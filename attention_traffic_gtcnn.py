import argparse
import csv
import time
from pathlib import Path
from functools import lru_cache
import matplotlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime

matplotlib.use("Agg")

# Every district runs the same local GTCNN. Each district then sums itself up in one token, and every
# district reads the K most useful tokens of the other districts (TOKEN_MODEL_PLAN.md).
#
# python attention_traffic_gtcnn.py                                # aware router, K=2 (our model)
# python attention_traffic_gtcnn.py --router traffic               # router without the support descriptor
# python attention_traffic_gtcnn.py --k 0                          # partitioned, no exchange
# python attention_traffic_gtcnn.py --partition none               # full city as one graph (K=0)
# python attention_traffic_gtcnn.py --k 0 --attn none              # plain GTCNN baseline
# python attention_traffic_gtcnn.py --gate off / --prior / --exchange full   # ablations
# python attention_traffic_gtcnn.py --seed 1                       # run seeds 0, 1, 2 for mean +- std
# python attention_traffic_gtcnn.py --data-dir data/forecast_dataset_full    # 21 districts, 4,143 sensors

"""
Hyper-Perameters with explination
"""
DATA_DIR = Path("data/forecast_dataset")
DISTRICTS = 4  # None = every district in the dataset
WINDOW = 6  # input window in timesteps
HORIZONS = 2  # predicted at once: 15, 30, 45, 60 min ahead
C_TAPS = 3  # graph filter order (number of taps); 1 = self only
A_TAPS = 1  # attention neighbourhood in hops; 1 = 1-hop neighbours (+ self)
N_FEAT = 32  # features before concat and graph filter. 8-12 is good
HIDDEN = 64  # features after concat, these are much cheaper. Also the size of a district token
DISTRICT_K = 5  # District node picks this many nodes using attention
ATTN = "district"  # local attention: "district", "shared" (neighbours only) or "none". --attn on the command line
K = 2  # other-district tokens each district reads; 0 = partitioned, no exchange. --k
ROUTER = "aware"  # "aware": tokens carry their support (live share, age); "traffic": tokens only. --router

EPOCHS = 1000       # upper bound; early stopping on validation can stop earlier
PATIENCE = 25       # stop after this many epochs without val improvement
MIN_DELTA = 0.01    # vehicles/hour; smaller val improvements do not count
LR = 1e-3           # learning rate
Gamma_LR = 0.985    # LR = LR * Gamma_LR ** epochs. An epoch is one pass over every district
SEED = 0

"""
Less hyper perameters probalby do not change
"""
COMPILE = True  # one compiled graph per district size
BATCH_SIZE = int(2 ** 8)  # origins per step; every district runs on the same origins
GPU_MARGIN_GB = 0.3  # the allocator is capped at the free GPU memory minus this (see main)
PLOT_FIGURES = False  # False skips the 10 figures per district, good for debug
TIME_FEATURES = ("tod_sin", "tod_cos", "dow_sin", "dow_cos")  # from features.npz time_features
INPUT_CHANNELS = ("intensity", "occupancy", "intensity_mask", "intensity_age") + TIME_FEATURES  # last axis of X
N_CHANNELS = len(INPUT_CHANNELS)
FULL_CITY = 0  # district id used for the whole city as one graph

MODEL_NAMES = {"district": "GTCNN + district attention",
               "shared": "GTCNN + neighbour attention",
               "none": "GTCNN"}

HORIZONS = (HORIZONS,) if isinstance(HORIZONS, int) else tuple(HORIZONS)  # be able to do both 2 and (1, 2, 3, 4)
if DISTRICTS is not None:                                                 # be able to train on a single district
    DISTRICTS = [int(DISTRICTS)] if isinstance(DISTRICTS, (int, np.integer)) else sorted({int(d) for d in DISTRICTS})

# ---------------------------------------------------------------- data

def district_ids():
    if DISTRICTS is not None:
        return list(DISTRICTS)
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in (DATA_DIR / "graph").glob("nodes_district_*.csv"))


def load_graph(n_sensors: int, edge_file: str) -> torch.Tensor:
    """Spatial shift operator of a route graph, normalised D^-1/2 A D^-1/2."""
    A = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    with open(DATA_DIR / "graph" / edge_file) as f:
        for e in csv.DictReader(f):
            i, j = int(e["target_index"]), int(e["source_index"])
            A[i, j] = A[j, i] = 1.0
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, d, 1.0) ** -0.5 * (d > 0)
    d_inv_sqrt = d_inv_sqrt.astype(np.float32)
    return torch.from_numpy(d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])


def load_sensors(district: int) -> np.ndarray:
    """Global feature column of every sensor of the district (in local index order)."""
    with open(DATA_DIR / "graph" / f"nodes_district_{district:02d}.csv") as f:
        nodes = sorted(csv.DictReader(f), key=lambda r: int(r["local_index"]))
    return np.array([int(r["global_index"]) for r in nodes], dtype=np.int64)


def neighbourhood(S: torch.Tensor, hops: int) -> torch.Tensor:
    """[N, N] True where j is within `hops` hops of i (self included): the attention's local keys."""
    N = S.shape[0]
    R = ((S > 0) | torch.eye(N, dtype=torch.bool, device=S.device)).float()
    return torch.linalg.matrix_power(R, hops) > 0


def cross_district_prior(ids, device) -> torch.Tensor:
    """[D, D] log(1 + total route-edge weight from district j into district i). Rows are receivers."""
    pos = {d: i for i, d in enumerate(ids)}
    w = np.zeros((len(ids), len(ids)), dtype=np.float32)
    path = DATA_DIR / "graph" / "edges_cross_district.csv"
    if len(ids) > 1 and path.exists():
        with open(path) as f:
            for e in csv.DictReader(f):
                i, j = pos.get(int(e["target_district"])), pos.get(int(e["source_district"]))
                if i is not None and j is not None:
                    w[i, j] += float(e["weight"])
    return torch.from_numpy(np.log1p(w)).to(device)


@lru_cache(maxsize=1)
def shared_data():
    """Everything that does not depend on the district, read from disk once per run."""
    t0 = time.perf_counter()
    z = np.load(DATA_DIR / "features.npz")
    names = list(z["time_feature_names"])
    tf = z["time_features"][:, [names.index(n) for n in TIME_FEATURES]].astype(np.float32)  # [T, 4]

    # every window of each split, like the baselines (not only the routine ones)
    with open(DATA_DIR / "windows.csv") as f:
        rows = list(csv.DictReader(f))
    origins = {s: np.array([int(r["origin_index"]) for r in rows if r["split"] == s], dtype=np.int64)
               for s in ("train", "val", "test")}
    p = np.load(DATA_DIR / "params.npz")

    sh = dict(intensity=z["intensity"].astype(np.float32), observed=z["intensity_mask"].astype(bool),
              occupancy=z["occupancy"].astype(np.float32), age=z["intensity_age"].astype(np.float32), time=tf,
              origins=origins, mean=p["scaler_mean"].astype(np.float32), std=p["scaler_std"].astype(np.float32),
              timestamps=z["timestamps"], sensor_ids=z["sensor_ids"])
    print(f"  shared data loaded once in {time.perf_counter() - t0:.1f}s")
    return sh


class District:
    """One district on the device: its graph and its [T, N] arrays. Windows are cut per batch,
    so every district fits in memory at once. District FULL_CITY is every sensor on the full graph."""

    def __init__(self, district: int, device):
        sh = shared_data()
        self.id = district
        if district == FULL_CITY:
            self.cols = np.arange(sh["intensity"].shape[1])
            self.S = load_graph(len(self.cols), "edges_full.csv").to(device)  # indices are feature columns
        else:
            self.cols = load_sensors(district)
            self.S = load_graph(len(self.cols), f"edges_district_{district:02d}.csv").to(device)  # [N, N]
        self.n = len(self.cols)
        self.reach = neighbourhood(self.S, A_TAPS)  # [N, N]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[:, self.cols])).to(device)
        self.intensity, self.occupancy, self.observed = t(sh["intensity"]), t(sh["occupancy"]), t(sh["observed"])
        self.age = t(sh["age"])  # steps since the last real reading / 96, 0 = real now
        self.mean = torch.from_numpy(sh["mean"][self.cols]).to(device)
        self.std = torch.from_numpy(sh["std"][self.cols]).to(device)
        self.origins = {s: torch.from_numpy(o).to(device) for s, o in sh["origins"].items()}


class City:
    """Every district of a run (or the whole city as one district), plus the shared time features
    and the cross-district edge prior."""

    def __init__(self, device, partition="district"):
        t0 = time.perf_counter()
        ids = [FULL_CITY] if partition == "none" else district_ids()
        self.districts = [District(d, device) for d in ids]
        self.time = torch.from_numpy(shared_data()["time"]).to(device)
        self.log_w = cross_district_prior(ids, device)
        self.origins = self.districts[0].origins  # the same windows for every district
        print(f"  loaded districts {ids} in {time.perf_counter() - t0:.1f}s")

    @property
    def n_sensors(self):
        return sum(d.n for d in self.districts)


@lru_cache(maxsize=1)
def window_offsets(device):
    return (torch.arange(-WINDOW + 1, 1, device=device), torch.tensor(HORIZONS, device=device))


def gather(dist: District, time_feats: torch.Tensor, origins: torch.Tensor):
    """origins [B] -> X [B, WINDOW, N, C] inputs (INPUT_CHANNELS), y [B, N, H] standardised targets,
    mask [B, N, H]."""
    hist, ahead = window_offsets(origins.device)
    rows = origins[:, None] + hist  # [B, WINDOW]
    B, N = len(origins), dist.n
    X = torch.cat([dist.intensity[rows][..., None], dist.occupancy[rows][..., None],
                   dist.observed[rows][..., None].float(), dist.age[rows][..., None],
                   time_feats[rows][:, :, None, :].expand(B, WINDOW, N, len(TIME_FEATURES))], dim=-1)
    targets = origins[:, None] + ahead  # [B, H]
    y = dist.intensity[targets].transpose(1, 2)
    mask = dist.observed[targets].transpose(1, 2)
    return X, y, mask


def support(dist: District, origins: torch.Tensor) -> torch.Tensor:
    """[B, 2]: share of the district's sensors with a real reading at the origin, and their mean age.
    Taken from the mask and age, never from the filled values (those look like real traffic)."""
    return torch.stack([dist.observed[origins].float().mean(1), dist.age[origins].mean(1)], dim=1)


def save_checkpoint(model, path, trained_districts, best_epoch, seed, run):
    ckpt = dict(
        format=4,  # 4: district tokens and router
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},   # weights only, no graph buffers
        config=dict(WINDOW=WINDOW, HORIZONS=list(HORIZONS), C_TAPS=C_TAPS, A_TAPS=A_TAPS, N_FEAT=N_FEAT,
                    HIDDEN=HIDDEN),
        model=model.kwargs,  # what GTCNN(**model) needs
        run=run,  # partition and exchange
        input_channels=list(INPUT_CHANNELS),   # last axis of X, in order
        trained_districts=list(trained_districts),
        best_epoch=best_epoch, seed=seed,
        torch=str(torch.__version__),                    # str(): the raw version object can't be loaded safely
        saved=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    torch.save(ckpt, path)


def load_trained(path, device="cpu"):
    """Checkpoint -> model (eval mode) and the City it was trained on, on `device`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)     # plain tensors/dicts only: safe to load
    assert ckpt.get("format") == 4, "checkpoint is from an older version of this file (other inputs/outputs)"
    cfg = ckpt["config"]
    mine = dict(WINDOW=WINDOW, HORIZONS=list(HORIZONS), C_TAPS=C_TAPS, A_TAPS=A_TAPS, N_FEAT=N_FEAT, HIDDEN=HIDDEN)
    bad = {k: (cfg[k], mine[k]) for k in mine if cfg[k] != mine[k]}
    assert not bad, f"constants differ from the checkpoint (checkpoint, this file): {bad}"
    device = torch.device(device)
    model = GTCNN(**ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    city = City(device, ckpt["run"]["partition"])
    return model.eval(), city


# --------------------------------------------------------------- model

class SharedGATAttention(nn.Module):
    """GAT-style spatial attention restricted to nodes within `hops` hops (self included).
    Scores are e_ij = LeakyReLU(a_src . h_i + a_dst . h_j), computed once from the last
    time step and reused for every step in the window. Residual connection.
    No node-specific parameters, so it is permutation-equivariant over sensors."""

    def __init__(self, c: int):
        super().__init__()
        self.a_src = nn.Linear(c, 1, bias=False)
        self.a_dst = nn.Linear(c, 1, bias=False)
        self.v = nn.Linear(c, c)

    def forward(self, x, reach):  # x: [B, T, N, C], reach: [N, N] local keys
        B, T, N, C = x.shape
        h = x[:, -1]
        e = F.leaky_relu(self.a_src(h) + self.a_dst(h).transpose(1, 2), 0.2)
        alpha = F.softmax(e.masked_fill(~reach, float("-inf")), dim=-1)  # [B, N, N]
        v = self.v(x).permute(0, 2, 1, 3).reshape(B, N, T * C)  # [B, N, T*C]
        out = torch.bmm(alpha, v).reshape(B, N, T, C).permute(0, 2, 1, 3)
        return x + out


class DistrictGATAttention(SharedGATAttention):
    """Dense neighbour attention plus K district picks as extra keys in the same softmax."""

    def __init__(self, c: int, k: int = DISTRICT_K):
        super().__init__(c)
        self.k = k
        self.a_sel = nn.Linear(c, 1)  # importance score (5 params)

    def forward(self, x, reach):  # x: [B, T, N, C]
        B, T, N, C = x.shape
        h = x[:, -1]
        s = self.a_src(h)  # [B, N, 1]
        t = self.a_dst(h).squeeze(-1)  # [B, N]
        u_top, pick = self.a_sel(h).squeeze(-1).topk(min(self.k, N), dim=1)  # [B, K]

        picked = torch.zeros(B, N, dtype=torch.bool, device=x.device).scatter(1, pick, True)
        gate = torch.ones(B, N, device=x.device, dtype=h.dtype).scatter(1, pick, torch.sigmoid(u_top))

        keep = reach[None] | picked[:, None, :]  # local keys OR picked sensors
        e = F.leaky_relu(s + t[:, None, :], 0.2).masked_fill(~keep, float("-inf"))
        alpha = F.softmax(e, dim=-1)  # [B, N, N]
        v = (self.v(x) * gate[:, None, :, None]).permute(0, 2, 1, 3).reshape(B, N, T * C)
        out = torch.bmm(alpha, v).reshape(B, N, T, C).permute(0, 2, 1, 3)
        return x + out


class StructuredGraphFilter(nn.Module):
    """Graph-time filter on the product graph, without building S_product:
    (I⊗S + S_time⊗I) x  =  S x_t + x_{t-1}   (per slice, plus a time shift)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(C_TAPS, n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.xavier_uniform_(self.weight)

    @staticmethod
    def product_step(x, S):  # x: [B, T, N, F]
        B, T, N, Fd = x.shape
        xs = x.permute(2, 0, 1, 3).reshape(N, -1)  # [N, B*T*F]
        spatial = (S @ xs).reshape(N, B, T, Fd).permute(1, 2, 0, 3)
        past = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return spatial + past

    def forward(self, x, S):  # [B, T, N, F_in]
        taps = [x]
        for _ in range(1, C_TAPS):
            taps.append(self.product_step(taps[-1], S))  # S_product^k x, iteratively
        W = self.weight.permute(2, 0, 1).reshape(self.weight.shape[2], -1)
        return chunked_linear(torch.cat(taps, dim=-1), W, self.bias)


class SupportRouter(nn.Module):
    """Each district reads the K most useful tokens of the other districts

    score  e(d<-d') = (Wq u_d) . (Wk [u_d' | s_d']) / sqrt(F)  (+ beta log w(d'->d))
    keep the top K, softmax -> alpha, then scale by the gate g = sigmoid(wg . [s_d' | s_d] + bg)
    context r_d = sum alpha * g * Wv u_d'

    aware=False is the traffic-only router: no support in the keys and no gate."""

    def __init__(self, f: int, k: int = K, aware: bool = True, gate: bool = True, prior: bool = False):
        super().__init__()
        self.k, self.aware = k, aware
        s = 2 if aware else 0
        self.q = nn.Linear(f, f, bias=False)
        self.key = nn.Linear(f + s, f, bias=False)
        self.v = nn.Linear(f, f, bias=False)
        self.gate = nn.Linear(2 * s, 1) if aware and gate else None
        self.beta = nn.Parameter(torch.zeros(())) if prior else None
        self.last = None  # (picked districts [B, D, K], weights [B, D, K]) of the last call, for analysis

    def forward(self, u, s, log_w):  # u [B, D, F] tokens, s [B, D, 2] support, log_w [D, D]
        B, D, f = u.shape
        if self.k == 0 or D == 1:
            return torch.zeros_like(u)
        keys = self.key(torch.cat([u, s], -1) if self.aware else u)
        e = self.q(u) @ keys.transpose(1, 2) / f ** 0.5  # [B, D receivers, D sources]
        if self.beta is not None:
            e = e + self.beta * log_w
        e = e.masked_fill(torch.eye(D, dtype=torch.bool, device=u.device), float("-inf"))  # never itself
        top, idx = e.topk(min(self.k, D - 1), dim=-1)  # [B, D, K]
        alpha = top.softmax(-1)
        if self.gate is not None:
            # pair[b, i, j] = [support of source j | support of receiver i]
            pair = torch.cat([s[:, None].expand(B, D, D, 2), s[:, :, None].expand(B, D, D, 2)], -1)
            alpha = alpha * torch.sigmoid(self.gate(pair)).squeeze(-1).gather(2, idx)
        vals = self.v(u)[torch.arange(B, device=u.device)[:, None, None], idx]  # [B, D, K, F]
        self.last = (idx.detach(), alpha.detach())
        return (alpha[..., None] * vals).sum(2)  # [B, D, F]

    def scalars_received(self, n_districts: int) -> int:
        """Numbers each district receives per origin: K tokens, plus their support if aware."""
        if self.k == 0 or n_districts == 1:
            return 0
        return min(self.k, n_districts - 1) * (self.q.in_features + (2 if self.aware else 0))


class GTCNN(nn.Module):
    """Local encoder (spatial attention, a space-time filter layer, hidden layers) on one district,
    a router that brings in other districts' tokens, and one shared head that predicts every horizon.
    Every node runs the exact same network with the same weights, and the graph is an input,
    so one model serves every district."""

    def __init__(self, attn: str = ATTN, k: int = K, router: str = ROUTER, gate: bool = True, prior: bool = False):
        super().__init__()
        self.kwargs = dict(attn=attn, k=k, router=router, gate=gate, prior=prior)
        self.attn_kind = attn
        self.attn = {"district": lambda: DistrictGATAttention(N_CHANNELS),
                     "shared": lambda: SharedGATAttention(N_CHANNELS),
                     "none": lambda: None}[attn]()
        self.filter1 = StructuredGraphFilter(N_CHANNELS, N_FEAT)
        self.fc = nn.Linear(WINDOW * N_FEAT, HIDDEN)  # <-- extra hidden layer
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)  # <-- extra hidden layer
        self.fc3 = nn.Linear(HIDDEN, HIDDEN)  # <-- extra hidden layer
        self.router = SupportRouter(HIDDEN, k, aware=router == "aware", gate=gate, prior=prior)
        self.head = nn.Linear(2 * HIDDEN, len(HORIZONS))  # own state | context from other districts

    def encode(self, x: torch.Tensor, S: torch.Tensor, reach: torch.Tensor):  # x: [B, WINDOW, N, C]
        """Node states h [B, N, HIDDEN] and the latest intensity [B, N, 1]."""
        last = x[:, -1, :, :1]  # latest intensity, before attention changes x
        if self.attn is not None:
            x = self.attn(x, reach)  # [B, T, N, C]
        z = torch.relu(self.filter1(x, S))  # [B, T, N, F]
        z = z.permute(0, 2, 1, 3).flatten(2)  # [B, N, T*F]
        z = torch.relu(self.fc(z))  # <-- hidden layer + activation
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        return z, last

    def decode(self, h, last, r=None):  # r [B, HIDDEN] context for this district, None = no exchange
        r = torch.zeros_like(h) if r is None else r[:, None, :].expand_as(h)
        return last + self.head(torch.cat([h, r], -1))  # [B, N, H] latest reading + change per horizon

    def forward(self, x, S, reach):  # one district on its own, no exchange
        return self.decode(*self.encode(x, S, reach))


def masked_mae(pred, y, mask):
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp(min=1)


def chunked_linear(x, W, bias, chunks=512):  # x [..., K], W [F_out, K]
    K = x.shape[-1]
    if (x.numel() // K) % chunks:
        return F.linear(x, W) + bias
    out = torch.bmm(x.reshape(chunks, -1, K), W.t().unsqueeze(0).expand(chunks, -1, -1))
    return out.reshape(*x.shape[:-1], -1) + bias


# --------------------------------------------------------------- city forward pass

CHUNK = BATCH_SIZE  # origins per forward pass, halved after an out-of-memory error


def out_of_memory(e: BaseException) -> bool:
    """True for an out-of-memory error, also when wrapped (torch.compile raises it as BackendCompilerFailed)."""
    while e is not None:
        if isinstance(e, torch.OutOfMemoryError):
            return True
        e = e.__cause__ or e.__context__
    return False


def chunked(o, step, reset):
    """Run step(chunk) over the origins o in chunks of CHUNK. After an out-of-memory error: reset(),
    halve CHUNK and start o over. Losses and scores are sums over origins, so the chunk size only
    changes memory and time, never the result (a batch is still BATCH_SIZE origins per optimizer step)."""
    global CHUNK
    while True:
        try:
            for c in range(0, len(o), CHUNK):
                step(o[c:c + CHUNK])
            return
        except Exception as e:
            if CHUNK == 1 or not out_of_memory(e):
                raise
        reset()  # outside the except, so the failed pass's tensors are already freed
        torch.cuda.empty_cache()
        CHUNK //= 2
        print(f"  out of GPU memory: {CHUNK} origins per forward pass from now on", flush=True)


def talks(model, city) -> bool:
    return model.router.k > 0 and len(city.districts) > 1


@torch.no_grad()
def tokens(model, city, origins, encode):
    """Pass 1: every district's token u [B, D, F] and support s [B, D, 2] at these origins."""
    u = torch.stack([encode(gather(d, city.time, origins)[0], d.S, d.reach)[0].mean(1) for d in city.districts], 1)
    s = torch.stack([support(d, origins) for d in city.districts], 1)
    return u, s


def city_outputs(model, city, origins, encode, exchange="detached"):
    """Yields (pred, y, mask) for every district at these origins, standardised, one district at a time.

    detached: pass 1 makes every token without gradients, pass 2 runs each district with its own token
    live and the other districts' tokens fixed. Memory stays at the size of one district, and because
    the weights are shared every weight is still trained. full: one pass, gradients everywhere."""
    ds = city.districts
    if not talks(model, city):
        for d in ds:
            X, y, mask = gather(d, city.time, origins)
            h, last = encode(X, d.S, d.reach)
            yield model.decode(h, last), y, mask
        return

    if exchange == "full" and torch.is_grad_enabled():
        outs = []
        for d in ds:
            X, y, mask = gather(d, city.time, origins)
            outs.append((*encode(X, d.S, d.reach), y, mask))
        u = torch.stack([h.mean(1) for h, _, _, _ in outs], 1)
        s = torch.stack([support(d, origins) for d in ds], 1)
        r = model.router(u, s, city.log_w)
        for di, (h, last, y, mask) in enumerate(outs):
            yield model.decode(h, last, r[:, di]), y, mask
        return

    u, s = tokens(model, city, origins, encode)
    training = torch.is_grad_enabled()
    if not training:  # evaluation: the tokens are exact, so one router call serves every district
        r = model.router(u, s, city.log_w)
    for di, d in enumerate(ds):
        X, y, mask = gather(d, city.time, origins)
        h, last = encode(X, d.S, d.reach)
        if training:
            u_d = u.clone()
            u_d[:, di] = h.mean(1)  # own token with gradients, the others fixed
            r_d = model.router(u_d, s, city.log_w)[:, di]
        else:
            r_d = r[:, di]
        yield model.decode(h, last, r_d), y, mask


# --------------------------------------------------------------- train

def train_epoch(model, city, optimizer, gen, encode, exchange="detached"):
    """One pass over every training window. A step covers every district at the same origins.
    Loss: masked MAE on standardised intensity, all horizons, pooled over every sensor."""
    model.train()
    o = city.origins["train"]
    o = o[torch.randperm(len(o), generator=gen).to(o.device)]
    _, ahead = window_offsets(o.device)
    total = torch.zeros((), device=o.device)
    n_steps = 0
    for b in range(0, len(o), BATCH_SIZE):
        ob = o[b:b + BATCH_SIZE]
        n_targets = sum(d.observed[ob[:, None] + ahead].sum() for d in city.districts).clamp(min=1)
        losses = []

        def step(oc):  # gradients add up over the chunks of the batch
            if exchange == "full":
                loss = sum(((p - y).abs() * m).sum() for p, y, m in city_outputs(model, city, oc, encode, "full"))
                loss = loss / n_targets
                loss.backward()
                losses.append(loss.detach())
            else:
                for p, y, m in city_outputs(model, city, oc, encode):
                    loss = ((p - y).abs() * m).sum() / n_targets
                    loss.backward()  # gradients add up over the districts, memory is freed per district
                    losses.append(loss.detach())

        optimizer.zero_grad()
        chunked(ob, step, reset=lambda: (optimizer.zero_grad(), losses.clear()))
        optimizer.step()
        total += sum(losses)
        n_steps += 1
    return (total / n_steps).item()  # one sync per epoch


@torch.no_grad()
def score(model, city, split, encode=None):
    """Pooled over every district: per horizon MAE and RMSE in vehicles/hour, plus per-district MAE."""
    model.eval()
    encode = encode or model.encode
    D, H = len(city.districts), len(HORIZONS)
    sums = torch.zeros(3, D, H, device=city.time.device, dtype=torch.float32)  # abs, squared, count
    o = city.origins[split]

    def add(part, oc):
        for di, (pred, y, mask) in enumerate(city_outputs(model, city, oc, encode)):
            m = mask
            err = ((pred - y) * city.districts[di].std[None, :, None]) * m  # veh/h, the mean cancels
            part[:, di] += torch.stack([err.abs().sum((0, 1)), err.square().sum((0, 1)), m.sum((0, 1))])

    for b in range(0, len(o), BATCH_SIZE):
        part = torch.zeros_like(sums)
        chunked(o[b:b + BATCH_SIZE], lambda oc: add(part, oc), reset=part.zero_)
        sums += part
    abs_sum, sq_sum, count = sums
    per_district = {d.id: (abs_sum[i] / count[i].clamp(min=1)).tolist() for i, d in enumerate(city.districts)}
    abs_sum, sq_sum, count = abs_sum.sum(0), sq_sum.sum(0), count.sum(0)
    count_l = count.long().tolist()
    if DISTRICTS is None:  # all districts together are exactly the sensors the baselines score
        assert count_l == expected_targets(split), \
            f"{split}: scored {count_l} targets, baselines score {expected_targets(split)}"
    mae = (abs_sum / count.clamp(min=1)).tolist()
    rmse = (sq_sum / count.clamp(min=1)).sqrt().tolist()
    return mae, rmse, count_l, per_district


@torch.no_grad()
def predict(model, city, di, split, encode=None):
    """One district's predictions, targets (both vehicles/hour) and genuine-target mask: [n_windows, N, H]."""
    model.eval()
    encode = encode or model.encode
    d = city.districts[di]
    o = city.origins[split]
    parts = []
    chunked(o, lambda oc: parts.append(list(city_outputs(model, city, oc, encode))[di]), reset=parts.clear)
    pred, y, mask = (torch.cat(t) for t in zip(*parts))
    scale = lambda t: t * d.std[None, :, None] + d.mean[None, :, None]
    return scale(pred), scale(y), mask


def expected_targets(split):
    """Genuine targets the baselines score in `split`, per horizon, over every sensor."""
    sh = shared_data()
    o = sh["origins"][split]
    return [int(sh["observed"][o + h].sum()) for h in HORIZONS]


@torch.no_grad()
def router_stats(model, city, split, path):
    """Save which districts every district picked, their weights and every token's support (for the figures)."""
    if not talks(model, city):
        return
    model.eval()
    o = city.origins[split]
    picks, weights, sup = [], [], []

    def step(oc):
        u, s = tokens(model, city, oc, model.encode)
        model.router(u, s, city.log_w)
        idx, alpha = model.router.last
        picks.append(idx.cpu()), weights.append(alpha.cpu()), sup.append(s.cpu())

    chunked(o, step, reset=lambda: (picks.clear(), weights.clear(), sup.clear()))
    ids = np.array([d.id for d in city.districts])
    np.savez(path, district_ids=ids, origins=o.cpu().numpy(), picks=ids[torch.cat(picks).numpy()],
             weights=torch.cat(weights).numpy(), support=torch.cat(sup).numpy())
    print(f"  router picks: {path}")


COST_FIELDS = ["n_params", "epochs", "train_s", "epoch_s", "infer_s", "peak_gb"]  # same columns as rq1/NN.py


def save_result(model_name: str, seed: int, results: dict, cost: dict):
    """Write this run's val and test scores to reports/graph_results.csv (same columns as the RQ1
    baselines, plus the seed and the cost columns), replacing any earlier run of the same model and seed."""
    path = DATA_DIR / "reports" / "graph_results.csv"
    fields = ["split", "model", "horizon_min", "MAE", "RMSE", "seed"] + COST_FIELDS
    rows = []
    if path.exists():
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
    rows =[r for r in rows if (r["model"], r.get("seed", "")) != (model_name, str(seed))]
    for split, (mae, rmse) in results.items():
        for h, a, s in zip(HORIZONS, mae, rmse):
            rows.append({"split": split, "model": model_name, "horizon_min": h * 15,
                         "MAE": a, "RMSE": s, "seed": seed, **cost})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, restval="")  # older rows have no cost columns
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved to {path}")


def plot_sensors(model, city, di, h_index=1):
    """Last week of test at 10 sensors of one district, `h_index` = 1 is 30 min ahead."""
    sh = shared_data()
    d = city.districts[di]
    pred, y, mask = predict(model, city, di, "test")
    t_test = sh["timestamps"][d.origins["test"].cpu().numpy() + HORIZONS[h_index]]
    pred_np, y_np = pred[..., h_index].cpu().numpy(), y[..., h_index].cpu().numpy()
    y_np[~mask[..., h_index].cpu().numpy()] = np.nan
    week = t_test >= t_test[-1] - np.timedelta64(7, "D")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    for i in np.linspace(0, d.n - 1, 10).astype(int):
        sid = sh["sensor_ids"][d.cols[i]]
        err = np.sqrt(np.nanmean((pred_np[week, i] - y_np[week, i]) ** 2))
        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.plot(t_test[week], y_np[week, i], lw=0.8, color="steelblue", label="true")
        ax.plot(t_test[week], pred_np[week, i], lw=0.8, alpha=0.85, color="darkorange",
                label=f"predicted ({HORIZONS[h_index] * 15} min ahead)")
        ax.set_title(f"sensor {sid} - last week of test, RMSE {err:.1f} vehicles/hour")
        ax.set_ylabel("vehicles/hour")
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / f"sensor_{sid}_district_{d.id}.png", dpi=250)
        plt.close(fig)
    print(f"  figures: {fig_dir}/sensor_*_district_{d.id}.png")


# --------------------------------------------------------------- checks

def district_check(d: District, time_feats, tol=1e-4):
    m = DistrictGATAttention(N_CHANNELS).to(d.S.device)
    xb, _, _ = gather(d, time_feats, d.origins["train"][:64])
    with torch.no_grad():
        B, T, N, C = xb.shape
        h = xb[:, -1]
        s, t = m.a_src(h), m.a_dst(h).squeeze(-1)
        u_top, pick = m.a_sel(h).squeeze(-1).topk(m.k, dim=1)
        keep = d.reach[None].expand(B, -1, -1).clone()
        keep.scatter_(2, pick[:, None, :].expand(-1, N, -1), True)
        alpha = F.softmax(F.leaky_relu(s + t[:, None, :], 0.2).masked_fill(~keep, float("-inf")), dim=-1)
        gate = torch.ones(B, N, device=xb.device, dtype=h.dtype)
        for b in range(B):  # plain loop: obviously correct
            gate[b, pick[b]] = torch.sigmoid(u_top[b])
        v = (m.v(xb) * gate[:, None, :, None]).permute(0, 2, 1, 3).reshape(B, N, T * C)
        want = xb + torch.bmm(alpha, v).reshape(B, N, T, C).permute(0, 2, 1, 3)
        err = (m(xb, d.reach) - want).abs().max().item()
    print(f"district attention vs explicit softmax reference: max |diff| {err:.2e}")
    assert err < tol


def router_check(router: SupportRouter, u, s, log_w, tol=1e-5):
    """Check the router against a plain loop over every receiver, batch element and pick."""
    with torch.no_grad():
        got = router(u, s, log_w)
        B, D, f = u.shape
        want = torch.zeros_like(u)
        for b in range(B):  # plain loops: obviously correct
            for i in range(D):
                q = router.q(u[b, i])
                e = []
                for j in range(D):
                    key_in = torch.cat([u[b, j], s[b, j]]) if router.aware else u[b, j]
                    e.append(-float("inf") if j == i else
                             float(q @ router.key(key_in) / f ** 0.5
                                   + (router.beta * log_w[i, j] if router.beta is not None else 0)))
                e = torch.tensor(e)
                top, idx = e.topk(min(router.k, D - 1))
                alpha = top.softmax(0)
                for a, j in zip(alpha, idx.tolist()):
                    g = torch.sigmoid(router.gate(torch.cat([s[b, j], s[b, i]]))).item() if router.gate is not None else 1.0
                    want[b, i] += a * g * router.v(u[b, j])
        err = (got - want).abs().max().item()
    print(f"router vs plain loop: max |diff| {err:.2e}")
    assert err < tol


def district_bench(d: District, time_feats, reps=10):
    device = d.S.device
    xb, _, _ = gather(d, time_feats, d.origins["train"][:BATCH_SIZE])
    sync = (torch.mps.synchronize if device.type == "mps"
            else torch.cuda.synchronize if device.type == "cuda" else (lambda: None))

    def ms(mod):
        xi = xb.clone().requires_grad_(True)

        def run():
            xi.grad = None
            mod(xi, d.reach).sum().backward()

        for _ in range(3): run()
        sync();
        t0 = time.perf_counter()
        for _ in range(reps): run()
        sync()
        return (time.perf_counter() - t0) / reps * 1e3

    rows = [("dense_bmm", SharedGATAttention(N_CHANNELS))]
    rows += [(f"district K={k}", DistrictGATAttention(N_CHANNELS, k=k)) for k in (3, 5)]
    print("attention fwd+bwd [ms]        eager  compiled")
    for name, mod in rows:
        mod = mod.to(device)
        print(f"  {name:16s} {ms(mod):8.1f} {ms(torch.compile(mod)):8.1f}")


# --------------------------------------------------------------- main

def run_name(args) -> str:
    """Model name in graph_results.csv, built from the flags."""
    name = MODEL_NAMES[args.attn]
    if args.partition == "none":
        return f"Full-city {name}"
    if args.k == 0:
        return f"{name} (no exchange)"
    name += f", {args.router} router K={args.k}"
    if args.router == "aware" and args.gate == "off":
        name += ", no gate"
    if args.prior:
        name += ", edge prior"
    if args.exchange == "full":
        name += ", full exchange"
    return name


def main():
    global EPOCHS, DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", choices=list(MODEL_NAMES), default=ATTN, help="local attention inside a district")
    parser.add_argument("--k", type=int, default=K, help="other-district tokens each district reads, 0 = none")
    parser.add_argument("--router", choices=("aware", "traffic"), default=ROUTER)
    parser.add_argument("--gate", choices=("on", "off"), default="on", help="support gate (aware router only)")
    parser.add_argument("--prior", action="store_true", help="add beta * log(cross-district edge weight)")
    parser.add_argument("--exchange", choices=("detached", "full"), default="detached",
                        help="detached: other districts' tokens carry no gradient (memory of one district)")
    parser.add_argument("--partition", choices=("district", "none"), default="district",
                        help="none = the whole city as one graph, no exchange")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    EPOCHS, DATA_DIR = args.epochs, args.data_dir
    if args.partition == "none":
        args.k = 0  # one graph, nobody to exchange with

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if device.type == "cuda":
        # beyond the free memory the driver can spill into shared system memory (WSL/Windows), which is
        # very slow and makes the timings meaningless: raise out-of-memory instead, so chunked() splits the batch
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(max(0.1, (free - GPU_MARGIN_GB * 1e9) / total))
        print(f"GPU memory cap {(free - GPU_MARGIN_GB * 1e9) / 1e9:.2f} GB of {total / 1e9:.2f} GB")

    city = City(device, args.partition)
    model = GTCNN(args.attn, args.k, args.router, args.gate == "on", args.prior).to(device)  # one model for every district
    model_name = run_name(args)
    print(f"device={device} data={DATA_DIR} model='{model_name}' seed={args.seed} "
          f"districts={[d.id for d in city.districts]} sensors={city.n_sensors} "
          f"largest district={max(d.n for d in city.districts)} "
          f"parameters={sum(p.numel() for p in model.parameters()):,} "
          f"scalars received per district per origin={model.router.scalars_received(len(city.districts))}")
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(city.origins[s]):,} windows")

    torch._dynamo.config.cache_size_limit = 64  # a few graphs per district size
    encode = torch.compile(model.encode) if COMPILE and not args.no_compile else model.encode

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=Gamma_LR)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync = (torch.cuda.synchronize if device.type == "cuda"
            else torch.mps.synchronize if device.type == "mps" else (lambda: None))  # so timings are real
    best_val, best_epoch, best_state = float("inf"), 0, None
    epoch_times = []
    train_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr = train_epoch(model, city, optimizer, gen, encode, args.exchange)
        va_mae, _, _, _ = score(model, city, "val", encode)
        va = float(np.mean(va_mae))  # model selection: val MAE in vehicles/hour, mean over horizons
        sync()
        epoch_times.append(time.perf_counter() - t0)  # train + val pass, like rq1/NN.py
        scheduler.step()
        if va < best_val - MIN_DELTA:
            best_val, best_epoch = va, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  train mae {tr:.4f} (std units)  val mae {va:7.2f} veh/h"
              f" [{' '.join(f'{v:.1f}' for v in va_mae)}]"
              f"  {time.perf_counter() - t0:.1f}s/epoch  lr {scheduler.get_last_lr()[0]:.1e}"
              f"{'  *' if best_epoch == epoch else ''}", flush=True)
        if epoch - best_epoch >= PATIENCE:
            print(f"early stop at epoch {epoch}: no val improvement > {MIN_DELTA} veh/h for {PATIENCE} epochs")
            break
    train_seconds = time.perf_counter() - train_start
    print(f"training time {train_seconds:.1f}s ({epoch} epochs, median {np.median(epoch_times):.1f}s/epoch)")
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
    print(f"peak GPU memory {peak_gb:.2f} GB, {CHUNK} origins per forward pass")

    model.load_state_dict(best_state)  # the epoch with the best validation score, never the test score
    print(f"best val mae {best_val:.2f} veh/h at epoch {best_epoch}")

    #Saving the model!
    tag = time.strftime("%Y%m%d_%H%M")
    Path("checkpoints").mkdir(exist_ok=True)
    slug = f"{args.partition}_{args.attn}_k{args.k}_{args.router}" + ("_nogate" if args.gate == "off" else "") \
           + ("_prior" if args.prior else "") + ("_full" if args.exchange == "full" else "")
    ckpt_path = f"checkpoints/gtcnn_{slug}_seed{args.seed}_{tag}.pt"
    save_checkpoint(model, ckpt_path, [d.id for d in city.districts], best_epoch, args.seed,
                    dict(partition=args.partition, exchange=args.exchange, data_dir=str(DATA_DIR)))
    reloaded = GTCNN(**model.kwargs).to(device)
    reloaded.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)["state_dict"])
    model.eval(), reloaded.eval()
    with torch.no_grad():
        o = city.origins["train"][:8]
        for (a, _, _), (b, _, _) in zip(city_outputs(model, city, o, model.encode),
                                        city_outputs(reloaded, city, o, reloaded.encode)):
            assert torch.allclose(a, b, atol=1e-6), "checkpoint round-trip mismatch"
    print(f"saved {ckpt_path}")

    # final scores with the best weights, eager (no compile)
    results = {}
    for split in ("val", "test"):  # val first, so the GPU is warm when test is timed
        sync()
        t0 = time.perf_counter()
        mae, rmse, count, per_district = score(model, city, split)  # predict every window and sensor
        sync()
        infer_seconds = time.perf_counter() - t0
        results[split] = (mae, rmse)
    cost = dict(n_params=sum(p.numel() for p in model.parameters()), epochs=epoch, train_s=train_seconds,
                epoch_s=float(np.median(epoch_times)), infer_s=infer_seconds, peak_gb=peak_gb)
    print(f"test inference {infer_seconds:.2f}s for {len(city.origins['test']):,} windows")
    print(f"\n{model_name}, test, every window and sensor, genuine targets (vehicles/hour):")
    print(f"  {'horizon':>8} {'MAE':>8} {'RMSE':>8} {'targets':>10}")
    for h, a, s, c in zip(HORIZONS, *results["test"], count):
        print(f"  {h * 15:>5} min {a:8.2f} {s:8.2f} {c:10,d}")
    print(f"\n  test MAE per district: " + ", ".join(f"{k}: {v[0]:.1f}/{v[-1]:.1f}" for k, v in per_district.items())
          + f"  ({HORIZONS[0] * 15}/{HORIZONS[-1] * 15} min)")
    save_result(model_name, args.seed, results, cost)
    router_stats(model, city, "test", DATA_DIR / "reports" / f"router_{slug}_seed{args.seed}.npz")

    if PLOT_FIGURES:
        for di in range(len(city.districts)):
            plot_sensors(model, city, di)


if __name__ == "__main__":
    main()
