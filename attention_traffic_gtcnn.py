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

# python attention_traffic_gtcnn.py                      # GTCNN + district attention, seed 0
# python attention_traffic_gtcnn.py --attn shared        # ablation: neighbour attention only
# python attention_traffic_gtcnn.py --attn none          # ablation: plain GTCNN, no attention
# python attention_traffic_gtcnn.py --seed 1             # run seeds 0, 1, 2 for mean +- std

"""
Hyper-Perameters with explination
"""
DATA_DIR = Path("data/forecast_dataset")
DISTRICTS = None  # None = every district in the dataset
WINDOW = 6  # input window in timesteps
HORIZONS = (1, 2, 3, 4)  # predicted at once: 15, 30, 45, 60 min ahead
C_TAPS = 3  # graph filter order (number of taps); 1 = self only
A_TAPS = 1  # attention neighbourhood in hops; 1 = 1-hop neighbours (+ self)
N_FEAT = 32  # features before concat and graph filter. 8-12 is good
HIDDEN = 64  # features after concat, these are much cheaper. Maybe we add some extra layers here?
DISTRICT_K = 5  # District node picks this many nodes using attention
ATTN = "district"  # "district", "shared" (neighbours only) or "none" (plain GTCNN). --attn on the command line

EPOCHS = 1000       # upper bound; early stopping on validation can stop earlier
PATIENCE = 25       # stop after this many epochs without val improvement
MIN_DELTA = 0.01    # vehicles/hour; smaller val improvements do not count
LR = 1e-3           # learning rate
Gamma_LR = 0.985    # LR = LR * Gamma_LR ** epochs. An epoch is now one pass over every district
SEED = 0

"""
Less hyper perameters probalby do not change
"""
COMPILE = True  # one compiled graph per district size
BATCH_SIZE = int(2 ** 10) # How to batch
PLOT_FIGURES = False  # False skips the 10 figures per district, good for debug
TIME_FEATURES = ("tod_sin", "tod_cos", "dow_sin", "dow_cos")  # from features.npz time_features
INPUT_CHANNELS = ("intensity", "occupancy", "intensity_mask", "intensity_age") + TIME_FEATURES  # last axis of X
N_CHANNELS = len(INPUT_CHANNELS)

MODEL_NAMES = {"district": "GTCNN + district attention",
               "shared": "GTCNN + neighbour attention",
               "none": "GTCNN"}

# ---------------------------------------------------------------- data

def district_ids():
    if DISTRICTS is not None:
        return list(DISTRICTS)
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in (DATA_DIR / "graph").glob("nodes_district_*.csv"))


def load_district_graph(n_sensors: int, district: int) -> torch.Tensor:
    """Spatial shift operator of the district route graph, normalised D^-1/2 A D^-1/2."""
    A = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    with open(DATA_DIR / "graph" / f"edges_district_{district:02d}.csv") as f:
        for e in csv.DictReader(f):
            i, j = int(e["target_index"]), int(e["source_index"])
            A[i, j] = A[j, i] = 1.0
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0).astype(np.float32)
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
    first = min(o.min() for o in origins.values())
    last = max(o.max() for o in origins.values()) + max(HORIZONS)
    assert first >= WINDOW - 1 and last < len(tf), "a window reaches outside the year"
    print(f"  shared data loaded once in {time.perf_counter() - t0:.1f}s")
    return sh


class District:
    """One district on the device: its graph and its [T, N] arrays. Windows are cut per batch,
    so every district fits in memory at once and training can mix them."""

    def __init__(self, district: int, device):
        sh = shared_data()
        self.id = district
        self.cols = load_sensors(district)
        self.n = len(self.cols)
        self.S = load_district_graph(self.n, district).to(device)  # [N, N]
        self.reach = neighbourhood(self.S, A_TAPS)  # [N, N]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[:, self.cols])).to(device)
        self.intensity, self.occupancy, self.observed = t(sh["intensity"]), t(sh["occupancy"]), t(sh["observed"])
        self.age = t(sh["age"])  # steps since the last real reading / 96, 0 = real now
        self.mean = torch.from_numpy(sh["mean"][self.cols]).to(device)
        self.std = torch.from_numpy(sh["std"][self.cols]).to(device)
        self.origins = {s: torch.from_numpy(o).to(device) for s, o in sh["origins"].items()}


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


def load_all(device):
    t0 = time.perf_counter()
    districts = [District(d, device) for d in district_ids()]
    time_feats = torch.from_numpy(shared_data()["time"]).to(device)
    print(f"  loaded districts {[d.id for d in districts]} in {time.perf_counter() - t0:.1f}s")
    return districts, time_feats


def save_checkpoint(model, path, trained_districts, best_epoch, seed):
    ckpt = dict(
        format=3,  # 3: 8 input channels and output = latest reading + change
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},   # weights only, no graph buffers
        config=dict(WINDOW=WINDOW, HORIZONS=list(HORIZONS), C_TAPS=C_TAPS, A_TAPS=A_TAPS, N_FEAT=N_FEAT,
                    HIDDEN=HIDDEN, attn=model.attn_kind, k=getattr(model.attn, "k", None)),
        input_channels=list(INPUT_CHANNELS),   # last axis of X, in order
        trained_districts=list(trained_districts),
        best_epoch=best_epoch, seed=seed,
        torch=str(torch.__version__),                    # str(): the raw version object can't be loaded safely
        saved=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    torch.save(ckpt, path)


def load_trained(path, device="cpu"):
    """Checkpoint -> model (eval mode), plus every district and the time features on `device`."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)     # plain tensors/dicts only: safe to load
    assert ckpt.get("format") == 3, "checkpoint is from an older version of this file (other inputs/outputs)"
    cfg = ckpt["config"]
    mine = dict(WINDOW=WINDOW, HORIZONS=list(HORIZONS), C_TAPS=C_TAPS, A_TAPS=A_TAPS, N_FEAT=N_FEAT, HIDDEN=HIDDEN)
    bad = {k: (cfg[k], mine[k]) for k in mine if cfg[k] != mine[k]}
    assert not bad, f"constants differ from the checkpoint (checkpoint, this file): {bad}"
    device = torch.device(device)
    model = GTCNN(cfg["attn"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    districts, time_feats = load_all(device)
    return model.eval(), districts, time_feats


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


class GTCNN(nn.Module):
    """Spatial attention, then a space-time filter layer, two hidden layers and one shared head
    that predicts every horizon. Every node runs the exact same network with the same weights,
    and the graph is an input, so one model serves every district."""

    def __init__(self, attn: str = ATTN):
        super().__init__()
        self.attn_kind = attn
        self.attn = {"district": lambda: DistrictGATAttention(N_CHANNELS),
                     "shared": lambda: SharedGATAttention(N_CHANNELS),
                     "none": lambda: None}[attn]()
        self.filter1 = StructuredGraphFilter(N_CHANNELS, N_FEAT)
        self.fc = nn.Linear(WINDOW * N_FEAT, HIDDEN)  # <-- extra hidden layer
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)  # <-- extra hidden layer
        self.fc3 = nn.Linear(HIDDEN, HIDDEN)  # <-- extra hidden layer
        self.head = nn.Linear(HIDDEN, len(HORIZONS))  # the same head at every node, one output per horizon

    def forward(self, x: torch.Tensor, S: torch.Tensor, reach: torch.Tensor) -> torch.Tensor:  # x: [B, WINDOW, N, C]
        last = x[:, -1, :, :1]  # [B, N, 1] latest intensity, before attention changes x
        if self.attn is not None:
            x = self.attn(x, reach)  # [B, T, N, C]
        z = torch.relu(self.filter1(x, S))  # [B, T, N, F]
        z = z.permute(0, 2, 1, 3).flatten(2)  # [B, N, T*F]
        z = torch.relu(self.fc(z))  # <-- hidden layer + activation
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        return last + self.head(z)  # [B, N, H] latest reading + predicted change per horizon


def masked_mae(pred, y, mask):
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp(min=1)


def chunked_linear(x, W, bias, chunks=512):  # x [..., K], W [F_out, K]
    K = x.shape[-1]
    if (x.numel() // K) % chunks:
        return F.linear(x, W) + bias
    out = torch.bmm(x.reshape(chunks, -1, K), W.t().unsqueeze(0).expand(chunks, -1, -1))
    return out.reshape(*x.shape[:-1], -1) + bias


# --------------------------------------------------------------- train

def train_epoch(model, districts, time_feats, optimizer, gen):
    """One pass over every training window of every district, batches of all districts shuffled
    together so no district is forgotten. Loss: masked MAE on standardised intensity, all horizons."""
    model.train()
    batches = []
    for di, d in enumerate(districts):
        o = d.origins["train"]
        o = o[torch.randperm(len(o), generator=gen).to(o.device)]
        batches += [(di, o[b:b + BATCH_SIZE]) for b in range(0, len(o), BATCH_SIZE)]
    total = torch.zeros((), device=time_feats.device)
    for i in torch.randperm(len(batches), generator=gen).tolist():
        di, ob = batches[i]
        d = districts[di]
        X, y, mask = gather(d, time_feats, ob)
        loss = masked_mae(model(X, d.S, d.reach), y, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.detach()
    return (total / len(batches)).item()  # one sync per epoch


@torch.no_grad()
def predict(model, d: District, time_feats, split):
    """Predictions, targets (both vehicles/hour) and genuine-target mask: [n_windows, N, H]."""
    model.eval()
    o = d.origins[split]
    preds, ys, masks = [], [], []
    for b in range(0, len(o), BATCH_SIZE):
        X, y, mask = gather(d, time_feats, o[b:b + BATCH_SIZE])
        preds.append(model(X, d.S, d.reach))
        ys.append(y)
        masks.append(mask)
    scale = lambda t: t * d.std[None, :, None] + d.mean[None, :, None]
    return scale(torch.cat(preds)), scale(torch.cat(ys)), torch.cat(masks)


def expected_targets(split):
    """Genuine targets the baselines score in `split`, per horizon, over every sensor."""
    sh = shared_data()
    o = sh["origins"][split]
    return [int(sh["observed"][o + h].sum()) for h in HORIZONS]


def score(model, districts, time_feats, split):
    """Pooled over every district: per horizon MAE and RMSE in vehicles/hour, plus per-district MAE."""
    abs_sum = torch.zeros(len(HORIZONS), device=time_feats.device, dtype=torch.float64)
    sq_sum, count = abs_sum.clone(), abs_sum.clone()
    per_district = {}
    for d in districts:
        pred, y, mask = predict(model, d, time_feats, split)
        m = mask.double()
        err = (pred - y).double() * m
        a, s, c = err.abs().sum((0, 1)), err.square().sum((0, 1)), m.sum((0, 1))
        abs_sum, sq_sum, count = abs_sum + a, sq_sum + s, count + c
        per_district[d.id] = (a / c.clamp(min=1)).tolist()
    count_l = count.long().tolist()
    if DISTRICTS is None:  # all districts together are exactly the sensors the baselines score
        assert count_l == expected_targets(split), \
            f"{split}: scored {count_l} targets, baselines score {expected_targets(split)}"
    mae = (abs_sum / count.clamp(min=1)).tolist()
    rmse = (sq_sum / count.clamp(min=1)).sqrt().tolist()
    return mae, rmse, count_l, per_district


def save_result(model_name: str, seed: int, results: dict):
    """Write this run's val and test scores to reports/graph_results.csv (same columns as the RQ1
    baselines, plus the seed), replacing any earlier run of the same model and seed."""
    path = DATA_DIR / "reports" / "graph_results.csv"
    fields = ["split", "model", "horizon_min", "MAE", "RMSE", "seed"]
    rows = []
    if path.exists():
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
    rows =[r for r in rows if (r["model"], r.get("seed", "")) != (model_name, str(seed))]
    for split, (mae, rmse) in results.items():
        for h, a, s in zip(HORIZONS, mae, rmse):
            rows.append({"split": split, "model": model_name, "horizon_min": h * 15,
                         "MAE": a, "RMSE": s, "seed": seed})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved to {path}")


def plot_sensors(model, d: District, time_feats, h_index=1):
    """Last week of test at 10 sensors of the district, `h_index` = 1 is 30 min ahead."""
    sh = shared_data()
    pred, y, mask = predict(model, d, time_feats, "test")
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

def main():
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn", choices=list(MODEL_NAMES), default=ATTN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    EPOCHS = args.epochs

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    districts, time_feats = load_all(device)
    model = GTCNN(args.attn).to(device)  # one model for every district
    model_name = MODEL_NAMES[args.attn]
    print(f"device={device} model='{model_name}' seed={args.seed} districts={[d.id for d in districts]} "
          f"sensors={sum(d.n for d in districts)} parameters={sum(p.numel() for p in model.parameters()):,}")
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(districts[0].origins[s]):,} windows")

    torch._dynamo.config.cache_size_limit = 64  # a few graphs per district size
    run_model = torch.compile(model) if COMPILE and not args.no_compile else model

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=Gamma_LR)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_val, best_epoch, best_state = float("inf"), 0, None
    train_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr = train_epoch(run_model, districts, time_feats, optimizer, gen)
        va_mae, _, _, _ = score(run_model, districts, time_feats, "val")
        va = float(np.mean(va_mae))  # model selection: val MAE in vehicles/hour, mean over horizons
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
    print(f"training time {train_seconds:.1f}s ({epoch} epochs, {train_seconds / epoch:.1f}s/epoch)")
    if device.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    model.load_state_dict(best_state)  # the epoch with the best validation score, never the test score
    print(f"best val mae {best_val:.2f} veh/h at epoch {best_epoch}")

    #Saving the model!
    tag = time.strftime("%Y%m%d_%H%M")
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = f"checkpoints/gtcnn_{args.attn}_seed{args.seed}_{tag}.pt"
    save_checkpoint(model, ckpt_path, [d.id for d in districts], best_epoch, args.seed)
    reloaded = GTCNN(args.attn).to(device)
    reloaded.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True)["state_dict"])
    model.eval(), reloaded.eval()
    with torch.no_grad():
        d = districts[0]
        X, _, _ = gather(d, time_feats, d.origins["train"][:8])
        assert torch.allclose(model(X, d.S, d.reach), reloaded(X, d.S, d.reach), atol=1e-6), \
            "checkpoint round-trip mismatch"
    print(f"saved {ckpt_path}")

    # final scores with the best weights, eager (no compile)
    results = {}
    for split in ("val", "test"):
        mae, rmse, count, per_district = score(model, districts, time_feats, split)
        results[split] = (mae, rmse)
    print(f"\n{model_name}, test, every window and sensor, genuine targets (vehicles/hour):")
    print(f"  {'horizon':>8} {'MAE':>8} {'RMSE':>8} {'targets':>10}")
    for h, a, s, c in zip(HORIZONS, *results["test"], count):
        print(f"  {h * 15:>5} min {a:8.2f} {s:8.2f} {c:10,d}")
    print(f"\n  test MAE per district: " + ", ".join(f"{k}: {v[0]:.1f}/{v[-1]:.1f}" for k, v in per_district.items())
          + f"  ({HORIZONS[0] * 15}/{HORIZONS[-1] * 15} min)")
    save_result(model_name, args.seed, results)

    if PLOT_FIGURES:
        for d in districts:
            plot_sensors(model, d, time_feats)


if __name__ == "__main__":
    main()
