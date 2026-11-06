import argparse
import csv
import time
from pathlib import Path
from functools import lru_cache
import datetime
import matplotlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

matplotlib.use("Agg")

"""Shared-weight GTCNN trained sequentially, one district in memory at a time.

Each district has its own local road graph and local spatial attention. Districts do
not exchange tokens, features, predictions, or gradients. The same GTCNN weights are
carried from one district to the next.
"""

# ---------------------------------------------------------------- hyperparameters
DATA_DIR = Path("data/forecast_dataset")
DISTRICTS = [4, 3, 2]              # sequential training order
EVAL_DISTRICTS = list(range(1, 22))
WINDOW = 6
HORIZONS = 2                       # 2 -> 30 min ahead; tuple also supported
C_TAPS = 3
A_TAPS = 1
N_FEAT = 24
HIDDEN = 32
DISTRICT_K = 5
ATTN = "district"                  # "district", "shared", or "none"

EPOCHS = 1000                     # total epoch budget across sequential district blocks
EPOCHS_PER_DISTRICT = 40          # switch district after this many epochs
LR = 1e-2
WEIGHT_DECAY = 1e-4
PATIENCE = 160                     # early stopping patience, reset for each district block
MIN_DELTA = 0.01                  # validation MAE improvement, vehicles/hour

# ReduceLROnPlateau monitors this district's validation MAE. Scheduler resets on district switch.
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 8
PLATEAU_THRESHOLD = MIN_DELTA
PLATEAU_COOLDOWN = 2
MIN_LR = 1e-5

SEED = 0
COMPILE = True
BATCH_SIZE = int(2 ** 9)
TRAIN_WINDOW_SIZE = 2048*8           # consecutive circular training origins per epoch
GPU_MARGIN_GB = 0.3
PLOT_FIGURES = False
TIME_FEATURES = ("tod_sin", "tod_cos", "dow_sin", "dow_cos")
INPUT_CHANNELS = ("intensity", "occupancy", "intensity_mask", "intensity_age") + TIME_FEATURES
N_CHANNELS = len(INPUT_CHANNELS)
HORIZONS = (HORIZONS,) if isinstance(HORIZONS, int) else tuple(HORIZONS)
DEFAULT_SIZE = (WINDOW, N_FEAT, HIDDEN)
MODEL_NAMES = {"district": "GTCNN + district attention", "shared": "GTCNN + neighbour attention", "none": "GTCNN"}

# ---------------------------------------------------------------- data

def district_ids():
    return list(DISTRICTS)


def load_graph(n_sensors: int, edge_file: str) -> torch.Tensor:
    """Symmetric, degree-normalised district road graph D^-1/2 A D^-1/2."""
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
    """Global feature columns for this district, in local-index order."""
    with open(DATA_DIR / "graph" / f"nodes_district_{district:02d}.csv") as f:
        nodes = sorted(csv.DictReader(f), key=lambda r: int(r["local_index"]))
    return np.array([int(r["global_index"]) for r in nodes], dtype=np.int64)


@lru_cache(maxsize=1)
def shared_data():
    """Load city-wide arrays and split origins once; district tensors are sliced on demand."""
    t0 = time.perf_counter()
    z = np.load(DATA_DIR / "features.npz")
    names = list(z["time_feature_names"])
    tf = z["time_features"][:, [names.index(n) for n in TIME_FEATURES]].astype(np.float32)
    with open(DATA_DIR / "windows.csv") as f:
        rows = list(csv.DictReader(f))
    origins = {s: np.array([int(r["origin_index"]) for r in rows if r["split"] == s], dtype=np.int64)
               for s in ("train", "val", "test")}
    p = np.load(DATA_DIR / "params.npz")
    sh = dict(intensity=z["intensity"].astype(np.float32),
              observed=z["intensity_mask"].astype(bool),
              occupancy=z["occupancy"].astype(np.float32),
              age=z["intensity_age"].astype(np.float32), time=tf, origins=origins,
              mean=p["scaler_mean"].astype(np.float32), std=p["scaler_std"].astype(np.float32),
              timestamps=z["timestamps"], sensor_ids=z["sensor_ids"])
    print(f"  shared data loaded once in {time.perf_counter() - t0:.1f}s")
    return sh


class District:
    """One district's graph and sensor arrays. Only the active district is put on device."""
    def __init__(self, district: int, device):
        t0 = time.perf_counter()
        sh = shared_data()
        self.id = int(district)
        self.cols = load_sensors(self.id)
        self.S = load_graph(len(self.cols), f"edges_district_{self.id:02d}.csv").to(device)
        self.n = len(self.cols)
        self.reach = neighbourhood(self.S, A_TAPS)
        take = lambda a: torch.from_numpy(np.ascontiguousarray(a[:, self.cols])).to(device)
        self.intensity = take(sh["intensity"])
        self.occupancy = take(sh["occupancy"])
        self.observed = take(sh["observed"])
        self.age = take(sh["age"])
        self.mean = torch.from_numpy(sh["mean"][self.cols]).to(device)
        self.std = torch.from_numpy(sh["std"][self.cols]).to(device)
        self.origins = {s: torch.from_numpy(o).to(device) for s, o in sh["origins"].items()}
        print(f"  loaded district {self.id} ({self.n} sensors) in {time.perf_counter() - t0:.1f}s")


def neighbourhood(S: torch.Tensor, hops: int) -> torch.Tensor:
    N = S.shape[0]
    R = ((S > 0) | torch.eye(N, dtype=torch.bool, device=S.device)).float()
    return torch.linalg.matrix_power(R, hops) > 0


@lru_cache(maxsize=1)
def window_offsets(device):
    return (torch.arange(-WINDOW + 1, 1, device=device), torch.tensor(HORIZONS, device=device))


def gather(dist: District, origins: torch.Tensor):
    """Build [B, WINDOW, N, C] inputs and standardised [B, N, H] targets."""
    hist, ahead = window_offsets(origins.device)
    rows = origins[:, None] + hist
    B, N = len(origins), dist.n
    X = torch.cat([dist.intensity[rows][..., None], dist.occupancy[rows][..., None],
                   dist.observed[rows][..., None].float(), dist.age[rows][..., None],
                   shared_time()[rows][:, :, None, :].expand(B, WINDOW, N, len(TIME_FEATURES))], dim=-1)
    targets = origins[:, None] + ahead
    y = dist.intensity[targets].transpose(1, 2)
    mask = dist.observed[targets].transpose(1, 2)
    return X, y, mask


@lru_cache(maxsize=1)
def shared_time():
    return torch.from_numpy(shared_data()["time"]).to(next_device())

# Device is explicitly set in main; cache the selected device without retaining a district.
_ACTIVE_DEVICE = torch.device("cpu")
def next_device():
    return _ACTIVE_DEVICE

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
    """Shared local GTCNN. It has no cross-district router or district-context pathway."""
    def __init__(self, attn: str = ATTN):
        super().__init__()
        self.attn_kind = attn
        self.attn = {"district": lambda: DistrictGATAttention(N_CHANNELS),
                     "shared": lambda: SharedGATAttention(N_CHANNELS),
                     "none": lambda: None}[attn]()
        self.filter1 = StructuredGraphFilter(N_CHANNELS, N_FEAT)
        self.fc = nn.Linear(WINDOW * N_FEAT, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, HIDDEN)
        self.head = nn.Linear(HIDDEN, len(HORIZONS))

    def encode(self, x: torch.Tensor, S: torch.Tensor, reach: torch.Tensor):
        last = x[:, -1, :, :1]
        if self.attn is not None:
            x = self.attn(x, reach)
        z = torch.relu(self.filter1(x, S))
        z = z.permute(0, 2, 1, 3).flatten(2)
        z = torch.relu(self.fc(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        return z, last

    def forward(self, x, S, reach):
        h, last = self.encode(x, S, reach)
        return last + self.head(h)


def masked_mae(pred, y, mask):
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp(min=1)


def chunked_linear(x, W, bias, chunks=512):
    K = x.shape[-1]
    if (x.numel() // K) % chunks:
        return F.linear(x, W) + bias
    out = torch.bmm(x.reshape(chunks, -1, K), W.t().unsqueeze(0).expand(chunks, -1, -1))
    return out.reshape(*x.shape[:-1], -1) + bias

CHUNK = BATCH_SIZE

def out_of_memory(e: BaseException) -> bool:
    while e is not None:
        if isinstance(e, torch.OutOfMemoryError):
            return True
        e = e.__cause__ or e.__context__
    return False


def chunked(o, step, reset):
    global CHUNK
    while True:
        try:
            for c in range(0, len(o), CHUNK):
                step(o[c:c + CHUNK])
            return
        except Exception as e:
            if CHUNK == 1 or not out_of_memory(e):
                raise
        reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        CHUNK = max(1, CHUNK // 2)
        print(f"  out of memory: {CHUNK} origins per forward pass from now on", flush=True)


# ---------------------------------------------------------------- train and validation

def train_epoch(model, dist, optimizer, epoch_index, encode):
    """Train this district on a circular window of TRAIN_WINDOW_SIZE origins."""
    model.train()
    all_origins = dist.origins["train"]
    n_origins = len(all_origins)
    if n_origins == 0:
        raise ValueError(f"district {dist.id} has no training origins")
    window_size = min(TRAIN_WINDOW_SIZE, n_origins)
    window_start = (epoch_index * TRAIN_WINDOW_SIZE) % n_origins
    indices = (torch.arange(window_size, device=all_origins.device) + window_start) % n_origins
    origins = all_origins[indices]
    total_abs = torch.zeros((), device=origins.device)
    total_valid = torch.zeros((), device=origins.device)
    _, ahead = window_offsets(origins.device)
    for b in range(0, len(origins), BATCH_SIZE):
        ob = origins[b:b + BATCH_SIZE]
        n_targets = dist.observed[ob[:, None] + ahead].sum().clamp(min=1)
        batch_abs = torch.zeros((), device=origins.device)
        def step(oc):
            X, y, mask = gather(dist, oc)
            pred = encode_model(model, X, dist, encode)
            error_sum = ((pred - y).abs() * mask).sum()
            (error_sum / n_targets).backward()
            batch_abs.add_(error_sum.detach())
        optimizer.zero_grad(set_to_none=True)
        chunked(ob, step, reset=lambda: (optimizer.zero_grad(set_to_none=True), batch_abs.zero_()))
        optimizer.step()
        total_abs.add_(batch_abs)
        total_valid.add_(n_targets)
    return (total_abs / total_valid.clamp(min=1)).item()

def encode_model(model, X, dist, encode):
    h, last = encode(X, dist.S, dist.reach)
    return last + model.head(h)


@torch.no_grad()
def score(model, dist, split, encode=None):
    """Full-split per-horizon MAE/RMSE in physical vehicles/hour."""
    model.eval()
    origins = dist.origins[split]
    H = len(HORIZONS)
    abs_sum = torch.zeros(H, device=dist.S.device)
    sq_sum = torch.zeros(H, device=dist.S.device)
    count = torch.zeros(H, device=dist.S.device)
    def step(oc):
        X, y, mask = gather(dist, oc)
        pred = encode_model(model, X, dist, encode or model.encode)
        err = (pred - y) * dist.std[None, :, None]
        abs_sum.add_((err.abs() * mask).sum((0, 1)))
        sq_sum.add_((err.square() * mask).sum((0, 1)))
        count.add_(mask.sum((0, 1)))
    chunked(origins, step, reset=lambda: (abs_sum.zero_(), sq_sum.zero_(), count.zero_()))
    mae = (abs_sum / count.clamp(min=1)).cpu().tolist()
    rmse = (sq_sum / count.clamp(min=1)).sqrt().cpu().tolist()
    return mae, rmse, count.long().cpu().tolist()


@torch.no_grad()
def evaluate_test(model, dist, make_plots=PLOT_FIGURES):
    """Test one district, including persistence baseline and optional sensor plots."""
    model.eval()
    origins = dist.origins["test"]
    H = len(HORIZONS)
    abs_sum = torch.zeros(H, device=dist.S.device)
    sq_sum = torch.zeros(H, device=dist.S.device)
    count = torch.zeros(H, device=dist.S.device)
    p_abs = torch.zeros(H, device=dist.S.device)
    p_sq = torch.zeros(H, device=dist.S.device)
    p_count = torch.zeros(H, device=dist.S.device)
    preds, truths, masks = [], [], []
    for b in range(0, len(origins), CHUNK):
        oc = origins[b:b + CHUNK]
        X, y, mask = gather(dist, oc)
        pred = encode_model(model, X, dist, model.encode)
        scale = dist.std[None, :, None]
        err = (pred - y) * scale
        persistence = (X[:, -1, :, 0, None] - y) * scale
        abs_sum += (err.abs() * mask).sum((0, 1))
        sq_sum += (err.square() * mask).sum((0, 1))
        count += mask.sum((0, 1))
        p_abs += (persistence.abs() * mask).sum((0, 1))
        p_sq += (persistence.square() * mask).sum((0, 1))
        p_count += mask.sum((0, 1))
        if make_plots:
            preds.append((pred * scale + dist.mean[None, :, None]).cpu())
            truths.append((y * scale + dist.mean[None, :, None]).cpu())
            masks.append(mask.cpu())
    mae = (abs_sum / count.clamp(min=1)).cpu().tolist()
    rmse = (sq_sum / count.clamp(min=1)).sqrt().cpu().tolist()
    pmae = (p_abs / p_count.clamp(min=1)).cpu().tolist()
    prmse = (p_sq / p_count.clamp(min=1)).sqrt().cpu().tolist()
    print(f"  district {dist.id} test:")
    for i, h in enumerate(HORIZONS):
        print(f"    {h * 15:>3} min: GTCNN MAE {mae[i]:.2f}, RMSE {rmse[i]:.2f} | "
              f"persistence MAE {pmae[i]:.2f}, RMSE {prmse[i]:.2f} vehicles/hour")
    if make_plots and preds:
        save_sensor_plots(dist, origins, torch.cat(preds), torch.cat(truths), torch.cat(masks))
    return mae, rmse, pmae, prmse, count.long().cpu().tolist()


def save_sensor_plots(dist, origins, pred, truth, mask):
    sh = shared_data()
    timestamps = sh["timestamps"][origins.cpu().numpy()[:, None] + np.array(HORIZONS)[None, :]]
    # For each horizon, use the corresponding target timestamp and plot the last seven days.
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    for hi, horizon in enumerate(HORIZONS):
        t = timestamps[:, hi]
        order = np.argsort(t)
        t = t[order]
        p = pred[order, :, hi].numpy()
        y = truth[order, :, hi].numpy()
        m = mask[order, :, hi].numpy().astype(bool)
        y[~m] = np.nan
        week = t >= t[-1] - np.timedelta64(7, "D")
        for i in np.linspace(0, dist.n - 1, min(10, dist.n)).astype(int):
            sid = sh["sensor_ids"][dist.cols[i]]
            err = np.sqrt(np.nanmean((p[week, i] - y[week, i]) ** 2))
            fig, ax = plt.subplots(figsize=(14, 3.5))
            ax.plot(t[week], y[week, i], lw=0.8, label="true")
            ax.plot(t[week], p[week, i], lw=0.8, alpha=0.85, label=f"predicted ({horizon * 15} min ahead)")
            ax.set_title(f"sensor {sid}, district {dist.id} - last week, RMSE {err:.1f} vehicles/hour")
            ax.set_ylabel("vehicles/hour")
            ax.legend(loc="upper right")
            fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(fig_dir / f"sensor_{sid}_district_{dist.id}_h{horizon}.png", dpi=200)
            plt.close(fig)


def save_checkpoint(model, path, trained_districts, epochs_completed, best_epochs):
    ckpt = dict(format=5,
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                config=dict(WINDOW=WINDOW, HORIZONS=list(HORIZONS), C_TAPS=C_TAPS, A_TAPS=A_TAPS,
                            N_FEAT=N_FEAT, HIDDEN=HIDDEN, ATTN=ATTN),
                input_channels=list(INPUT_CHANNELS), trained_districts=list(trained_districts),
                epochs=epochs_completed, epochs_per_district=EPOCHS_PER_DISTRICT,
                best_epochs=dict(best_epochs), torch=str(torch.__version__),
                saved=datetime.datetime.now().isoformat(timespec="seconds"))
    torch.save(ckpt, path)


def save_result_rows(results, seed, cost):
    """Append aggregate val/test horizon metrics to the standard graph_results.csv."""
    path = DATA_DIR / "reports" / "graph_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "model", "horizon_min", "MAE", "RMSE", "seed", "n_params", "epochs", "train_s", "epoch_s", "infer_s", "peak_gb"]
    rows = []
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))
    model_name = f"{MODEL_NAMES[ATTN]} (sequential districts, no exchange)"
    rows = [r for r in rows if (r.get("model"), r.get("seed")) != (model_name, str(seed))]
    for split, values in results.items():
        mae, rmse, _ = values
        for h, a, r in zip(HORIZONS, mae, rmse):
            rows.append(dict(split=split, model=model_name, horizon_min=h * 15, MAE=a, RMSE=r, seed=seed, **cost))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"  aggregate results saved to {path}")

# ---------------------------------------------------------------- main

def main():
    global EPOCHS, DATA_DIR, _ACTIVE_DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    EPOCHS, DATA_DIR = args.epochs, args.data_dir
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    _ACTIVE_DEVICE = device
    shared_time.cache_clear()
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(max(0.1, (free - GPU_MARGIN_GB * 1e9) / total))
        print(f"GPU memory cap {(free - GPU_MARGIN_GB * 1e9) / 1e9:.2f} GB of {total / 1e9:.2f} GB")

    model = GTCNN(ATTN).to(device)
    print(f"device={device} training districts={DISTRICTS} evaluation districts={EVAL_DISTRICTS} "
          f"epoch budget={EPOCHS} epochs/district block={EPOCHS_PER_DISTRICT} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")

    # AdamW: decay matrix/tensor weights, but not biases or one-dimensional parameters.
    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    param_groups = [{"params": decay_params, "weight_decay": WEIGHT_DECAY},
                    {"params": no_decay_params, "weight_decay": 0.0}]
    opt_kwargs = dict(lr=LR, betas=(0.9, 0.999), eps=1e-8)
    if device.type == "cuda":
        opt_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(param_groups, **opt_kwargs)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sync = (torch.cuda.synchronize if device.type == "cuda" else
            torch.mps.synchronize if device.type == "mps" else (lambda: None))
    use_compile = COMPILE and not args.no_compile
    epoch_times = []
    best_epochs = {}
    total_epochs_run = 0
    train_start = time.perf_counter()
    block_index = 0
    district_epochs_seen = {d: 0 for d in DISTRICTS}

    # One district is loaded at a time. Shared model weights and AdamW moments persist
    # across district switches; the plateau scheduler is reset for each district.
    while total_epochs_run < EPOCHS:
        district = DISTRICTS[block_index % len(DISTRICTS)]
        active_dist = District(district, device)
        if device.type == "cuda":
            torch._dynamo.reset()
        run_encode = torch.compile(model.encode) if use_compile else model.encode
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE,
            threshold=PLATEAU_THRESHOLD, threshold_mode="abs", cooldown=PLATEAU_COOLDOWN,
            min_lr=MIN_LR)
        best_val, best_state, best_epoch_local = float("inf"), None, 0
        print(f"\n--- district {district}: {active_dist.n} sensors; "
              f"train/val/test windows=" + "/".join(f"{len(active_dist.origins[s]):,}" for s in ("train", "val", "test")))

        for block_epoch in range(EPOCHS_PER_DISTRICT):
            if total_epochs_run >= EPOCHS:
                break
            t0 = time.perf_counter()
            tr_mae = train_epoch(model, active_dist, optimizer, district_epochs_seen[district], run_encode)
            district_epochs_seen[district] += 1
            va_mae, _, _ = score(model, active_dist, "val", run_encode)
            va = float(np.mean(va_mae))
            sync()
            epoch_times.append(time.perf_counter() - t0)
            scheduler.step(va)
            total_epochs_run += 1
            if va < best_val - MIN_DELTA:
                best_val = va
                best_epoch_local = block_epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"epoch {total_epochs_run:4d} d{district:<2d} block_epoch {block_epoch + 1:3d} "
                  f"train MAE {tr_mae:.4f} std  val MAE {va:7.2f} veh/h "
                  f"[{ ' '.join(f'{v:.1f}' for v in va_mae)}] "
                  f"{time.perf_counter() - t0:.1f}s/epoch  lr {optimizer.param_groups[0]['lr']:.1e}"
                  f"{'  *' if best_epoch_local == block_epoch + 1 else ''}", flush=True)
            if block_epoch + 1 - best_epoch_local >= PATIENCE:
                print(f"  district {district}: early stopping after {PATIENCE} epochs without a >{MIN_DELTA} veh/h improvement")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            best_epochs[district] = best_epoch_local
            print(f"  restored district {district}'s best validation weights from block epoch {best_epoch_local} (MAE {best_val:.2f})")
        del active_dist, run_encode, scheduler, best_state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
        block_index += 1

    sync()
    train_seconds = time.perf_counter() - train_start
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
    print(f"\ntraining time {train_seconds:.1f}s ({total_epochs_run} epochs, median {np.median(epoch_times):.1f}s/epoch)")
    print(f"peak GPU memory {peak_gb:.2f} GB; {CHUNK} origins per forward pass")

    tag = time.strftime("%Y%m%d_%H%M")
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = Path("checkpoints") / f"gtcnn_sequential_no_exchange_seed{args.seed}_{tag}.pt"
    save_checkpoint(model, ckpt_path, DISTRICTS, total_epochs_run, best_epochs)
    reloaded = GTCNN(ATTN).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(ckpt["state_dict"])
    reloaded.eval()
    print(f"saved {ckpt_path}; checkpoint architecture has no support router")

    # Final evaluation: load one district at a time, keeping only shared model weights.
    agg = {s: [torch.zeros(len(HORIZONS), dtype=torch.float64) for _ in range(3)] for s in ("val", "test")}
    district_rows = []
    infer_seconds = 0.0
    for d_id in EVAL_DISTRICTS:
        if device.type == "cuda": torch.cuda.empty_cache()
        elif device.type == "mps": torch.mps.empty_cache()
        try:
            dist = District(d_id, device)
        except FileNotFoundError as e:
            print(f"district {d_id}: skipped ({e.filename} not found)")
            continue
        sync(); t0 = time.perf_counter()
        mae_val, rmse_val, count_val = score(reloaded, dist, "val")
        sync()
        for hi, n in enumerate(count_val):
            agg["val"][0][hi] += mae_val[hi] * n
            agg["val"][1][hi] += (rmse_val[hi] ** 2) * n
            agg["val"][2][hi] += n

        sync(); t0 = time.perf_counter()
        mae, rmse, pmae, prmse, count_test = evaluate_test(
            reloaded, dist, make_plots=(PLOT_FIGURES and not args.no_plots))
        sync(); infer_seconds += time.perf_counter() - t0
        for hi, n in enumerate(count_test):
            agg["test"][0][hi] += mae[hi] * n
            agg["test"][1][hi] += (rmse[hi] ** 2) * n
            agg["test"][2][hi] += n
        district_rows.append((d_id, d_id in DISTRICTS, mae, rmse, pmae, prmse))
        del dist

    aggregate_results = {}
    for split in ("val", "test"):
        a, q, n = agg[split]
        aggregate_results[split] = ((a / n.clamp(min=1)).tolist(),
                                    (q / n.clamp(min=1)).sqrt().tolist(), n.long().tolist())
    print("\nAggregate evaluation across available evaluation districts (vehicles/hour):")
    print(f"  {'split':<8} {'horizon':>8} {'MAE':>10} {'RMSE':>10} {'targets':>12}")
    for split, (mae, rmse, count) in aggregate_results.items():
        for h, a, r, n in zip(HORIZONS, mae, rmse, count):
            print(f"  {split:<8} {h * 15:>5} min {a:10.2f} {r:10.2f} {n:12,d}")

    print("\nPer-district test results (vehicles/hour):")
    for d, trained, mae, rmse, pmae, prmse in district_rows:
        print(f"  d{d:02d} {'trained' if trained else 'unseen':>7}: "
              f"GTCNN MAE {mae[-1]:.2f} RMSE {rmse[-1]:.2f}; "
              f"persistence MAE {pmae[-1]:.2f} RMSE {prmse[-1]:.2f} ({HORIZONS[-1] * 15} min)")
    save_result_rows(aggregate_results, args.seed)


if __name__ == "__main__":
    main()
