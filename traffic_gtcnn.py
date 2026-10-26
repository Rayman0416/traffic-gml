# Copy of the Lab 2.3 GTCNN (graph-time CNN) applied to GO-MO Madrid traffic.
# Takes intensity (one DISTRICT, or all of them) + occupancy + time-of-day (tod_sin, tod_cos)
# + time since the last real reading over the last WINDOW steps and predicts the intensity
# HORIZON steps ahead. The network
# runs a K-tap graph filter on a Cartesian product graph over space x time, as in Lab 2.3.
#
# Only reads data/forecast_dataset (built by `python -m preprocess.run`).

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("data/forecast_dataset")
DISTRICT = None           # None = every district in the dataset (full graph), or a district number
WINDOW = 6                # input window in timesteps (features.npz history_length)
HORIZON = 2               # target steps ahead up to 4 (1 = 15 min, 2 = 30 min, 3 = 45 min, 4 = 60 min)
N_TAPS = 3                # graph filter order
N_FEAT = 32               # hidden features per layer
BATCH_SIZE = 256          # full graph (1054 sensors), WINDOW = 6: lower it if the GPU runs out of memory
EPOCHS = 100
LR = 1e-3
LR_FACTOR = 0.5           # ReduceLROnPlateau: multiply lr by this when val stops improving
LR_PATIENCE = 5           # ReduceLROnPlateau: epochs without a val improvement before lowering lr
PATIENCE = 10             # early stopping: stop after this many epochs without a val improvement
MIN_DELTA = 1e-4          # smallest val loss drop that counts as an improvement
LOSS = "mae"             # training loss: "rmse" or "mae" (both are reported on test)
N_CHANNELS = 8            # intensity, occupancy, tod_sin, tod_cos, observed, dow_sin, dow_cos, age


# ---------------------------------------------------------------- data

def graph_files():
    """Node file, edge file and node index column for DISTRICT (None = full graph)."""
    if DISTRICT is None:
        return "nodes.csv", "edges_full.csv", "array_index"
    return f"nodes_district_{DISTRICT:02d}.csv", f"edges_district_{DISTRICT:02d}.csv", "local_index"


def district_label() -> str:
    return "all districts" if DISTRICT is None else f"district {DISTRICT}"


def load_district_graph(n_sensors: int) -> torch.Tensor:
    """Spatial shift operator of the route graph, normalised D^-1/2 A D^-1/2."""
    edges = csv.DictReader(open(DATA_DIR / "graph" / graph_files()[1]))
    A = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    for e in edges:
        i, j = int(e["target_index"]), int(e["source_index"])
        A[i, j] = A[j, i] = 1.0        # treat routes as undirected for influence
    d = A.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d)
    np.power(d, -0.5, out=d_inv_sqrt, where=d > 0)   # isolated sensors (degree 0) stay 0
    return torch.from_numpy(d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])


def load_sensors() -> np.ndarray:
    """Global feature column of every sensor in the graph (in local index order)."""
    node_file, _, key = graph_files()
    nodes = sorted(csv.DictReader(open(DATA_DIR / "graph" / node_file)), key=lambda r: int(r[key]))
    # nodes.csv indexes by array_index (= feature column), district files by global_index
    col = "global_index" if DISTRICT is not None else "array_index"
    return np.array([int(r[col]) for r in nodes], dtype=np.int64)


def load_split_tensors(cols: np.ndarray, split: str, routine_only: bool = True):
    """X [n, WINDOW, N, 8] input windows (intensity, occupancy, tod_sin, tod_cos, observed, dow_sin, dow_cos, age),
    y [n, N] intensity at HORIZON steps ahead, the genuine-target mask, and the target row indices
    (for plotting against the clock)."""
    z = np.load(DATA_DIR / "features.npz")
    intensity = z["intensity"][:, cols]        # [T, N] scaled intensity (gaps hold the causal fill)
    observed = z["intensity_mask"][:, cols]    # [T, N] genuine observations
    occupancy = z["occupancy"][:, cols]        # [T, N] scaled occupancy (gaps hold the causal fill)
    age = z["intensity_age"][:, cols]          # [T, N] steps since the last real reading / 96, 0 = real now

    # time features: tod_sin, tod_cos, dow_sin, dow_cos
    names = list(z["time_feature_names"])
    tf = z["time_features"]                       # [T, 4] tod_sin, tod_cos, dow_sin, dow_cos
    tod_sin = tf[:, names.index("tod_sin")]
    tod_cos = tf[:, names.index("tod_cos")]
    dow_sin = tf[:, names.index("dow_sin")]       # sin/cos(2*pi*dow/7), Monday = 0, constant per day
    dow_cos = tf[:, names.index("dow_cos")]
    assert np.allclose(tod_cos, np.cos(2 * np.pi * np.arange(len(tf)) / 96.0), atol=1e-6), \
        "tod_cos must be cos(2*pi*t/96): 1 at 00:00, -1 at noon"

    origins = np.array([int(r["origin_index"])
                        for r in csv.DictReader(open(DATA_DIR / "windows.csv"))
                        if r["split"] == split and (r["eligible_routine"] == "True" or not routine_only)],
                       dtype=np.int64)
    targets = origins + HORIZON              # target row index for each window
    ok = targets < len(intensity)            # drop windows whose target runs past the dataset end
    origins, targets = origins[ok], targets[ok]

    rows = origins[:, None] + np.arange(-WINDOW + 1, 1)[None, :]   # [n, WINDOW]
    X = np.empty(rows.shape + (len(cols), N_CHANNELS), dtype=np.float32)
    X[..., 0] = intensity[rows]                 # traffic intensity
    X[..., 1] = occupancy[rows]                 # occupancy
    X[..., 2] = tod_sin[rows][:, :, None]       # time-of-day, same at every node
    X[..., 3] = tod_cos[rows][:, :, None]
    X[..., 4] = observed[rows]                  # 1 = real reading, 0 = filled estimate
    X[..., 5] = dow_sin[rows][:, :, None]       # day-of-week, weekday vs weekend
    X[..., 6] = dow_cos[rows][:, :, None]
    X[..., 7] = age[rows]                       # how stale a filled value is: a 15 min gap vs a 6 h gap
    y = torch.from_numpy(intensity[targets])    # [n, N], filled where the mask is 0 (never scored)
    mask = torch.from_numpy(observed[targets])  # [n, N]
    return torch.from_numpy(X), y, mask, torch.from_numpy(targets)


# ------------------------------------------------------------ network

class GraphFilter(nn.Module):
    """One LSIGF layer from the GTCNN: y = sum_k S^k x W_k + b.
    S is kept sparse and S^k x is built as S(S(...x)), so no dense NT x NT powers are stored."""

    def __init__(self, S: torch.Tensor, n_feat_in: int, n_feat_out: int):
        super().__init__()
        self.register_buffer("S", S.to_sparse().coalesce())       # [NT, NT] sparse
        self.weight = nn.Parameter(torch.empty(N_TAPS, n_feat_in, n_feat_out))
        self.bias = nn.Parameter(torch.zeros(n_feat_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # x: [B, NT, F_in]
        B, NT, F = x.shape
        taps = [x]                                    # S^0 x = x, no multiply needed
        z = x.permute(1, 0, 2).reshape(NT, B * F)     # [NT, B*F]: one sparse matmul for the whole batch
        for _ in range(1, N_TAPS):
            z = torch.sparse.mm(self.S, z)            # S^k x = S (S^(k-1) x)
            taps.append(z.reshape(NT, B, F).permute(1, 0, 2))
        # all taps in one contraction: y = sum_k S^k x W_k + b (fewer MPS kernel launches)
        W = self.weight.permute(2, 0, 1).reshape(self.weight.shape[2], -1)
        return nn.functional.linear(torch.cat(taps, dim=-1), W) + self.bias


class GTCNN(nn.Module):
    """Three product-graph filter layers and one shared head. Every node runs the exact
    same network with the same weights: shared filters (neighbour mixing) plus the same
    per-node readout. No node-private parameters. The head predicts the change from the
    current reading (persistence), so an untrained network starts close to persistence."""

    def __init__(self, S: torch.Tensor, n_sensors: int):
        super().__init__()
        self.n_sensors = n_sensors
        self.filter1 = GraphFilter(S, N_CHANNELS, N_FEAT)   # channels: intensity, occupancy, tod_sin, tod_cos, observed, dow_sin, dow_cos, age
        self.filter2 = GraphFilter(S, N_FEAT, N_FEAT)
        self.filter3 = GraphFilter(S, N_FEAT, N_FEAT)
        self.filter4 = GraphFilter(S, N_FEAT, N_FEAT)
        self.head = nn.Linear(WINDOW * N_FEAT, 1)      # the same head at every node

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: [B, WINDOW, N, 8]
        z = x.reshape(x.shape[0], -1, x.shape[-1])                 # [B, WINDOW*N, C]
        z = torch.relu(self.filter1(z)) #was relu
        z = torch.relu(self.filter2(z)) #was relu
        z = torch.relu(self.filter3(z))     # [B, WINDOW*N, F]
        z = torch.relu(self.filter4(z))     # [B, WINDOW*N, F]
        z = z.reshape(x.shape[0], WINDOW, self.n_sensors, -1)      # [B, T, N, F]
        z = z.permute(0, 2, 1, 3).flatten(2)                       # [B, N, T*F]
        # residual on persistence: current scaled intensity (filled if missing) + learned change
        return x[:, -1, :, 0] + self.head(z).squeeze(-1)           # [B, N]


def check_node_shared(model, S_product, x, n_sensors):
    """Check that every node runs the exact same network: relabel the sensors (and the
    graph with them) and the outputs must relabel identically."""
    pi = torch.randperm(n_sensors, device=x.device)
    idx = (torch.arange(WINDOW, device=x.device)[:, None] * n_sensors + pi).flatten()
    twin = GTCNN(S_product[idx][:, idx], n_sensors).to(x.device)
    for a, b in zip(twin.parameters(), model.parameters()):
        a.data.copy_(b.data)
    assert torch.allclose(model(x)[:, pi], twin(x[:, :, pi]), atol=1e-5), "node sharing violated"


def masked_rmse(pred, y, mask):
    """Root mean squared error over the observed targets only (mask=1)."""
    return torch.sqrt((torch.square(pred - y) * mask).sum() / mask.sum().clamp(min=1))

def masked_mae(pred, y, mask):
    """Mean absolute error over the observed targets only (mask=1)."""
    return (torch.abs(pred - y) * mask).sum() / mask.sum().clamp(min=1)

LOSSES = {"rmse": masked_rmse, "mae": masked_mae}


# --------------------------------------------------------------- train

def run_epoch(model, X, y, mask, optimizer=None):
    model.train(optimizer is not None)
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for b in range(0, len(X), BATCH_SIZE):
        # data stays on the CPU (too big for the GPU at WINDOW = 6), one batch at a time goes to the device
        xb, yb, mb = (t[b:b + BATCH_SIZE].to(device) for t in (X, y, mask))
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(xb)
            loss = LOSSES[LOSS](pred, yb, mb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    return total / count


def save_result(model_name: str, mae: float, rmse: float):
    """Write this run's test score to reports/graph_results.csv (same columns as the RQ1 baselines),
    replacing any earlier run of the same model and horizon."""
    if DISTRICT is not None:
        model_name += f" (district {DISTRICT})"
    path = DATA_DIR / "reports" / "graph_results.csv"
    fields = ["split", "model", "horizon_min", "MAE", "RMSE"]
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    rows = [r for r in rows if (r["model"], int(r["horizon_min"])) != (model_name, HORIZON * 15)]
    rows.append({"split": "test", "model": model_name, "horizon_min": HORIZON * 15, "MAE": mae, "RMSE": rmse})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved to {path}")


def main():
    global HORIZON, EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=HORIZON, help="steps ahead, 1..4 (15 min each)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    HORIZON, EPOCHS = args.horizon, args.epochs

    torch.manual_seed(0) # reproducible starting point for model weights
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # check HORIZON is below/equal to 4 (preprocessed dataset limit)
    assert 1 <= HORIZON <= 4, "HORIZON must be between 1 and 4 (inclusive)"
    assert LOSS in LOSSES, f"LOSS must be one of {list(LOSSES)}"

    # load data
    cols = load_sensors()
    n_sensors = len(cols)
    S = load_district_graph(n_sensors)

    # Cartesian product graph over space x time (Lab 2.3: s01 = s10 = 1, s00 = s11 = 0)
    S_time = torch.diag(torch.ones(WINDOW - 1), -1) # past -> present chain
    S_product = torch.kron(torch.eye(WINDOW), S) + torch.kron(S_time, torch.eye(n_sensors))
    assert S_product.shape == (WINDOW * n_sensors, WINDOW * n_sensors)

    # load data in train/val/test splits
    # kept on the CPU: at WINDOW = 6 the full graph's windows are ~6.5 GB, run_epoch moves batches to the device
    # train/val on routine windows only; test on every window, so the score matches the RQ1 baselines
    data = {s: load_split_tensors(cols, s, routine_only=s != "test") for s in ("train", "val", "test")}
    print("device:", device)
    print(district_label(), "with", n_sensors, "sensors")

    # n = number of windows, N = number of sensors, 
    # 8 features per node (intensity, occupancy, tod_sin, tod_cos, observed, dow_sin, dow_cos, age)
    # X [n, WINDOW, N, 8] input windows
    # y [n, N] intensity at HORIZON steps ahead
    # mask [n, N] ground truth mask (1 = observed, 0 = missing) -> dont use missing targets to predict on
    # targets [n] row indices of the target timestep for each window
    for s, (X, y, mask, targets) in data.items():
        missing = (~mask).sum().item()
        print(s, "windows:", len(X), "missing targets:", missing, "of", mask.numel())
        assert torch.isfinite(X).all() and torch.isfinite(y).all(), f"{s}: non-finite data"

    model = GTCNN(S_product.to(device), n_sensors).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    check_node_shared(model, S_product.to(device), data["train"][0][:8].to(device), n_sensors) # TO:DO remove this check in the final version
    model = torch.compile(model) # fuses the per-node ops, measured ~1.4x on top of the fused filter

    # checks
    assert torch.isfinite(S).all(), "non-finite S"
    assert torch.isfinite(S_product).all(), "non-finite S_product"
    assert torch.isfinite(model(data["train"][0][:2].to(device))).all(), "non-finite output"

    # hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, threshold=MIN_DELTA, threshold_mode="abs")

    # train loop
    best_val, best_epoch, best_state = float("inf"), 0, None
    train_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, *data["train"][:3], optimizer)
        va = run_epoch(model, *data["val"][:3])
        scheduler.step(va)

        # keep the weights with the lowest val loss
        if va < best_val - MIN_DELTA:
            best_val, best_epoch = va, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}  train {LOSS} {tr:.4f}  val {LOSS} {va:.4f}"
              f"  {time.perf_counter() - t0:.1f}s/epoch  lr {optimizer.param_groups[0]['lr']:.1e}",
              flush=True)

        # early stopping: no real val improvement for PATIENCE epochs
        if epoch - best_epoch >= PATIENCE:
            print(f"early stop at epoch {epoch}: no val improvement > {MIN_DELTA} for {PATIENCE} epochs")
            break

    train_seconds = time.perf_counter() - train_start
    print(f"training time {train_seconds:.1f}s ({epoch} epochs, {train_seconds / epoch:.1f}s/epoch)")

    # evaluate the best-val epoch
    model.load_state_dict(best_state)
    print(f"best val {LOSS} {best_val:.4f} at epoch {best_epoch}")

    # test error in vehicles per hour, undoing the per-sensor standardisation
    p = np.load(DATA_DIR / "params.npz")
    mean = torch.from_numpy(p["scaler_mean"][cols]).to(device)
    std = torch.from_numpy(p["scaler_std"][cols]).to(device)
    X_test, y, mask, targets_test = data["test"]
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X_test[b:b + BATCH_SIZE].to(device)) for b in range(0, len(X_test), BATCH_SIZE)])
    y, mask = y.to(device), mask.to(device)
    X_now = X_test[:, -1].to(device)                          # [n, N, 8] last input step, for persistence

    # convert back to vehicles/hour
    pred = pred * std + mean
    y_phys = y * std + mean

    # persistence baseline: predict the current reading (last timestep of the input window) for every target,
    # scored only where that current reading is observed (a missing one holds the fill, not a measurement)
    now = X_now[..., 0] * std + mean
    mask_now = mask & (X_now[..., 4] > 0.5)

    # overview: test error in vehicles/hour, both metrics, GTCNN vs persistence
    rows = [("GTCNN", pred, mask, "all observed targets"),
            ("GTCNN", pred, mask_now, "current reading also observed"),
            ("persistence", now, mask_now, "current reading also observed")]
    print(f"\ntest overview ({HORIZON * 15} min ahead, {district_label()}, trained on {LOSS}, vehicles/hour):")
    print(f"  {'model':<12} {'RMSE':>8} {'MAE':>8} {'targets':>9}  subset")
    for name, p_, m_, subset in rows:
        print(f"  {name:<12} {masked_rmse(p_, y_phys, m_).item():8.2f} {masked_mae(p_, y_phys, m_).item():8.2f}"
              f" {m_.sum().item():9d}  {subset}")
    save_result("GTCNN", masked_mae(pred, y_phys, mask).item(), masked_rmse(pred, y_phys, mask).item())

    # ----------------- figures: 10 sensors, prediction vs truth, last week of the test set
    z = np.load(DATA_DIR / "features.npz")
    t_test = z["timestamps"][targets_test.cpu().numpy()]     # target time of each window
    sid_all = z["sensor_ids"]
    order = np.argsort(t_test)
    t_test = t_test[order]
    pred_np = pred.cpu().numpy()[order]
    y_np = y_phys.cpu().numpy()[order]
    y_np[mask.cpu().numpy()[order] == 0] = np.nan  # gap the truth line where target is missing

    week = t_test >= t_test[-1] - np.timedelta64(7, "D")   # zoom in: last week of the test set
    t_test, pred_np, y_np = t_test[week], pred_np[week], y_np[week]

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    for i in np.linspace(0, n_sensors - 1, 10).astype(int):  # 10 sensors spread over the district
        sid = sid_all[cols[i]]
        err = np.sqrt(np.nanmean((pred_np[:, i] - y_np[:, i]) ** 2))
        fig, ax = plt.subplots(figsize=(14, 3.5))
        ax.plot(t_test, y_np[:, i], lw=0.8, color="steelblue", label="true")
        ax.plot(t_test, pred_np[:, i], lw=0.8, alpha=0.85, color="darkorange",
                label=f"predicted ({HORIZON * 15} min ahead)")
        ax.set_title(f"sensor {sid} - last week of test, RMSE {err:.1f} vehicles/hour")
        ax.set_ylabel("vehicles/hour")
        ax.legend(loc="upper right")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(fig_dir / f"sensor_{sid}.png", dpi=250)
        plt.close(fig)
    print(f"  figures: {fig_dir}/sensor_*.png (10 sensors, last week of the test set)")


if __name__ == "__main__":
    main()
