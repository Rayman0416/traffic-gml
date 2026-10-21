# Copy of the Lab 2.3 GTCNN (graph-time CNN) applied to GO-MO Madrid traffic.
# Takes district-1 intensity + occupancy + time-of-day (tod_sin, sin(t*pi/96)^2)
# over the last WINDOW steps and predicts the intensity HORIZON steps ahead. The network
# runs a K-tap graph filter on a Cartesian product graph over space x time, as in Lab 2.3.
#
# Only reads data/forecast_dataset (built by `python -m preprocess.run`).

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
DISTRICT = 1              # only district 1, for faster training
WINDOW = 3                # input window in timesteps (features.npz history_length)
HORIZON = 2               # target steps ahead up to 4 (1 = 15 min, 2 = 30 min, 3 = 45 min, 4 = 60 min)
N_TAPS = 3                # graph filter order
N_FEAT = 12                # hidden features per layer
BATCH_SIZE = int(16384*2)
EPOCHS = 300
LR = 5e-3


# ---------------------------------------------------------------- data

def load_district_graph(n_sensors: int) -> torch.Tensor:
    """Spatial shift operator of the district route graph, normalised D^-1/2 A D^-1/2."""
    edges = csv.DictReader(open(DATA_DIR / "graph" / f"edges_district_{DISTRICT:02d}.csv"))
    A = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    for e in edges:
        i, j = int(e["target_index"]), int(e["source_index"])
        A[i, j] = A[j, i] = 1.0        # treat routes as undirected for influence
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0).astype(np.float32)
    return torch.from_numpy(d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])


def load_sensors() -> np.ndarray:
    """Global feature column of every district-1 sensor (in local index order)."""
    nodes = sorted(csv.DictReader(open(DATA_DIR / "graph" / f"nodes_district_{DISTRICT:02d}.csv")),
                   key=lambda r: int(r["local_index"]))
    return np.array([int(r["global_index"]) for r in nodes], dtype=np.int64)


def load_split_tensors(cols: np.ndarray, split: str):
    """X [n, WINDOW, N, 5] input windows (intensity, occupancy, tod_sin, day_sin2, observed),
    y [n, N] intensity at HORIZON steps ahead, the genuine-target mask, and the target row indices
    (for plotting against the clock)."""
    z = np.load(DATA_DIR / "features.npz")
    intensity = z["intensity"][:, cols]        # [T, N] scaled intensity
    observed = z["intensity_mask"][:, cols]    # [T, N] genuine observations
    occupancy = z["occupancy"][:, cols]        # [T, N] scaled occupancy (missing = 0)

    # tod_sin = sin(2*pi*slot/96) from the dataset (formula audited against the timestamps),
    # plus the day_sin2 = sin(pi*t/96)^2 bump on the absolute timestep: squaring gives an
    # exact 96-row period (no 48 h issue), 0 at 00:00, 1 at noon, same every day. It equals
    # (1 - tod_cos)/2 exactly (asserted below), so with tod_sin the pair carries the full
    # day phase again - noon and midnight are distinguishable. Selected by name, not position.
    names = list(z["time_feature_names"])
    tf = z["time_features"]                       # [T, 4] tod_sin, tod_cos, dow_sin, dow_cos
    tod_sin = tf[:, names.index("tod_sin")]
    tod_cos = tf[:, names.index("tod_cos")]
    day_sin2 = np.sin(np.pi * np.arange(len(tf)) / 96.0) ** 2
    assert np.allclose(day_sin2, (1.0 - tod_cos) / 2.0, atol=1e-6), "sin(pi*t/96)^2 must be (1-tod_cos)/2"
    assert abs(day_sin2[0]) < 1e-6 and abs(day_sin2[48] - 1) < 1e-6, "must be 0 at 00:00, 1 at noon"

    origins = np.array([int(r["origin_index"])
                        for r in csv.DictReader(open(DATA_DIR / "windows.csv"))
                        if r["split"] == split and r["eligible_routine"] == "True"],
                       dtype=np.int64)
    targets = origins + HORIZON              # target row index for each window
    ok = targets < len(intensity)            # drop windows whose target runs past the dataset end
    origins, targets = origins[ok], targets[ok]

    rows = origins[:, None] + np.arange(-WINDOW + 1, 1)[None, :]   # [n, WINDOW]
    X = np.empty(rows.shape + (len(cols), 5), dtype=np.float32)
    X[..., 0] = intensity[rows]                 # traffic intensity
    X[..., 1] = occupancy[rows]                 # occupancy
    X[..., 2] = tod_sin[rows][:, :, None]       # time-of-day, same at every node
    X[..., 3] = day_sin2[rows][:, :, None]      # sin(pi*t/96)^2 day bump
    X[..., 4] = observed[rows]                  # 1 = real reading, 0 = missing/interpolated (intensity stored as 0)
    y = torch.from_numpy(intensity[targets])    # [n, N]
    mask = torch.from_numpy(observed[targets])  # [n, N]
    return torch.from_numpy(X), y, mask, torch.from_numpy(targets)


# ------------------------------------------------------------ network

class GraphFilter(nn.Module):
    """One LSIGF layer from the GTCNN: y = sum_k S^k x W_k + b."""

    def __init__(self, S: torch.Tensor, n_feat_in: int, n_feat_out: int):
        super().__init__()
        powers = [torch.eye(S.shape[0], dtype=S.dtype, device=S.device)]
        for _ in range(N_TAPS - 1):
            powers.append(powers[-1] @ S)
        self.register_buffer("powers", torch.stack(powers))       # [N_TAPS, NT, NT]
        self.weight = nn.Parameter(torch.empty(N_TAPS, n_feat_in, n_feat_out))
        self.bias = nn.Parameter(torch.zeros(n_feat_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # x: [B, NT, F_in]
        taps = [x]                                    # S^0 x = x, no multiply needed
        for k in range(1, N_TAPS):
            taps.append(torch.einsum("nm,bmf->bnf", self.powers[k], x))
        # all taps in one contraction: y = sum_k S^k x W_k + b (fewer MPS kernel launches)
        W = self.weight.permute(2, 0, 1).reshape(self.weight.shape[2], -1)
        return nn.functional.linear(torch.cat(taps, dim=-1), W) + self.bias


class GTCNN(nn.Module):
    """Three product-graph filter layers and one shared head. Every node runs the exact
    same network with the same weights: shared filters (neighbour mixing) plus the same
    per-node readout. No node-private parameters."""

    def __init__(self, S: torch.Tensor, n_sensors: int):
        super().__init__()
        self.n_sensors = n_sensors
        self.filter1 = GraphFilter(S, 5, N_FEAT)   # channels: intensity, occupancy, tod_sin, sin(pi*t/96)^2, observed
        self.filter2 = GraphFilter(S, N_FEAT, N_FEAT)
        self.filter3 = GraphFilter(S, N_FEAT, N_FEAT)
        self.head = nn.Linear(WINDOW * N_FEAT, 1)      # the same head at every node

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: [B, WINDOW, N, 5]
        z = x.reshape(x.shape[0], -1, x.shape[-1])                 # [B, WINDOW*N, C]
        z = torch.relu(self.filter1(z))
        z = torch.relu(self.filter2(z))
        z = torch.relu(self.filter3(z))                            # [B, WINDOW*N, F]
        z = z.reshape(x.shape[0], WINDOW, self.n_sensors, -1)      # [B, T, N, F]
        z = z.permute(0, 2, 1, 3).flatten(2)                       # [B, N, T*F]
        return self.head(z).squeeze(-1)                            # [B, N]


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


# --------------------------------------------------------------- train

def run_epoch(model, X, y, mask, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    for b in range(0, len(X), BATCH_SIZE):
        xb, yb, mb = X[b:b + BATCH_SIZE], y[b:b + BATCH_SIZE], mask[b:b + BATCH_SIZE]
        with torch.set_grad_enabled(optimizer is not None):
            loss = masked_rmse(model(xb), yb, mb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    return total / count


def main():
    torch.manual_seed(0) # reproducible starting point for model weights
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # load data
    cols = load_sensors()
    n_sensors = len(cols)
    S = load_district_graph(n_sensors)

    # Cartesian product graph over space x time (Lab 2.3: s01 = s10 = 1, s00 = s11 = 0)
    S_time = torch.diag(torch.ones(WINDOW - 1), -1) # past -> present chain
    S_product = torch.kron(torch.eye(WINDOW), S) + torch.kron(S_time, torch.eye(n_sensors))
    assert S_product.shape == (WINDOW * n_sensors, WINDOW * n_sensors)

    # load data in train/val/test splits
    data = {s: tuple(t.to(device) for t in load_split_tensors(cols, s)) for s in ("train", "val", "test")}
    print("device:", device)
    print("district", DISTRICT, "with", n_sensors, "sensors")

    # n = number of windows, N = number of sensors, 
    # 5 features per node (intensity, occupancy, tod_sin, day_sin2, observed)
    # X [n, WINDOW, N, 5] input windows
    # y [n, N] intensity at HORIZON steps ahead
    # mask [n, N] ground truth mask (1 = observed, 0 = missing) -> dont use missing targets to predict on
    # targets [n] row indices of the target timestep for each window
    for s, (X, y, mask, targets) in data.items():
        missing = (~mask).sum().item()
        print(s, "windows:", len(X), "missing targets:", missing, "of", mask.numel())
        assert (y[~mask] == 0).all(), f"{s}: missing targets should be stored as 0"
        assert torch.isfinite(X).all() and torch.isfinite(y).all(), f"{s}: non-finite data"

    model = GTCNN(S_product.to(device), n_sensors).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")
    check_node_shared(model, S_product.to(device), data["train"][0][:8], n_sensors) # TO:DO remove this check in the final version
    model = torch.compile(model) # fuses the per-node ops, measured ~1.4x on top of the fused filter

    # checks
    assert torch.isfinite(S).all(), "non-finite S"
    assert torch.isfinite(S_product).all(), "non-finite S_product"
    assert torch.isfinite(model(data["train"][0][:2])).all(), "non-finite output"

    # hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # train loop
    best_val, best_epoch, best_state = float("inf"), 0, None
    train_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr = run_epoch(model, *data["train"][:3], optimizer)
        va = run_epoch(model, *data["val"][:3])
        scheduler.step()

        # keep the weights with the lowest val mse
        if va < best_val:
            best_val, best_epoch = va, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}  train mse {tr:.4f}  val mse {va:.4f}"
              f"  {time.perf_counter() - t0:.1f}s/epoch  lr {scheduler.get_last_lr()[0]:.1e}",
              flush=True)
        
    train_seconds = time.perf_counter() - train_start
    print(f"training time {train_seconds:.1f}s ({EPOCHS} epochs, {train_seconds / EPOCHS:.1f}s/epoch)")

    # evaluate the best-val epoch
    model.load_state_dict(best_state)
    print(f"best val mse {best_val:.4f} at epoch {best_epoch}")

    # test error in vehicles per hour, undoing the per-sensor standardisation
    p = np.load(DATA_DIR / "params.npz")
    mean = torch.from_numpy(p["scaler_mean"][cols]).to(device)
    std = torch.from_numpy(p["scaler_std"][cols]).to(device)
    X_test, y, mask, targets_test = data["test"]
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X_test[b:b + BATCH_SIZE]) for b in range(0, len(X_test), BATCH_SIZE)])

    # convert back to vehicles/hour
    pred = pred * std + mean
    y_phys = y * std + mean
    rmse = masked_rmse(pred, y_phys, mask).item()

    # persistence baseline: predict the current reading (last timestep of the input window) for every target
    now = X_test[:, -1, :, 0] * std + mean       
    mask_now = mask & (X_test[:, -1, :, 4] > 0.5)
    rmse_now = masked_rmse(pred, y_phys, mask_now).item()
    rmse_persistence = masked_rmse(now, y_phys, mask_now).item()
    print(f"test rmse ({HORIZON * 15} min ahead, district {DISTRICT}):")
    print(f"  GTCNN       {rmse:.2f} vehicles/hour (all observed targets)")
    print(f"  GTCNN       {rmse_now:.2f} vehicles/hour (current reading also observed: "
          f"{mask_now.sum().item()} of {mask.sum().item()} targets)")
    print(f"  persistence {rmse_persistence:.2f} vehicles/hour (predict current intensity, same subset)")

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
