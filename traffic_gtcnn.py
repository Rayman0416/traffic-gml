# Copy of the Lab 2.3 GTCNN (graph-time CNN) applied to GO-MO Madrid traffic.
# Takes district-1 intensity + occupancy + time-of-day (tod_sin, sin(t*pi/96)^2)
# over the last WINDOW steps and predicts the intensity 8 steps (2 hours) ahead. The network
# runs a K-tap graph filter on a Cartesian product graph over space x time, as in Lab 2.3.
#
# python traffic_gtcnn.py        # device=mps, batch size 16384
#
# Only reads data/forecast_dataset (built by `python -m preprocess.run`).

"""
test rmse/mae (30 min ahead, district 1):
  GTCNN        rmse 111.16  mae 60.09 vehicles/hour
  persistence  rmse 115.19  mae 64.44 vehicles/hour (predict current intensity)
"""


import csv
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("data/forecast_dataset")
DISTRICT = 1              # only district 1, for faster training
WINDOW = 5                # input window in timesteps (features.npz history_length)
HORIZON = 2               # target steps ahead (15 min per step, so 2 = 30 min)
N_TAPS = 3                # graph filter order
N_FEAT = 15                # hidden features per layer
BATCH_SIZE = int(16384*0.5)
EPOCHS = 60
LR = 5e-2


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
    """X [n, WINDOW, N, 4] input windows (intensity, occupancy, tod_sin, day_sin2),
    y [n, N] intensity at +2 hours (HORIZON steps), the genuine-target mask, and the target row indices
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
    targets = origins + HORIZON              # windows.csv stops at h4, so compute h8 directly
    ok = targets < len(intensity)            # drop windows whose target runs past the dataset end
    origins, targets = origins[ok], targets[ok]

    rows = origins[:, None] + np.arange(-WINDOW + 1, 1)[None, :]   # [n, WINDOW]
    X = np.empty(rows.shape + (len(cols), 4), dtype=np.float32)
    X[..., 0] = intensity[rows]                 # traffic intensity
    X[..., 1] = occupancy[rows]                 # occupancy
    X[..., 2] = tod_sin[rows][:, :, None]       # time-of-day, same at every node
    X[..., 3] = day_sin2[rows][:, :, None]      # sin(pi*t/96)^2 day bump
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
        self.filter1 = GraphFilter(S, 4, N_FEAT)   # channels: intensity, occupancy, tod_sin, sin(pi*t/96)^2
        self.filter2 = GraphFilter(S, N_FEAT, N_FEAT)
        self.filter3 = GraphFilter(S, N_FEAT, N_FEAT)
        self.head = nn.Linear(WINDOW * N_FEAT, 1)      # the same head at every node

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: [B, WINDOW, N, 4]
        z = x.reshape(x.shape[0], -1, x.shape[-1])                 # [B, WINDOW*N, C]
        z = torch.relu(self.filter1(z)) #was relu
        z = torch.relu(self.filter2(z)) #was relu
        z = torch.relu(self.filter3(z))     # [B, WINDOW*N, F]
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


def masked_mse(pred, y, mask):
    return ((pred - y) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def masked_mae(pred, y, mask):
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------- train

def run_epoch(model, X, y, mask, optimizer=None):
    model.train(optimizer is not None)
    total, mae_total, count = 0.0, 0.0, 0
    for b in range(0, len(X), BATCH_SIZE):
        xb, yb, mb = X[b:b + BATCH_SIZE], y[b:b + BATCH_SIZE], mask[b:b + BATCH_SIZE]
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(xb)
            loss = masked_mse(pred, yb, mb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * len(xb)
        mae_total += masked_mae(pred.detach(), yb, mb).item() * len(xb)   # metric only, MSE stays the loss
        count += len(xb)
    return total / count, mae_total / count


def main():
    torch.manual_seed(0)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    cols = load_sensors()
    n_sensors = len(cols)
    S = load_district_graph(n_sensors)

    # Cartesian product graph over space x time (Lab 2.3: s01 = s10 = 1, s00 = s11 = 0)
    S_time = torch.diag(torch.ones(WINDOW - 1), -1)                # past -> present chain
    S_product = torch.kron(torch.eye(WINDOW), S) + torch.kron(S_time, torch.eye(n_sensors))
    assert S_product.shape == (WINDOW * n_sensors, WINDOW * n_sensors)

    data = {s: tuple(t.to(device) for t in load_split_tensors(cols, s))
            for s in ("train", "val", "test")}
    print(f"device={device} district={DISTRICT} sensors={n_sensors} "
          f"space-time nodes={WINDOW * n_sensors} batch size={BATCH_SIZE}")
    for s, (X, y, *_) in data.items():
        print(f"  {s:5s} {len(X):,} windows")

    model = GTCNN(S_product.to(device), n_sensors).to(device)   # default init, seeded by manual_seed above
    n_params = sum(p.numel() for p in model.parameters())
    assert torch.isfinite(model(data["train"][0][:2])).all(), "non-finite output"
    print(f"  model parameters: {n_params:,}")
    check_node_shared(model, S_product.to(device), data["train"][0][:8], n_sensors)
    model = torch.compile(model)      # fuses the per-node ops, measured ~1.4x on top of the fused filter

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # smooth, gentle decay: lr * GAMMA^epoch (halves over ~69 epochs, never bottoms out)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    train_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_mse, tr_mae = run_epoch(model, *data["train"][:3], optimizer)
        va_mse, va_mae = run_epoch(model, *data["val"][:3])
        scheduler.step()
        print(f"epoch {epoch:2d}  train mse {tr_mse:.4f} mae {tr_mae:.4f}"
              f"  val mse {va_mse:.4f} mae {va_mae:.4f}"
              f"  {time.perf_counter() - t0:.1f}s/epoch  lr {scheduler.get_last_lr()[0]:.1e}",
              flush=True)
    train_seconds = time.perf_counter() - train_start
    print(f"training time {train_seconds:.1f}s ({EPOCHS} epochs, "
          f"{train_seconds / EPOCHS:.1f}s/epoch)")

    # test error in vehicles per hour, undoing the per-sensor standardisation
    p = np.load(DATA_DIR / "params.npz")
    mean = torch.from_numpy(p["scaler_mean"][cols]).to(device)
    std = torch.from_numpy(p["scaler_std"][cols]).to(device)
    X_test, y, mask, targets_test = data["test"]
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X_test[b:b + BATCH_SIZE])
                          for b in range(0, len(X_test), BATCH_SIZE)])
    pred = pred * std + mean
    y_phys = y * std + mean
    rmse = torch.sqrt(masked_mse(pred, y_phys, mask)).item()
    mae = masked_mae(pred, y_phys, mask).item()
    now = X_test[:, -1, :, 0] * std + mean       # persistence: traffic now = traffic at horizon
    rmse_persistence = torch.sqrt(masked_mse(now, y_phys, mask)).item()
    mae_persistence = masked_mae(now, y_phys, mask).item()
    print(f"test rmse/mae ({HORIZON * 15} min ahead, district {DISTRICT}):")
    print(f"  GTCNN        rmse {rmse:.2f}  mae {mae:.2f} vehicles/hour")
    print(f"  persistence  rmse {rmse_persistence:.2f}  mae {mae_persistence:.2f}"
          f" vehicles/hour (predict current intensity)")

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
