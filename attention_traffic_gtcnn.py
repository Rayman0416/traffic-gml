# Copy of the Lab 2.3 GTCNN (graph-time CNN) applied to GO-MO Madrid traffic.
# Takes district-1 intensity + occupancy + time-of-day (tod_sin, sin(t*pi/96)^2)
# over the last WINDOW steps and predicts the intensity HORIZON steps ahead (15 min per
# step, so HORIZON = 2 is 30 min). The network runs a K-tap graph filter on a Cartesian
# product graph over space x time, as in Lab 2.3, preceded by a GAT-style spatial
# attention layer that is restricted to nodes within N_TAPS hops.
#
# python traffic_gtcnn.py        # device=mps, batch size 32768
#
# Only reads data/forecast_dataset (built by `python -m preprocess.run`).
"""
test rmse/mae (30 min ahead, district 1):
  GTCNN        rmse 107.74  mae 58.27 vehicles/hour
  persistence  rmse 115.19  mae 64.44 vehicles/hour (predict current intensity)
  figures: figures/sensor_*.png (10 sensors, last week of the test set)
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
DISTRICT = 4              # only district 1, for faster training
WINDOW = 4                # input window in timesteps (features.npz history_length)
HORIZON = 2               # target steps ahead (15 min per step, so 2 = 30 min)
C_TAPS = 3                # graph filter order (number of taps); 1 = self only
A_TAPS = 1                # attention neighbourhood in hops; 1 = 1-hop neighbours (+ self)
N_FEAT = 12               # hidden features per layer
HIDDEN = 12
BATCH_SIZE = int(2**11)
EPOCHS = 200
LR = 5e-2


# ---------------------------------------------------------------- data

def load_district_graph(n_sensors: int) -> torch.Tensor:
    """Spatial shift operator of the district route graph, normalised D^-1/2 A D^-1/2."""
    A = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    with open(DATA_DIR / "graph" / f"edges_district_{DISTRICT:02d}.csv") as f:
        for e in csv.DictReader(f):
            i, j = int(e["target_index"]), int(e["source_index"])
            A[i, j] = A[j, i] = 1.0        # treat routes as undirected for influence
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0).astype(np.float32)
    return torch.from_numpy(d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])


def load_sensors() -> np.ndarray:
    """Global feature column of every district-1 sensor (in local index order)."""
    with open(DATA_DIR / "graph" / f"nodes_district_{DISTRICT:02d}.csv") as f:
        nodes = sorted(csv.DictReader(f), key=lambda r: int(r["local_index"]))
    return np.array([int(r["global_index"]) for r in nodes], dtype=np.int64)


def load_split_tensors(cols: np.ndarray, split: str):
    """X [n, WINDOW, N, 4] input windows (intensity, occupancy, tod_sin, day_sin2),
    y [n, N] intensity HORIZON steps ahead, the genuine-target mask, and the target row
    indices (for plotting against the clock)."""
    z = np.load(DATA_DIR / "features.npz")
    # float32 everywhere: MPS has no float64 support
    intensity = z["intensity"][:, cols].astype(np.float32)        # [T, N] scaled intensity
    observed = z["intensity_mask"][:, cols].astype(np.float32)    # [T, N] genuine observations
    occupancy = z["occupancy"][:, cols].astype(np.float32)        # [T, N] scaled occupancy (missing = 0)

    # tod_sin = sin(2*pi*slot/96) from the dataset (formula audited against the timestamps),
    # plus the day_sin2 = sin(pi*t/96)^2 bump on the absolute timestep: squaring gives an
    # exact 96-row period (no 48 h issue), 0 at 00:00, 1 at noon, same every day. It equals
    # (1 - tod_cos)/2 exactly (asserted below), so with tod_sin the pair carries the full
    # day phase again - noon and midnight are distinguishable. Selected by name, not position.
    names = list(z["time_feature_names"])
    tf = z["time_features"]                       # [T, 4] tod_sin, tod_cos, dow_sin, dow_cos
    tod_sin = tf[:, names.index("tod_sin")].astype(np.float32)
    tod_cos = tf[:, names.index("tod_cos")]
    day_sin2 = (np.sin(np.pi * np.arange(len(tf)) / 96.0) ** 2).astype(np.float32)
    assert np.allclose(day_sin2, (1.0 - tod_cos) / 2.0, atol=1e-5), "sin(pi*t/96)^2 must be (1-tod_cos)/2"
    assert abs(day_sin2[0]) < 1e-6 and abs(day_sin2[48] - 1) < 1e-6, "must be 0 at 00:00, 1 at noon"

    with open(DATA_DIR / "windows.csv") as f:
        origins = np.array([int(r["origin_index"]) for r in csv.DictReader(f)
                            if r["split"] == split and r["eligible_routine"] == "True"],
                           dtype=np.int64)
    targets = origins + HORIZON              # windows.csv has fixed horizons, so compute ours directly
    # drop windows whose input would start before the dataset (negative rows would wrap
    # around silently) or whose target runs past the dataset end
    ok = (origins >= WINDOW - 1) & (targets < len(intensity))
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

class SharedGATAttention(nn.Module):
    """GAT-style spatial attention restricted to nodes within `hops` hops (self included).
    Scores are e_ij = LeakyReLU(a_src . h_i + a_dst . h_j), computed once from the last
    time step and reused for every step in the window. Residual connection.
    No node-specific parameters, so it is permutation-equivariant over sensors."""

    def __init__(self, S_space: torch.Tensor, c: int, hops: int = A_TAPS):
        super().__init__()
        N = S_space.shape[0]
        # reachability on CPU (linalg ops are patchy on MPS); the buffer moves with the model
        R = ((S_space.cpu() > 0) | torch.eye(N, dtype=torch.bool)).float()    # edges + self loops
        reach = torch.linalg.matrix_power(R, hops) > 0                        # reachable in <= hops steps
        self.register_buffer("mask", reach)                                   # [N, N]
        self.a_src = nn.Linear(c, 1, bias=False)
        self.a_dst = nn.Linear(c, 1, bias=False)
        self.v = nn.Linear(c, c)

    def forward(self, x):                                           # x: [B, T, N, C]
        B, T, N, C = x.shape
        h = x[:, -1]
        e = F.leaky_relu(self.a_src(h) + self.a_dst(h).transpose(1, 2), 0.2)
        alpha = F.softmax(e.masked_fill(~self.mask, float("-inf")), dim=-1)    # [B, N, N]
        v = self.v(x).permute(0, 2, 1, 3).reshape(B, N, T * C)      # [B, N, T*C]
        out = torch.bmm(alpha, v).reshape(B, N, T, C).permute(0, 2, 1, 3)
        return x + out


class StructuredGraphFilter(nn.Module):
    """Same filter as GraphFilter, but S_product is never built:
    (I⊗S + S_time⊗I) x  =  S x_t + x_{t-1}   (per slice, plus a time shift)."""

    def __init__(self, S_space: torch.Tensor, n_in: int, n_out: int):
        super().__init__()
        self.register_buffer("S", S_space)                       # [N, N]
        self.weight = nn.Parameter(torch.empty(C_TAPS, n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.xavier_uniform_(self.weight)

    def product_step(self, x):                                   # x: [B, T, N, F]
        B, T, N, Fd = x.shape
        xs = x.permute(2, 0, 1, 3).reshape(N, -1)                # [N, B*T*F]
        spatial = (self.S @ xs).reshape(N, B, T, Fd).permute(1, 2, 0, 3)
        past = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return spatial + past

    def forward(self, x):                                        # [B, T, N, F_in]
        taps = [x]
        for _ in range(1, C_TAPS):
            taps.append(self.product_step(taps[-1]))             # S_product^k x, iteratively
        W = self.weight.permute(2, 0, 1).reshape(self.weight.shape[2], -1)
        return chunked_linear(torch.cat(taps, dim=-1), W, self.bias)


class GTCNN(nn.Module):
    """Spatial attention, then three space-time filter layers and one shared head.
    Every node runs the exact same network with the same weights."""

    def __init__(self, S: torch.Tensor, n_sensors: int):     # S: spatial [N, N]
        super().__init__()
        self.n_sensors = n_sensors
        self.attn = SharedGATAttention(S, 4, hops=A_TAPS)
        self.filter1 = StructuredGraphFilter(S, 4, N_FEAT)   # channels: intensity, occupancy, tod_sin, sin(pi*t/96)^2
        self.fc = nn.Linear(WINDOW * N_FEAT, HIDDEN)   # <-- extra hidden layer
        self.head = nn.Linear(HIDDEN, 1)             # the same head at every node

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, WINDOW, N, 4]
        x = self.attn(x)                                      # [B, T, N, 4]
        z = torch.relu(self.filter1(x))                       # [B, T, N, F]
        z = z.permute(0, 2, 1, 3).flatten(2)                  # [B, N, T*F]
        z = torch.relu(self.fc(z))                  # <-- hidden layer + activation
        return self.head(z).squeeze(-1)                       # [B, N]


def check_node_shared(model, S, x, n_sensors):
    """Relabel the sensors (and the graph with them); outputs must relabel identically."""
    pi = torch.randperm(n_sensors, device=x.device)
    twin = GTCNN(S[pi][:, pi], n_sensors).to(x.device)
    for a, b in zip(twin.parameters(), model.parameters()):
        a.data.copy_(b.data)
    assert torch.allclose(model(x)[:, pi], twin(x[:, :, pi]), atol=1e-4), "node sharing violated"


def masked_mse(pred, y, mask):
    return ((pred - y) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def masked_mae(pred, y, mask):
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------- train

def run_epoch(model, X, y, mask, optimizer=None):
    model.train(optimizer is not None)

    for b in range(0, len(X), BATCH_SIZE):
        xb, yb, mb = X[b:b + BATCH_SIZE], y[b:b + BATCH_SIZE], mask[b:b + BATCH_SIZE]
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(xb)
            loss = masked_mse(pred, yb, mb)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Zero syncs or accumulation in the loop. Return the raw tensors from the LAST batch.
    return loss.detach(), masked_mae(pred.detach(), yb, mb)

def chunked_linear(x, W, bias, chunks=512):          # x [..., K], W [F_out, K]
    K = x.shape[-1]
    if (x.numel() // K) % chunks:
        return F.linear(x, W) + bias
    out = torch.bmm(x.reshape(chunks, -1, K), W.t().unsqueeze(0).expand(chunks, -1, -1))
    return out.reshape(*x.shape[:-1], -1) + bias


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


    data = {s: tuple(t.to(device) for t in load_split_tensors(cols, s))
            for s in ("train", "val", "test")}
    print(f"device={device} district={DISTRICT} sensors={n_sensors} "
          f"space-time nodes={WINDOW * n_sensors} batch size={BATCH_SIZE}")
    for s, (X, y, *_) in data.items():
        print(f"  {s:5s} {len(X):,} windows")

    model = GTCNN(S.to(device), n_sensors).to(device)   # default init, seeded by manual_seed above
    n_params = sum(p.numel() for p in model.parameters())
    reach = model.attn.mask
    print(f"  attention: {A_TAPS} hops, max neighbourhood {int(reach.sum(1).max())} of {n_sensors} sensors, "
          f"mask density {reach.float().mean().item():.2f}")
    assert torch.isfinite(model(data["train"][0][:2])).all(), "non-finite output"
    print(f"  model parameters: {n_params:,}")
    check_node_shared(model, S.to(device), data["train"][0][:8], n_sensors)
    model = torch.compile(model)      # fuses the per-node ops, measured ~1.4x on top of the fused filter

    #graph_report(S, model, data["train"][0], n_sensors) #remove this line later. This is just diagnostics /Calle
    #filter_breakdown(device, n_sensors) #this too
    #linear_variants(device, n_sensors) #this too

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # smooth, gentle decay: lr * GAMMA^epoch (halves every ~34 epochs, ~0.13x after 100)
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
    mean = torch.from_numpy(p["scaler_mean"][cols].astype(np.float32)).to(device)
    std = torch.from_numpy(p["scaler_std"][cols].astype(np.float32)).to(device)
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
        fig.savefig(fig_dir / f"sensor_{sid}_district_{DISTRICT}.png", dpi=250)
        plt.close(fig)
    print(f"  figures: {fig_dir}/sensor_*.png (10 sensors, last week of the test set)")


if __name__ == "__main__":
    main()