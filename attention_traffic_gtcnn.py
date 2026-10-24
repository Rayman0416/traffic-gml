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
from functools import lru_cache


DATA_DIR = Path("data/forecast_dataset")
DISTRICTS = [1, 4, 6]     # trained one after another, same weights; add as many as you like
EVAL_DISTRICTS = list(range(1, 22))      # evaluated on after training: districts 1-21
PLOT_FIGURES = True                      # False skips the 10 figures per district (210 files for 21)
EPOCHS_PER_DISTRICT = 2
COMPILE = True              # recompiles at every district switch (N changes); set False to compare
WINDOW = 4                # input window in timesteps (features.npz history_length)
HORIZON = 2               # target steps ahead (15 min per step, so 2 = 30 min)
C_TAPS = 2                # graph filter order (number of taps); 1 = self only
A_TAPS = 1                # attention neighbourhood in hops; 1 = 1-hop neighbours (+ self)
N_FEAT = 12               # Catination later features
HIDDEN = 12               # features after catination, these are much cheaper
BATCH_SIZE = int(2**10)
EPOCHS = 100 #time to beat 337 sec
LR = 5e-2
DISTRICT_K = 3     # picks per window

#  GTCNN        rmse 138.19  mae 69.61 vehicles/hour utan c taps
#  GTCNN        rmse 137.33  mae 69.34 vehicles/hour med c=2

# ---------------------------------------------------------------- data

def load_district_graph(n_sensors: int, district: int) -> torch.Tensor:
    """Spatial shift operator of the district route graph, normalised D^-1/2 A D^-1/2."""
    A = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    with open(DATA_DIR / "graph" / f"edges_district_{district:02d}.csv") as f:
        for e in csv.DictReader(f):
            i, j = int(e["target_index"]), int(e["source_index"])
            A[i, j] = A[j, i] = 1.0        # treat routes as undirected for influence
    d = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0).astype(np.float32)
    return torch.from_numpy(d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])

def load_district(district, device):
    cols = load_sensors(district)
    S = load_district_graph(len(cols), district).to(device)
    data = {s: tuple(t.to(device) for t in load_split_tensors(cols, s))
            for s in ("train", "val", "test")}
    return cols, S, data


def evaluate(model, data, cols, n_sensors, district, device):
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
    now = X_test[:, -1, :, 0] * std + mean
    rmse_persistence = torch.sqrt(masked_mse(now, y_phys, mask)).item()
    mae_persistence = masked_mae(now, y_phys, mask).item()
    print(f"test rmse/mae ({HORIZON * 15} min ahead, district {district}):")
    print(f"  GTCNN        rmse {rmse:.2f}  mae {mae:.2f} vehicles/hour")
    print(f"  persistence  rmse {rmse_persistence:.2f}  mae {mae_persistence:.2f}"
          f" vehicles/hour (predict current intensity)")

    z = np.load(DATA_DIR / "features.npz")
    t_test = z["timestamps"][targets_test.cpu().numpy()]
    sid_all = z["sensor_ids"]
    order = np.argsort(t_test)
    t_test = t_test[order]
    pred_np = pred.cpu().numpy()[order]
    y_np = y_phys.cpu().numpy()[order]
    y_np[mask.cpu().numpy()[order] == 0] = np.nan

    week = t_test >= t_test[-1] - np.timedelta64(7, "D")
    t_test, pred_np, y_np = t_test[week], pred_np[week], y_np[week]

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    for i in np.linspace(0, n_sensors - 1, 10).astype(int):
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
        fig.savefig(fig_dir / f"sensor_{sid}_district_{district}.png", dpi=250)
        plt.close(fig)
    print(f"  figures: {fig_dir}/sensor_*_district_{district}.png")
    return rmse, rmse_persistence


def load_sensors(district: int) -> np.ndarray:
    """Global feature column of every district-1 sensor (in local index order)."""
    with open(DATA_DIR / "graph" / f"nodes_district_{district:02d}.csv") as f:
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

class DistrictGATAttention(SharedGATAttention):
    """Dense neighbour attention plus K district picks as extra keys in the same softmax."""

    def __init__(self, S_space: torch.Tensor, c: int, hops: int = A_TAPS, k: int = DISTRICT_K):
        super().__init__(S_space, c, hops)
        self.k = k
        self.a_sel = nn.Linear(c, 1)                                   # importance score (5 params)

    def forward(self, x):  # x: [B, T, N, C]
        B, T, N, C = x.shape
        h = x[:, -1]
        s = self.a_src(h)  # [B, N, 1]
        t = self.a_dst(h).squeeze(-1)  # [B, N]
        u_top, pick = self.a_sel(h).squeeze(-1).topk(self.k, dim=1)  # [B, K]

        picked = torch.zeros(B, N, dtype=torch.bool, device=x.device).scatter(1, pick, True)
        gate = torch.ones(B, N, device=x.device, dtype=h.dtype).scatter(1, pick, torch.sigmoid(u_top))

        keep = self.mask[None] | picked[:, None, :]  # local keys OR picked sensors
        e = F.leaky_relu(s + t[:, None, :], 0.2).masked_fill(~keep, float("-inf"))
        alpha = F.softmax(e, dim=-1)  # [B, N, N]
        v = (self.v(x) * gate[:, None, :, None]).permute(0, 2, 1, 3).reshape(B, N, T * C)
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
        #self.attn = SharedGATAttention(S, 4, hops=A_TAPS) #old way without distict node!
        self.attn = DistrictGATAttention(S, 4, hops=A_TAPS) #New district node!
        self.filter1 = StructuredGraphFilter(S, 4, N_FEAT)   # channels: intensity, occupancy, tod_sin, sin(pi*t/96)^2
        self.fc = nn.Linear(WINDOW * N_FEAT, HIDDEN)   # <-- extra hidden layer
        self.head = nn.Linear(HIDDEN, 1)             # the same head at every node

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [B, WINDOW, N, 4]
        x = self.attn(x)                                      # [B, T, N, 4]
        z = torch.relu(self.filter1(x))                       # [B, T, N, F]
        z = z.permute(0, 2, 1, 3).flatten(2)                  # [B, N, T*F]
        z = torch.relu(self.fc(z))                  # <-- hidden layer + activation
        return self.head(z).squeeze(-1)                       # [B, N]

    def set_graph(self, S: torch.Tensor, n_sensors: int):
        """Swap in another district's graph. Weights are untouched; only graph buffers change."""
        self.n_sensors = n_sensors
        fresh = type(self.attn)(S, 4, hops=A_TAPS)               # rebuilds the attention's own buffers
        for name, buf in fresh.named_buffers():
            setattr(self.attn, name, buf.to(S.device))
        for f in (self.filter1, self.fc):
            f.S = S

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

def district_check(S, x, tol=1e-4):
    m = DistrictGATAttention(S, 4, hops=A_TAPS).to(x.device)
    xb = x[:64]
    with torch.no_grad():
        B, T, N, C = xb.shape
        h = xb[:, -1]
        s, t = m.a_src(h), m.a_dst(h).squeeze(-1)
        u_top, pick = m.a_sel(h).squeeze(-1).topk(m.k, dim=1)
        keep = m.mask[None].expand(B, -1, -1).clone()
        keep.scatter_(2, pick[:, None, :].expand(-1, N, -1), True)
        alpha = F.softmax(F.leaky_relu(s + t[:, None, :], 0.2).masked_fill(~keep, float("-inf")), dim=-1)
        gate = torch.ones(B, N, device=x.device, dtype=h.dtype)
        for b in range(B):                                             # plain loop: obviously correct
            gate[b, pick[b]] = torch.sigmoid(u_top[b])
        v = (m.v(xb) * gate[:, None, :, None]).permute(0, 2, 1, 3).reshape(B, N, T * C)
        want = xb + torch.bmm(alpha, v).reshape(B, N, T, C).permute(0, 2, 1, 3)
        err = (m(xb) - want).abs().max().item()
    print(f"district attention vs explicit softmax reference: max |diff| {err:.2e}")
    assert err < tol


def district_bench(S, x, reps=10):
    device, xb = x.device, x[:BATCH_SIZE]
    sync = (torch.mps.synchronize if device.type == "mps"
            else torch.cuda.synchronize if device.type == "cuda" else (lambda: None))

    def ms(mod):
        xi = xb.clone().requires_grad_(True)
        def run():
            xi.grad = None
            mod(xi).sum().backward()
        for _ in range(3): run()
        sync(); t0 = time.perf_counter()
        for _ in range(reps): run()
        sync()
        return (time.perf_counter() - t0) / reps * 1e3

    rows = [("dense_bmm", SharedGATAttention(S, 4, hops=A_TAPS))]
    rows += [(f"district K={k}", DistrictGATAttention(S, 4, hops=A_TAPS, k=k)) for k in (3, 5)]
    print("attention fwd+bwd [ms]        eager  compiled")
    for name, mod in rows:
        mod = mod.to(device)
        print(f"  {name:16s} {ms(mod):8.1f} {ms(torch.compile(mod)):8.1f}")


def main():
    torch.manual_seed(0)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    free = lambda: torch.mps.empty_cache() if device.type == "mps" else None

    cols, S, data = load_district(DISTRICTS[0], device)
    model = GTCNN(S, len(cols)).to(device)          # one model for every district
    print(f"device={device} districts={DISTRICTS} epochs/district={EPOCHS_PER_DISTRICT} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    current, run_model = None, model
    train_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        d = DISTRICTS[((epoch - 1) // EPOCHS_PER_DISTRICT) % len(DISTRICTS)]
        if d != current:                            # new block: one district in memory at a time
            if current is not None:
                data = None; free()
                cols, S, data = load_district(d, device)
                model.set_graph(S, len(cols))
            check_node_shared(model, S, data["train"][0][:8], len(cols))
            torch._dynamo.reset()                   # N changed: don't hit the recompile limit
            run_model = torch.compile(model) if COMPILE else model
            print(f"--- district {d}: {len(cols)} sensors, "
                  + ", ".join(f"{s} {len(data[s][0]):,}" for s in data) + " windows")
            current = d
        t0 = time.perf_counter()
        tr_mse, tr_mae = run_epoch(run_model, *data["train"][:3], optimizer)
        va_mse, va_mae = run_epoch(run_model, *data["val"][:3])
        scheduler.step()
        print(f"epoch {epoch:2d} d{d:<2d} train mse {tr_mse:.4f} mae {tr_mae:.4f}"
              f"  val mse {va_mse:.4f} mae {va_mae:.4f}"
              f"  {time.perf_counter() - t0:.1f}s/epoch  lr {scheduler.get_last_lr()[0]:.1e}",
              flush=True)
    train_seconds = time.perf_counter() - train_start
    print(f"training time {train_seconds:.1f}s ({EPOCHS} epochs, {train_seconds / EPOCHS:.1f}s/epoch)")

    # final test on every district with the one set of weights, loading one at a time (eager, no compile)
    # final test on every evaluation district with the one set of weights, one district in memory at a time
    results = []
    for d in EVAL_DISTRICTS:
        data = None;
        free()
        try:
            cols, S, data = load_district(d, device)
        except FileNotFoundError as e:
            print(f"district {d}: skipped ({e.filename} not found)")
            continue
        model.set_graph(S, len(cols))
        rmse, rmse_p = evaluate(model, data, cols, len(cols), d, device)
        results.append((d, rmse, rmse_p))

    print(f"\n{'district':>8} {'trained':>8} {'GTCNN':>8} {'persist':>8} {'gain':>7}   (rmse, vehicles/hour)")
    for d, rmse, rmse_p in results:
        print(f"{d:8d} {'yes' if d in DISTRICTS else 'no':>8} {rmse:8.2f} {rmse_p:8.2f} {1 - rmse / rmse_p:7.1%}")
    for label, flag in (("trained", True), ("unseen", False)):
        g = [1 - r / rp for d, r, rp in results if (d in DISTRICTS) == flag]
        if g:
            print(f"mean gain over persistence, {label} districts: {sum(g) / len(g):.1%}")

if __name__ == "__main__":
    main()