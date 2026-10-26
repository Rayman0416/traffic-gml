# Non-graph neural network for RQ1: a small MLP shared by every sensor.
# For each (forecast origin, sensor) it reads that sensor's last WINDOW intensity readings and
# their observed flags, the current occupancy and staleness, the time features and a learned
# sensor embedding, and predicts the intensity 15, 30, 45 and 60 minutes ahead at once.
# No neighbours are used, so it is the non-graph counterpart of the GTCNN.
#
# Scored like rq1/baseline_nongraph.ipynb: every val/test window, every sensor, observed targets
# only, MAE and RMSE in vehicles/hour. Results go to reports/nn_results.csv.
#
# python NN.py

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

DATA_DIR = Path("data/forecast_dataset")
WINDOW = 6                # input steps up to and including the origin
HORIZONS = (1, 2, 3, 4)   # steps ahead: 15, 30, 45, 60 min
HIDDEN = 128
SENSOR_EMB_DIM = 8        # learned per-sensor embedding (0 = off)
BATCH_SIZE = 8192         # (origin, sensor) pairs per step
STEPS_PER_EPOCH = 1000
VAL_SAMPLES = 1_000_000   # fixed random (origin, sensor) pairs used for early stopping
EPOCHS = 40
LR = 1e-3
PATIENCE = 5
MIN_DELTA = 1e-4
EVAL_WINDOWS = 512        # windows per chunk when scoring the full val/test splits


class Data:
    """Feature arrays on the device, and gather() to build (origin, sensor) samples."""

    def __init__(self, device):
        z = np.load(DATA_DIR / "features.npz")
        t = lambda k, dtype=torch.float32: torch.as_tensor(z[k], dtype=dtype, device=device)
        self.intensity = t("intensity")                    # [T, N] standardised, gaps hold the fill
        self.observed = t("intensity_mask", torch.bool)    # [T, N] genuine readings
        self.occupancy = t("occupancy")
        self.age = t("intensity_age")                      # steps since last real reading / 96
        self.time_features = t("time_features")            # [T, 4]
        self.n_sensors = self.intensity.shape[1]
        self.hist = torch.arange(-WINDOW + 1, 1, device=device)
        self.ahead = torch.tensor(HORIZONS, device=device)

    def gather(self, o, s):
        """o, s [B] origin rows and sensor columns -> x [B, F], sensor ids, y [B, H], mask [B, H]."""
        rows, col = o[:, None] + self.hist, s[:, None]
        x = torch.cat([self.intensity[rows, col],
                       self.observed[rows, col].float(),
                       self.occupancy[o, s, None],
                       self.age[o, s, None],
                       self.time_features[o]], dim=1)
        targets = o[:, None] + self.ahead
        return x, s, self.intensity[targets, col], self.observed[targets, col]


class MLP(nn.Module):
    def __init__(self, n_in, n_sensors):
        super().__init__()
        self.emb = nn.Embedding(n_sensors, SENSOR_EMB_DIM) if SENSOR_EMB_DIM else None
        self.layers = nn.Sequential(
            nn.Linear(n_in + SENSOR_EMB_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, len(HORIZONS)),
        )

    def forward(self, x, s):
        h = torch.cat([x, self.emb(s)], dim=1) if self.emb is not None else x
        return x[:, WINDOW - 1, None] + self.layers(h)   # residual on the latest reading


def masked_mae(pred, y, mask):
    return (torch.abs(pred - y) * mask).sum() / mask.sum().clamp(min=1)


def pairs(origins, n_sensors):
    """Every (origin, sensor) pair of a set of windows."""
    return origins.repeat_interleave(n_sensors), torch.arange(n_sensors, device=origins.device).repeat(len(origins))


@torch.no_grad()
def score(model, d, origins, std):
    """MAE and RMSE in vehicles/hour per horizon over every observed target of these windows."""
    model.eval()
    abs_sum = torch.zeros(len(HORIZONS), device=origins.device, dtype=torch.float64)
    sq_sum, count = abs_sum.clone(), abs_sum.clone()
    for b in range(0, len(origins), EVAL_WINDOWS):
        o, s = pairs(origins[b:b + EVAL_WINDOWS], d.n_sensors)
        x, s, y, m = d.gather(o, s)
        err = (model(x, s) - y) * std[s, None]            # z-score error -> vehicles/hour
        abs_sum += (err.abs() * m).sum(0).double()
        sq_sum += (err.square() * m).sum(0).double()
        count += m.sum(0).double()
    return (abs_sum / count).cpu().numpy(), torch.sqrt(sq_sum / count).cpu().numpy(), count.cpu().numpy()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    d = Data(device)
    win = pd.read_csv(DATA_DIR / "windows.csv")
    origins = {s: torch.as_tensor(win.loc[win["split"] == s, "origin_index"].to_numpy(), device=device)
               for s in ("train", "val", "test")}
    print("device:", device, "|", d.n_sensors, "sensors |",
          ", ".join(f"{s}: {len(o)} windows" for s, o in origins.items()))

    g = torch.Generator(device=device).manual_seed(0)
    def sample(split, n):
        o = origins[split][torch.randint(len(origins[split]), (n,), device=device, generator=g)]
        return o, torch.randint(d.n_sensors, (n,), device=device, generator=g)
    val_x, val_s, val_y, val_m = d.gather(*sample("val", VAL_SAMPLES))

    model = MLP(val_x.shape[1], d.n_sensors).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val, best_epoch, best_state = float("inf"), 0, None
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        total = 0.0
        for _ in range(STEPS_PER_EPOCH):
            x, s, y, m = d.gather(*sample("train", BATCH_SIZE))
            loss = masked_mae(model(x, s), y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            va = masked_mae(model(val_x, val_s), val_y, val_m).item()
        if va < best_val - MIN_DELTA:
            best_val, best_epoch = va, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}  train mae {total / STEPS_PER_EPOCH:.4f}  val mae {va:.4f}"
              f"  {time.perf_counter() - t0:.1f}s", flush=True)
        if epoch - best_epoch >= PATIENCE:
            print(f"early stop at epoch {epoch}")
            break
    print(f"training time {time.perf_counter() - start:.1f}s, best val mae {best_val:.4f} at epoch {best_epoch}")
    model.load_state_dict(best_state)

    std = torch.as_tensor(np.load(DATA_DIR / "params.npz")["scaler_std"], dtype=torch.float32, device=device)
    rows = []
    for split in ("val", "test"):
        mae, rmse, count = score(model, d, origins[split], std)
        for h, a, r in zip(HORIZONS, mae, rmse):
            rows.append({"split": split, "model": "mlp", "horizon_min": h * 15, "MAE": a, "RMSE": r})
        print(f"{split}: scored {int(count.sum()):,} targets")
    results = pd.DataFrame(rows)
    print(results.round(1).to_string(index=False))
    out = DATA_DIR / "reports" / "nn_results.csv"
    results.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
