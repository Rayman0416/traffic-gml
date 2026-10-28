# MLP baseline. Run from the repo root:  python rq1/NN.py [--seed 0] [--data-dir data/forecast_dataset]
# Writes val and test MAE/RMSE per horizon, plus parameters and timings, to <data_dir>/reports/nn_results.csv

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data-dir", default="data/forecast_dataset")
parser.add_argument("--epochs", type=int, default=1000)  # upper bound, early stopping usually stops earlier
args = parser.parse_args()

data_dir = args.data_dir
seed = args.seed
torch.manual_seed(seed)
gen = torch.Generator().manual_seed(seed)  # shuffling, on the CPU like the graph models
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

nbr_timestamps = 6
epochs = args.epochs
patience = 25      # stop after this many epochs without val improvement, same as the graph models
min_delta = 0.01   # vehicles/hour; smaller val improvements do not count
batch_size = 2048
learning_rate = 0.001
gamma_lr = 0.985   # LR = learning_rate * gamma_lr ** epochs, same as the graph models
horizons = {"h1": 15, "h2": 30, "h3": 45, "h4": 60}


def sync():
    # wait for the GPU, so the timings are real
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

features = np.load(f"{data_dir}/features.npz")
params = np.load(f"{data_dir}/params.npz")  
windows = pd.read_csv(f"{data_dir}/windows.csv")

intensity = features["intensity"]       
mask = features["intensity_mask"].astype(bool)       
intensity = np.where(mask, intensity, np.nan) 

timestamps = pd.DatetimeIndex(features["timestamps"])
mean = params["scaler_mean"]
std = params["scaler_std"]
nbr_sensors = intensity.shape[1]

slot = timestamps.hour.to_numpy() * 4 + timestamps.minute.to_numpy() // 15
time_of_week = timestamps.dayofweek.to_numpy() * 96 + slot   # weekday x 15-min slot, 0..671

history_steps = np.arange(-5, 1)   # the 6 input steps up to and including the origin

def history_rows(origins):
    # standardised intensity over the 6-step input window: [windows, 6, N], gaps -> 0
    return np.nan_to_num(intensity[origins[:, None] + history_steps[None, :]])

target_indexes = [f"target_index_{h}" for h in horizons]   

train_w = windows[windows["split"] == "train"]
test_w = windows[windows["split"] == "test"]
val_w = windows[windows["split"] == "val"]

train_origins = train_w["origin_index"].to_numpy()
train_targets = train_w[target_indexes].to_numpy()      
# train_origins, train_targets = train_origins[::10], train_targets[::10]  # train only every 10th window to make it quicker

test_origins = test_w["origin_index"].to_numpy()
test_targets = test_w[target_indexes].to_numpy()

val_origins = val_w["origin_index"].to_numpy()
val_targets = val_w[target_indexes].to_numpy()

nbr_features = 1 + 1 + nbr_timestamps 

class TrafficNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb = nn.Embedding(672, 8)
        self.layers = nn.Sequential(
            nn.Linear(nbr_features + 7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4) # outputs one prediction per horizon
        )

    def forward(self, x):
        t = self.time_emb(x[:, -1].long())              
        return self.layers(torch.cat([x[:, :-1], t], dim=1))


# The [T, N] arrays live on the GPU, like the graph models, and every batch is cut from them, so memory
# does not grow with windows x sensors. A sample is one (window, sensor) pair with at least one measured target.
filled = torch.tensor(np.where(mask, intensity, 0.0), dtype=torch.float32, device=device)  # gaps -> 0
measured_all = torch.tensor(mask, device=device)
time_of_week_t = torch.tensor(time_of_week, device=device)
std_t = torch.tensor(std, dtype=torch.float32, device=device)
history_offsets = torch.arange(-(nbr_timestamps - 1), 1, device=device)


class Split:
    def __init__(self, origins, targets):
        keep = np.zeros((len(origins), nbr_sensors), dtype=bool)
        for k in range(len(horizons)):
            keep |= mask[targets[:, k]]
        self.origins = torch.tensor(origins, device=device)
        self.targets = torch.tensor(targets, device=device)  # [windows, H]
        self.samples = torch.tensor(np.flatnonzero(keep), device=device)  # window * N + sensor, row-major
        self.n = len(self.samples)

    def batch(self, idx):
        """X [B, 8] (last reading, 6-step history, time of week), y [B, H], measured [B, H], std [B, 1]."""
        window, sensor = self.samples[idx] // nbr_sensors, self.samples[idx] % nbr_sensors
        origin = self.origins[window]
        history = filled[origin[:, None] + history_offsets, sensor[:, None]]  # [B, 6]
        X = torch.cat([history[:, -1:], history, time_of_week_t[origin][:, None].float()], dim=1)
        t = self.targets[window]
        return X, filled[t, sensor[:, None]], measured_all[t, sensor[:, None]].float(), std_t[sensor][:, None]


train, val, test = (Split(w["origin_index"].to_numpy(), w[target_indexes].to_numpy()) for w in (train_w, val_w, test_w))


def loss_fn(pred, y, measured):
    return ((pred - y).abs() * measured).sum() / measured.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(split, chunk=2 ** 20):
    """Per horizon MAE and RMSE in vehicles/hour over every measured target of the split."""
    sums = torch.zeros(3, len(horizons), device=device, dtype=torch.float64)  # abs, squared, count
    for i in range(0, split.n, chunk):
        X, y, measured, std_s = split.batch(torch.arange(i, min(i + chunk, split.n), device=device))
        errors = ((model(X) - y) * std_s).double() * measured
        sums += torch.stack([errors.abs().sum(0), errors.square().sum(0), measured.sum(0)])
    n_measured = sums[2].clamp(min=1.0)
    return sums[0] / n_measured, (sums[1] / n_measured).sqrt(), sums[2].long()


model = TrafficNetwork().to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma_lr)
n_params = sum(p.numel() for p in model.parameters())
print(f"device {device}, {n_params:,} parameters, {train.n:,} training samples")


n_train = train.n
lowest_mae = float("inf")
best_state = None
best_epoch = 0
epoch_times = []
sync()
train_start = time.perf_counter()

for epoch in range(epochs):
    epoch_start = time.perf_counter()
    model.train()
    order = torch.randperm(n_train, generator=gen).to(device)  # on the GPU it needs ~5 GB scratch for the full dataset

    for i in range(0, n_train, batch_size):
        X_batch, y_batch, measured_batch, _ = train.batch(order[i:i + batch_size])

        opt.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch, measured_batch)
        loss.backward()
        opt.step()

    # evaluatoin on validation set
    model.eval()
    val_mae, val_rmse, _ = evaluate(val)
    val_score = val_mae.mean().item()  # model selection like the graph models: veh/h, mean over horizons

    sync()
    epoch_times.append(time.perf_counter() - epoch_start)
    scheduler.step()

    if val_score < lowest_mae - min_delta:
        lowest_mae = val_score
        best_epoch = epoch + 1
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Validation MAE: {val_score:.2f} veh/h [{' '.join(f'{v:.1f}' for v in val_mae.tolist())}], "
        f"Validation RMSE: {val_rmse.mean().item():.2f} veh/h, "
        f"{epoch_times[-1]:.1f}s, lr {scheduler.get_last_lr()[0]:.1e}"
        f"{'  *' if best_epoch == epoch + 1 else ''}"
    )
    
    if epoch + 1 - best_epoch >= patience:
        print(f"early stop at epoch {epoch + 1}: no val improvement > {min_delta} veh/h for {patience} epochs")
        break

train_seconds = time.perf_counter() - train_start
peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
print(f"training time {train_seconds:.1f}s, best epoch {best_epoch}, peak GPU memory {peak_gb:.2f} GB")

# save checkpoints
model.load_state_dict(best_state)

model.eval()
results = {"val": evaluate(val)}  # val first, so the GPU is warm when test is timed (like the graph models)
sync()
t0 = time.perf_counter()
results["test"] = evaluate(test)  # every test window and sensor
sync()
infer_seconds = time.perf_counter() - t0

rows = []
for split, (split_mae, split_rmse, n_measured) in results.items():
    if split == "test":
        print(f"\nEvaluation results for {nbr_sensors} sensors (test):")
    for k, minutes in enumerate(horizons.values()):
        mae, rmse, n = split_mae[k].item(), split_rmse[k].item(), n_measured[k].item()
        rows.append({"split": split, "model": "MLP", "horizon_min": minutes, "MAE": mae, "RMSE": rmse,
                     "seed": seed, "n_params": n_params, "epochs": len(epoch_times),
                     "train_s": train_seconds, "epoch_s": float(np.median(epoch_times)), "infer_s": infer_seconds,
                     "peak_gb": peak_gb})
        if split == "test":
            print(f"{minutes} min:  MAE {mae:.1f}  RMSE {rmse:.1f} veh/h   (n = {n})")
print(f"test inference {infer_seconds:.3f}s for {len(test_origins):,} windows")

# one file with every seed; a rerun of the same seed replaces its rows
out = f"{data_dir}/reports/nn_results.csv"
old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
if "seed" in old:
    old = old[old["seed"] != seed]
else:
    old = pd.DataFrame()  # rows of the old MLP (no seed column) are dropped
pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(out, index=False)
print(f"saved {out}")