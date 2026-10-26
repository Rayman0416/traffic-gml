import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

data_dir = "data/forecast_dataset"

nbr_timestamps = 6
epochs = 15
batch_size = 2048
learning_rate = 0.001
horizons = {"h1": 15, "h2": 30, "h3": 45, "h4": 60}

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


def _create_input(origins, targets):
    steps = origins[:, None] + np.arange(-(nbr_timestamps - 1), 1)[None, :] 
    history = np.where(mask[steps], intensity[steps], 0.0).transpose(0,2,1)

    tow = np.repeat(time_of_week[origins][:, None], nbr_sensors, axis=1)   

    X = np.concatenate([history[:, :, -1:], history, tow[:, :, None]], axis=2)     
    y = np.stack([intensity[targets[:, k]] for k in range(len(horizons))], axis=2) 
    measured = np.stack([mask[targets[:, k]] for k in range(len(horizons))], axis=2)   
    stds = np.broadcast_to(std[None, :], measured.shape[:2]) 

    y = np.where(measured, y, 0.0)

    keep = measured.any(axis=2)  
    X, y, measured, stds = X[keep], y[keep], measured[keep], stds[keep]

    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(measured, dtype=torch.float32),
            torch.tensor(stds, dtype=torch.float32).unsqueeze(1))

X_train, y_train, measured_train, std_train = _create_input(train_origins, train_targets)  
X_test, y_test, measured_test, std_test = _create_input(test_origins, test_targets)
X_val, y_val, measured_val, std_val = _create_input(val_origins, val_targets)

def loss_fn(pred, y, measured):
    return ((pred - y).abs() * measured).sum() / measured.sum().clamp(min=1.0)

model = TrafficNetwork()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_train = len(X_train)
lowest_mae = float("inf")
best_state = None

for epoch in range(epochs):
    model.train()
    order = torch.randperm(n_train)                     

    for i in range(0, n_train, batch_size):
        idx = order[i:i + batch_size]
        X_batch, y_batch, measured_batch = X_train[idx], y_train[idx], measured_train[idx]

        opt.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch, measured_batch)
        loss.backward()
        opt.step()

    # evaluatoin on validation set
    model.eval()
    with torch.no_grad():
        predictions = model(X_val)

        errors = predictions - y_val
        val_mae = ((errors.abs() * measured_val).sum() / measured_val.sum().clamp(min=1.0)).item()
        val_rmse = (((errors.square() * measured_val).sum() / measured_val.sum().clamp(min=1.0)).sqrt()).item()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Validation MAE: {val_mae:.4f}, "
        f"Validation RMSE: {val_rmse:.4f}"
    )

    if val_mae < lowest_mae:
        lowest_mae = val_mae
        best_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

# save checkpoints
model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    predictions = model(X_test)

err = (predictions - y_test) * std_test        

print(f"\nEvaluation results for {nbr_sensors} sensors (test):")
for k, minutes in enumerate(horizons.values()):
    e = err[:, k][measured_test[:, k].bool()]             
    print(f"{minutes} min:  MAE {e.abs().mean().item():.1f}  "
          f"RMSE {e.pow(2).mean().sqrt().item():.1f} veh/h   (n = {len(e)})")