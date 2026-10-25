import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

data_dir = "data/forecast_dataset"

sensor_idx = 30

nbr_timestamps = 6
epochs = 10
batch_size = 1000
learning_rate = 0.001
horizons = {"h1": 15, "h2": 30, "h3": 45, "h4": 60}

features = np.load(f"{data_dir}/features.npz")
params = np.load(f"{data_dir}/params.npz")
windows = pd.read_csv(f"{data_dir}/windows.csv")

intensity = features["intensity"]        # [T, N] standardised per sensor
mask = features["intensity_mask"]        # [T, N] True where genuinely observed

intensity = np.where(mask, intensity, np.nan) 

timestamps = pd.DatetimeIndex(features["timestamps"])
mean = params["scaler_mean"]
std = params["scaler_std"][sensor_idx]

slot = timestamps.hour.to_numpy() * 4 + timestamps.minute.to_numpy() // 15
time_of_week = timestamps.dayofweek.to_numpy() * 96 + slot   # weekday x 15-min slot, 0..671

history_steps = np.arange(-5, 1)   # the 6 input steps up to and including the origin

def history_rows(origins):
    # standardised intensity over the 6-step input window: [windows, 6, N], gaps -> 0
    return np.nan_to_num(intensity[origins[:, None] + history_steps[None, :]])

train_origins = windows.loc[windows["split"] == "train", "origin_index"].to_numpy()
test_origins = windows.loc[windows["split"] == "test", "origin_index"].to_numpy()

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
            nn.Linear(16, 1) 
        )

    def forward(self, x):
        t = self.time_emb(x[:, -1].long())              
        return self.layers(torch.cat([x[:, :-1], t], dim=1))


def _create_input(origins):
    steps = origins[:, None] + np.arange(-(nbr_timestamps - 1), 1)[None, :] 
    history = np.where(mask[steps, sensor_idx], intensity[steps, sensor_idx], 0.0)

    X = np.column_stack([history[:, -1], history, time_of_week[origins + 1]])
    y = intensity[origins + 1, sensor_idx]

    measured = mask[origins + 1, sensor_idx]        
    X, y = X[measured], y[measured]

    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1))

X_train, y_train = _create_input(train_origins)
X_eval, y_eval = _create_input(test_origins)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = TrafficNetwork()

loss_fn = nn.L1Loss()  # MAE
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_mae = 0.0
    total_mse = 0.0

    for X_batch, y_batch in train_loader:
        opt.zero_grad()

        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)

        loss.backward()
        opt.step()

        errors = predictions - y_batch
        total_mae += torch.abs(errors).sum().item()
        total_mse += (errors ** 2).sum().item()

    average_mae = total_mae / len(train_dataset)
    average_mse = total_mse / len(train_dataset)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Training MAE: {average_mae:.4f}, training MSE: {average_mse:.4f}"
    )


model.eval()

with torch.no_grad():
    predictions = model(X_eval)

    eval_mse = nn.functional.mse_loss(predictions, y_eval).item()
    eval_mae = nn.functional.l1_loss(predictions, y_eval).item()

print(f"\nEvaluation results for sensor {sensor_idx}:")
print(f"MAE:  {eval_mae * std:.1f} veh/h")
print(f"RMSE: {(eval_mse ** 0.5) * std:.1f} veh/h")