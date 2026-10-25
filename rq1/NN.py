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

train_origins = windows.loc[windows["split"] == "train", "origin_index"].to_numpy()
train_origins = train_origins[::10]       # train only every 10th window to make it quicker
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
    history = np.where(mask[steps], intensity[steps], 0.0).transpose(0,2,1)

    tow = np.repeat(time_of_week[origins + 1][:, None], nbr_sensors, axis=1)   
    X = np.concatenate([history[:, :, -1:], history, tow[:, :, None]], axis=2)     
    y = intensity[origins + 1]
    measured = mask[origins + 1]        
    stds = np.broadcast_to(std[None, :], measured.shape) 

    X, y, stds = X[measured], y[measured], stds[measured]

    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
            torch.tensor(stds, dtype=torch.float32).unsqueeze(1))

X_train, y_train, std_train = _create_input(train_origins)
X_eval, y_eval, std_eval = _create_input(test_origins)

model = TrafficNetwork()
loss_fn = nn.L1Loss()  # MAE
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_train = len(X_train)

for epoch in range(epochs):
    model.train()
    total_mae = 0.0
    total_mse = 0.0
    order = torch.randperm(n_train)                     

    for i in range(0, n_train, batch_size):
        idx = order[i:i + batch_size]
        X_batch, y_batch = X_train[idx], y_train[idx]

        opt.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        opt.step()

        errors = predictions - y_batch
        total_mae += torch.abs(errors).sum().item()
        total_mse += (errors ** 2).sum().item()

    average_mae = total_mae / n_train
    average_mse = total_mse / n_train
    print(f"Epoch {epoch + 1}/{epochs}, Training MAE: {average_mae:.4f}, training MSE: {average_mse:.4f}")


model.eval()

with torch.no_grad():
    predictions = model(X_eval)

err = (predictions - y_eval) * std_eval        
eval_mae = err.abs().mean().item()
eval_rmse = (err ** 2).mean().sqrt().item()

print(f"\nEvaluation results for {nbr_sensors} sensors (15 min ahead, test):")
print(f"MAE:  {eval_mae:.1f} veh/h")    
print(f"RMSE: {eval_rmse:.1f} veh/h")