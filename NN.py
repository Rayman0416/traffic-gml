import torch 
import torch.nn as nn
from preprocess import features, windows
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# --------------------------------
# THIS IS THE SENSOR WE ARE PREDICTING FOR
sensor_idx = 30
# --------------------------------

nbr_sensors = 500
nbr_timestamps = 6
epochs = 50
batch_size = 64
learning_rate = 0.001
nbr_iterations = 10 # just to start 

f = features.load()
win = windows.load()

train_origins = win.loc[win["split"] == "train", "origin_index"].to_numpy()
eval_origins = win.loc[win["split"] == "val", "origin_index"].to_numpy()

train_data = features.window_features(f, train_origins)
eval_data = features.window_features(f, eval_origins)

data = features.window_features(f, train_origins)

nbr_features = 1 + 1 + 1 + nbr_timestamps 

class TrafficNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(nbr_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) 
        )

    def forward(self, x):
        return self.layers(x)


def _create_input(data, origins):
    intensity = data["intensity"][:, -1, sensor_idx]
    occupancy = data["occupancy"][:, -1, sensor_idx]
    time_sin = data["time_features"][:, 0]
    history = data["intensity"][:, -nbr_timestamps:, sensor_idx]

    X = np.column_stack([
        intensity,
        occupancy,
        time_sin,
        history,
    ])

    # the target time
    y = f["intensity"][origins + 1, sensor_idx]
    # remove missing observations
    valid = f["intensity_mask"][origins + 1, sensor_idx].astype(bool)
    valid &= data["intensity_mask"][:, -nbr_timestamps:, sensor_idx].all(axis=1)
    valid &= data["occupancy_mask"][:, -1, sensor_idx].astype(bool)
    X = X[valid]
    y = y[valid]

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )


X_train, y_train = _create_input(train_data, train_origins)
X_eval, y_eval = _create_input(eval_data, eval_origins)

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
print(f"MSE: {eval_mse:.4f}")
print(f"MAE: {eval_mae:.4f}")