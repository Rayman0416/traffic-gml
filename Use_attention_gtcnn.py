import attention_traffic_gtcnn as m
from pathlib import Path
import numpy as np
import torch


m.DATA_DIR = Path("data/forecast_dataset")
CKPT = "checkpoints/gtcnn_20261101_0141.pt"
trained = torch.load(CKPT, map_location="cpu", weights_only=True)["trained_districts"]
m.shared_data.cache_clear()          # otherwise the old folder's arrays are still served

results = []
for d in range(1, 22):
    model, cols, S, data = m.load_trained(CKPT, d, "mps")
    n = len(cols)
    rmse, rmse_p, mae = m.evaluate(model, data, cols, n, d, next(model.parameters()).device)
    results.append((d, n, rmse, rmse_p, mae))

print(f"\n{'district':>8} {'nodes':>6} {'trained':>8} {'rmse':>8} {'gain':>7} {'mae':>8}")
for d, n, rmse, rmse_p, mae in results:
    print(f"{d:8d} {n:6d} {'yes' if d in trained else 'no':>8} {rmse:8.2f} {1 - rmse / rmse_p:7.1%} {mae:8.2f}")


def pooled(rows):
    """Node-weighted RMSE (pooled via MSE), persistence RMSE, and MAE."""
    w = np.array([r[1] for r in rows], dtype=float)
    rmse = np.sqrt(np.sum(w * np.array([r[2] for r in rows]) ** 2) / w.sum())
    rmse_p = np.sqrt(np.sum(w * np.array([r[3] for r in rows]) ** 2) / w.sum())
    mae = np.sum(w * np.array([r[4] for r in rows])) / w.sum()
    return int(w.sum()), rmse, rmse_p, mae


print(f"\n{'subset':>10} {'nodes':>6} {'rmse':>8} {'gain':>7} {'mae':>8}")
for name, rows in [
    ("all", results),
    ("trained", [r for r in results if r[0] in trained]),
    ("untrained", [r for r in results if r[0] not in trained]),
]:
    if rows:
        n, rmse, rmse_p, mae = pooled(rows)
        print(f"{name:>10} {n:6d} {rmse:8.2f} {1 - rmse / rmse_p:7.1%} {mae:8.2f}")