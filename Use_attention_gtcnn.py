import attention_traffic_gtcnn as m
from pathlib import Path
import torch


m.DATA_DIR = Path("data/forecast_dataset")
CKPT = "checkpoints/gtcnn_post_20261025_1813.pt"
trained = torch.load(CKPT, map_location="cpu", weights_only=True)["trained_districts"]
m.shared_data.cache_clear()          # otherwise the old folder's arrays are still served

results = []
for d in range(1, 22):
    model, cols, S, data = m.load_trained(CKPT, d, "mps")
    rmse, rmse_p, mae = m.evaluate(model, data, cols, len(cols), d, next(model.parameters()).device)
    results.append((d, rmse, rmse_p, mae))

print(f"\n{'district':>8} {'trained':>8} {'rmse':>8} {'gain':>7} {'mae':>8}")
for d, rmse, rmse_p, mae in results:
    print(f"{d:8d} {'yes' if d in trained else 'no':>8} {rmse:8.2f} {1 - rmse / rmse_p:7.1%} {mae:8.2f}")