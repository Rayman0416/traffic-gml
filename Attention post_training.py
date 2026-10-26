import time
import torch
import attention_traffic_gtcnn as m
from pathlib import Path

CKPT = "checkpoints/gtcnn_20261025_1526.pt"
DEVICE = "mps"                         # Set to cuda if you have Nvidia card
POST_DISTRICTS = list(range(1, 22))    # gradients are collected over all of these, then one update
POST_EPOCHS = 20                       # = 10 weight updates in total
POST_LR = 1e-4                         # see the note on step size below

m.DATA_DIR = Path("data/forecast_dataset")
m.shared_data.cache_clear()           # otherwise the old folder's arrays are still served
m.PLOT_FIGURES = True                # evaluate() would otherwise redraw 10 figures per district


def accumulate_district(model, data, scale):
    """Backward over every training batch of one district; gradients add up in .grad.
    The district's loss is its exact masked mean, and `scale` (1 / n_districts) gives each
    district the same weight in the update. Returns the district's mean train MSE."""
    X, y, mask = data["train"][:3]
    total = mask.sum().clamp(min=1)
    district_loss = torch.zeros((), device=X.device)
    model.train()
    for b in range(0, len(X), m.BATCH_SIZE):
        xb, yb, mb = X[b:b + m.BATCH_SIZE], y[b:b + m.BATCH_SIZE], mask[b:b + m.BATCH_SIZE]
        loss = m.masked_mse(model(xb), yb, mb) * mb.sum() / total     # this batch's share of the district mean
        (loss * scale).backward()
        district_loss += loss.detach()
    return district_loss.item()


ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
pretrained = ckpt["trained_districts"]

model, cols, S, data = m.load_trained(CKPT, POST_DISTRICTS[0], DEVICE)
dev = next(model.parameters()).device
optimizer = torch.optim.Adam(model.parameters(), lr=POST_LR)       # fresh state: the checkpoint has none

state = dict(d=POST_DISTRICTS[0], cols=cols, data=data)
def use(d):
    """Make district d the one in memory (only one at a time) and point the model's graph at it."""
    if state["d"] != d:
        state["data"] = None
        if dev.type == "mps": torch.mps.empty_cache()
        cols, S, state["data"] = m.load_district(d, dev)
        model.set_graph(S, len(cols))
        state.update(d=d, cols=cols)
    return state["cols"], state["data"]

before = {}
t0 = time.perf_counter()
for epoch in range(1, POST_EPOCHS + 1):
    optimizer.zero_grad()
    losses = []
    for d in POST_DISTRICTS:
        cols, data = use(d)
        if epoch == 1:                                    # baseline for free: data is already loaded
            before[d] = m.evaluate(model, data, cols, len(cols), d, dev)
        losses.append(accumulate_district(model, data, 1 / len(POST_DISTRICTS)))
    optimizer.step()                                      # one update, after every district contributed
    print(f"post epoch {epoch}: mean train mse over districts {sum(losses) / len(losses):.4f}"
          f"  ({time.perf_counter() - t0:.0f}s)", flush=True)

# save a new checkpoint (keep the original training metadata accurate, add the post-training info)
m.EPOCHS, m.EPOCHS_PER_DISTRICT = ckpt["epochs"], ckpt["epochs_per_district"]
tag = time.strftime("%Y%m%d_%H%M")
out_path = f"checkpoints/gtcnn_post_{tag}.pt"
m.save_checkpoint(model, out_path, sorted(set(pretrained) | set(POST_DISTRICTS)))
new = torch.load(out_path, map_location="cpu", weights_only=True)
assert all(torch.equal(new["state_dict"][k], v.detach().cpu()) for k, v in model.named_parameters())
new["post_training"] = dict(base_checkpoint=CKPT, epochs=POST_EPOCHS, districts=POST_DISTRICTS, lr=POST_LR)
torch.save(new, out_path)
print(f"saved {out_path}")

# evaluate after post-training, same table as before plus the pre-post-training RMSE
results = []
for d in POST_DISTRICTS:
    cols, data = use(d)
    results.append((d, *m.evaluate(model, data, cols, len(cols), d, dev)))

print(f"\n{'district':>8} {'pretrain':>8} {'rmse_old':>9} {'rmse':>8} {'gain':>7} {'mae':>8}")
for d, rmse, rmse_p, mae in results:
    print(f"{d:8d} {'yes' if d in pretrained else 'no':>8} {before[d][0]:9.2f} {rmse:8.2f} "
          f"{1 - rmse / before[d][0]:7.1%} {mae:8.2f}")
for label, flag in (("pretrained", True), ("other", False)):
    g_old = [1 - before[d][0] / before[d][1] for d, *_ in results if (d in pretrained) == flag]
    g_new = [1 - r / rp for d, r, rp, _ in results if (d in pretrained) == flag]
