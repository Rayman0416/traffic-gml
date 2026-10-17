# GO-MO Madrid traffic preprocessing

## Run

Put `traffic_data_2024.csv`, `traffic_sensors.csv` and `routes-graph.pkl` from
[GO-MO Madrid](https://huggingface.co/datasets/double-blind-anonymous/go-mo-dataset/tree/main)
in `data/`, then:

```bash
conda env create -f environment.yaml && conda activate gml
python -m preprocess.run
```

Use `python -m preprocess.run --selection region --n-nodes 500` for a small, fast build.

## Model input features

Written to `data/forecast_dataset/features.npz`. `T` = 15-minute timesteps in 2024,
`N` = sensors.

| Key | Shape | Meaning |
|-----|-------|---------|
| `intensity` | `[T, N]` | traffic intensity, standardised per sensor (training mean/std) |
| `intensity_mask` | `[T, N]` | `True` where the intensity is a genuine observation |
| `occupancy` | `[T, N]` | occupancy, standardised per sensor (training mean/std) |
| `occupancy_mask` | `[T, N]` | `True` where the occupancy is a genuine observation |
| `time_features` | `[T, 4]` | `tod_sin`, `tod_cos`, `dow_sin`, `dow_cos` |
| `sensor_ids` | `[N]` | sensor id of each column |
| `timestamps` | `[T]` | start of each 15-minute interval |
| `history_length` | `[]` | number of timesteps in the history |

Missing values are `0` and flagged by the masks. Get the inputs for a set of forecast
origins (the last `history_length` steps up to and including each origin):

```python
from preprocess import features, windows

f = features.load()
win = windows.load()
origins = win.loc[win["split"] == "train", "origin_index"].to_numpy()[:64]
x = features.window_features(f, origins)
# x["intensity"], x["intensity_mask"], x["occupancy"], x["occupancy_mask"]: [B, L, N]
# x["time_features"]: [B, 4]
```
