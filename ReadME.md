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

## Fixes for the current data

The GO-MO dataset was updated and dropped three columns the code still expects:
`original_traffic_intensity`, `max_capacity` and `has_data`. On a fresh download
the pipeline crashes partway through. The edits below get it running end to end.
They belong on `main` so nobody has to redo them.

**1. `preprocess/raw.py`.** In `READINGS_SCHEMA`, delete the line
`("original_traffic_intensity", pa.float32()),`. The readings CSV has 7 columns now.

**2. `preprocess/audit.py`.** In the column list around line 48, remove
`"original_traffic_intensity"`.

**3. `preprocess/sensor_selection.py`, metadata copy.** The loop near line 158 that
copies sensor metadata crashes on the dropped columns. Change `df[col] = sensors[col]`
to:

```python
df[col] = sensors[col] if col in sensors.columns else np.nan
```

It copies the columns that still exist and leaves the dropped ones blank, which the
rest of the stage handles.

**4. `preprocess/sensor_selection.py`, region selection.** `--selection region` is a
stub. Replace `raise ValueError("implement region selection later")` with:

```python
chosen = _grow_region(graph, sensors, eligible, n_nodes)
```

`_grow_region` is already defined in the file. You only need this for
`--selection region`; `--selection all` does not use it.

The audit stage still trips on the missing `has_data` column. Audit only writes a
diagnostics report, so skipping it is the simplest route to a dataset:

```bash
python -m preprocess.run --selection all --stages select graphs observed params windows features
```

If the download itself fails with a Xet "CAS client" error on the large CSV, disable
the Xet backend and run it again:

```bash
set HF_HUB_DISABLE_XET=1
```

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
