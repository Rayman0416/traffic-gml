# Stage 7: Build the feature arrays for model input

import numpy as np
import pandas as pd
from . import config as C
from . import params, splits, timeline
from . import observed as observed_mod

# Per-sensor channels, in the order they are documented and summarised.
SENSOR_CHANNELS = ("intensity", "occupancy")
TIME_FEATURES = timeline.CYCLICAL_TIME_FEATURES


def scale_channel(observed: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Scale a single channel of observed data to zero mean and unit variance, using the provided scalers."""
    z = (np.nan_to_num(observed, nan=0.0) - mean[None, :]) / std[None, :]
    return np.where(mask, z, np.float32(C.FEATURE_MISSING_VALUE)).astype(np.float32)


def build_arrays(obs: dict, p: dict, index: pd.DatetimeIndex) -> dict:
    """
    Build the feature arrays for model input, given the observed data and the scalers.
    """
    time_feats, _ = timeline.cyclical_time_features(index)
    return {
        "intensity": scale_channel(obs["observed_intensity"], obs["mask"],
                                   p["scaler_mean"], p["scaler_std"]),
        "intensity_mask": obs["mask"].astype(bool),
        "occupancy": scale_channel(obs["observed_occupancy"], obs["occupancy_mask"],
                                   p["occupancy_scaler_mean"],
                                   p["occupancy_scaler_std"]),
        "occupancy_mask": obs["occupancy_mask"].astype(bool),
        "time_features": time_feats,
    }


def history_index(origin_index: np.ndarray, history_length: int) -> np.ndarray:
    """
    Given a set of forecast origins, return the indices of the history rows for each origin.
    The returned array has shape ``[B, L]`` where ``B`` is the number of origins and ``L`` 
    is the history length.
    """
    origin_index = np.asarray(origin_index, dtype=np.int64)
    L = int(history_length)

    if (origin_index < L - 1).any():
        raise ValueError(f"origins before step {L - 1} have no full {L}-step history")
    
    return origin_index[:, None] + np.arange(-L + 1, 1, dtype=np.int64)[None, :]


def window_features(feats: dict, origin_index: np.ndarray, sensor_index: np.ndarray | None = None) -> dict:
    """
    Given a set of forecast origins and an optional set of sensors, 
    return the feature arrays for the corresponding windows.
    The returned dictionary contains the following keys:
    - "intensity": [B, L, N] array of scaled intensity values
    - "intensity_mask": [B, L, N] boolean array indicating valid intensity values
    - "occupancy": [B, L, N] array of scaled occupancy values
    - "occupancy_mask": [B, L, N] boolean array indicating valid occupancy values
    - "time_features": [B, F] array of time features for each origin
    - "time_feature_names": list of time feature names
    - "history_index": [B, L] array of history indices for each origin
    - "origin_index": [B] array of origin indices
    - "sensor_index": [N] array of sensor indices
    - "sensor_ids": [N] array of sensor IDs corresponding to the sensor indices
    """
    L = int(feats["history_length"])
    origin_index = np.asarray(origin_index, dtype=np.int64)
    rows = history_index(origin_index, L)
    if sensor_index is None:
        sensor_index = np.arange(len(feats["sensor_ids"]))
    cols = np.asarray(sensor_index, dtype=np.int64)
    take = (rows[:, :, None], cols[None, None, :])

    out = {name: feats[name][take] for name in ("intensity", "intensity_mask", "occupancy", "occupancy_mask")}
    out.update({
        "time_features": feats["time_features"][origin_index],
        "time_feature_names": list(feats["time_feature_names"]),
        "history_index": rows,
        "origin_index": origin_index,
        "sensor_index": cols,
        "sensor_ids": feats["sensor_ids"][cols],
    })

    return out


def _channel_stats(values: np.ndarray, mask: np.ndarray, sel: np.ndarray) -> dict:
    """Compute summary statistics for a single channel, given a selection of rows."""
    v = values[sel][mask[sel]].astype(np.float64)
    return {
        "valid_fraction": float(mask[sel].mean()),
        "scaled_mean": float(v.mean()) if v.size else None,
        "scaled_std": float(v.std()) if v.size else None,
    }


def build(out_dir=C.OUT_DIR, history_length=C.INPUT_LENGTH, verbose=True) -> dict:
    """
    Build the feature arrays for model input, and write them to ``features.npz``.
    """

    # load the observed data and the params for the scalers
    obs = observed_mod.load(out_dir)
    p = params.load(out_dir)

    # check that the sensor order is consistent and that the occupancy scaler is present
    if not np.array_equal(obs["sensor_ids"], p["sensor_ids"]):
        raise ValueError("observed.npz and params.npz disagree on sensor order")
    if "occupancy_scaler_mean" not in p:
        raise KeyError("params.npz has no occupancy scaler - re-run the `params` stage")
    
    index = obs["timestamps"]
    L = int(history_length)
    if L < 1:
        raise ValueError(f"history length must be at least 1, got {L}")

    # build the feature arrays and check for non-finite values
    arr = build_arrays(obs, p, index)
    for name in ("intensity", "occupancy", "time_features"):
        if not np.isfinite(arr[name]).all():
            raise ValueError(f"{name} feature contains non-finite values")

    np.savez_compressed(
        out_dir / "features.npz",
        sensor_ids=obs["sensor_ids"],
        timestamps=index.values.astype("datetime64[s]"),
        history_length=np.int64(L),
        sensor_channels=np.array(SENSOR_CHANNELS),
        time_feature_names=np.array(TIME_FEATURES),
        **arr,
    )

    # compute summary statistics per split
    split_code = obs["split_code"]
    per_split = {}
    for name, code in splits.SPLIT_CODE.items():
        sel = split_code == code
        per_split[name] = {
            ch: _channel_stats(arr[ch], arr[f"{ch}_mask"], sel)
            for ch in SENSOR_CHANNELS
        }
        per_split[name]["both_valid_fraction"] = float((arr["intensity_mask"][sel] & arr["occupancy_mask"][sel]).mean())
    
    summary = {
        "shape": {"T": int(len(index)), "N": int(len(obs["sensor_ids"])), "history_length": L},
        "per_split": per_split,
    }

    C.write_json(summary, C.reports_dir(out_dir) / "features_summary.json")

    if verbose:
        tr = per_split["train"]
        print(f"  history {L} steps; train valid: intensity "
              f"{tr['intensity']['valid_fraction']:.2%}, occupancy "
              f"{tr['occupancy']['valid_fraction']:.2%}")
        print(f"  wrote {out_dir / 'features.npz'}")

    return summary


def load(out_dir=C.OUT_DIR) -> dict:
    z = np.load(out_dir / "features.npz")
    out = {k: z[k] for k in z.files}
    out["timestamps"] = pd.DatetimeIndex(z["timestamps"])
    out["history_length"] = int(z["history_length"])
    out["sensor_channels"] = [str(s) for s in z["sensor_channels"]]
    out["time_feature_names"] = [str(s) for s in z["time_feature_names"]]
    return out
