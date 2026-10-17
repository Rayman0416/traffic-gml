# Stage 5: Fit and save per-sensor parameters for filling and standardisation

import numpy as np
import pandas as pd
from . import config as C
from . import splits, timeline


# TO:DO can be used for filling missing values, but not used in the current pipeline
def fit_profile(observed: np.ndarray, mask: np.ndarray,
                index: pd.DatetimeIndex) -> dict:
    """
    Fit a time-of-week profile for each sensor from genuine training observations.
    """

    # get only training data
    train = splits.train_mask(index)
    tow = timeline.time_of_week(index)[train]
    vals = observed[train]
    obs = mask[train] & np.isfinite(vals)

    # init totals and counts
    T, N = vals.shape
    sums = np.zeros((C.STEPS_PER_WEEK, N), dtype=np.float64)
    counts = np.zeros((C.STEPS_PER_WEEK, N), dtype=np.int64)

    # compute sums and counts for each sensor at each time-of-week
    contrib = np.where(obs, vals, 0.0).astype(np.float64)
    np.add.at(sums, tow, contrib)
    np.add.at(counts, tow, obs.astype(np.int64))

    enough = counts >= C.PROFILE_MIN_SUPPORT
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(enough, sums / np.maximum(counts, 1), np.nan)

    # compute per-sensor and global training means for fallback
    sensor_total = np.where(obs, vals, 0.0).sum(axis=0)
    sensor_count = obs.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sensor_mean = np.where(sensor_count > 0, sensor_total / np.maximum(sensor_count, 1), np.nan)
    global_mean = float(sensor_total.sum() / max(sensor_count.sum(), 1))

    return {
        "profile": profile.astype(np.float32), # [672, N]
        "profile_support": counts.astype(np.int32), # [672, N]
        "sensor_train_mean": np.nan_to_num(sensor_mean, nan=global_mean).astype(np.float32),
        "global_train_mean": np.float32(global_mean),
        "info": {
            "n_profile_cells": int(profile.size),
            "n_profile_cells_unsupported": int((~enough).sum()),
            "fraction_unsupported": float((~enough).mean()),
            "median_support": float(np.median(counts)),
            "sensors_with_no_training_data": int((sensor_count == 0).sum()),
            "global_train_mean": global_mean,
        },
    }


def profile_matrix(profile: np.ndarray, sensor_train_mean: np.ndarray,
                   global_train_mean: float,
                   index: pd.DatetimeIndex) -> np.ndarray:
    """
    Return the per-timestep profile, with fallbacks for unsupported cells.
    """
    tow = timeline.time_of_week(index)
    out = profile[tow].astype(np.float32)             # [T, N]
    sm = np.broadcast_to(sensor_train_mean[None, :], out.shape)
    out = np.where(np.isfinite(out), out, sm)
    return np.where(np.isfinite(out), out, np.float32(global_train_mean)).astype(np.float32)


def fit_scaler(observed: np.ndarray, mask: np.ndarray,
               index: pd.DatetimeIndex,
               min_std: float = C.MIN_STD_VEH_PER_HOUR) -> dict:
    """
    Fit a per-sensor scaler from genuine training observations.
    """
    train = splits.train_mask(index)
    vals = np.where(mask[train], observed[train], np.nan).astype(np.float64)

    count = np.isfinite(vals).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(vals, axis=0)
        std = np.nanstd(vals, axis=0)

    global_mean = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
    global_std = float(np.nanstd(vals)) if np.isfinite(vals).any() else 1.0

    mean = np.where(count > 0, mean, global_mean)
    std = np.where(count > 0, std, global_std)
    floored = std < min_std
    std = np.where(floored, min_std, std)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "info": {
            "n_sensors_std_floored": int(floored.sum()),
            "n_sensors_without_training_data": int((count == 0).sum()),
            "mean_range": [float(mean.min()), float(mean.max())],
            "std_range": [float(std.min()), float(std.max())],
        },
    }


def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise ``[..., N]`` using per-sensor statistics."""
    return ((x - mean) / std).astype(np.float32)


def inverse_transform(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Back to vehicles/hour.  Always call this before reporting MAE or RMSE."""
    return (z * std + mean).astype(np.float32)


def build(observed: np.ndarray, mask: np.ndarray, index: pd.DatetimeIndex,
          sensor_ids: np.ndarray, observed_occupancy: np.ndarray,
          occupancy_mask: np.ndarray, out_dir=C.OUT_DIR, verbose=True) -> dict:
    """Fit and save the per-sensor profile and standardisation parameters."""
    prof = fit_profile(observed, mask, index)
    scal = fit_scaler(observed, mask, index)
    occ = fit_scaler(observed_occupancy, occupancy_mask, index, min_std=C.MIN_STD_OCCUPANCY_PERCENT)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "params.npz",
        sensor_ids=sensor_ids,
        profile=prof["profile"],
        profile_support=prof["profile_support"],
        sensor_train_mean=prof["sensor_train_mean"],
        global_train_mean=prof["global_train_mean"],
        scaler_mean=scal["mean"],
        scaler_std=scal["std"],
        occupancy_scaler_mean=occ["mean"],
        occupancy_scaler_std=occ["std"],
    )

    summary = {
        "profile": prof["info"],
        "scaler": scal["info"],
        "occupancy_scaler": occ["info"],
        "split_used": splits.describe()["train"],
    }

    C.write_json(summary, C.reports_dir(out_dir) / "params_summary.json")

    if verbose:
        print(f"  profile: {prof['info']['fraction_unsupported']:.2%} of "
              f"{prof['info']['n_profile_cells']:,} cells unsupported; "
              f"scaler: {scal['info']['n_sensors_std_floored']} std floored, "
              f"occupancy scaler: {occ['info']['n_sensors_std_floored']} floored")
        
    return summary


def load(out_dir=C.OUT_DIR) -> dict:
    """Load the per-sensor profile and standardisation parameters."""
    z = np.load(out_dir / "params.npz")
    return {k: z[k] for k in z.files}
