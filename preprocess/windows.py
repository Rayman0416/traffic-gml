# Stage 6: Build the window table for training and evaluation, with eligibility flags

import numpy as np
import pandas as pd
from . import config as C
from . import sensor_selection, splits
from . import observed as observed_mod

WINDOW_COLUMNS = [
    "window_id", "origin_index", "origin_timestamp", "split",
    "input_start_index", "input_end_index",
    "input_start_timestamp", "input_end_timestamp",
    "target_index_h1", "target_index_h2", "target_index_h3", "target_index_h4",
    "target_timestamp_h1", "target_timestamp_h2",
    "target_timestamp_h3", "target_timestamp_h4",
    "n_input_cells", "n_input_observed", "available_input",
    "available_input_worst_district", "worst_district",
    "n_valid_targets", "available_targets",
    "eligible_routine", "low_availability",
]

def _window_sums(counts: np.ndarray, L: int) -> np.ndarray:
    """
    Compute the sum of counts over a sliding window of length L for each sensor.
    counts: [T, N] array of counts per time step and sensor
    Returns: [T, N] array of sums over the last L time steps for each
    """
    csum = np.concatenate([np.zeros((1,) + counts.shape[1:], dtype=np.int64),
                           np.cumsum(counts, axis=0, dtype=np.int64)], axis=0)
    T = counts.shape[0]
    t = np.arange(T)
    lo = np.maximum(t - L + 1, 0)
    return csum[t + 1] - csum[lo]


def build(out_dir=C.OUT_DIR, input_length=C.INPUT_LENGTH,
          horizons=C.HORIZONS, verbose=True) -> dict:
    """
    Build the window table for training and evaluation, with eligibility flags.
    """

    # load the observed data and masks
    data = observed_mod.load(out_dir)
    mask = data["mask"] # [T, N] boolean array indicating valid observations
    index = data["timestamps"] # [T] array of timestamps
    T, N = mask.shape
    L = int(input_length)
    H = tuple(int(h) for h in horizons)
    max_h = max(H)

    inc = sensor_selection.included(out_dir)
    districts = inc["district"].to_numpy()
    uniq_d = np.array(sorted({int(d) for d in districts if pd.notna(d)}))

    # availability over the input window for all sensors and for each district
    per_step_all = mask.sum(axis=1).astype(np.int64)
    win_all = _window_sums(per_step_all[:, None], L)[:, 0]

    per_step_d = np.stack([mask[:, districts == d].sum(axis=1).astype(np.int64) for d in uniq_d], axis=1)
    win_d = _window_sums(per_step_d, L)
    size_d = np.array([int((districts == d).sum()) for d in uniq_d], dtype=np.int64)
    avail_d = win_d / (L * size_d[None, :])

    worst_pos = np.argmin(avail_d, axis=1)
    avail_worst = avail_d[np.arange(T), worst_pos]
    worst_district = uniq_d[worst_pos]

    avail_all = win_all / float(L * N)

    # candidate origins
    origins = np.arange(L - 1, T - max_h, dtype=np.int64)

    # Split assignment: all four targets must share a split.
    # so they should all belong to train split if the origin is in train split, and so on.
    split_code = data["split_code"]
    tgt_idx = np.stack([origins + h for h in H], axis=1)
    tgt_split = split_code[tgt_idx]
    same = (tgt_split == tgt_split[:, :1]).all(axis=1)
    n_straddle = int((~same).sum())
    origins = origins[same]
    tgt_idx = tgt_idx[same]
    win_split = tgt_split[same, 0]

    # target availability
    tgt_obs = mask[tgt_idx] # [n_windows, n_horizons, n_sensors] boolean array indicating valid targets
    n_valid_targets = tgt_obs.sum(axis=(1, 2)).astype(np.int64)
    avail_targets = n_valid_targets / float(len(H) * N)

    a_in = avail_all[origins]
    a_in_worst = avail_worst[origins]
    eligible = ((a_in >= C.MIN_INPUT_AVAILABILITY) & (a_in_worst >= C.MIN_DISTRICT_INPUT_AVAILABILITY))

    df = pd.DataFrame({
        "window_id": np.arange(len(origins), dtype=np.int64),
        "origin_index": origins,
        "origin_timestamp": index[origins],
        "split": [splits.CODE_SPLIT[int(c)] for c in win_split],
        "input_start_index": origins - L + 1,
        "input_end_index": origins,
        "input_start_timestamp": index[origins - L + 1],
        "input_end_timestamp": index[origins],
        "n_input_cells": np.int64(L * N),
        "n_input_observed": win_all[origins],
        "available_input": a_in.astype(np.float32),
        "available_input_worst_district": a_in_worst.astype(np.float32),
        "worst_district": worst_district[origins],
        "n_valid_targets": n_valid_targets,
        "available_targets": avail_targets.astype(np.float32),
        "eligible_routine": eligible,
        "low_availability": ~eligible,
    })

    for k, h in enumerate(H):
        df[f"target_index_h{k + 1}"] = tgt_idx[:, k]
        df[f"target_timestamp_h{k + 1}"] = index[tgt_idx[:, k]]

    df = df[WINDOW_COLUMNS]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "windows.csv", index=False)
    np.savez_compressed(
        out_dir / "windows.npz",
        window_id=df["window_id"].to_numpy(),
        origin_index=df["origin_index"].to_numpy(),
        target_index=tgt_idx,
        split_code=win_split.astype(np.int8),
        available_input=df["available_input"].to_numpy(),
        available_input_worst_district=df["available_input_worst_district"].to_numpy(),
        eligible_routine=df["eligible_routine"].to_numpy(),
        n_valid_targets=n_valid_targets,
    )

    # compute summary statistics per split
    per_split = {}
    for name in splits.SPLIT_NAMES:
        sel = df["split"] == name
        sub = df[sel]
        elig = sub[sub["eligible_routine"]]
        per_split[name] = {
            "n_windows": int(len(sub)),
            "n_eligible_routine": int(len(elig)),
            "n_low_availability": int((~sub["eligible_routine"]).sum()),
            "available_input": {
                "min": float(sub["available_input"].min()) if len(sub) else None,
                "p01": float(sub["available_input"].quantile(0.01)) if len(sub) else None,
                "median": float(sub["available_input"].median()) if len(sub) else None,
            },
            "available_input_worst_district_p01": float(
                sub["available_input_worst_district"].quantile(0.01)) if len(sub) else None,
            "n_valid_targets_total": int(sub["n_valid_targets"].sum()),
            "n_valid_targets_eligible_only": int(elig["n_valid_targets"].sum()),
            "windows_with_zero_valid_targets": int((sub["n_valid_targets"] == 0).sum()),
        }

    # write summary statistics to JSON
    summary = {
        "n_candidate_origins": int(T - max_h - (L - 1)),
        "n_windows": int(len(df)),
        "n_dropped_straddling_split_boundary": n_straddle,
        "per_split": per_split,
    }

    C.write_json(summary, C.reports_dir(out_dir) / "windows_summary.json")

    if verbose:
        for name in splits.SPLIT_NAMES:
            s = per_split[name]
            print(f"  {name:5s}: {s['n_windows']:6,} windows, "
                  f"{s['n_eligible_routine']:6,} eligible, "
                  f"{s['n_valid_targets_total']:,} genuine targets")
        print(f"  wrote {out_dir / 'windows.csv'}")

    return summary


def load(out_dir=C.OUT_DIR) -> pd.DataFrame:
    """The shared window table (timestamps parsed)."""
    df = pd.read_csv(out_dir / "windows.csv", parse_dates=[
        c for c in WINDOW_COLUMNS if c.endswith("timestamp")
        or "timestamp_h" in c])
    return df


def load_arrays(out_dir=C.OUT_DIR) -> dict:
    """The shared window table as arrays, for fast access."""
    z = np.load(out_dir / "windows.npz")
    return {k: z[k] for k in z.files}
