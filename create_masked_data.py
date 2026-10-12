"""Create per-sensor-normalized and ground-truth-filled traffic subgraphs."""

import argparse
import shutil
from pathlib import Path
import pickle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "subgraph"
DEFAULT_NORMAL = ROOT / "data" / "subgraph_normal"
DEFAULT_GROUNDTRUTH = ROOT / "data" / "subgraph_groundtruth_normal"
DEFAULT_RANDOMISED = ROOT / "data" / "randomised_normal"
DEFAULT_RANDOMISED_SINE = ROOT / "data" / "randomised_sine_normal"
EXPECTED_FILES = ("readings.npz", "edges.csv", "sensors.csv")


def check_output_directories(source_dir, output_dirs, overwrite):
    source = source_dir.resolve()
    resolved_outputs = [p.resolve() for p in output_dirs]

    if any(p == source for p in resolved_outputs):
        raise ValueError("An output directory cannot be the source directory.")
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Output directories must be different.")

    for out_dir in output_dirs:
        if out_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}\n"
                "Choose another output path or rerun with --overwrite."
            )


def load_subgraph(source_dir):
    """Load the matrix and its associated graph/metadata files."""
    npz_path = source_dir / "readings.npz"
    edges_path = source_dir / "edges.csv"
    sensors_path = source_dir / "sensors.csv"

    missing = [str(p) for p in (npz_path, edges_path, sensors_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "The source subgraph is missing required file(s):\n  "
            + "\n  ".join(missing)
        )

    with np.load(npz_path, allow_pickle=False) as archive:
        # Preserve any additional arrays that may be stored in readings.npz.
        arrays = {key: archive[key].copy() for key in archive.files}

    required_keys = {"sensor_ids", "timestamps", "intensity"}
    missing_keys = required_keys - arrays.keys()
    if missing_keys:
        raise KeyError(f"readings.npz is missing array(s): {sorted(missing_keys)}")

    sensor_ids = arrays["sensor_ids"]
    intensity = arrays["intensity"]

    if intensity.ndim != 2:
        raise ValueError(
            f"Expected intensity to have shape [time, sensors], got {intensity.shape}."
        )
    if intensity.shape[1] != len(sensor_ids):
        raise ValueError(
            f"intensity has {intensity.shape[1]} sensor columns, "
            f"but sensor_ids contains {len(sensor_ids)} IDs."
        )

    edges = pd.read_csv(edges_path)
    required_edge_columns = {"source", "target"}
    if not required_edge_columns.issubset(edges.columns):
        raise ValueError(
            f"edges.csv must contain columns {sorted(required_edge_columns)}."
        )

    return arrays, edges


def normalize_per_sensor(intensity):
    """Scale each sensor independently so its minimum is 0 and maximum is 1.

    The requested fixed minimum is 0, rather than the observed sensor minimum.
    NaNs are retained. Each sensor's maximum is calculated from its available
    values across the supplied 2024 matrix.
    """
    normalized = intensity.astype(np.float32, copy=True)
    maxima = np.full(normalized.shape[1], np.nan, dtype=np.float64)

    for col in range(normalized.shape[1]):
        values = normalized[:, col]
        valid = ~np.isnan(values)

        if not valid.any():
            continue

        sensor_max = float(np.max(values[valid]))
        maxima[col] = sensor_max

        if sensor_max < 0:
            raise ValueError(
                f"Sensor column {col} has a negative maximum ({sensor_max}); "
                "zero-minimum normalization is not valid for this column."
            )
        if sensor_max > 0:
            normalized[valid, col] = values[valid] / sensor_max
        else:
            # All valid values are zero when the maximum is zero and the data
            # are non-negative. Keep them at zero and avoid division by zero.
            normalized[valid, col] = 0.0

    return normalized, maxima


def build_neighbour_columns(sensor_ids, edges):
    """Build undirected neighbour lists from the directed route-edge table."""
    column_by_id = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    neighbours = [set() for _ in sensor_ids]

    for source, target in edges[["source", "target"]].itertuples(index=False, name=None):
        if source == target:
            continue
        source_col = column_by_id.get(source)
        target_col = column_by_id.get(target)
        if source_col is None or target_col is None:
            continue

        # Treat a road connection as adjacency in either direction, so a sensor
        # can use both incoming and outgoing route neighbours.
        neighbours[source_col].add(target_col)
        neighbours[target_col].add(source_col)

    return [sorted(cols) for cols in neighbours]



def fill_groundtruth(normalized, sensor_ids, edges, radius=4):
    """Iteratively fill NaNs using temporal lookup and spatial neighbour means.

    Each iteration uses a snapshot of the previous iteration, so newly filled
    values only propagate during subsequent iterations.

    If no more progress is possible, any remaining NaNs are filled using the
    global mean of available normalized readings. If the entire dataset is
    NaN, use 0.0 as the final fallback.

    Returns the filled matrix and fill statistics.
    """
    filled = normalized.astype(np.float32, copy=True)
    n_times, n_sensors = filled.shape
    neighbours = build_neighbour_columns(sensor_ids, edges)

    temporal_fills = 0
    spatial_fills = 0
    fallback_fills = 0
    iterations = 0

    while np.isnan(filled).any():
        iterations += 1

        # All decisions in this iteration use the same snapshot.
        previous = filled.copy()
        updated = previous.copy()
        progress = 0

        for sensor_col in range(n_sensors):
            missing_times = np.flatnonzero(
                np.isnan(previous[:, sensor_col])
            )

            for t in missing_times:
                start = max(0, t - radius)
                stop = min(n_times, t + radius + 1)

                # Rule 1: nearest valid reading from the same sensor.
                window = previous[start:stop, sensor_col]
                valid_offsets = np.flatnonzero(~np.isnan(window))

                if valid_offsets.size:
                    valid_times = start + valid_offsets
                    nearest_t = valid_times[
                        np.argmin(np.abs(valid_times - t))
                    ]
                    updated[t, sensor_col] = previous[
                        nearest_t, sensor_col
                    ]
                    temporal_fills += 1
                    progress += 1
                    continue

                # Rule 2: mean of valid graph neighbours at this timestamp.
                neighbour_cols = neighbours[sensor_col]

                if neighbour_cols:
                    values = previous[t, neighbour_cols]
                    values = values[~np.isnan(values)]
                else:
                    values = np.empty(0, dtype=np.float32)

                if values.size:
                    updated[t, sensor_col] = np.mean(values)
                    spatial_fills += 1
                    progress += 1

        filled = updated

        print(
            f"Iteration {iterations}: "
            f"{np.isnan(filled).sum():,} NaNs remaining"
        )

        # If nothing could be filled, another iteration cannot help without
        # introducing a fallback value.
        if progress == 0:
            break

    # Final fallback: guarantee that no NaNs remain.
    remaining = np.isnan(filled)
    fallback_fills = int(remaining.sum())

    if fallback_fills:
        valid_values = filled[~np.isnan(filled)]

        if valid_values.size:
            fallback_value = float(np.mean(valid_values))
        else:
            fallback_value = 0.0

        filled[remaining] = fallback_value

        print(
            f"Fallback: filled {fallback_fills:,} remaining NaNs "
            f"with {fallback_value:.6f}"
        )

    stats = {
        "temporal_fills": temporal_fills,
        "spatial_fills": spatial_fills,
        "fallback_fills": fallback_fills,
        "unresolved": int(np.isnan(filled).sum()),
        "remaining_nan": int(np.isnan(filled).sum()),
        "iterations": iterations,
    }

    return filled, stats

def create_randomised_normal(
    groundtruth_dir=DEFAULT_GROUNDTRUTH,
    output_dir=ROOT / "data" / "randomised_normal",
    missing_ratio=0.05,
    overwrite=True,
    seed=42,
):
    check_output_directories(groundtruth_dir, [output_dir], overwrite)

    # Load groundtruth normal data
    arrays, _ = load_subgraph(groundtruth_dir)
    groundtruth = arrays["intensity"]

    # Generate a random mask for 10% missing values
    np.random.seed(seed)
    randomised = groundtruth.astype(np.float32, copy=True)
    mask = np.random.rand(*randomised.shape) < missing_ratio
    randomised[mask] = np.nan

    # Save to readings.npz along with existing metadata files
    write_subgraph(groundtruth_dir, output_dir, arrays, randomised)
    print(f"Created randomised-normal dataset: {output_dir}")
    print(f"Masked {mask.mean():.2%} of values as NaN.")

    return randomised

def write_subgraph(source_dir, output_dir, arrays, intensity):
    """Write a subgraph copy with a replacement intensity matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output_arrays = {key: value.copy() for key, value in arrays.items()}
    output_arrays["intensity"] = intensity.astype(np.float32, copy=False)
    np.savez_compressed(output_dir / "readings.npz", **output_arrays)

    for filename in ("edges.csv", "sensors.csv"):
        shutil.copy2(source_dir / filename, output_dir / filename)

def create_datasets(source_dir=DEFAULT_SOURCE, normal_dir=DEFAULT_NORMAL, groundtruth_dir=DEFAULT_GROUNDTRUTH, randomised_dir=DEFAULT_RANDOMISED, randomised_sine_dir=DEFAULT_RANDOMISED_SINE, radius=4, missing_ratio=0.05, seed=42, overwrite=True,):

    check_output_directories(source_dir,  [normal_dir, groundtruth_dir, randomised_dir, randomised_sine_dir], overwrite,)

    arrays, edges = load_subgraph(source_dir)
    sensor_ids = arrays["sensor_ids"]
    original = arrays["intensity"]
    timestamps = arrays["timestamps"]

    normalized, maxima = normalize_per_sensor(original)
    groundtruth, stats = fill_groundtruth(
        normalized, sensor_ids, edges, radius=radius
    )

    np.random.seed(seed)

    # 1. Standard Uniform Random Masking
    randomised = groundtruth.astype(np.float32, copy=True)
    mask = np.random.rand(*randomised.shape) < missing_ratio
    randomised[mask] = np.nan

    # 2. Sine-Weighted Periodic Random Masking (24-hour period)
    # Convert timestamps to fractional hour of the day [0.0, 24.0)
    ts_pd = pd.to_datetime(timestamps)
    hours = (ts_pd.hour + ts_pd.minute / 60.0 + ts_pd.second / 3600.0).to_numpy()

    # |sin(pi * hour / 24)| has a period of 24 hours
    sine_weights = np.abs(np.sin(np.pi * hours / 24.0)**10)

    # Scale probabilities so the overall mean missing rate matches missing_ratio
    probabilities = sine_weights * (missing_ratio / sine_weights.mean())
    probabilities = np.clip(probabilities, 0.0, 1.0)

    # Broadcast probabilities across all sensor columns: shape [T, N]
    sine_prob_matrix = np.tile(probabilities[:, None], (1, groundtruth.shape[1]))

    randomised_sine = groundtruth.astype(np.float32, copy=True)
    mask_sine = np.random.rand(*randomised_sine.shape) < sine_prob_matrix
    randomised_sine[mask_sine] = np.nan

    # Save all four subgraphs
    write_subgraph(source_dir, normal_dir, arrays, normalized)
    write_subgraph(source_dir, groundtruth_dir, arrays, groundtruth)
    write_subgraph(source_dir, randomised_dir, arrays, randomised)
    write_subgraph(source_dir, randomised_sine_dir, arrays, randomised_sine)

    print(f"Created normalized dataset: {normal_dir}")
    print(f"Created ground-truth dataset: {groundtruth_dir}")
    print(f"Created randomised-normal dataset: {randomised_dir}")
    print(f"Created randomised-sine dataset: {randomised_sine_dir}")
    print(f"Temporal fills: {stats['temporal_fills']:,}")
    print(f"Spatial fills: {stats['spatial_fills']:,}")
    print(f"Unresolved NaNs: {stats['unresolved']:,}")
    print(f"Fallback fills: {stats['fallback_fills']:,}")
    print(f"Randomly masked NaNs: {mask.sum():,} ({mask.mean():.2%})")
    print(f"Sine-masked NaNs: {mask_sine.sum():,} ({mask_sine.mean():.2%})")

    return stats, groundtruth, randomised, randomised_sine


# Inside create_masked_data.py

def load_datasets():
    groundtruth = np.load(DEFAULT_GROUNDTRUTH / "readings.npz")["intensity"]
    randomised = np.load(DEFAULT_RANDOMISED / "readings.npz")["intensity"]
    randomised_sine = np.load(DEFAULT_RANDOMISED_SINE / "readings.npz")["intensity"]

    return groundtruth, randomised, randomised_sine

def main():
    parser = argparse.ArgumentParser(
        description="Create normalized and ground-truth-filled traffic subgraphs."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--normal-dir", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument("--groundtruth-dir", type=Path, default=DEFAULT_GROUNDTRUTH)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.radius < 0:
        parser.error("--radius must be non-negative")

    create_datasets(
        source_dir=args.source_dir,
        normal_dir=args.normal_dir,
        groundtruth_dir=args.groundtruth_dir,
        radius=args.radius,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()