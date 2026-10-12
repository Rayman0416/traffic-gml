"""Loading and preprocessing for the GO-MO Madrid traffic dataset.

Features:
- convert the raw readings CSV to Parquet for faster reads and memory efficiency,
- load sensor metadata and the route graph,
- choose a connected subgraph region of sensors with good training coverage,
- build a regular 15-minute grid of readings for that region.
"""

from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "traffic_data_2024.csv"
SENSORS_CSV = DATA_DIR / "traffic_sensors.csv"
ROUTE_GRAPH = DATA_DIR / "routes-graph.pkl"
READINGS_PARQUET = DATA_DIR / "traffic_2024.parquet"
SUBGRAPH_DIR = DATA_DIR / "subgraph"

FREQ = "15min" # timestamp frequency
TRAIN_END = pd.Timestamp("2024-09-01")  # train: Jan-Aug, val: Sep-Oct, test: Nov-Dec
VAL_END = pd.Timestamp("2024-11-01")
MADRID_CENTRE = (40.4168, -3.7038)  # Puerta del Sol, city centre madrid (lat, lon) roughly

READINGS_SCHEMA = pa.schema([
    ("sensor_id", pa.int32()),
    ("entry_date", pa.timestamp("s")),
    ("traffic_intensity", pa.float32()),
    ("sensor_occupancy", pa.float32()),
    ("avg_speed", pa.float32()),
    ("read_error", pa.string()),
    # Three-character flag, one digit per variable (intensity, occupancy, speed):
    # "0" = all original, "010" = occupancy interpolated, "111" = all interpolated.
    ("is_interpolated", pa.string()), # use string so it does not get converted to a number and lose leading zeros
])

# Load/Convert raw files --------------------------------------------------------------

def convert_readings_to_parquet(csv_path=RAW_CSV, out_path=READINGS_PARQUET, overwrite=False):
    """Stream the raw CSV into a typed Parquet file"""

    # Create the output path, skip if file already exists and the user did not request overwrite
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")

    # Stream the CSV in batches to avoid loading the whole file into memory
    reader = pcsv.open_csv(
        csv_path,
        read_options=pcsv.ReadOptions(block_size=64 << 20), # batch size 64 MiB
        convert_options=pcsv.ConvertOptions(
            column_types=READINGS_SCHEMA,
            timestamp_parsers=["%Y-%m-%d %H:%M:%S"],
        ),
    )

    # Write the batches to Parquet format
    n_rows = 0
    with pq.ParquetWriter(tmp_path, READINGS_SCHEMA, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
            n_rows += batch.num_rows
    tmp_path.rename(out_path)
    print(f"wrote {n_rows:,} rows to {out_path}")

    return out_path


def load_sensors(path=SENSORS_CSV):
    """Load sensor metadata indexed by sensor id"""
    return pd.read_csv(path).set_index("id")


def load_route_graph(path=ROUTE_GRAPH):
    """Load the directed route graph: nodes are sensor ids, edges carry `distance` (m) and `weight`."""
    with open(path, "rb") as f:
        return pickle.load(f)


def readings_dataset(path=READINGS_PARQUET):
    """Load pyarrow dataset of the traffic readings Parquet file"""
    return ds.dataset(path, format="parquet")


# masks ------------------------------------------------------------

def valid_reading_expr():
    """A traffic_intensity reading is trustworthy if it exists, the source flagged it
    as N (no error), and its own interpolation digit (the first one) is 0.

    Rows with a blank read_error are excluded.
    """
    return (
        pc.field("traffic_intensity").is_valid()
        & (pc.field("read_error") == "N")
        & ~pc.starts_with(pc.field("is_interpolated"), "1")
    )


def sensor_coverage(start=None, end=TRAIN_END, path=READINGS_PARQUET):
    """Fraction of 15-minute slots in [start, end) with a valid reading, per sensor."""

    # Get the number of 15-minute slots in the time range
    start = pd.Timestamp(start or "2024-01-01")
    n_slots = len(pd.date_range(start, end, freq=FREQ, inclusive="left"))

    # Time + valid reading filter
    filter = (
        (pc.field("entry_date") >= pa.scalar(start.to_pydatetime(), pa.timestamp("s")))
        & (pc.field("entry_date") < pa.scalar(end.to_pydatetime(), pa.timestamp("s")))
        & valid_reading_expr()
    )

    # Apply filter and count distinct timestamps per sensor, then divide by the number of slots
    table = readings_dataset(path).to_table(columns=["sensor_id", "entry_date"], filter=filter)
    counts = table.group_by("sensor_id").aggregate([("entry_date", "count_distinct")])
    s = counts.to_pandas().set_index("sensor_id")["entry_date_count_distinct"]

    return (s / n_slots).rename("coverage").sort_values(ascending=False)


# subgraph region ------------------------------------------------------------

def undirected_simple(graph, nodes=None):
    """
    Collapse the directed multigraph to a simple undirected graph. 
    Parallel and reciprocal edges keep the largest weight.
    """
    sub = graph.subgraph(nodes) if nodes is not None else graph
    g = nx.Graph()
    g.add_nodes_from(sub.nodes(data=True))
    for u, v, d in sub.edges(data=True):
        if u == v:
            continue
        if g.has_edge(u, v):
            if d["weight"] > g[u][v]["weight"]:
                g[u][v].update(weight=d["weight"], distance=d["distance"])
        else:
            g.add_edge(u, v, weight=d["weight"], distance=d["distance"])
    return g


def nearest_sensor(sensors, lat, lon, candidates):
    """Return the sensor id in candidates nearest to (lat, lon) in degrees."""
    cand = sensors.loc[sorted(candidates)]
    d2 = (cand["latitude"] - lat) ** 2 + ((cand["longitude"] - lon) * np.cos(np.radians(lat))) ** 2
    return int(d2.idxmin())


def select_subgraph(graph, sensors, coverage, n_nodes=500, min_coverage=0.8, centre=MADRID_CENTRE, max_nan_run=5000, path=READINGS_PARQUET):
    """
    Grow a connected region of n_nodes well-covered sensors outward from centre.
    Candidates are sensors with at least min_coverage valid training readings.
    Breadth-first search on the undirected graph restricted to candidates keeps
    the region connected.
    """
    candidates = set(coverage.index[coverage >= min_coverage]) & set(graph.nodes)

    # Exclude sensors with a too long consecutive stretch of NaN
    if max_nan_run is not None:
        runs = sensor_max_nan_run(sorted(candidates), path)
        n_before = len(candidates)
        candidates = set(runs.index[runs <= max_nan_run])
        print(f"{n_before - len(candidates):,} of {n_before:,} candidates excluded: "
              f"longest NaN run > {max_nan_run:,} steps")

    g = undirected_simple(graph, candidates)
    seed = nearest_sensor(sensors, *centre, max(nx.connected_components(g), key=len))

    # Visit nodes nearest-hop first, breaking ties based on distance to the seed.
    selected = []
    for layer in nx.bfs_layers(g, seed):
        layer = sorted(layer, key=lambda n: _dist(sensors, seed, n))
        selected.extend(layer[: n_nodes - len(selected)])
        if len(selected) >= n_nodes:
            break

    if len(selected) < n_nodes:
        print(f"warning: only {len(selected)} sensors available (wanted {n_nodes})")
    return sorted(selected)

    return sorted(selected)


def _dist(sensors, a, b):
    """get the Euclidean distance in meters between two sensors a and b"""
    return np.hypot(sensors.at[a, "utm_x"] - sensors.at[b, "utm_x"],
                    sensors.at[a, "utm_y"] - sensors.at[b, "utm_y"])


# regular grid -------------------------------------------------------- 

def build_grid(sensor_ids, path=READINGS_PARQUET, zeros_to_nan=False):
    """Readings for `sensor_ids` on a regular 15-minute grid covering all of 2024.

    Returns (timestamps, intensity, valid) where intensity is float32 [T, N]
    with NaN wherever no valid reading exists, and valid is the matching bool mask.
    Rows flagged by `read_error` or `is_interpolated` are treated as missing.
    """
    sensor_ids = np.asarray(sensor_ids)
    timestamps = pd.date_range("2024-01-01", "2025-01-01", freq=FREQ, inclusive="left")
    flt = pc.field("sensor_id").isin(pa.array(sensor_ids, pa.int32())) & valid_reading_expr()
    df = readings_dataset(path).to_table(
        columns=["sensor_id", "entry_date", "traffic_intensity"], filter=flt
    ).to_pandas()
    df = df.drop_duplicates(["sensor_id", "entry_date"], keep="last")

    col = pd.Index(sensor_ids).get_indexer(df["sensor_id"])
    row = timestamps.get_indexer(df["entry_date"])
    keep = row >= 0  # drop readings that are off the 15-minute grid
    intensity = np.full((len(timestamps), len(sensor_ids)), np.nan, dtype=np.float32)
    intensity[row[keep], col[keep]] = df["traffic_intensity"].to_numpy()[keep]

    if zeros_to_nan:
        intensity[intensity == 0] = np.nan

    return timestamps, intensity, ~np.isnan(intensity)


def remove_spikes(timestamps, intensity, factor=3.0):
    """
    Set readings above factor x each sensor's training 99th percentile to NaN.
    Catches isolated glitches (e.g. 82,300 veh/h on a street whose median is ~950).
    A per-lane cap is not used because the lanes metadata is unreliable for some
    sensors. Only training months set the threshold, so no test information leaks.
    """
    train = timestamps < TRAIN_END
    limit = factor * np.nanpercentile(intensity[train], 99, axis=0)
    spikes = intensity > limit
    cleaned = np.where(spikes, np.nan, intensity).astype(np.float32)
    return cleaned, int(spikes.sum())


def subgraph_edges(graph, sensor_ids):
    """Directed edge list of the route graph restricted to the given sensor_ids"""
    sub = graph.subgraph(sensor_ids)

    edges = []
    for u, v, d in sub.edges(data=True):
        if u != v:
            edges.append((u, v, d["distance"], d["weight"]))

    return pd.DataFrame(edges, columns=["source", "target", "distance", "weight"])


def save_subgraph(sensor_ids, timestamps, intensity, edges, sensors, out_dir=SUBGRAPH_DIR):
    """Save the subgraph region"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "readings.npz",
        sensor_ids=np.asarray(sensor_ids),
        timestamps=timestamps.values.astype("datetime64[s]"),
        intensity=intensity,
    )
    edges.to_csv(out_dir / "edges.csv", index=False)
    sensors.loc[sensor_ids].to_csv(out_dir / "sensors.csv")


def load_subgraph(out_dir=SUBGRAPH_DIR):
    """Load the saved subgraph: (sensor_ids, timestamps, intensity [T, N], edges, sensors)."""
    out_dir = Path(out_dir)
    z = np.load(out_dir / "readings.npz")
    return (
        z["sensor_ids"],
        pd.DatetimeIndex(z["timestamps"]),
        z["intensity"],
        pd.read_csv(out_dir / "edges.csv"),
        pd.read_csv(out_dir / "sensors.csv", index_col="id"),
    )

def remove_outage_steps(timestamps, intensity, max_bad_fraction=0.5, drop=True):
    bad = np.isnan(intensity)
    outage = bad.mean(axis=1) > max_bad_fraction

    if drop:
        return timestamps[~outage], intensity[~outage], int(outage.sum())

    cleaned = intensity.copy()
    cleaned[outage] = np.nan
    return timestamps, cleaned, int(outage.sum())

def longest_nan_run(intensity):
    """Longest run of consecutive NaN time steps."""
    nan = np.isnan(intensity)
    run = np.zeros(nan.shape[1], dtype=np.int64)
    best = np.zeros(nan.shape[1], dtype=np.int64)
    for row in nan:  # one pass over time; run resets to 0 whenever a value is present
        run = (run + 1) * row
        best = np.maximum(best, run)
    return best #well "worst"


def sensor_max_nan_run(sensor_ids, path=READINGS_PARQUET, chunk_size=1000):
    """Longest consecutive NaN run (in 15-min steps) per sensor over the whole 2024 grid.

    Uses the same NaN definition as the final data: invalid readings plus removed spikes.
    Sensors are processed in chunks to limit memory use.
    """
    sensor_ids = np.asarray(sensor_ids)
    if len(sensor_ids) == 0:
        return pd.Series([], dtype="int64", name="max_nan_run")
    runs = []
    for i in range(0, len(sensor_ids), chunk_size):
        timestamps, intensity, _ = build_grid(sensor_ids[i:i + chunk_size], path)
        intensity, _ = remove_spikes(timestamps, intensity)
        runs.append(longest_nan_run(intensity))
    return pd.Series(np.concatenate(runs), index=sensor_ids, name="max_nan_run")