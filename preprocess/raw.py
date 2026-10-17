# CSV -> Parquet conversion, sensor metadata, route graph. Since the csv is large,
# we use pyarrow to stream it into a typed parquet file.  The parquet file is
# partitioned by month so that later stages can scan a single month cheaply.

from pathlib import Path
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from . import config as C

# The raw CSV header is:
#   sensor_id, entry_date, traffic_intensity, sensor_occupancy, avg_speed,
#   read_error, original_traffic_intensity, is_interpolated
#
# original_traffic_intensity is empty everywhere so not sure what it is for.
#
# traffic_intensity is vehicles per hour
#
# is_interpolated is a positional flag.
# first digit: traffic_intensity, second: sensor_occupancy, third: avg_speed.
# 1 means the value was interpolated, 0 means it was not.

# schema for reading the raw CSV into a typed parquet file
READINGS_SCHEMA = pa.schema([
    ("sensor_id", pa.int32()),
    ("entry_date", pa.timestamp("s")),
    ("traffic_intensity", pa.float32()),
    ("sensor_occupancy", pa.float32()),
    ("avg_speed", pa.float32()),
    ("read_error", pa.string()),
    ("original_traffic_intensity", pa.float32()),
    ("is_interpolated", pa.string()),
])

# columns to read from the CSV, rest is ignored
READ_COLUMNS = [
    "sensor_id", "entry_date", "traffic_intensity",
    "read_error", "is_interpolated",
]


def convert_readings_to_parquet(
    csv_path: Path = C.RAW_CSV,
    out_path: Path = C.READINGS_PARQUET,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Stream the raw readings CSV into a typed Parquet file.
    Partitions by month so that later stages can scan a single month cheaply;
    """

    # skip the conversion if the parquet already exists and overwrite is False
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"  parquet already present: {out_path}")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")

    # open the CSV reader in batches with the reading schema and timestamp parser
    reader = pcsv.open_csv(
        csv_path,
        read_options=pcsv.ReadOptions(block_size=128 << 20),
        parse_options=pcsv.ParseOptions(newlines_in_values=False),
        convert_options=pcsv.ConvertOptions(
            column_types=READINGS_SCHEMA,
            timestamp_parsers=["%Y-%m-%d %H:%M:%S"],
            strings_can_be_null=True,
        ),
    )

    # convert the CSV to Parquet in batches, writing to a temporary file first
    n_rows = 0
    with pq.ParquetWriter(tmp_path, READINGS_SCHEMA, compression="zstd") as writer:
        for batch in reader:
            # write the batch to the parquet file and update the row count
            writer.write_batch(batch)
            n_rows += batch.num_rows

            # print progress 
            if verbose and n_rows % 20_000_000 < batch.num_rows:
                print(f"    {n_rows:,} rows...", flush=True)

    tmp_path.rename(out_path)
    if verbose:
        print(f"  wrote {n_rows:,} rows to {out_path}")

    return out_path


def readings_dataset(path: Path = C.READINGS_PARQUET) -> ds.Dataset:
    """return pyarrow dataset for the given parquet file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run the `convert` stage first "
            f"(python -m preprocess.run --stages convert)"
        )
    
    return ds.dataset(path, format="parquet")

def reading_sensor_ids(path: Path = C.READINGS_PARQUET,
                       cache: Path | None = None,
                       verbose: bool = True) -> np.ndarray:
    """
    Return the unique sensor ids present in the given parquet file.
    """
    cache = Path(cache) if cache is not None else C.OUT_DIR / "_reading_sensor_ids.npy"
    if cache.exists():
        return np.load(cache)

    if verbose:
        print("  scanning sensor ids in the readings...", flush=True)
    table = readings_dataset(path).to_table(columns=["sensor_id"])
    ids = pc.unique(table.column("sensor_id")).to_numpy(zero_copy_only=False)
    ids = np.sort(np.asarray(ids, dtype=np.int64))

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, ids)
    if verbose:
        print(f"    {len(ids):,} sensors with readings", flush=True)
    return ids

def intensity_interpolated_expr():
    """
    First digit of is_interpolated is 1 for rows where traffic_intensity was interpolated.
    """
    flag = pc.field("is_interpolated")
    # `is_valid` first so a null flag yields False rather than null: pyarrow
    # drops null filter rows silently, which would hide bad data.
    return (pc.is_valid(flag)
            & pc.equal(pc.binary_length(flag), 3)
            & pc.starts_with(flag, "1"))


def occupancy_interpolated_expr():
    """
    Second digit of is_interpolated is 1 for rows where sensor_occupancy was interpolated.
    """
    flag = pc.field("is_interpolated")
    return (pc.is_valid(flag)
            & pc.equal(pc.binary_length(flag), 3)
            & pc.equal(pc.utf8_slice_codeunits(flag, 1, 2), "1"))


def read_error_expr():
    """
    True for rows with a non-"N" read_error, which is the only valid value.
    """
    re = pc.field("read_error")
    return ~(pc.is_valid(re) & pc.equal(re, "N"))


def load_sensors(path: Path = C.SENSORS_CSV) -> pd.DataFrame:
    """
    Load the sensor metadata CSV into a pandas DataFrame
    """
    df = pd.read_csv(path)
    if df["id"].duplicated().any():
        raise ValueError("traffic_sensors.csv has duplicate ids")
    df["id"] = df["id"].astype("int64")
    df = df.set_index("id").sort_index()
    df["district"] = df["district"].astype("Int64")
    return df


def load_route_graph(path: Path = C.ROUTE_GRAPH_PKL) -> nx.MultiDiGraph:
    """
    Load the route graph from a pickle file into a networkx MultiDiGraph.
    """
    with open(path, "rb") as f:
        g = pickle.load(f)
    return nx.relabel_nodes(g, {n: int(n) for n in g.nodes}, copy=True)
