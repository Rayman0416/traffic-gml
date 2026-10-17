# Stage 2: select the sensors based on training-period coverage and graph connectivity, and save the selection table.

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import networkx as nx
from . import config as C
from . import raw, splits, timeline

SENSOR_SELECTION_COLUMNS = [
    "sensor_id", "array_index", "latitude", "longitude", "utm_x", "utm_y",
    "district", "element_type", "lanes", "max_speed", "max_capacity",
    "has_data", "in_graph", "in_readings", "graph_degree",
    "train_coverage", "train_max_gap_steps", "train_n_observed",
    "included", "exclusion_reason",
]


def _ts(t):
    return pa.scalar(pd.Timestamp(t).to_pydatetime(), pa.timestamp("s"))


def genuine_filter():
    """
    Returns rows that are not flagged as read errors or outside the hard cap.
    Interpolated intensities are accepted, matching ``Flag.REJECTING``.
    """
    return (
        pc.field("traffic_intensity").is_valid()
        & ~raw.read_error_expr()
        & (pc.field("traffic_intensity") >= 0)
        & (pc.field("traffic_intensity") <= C.HARD_CAP_VEH_PER_HOUR)
    )


def train_presence(sensor_ids: np.ndarray, path=C.READINGS_PARQUET,
                   verbose=True) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Return a boolean array of shape (n_train_slots, n_sensors) indicating which sensors have genuine 
    readings in each training-period time slot, and the corresponding index of time slots.
    """
    index = timeline.build_index()
    tmask = splits.train_mask(index)
    train_index = index[tmask]

    sensor_ids = np.asarray(sensor_ids, dtype=np.int64)
    col_of = pd.Index(sensor_ids)
    present = np.zeros((len(train_index), len(sensor_ids)), dtype=bool)

    dset = raw.readings_dataset(path)
    edges = pd.date_range(C.TRAIN_START, C.TRAIN_END, freq="MS")
    edges = pd.DatetimeIndex([C.TRAIN_START]).union(edges).union(
        pd.DatetimeIndex([C.TRAIN_END]))

    # Scan the readings dataset, filtering to the training period and genuine readings
    # and mark the presence of each sensor in the corresponding time slot.
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo >= hi:
            continue
        flt = (genuine_filter()
               & (pc.field("entry_date") >= _ts(lo))
               & (pc.field("entry_date") < _ts(hi)))
        tb = dset.to_table(columns=["sensor_id", "entry_date"], filter=flt)
        if tb.num_rows == 0:
            continue
        sid = tb.column("sensor_id").to_numpy(zero_copy_only=False).astype(np.int64)
        ed = pd.DatetimeIndex(tb.column("entry_date").to_numpy(zero_copy_only=False))
        col = col_of.get_indexer(sid)
        row = train_index.get_indexer(ed)
        keep = (row >= 0) & (col >= 0)
        present[row[keep], col[keep]] = True
        if verbose:
            print(f"    {lo.date()}: {int(keep.sum()):,} genuine rows", flush=True)

    return present, train_index


def longest_false_run(present: np.ndarray) -> np.ndarray:
    """Return the longest run of consecutive ``False`` values per column, in time steps."""
    run = np.zeros(present.shape[1], dtype=np.int64)
    best = np.zeros(present.shape[1], dtype=np.int64)
    for row in present:
        run = (run + 1) * ~row   # resets to 0 wherever a value is present
        np.maximum(best, run, out=best)
    return best


def _undirected_simple(graph, nodes=None):
    """
    Return a simple undirected graph with the same nodes and edges as the given directed graph.
    """
    sub = graph.subgraph(nodes) if nodes is not None else graph
    g = nx.Graph()
    g.add_nodes_from(sub.nodes(data=True))
    for u, v, d in sub.edges(data=True):
        if u == v:
            continue
        w, dist = float(d["weight"]), float(d["distance"])
        if g.has_edge(u, v):
            if w > g[u][v]["weight"]:
                g[u][v].update(weight=w, distance=dist)
        else:
            g.add_edge(u, v, weight=w, distance=dist)
    return g


def _nearest(sensors: pd.DataFrame, lat: float, lon: float, candidates) -> int:
    """
    Return the sensor id of the nearest candidate to the given latitude and longitude.
    """
    cand = sensors.loc[sorted(candidates)]
    d2 = ((cand["latitude"] - lat) ** 2
          + ((cand["longitude"] - lon) * np.cos(np.radians(lat))) ** 2)
    return int(d2.idxmin())


def _grow_region(graph, sensors: pd.DataFrame, eligible: set[int],
                 n_nodes: int, centre=C.MADRID_CENTRE) -> list[int]:
    """
    Grow a connected region of sensors from the centre, returning up to ``n_nodes``.
    The region is grown in breadth-first order, with ties broken by distance to the centre.
    """
    g = _undirected_simple(graph, eligible)
    if g.number_of_nodes() == 0:
        return []
    biggest = max(nx.connected_components(g), key=len)
    seed = _nearest(sensors, *centre, biggest)

    def dist(n):
        return float(np.hypot(sensors.at[seed, "utm_x"] - sensors.at[n, "utm_x"],
                              sensors.at[seed, "utm_y"] - sensors.at[n, "utm_y"]))

    selected: list[int] = []
    for layer in nx.bfs_layers(g, seed):
        for n in sorted(layer, key=dist):
            selected.append(int(n))
            if len(selected) >= n_nodes:
                return sorted(selected)
    return sorted(selected)


def build(path=C.READINGS_PARQUET, out_dir=C.OUT_DIR,
          min_coverage=C.MIN_TRAIN_COVERAGE,
          max_gap=C.MAX_TRAIN_GAP_STEPS,
          selection=C.SELECTION, n_nodes=C.N_NODES,
          verbose=True) -> tuple[pd.DataFrame, dict]:
    """Choose the sensors and save the selection table."""

    # load the sensor metadata, route graph, and the unique sensor ids
    sensors = raw.load_sensors()
    graph = raw.load_route_graph()
    reading_ids = raw.reading_sensor_ids(path, cache=out_dir / "_reading_sensor_ids.npy", verbose=verbose)

    df = pd.DataFrame(index=sensors.index.copy())
    df.index.name = "sensor_id"
    for col in ("latitude", "longitude", "utm_x", "utm_y", "district", "element_type", "lanes", "max_speed", "max_capacity", "has_data"):
        df[col] = sensors[col]

    # get the graph membership and degree, and whether the sensor is in the dataset at all
    gdeg = dict(_undirected_simple(graph).degree())
    df["in_graph"] = df.index.isin([int(n) for n in graph.nodes])
    df["graph_degree"] = [int(gdeg.get(int(i), 0)) for i in df.index]
    df["in_readings"] = df.index.isin(reading_ids.tolist())

    # Calculate the training coverage
    if verbose:
        print("measuring training coverage for "f"{len(reading_ids):,} sensors...", flush=True)
    present, train_index = train_presence(reading_ids, path, verbose=verbose)
    n_train_slots = present.shape[0]

    cov = pd.Series(present.mean(axis=0), index=reading_ids, dtype="float64")
    nobs = pd.Series(present.sum(axis=0), index=reading_ids, dtype="int64")
    gap = pd.Series(longest_false_run(present), index=reading_ids, dtype="int64")

    df["train_coverage"] = cov.reindex(df.index).astype("float64")
    df["train_n_observed"] = nobs.reindex(df.index).fillna(0).astype("int64")
    df["train_max_gap_steps"] = gap.reindex(df.index)
    df.loc[df["train_max_gap_steps"].isna(), "train_max_gap_steps"] = n_train_slots
    df["train_max_gap_steps"] = df["train_max_gap_steps"].astype("int64")
    df["train_coverage"] = df["train_coverage"].fillna(0.0)

    # create eligible sensors, and log reason for exclusion of each ineligible sensor
    rules = [
        ("no_readings", ~df["in_readings"]),
        ("missing_coordinates", df[["latitude", "longitude"]].isna().any(axis=1)),
        ("missing_district", df["district"].isna() if C.REQUIRE_DISTRICT
         else pd.Series(False, index=df.index)),
        ("not_in_graph", ~df["in_graph"] if C.REQUIRE_GRAPH_MEMBERSHIP
         else pd.Series(False, index=df.index)),
        (f"train_coverage_below_{min_coverage}", df["train_coverage"] < min_coverage),
        (f"train_gap_above_{max_gap}", df["train_max_gap_steps"] > max_gap),
    ]
    reason = pd.Series("", index=df.index, dtype="object")
    for name, failed in rules:
        reason = reason.where(~(failed & (reason == "")), name)
    df["eligible"] = reason == ""

    # select the sensors to include
    # drop sensors in districts that are too small
    eligible = set(int(i) for i in df.index[df["eligible"]])
    district_of = df["district"]
    banned: set[int] = set()   # districts below MIN_DISTRICT_SENSORS

    def too_small(ids):
        counts = district_of.loc[sorted(ids)].value_counts(dropna=True)
        return {int(d) for d, n in counts.items() if n < C.MIN_DISTRICT_SENSORS}

    if selection == "all":
        chosen = sorted(eligible)
        banned = too_small(chosen)
        chosen = [s_ for s_ in chosen
                  if pd.notna(district_of.get(s_))
                  and int(district_of[s_]) not in banned]
    elif selection == "region":
        raise ValueError("implement region selection later")
    else:
        raise ValueError(f"unknown selection: {selection!r}")

    df["included"] = df.index.isin(chosen)
    df["exclusion_reason"] = reason.where(~df["included"], "")

    # fix array ordering
    df["array_index"] = -1
    df.loc[chosen, "array_index"] = np.arange(len(chosen), dtype=np.int64)
    df = df.reset_index()
    df = df[SENSOR_SELECTION_COLUMNS + ["eligible"]]
    df = df.sort_values(["included", "array_index", "sensor_id"], ascending=[False, True, True]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "sensor_selection.csv", index=False)

    inc = df[df["included"]].sort_values("array_index")

    
    summary = {
        "districts_dropped_as_too_small": sorted(banned),
        "n_train_slots": int(n_train_slots),
        "n_in_metadata": int(len(df)),
        "n_eligible": int(df["eligible"].sum()),
        "n_included": int(len(inc)),
        "exclusion_reasons": df.loc[~df["included"], "exclusion_reason"]
            .value_counts().to_dict(),
        "included_train_coverage": {
            "min": float(inc["train_coverage"].min()),
            "mean": float(inc["train_coverage"].mean()),
            "max": float(inc["train_coverage"].max()),
        },
        "included_max_gap_steps": {
            "max": int(inc["train_max_gap_steps"].max()),
            "mean": float(inc["train_max_gap_steps"].mean()),
        },
        "included_isolated_in_graph": int((inc["graph_degree"] == 0).sum()),
        "districts": {str(k): int(v) for k, v in
                      inc["district"].value_counts().sort_index().items()},
    }

    C.write_json(summary, C.reports_dir(out_dir) / "sensor_selection_summary.json")

    if verbose:
        print(f"  {summary['n_eligible']:,} eligible of {len(df):,}; "
              f"{summary['n_included']:,} included across "
              f"{len(summary['districts'])} districts")
        print(f"  wrote {out_dir / 'sensor_selection.csv'}")
    return df, summary


def load(out_dir=C.OUT_DIR) -> pd.DataFrame:
    """Load the selection table; included rows are ordered by ``array_index``."""
    df = pd.read_csv(out_dir / "sensor_selection.csv")
    df["district"] = df["district"].astype("Int64")
    return df


def included(out_dir=C.OUT_DIR) -> pd.DataFrame:
    """load and return only the included sensors"""
    df = load(out_dir)
    return df[df["included"]].sort_values("array_index").reset_index(drop=True)


def sensor_order(out_dir=C.OUT_DIR) -> np.ndarray:
    """Return the sensor ids of the included sensors, in array order."""
    return included(out_dir)["sensor_id"].to_numpy(dtype=np.int64)
