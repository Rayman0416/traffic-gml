# Stage 1: Create overview of the raw readings and metadata.

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from . import config as C
from . import raw, splits


def _ts(t: pd.Timestamp):
    """Convert a pandas Timestamp to a pyarrow timestamp"""
    return pa.scalar(pd.Timestamp(t).to_pydatetime(), pa.timestamp("s"))


def month_slices(start=C.PERIOD_START, end=C.PERIOD_END):
    """Create monthly slices for the given period"""
    edges = pd.date_range(start, end, freq="MS")
    edges = pd.DatetimeIndex([start]).union(edges).union(pd.DatetimeIndex([end]))
    return list(zip(edges[:-1], edges[1:]))


def check_columns(path=C.READINGS_PARQUET) -> dict:
    """Check the columns and value domains of the given readings parquet file"""
    dataset = raw.readings_dataset(path)

    # json cannot store a pyarrow schema object, so we write the columns
    # down as plain strings: {column name: type}
    schema = {}
    for field in dataset.schema:
        schema[field.name] = str(field.type)

    n_rows = 0
    read_error_counts: dict[str, int] = {}
    interp_counts: dict[str, int] = {}
    intensity_null = 0
    intensity_nonfinite = 0
    neg = 0
    zero = 0
    above_hard = 0
    vmax = -np.inf
    sensor_ids: set[int] = set()

    for lo, hi in month_slices():
        filter = ((pc.field("entry_date") >= _ts(lo)) & (pc.field("entry_date") < _ts(hi)))
        table = dataset.to_table(
            columns=["sensor_id", "traffic_intensity", "read_error",
                     "is_interpolated", "original_traffic_intensity"],
            filter=filter,
        )
        n_rows += table.num_rows

        # Count the values of the categorical columns read_error and is_interpolated
        for col, acc in (("read_error", read_error_counts), ("is_interpolated", interp_counts)):
            vc = table.column(col).value_counts()
            for item in vc.to_pylist():
                key = item["values"]
                key_str = "<null>" if key is None else key
                acc[key_str] = acc.get((key_str), 0) + item["counts"]

        # Analyse the traffic_intensity domain
        col_intensity = table.column("traffic_intensity")
        intensity_null += col_intensity.null_count
        v = col_intensity.to_numpy(zero_copy_only=False)
        finite_mask = np.isfinite(v)
        intensity_nonfinite += np.count_nonzero(~finite_mask)
        vv = v[finite_mask]
        if vv.size > 0:
            neg += np.count_nonzero(vv < 0)
            zero += np.count_nonzero(vv == 0)
            above_hard += np.count_nonzero(vv > C.HARD_CAP_VEH_PER_HOUR)
            vmax = max(vmax, float(vv.max()))

        # Collect the sensor ids seen in this slice
        sensor_ids.update(table.column("sensor_id").to_pandas())

    return {
        "schema": schema,
        "n_rows": n_rows,
        "read_error_values": read_error_counts,
        "is_interpolated_values": interp_counts,
        "intensity_null": intensity_null,
        "intensity_non_finite": intensity_nonfinite,
        "intensity_negative": neg,
        "intensity_zero": zero,
        "intensity_zero_note": "zero is treated as potentially valid traffic",
        "intensity_above_hard_cap": above_hard,
        "intensity_max": None if vmax == -np.inf else vmax,
        "n_distinct_sensor_ids": len(sensor_ids),
        "sensor_ids": sorted(int(s) for s in sensor_ids),
    }


def check_interpolation(path=C.READINGS_PARQUET) -> dict:
    """
    Check the interpolation and read-error flags in the given readings parquet file.
    """
    dataset = raw.readings_dataset(path)
    n_int = 0 # number of rows where intensity is interpolated
    n_occ = 0 # number of rows where occupancy is interpolated
    n_err = 0 # number of rows where read_error is not 'N' no error
    n_err_not_interp = 0 # number of rows where read_error is not 'N' and intensity is not interpolated
    n_interp_no_err = 0 # number of rows where intensity is interpolated and read_error is 'N' no error


    for lo, hi in month_slices():
        filter = ((pc.field("entry_date") >= _ts(lo)) & (pc.field("entry_date") < _ts(hi)))
        table = dataset.to_table(columns=["read_error", "is_interpolated"], filter=filter)

        # turn the two columns into normal python lists so we can loop over them
        flag_list = table.column("is_interpolated").to_pylist()
        error_list = table.column("read_error").to_pylist()

        for i in range(len(flag_list)):
            flag = flag_list[i]
            error = error_list[i]

            # the flag is a string of 3 digits, for example "100".
            # digit 0 = intensity, digit 1 = occupancy, 1 means interpolated.
            # if the value is missing or does not have 3 digits we cannot
            # trust it, so we count it as "not interpolated".
            if flag is not None and len(flag) == 3:
                intensity_is_interpolated = (flag[0] == "1")
                occupancy_is_interpolated = (flag[1] == "1")
            else:
                intensity_is_interpolated = False
                occupancy_is_interpolated = False

            # "N" means no error. anything else (also a missing value)
            # is counted as an error.
            has_error = (error != "N")

            if intensity_is_interpolated:
                n_int = n_int + 1
            if occupancy_is_interpolated:
                n_occ = n_occ + 1
            if has_error:
                n_err = n_err + 1
            if has_error and intensity_is_interpolated == False:
                n_err_not_interp = n_err_not_interp + 1
            if intensity_is_interpolated and has_error == False:
                n_interp_no_err = n_interp_no_err + 1

    return {
        "intensity_interpolated_rows": n_int,
        "occupancy_interpolated_rows": n_occ,
        "read_error_rows": n_err,
        "read_error_and_not_intensity_interpolated": n_err_not_interp,
        "intensity_interpolated_and_no_read_error": n_interp_no_err,
        "rule": (
            "intensity is genuine iff read_error == 'N' and the intensity digit of is_interpolated is 0"
        ),
    }


def check_duplicates(path=C.READINGS_PARQUET) -> dict:
    """
    Check for duplicate (sensor_id, entry_date) pairs, and whether they conflict.
    """
    dataset = raw.readings_dataset(path)
    n_pairs = n_dup_pairs = n_dup_rows = n_conflicting = 0
    examples = []

    for lo, hi in month_slices():
        filter = ((pc.field("entry_date") >= _ts(lo)) & (pc.field("entry_date") < _ts(hi)))
        table = dataset.to_table(columns=["sensor_id", "entry_date", "traffic_intensity"], filter=filter)
        grp = table.group_by(["sensor_id", "entry_date"]).aggregate([
            ("traffic_intensity", "count"),
            ("traffic_intensity", "min"),
            ("traffic_intensity", "max"),
        ])
        cnt = grp.column("traffic_intensity_count").to_numpy(zero_copy_only=False)
        vmin = grp.column("traffic_intensity_min").to_numpy(zero_copy_only=False)
        vmax = grp.column("traffic_intensity_max").to_numpy(zero_copy_only=False)

        n_pairs += len(cnt)
        dup = cnt > 1
        n_dup_pairs += int(dup.sum())
        n_dup_rows += int(cnt[dup].sum() - dup.sum())
        conflict = dup & ~np.isclose(np.nan_to_num(vmin, nan=0.0), np.nan_to_num(vmax, nan=0.0))
        n_conflicting += int(conflict.sum())
        if conflict.any() and len(examples) < 5:
            sid = grp.column("sensor_id").to_numpy(zero_copy_only=False)
            ed = grp.column("entry_date").to_pylist()
            for i in np.flatnonzero(conflict)[:5 - len(examples)]:
                examples.append({
                    "sensor_id": int(sid[i]),
                    "entry_date": pd.Timestamp(ed[i]).isoformat(),
                    "min": float(vmin[i]), "max": float(vmax[i]),
                })

    return {
        "distinct_sensor_timestamp_pairs": n_pairs,
        "duplicated_pairs": n_dup_pairs,
        "surplus_duplicate_rows": n_dup_rows,
        "conflicting_duplicated_pairs": n_conflicting,
        "conflict_examples": examples,
        "policy": (
            "agreeing duplicates are collapsed and flagged DUPLICATE_RESOLVED; "
            "conflicting duplicates are rejected and flagged DUPLICATE_CONFLICT "
            "- never averaged"
        ),
    }


def check_sensor_sets(reading_sensor_ids: list[int]) -> dict:
    """Which sensor ids appear in the readings, the metadata and the graph."""
    sensors = raw.load_sensors()
    graph = raw.load_route_graph()

    meta = set(int(i) for i in sensors.index)
    gnodes = set(int(n) for n in graph.nodes)
    rsens = set(int(s) for s in reading_sensor_ids)

    static_cols = ["latitude", "longitude", "utm_x", "utm_y", "district"]
    return {
        "n_metadata": len(meta),
        "n_graph_nodes": len(gnodes),
        "n_reading_sensors": len(rsens),
        "readings_and_metadata": len(rsens & meta),
        "readings_without_metadata": sorted(rsens - meta)[:20],
        "n_readings_without_metadata": len(rsens - meta),
        "metadata_without_readings": len(meta - rsens),
        "graph_not_in_metadata": sorted(gnodes - meta)[:20],
        "n_graph_not_in_metadata": len(gnodes - meta),
        "metadata_not_in_graph": len(meta - gnodes),
        "readings_not_in_graph": len(rsens - gnodes),
        "metadata_is_static_snapshot": True,
        "metadata_static_note": (
            "traffic_sensors.csv holds exactly one row per sensor id, so "
            f"{static_cols} cannot change within the period; any real "
            "relocation would be invisible in this source"
        ),
        "has_data_flag": sensors["has_data"].value_counts().to_dict(),
        "district_missing": int(sensors["district"].isna().sum()),
        "lanes_missing": int(sensors["lanes"].isna().sum()),
        "max_capacity_missing": int(sensors["max_capacity"].isna().sum()),
    }


def run(path=C.READINGS_PARQUET, out_dir=C.OUT_DIR, verbose=True) -> dict:
    if verbose:
        print("  columns and value domains...", flush=True)
    cols = check_columns(path)
    reading_ids = cols.pop("sensor_ids")

    if verbose:
        print("  interpolation and read-error flags...", flush=True)
    interp = check_interpolation(path)

    if verbose:
        print("  duplicate sensor-timestamp pairs...", flush=True)
    dups = check_duplicates(path)

    if verbose:
        print("  sensor id sets...", flush=True)
    sets_ = check_sensor_sets(reading_ids)

    report = {
        "stage": "audit",
        "intensity_unit": C.INTENSITY_UNIT,
        "columns": cols,
        "quality_flags": interp,
        "duplicates": dups,
        "sensor_sets": sets_,
        "splits": {"bounds": splits.describe()},
    }
    
    C.write_json(report, C.reports_dir(out_dir) / "audit.json")
    # Same cache `raw.reading_sensor_ids` uses, so whichever stage runs first
    # populates it and the other reuses it.
    np.save(out_dir / "_reading_sensor_ids.npy",
            np.sort(np.asarray(reading_ids, dtype=np.int64)))
    if verbose:
        print(f"  wrote {C.reports_dir(out_dir) / 'audit.json'}")
    return report

if __name__ == "__main__":
    run()
