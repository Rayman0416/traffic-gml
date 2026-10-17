# Stage 4: Build the observed-data and quality layers

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from . import config as C
from . import clean, raw, sensor_selection, splits, timeline
from .config import Flag


def _ts(t):
    """Convert a pandas Timestamp to a pyarrow timestamp scalar."""
    return pa.scalar(pd.Timestamp(t).to_pydatetime(), pa.timestamp("s"))


def _month_edges(start=C.PERIOD_START, end=C.PERIOD_END):
    """Return a list of (lo, hi) month edges covering the period."""
    edges = pd.date_range(start, end, freq="MS")
    edges = pd.DatetimeIndex([start]).union(edges).union(pd.DatetimeIndex([end]))
    return list(zip(edges[:-1], edges[1:]))


def build_raw_layer(sensor_ids: np.ndarray, path=C.READINGS_PARQUET,
                    verbose=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build the raw intensity and occupancy arrays, and the quality flags.
    Returns ``(raw_intensity, raw_occupancy, flags, report)``.
    """
    index = timeline.build_index()
    sensor_ids = np.asarray(sensor_ids, dtype=np.int64)
    T, N = len(index), len(sensor_ids) # time steps, sensors
    col_of = pd.Index(sensor_ids)

    # init arrays
    values = np.full((T, N), np.nan, dtype=np.float32)
    occ_values = np.full((T, N), np.nan, dtype=np.float32)
    flags = np.zeros((T, N), dtype=np.uint16) # see config for bit definitions
    seen = np.zeros((T, N), dtype=bool)

    # Read the raw data
    dataset = raw.readings_dataset(path)
    sid_arr = pa.array(sensor_ids, pa.int32())

    # Count rows and duplicates for report
    n_rows = 0
    dup_pairs = dup_conflicts = 0
    off_grid = 0

    # Loop over the months and process the data
    for lo, hi in _month_edges():
        flt = (pc.field("sensor_id").isin(sid_arr)
               & (pc.field("entry_date") >= _ts(lo))
               & (pc.field("entry_date") < _ts(hi)))
        
        tb = dataset.to_table(
            columns=["sensor_id", "entry_date", "traffic_intensity",
                     "sensor_occupancy", "read_error", "is_interpolated"],
            filter=flt,
        )

        if tb.num_rows == 0:
            continue
        n_rows += tb.num_rows

        # Convert the columns to numpy arrays
        sid = tb.column("sensor_id").to_numpy(zero_copy_only=False).astype(np.int64)
        ed = pd.DatetimeIndex(tb.column("entry_date").to_numpy(zero_copy_only=False))
        val = tb.column("traffic_intensity").to_numpy(zero_copy_only=False)
        val = np.asarray(val, dtype=np.float64)
        occ_val = np.asarray(tb.column("sensor_occupancy").to_numpy(zero_copy_only=False), dtype=np.float64)

        # check if flag is 3 bits long
        flag_col = tb.column("is_interpolated")
        three = pc.equal(pc.binary_length(flag_col), 3)

        # interpolated traffic intensity and occupancy flag (110)
        # itp is true when first bit of flag is 1, and occ is true when second bit of flag is 1
        itp = np.asarray(pc.fill_null(pc.and_(pc.is_valid(flag_col), 
                        pc.and_(three, pc.starts_with(flag_col, "1"))), False)
                        .to_numpy(zero_copy_only=False), dtype=bool)
        occ = np.asarray(pc.fill_null(pc.and_(pc.is_valid(flag_col), 
                        pc.and_(three,pc.equal(pc.utf8_slice_codeunits(flag_col, 1, 2), "1"))), False)
                        .to_numpy(zero_copy_only=False), dtype=bool)

        # read error flag (1 if read_error is not "N" -> error, 0 if read_error is "N" -> no error)
        re_col = tb.column("read_error")
        err = ~np.asarray(pc.fill_null(pc.equal(re_col, "N"), False).to_numpy(zero_copy_only=False), dtype=bool)


        col = col_of.get_indexer(sid) # get the column index for each sensor_id in the sensor_ids array
        row = index.get_indexer(ed) # get the row index for each entry_date in the timeline index
        keep = (row >= 0) & (col >= 0) # keep only the rows and columns that are in the grid
        off_grid += int((~keep).sum()) # count the number of rows that are off-grid
        row, col = row[keep], col[keep]
        val, itp, occ, err, occ_val = val[keep], itp[keep], occ[keep], err[keep], occ_val[keep]

        # Check for duplicates
        counts = np.zeros((T, N), dtype=np.int8)
        np.add.at(counts, (row, col), 1) # count the number of occurrences of each (row, col) pair
        if counts.max() > 1:
            dmask = counts[row, col] > 1
            dup_pairs += int(np.unique(np.stack([row[dmask], col[dmask]]), axis=1).shape[1])
            df = pd.DataFrame({"row": row[dmask], "col": col[dmask], "val": val[dmask], "occ": occ_val[dmask]})
            agg = df.groupby(["row", "col"])[["val", "occ"]].agg(["min", "max"])
            conflict = np.zeros(len(agg), dtype=bool)

            # if the min and max for val and occ are not close enough, then there is a conflict
            for c in ("val", "occ"):
                conflict |= ~np.isclose(agg[(c, "min")].to_numpy(), agg[(c, "max")].to_numpy(), equal_nan=True)

            # assign duplicate resolved flag
            cr = agg.index.get_level_values(0).to_numpy()
            cc = agg.index.get_level_values(1).to_numpy()
            flags[cr, cc] |= Flag.DUPLICATE_RESOLVED

            # if there is a conflict, assign the duplicate conflict flag
            if conflict.any():
                dup_conflicts += int(conflict.sum())
                flags[cr[conflict], cc[conflict]] |= Flag.DUPLICATE_CONFLICT

        del counts # delete counts to free memory immediately

        # Fill the arrays with the values, occupancy values, and flags
        values[row, col] = val 
        occ_values[row, col] = occ_val
        seen[row, col] = True
        flags[row[itp], col[itp]] |= Flag.INTENSITY_INTERPOLATED
        flags[row[occ], col[occ]] |= Flag.OCCUPANCY_INTERPOLATED
        flags[row[err], col[err]] |= Flag.READ_ERROR

        if verbose:
            print(f"{lo.date()}: {tb.num_rows:,} rows", flush=True)

    # Cells with no raw row at all.
    flags[~seen] |= Flag.RAW_MISSING

    # Present but unusable as a number.
    bad = seen & ~np.isfinite(values)
    flags[bad] |= Flag.NON_NUMERIC
    neg = seen & np.isfinite(values) & (values < 0)
    if C.REJECT_NEGATIVE:
        flags[neg] |= Flag.NEGATIVE

    # Occupancy flags: non-numeric and out-of-range
    occ_bad = seen & ~np.isfinite(occ_values)
    flags[occ_bad] |= Flag.OCCUPANCY_NON_NUMERIC
    occ_range = seen & np.isfinite(occ_values) & (
        (occ_values < C.OCCUPANCY_MIN_PERCENT)
        | (occ_values > C.OCCUPANCY_MAX_PERCENT))
    flags[occ_range] |= Flag.OCCUPANCY_OUT_OF_RANGE

    # summary
    report = {
        "n_cells": int(T * N),
        "n_cells_with_raw_row": int(seen.sum()),
        "raw_row_presence": float(seen.mean()),
        "off_grid_rows_dropped": off_grid,
        "duplicate_pairs": dup_pairs,
        "duplicate_conflicts": dup_conflicts,
        "non_numeric_cells": int(bad.sum()),
        "negative_cells": int(neg.sum()),
        "occupancy_non_numeric_cells": int(occ_bad.sum()),
        "occupancy_out_of_range_cells": int(occ_range.sum()),
    }

    return values, occ_values, flags, report


def finalise(raw_intensity: np.ndarray, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the mask and return the observed intensity and mask arrays"""
    mask = (flags & Flag.REJECTING) == 0
    # A cell with no rejecting flag must still carry a finite number.
    mask &= np.isfinite(raw_intensity)
    observed = np.where(mask, raw_intensity, np.nan).astype(np.float32)
    return observed, mask


def finalise_occupancy(raw_occupancy: np.ndarray,
                       flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the mask and return the observed occupancy and mask arrays"""
    mask = (flags & Flag.OCCUPANCY_REJECTING) == 0
    mask &= np.isfinite(raw_occupancy)
    observed = np.where(mask, raw_occupancy, np.nan).astype(np.float32)
    return observed, mask


def build(out_dir=C.OUT_DIR, path=C.READINGS_PARQUET, verbose=True) -> dict:
    """Build and save the observed-data and quality layers."""
    index = timeline.build_index() # build timeline
    sensor_ids = sensor_selection.sensor_order(out_dir) # selected sensors from previous stage

    if verbose:
        print(f"  gridding {len(sensor_ids)} sensors x {len(index):,} slots...", flush=True)

    # get raw intensity, occupancy, and flags arrays, along with a grid report
    raw_intensity, raw_occupancy, flags, grid_report = build_raw_layer(sensor_ids, path, verbose)

    # before cleaning
    _, mask_source = finalise(raw_intensity, flags)

    if verbose:
        print("  applying cleaning rules...", flush=True)
    flags = clean.apply(raw_intensity, flags, out_dir=out_dir, verbose=verbose)

    observed, mask = finalise(raw_intensity, flags)
    observed_occ, occ_mask = finalise_occupancy(raw_occupancy, flags)

    # compute coverage per data split
    split_code = splits.split_of(index)
    coverage = {}
    for name, code in splits.SPLIT_CODE.items():
        sel = split_code == code
        coverage[name] = {
            "n_steps": int(sel.sum()),
            "observed_fraction": float(mask[sel].mean()),
            "raw_row_fraction": float(((flags[sel] & Flag.RAW_MISSING) == 0).mean()),
            "occupancy_observed_fraction": float(occ_mask[sel].mean()),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "observed.npz",
        sensor_ids=sensor_ids,
        timestamps=index.values.astype("datetime64[s]"),
        raw_intensity=raw_intensity,
        observed_intensity=observed,
        mask=mask,
        observed_occupancy=observed_occ,
        raw_occupancy=raw_occupancy,
        occupancy_mask=occ_mask,
        flags=flags,
        split_code=split_code,
    )

    summary = {
        "shape": {"T": int(len(index)), "N": int(len(sensor_ids))},
        "grid": grid_report,
        "flag_counts": Flag.decompose(flags),
        "mask": {
            "observed_fraction_overall": float(mask.mean()),
            "observed_fraction_before_cleaning": float(mask_source.mean()),
            "per_split": coverage,
        },
        "observed_intensity_stats": {
            "min": float(np.nanmin(observed)),
            "median": float(np.nanmedian(observed)),
            "p99": float(np.nanpercentile(observed, 99)),
            "max": float(np.nanmax(observed)),
            "n_zero": int(np.nansum(observed == 0)),
        },
        "occupancy_mask": {
            "observed_fraction_overall": float(occ_mask.mean()),
            "both_observed_fraction": float((mask & occ_mask).mean()),
            "intensity_only_fraction": float((mask & ~occ_mask).mean()),
            "occupancy_only_fraction": float((~mask & occ_mask).mean()),
        },
        "observed_occupancy_stats": {
            "min": float(np.nanmin(observed_occ)),
            "median": float(np.nanmedian(observed_occ)),
            "p99": float(np.nanpercentile(observed_occ, 99)),
            "max": float(np.nanmax(observed_occ)),
            "n_zero": int(np.nansum(observed_occ == 0)),
        },
    }

    C.write_json(summary, C.reports_dir(out_dir) / "observed_summary.json")

    if verbose:
        print(f"  observed fraction {mask.mean():.2%} "
              f"(train {coverage['train']['observed_fraction']:.2%}, "
              f"val {coverage['val']['observed_fraction']:.2%}, "
              f"test {coverage['test']['observed_fraction']:.2%}); "
              f"occupancy {occ_mask.mean():.2%}")
        print(f"  wrote {out_dir / 'observed.npz'}")

    return summary


def load(out_dir=C.OUT_DIR) -> dict:
    """Load the observed and quality layers."""
    z = np.load(out_dir / "observed.npz")

    if "occupancy_mask" not in z.files:
        raise KeyError(f"{out_dir / 'observed.npz'} predates the occupancy layer - re-run the `observed` stage")
    
    return {
        "sensor_ids": z["sensor_ids"],
        "timestamps": pd.DatetimeIndex(z["timestamps"]),
        "raw_intensity": z["raw_intensity"],
        "observed_intensity": z["observed_intensity"],
        "mask": z["mask"],
        "observed_occupancy": z["observed_occupancy"],
        "occupancy_mask": z["occupancy_mask"],
        "flags": z["flags"],
        "split_code": z["split_code"],
    }
