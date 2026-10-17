# Build the canonical timeline and its calendar features, and check its integrity.

import numpy as np
import pandas as pd
from . import config as C


def build_index(start: pd.Timestamp = C.PERIOD_START, end: pd.Timestamp = C.PERIOD_END) -> pd.DatetimeIndex:
    """Build the canonical timeline index for the period, with 15-min steps."""
    idx = pd.date_range(start, end, freq=C.FREQ, inclusive="left")
    return pd.DatetimeIndex(idx, name="timestamp")


def check_regular(index: pd.DatetimeIndex) -> dict:
    """Assert the index is strictly increasing with a constant 15-min step."""
    deltas = np.diff(index.values).astype("timedelta64[m]").astype(int)
    report = {
        "n_steps": int(len(index)),
        "start": index[0].isoformat(),
        "end": index[-1].isoformat(),
        "unique_deltas_minutes": sorted(set(deltas.tolist())),
        "monotonic_increasing": bool(index.is_monotonic_increasing),
        "has_duplicates": bool(index.duplicated().any()),
    }
    if report["unique_deltas_minutes"] != [15]:
        raise ValueError(f"timeline is not regular: {report}")
    if not report["monotonic_increasing"] or report["has_duplicates"]:
        raise ValueError(f"timeline is not strictly increasing: {report}")
    return report


def dst_resolution(index: pd.DatetimeIndex, tz: str = C.CALENDAR_TZ) -> dict:
    """
    Check whether the index localizes cleanly to the given timezone, and count
    the number of nonexistent and ambiguous timestamps.
    """
    nonexistent = ambiguous = 0
    try:
        index.tz_localize(tz, nonexistent="raise", ambiguous="raise")
        clean = True
    except Exception:
        clean = False
        # Count precisely which stamps fail, by probing both DST folds.
        shifted = index.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
        # A nonexistent stamp is moved by shift_forward.
        nonexistent = int((shifted.tz_localize(None) != index).sum())
        utc_first = index.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
        utc_last = index.tz_localize(tz, nonexistent="shift_forward", ambiguous=False)
        ambiguous = int((utc_first != utc_last).sum())

    return {
        "interval_label": C.INTERVAL_LABEL,
        "localizes_cleanly_as_local_time": clean,
        "stamps_nonexistent_in_local_time": nonexistent,
        "stamps_ambiguous_in_local_time": ambiguous,
        "calendar_features_source": "naive stamps (already Madrid-civil-phased)",
        "nominal_utc_offset_hours": C.NOMINAL_UTC_OFFSET_HOURS,
    }


def nominal_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Return the index shifted to nominal UTC, which is the same as Madrid civil time
    """
    return index - pd.Timedelta(hours=C.NOMINAL_UTC_OFFSET_HOURS)


# --------------------------------------------------------------------------
# Calendar features
# --------------------------------------------------------------------------

def time_of_week(index: pd.DatetimeIndex) -> np.ndarray:
    """Slot within the week, ``0 .. 671`` (Monday 00:00 == 0)."""
    tod = index.hour * C.STEPS_PER_HOUR + index.minute // 15
    return (index.dayofweek * C.STEPS_PER_DAY + tod).to_numpy().astype(np.int16)


def time_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Slot within the day, ``0 .. 95``."""
    return (index.hour * C.STEPS_PER_HOUR + index.minute // 15).to_numpy().astype(np.int16)

# time of day and day of week
CYCLICAL_TIME_FEATURES = ("tod_sin", "tod_cos", "dow_sin", "dow_cos")


def cyclical_time_features(index: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """
    Compute cyclical time features for the given index.
    Returns a tuple of (array, feature_names), where array has shape [T, F] and F is the number of features.
    """
    tod = time_of_day(index).astype(np.float64)
    dow = index.dayofweek.to_numpy().astype(np.float64)

    two_pi = 2.0 * np.pi
    feats = {
        "tod_sin": np.sin(two_pi * tod / C.STEPS_PER_DAY),
        "tod_cos": np.cos(two_pi * tod / C.STEPS_PER_DAY),
        "dow_sin": np.sin(two_pi * dow / 7.0),
        "dow_cos": np.cos(two_pi * dow / 7.0),
    }
    names = list(CYCLICAL_TIME_FEATURES)
    arr = np.stack([feats[n] for n in names], axis=1).astype(np.float32)
    return arr, names


def calendar_features(index: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """Cyclical time features plus a weekend indicator."""
    cyc, names = cyclical_time_features(index)
    weekend = (index.dayofweek.to_numpy() >= 5).astype(np.float32)
    arr = np.concatenate([cyc, weekend[:, None]], axis=1)
    return arr, names + ["is_weekend"]
