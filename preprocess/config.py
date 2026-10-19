# Config for preprocessing pipeline.

from pathlib import Path
import json
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

RAW_CSV = DATA_DIR / "traffic_data_2024.csv"
SENSORS_CSV = DATA_DIR / "traffic_sensors.csv"
ROUTE_GRAPH_PKL = DATA_DIR / "routes-graph.pkl"

READINGS_PARQUET = DATA_DIR / "traffic_2024.parquet"
OUT_DIR = DATA_DIR / "forecast_dataset"

# Stage summaries, the audit, the checks and the quality roll-up all live in
# their own subdirectory, so the top level of OUT_DIR holds only the dataset a
# model actually consumes.
REPORTS_SUBDIR = "reports"


def reports_dir(out_dir: Path = OUT_DIR) -> Path:
    """``<out_dir>/reports``, created on demand."""
    d = Path(out_dir) / REPORTS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d

# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------

FREQ = "15min" # timestamp frequency in dataset
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96
STEPS_PER_WEEK = 7 * STEPS_PER_DAY  # 672 time-of-week slots

PERIOD_START = pd.Timestamp("2024-01-01 00:00:00") # timestamp start of dataset
PERIOD_END = pd.Timestamp("2025-01-01 00:00:00")  # timestamp end of dataset (exclusive)

INTERVAL_LABEL = "start"
NOMINAL_UTC_OFFSET_HOURS = 1  # CET
CALENDAR_TZ = "Europe/Madrid"

INTENSITY_UNIT = "vehicles/hour"  # recorded every 15 min, but a rate
OCCUPANCY_UNIT = "percent"        # share of the interval the loop was occupied

# --------------------------------------------------------------------------
# Chronological splits
# --------------------------------------------------------------------------

TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2024-09-01")  # train: Jan-Aug
VAL_END = pd.Timestamp("2024-11-01")    # val:   Sep-Oct
TEST_END = PERIOD_END                   # test:  Nov-Dec

SPLIT_NAMES = ("train", "val", "test")

# --------------------------------------------------------------------------
# Forecasting task
# --------------------------------------------------------------------------

# L: history length in steps, *including* the forecast origin, so the input
# window of origin t is [t-L+1, t].  Shared by the window index and the input
# features; override per run with `--history-length`.
INPUT_LENGTH = 6        # 6 steps = 1h30 of history
HORIZONS = (1, 2, 3, 4)  # +15, +30, +45, +60 minutes
HORIZON_MINUTES = (15, 30, 45, 60)

# --------------------------------------------------------------------------
# Sensor eligibility (decided on TRAINING data only)
# --------------------------------------------------------------------------

MIN_TRAIN_COVERAGE = 0.80     # fraction of training slots genuinely observed
MAX_TRAIN_GAP_STEPS = 500     # longest run of consecutive training gaps allowed
REQUIRE_GRAPH_MEMBERSHIP = False  # graph isolation alone must not exclude a sensor
REQUIRE_DISTRICT = True       # district partitioning needs a district label

# "all" or "districts". all keeps every eligible sensor in the network; districts
# takes every eligible sensor in a connected block of N_DISTRICTS whole districts
# grown from the centre
SELECTION = "all"
N_DISTRICTS = 6 # districts to select when SELECTION == "districts" only

# A district needs enough sensors to be worth modelling as a graph at all
MIN_DISTRICT_SENSORS = 10
MADRID_CENTRE = (40.4168, -3.7038)  # Puerta del Sol (lat, lon)

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

# Zero is treated as potentially valid traffic and is never rejected but negative values are.
REJECT_NEGATIVE = True
HARD_CAP_VEH_PER_HOUR = 20000.0   # far above any plausible urban 15-min rate

# Occupancy is an integer percentage; anything outside [0, 100] is impossible.
OCCUPANCY_MIN_PERCENT = 0.0
OCCUPANCY_MAX_PERCENT = 100.0

# --------------------------------------------------------------------------
# Causal filling
# --------------------------------------------------------------------------

FFILL_MAX_STEPS = 4           # carry a genuine observation at most 1 hour
PROFILE_MIN_SUPPORT = 4       # observations needed per (sensor, time-of-week)
AGE_CLIP_STEPS = 96           # cap the elapsed-time feature at one day

# --------------------------------------------------------------------------
# Window eligibility
# --------------------------------------------------------------------------

MIN_INPUT_AVAILABILITY = 0.70          # citywide, over the input window
MIN_DISTRICT_INPUT_AVAILABILITY = 0.50  # worst district, over the input window

# --------------------------------------------------------------------------
# Scaling
# --------------------------------------------------------------------------

SCALER = "per_sensor_standard"
MIN_STD_VEH_PER_HOUR = 1.0    # safeguard for near-constant sensors
MIN_STD_OCCUPANCY_PERCENT = 0.5

# --------------------------------------------------------------------------
# Input features
# --------------------------------------------------------------------------

# Value written into a scaled feature wherever its validity mask is False.
# 0 in standardised units is the sensor's *training* mean, so a missing input
# carries no information from outside the training period or after the origin;
# the mask is what tells a model the value is not a measurement.
FEATURE_MISSING_VALUE = 0.0

RANDOM_SEED = 20241015


# --------------------------------------------------------------------------
# Quality flag bits
# --------------------------------------------------------------------------

class Flag:
    """
    Bit flags for the quality of each (sensor, timestamp) cell in the
    preprocessed dataset.
    """

    RAW_MISSING = 1 << 0            # no raw row existed for this (sensor, t)
    INTENSITY_INTERPOLATED = 1 << 1  # GO-MO supplied an interpolated intensity
    READ_ERROR = 1 << 2             # source read_error flag was not "N"
    NON_NUMERIC = 1 << 3            # intensity present but not a finite number
    DUPLICATE_CONFLICT = 1 << 4     # duplicate rows disagreed on a value
    NEGATIVE = 1 << 5               # negative intensity
    ABOVE_HARD_CAP = 1 << 6         # above the absolute physical cap
    OCCUPANCY_INTERPOLATED = 1 << 7  # interpolated occupancy only, not intensity
    DUPLICATE_RESOLVED = 1 << 8      # duplicates agreed and were collapsed
    OCCUPANCY_NON_NUMERIC = 1 << 9   # occupancy present but not a finite number
    OCCUPANCY_OUT_OF_RANGE = 1 << 10  # occupancy outside [0, 100] percent

    # GO-MO's interpolated values are accepted for now, but might have to be filled in manually in the future
    # to prevent test leakage

    # Traffic intensity rejecting flags
    REJECTING = (
        RAW_MISSING
        | READ_ERROR
        | NON_NUMERIC
        | DUPLICATE_CONFLICT
        | NEGATIVE
        | ABOVE_HARD_CAP
    )

    # Occupancy rejecting flags
    OCCUPANCY_REJECTING = (
        RAW_MISSING
        | READ_ERROR
        | DUPLICATE_CONFLICT
        | OCCUPANCY_NON_NUMERIC
        | OCCUPANCY_OUT_OF_RANGE
    )

    NAMES = {
        RAW_MISSING: "raw_missing",
        INTENSITY_INTERPOLATED: "intensity_interpolated",
        READ_ERROR: "read_error",
        NON_NUMERIC: "non_numeric",
        DUPLICATE_CONFLICT: "duplicate_conflict",
        NEGATIVE: "negative",
        ABOVE_HARD_CAP: "above_hard_cap",
        OCCUPANCY_INTERPOLATED: "occupancy_interpolated",
        DUPLICATE_RESOLVED: "duplicate_resolved",
        OCCUPANCY_NON_NUMERIC: "occupancy_non_numeric",
        OCCUPANCY_OUT_OF_RANGE: "occupancy_out_of_range",
    }

    @classmethod
    def decompose(cls, flags: np.ndarray) -> dict[str, int]:
        """Count how many cells carry each individual flag bit."""
        return {name: int((flags & bit).astype(bool).sum())
                for bit, name in cls.NAMES.items()}


# --------------------------------------------------------------------------
# Fill source codes (which rule produced a model input value)
# --------------------------------------------------------------------------

class FillSource:
    OBSERVED = 0        # genuine observation
    FFILL = 1           # forward-filled from a recent genuine observation
    PROFILE = 2         # sensor time-of-week profile (training)
    SENSOR_MEAN = 3     # sensor training mean (profile had no support)
    GLOBAL_MEAN = 4     # global training mean (sensor had no training data)

    NAMES = {
        OBSERVED: "observed",
        FFILL: "ffill",
        PROFILE: "time_of_week_profile",
        SENSOR_MEAN: "sensor_train_mean",
        GLOBAL_MEAN: "global_train_mean",
    }


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------

# settings of produced preprocessed data
RECORDED_SETTINGS = (
    # timeline
    "FREQ", "PERIOD_START", "PERIOD_END",
    # splits
    "TRAIN_START", "TRAIN_END", "VAL_END", "TEST_END",
    # forecasting task
    "INPUT_LENGTH", "HORIZONS",
    # sensor eligibility
    "SELECTION", "N_DISTRICTS", "MIN_TRAIN_COVERAGE", "MAX_TRAIN_GAP_STEPS",
    "REQUIRE_DISTRICT", "REQUIRE_GRAPH_MEMBERSHIP", "MIN_DISTRICT_SENSORS",
    # cleaning
    "REJECT_NEGATIVE", "HARD_CAP_VEH_PER_HOUR",
    "OCCUPANCY_MIN_PERCENT", "OCCUPANCY_MAX_PERCENT",
    # causal filling
    "FFILL_MAX_STEPS", "PROFILE_MIN_SUPPORT", "AGE_CLIP_STEPS",
    # window eligibility
    "MIN_INPUT_AVAILABILITY", "MIN_DISTRICT_INPUT_AVAILABILITY",
    # scaling
    "SCALER", "MIN_STD_VEH_PER_HOUR", "MIN_STD_OCCUPANCY_PERCENT",
    # input features
    "FEATURE_MISSING_VALUE",
    # reproducibility
    "RANDOM_SEED",
)


def settings() -> dict:
    """The tunables this run was parameterised with, in declaration order."""
    g = globals()
    unknown = [k for k in RECORDED_SETTINGS if k not in g]
    if unknown:
        raise KeyError(f"RECORDED_SETTINGS names non-existent settings: {unknown}")
    out = {}
    for k in RECORDED_SETTINGS:
        v = g[k]
        if isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def source(extra: dict | None = None, overrides: dict | None = None) -> dict:
    """Run record: the settings the data was preprocessed with.

    ``overrides`` replaces recorded settings that were changed on the command
    line, so the record shows the values actually used.
    """
    rec = dict(extra or {})
    rec["settings"] = {**settings(), **(overrides or {})}
    return rec


def _round(o, sig: int = 6):
    """
    Round floats to significant figures for JSON output, recursively through dicts/lists.
    """

    if isinstance(o, float):
        return o if (o != o or o in (float("inf"), float("-inf"))) \
            else float(f"{o:.{sig}g}")
    if isinstance(o, (np.floating,)):
        return _round(float(o), sig)
    if isinstance(o, dict):
        return {k: _round(v, sig) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, sig) for v in o]
    return o


def write_json(obj, path: Path) -> Path:
    """Write obj to json file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = _round(obj)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not serialisable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=default)
    return path
