# Chronological splits of the data into train, val, and test sets.

import numpy as np
import pandas as pd
from . import config as C

SPLIT_NAMES = C.SPLIT_NAMES
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
CODE_SPLIT = {v: k for k, v in SPLIT_CODE.items()}

def split_bounds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the start and end timestamps for each split, as a dict of (start, end) pairs."""
    return {
        "train": (C.TRAIN_START, C.TRAIN_END),
        "val": (C.TRAIN_END, C.VAL_END),
        "test": (C.VAL_END, C.TEST_END),
    }


def split_of(index: pd.DatetimeIndex) -> np.ndarray:
    """Split code per timestamp (``0`` train, ``1`` val, ``2`` test)."""
    code = np.full(len(index), -1, dtype=np.int8)
    for name, (lo, hi) in split_bounds().items():
        sel = (index >= lo) & (index < hi)
        code[sel] = SPLIT_CODE[name]
    if (code < 0).any():
        bad = index[code < 0][:3]
        raise ValueError(f"timestamps outside every split, e.g. {list(bad)}")
    return code


def masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Boolean membership mask per split."""
    code = split_of(index)
    return {name: code == c for name, c in SPLIT_CODE.items()}


def train_mask(index: pd.DatetimeIndex) -> np.ndarray:
    return masks(index)["train"]


def describe() -> dict:
    bounds = split_bounds()
    out = {}
    for name, (lo, hi) in bounds.items():
        n = len(pd.date_range(lo, hi, freq=C.FREQ, inclusive="left"))
        out[name] = {
            "start": lo.isoformat(),
            "end_exclusive": hi.isoformat(),
            "n_steps": int(n),
            "n_days": float((hi - lo) / pd.Timedelta(days=1)),
        }
        
    return out
