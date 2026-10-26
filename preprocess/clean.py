# cleaning and capping of the raw intensity data.

import numpy as np
from . import config as C
from .config import Flag


def apply(raw_intensity: np.ndarray, flags: np.ndarray, out_dir=C.OUT_DIR,
          verbose=True) -> np.ndarray:
    """
    Apply the cleaning rules to the raw intensity array, returning the updated flags.
    """

    flags = flags.copy()
    finite = np.isfinite(raw_intensity)
    usable = ((flags & Flag.REJECTING) == 0) & finite
    n_usable = int(usable.sum())

    rules = {}

    # apply hard cap on intensity to remove extreme outliers
    hard = usable & (raw_intensity > C.HARD_CAP_VEH_PER_HOUR)
    flags[hard] |= Flag.ABOVE_HARD_CAP
    rules["above_hard_cap"] = {
        "cap_veh_per_hour": C.HARD_CAP_VEH_PER_HOUR,
        "cells_rejected": int(hard.sum())}

    # 2. spike filter — recompute usable so spikes are judged against
    #    post-cap values only; hard-cap-flagged cells are not references.
    usable = ((flags & Flag.REJECTING) == 0) & finite
    spike = _spike_mask(raw_intensity, usable)
    flags[spike] |= Flag.INTENSITY_SPIKE
    rules["intensity_spike"] = {
        "factor": C.SPIKE_FACTOR,
        "min_prev_veh_per_hour": C.SPIKE_MIN_PREV_VEH_PER_HOUR,
        "cells_rejected": int(spike.sum()),}

    rejected = hard | spike
    report = {
        "usable_cells_before_cleaning": n_usable,
        "cells_rejected_by_cleaning": int(rejected.sum()),
        "additional_missingness_fraction": float(
            rejected.sum() / max(n_usable, 1)),
        "rules": rules,
    }

    # TO:DO add more cleaning rules here, e.g. based on road lanes

    C.write_json(report, C.reports_dir(out_dir) / "cleaning_report.json")

    if verbose:
        print(f" rejected {rejected.sum():,} of {n_usable:,} usable cells "f"({rejected.sum() / max(n_usable, 1):.4%})")

    return flags

def _spike_mask(raw_intensity: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Flag cells whose value exceeds SPIKE_FACTOR × the last *accepted* value
    at the same sensor.  Uses a forward-fill reference so that a spike is not
    itself accepted as the baseline for the next step, and gaps are skipped."""
    spike = np.zeros_like(usable)
    last = np.full(raw_intensity.shape[1], np.nan, dtype=raw_intensity.dtype)
    for t in range(raw_intensity.shape[0]):
        cur = raw_intensity[t]
        ok = usable[t]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = cur / last
        s = (ok & np.isfinite(last)
             & (last >= C.SPIKE_MIN_PREV_VEH_PER_HOUR)
             & (ratio > C.SPIKE_FACTOR))
        spike[t] = s
        # A spike is not trusted as the reference for the next timestep.
        last = np.where(ok & ~s & np.isfinite(cur), cur, last)
    return spike
