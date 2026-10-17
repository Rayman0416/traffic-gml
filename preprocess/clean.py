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

    rejected = hard
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
