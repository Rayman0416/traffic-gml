# Fills in missing readings, using only the past and the present (never the future).
#
# We don't fill in how many cars there are, but how busy the road is compared to
# its normal for that time of the week:
#
#     r = (level - profile[time-of-week]) / s
#
# level is log(1 + veh/h) for intensity and plain percent for occupancy.
# r = 0 means a normal day, r > 0 busier than normal. This way a highway
# (3000 veh/h) and a side street (200 veh/h) next to it can still help each other.
#
# For a missing reading we guess r from two things:
#   1. neighbours: what r the connected sensors have right now. We only use
#      neighbours that moved together with this sensor in the training months
#      (correlation >= FILL_MIN_EDGE_CORR).
#   2. own past: the sensor's last real r, which fades back to 0 the longer the
#      gap is (r_last * phi^age), at most FILL_LOOKBACK_STEPS back.
#
# The two guesses are mixed, trusting the more accurate one more, and turned
# back into veh/h with this sensor's own profile. If we have neither, we use the
# profile, then the sensor mean, then the global mean.

import numpy as np
import pandas as pd
from scipy import sparse
from . import config as C
from . import params, splits, timeline
from .config import FillSource

# channel -> (observed key, mask key, log level, upper clip, residual std floor)
CHANNELS = {
    "intensity": ("observed_intensity", "mask", True,
                  C.HARD_CAP_VEH_PER_HOUR, C.FILL_MIN_RESID_STD),
    "occupancy": ("observed_occupancy", "occupancy_mask", False,
                  C.OCCUPANCY_MAX_PERCENT, C.MIN_STD_OCCUPANCY_PERCENT),
}

# Fitted per-channel arrays, saved in params.npz as fill_<channel>_<key>.
PARAM_KEYS = ("profile", "profile_support", "sensor_train_mean",
              "global_train_mean", "resid_std", "phi", "spatial_var",
              "edge_target", "edge_source", "edge_beta", "edge_weight")

CHUNK_STEPS = 2048  # timesteps per chunk, so we don't run out of memory


def context_steps() -> int:
    """How far back in time a fill or an age value can look."""
    return max(C.FILL_LOOKBACK_STEPS, C.AGE_CLIP_STEPS)


def last_observed_index(mask: np.ndarray) -> np.ndarray:
    """For every cell, the last step (up to now) with a real reading. -1 if there is none."""
    T = mask.shape[0]
    steps = np.arange(T, dtype=np.int32)[:, None]
    idx = np.where(mask, steps, np.int32(-1))
    return np.maximum.accumulate(idx, axis=0)


def elapsed_steps(mask: np.ndarray, last: np.ndarray | None = None,
                  clip: int = C.AGE_CLIP_STEPS) -> tuple[np.ndarray, np.ndarray]:
    """Steps since the last real reading, clipped and unclipped.

    If a sensor has never had a reading yet, its age is set to `clip`.
    """
    if last is None:
        last = last_observed_index(mask)
    T = mask.shape[0]
    steps = np.arange(T, dtype=np.int32)[:, None]
    raw_age = steps - last
    never = last < 0
    age = np.where(never, clip, np.minimum(raw_age, clip)).astype(np.int16)
    return age, raw_age


def gap_lengths(mask: np.ndarray) -> np.ndarray:
    """Length of every gap (missing steps in a row), for all sensors."""
    T, N = mask.shape
    miss = np.zeros((T + 2, N), dtype=np.int8)
    miss[1:-1] = ~mask
    d = np.diff(miss, axis=0).T  # one sensor after another, so each start lines up with its end
    starts = np.nonzero(d == 1)[1]
    ends = np.nonzero(d == -1)[1]
    return (ends - starts).astype(np.int64)


def _to_level(values: np.ndarray, log: bool) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return (np.log1p(np.maximum(values, 0)) if log else values).astype(np.float32)


def _profile_level(fp: dict, index: pd.DatetimeIndex) -> np.ndarray:
    return params.profile_matrix(fp["profile"], fp["sensor_train_mean"],
                                 float(fp["global_train_mean"]), index)


def _residuals(level: np.ndarray, m: np.ndarray, prof: np.ndarray,
               std: np.ndarray) -> np.ndarray:
    """r for every real reading (clipped), 0 where the reading is missing."""
    with np.errstate(invalid="ignore"):
        r = np.clip((level - prof) / std[None, :], -C.FILL_RESID_CLIP, C.FILL_RESID_CLIP)
    return np.where(m, r, 0).astype(np.float32)


def _undirected_pairs(edges: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s = edges["source_index"].to_numpy(dtype=np.int64)
    t = edges["target_index"].to_numpy(dtype=np.int64)
    pairs = np.unique(np.stack([np.minimum(s, t), np.maximum(s, t)], axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return pairs[:, 0], pairs[:, 1]


def _spatial_mats(fp: dict, N: int) -> tuple:
    """Sparse [N, N] neighbour matrices: weight * slope, weight only, and 1 per edge (to count neighbours)."""
    ij = (fp["edge_target"], fp["edge_source"])
    w = fp["edge_weight"].astype(np.float32)
    wb = sparse.csr_matrix((w * fp["edge_beta"].astype(np.float32), ij), shape=(N, N))
    ww = sparse.csr_matrix((w, ij), shape=(N, N))
    wc = sparse.csr_matrix((np.ones_like(w), ij), shape=(N, N))
    return wb, ww, wc


def _spatial(r: np.ndarray, mf: np.ndarray, mats: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Guess r for every cell from the real neighbour readings at the same step.
    Returns the guess, and where we had enough neighbours to make one."""
    wb, ww, wc = mats
    num = (wb @ r.T).T
    den = (ww @ mf.T).T
    cnt = (wc @ mf.T).T
    ok = (cnt >= C.FILL_MIN_NEIGHBOURS) & (den > 0)
    est = np.where(ok, num / np.where(ok, den, 1), 0).astype(np.float32)
    return est, ok


def fit_fill_params(values: np.ndarray, mask: np.ndarray, index: pd.DatetimeIndex,
                    edges: pd.DataFrame, log: bool, min_std: float,
                    min_corr: float = C.FILL_MIN_EDGE_CORR) -> dict:
    """
    Learn the fill settings for one channel, from real readings in the training months only:
    the weekly profile, the spread s, how fast r fades (phi), which neighbours to trust
    and how much, and how accurate the neighbour guess is per sensor.
    """
    train = splits.train_mask(index)
    tidx = index[train]
    level = _to_level(values[train], log)
    m = mask[train] & np.isfinite(level)
    N = m.shape[1]

    prof = params.fit_profile(level, m, tidx)
    P = _profile_level(prof, tidx)
    with np.errstate(invalid="ignore"):
        std = np.nanstd(np.where(m, level - P, np.nan), axis=0)
    std = np.where(np.isfinite(std), np.maximum(std, min_std), 1.0).astype(np.float32)
    r = _residuals(level, m, P, std)
    del level, P
    mf = m.astype(np.float32)

    # phi: how much r at one step looks like r at the next step (correlation).
    # Only pairs where both readings are real count; r is 0 otherwise, so the
    # products below drop out by themselves.
    n_lag = (mf[1:] * mf[:-1]).sum(axis=0)
    sxy = (r[1:] * r[:-1]).sum(axis=0, dtype=np.float64)
    sxx = (r[:-1] ** 2 * mf[1:]).sum(axis=0, dtype=np.float64)
    syy = (r[1:] ** 2 * mf[:-1]).sum(axis=0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = sxy / np.sqrt(sxx * syy)
    phi = np.where((n_lag >= C.FILL_MIN_EDGE_OVERLAP) & np.isfinite(phi),
                   np.clip(phi, 0.0, 0.99), 0.0).astype(np.float32)

    # For every edge: how well do the two sensors move together? Only steps where
    # both are real count. Done 1024 edges at a time to save memory.
    u, v = _undirected_pairs(edges)
    n = np.zeros(len(u)); cov = np.zeros(len(u)); va = np.zeros(len(u)); vb = np.zeros(len(u))
    for k in range(0, len(u), 1024):
        uu, vv = u[k:k + 1024], v[k:k + 1024]
        a, b, w = r[:, uu], r[:, vv], mf[:, uu] * mf[:, vv]
        nk = w.sum(axis=0, dtype=np.float64)
        sa = (a * w).sum(axis=0, dtype=np.float64)
        sb = (b * w).sum(axis=0, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ma, mb = sa / nk, sb / nk
            cov[k:k + 1024] = (a * b * w).sum(axis=0, dtype=np.float64) / nk - ma * mb
            va[k:k + 1024] = (a * a * w).sum(axis=0, dtype=np.float64) / nk - ma ** 2
            vb[k:k + 1024] = (b * b * w).sum(axis=0, dtype=np.float64) / nk - mb ** 2
        n[k:k + 1024] = nk
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = cov / np.sqrt(va * vb)
    keep = (n >= C.FILL_MIN_EDGE_OVERLAP) & (va > 0) & (vb > 0) & (rho >= min_corr)
    u, v, rho, cov, va, vb = u[keep], v[keep], rho[keep], cov[keep], va[keep], vb[keep]
    # Keep each edge in both directions: `target` gets guessed from `source`.
    fp = {
        **{k: prof[k] for k in ("profile", "profile_support",
                                "sensor_train_mean", "global_train_mean")},
        "resid_std": std,
        "phi": phi,
        "edge_target": np.concatenate([u, v]).astype(np.int32),
        "edge_source": np.concatenate([v, u]).astype(np.int32),
        "edge_beta": np.concatenate([cov / vb, cov / va]).astype(np.float32),
        "edge_weight": np.concatenate([rho ** 2, rho ** 2]).astype(np.float32),
    }

    # How wrong is the neighbour guess on real training readings? If it is no
    # better than just using the profile (error >= 1), that sensor never uses neighbours.
    mats = _spatial_mats(fp, N)
    se = np.zeros(N); cnt = np.zeros(N)
    for t0 in range(0, len(tidx), CHUNK_STEPS):
        sl = slice(t0, t0 + CHUNK_STEPS)
        est, ok = _spatial(r[sl], mf[sl], mats)
        use = ok & m[sl]
        se += np.where(use, (r[sl] - est) ** 2, 0).sum(axis=0, dtype=np.float64)
        cnt += use.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        svar = se / cnt
    usable = (cnt >= C.FILL_MIN_EDGE_OVERLAP) & (svar < 1.0)
    fp["spatial_var"] = np.where(usable, svar, np.inf).astype(np.float32)

    fp["info"] = {
        "n_candidate_edges": int(len(keep)),
        "n_edges_kept": int(keep.sum()),
        "min_edge_corr": float(min_corr),
        "median_phi": float(np.median(phi)),
        "n_sensors_with_spatial": int(usable.sum()),
        "median_spatial_var": float(np.median(svar[usable])) if usable.any() else None,
        "median_resid_std": float(np.median(std)),
    }
    return fp


def channel_params(p: dict, channel: str) -> dict:
    """Get the fill settings of one channel out of a loaded params.npz."""
    prefix = f"fill_{channel}_"
    fp = {k[len(prefix):]: v for k, v in p.items() if k.startswith(prefix)}
    missing = [k for k in PARAM_KEYS if k not in fp]
    if missing:
        raise KeyError(f"params.npz has no fill parameters {missing} for {channel} "
                       "- re-run the `params` stage")
    return fp


def fill_channel(values: np.ndarray, mask: np.ndarray, index: pd.DatetimeIndex,
                 fp: dict, log: bool, upper: float,
                 use_spatial: bool = True, use_own: bool = True) -> dict:
    """
    Fill every missing cell of one channel, without looking at the future.
    Real readings stay as they are. Returns:
      filled       [T, N]  the values with the gaps filled
      fill_source  [T, N]  how each cell was filled (FillSource codes)
      age          [T, N]  steps since the last real reading, max AGE_CLIP_STEPS
    """
    T, N = values.shape
    filled = np.empty((T, N), dtype=np.float32)
    source = np.empty((T, N), dtype=np.uint8)
    age = np.empty((T, N), dtype=np.int16)

    std, phi = fp["resid_std"], fp["phi"]
    svar = fp["spatial_var"]
    mats = _spatial_mats(fp, N)
    has_spatial = np.isfinite(svar)[None, :]
    supported = fp["profile_support"] >= C.PROFILE_MIN_SUPPORT
    no_train = (fp["profile_support"].sum(axis=0) == 0)[None, :]
    tow = timeline.time_of_week(index)
    cols = np.arange(N)[None, :]

    # Remembered from one chunk to the next: each sensor's last real step and its r.
    last_t = np.full(N, -1, dtype=np.int64)
    last_r = np.zeros(N, dtype=np.float32)

    for t0 in range(0, T, CHUNK_STEPS):
        t1 = min(T, t0 + CHUNK_STEPS)
        sl = slice(t0, t1)
        level = _to_level(values[sl], log)
        m = mask[sl] & np.isfinite(level)
        P = _profile_level(fp, index[sl])
        r = _residuals(level, m, P, std)

        steps = np.arange(t0, t1, dtype=np.int64)[:, None]
        idx = np.maximum.accumulate(
            np.vstack([last_t[None, :], np.where(m, steps, -1)]), axis=0)[1:]
        never = idx < 0
        raw_age = steps - idx
        r_last = np.where(idx >= t0, r[np.clip(idx - t0, 0, None), cols], last_r[None, :])

        own_ok = (use_own & ~never & (raw_age <= C.FILL_LOOKBACK_STEPS)
                  & (phi > 0)[None, :])
        with np.errstate(under="ignore"):
            decay = np.where(own_ok, phi[None, :] ** np.where(own_ok, raw_age, 0), 0)
        own = (decay * r_last).astype(np.float32)
        own_prec = np.where(own_ok, 1.0 / np.maximum(1.0 - decay ** 2, 1e-6), 0)

        if use_spatial:
            sp, sp_ok = _spatial(r, m.astype(np.float32), mats)
            sp_ok &= has_spatial
        else:
            sp, sp_ok = np.zeros_like(r), np.zeros_like(m)
        sp_prec = np.where(sp_ok, 1.0 / np.where(sp_ok, svar[None, :], 1), 0)

        tot = own_prec + sp_prec
        rhat = np.where(tot > 0, (own_prec * own + sp_prec * sp) / np.where(tot > 0, tot, 1), 0)
        x = P + std[None, :] * rhat
        if log:
            x = np.expm1(x)
        x = np.clip(x, 0, upper)

        filled[sl] = np.where(m, values[sl], x)
        base = np.where(supported[tow[sl]], FillSource.PROFILE, FillSource.SENSOR_MEAN)
        base = np.where(no_train, FillSource.GLOBAL_MEAN, base)
        src = np.where(own_ok & sp_ok, FillSource.SPATIOTEMPORAL,
                       np.where(sp_ok, FillSource.SPATIAL,
                                np.where(own_ok, FillSource.OWN_DECAY, base)))
        source[sl] = np.where(m, FillSource.OBSERVED, src)
        age[sl] = np.where(never, C.AGE_CLIP_STEPS,
                           np.minimum(raw_age, C.AGE_CLIP_STEPS))

        last_t, last_r = idx[-1], r_last[-1]

    if not np.isfinite(filled).all():
        raise ValueError("fill produced non-finite values")
    return {"filled": filled, "fill_source": source, "age": age}


def source_fractions(source: np.ndarray) -> dict:
    counts = np.bincount(source.ravel(), minlength=len(FillSource.NAMES))
    return {name: float(counts[code] / source.size)
            for code, name in FillSource.NAMES.items()}


# --------------------------------------------------------------------------
# Testing the fill: hide real validation readings, fill them, and see how close we get
# --------------------------------------------------------------------------

AGE_BUCKETS = ((1, 1), (2, 4), (5, 16), (17, 96), (97, 10 ** 9))


def _scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    e = pred.astype(np.float64) - truth
    return {"n": int(e.size),
            "mae": float(np.abs(e).mean()) if e.size else None,
            "rmse": float(np.sqrt((e ** 2).mean())) if e.size else None}


def evaluate(out_dir=C.OUT_DIR, n_gaps=20000, n_outages=40, outage_steps=8,
             corr_grid=(0.3, 0.5, 0.7), seed=C.RANDOM_SEED, verbose=True) -> dict:
    """
    Compare fill methods on real validation readings that we hide on purpose.
    The hidden gaps have the same lengths as real gaps in training, plus some
    whole-district outages. Errors are in veh/h, split by gap age and by how busy
    the sensor normally is (quiet street vs. highway).
    """
    from . import graphs, sensor_selection
    from . import observed as observed_mod

    obs = observed_mod.load(out_dir)
    p = params.load(out_dir)
    index = obs["timestamps"]
    values, mask = obs["observed_intensity"], obs["mask"]
    N = values.shape[1]
    rng = np.random.default_rng(seed)

    val_rows = np.nonzero(obs["split_code"] == splits.SPLIT_CODE["val"])[0]
    v0, v1 = int(val_rows[0]), int(val_rows[-1]) + 1
    lo = max(0, v0 - context_steps())
    sub_v, sub_m, sub_idx = values[lo:v1], mask[lo:v1], index[lo:v1]
    first = v0 - lo

    lengths = np.minimum(gap_lengths(mask[splits.train_mask(index)]), C.STEPS_PER_WEEK)
    hide = np.zeros(sub_m.shape, dtype=bool)
    for s, c, n in zip(rng.integers(first, v1 - lo, n_gaps), rng.integers(0, N, n_gaps),
                       rng.choice(lengths, n_gaps)):
        hide[s:s + n, c] = True
    outage = np.zeros_like(hide)
    district = sensor_selection.included(out_dir)["district"].to_numpy()
    for d, s in zip(rng.choice(np.unique(district), n_outages),
                    rng.integers(first, v1 - lo - outage_steps, n_outages)):
        outage[s:s + outage_steps, district == d] = True
    hide = (hide | outage) & sub_m
    hide[:first] = False
    outage &= hide
    visible = sub_m & ~hide
    truth = sub_v[hide].astype(np.float64)

    fp = channel_params(p, "intensity")
    up = C.HARD_CAP_VEH_PER_HOUR
    blend = fill_channel(sub_v, visible, sub_idx, fp, True, up)
    age = blend["age"][hide]

    profile = params.profile_matrix(p["profile"], p["sensor_train_mean"],
                                    float(p["global_train_mean"]), sub_idx)
    last = last_observed_index(visible)
    _, raw_age = elapsed_steps(visible, last)
    ffill_ok = (last >= 0) & (raw_age <= 4)
    ffill = np.where(ffill_ok, sub_v[np.maximum(last, 0), np.arange(N)[None, :]], profile)

    preds = {
        "train_mean": np.broadcast_to(p["scaler_mean"][None, :], sub_v.shape)[hide],
        "profile": profile[hide],
        "ffill_1h_then_profile": ffill[hide],
        "own_decay_only": fill_channel(sub_v, visible, sub_idx, fp, True, up,
                                       use_spatial=False)["filled"][hide],
        "spatial_only": fill_channel(sub_v, visible, sub_idx, fp, True, up,
                                     use_own=False)["filled"][hide],
        "graph_ar": blend["filled"][hide],
    }

    level = np.broadcast_to(p["scaler_mean"][None, :], hide.shape)[hide]
    q = np.quantile(p["scaler_mean"], [0.2, 0.4, 0.6, 0.8])
    quint = np.searchsorted(q, level)
    is_out = outage[hide]

    methods = {}
    for name, pr in preds.items():
        methods[name] = {
            "overall": _scores(pr, truth),
            "random_gaps": _scores(pr[~is_out], truth[~is_out]),
            "district_outages": _scores(pr[is_out], truth[is_out]),
            "by_age_steps": {f"{a}-{b}": _scores(pr[(age >= a) & (age <= b)],
                                                 truth[(age >= a) & (age <= b)])
                             for a, b in AGE_BUCKETS},
            "by_level_quintile": {f"q{k + 1}": _scores(pr[quint == k], truth[quint == k])
                                  for k in range(5)},
        }

    # Try other neighbour correlation thresholds (learns the fill settings again for each).
    grid = {}
    if corr_grid:
        edges = graphs.load_full_edges(out_dir)
        for c in corr_grid:
            fpc = fit_fill_params(values, mask, index, edges, log=True,
                                  min_std=C.FILL_MIN_RESID_STD, min_corr=c)
            pr = fill_channel(sub_v, visible, sub_idx, fpc, True, up)["filled"][hide]
            grid[str(c)] = {"n_edges_kept": fpc["info"]["n_edges_kept"],
                            "overall": _scores(pr, truth),
                            "by_level_quintile": {
                                f"q{k + 1}": _scores(pr[quint == k], truth[quint == k])
                                for k in range(5)}}

    hidden_src = blend["fill_source"][hide]
    summary = {
        "n_hidden_cells": int(hide.sum()),
        "n_outage_cells": int(is_out.sum()),
        "level_quintile_edges_veh_per_hour": q.tolist(),
        "graph_ar_source_fractions": source_fractions(hidden_src),
        "methods": methods,
        "edge_corr_grid": grid,
    }
    C.write_json(summary, C.reports_dir(out_dir) / "fill_eval.json")

    if verbose:
        print(f"  hid {summary['n_hidden_cells']:,} val cells "
              f"({summary['n_outage_cells']:,} in outages)")
        for name, m in methods.items():
            print(f"  {name}: mae {m['overall']['mae']:.1f}")
        for c, g in grid.items():
            print(f"  min corr {c}: mae {g['overall']['mae']:.1f}")
        print(f"  wrote {C.reports_dir(out_dir) / 'fill_eval.json'}")
    return summary
