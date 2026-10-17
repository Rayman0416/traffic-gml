# Stage 8: Pipeline checks

import numpy as np
import pandas as pd
from . import config as C
from . import features, graphs, observed, params, sensor_selection, splits, timeline, windows

def _result(name: str, problems: list, **extra) -> dict:
    return {"name": name, "passed": not problems, "problems": problems, **extra}


def _history_length(df: pd.DataFrame) -> int:
    """The input length the window index was built with (it may be a CLI
    override of ``config.INPUT_LENGTH``)."""
    span = df["origin_index"] - df["input_start_index"] + 1
    if span.nunique() != 1:
        raise ValueError("windows.csv mixes input lengths")
    return int(span.iloc[0])


def check_sensor_order(out_dir=C.OUT_DIR) -> dict:
    """Sensor order is identical in the selection table, the arrays and the graph."""
    reg = sensor_selection.included(out_dir)
    nodes = pd.read_csv(out_dir / "graph" / "nodes.csv")
    reg_ids = reg["sensor_id"].to_numpy()
    problems = []

    sources = {
        "observed.npz": observed.load(out_dir)["sensor_ids"],
        "params.npz": params.load(out_dir)["sensor_ids"],
        "features.npz": np.load(out_dir / "features.npz")["sensor_ids"],
        "graph nodes.csv": nodes["sensor_id"].to_numpy(),
    }
    for name, ids in sources.items():
        if not np.array_equal(reg_ids, ids):
            problems.append(f"{name} disagrees with the sensor selection")

    for name, idx in (("graph node", nodes["array_index"]),
                      ("sensor_selection", reg["array_index"])):
        if not np.array_equal(idx.to_numpy(), np.arange(len(idx))):
            problems.append(f"{name} array_index is not 0..N-1 in row order")

    # Every edge endpoint resolves to the right array position.
    edges = graphs.load_full_edges(out_dir)
    pos = pd.Index(reg_ids)
    for end in ("source", "target"):
        if len(edges) and not np.array_equal(pos.get_indexer(edges[end]),
                                             edges[f"{end}_index"].to_numpy()):
            problems.append(f"edge {end}_index does not match sensor order")

    # District subgraph local indices map back correctly.
    for d in sorted(reg["district"].dropna().astype(int).unique()):
        dn = graphs.load_district_nodes(int(d), out_dir)
        want = np.flatnonzero(reg["district"].to_numpy() == d)
        if not np.array_equal(dn["global_index"].to_numpy(), want):
            problems.append(f"district {d} global_index mismatch")
        if not np.array_equal(reg_ids[want], dn["sensor_id"].to_numpy()):
            problems.append(f"district {d} sensor ids mismatch")

    return _result("sensor_order_matches_graph_order", problems,
                   n_sensors=int(len(reg_ids)))


def check_horizons(out_dir=C.OUT_DIR) -> dict:
    """Target timestamps are exactly +15, +30, +45, +60 minutes from the origin,
    and the input window is L steps ending at the origin."""
    df = windows.load(out_dir)
    origin = df["origin_timestamp"]
    problems = []

    def minutes(col):
        return (df[col] - origin).dt.total_seconds() / 60.0

    for k, m in enumerate(C.HORIZON_MINUTES):
        bad = int((minutes(f"target_timestamp_h{k + 1}") != m).sum())
        if bad:
            problems.append(f"h{k + 1}: {bad} windows are not +{m}min")

    want = -(_history_length(df) - 1) * 15
    bad = int((minutes("input_start_timestamp") != want).sum())
    if bad:
        problems.append(f"{bad} windows have an input span != {-want}min")

    return _result("target_timestamps_are_15_30_45_60_ahead", problems,
                   n_windows=int(len(df)))


def check_split_boundaries(out_dir=C.OUT_DIR) -> dict:
    """All four targets of a window lie in the window's own split."""
    df = windows.load(out_dir)
    code = observed.load(out_dir)["split_code"]
    want = df["split"].map(splits.SPLIT_CODE).to_numpy()
    problems = []

    for k in range(len(C.HORIZONS)):
        bad = int((code[df[f"target_index_h{k + 1}"].to_numpy()] != want).sum())
        if bad:
            problems.append(f"h{k + 1}: {bad} targets outside the window's split")

    return _result("targets_do_not_cross_split_boundaries", problems)


def check_targets_genuine(out_dir=C.OUT_DIR) -> dict:
    """Only genuine observations can be targets: the observed layer is NaN
    wherever the mask is False, and the window target counts follow the mask."""
    obs = observed.load(out_dir)
    wa = windows.load_arrays(out_dir)
    o, m = obs["observed_intensity"], obs["mask"]
    problems = []

    if np.isfinite(o[~m]).any():
        problems.append("observed_intensity is finite where mask is False")
    if not np.isfinite(o[m]).all():
        problems.append("observed_intensity is non-finite where mask is True")
    n_valid = m[wa["target_index"]].sum(axis=(1, 2))
    if not np.array_equal(n_valid, wa["n_valid_targets"]):
        problems.append("n_valid_targets disagrees with the observed mask")

    return _result("only_genuine_observations_are_targets", problems)


def check_timeline_complete(out_dir=C.OUT_DIR) -> dict:
    """The timeline was never compacted: consecutive rows are 15 min apart."""
    obs = observed.load(out_dir)
    index = obs["timestamps"]
    problems = []
    try:
        timeline.check_regular(index)
    except ValueError as e:
        problems.append(str(e))
    expected = timeline.build_index()
    if len(index) != len(expected) or not (index == expected).all():
        problems.append("timestamps differ from the canonical timeline")
    # Outages must be represented as missing, not as removed rows.
    if not (~obs["mask"]).any():
        problems.append("no missing cells at all - suspicious")
    return _result("timeline_complete_and_never_stitched", problems,
                   n_steps=int(len(index)))


def check_no_leakage(out_dir=C.OUT_DIR) -> dict:
    """Learned parameters depend on training cells only.

    Refits each parameter with validation and test values corrupted and asserts
    it does not move, then compares the clean refit with the saved parameters.
    """
    obs = observed.load(out_dir)
    saved = params.load(out_dir)
    index = obs["timestamps"]
    rng = np.random.default_rng(C.RANDOM_SEED)
    train = splits.train_mask(index)
    problems = []

    def corrupt(x, top):
        x = x.copy()
        x[~train] = rng.uniform(0, top, size=x[~train].shape).astype(np.float32)
        return x

    # (label, fit fn, values, mask, corruption range, {fitted key: saved key})
    fits = (
        ("scaler", params.fit_scaler, "observed_intensity", "mask", 1e5,
         {"mean": "scaler_mean", "std": "scaler_std"}),
        ("occupancy scaler",
         lambda o, m, i: params.fit_scaler(o, m, i, min_std=C.MIN_STD_OCCUPANCY_PERCENT),
         "observed_occupancy",
         "occupancy_mask", 100.0,
         {"mean": "occupancy_scaler_mean", "std": "occupancy_scaler_std"}),
        ("profile", params.fit_profile, "observed_intensity", "mask", 1e5,
         {"profile": "profile"}),
    )
    for label, fit, vk, mk, top, keys in fits:
        ref = fit(obs[vk], obs[mk], index)
        alt = fit(corrupt(obs[vk], top), obs[mk], index)
        for k, sk in keys.items():
            if not np.allclose(ref[k], alt[k], equal_nan=True):
                problems.append(f"{label} {k} changed when non-training values "
                                "were corrupted")
            if not np.allclose(ref[k], saved[sk], equal_nan=True):
                problems.append(f"saved {label} {k} differs from a "
                                "training-only refit")

    return _result("learned_parameters_use_training_data_only", problems)


def check_feature_definitions(out_dir=C.OUT_DIR) -> dict:
    """features.npz is exactly the documented function of the observed layer.

    Rebuilds every feature array from ``observed.npz`` and the saved training
    scalers and requires a bit-identical match; checks that masked-out cells
    hold the missing value and nothing else, and that the history length
    agrees with the window index.
    """
    obs = observed.load(out_dir)
    p = params.load(out_dir)
    f = features.load(out_dir)
    problems = []

    for k, v in features.build_arrays(obs, p, obs["timestamps"]).items():
        if not np.array_equal(v, f[k]):
            problems.append(f"{k} differs from a rebuild from observed + params")
    for ch in features.SENSOR_CHANNELS:
        v, m = f[ch], f[f"{ch}_mask"]
        if not np.isfinite(v).all():
            problems.append(f"{ch} contains non-finite values")
        if (v[~m] != np.float32(C.FEATURE_MISSING_VALUE)).any():
            problems.append(f"{ch} holds something other than the missing "
                            "value where its mask is False")
    if f["time_feature_names"] != list(features.TIME_FEATURES):
        problems.append(f"time features are {f['time_feature_names']}, "
                        f"expected {list(features.TIME_FEATURES)}")
    L = _history_length(windows.load(out_dir))
    if f["history_length"] != L:
        problems.append(f"features history_length {f['history_length']} != "
                        f"window input length {L}")

    return _result("features_match_their_definition", problems,
                   history_length=f["history_length"])


def check_feature_causality(out_dir=C.OUT_DIR, n_probe=24, seed=C.RANDOM_SEED,
                            future_steps=C.STEPS_PER_WEEK) -> dict:
    """An origin's features never read anything after the origin.

    For each probe origin ``t``, builds the features once on data truncated at
    ``t`` and once on data whose next ``future_steps`` are garbage with both
    masks flipped, and requires the gathered ``L``-step history to be
    bit-identical.  Also asserts the gathered rows end exactly at ``t``.
    """
    obs = observed.load(out_dir)
    p = params.load(out_dir)
    f = features.load(out_dir)
    index = obs["timestamps"]
    L = f["history_length"]
    T = len(index)
    rng = np.random.default_rng(seed)
    origins = sorted(int(o) for o in rng.choice(
        np.arange(L - 1, T - max(C.HORIZONS)), size=min(n_probe, T - L), replace=False))
    keys = ("observed_intensity", "mask", "observed_occupancy", "occupancy_mask")
    problems = []

    for t in origins:
        lo, hi = t - L + 1, min(T, t + 1 + future_steps)
        cut = {k: obs[k][lo: t + 1] for k in keys}
        bad = {k: obs[k][lo: hi].copy() for k in keys}
        for k, mk, top in (("observed_intensity", "mask", 9000.0),
                           ("observed_occupancy", "occupancy_mask", 100.0)):
            bad[k][L:] = rng.uniform(0, top, size=bad[k][L:].shape)
            bad[mk][L:] = ~bad[mk][L:]
        ref = features.build_arrays(cut, p, index[lo: t + 1])
        alt = features.build_arrays(bad, p, index[lo: hi])
        changed = [k for k in ref if not np.array_equal(ref[k], alt[k][:L])]
        if changed:
            problems.append(f"origin {t} ({index[t]}): {changed[0]} changed "
                            "when the future was perturbed")

        w = features.window_features(f, np.array([t]))
        if w["history_index"].shape[1] != L or w["history_index"][0, -1] != t:
            problems.append(f"origin {t}: gathered history does not end at the origin")
        stale = [k for k in ("intensity", "intensity_mask", "occupancy", "occupancy_mask")
                 if not np.array_equal(w[k][0], ref[k])]
        if stale:
            problems.append(f"origin {t}: stored {stale[0]} history differs "
                            "from the truncated rebuild")

    return _result("features_cannot_read_after_origin", problems,
                   n_origins_probed=len(origins))


ALL_CHECKS = (
    check_sensor_order,
    check_horizons,
    check_split_boundaries,
    check_targets_genuine,
    check_timeline_complete,
    check_no_leakage,
    check_feature_definitions,
    check_feature_causality,
)


def run(out_dir=C.OUT_DIR, strict=True, verbose=True) -> dict:
    results = []
    for fn in ALL_CHECKS:
        if verbose:
            print(f"  {fn.__name__}...", end=" ", flush=True)
        rec = fn(out_dir)
        results.append(rec)
        if verbose:
            print("ok" if rec["passed"] else "FAILED")
            for p in rec["problems"]:
                print(f"      - {p}")

    # An empty `problems` list is the normal case, so it is dropped: on a green
    # run the key is absent, and when it is present it matters.
    for r in results:
        if not r["problems"]:
            del r["problems"]
    failed = [r["name"] for r in results if not r["passed"]]
    report = {
        "n_checks": len(results),
        "n_failed": len(failed),
        "all_passed": not failed,
        "checks": results,
    }
    C.write_json(report, C.reports_dir(out_dir) / "checks_report.json")
    if strict and failed:
        raise AssertionError(f"{len(failed)} pipeline check(s) failed: {failed}")
    return report
