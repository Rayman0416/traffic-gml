"""
Quality report for the GO-MO Madrid 2024 traffic intensity dataset.
"""

import json
import numpy as np
import pandas as pd
from . import config as C
from . import observed, sensor_selection, splits, windows


def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def gap_statistics(out_dir=C.OUT_DIR) -> dict:
    """Distribution of consecutive-gap lengths per sensor, over the full year."""
    obs = observed.load(out_dir)
    mask = obs["mask"]
    T, N = mask.shape

    # Per-sensor gap runs: the diff of the missing indicator gives every run's
    # start and end, so one pass per sensor collects all gap lengths.
    longest = np.zeros(N, dtype=np.int32)
    pieces = []
    for j in range(N):
        col = (~mask[:, j]).view(np.int8)
        if not col.any():
            continue
        d = np.diff(np.concatenate(([0], col, [0])).astype(np.int8))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        runs = (ends - starts).astype(np.int32)
        longest[j] = runs.max()
        pieces.append(runs)
    gaps = np.concatenate(pieces) if pieces else np.array([], dtype=np.int32)

    hist = {}
    bucket_edges = [1, 2, 4, 12, 96, 672]
    for e in bucket_edges:
        hist[e] = int((gaps <= e).sum())
    return {
        "n_gaps": int(gaps.size),
        "gap_steps": {
            "mean": float(gaps.mean()) if gaps.size else None,
            "median": float(np.median(gaps)) if gaps.size else None,
            "p99": float(np.percentile(gaps, 99)) if gaps.size else None,
            "max": int(gaps.max()) if gaps.size else None,
        },
        "cumulative_gaps_at_most_n_steps": {
            f"{e}_steps_{e * 15}min": hist[e] for e in bucket_edges},
        "longest_gap_per_sensor": {
            "median": float(np.median(longest)),
            "max": int(longest.max()),
        },
    }


def _input_length(out_dir) -> int:
    df = windows.load(out_dir)
    return int((df["origin_index"] - df["input_start_index"]).iloc[0] + 1)


def build(out_dir=C.OUT_DIR, verbose=True) -> dict:
    reg = sensor_selection.load(out_dir)
    inc = reg[reg["included"]]

    rdir = C.reports_dir(out_dir)
    
    stages = {
        "audit": _read(rdir / "audit.json"),
        "sensor_selection": _read(rdir / "sensor_selection_summary.json"),
        "graphs": _read(rdir / "graph_summary.json"),
        "observed": _read(rdir / "observed_summary.json"),
        "cleaning": _read(rdir / "cleaning_report.json"),
        "params": _read(rdir / "params_summary.json"),
        "fill": _read(rdir / "fill_summary.json"),
        "features": _read(rdir / "features_summary.json"),
        "windows": _read(rdir / "windows_summary.json"),
        "checks": _read(rdir / "checks_report.json"),
    }
    chk = stages["checks"] or {}
    sensors = {
        "n_included": int(len(inc)),
        "n_excluded": int((~reg["included"]).sum()),
        "exclusion_reasons": reg.loc[~reg["included"], "exclusion_reason"]
            .value_counts().to_dict(),
        "districts": {str(int(k)): int(v) for k, v in
                      inc["district"].value_counts().sort_index().items()},
    }

    report = {
        "dataset": "GO-MO Madrid 2024 traffic intensity",
        "unit": C.INTENSITY_UNIT,
        "generated": pd.Timestamp.utcnow().isoformat(),
        "timeline": {
            "start": C.PERIOD_START.isoformat(),
            "end_exclusive": C.PERIOD_END.isoformat(),
            "freq": C.FREQ,
            "interval_label": C.INTERVAL_LABEL,
        },
        "task": {
            "input_length": _input_length(out_dir),
            "input_span_minutes": _input_length(out_dir) * 15,
            "horizons_minutes": list(C.HORIZON_MINUTES),
        },
        "splits": splits.describe(),
        # Gap statistics are the one thing no stage computes, so they live here.
        "gaps": gap_statistics(out_dir),
        "checks": {k: chk.get(k) for k in
                   ("n_checks", "n_failed", "all_passed")} if chk else None,
        # The per-stage numbers are NOT copied in: each stage already writes its
        # own file, and re-embedding all nine made this file 793 lines of
        # duplicates.  quality_report.md is the readable roll-up.
        "stage_reports": sorted(
            f.name for f in rdir.glob("*.json")
            if f.name != "quality_report.json"),
    }
    C.write_json(report, rdir / "quality_report.json")
    _write_markdown({**report, "stages": stages,
                     "sensors": sensors}, rdir / "quality_report.md")
    if verbose:
        print(f"  wrote {rdir / 'quality_report.json'} and .md")
    return report


def _write_markdown(r: dict, path) -> None:
    s = r["stages"]
    obs = s.get("observed") or {}
    win = s.get("windows") or {}
    fil = s.get("fill") or {}
    chk = s.get("checks") or {}
    reg = s.get("sensor_selection") or {}
    aud = s.get("audit") or {}

    L: list[str] = []
    L.append("# GO-MO Madrid forecasting dataset - quality report\n")
    L.append(f"Generated {r['generated']}  ")
    L.append(f"Unit: **{r['unit']}** throughout.\n")

    L.append("## Timeline\n")
    t = r["timeline"]
    L.append(f"- {t['start']} to {t['end_exclusive']} at {t['freq']}, "
             f"timestamps label the **{t['interval_label']}** of each interval")
    dst = (aud.get("timestamps") or {}).get("dst_resolution") or {}
    if dst:
        L.append(f"- Stamps nonexistent in Madrid local time: "
                 f"{dst.get('stamps_nonexistent_in_local_time')}; "
                 f"ambiguous: {dst.get('stamps_ambiguous_in_local_time')} "
                 "(the signature of a DST-free provider grid)")
    L.append("")

    L.append("## Sensors\n")
    L.append(f"- Included: **{r['sensors']['n_included']}**, "
             f"excluded: {r['sensors']['n_excluded']}")
    L.append("- Exclusion reasons:")
    for k, v in r["sensors"]["exclusion_reasons"].items():
        L.append(f"  - `{k}`: {v}")
    L.append(f"- Districts: {r['sensors']['districts']}")
    if reg.get("districts_dropped_as_too_small"):
        L.append(f"- Districts dropped as too small "
                 f"(< {reg.get('min_district_sensors')} sensors): "
                 f"{reg['districts_dropped_as_too_small']}")
    L.append("")

    L.append("## Observation coverage\n")
    m = obs.get("mask", {})
    L.append(f"- Genuine observations overall: **{m.get('observed_fraction_overall', 0):.2%}**")
    cl = s.get("cleaning") or {}
    L.append(f"- Removed by cleaning: "
             f"{cl.get('cells_rejected_by_cleaning', 0):,} cells")
    L.append("")
    L.append("| Split | Steps | Genuine observations |")
    L.append("|---|---|---|")
    for name, v in (m.get("per_split") or {}).items():
        L.append(f"| {name} | {v['n_steps']:,} | {v['observed_fraction']:.2%} |")
    L.append("")

    L.append("## Quality flags (cells)\n")
    L.append("| Flag | Cells |")
    L.append("|---|---|")
    for k, v in (obs.get("flag_counts") or {}).items():
        L.append(f"| `{k}` | {v:,} |")
    L.append("")

    L.append("## Gaps\n")
    g = r["gaps"]
    L.append(f"- {g['n_gaps']:,} gaps; median {g['gap_steps']['median']} steps, "
             f"max {g['gap_steps']['max']} steps")
    L.append(f"- Longest gap per sensor: median "
             f"{g['longest_gap_per_sensor']['median']}, "
             f"max {g['longest_gap_per_sensor']['max']} steps")
    L.append("")

    L.append("## Model inputs (causal fill)\n")
    L.append("| Source | Share |")
    L.append("|---|---|")
    for k, v in (fil.get("fill_source_fractions") or {}).items():
        L.append(f"| `{k}` | {v:.2%} |")
    L.append("")

    feat = s.get("features") or {}
    if feat:
        L.append("## Input features\n")
        L.append(f"- History: **{feat['shape']['history_length']} steps** "
                 "(including the origin) of scaled intensity and occupancy, each "
                 "with its own validity mask, plus tod/dow sine and cosine")
        L.append("")
        L.append("| Split | Intensity valid | Occupancy valid | Both valid "
                 "| Intensity z mean / std | Occupancy z mean / std |")
        L.append("|---|---|---|---|---|---|")
        for name, v in (feat.get("per_split") or {}).items():
            i, o = v["intensity"], v["occupancy"]
            L.append(f"| {name} | {i['valid_fraction']:.2%} | {o['valid_fraction']:.2%} "
                     f"| {v['both_valid_fraction']:.2%} "
                     f"| {i['scaled_mean']:.3f} / {i['scaled_std']:.3f} "
                     f"| {o['scaled_mean']:.3f} / {o['scaled_std']:.3f} |")
        L.append("")

    L.append("## Forecasting windows\n")
    L.append("| Split | Windows | Eligible (routine) | Genuine targets |")
    L.append("|---|---|---|---|")
    for name, v in (win.get("per_split") or {}).items():
        L.append(f"| {name} | {v['n_windows']:,} | {v['n_eligible_routine']:,} "
                 f"| {v['n_valid_targets_total']:,} |")
    L.append("")
    L.append(f"- Dropped for straddling a split boundary: "
             f"{win.get('n_dropped_straddling_split_boundary')}")
    L.append(f"- Eligibility: input availability >= "
             f"{C.MIN_INPUT_AVAILABILITY}, worst district >= "
             f"{C.MIN_DISTRICT_INPUT_AVAILABILITY}; "
             "depends on past input availability only")
    L.append("")

    L.append("## Checks\n")
    L.append(f"**{chk.get('n_checks', 0) - chk.get('n_failed', 0)}"
             f"/{chk.get('n_checks', 0)} passed**\n")
    L.append("| Check | Result |")
    L.append("|---|---|")
    for c in chk.get("checks", []):
        L.append(f"| `{c['name']}` | {'pass' if c['passed'] else 'FAIL'} |")
    L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L))
