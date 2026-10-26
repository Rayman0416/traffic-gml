# Pipeline orchestrator.

# python -m preprocess.run # runs all stages in order
# python -m preprocess.run --stages audit # runs only the audit stage
# python -m preprocess.run --from observed # resume from a stage onwards
# python -m preprocess.run --selection districts --n-districts 6 --min-coverage 0.9
# python -m preprocess.run --from windows --history-length 8
# python -m preprocess.run --stages fill_eval # score the input fill on hidden validation cells

# Stages run in dependency order:

# convert->audit->select->graphs->observed->params->windows->features->checks->report
# Optional stages (fill_eval) only run when named with --stages.

import argparse
import sys
import time
import numpy as np
from . import config as C
from . import audit, sensor_selection, graphs, observed, params, windows, features, fill, raw, checks, report


STAGES = ("convert", "audit", "select", "graphs", "observed", "params"
          , "windows", "features", "checks", "report")
OPTIONAL_STAGES = ("fill_eval",)


def _stage_convert(args):
    raw.convert_readings_to_parquet(overwrite=args.overwrite)


def _stage_audit(args):
    """
    Run the audit stage to generate an overview of the dataset.
    If the audit report already exists, skip unless explicitly requested.
    """
    explicit = bool(args.stages and "audit" in args.stages)
    target = C.reports_dir(args.out_dir) / "audit.json"
    if target.exists() and not explicit:
        print(f"  {target.name} already exists, skipping "
              f"(--stages audit to re-run)")
        return
    audit.run(out_dir=args.out_dir)


def _stage_select(args):
    """
    Run the sensor selection stage to select the relevant sensors.
    """
    sensor_selection.build(out_dir=args.out_dir, min_coverage=args.min_coverage,
                           max_gap=args.max_nan_run, selection=args.selection,
                           n_districts=args.n_districts)


def _stage_graphs(args):
    """
    Run the graph stage to generate the sensor graph and related visualizations.
    """
    graphs.build(out_dir=args.out_dir)


def _stage_observed(args):
    """
    Run the observed stage to generate the observed intensity and mask arrays.
    """
    observed.build(out_dir=args.out_dir)


def _stage_params(args):
    """
    Run the params stage to generate the statistical parameters for each sensor.
    """
    d = observed.load(args.out_dir)
    params.build(d["observed_intensity"], d["mask"], d["timestamps"],
                 d["sensor_ids"], d["observed_occupancy"], d["occupancy_mask"],
                 out_dir=args.out_dir)


def _stage_windows(args):
    """
    Run the windows stage to generate the training and test windows for each sensor.
    """
    windows.build(out_dir=args.out_dir, input_length=args.history_length)


def _stage_features(args):
    """
    Run the features stage to build the scaled, masked per-sensor input features.
    """
    features.build(out_dir=args.out_dir, history_length=args.history_length)


def _stage_checks(args):
    """
    Run the checks stage to validate the generated data.
    """
    checks.run(out_dir=args.out_dir, strict=not args.no_strict)


def _stage_fill_eval(args):
    """
    Score the input fill against genuine validation cells hidden as missing.
    """
    fill.evaluate(out_dir=args.out_dir)


def _stage_report(args):
    """
    Run the report stage to generate a summary report of the pipeline run.
    """
    report.build(out_dir=args.out_dir)


RUNNERS = {
    "convert": _stage_convert,
    "audit": _stage_audit,
    "select": _stage_select,
    "graphs": _stage_graphs,
    "observed": _stage_observed,
    "params": _stage_params,
    "windows": _stage_windows,
    "features": _stage_features,
    "checks": _stage_checks,
    "report": _stage_report,
    "fill_eval": _stage_fill_eval,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m preprocess.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stages", nargs="+", choices=STAGES + OPTIONAL_STAGES,
                   help="run only these stages (default: all)")
    p.add_argument("--from", dest="from_stage", choices=STAGES,
                   help="run from this stage to the end")
    p.add_argument("--out-dir", type=lambda s: __import__("pathlib").Path(s),
                   default=C.OUT_DIR, help="output directory")
    p.add_argument("--overwrite", action="store_true",
                   help="re-convert the raw CSV even if the Parquet exists")
    p.add_argument("--n-districts", type=int, default=C.N_DISTRICTS,
                   help="districts to select with --selection districts")
    p.add_argument("--min-coverage", type=float, default=C.MIN_TRAIN_COVERAGE,
                   help="minimum fraction of genuine training observations")
    p.add_argument("--max-nan-run", type=int, default=C.MAX_TRAIN_GAP_STEPS,
                   help="maximum consecutive training gap, in 15-min steps")
    p.add_argument("--selection", choices=("all", "districts"), default=C.SELECTION,
                   help="the whole network, or a connected block of whole "
                        "districts around the centre")
    p.add_argument("--history-length", type=int, default=C.INPUT_LENGTH,
                   help="input history in 15-min steps, including the origin "
                        "(windows and features stages)")
    p.add_argument("--no-strict", action="store_true",
                   help="report failing checks instead of raising")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.stages:
        stages = [s for s in STAGES + OPTIONAL_STAGES if s in set(args.stages)]
    elif args.from_stage:
        stages = list(STAGES[STAGES.index(args.from_stage):])
    else:
        stages = list(STAGES)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(C.RANDOM_SEED)

    # CLI overrides of recorded settings, so source.json reflects this run.
    overrides = {"INPUT_LENGTH": args.history_length,
                 "SELECTION": args.selection,
                 "N_DISTRICTS": args.n_districts}
    C.write_json(C.source({"stages_requested": stages}, overrides),
                 args.out_dir / "source.json")

    print(f"output: {args.out_dir}")
    print(f"stages: {' -> '.join(stages)}\n")

    timings = {}
    for name in stages:
        print(f"[{name}]")
        t0 = time.time()
        RUNNERS[name](args)
        timings[name] = round(time.time() - t0, 1)
        print(f"  done in {timings[name]}s\n")

    C.write_json(C.source({"stages_run": stages, "seconds": timings}, overrides),
                 args.out_dir / "source.json")
    print("pipeline complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
