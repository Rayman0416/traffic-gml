# Build the three tables of the paper from the saved test results.
# Each model writes its own results file in <data-dir>/reports/ (*_results.csv) with the same columns:
#   split, model, horizon_min, MAE, RMSE, seed, n_params, epochs, train_s, epoch_s, infer_s, peak_gb
# Every *_results.csv is read, so a new model only has to write its own file to show up here.
# graph_results.csv keeps every run ever made: only the graph models of the selected groups are kept.
#
#   Table 1 (main)      non-graph models + full-city, traffic router, aware router: MAE at 15 and 60 min
#   Table 2 (rq2)       cost of partitioning, all without local attention: memory, time, parameters, exchange
#   Table 3 (ablation)  our model against each ablation, MAE at 60 min and the difference paired by seed
#
# Run from the repo root, on one machine, with nothing else using the GPU (otherwise timings mean nothing):
#   python rq1/compare.py                          # compare whatever results are already saved
#   python rq1/compare.py --run                    # run every model for seeds 0 1 2 first
#   python rq1/compare.py --run --skip baseline nn # only (re)run the graph models
#   python rq1/compare.py --run --groups ablation  # only Table 3 (our model is always run)
#   python rq1/compare.py --run --seeds 0          # one seed, quicker
#   python rq1/compare.py --run --seeds 0 --epochs 1   # check that everything runs
#   python rq1/compare.py --data-dir data/forecast_dataset_full

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for the graph script
import attention_traffic_gtcnn as gt  # noqa: E402

NOTEBOOK = Path("rq1/baseline_nongraph.ipynb")
GRAPH_SCRIPT = "attention_traffic_gtcnn.py"
GRAPH_FILE = "graph_results.csv"
OURS = "Aware router (ours)"
# graph rows of the paper, one group per table: name in the table -> flags for attention_traffic_gtcnn.py
GROUPS = {
    "main": {
        "Full-city": ["--partition", "none"],
        "Traffic router": ["--router", "traffic"],
        OURS: [],
    },
    # no local attention, so memory measures the partitioning and not the dense N x N attention
    "rq2": {
        "Full-city (no attn)": ["--partition", "none", "--attn", "none"],
        "No exchange (no attn)": ["--k", "0", "--attn", "none"],
        "Aware router (no attn)": ["--attn", "none"],
    },
    "ablation": {
        "No exchange (K=0)": ["--k", "0"],
        "No gate": ["--gate", "off"],
        "K=1": ["--k", "1"],
        "K=4": ["--k", "4"],
        "Full exchange": ["--exchange", "full"],
        "Edge prior": ["--prior"],
    },
}
COST = ["n_params", "epochs", "train_s", "epoch_s", "infer_s", "peak_gb"]
# fixed categorical order (validated: dataviz palette, light mode) and a marker per series
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def graph_args(flags: list):
    """The flags parsed with the same defaults as attention_traffic_gtcnn.py."""
    p = argparse.ArgumentParser()
    p.add_argument("--attn", default=gt.ATTN)
    p.add_argument("--k", type=int, default=gt.K)
    p.add_argument("--router", default=gt.ROUTER)
    p.add_argument("--gate", default="on")
    p.add_argument("--prior", action="store_true")
    p.add_argument("--exchange", default="detached")
    p.add_argument("--partition", default="district")
    args = p.parse_args(flags)
    if args.partition == "none":
        args.k = 0  # as in the graph script: one graph, nobody to exchange with
    return args


def selected_runs(groups: list) -> dict:
    """Paper name -> flags for the selected groups, our model always first."""
    runs = {OURS: GROUPS["main"][OURS]}
    for g in groups:
        runs.update(GROUPS[g])
    return runs


def model_name(flags: list) -> str:
    return gt.run_name(graph_args(flags))


def scalars_exchanged(flags: list) -> int:
    """Numbers each district receives per origin: K tokens, plus their support if aware."""
    a = graph_args(flags)
    return a.k * (gt.HIDDEN + (2 if a.router == "aware" else 0))


def run_notebook(path: Path, data_dir: Path):
    """Execute the notebook's code cells in order, from the repo root (no nbconvert needed)."""
    cells = json.load(open(path))["cells"]
    namespace = {"__name__": "__main__"}
    for cell in cells:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"]).replace('data_dir = "data/forecast_dataset"', f'data_dir = "{data_dir}"')
            exec(compile(src, str(path), "exec"), namespace)
    plt.close("all")


def run(cmd):
    print("==", " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, check=True)


def run_all(skip: set, seeds, data_dir: Path, runs: dict, epochs=None):
    extra = ["--epochs", str(epochs)] if epochs else []  # only for quick tests
    if "baseline" not in skip:
        print(f"== baselines ({NOTEBOOK})", flush=True)
        run_notebook(NOTEBOOK, data_dir)  # deterministic, one run is enough
    unique = {model_name(f): f for f in runs.values()}  # a run in two groups is trained once
    for seed in seeds:
        if "nn" not in skip:
            run(["rq1/NN.py", "--seed", str(seed), "--data-dir", str(data_dir), *extra])
        if "graph" not in skip:
            for flags in unique.values():
                run([GRAPH_SCRIPT, *flags, "--seed", str(seed), "--data-dir", str(data_dir), *extra])


def load_results(reports: Path, runs: dict) -> pd.DataFrame:
    """Test rows with a seed from every *_results.csv; graph rows only for the selected runs, labelled."""
    frames = []
    for path in sorted(reports.glob("*_results.csv")):
        df = pd.read_csv(path)
        df["graph"] = path.name == GRAPH_FILE
        frames.append(df)
    if not frames:
        sys.exit(f"no *_results.csv in {reports}")
    results = pd.concat(frames, ignore_index=True)
    for c in ["seed"] + COST:
        if c not in results:
            results[c] = float("nan")
    test = results[(results["split"] == "test") & results["seed"].notna()].copy()  # old runs had no seed

    labels = {}
    for label, flags in runs.items():
        labels.setdefault(model_name(flags), label)
    graph = test["graph"]
    ignored = sorted(set(test.loc[graph, "model"]) - labels.keys())
    if ignored:
        print(f"ignored in {GRAPH_FILE} (not in the selected groups): " + "; ".join(ignored))
    test = test[~graph | test["model"].isin(labels.keys())]
    missing = [f"{lab} ({m})" for m, lab in labels.items() if m not in set(test["model"])]
    if missing:
        print("no results yet: " + "; ".join(missing))
    test["label"] = test["model"].map(labels).fillna(test["model"])
    return test.drop(columns="graph")


def mean_std(g: pd.Series, digits=1, sign=False) -> str:
    g = g.dropna()
    if g.empty:
        return "n/a"
    m = f"{g.mean():{'+' if sign else ''}.{digits}f}"
    return m if len(g) < 2 else f"{m} ± {g.std():.{digits}f}"


def mae_at(test: pd.DataFrame, label: str, horizon: int) -> pd.Series:
    """MAE of one model at one horizon, indexed by seed."""
    sub = test[(test["label"] == label) & (test["horizon_min"] == horizon)]
    return sub.set_index("seed")["MAE"]


def tex(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    s = s.replace("±", r"$\pm$").replace("Δ", r"$\Delta$").replace("—", "---")
    return re.sub(r"(?<!\w)-(?=\d)", "$-$", s)  # minus sign of a number, not a hyphen


def write_tex(table: pd.DataFrame, path: Path, rule_after=None):
    """booktabs tabular; `rule_after` = row positions followed by a \\midrule."""
    lines = [r"\begin{tabular}{l" + "r" * len(table.columns) + "}", r"\toprule",
             " & ".join(tex(c) for c in [table.index.name or ""] + list(table.columns)) + r" \\", r"\midrule"]
    for i, (idx, row) in enumerate(table.iterrows()):
        lines.append(" & ".join(tex(v) for v in [idx, *row]) + r" \\")
        if rule_after and i in rule_after and i < len(table) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"saved {path}")


def table1(test: pd.DataFrame, reports: Path) -> list:
    graph_labels = [lab for lab in GROUPS["main"] if lab in set(test["label"])]
    other = test[~test["label"].isin(set().union(*map(set, GROUPS.values())))]
    worst_first = (other[other["horizon_min"] == 60].groupby("label")["MAE"].mean()
                   .sort_values(ascending=False).index.tolist())
    order = worst_first + graph_labels
    sub = test[test["label"].isin(order)]

    for metric in ("MAE", "RMSE"):
        t = sub.groupby(["label", "horizon_min"])[metric].apply(mean_std).unstack().loc[order]
        t.columns = [f"{c} min" for c in t.columns]
        t.index.name = "model"
        print(f"\ntest {metric} at every horizon (vehicles/hour, mean ± std over seeds), lower is better")
        print(t.to_string())

    t = sub[sub["horizon_min"].isin([15, 60])].groupby(["label", "horizon_min"])["MAE"].apply(mean_std)
    t = t.unstack().loc[order]
    t.columns = [f"MAE {c} min" for c in t.columns]
    t.index.name = "Model"
    print("\nTable 1: test MAE (vehicles/hour, mean ± std over seeds)")
    print(t.to_string())
    write_tex(t, reports / "table1_main.tex", rule_after={len(worst_first) - 1})
    return order


def rq1_paired(test: pd.DataFrame):
    diff = (mae_at(test, OURS, 60) - mae_at(test, "Traffic router", 60)).dropna()
    print("\nRQ1, paired by seed: MAE at 60 min of aware router minus traffic router (< 0: aware is better)")
    if diff.empty:
        print("  no seed has both runs")
        return
    for seed, d in diff.items():
        print(f"  seed {int(seed)}: {d:+.2f}")
    print(f"  mean {mean_std(diff, 2, sign=True)}; aware router wins {(diff < 0).sum()} of {len(diff)} seeds")


def table2(test: pd.DataFrame, reports: Path):
    rows = {}
    for label, flags in GROUPS["rq2"].items():
        per_run = test[test["label"] == label].drop_duplicates("seed")
        rows[label] = {"peak_gb": per_run["peak_gb"].mean(), "epoch_s": per_run["epoch_s"].mean(),
                       "train_s": per_run["train_s"].mean(), "n_params": per_run["n_params"].mean(),
                       "scalars_exchanged": scalars_exchanged(flags),
                       "MAE 60 min": mean_std(mae_at(test, label, 60)), "seeds": len(per_run)}
    t = pd.DataFrame.from_dict(rows, orient="index")
    t.index.name = "model"
    t = t.round({"peak_gb": 2, "epoch_s": 1, "train_s": 1})
    t["n_params"] = t["n_params"].round().astype("Int64")
    print("\nTable 2: cost of partitioning, no local attention (mean over seeds): peak GPU memory [GB], "
          "time per epoch [s], total training time [s], parameters, scalars received per district per origin, "
          "test MAE at 60 min")
    print(t.to_string())
    t.to_csv(reports / "table2_cost.csv")
    print(f"saved {reports / 'table2_cost.csv'}")

    full, part, ours = (mae_at(test, lab, 60).mean() for lab in GROUPS["rq2"])
    gap = part - full
    if pd.isna(gap):
        print("partitioning gap: needs both the full-city and the no-exchange run")
        return
    print(f"partitioning gap MAE(K=0) - MAE(full) = {gap:.2f} veh/h at 60 min")
    if gap > 0 and not pd.isna(ours):
        print(f"share the tokens recover (MAE(K=0) - MAE(aware)) / gap = {(part - ours) / gap:.0%}")
    elif gap <= 0:
        print("no gap to recover: partitioning did not hurt, the share is undefined")


def table3(test: pd.DataFrame, reports: Path):
    base = mae_at(test, OURS, 60)
    rows = {OURS: {"MAE 60 min": mean_std(base), "Δ vs ours": "—"}}
    for label in GROUPS["ablation"]:
        mae = mae_at(test, label, 60)
        rows[label] = {"MAE 60 min": mean_std(mae), "Δ vs ours": mean_std((mae - base).dropna(), 2, sign=True)}
    t = pd.DataFrame.from_dict(rows, orient="index")
    t.index.name = "Variant"
    print("\nTable 3: ablations, test MAE at 60 min (vehicles/hour, mean ± std over seeds); "
          "Δ = variant - ours paired by seed, > 0 means the variant is worse")
    print(t.to_string())
    write_tex(t, reports / "table3_ablation.tex", rule_after={0})


def plot_table1(test: pd.DataFrame, order: list):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, label in enumerate(order):
        g = test[test["label"] == label].groupby("horizon_min")["MAE"]
        ax.errorbar(g.mean().index, g.mean(), yerr=g.std().fillna(0), capsize=3, lw=2, ms=8,
                    color=COLORS[i % len(COLORS)], marker=MARKERS[i % len(MARKERS)], label=label)
    ax.set_xlabel("forecast horizon (minutes)")
    ax.set_ylabel("test MAE (vehicles/hour)")
    ax.set_xticks([15, 30, 45, 60])
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/table1_mae_by_horizon.png", dpi=200)
    plt.close(fig)
    print("saved figures/table1_mae_by_horizon.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the models before comparing")
    parser.add_argument("--skip", nargs="*", default=[], choices=["baseline", "nn", "graph"], help="models not to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--groups", nargs="*", default=list(GROUPS), choices=list(GROUPS),
                        help="tables to build; main = Table 1, rq2 = Table 2, ablation = Table 3")
    parser.add_argument("--data-dir", type=Path, default=Path("data/forecast_dataset"))
    parser.add_argument("--epochs", type=int, default=None, help="override the epochs, for a quick test only")
    args = parser.parse_args()
    reports = args.data_dir / "reports"
    if not reports.exists():
        sys.exit(f"{reports} not found: run this from the repo root")
    runs = selected_runs(args.groups)

    if args.run:
        run_all(set(args.skip), args.seeds, args.data_dir, runs, args.epochs)

    test = load_results(reports, runs)
    test.to_csv(reports / "comparison_all.csv", index=False)
    print(f"saved {reports / 'comparison_all.csv'}")

    if "main" in args.groups:
        order = table1(test, reports)
        rq1_paired(test)
        plot_table1(test, order)
    if "rq2" in args.groups:
        table2(test, reports)
    if "ablation" in args.groups:
        table3(test, reports)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
