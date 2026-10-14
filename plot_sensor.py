"""Plot one sensor's time series from a GO-MO subgraph dataset, emphasising NaN readings.

Default: the first 100 sensors of data/subgraph/readings.npz, one figure per sensor,
saved in figures/sanity check/.

Figure (three panels):
    1. Full-year time series. NaN readings are shaded red, time steps that were dropped
       from the dataset entirely (outage steps removed in preprocessing) are shaded orange.
       The line breaks wherever there is no value.
    2. Availability map: one row per day, one column per 15-min slot of the day.
       light = valid reading, red = NaN, orange = time step removed from the dataset.
       This shows at a glance WHEN in the day / year the sensor is missing.
    3. Zoom on the window around the longest NaN run (or a window you choose),
       with every individual reading drawn as a dot and NaN steps in red.

Usage:
    python plot_sensor.py                                   # first 100 sensors -> figures/sanity check
    python plot_sensor.py --n-sensors 20
    python plot_sensor.py --data-dir data/subgraph --out-dir "figures/sanity check"
    python plot_sensor.py --start 2024-03-01 --end 2024-03-10   # same zoom window for every sensor
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

VALID, NAN, REMOVED = 0, 1, 2
COLOURS = {VALID: "#f2f2f2", NAN: "#e41a1c", REMOVED: "#ff9f1c"}
LINE_COLOUR = "#1f4e79"


# loading / preparation ------------------------------------------------------------

def load_readings(data_dir):
    """Load (sensor_ids, timestamps, intensity [T, N]) from a subgraph dir or readings.npz."""
    path = Path(data_dir)
    if path.is_dir():
        path = path / "readings.npz"
    z = np.load(path)
    return z["sensor_ids"], pd.DatetimeIndex(z["timestamps"]), z["intensity"]


def to_full_grid(timestamps, series):
    """Put a sensor's series on the complete regular 15-min grid of its year.

    Returns grid, values (NaN where there is no value) and state (VALID / NAN / REMOVED per step).
    REMOVED = the time step is not in the dataset at all (dropped in preprocessing).
    """
    year = timestamps[0].year
    grid = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", inclusive="left")
    idx = grid.get_indexer(timestamps)
    keep = idx >= 0

    values = np.full(len(grid), np.nan, dtype=np.float64)
    values[idx[keep]] = series[keep]
    present = np.zeros(len(grid), dtype=bool)
    present[idx[keep]] = True

    state = np.full(len(grid), VALID, dtype=np.uint8)
    state[~present] = REMOVED
    state[present & np.isnan(values)] = NAN
    return grid, values, state


def runs(mask):
    """(start, end_exclusive) index pairs of consecutive True values."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# plotting ------------------------------------------------------------

def shade(ax, x, mask, colour, alpha):
    """Shade the x-ranges where mask is True over the full axis height.

    The coloured edge keeps single-step gaps visible even when one step is less than a pixel wide.
    """
    ax.fill_between(x, 0, 1, where=mask, transform=ax.get_xaxis_transform(), step="post",
                    facecolor=colour, edgecolor=colour, linewidth=0.9, alpha=alpha, zorder=1)


def plot_sensor(grid, values, state, sensor_id, out_path, zoom, dpi=170):
    nan_mask, removed_mask = state == NAN, state == REMOVED
    nan_runs = runs(nan_mask)
    n_present = int((~removed_mask).sum())
    longest = max((e - s for s, e in nan_runs), default=0)

    fig = Figure(figsize=(16, 13))
    ax1, ax2, ax3 = fig.subplots(3, 1, gridspec_kw={"height_ratios": [3, 3.5, 3]})

    # 1. full-year series
    shade(ax1, grid, removed_mask, COLOURS[REMOVED], 0.35)
    shade(ax1, grid, nan_mask, COLOURS[NAN], 0.55)
    ax1.plot(grid, values, color=LINE_COLOUR, linewidth=0.6, zorder=2)
    ax1.set_xlim(grid[0], grid[-1])
    ax1.set_ylabel("intensity")
    ax1.set_title(
        f"Sensor {sensor_id}: NaN in {nan_mask.sum():,} of {n_present:,} kept steps "
        f"({nan_mask.sum() / max(n_present, 1):.1%}), {len(nan_runs):,} NaN gaps, "
        f"longest {longest:,} steps ({longest * 15 / 60:.1f} h); "
        f"{removed_mask.sum():,} steps removed from dataset",
        fontsize=12, loc="left")
    ax1.legend(handles=[
        Patch(facecolor=COLOURS[NAN], label="NaN reading"),
        Patch(facecolor=COLOURS[REMOVED], alpha=0.6, label="time step removed from dataset"),
    ], loc="upper right", framealpha=0.95)

    # 2. availability map: days x slots of the day
    img = state.reshape(-1, 96)
    ax2.imshow(img, cmap=ListedColormap([COLOURS[VALID], COLOURS[NAN], COLOURS[REMOVED]]),
               vmin=0, vmax=2, aspect="auto", interpolation="nearest",
               extent=(0, 24, img.shape[0], 0))
    month_days = [(m, (pd.Timestamp(grid[0].year, m, 1) - grid[0].normalize()).days) for m in range(1, 13)]
    ax2.set_yticks([d for _, d in month_days])
    ax2.set_yticklabels([pd.Timestamp(2000, m, 1).strftime("%b") for m, _ in month_days])
    ax2.set_xticks(range(0, 25, 3))
    ax2.set_xlabel("hour of day")
    ax2.set_title("Availability map (row = day)", fontsize=11, loc="left")

    # 3. zoom window
    start, end = zoom
    sel = (grid >= start) & (grid < end)
    gz, vz, sz = grid[sel], values[sel], state[sel]
    shade(ax3, gz, sz == REMOVED, COLOURS[REMOVED], 0.35)
    shade(ax3, gz, sz == NAN, COLOURS[NAN], 0.55)
    ax3.plot(gz, vz, color=LINE_COLOUR, linewidth=1.0, marker="o", markersize=2.5, zorder=2)
    ax3.set_xlim(gz[0], gz[-1])
    ax3.set_ylabel("intensity")
    ax3.set_title(f"Zoom: {gz[0]:%Y-%m-%d %H:%M} to {gz[-1]:%Y-%m-%d %H:%M}", fontsize=11, loc="left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)


def zoom_window(grid, state, start=None, end=None, zoom_days=7):
    """Zoom window: the given start/end, else around the sensor's longest NaN run."""
    if start or end:
        return (pd.Timestamp(start) if start else grid[0],
                pd.Timestamp(end) if end else grid[-1] + pd.Timedelta("15min"))
    nan_runs = runs(state == NAN)
    if nan_runs:
        s, e = max(nan_runs, key=lambda r: r[1] - r[0])
        mid = grid[s] + (grid[e - 1] - grid[s]) / 2
        half = max(pd.Timedelta(days=zoom_days / 2), (grid[e - 1] - grid[s]) / 2 + pd.Timedelta(days=1))
    else:
        mid, half = grid[0] + pd.Timedelta(days=zoom_days / 2), pd.Timedelta(days=zoom_days / 2)
    return max(mid - half, grid[0]), min(mid + half, grid[-1] + pd.Timedelta("15min"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/subgraph", help="folder with readings.npz (or the .npz itself)")
    p.add_argument("--n-sensors", type=int, default=100, help="plot the first N sensors (columns 0..N-1)")
    p.add_argument("--out-dir", default="figures/sanity check", help="folder for the figures")
    p.add_argument("--start", help="zoom window start for every sensor (default: around its longest NaN run)")
    p.add_argument("--end", help="zoom window end (exclusive)")
    p.add_argument("--zoom-days", type=float, default=7, help="minimum zoom window length in days")
    p.add_argument("--dpi", type=int, default=170)
    args = p.parse_args()

    sensor_ids, timestamps, intensity = load_readings(args.data_dir)
    n = min(args.n_sensors, len(sensor_ids))
    if n < args.n_sensors:
        print(f"warning: only {n} sensors available (wanted {args.n_sensors})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in range(n):
        sensor_id = int(sensor_ids[col])
        grid, values, state = to_full_grid(timestamps, intensity[:, col])
        zoom = zoom_window(grid, state, args.start, args.end, args.zoom_days)
        out = out_dir / f"sensor_{col:03d}_id{sensor_id}.png"
        plot_sensor(grid, values, state, sensor_id, out, zoom, args.dpi)
        print(f"[{col + 1}/{n}] sensor {sensor_id}: NaN {np.mean(state == NAN):.1%}, "
              f"removed steps {np.mean(state == REMOVED):.1%}; saved {out}")

    print(f"saved {n} figures to {out_dir}")


if __name__ == "__main__":
    main()