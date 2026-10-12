"""Sensor-availability pictures for GO-MO subgraph datasets (matplotlib only).

Each image has one row per sensor (y) and one column per time step (x):
    black = NaN (no valid reading)
    grey  = reading is exactly 0
    white = any other reading

Standalone usage (one picture per dataset directory or readings.npz):
    python visualize.py                          # defaults to data/subgraph
    python visualize.py data/subgraph data/subgraph_v2 --out-dir figures
    python visualize.py data/subgraph --time-bin 96 --y-scale 3
    python visualize.py data/subgraph --start 2024-11-01 --end 2024-12-01 --annotate
    python visualize.py data/subgraph --sort missing

From the main script or a notebook:
    import visualize
    visualize.make_picture(intensity, timestamps, "figures/a.png", time_bin=4)
    visualize.draw_graph(sensors.loc[sensor_ids], edges, "figures/graph.png")
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from matplotlib.collections import LineCollection
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, date2num
from matplotlib.figure import Figure
from matplotlib.patches import Patch

BLACK, GREY, WHITE = 0, 128, 255  # grayscale levels used in the image


# loading ------------------------------------------------------------

def load_readings(path):
    """Load (sensor_ids, timestamps, intensity [T, N]) from a subgraph dir or a readings.npz file."""
    path = Path(path)
    if path.is_dir():
        path = path / "readings.npz"
    z = np.load(path)
    return z["sensor_ids"], pd.DatetimeIndex(z["timestamps"]), z["intensity"]


# image building ------------------------------------------------------------
def slice_time(intensity, timestamps, start=None, end=None):
    """Restrict to timestamps in [start, end). Either bound may be None."""
    mask = np.ones(len(timestamps), dtype=bool)
    if start is not None:
        mask &= timestamps >= pd.Timestamp(start)
    if end is not None:
        mask &= timestamps < pd.Timestamp(end)
    return intensity[mask], timestamps[mask]


def to_image(intensity, time_bin=1):
    """Convert intensity [T, N] to a uint8 image [N, ceil(T / time_bin)].

    time_bin > 1 merges consecutive time steps into one pixel column. A merged pixel is
    white if any step in it has a non-zero reading, else grey if any step is exactly 0,
    else black (all NaN). The last bin is padded with NaN if T is not divisible.
    """
    x = np.asarray(intensity, dtype=np.float32)
    if time_bin > 1:
        pad = (-x.shape[0]) % time_bin
        if pad:
            x = np.vstack([x, np.full((pad, x.shape[1]), np.nan, dtype=np.float32)])
        x = x.reshape(-1, time_bin, x.shape[1])  # [T', bin, N]
        any_other = np.any(~np.isnan(x) & (x != 0), axis=1)
        any_zero = np.any(x == 0, axis=1)
    else:
        any_other = ~np.isnan(x) & (x != 0)
        any_zero = x == 0

    img = np.full(any_other.shape, BLACK, dtype=np.uint8)
    img[any_zero] = GREY
    img[any_other] = WHITE  # white wins over grey
    return img.T  # rows = sensors, columns = time


# saving ------------------------------------------------------------

def save_plain(img, out_path, x_scale=1, y_scale=1):
    """Save the raw pixel image: one pixel per sensor / time bin, enlarged by integer factors."""
    if x_scale != 1 or y_scale != 1:
        img = np.repeat(np.repeat(img, y_scale, axis=0), x_scale, axis=1)
    mpimg.imsave(out_path, img, cmap="gray", vmin=0, vmax=255)


def save_annotated(img, timestamps, out_path, title=None, dpi=150, figsize=(14, 6)):
    """Save a figure with a date axis, a sensor axis and a legend."""
    t0 = date2num(timestamps[0])
    step = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else pd.Timedelta("15min")
    t1 = date2num(timestamps[0] + step * len(timestamps))

    fig = Figure(figsize=figsize)  # object API: no pyplot, no global backend state
    ax = fig.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto",
              interpolation="nearest", extent=(t0, t1, img.shape[0], 0))
    ax.xaxis_date()
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.set_xlabel("time")
    ax.set_ylabel("sensor (row)")
    ax.set_title(title or "")
    ax.legend(handles=[
        Patch(facecolor="black", edgecolor="white", label="NaN"),
        Patch(facecolor=str(GREY / 255), edgecolor="white", label="0"),
        Patch(facecolor="white", edgecolor="black", label="> 0 / other"),
    ], loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)


# graph drawing ------------------------------------------------------------

def _components(ids, edges):
    """Connected-component label per sensor id (edges treated as undirected), via union-find."""
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in zip(edges["source"], edges["target"]):
        if u in parent and v in parent:
            parent[find(u)] = find(v)
    return np.array([find(i) for i in ids])


def draw_graph(sensors, edges, out_path, title=None, node_size=8, edge_width=0.5,
               edge_alpha=0.4, figsize=(10, 10), dpi=200):
    """Draw the sensors as nodes at their UTM position, with the route-graph edges between them.

    sensors: DataFrame indexed by sensor id with `utm_x` and `utm_y` columns
             (e.g. the `sensors` returned by load_subgraph, or sensors.loc[subgraph]).
    edges:   DataFrame with `source` and `target` columns (e.g. from subgraph_edges).
    Nodes in the largest connected component are blue; nodes outside it are red,
    so a disconnected region is easy to spot. Edges are drawn without direction.
    """
    ids = list(sensors.index)
    pos = dict(zip(ids, zip(sensors["utm_x"], sensors["utm_y"])))

    # Edges whose two endpoints both have a position
    inside = [(u, v) for u, v in zip(edges["source"], edges["target"]) if u in pos and v in pos]
    segments = [(pos[u], pos[v]) for u, v in inside]

    labels = _components(ids, edges)
    values, counts = np.unique(labels, return_counts=True)
    in_main = labels == values[np.argmax(counts)]
    xy = np.array([pos[i] for i in ids])

    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    ax.add_collection(LineCollection(segments, colors="0.3", linewidths=edge_width, alpha=edge_alpha))
    ax.scatter(xy[in_main, 0], xy[in_main, 1], s=node_size, c="tab:blue", zorder=3,
               label=f"largest component ({in_main.sum()})")
    if (~in_main).any():
        ax.scatter(xy[~in_main, 0], xy[~in_main, 1], s=node_size, c="tab:red", zorder=3,
                   label=f"other components ({(~in_main).sum()})")
        ax.legend(loc="upper right")
    ax.set_aspect("equal")  # UTM coordinates are metres, so keep x and y to the same scale
    ax.autoscale_view()
    ax.set_xlabel("UTM x (m)")
    ax.set_ylabel("UTM y (m)")
    ax.set_title(title or f"{len(ids)} sensors, {len(inside):,} edges, {len(values)} component(s)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    return len(values)  # number of connected components (1 = fully connected)


# high-level entry point ------------------------------------------------------------

def make_picture(intensity, timestamps, out_path, sensor_ids=None,
                 start=None, end=None, time_bin=1, x_scale=1, y_scale=1,
                 annotate=False, title=None):
    """Build and save one picture from in-memory arrays. Returns the uint8 image [N, T']."""
    intensity, timestamps = slice_time(intensity, timestamps, start, end)
    if intensity.shape[0] == 0:
        raise ValueError("no time steps left after slicing")
    img = to_image(intensity, time_bin)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if annotate:
        save_annotated(img, timestamps, out_path, title)
    else:
        save_plain(img, out_path, x_scale, y_scale)
    return img


def _stats(intensity):
    n = intensity.size
    nan = np.isnan(intensity).sum()
    zero = (intensity == 0).sum()
    return f"NaN {nan / n:.1%}, zero {zero / n:.1%}, other {(n - nan - zero) / n:.1%}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", default=["data/subgraph"],
                   help="subgraph directories or readings.npz files (one picture each)")
    p.add_argument("--out-dir", default="figures", help="where to write the pictures")
    p.add_argument("--suffix", default="", help="text appended to each output file name")
    p.add_argument("--start", help="first timestamp to include, e.g. 2024-11-01")
    p.add_argument("--end", help="end timestamp (exclusive)")
    p.add_argument("--time-bin", type=int, default=1, help="time steps per pixel column (96 = one day at 15 min)")
    p.add_argument("--x-scale", type=int, default=1, help="pixel enlargement along time (plain image)")
    p.add_argument("--y-scale", type=int, default=1, help="pixel enlargement along sensors (plain image)")
    p.add_argument("--annotate", action="store_true", help="figure with axes and legend")
    args = p.parse_args()

    for path in map(Path, args.paths):
        sensor_ids, timestamps, intensity = load_readings(path)
        name = (path.parent.name if path.suffix == ".npz" else path.name) or "dataset"
        out = Path(args.out_dir) / f"{name}{args.suffix}.png"
        img = make_picture(intensity, timestamps, out, sensor_ids, args.start, args.end,
                           args.time_bin, args.x_scale, args.y_scale, args.annotate, title=name)
        print(f"{path}: {intensity.shape[1]} sensors x {intensity.shape[0]:,} steps -> "
              f"{img.shape[1]}x{img.shape[0]} px, {_stats(intensity)}; saved {out}")


if __name__ == "__main__":
    main()