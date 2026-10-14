"""Hourly traffic-flow vector fields for a GO-MO subgraph dataset.

Readings are scalar (veh/h), so the flow DIRECTION of each sensor comes from the directed
route graph: a sensor's direction is the weight-averaged unit vector towards its downstream
neighbours (out-edges). Sensors without out-edges use the direction from their upstream
neighbours (in-edges). The vector is then

    v_i(hour) = mean_intensity_i(hour) * direction_i

where mean_intensity_i(hour) is the mean over all days of the year and over the four
15-minute steps in that hour (NaN steps are ignored).

Because the direction is static, the hourly variation comes from the magnitude only.
Where branches/opposite directions leave a sensor, the resultant is shorter than the
intensity (flow is split), which is intended.

Usage:
    python vector_field.py                                   # smooth field -> figures/vector_field_smooth
    python vector_field.py --data-dir data/subgraph
    python vector_field.py --mode sensors                    # original one-arrow-per-sensor plots
    python vector_field.py --sigma 600 --grid-cell 100 --quiver-step 8

Modes:
    smooth   (default) the sensor vectors are interpolated with a Gaussian kernel onto a dense
             regular grid, giving a continuous field. Drawn as a magnitude heat-map, streamlines
             (thicker = stronger) and a sparse quiver overlay. Cells further than roughly
             2 sigma from any sensor are left blank instead of extrapolated.
    sensors  one arrow per sensor (no interpolation)

Outputs (in --out-dir):
    hour_00.png ... hour_23.png   one figure per hour of the day
    overview.png                  all 24 hours in one 4x6 figure (shared colour/arrow scale)
    vector_fields.npz             sensor_ids, utm_xy, direction, hourly_intensity, hourly_vectors
                                  (+ grid_x, grid_y, grid_uv [24, Gy, Gx, 2] in smooth mode)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


# loading ------------------------------------------------------------

def load_dataset(data_dir):
    """Load (sensor_ids, timestamps, intensity [T, N], edges, sensors) from a subgraph directory."""
    data_dir = Path(data_dir)
    z = np.load(data_dir / "readings.npz")
    sensors = pd.read_csv(data_dir / "sensors.csv", index_col="id")
    edges = pd.read_csv(data_dir / "edges.csv")
    return z["sensor_ids"], pd.DatetimeIndex(z["timestamps"]), z["intensity"], edges, sensors


# vector field construction ------------------------------------------------------------

def sensor_directions(sensor_ids, sensors, edges):
    """Unit-ish flow direction [N, 2] per sensor (UTM metres frame) from the directed edges.

    Out-edges: weight-averaged unit vectors towards downstream neighbours.
    Fallback for sensors with no out-edge: weight-averaged unit vectors from upstream neighbours.
    Sensors with no edges at all get (0, 0).
    """
    index = {int(s): i for i, s in enumerate(sensor_ids)}
    xy = sensors.loc[sensor_ids, ["utm_x", "utm_y"]].to_numpy(dtype=float)
    n = len(sensor_ids)

    out_sum, in_sum = np.zeros((n, 2)), np.zeros((n, 2))
    out_w, in_w = np.zeros(n), np.zeros(n)

    for u, v, w in zip(edges["source"], edges["target"], edges["weight"]):
        if u not in index or v not in index or u == v:
            continue
        i, j = index[u], index[v]
        d = xy[j] - xy[i]
        norm = np.hypot(*d)
        if norm == 0:
            continue
        d = d / norm
        out_sum[i] += w * d
        out_w[i] += w
        in_sum[j] += w * d
        in_w[j] += w

    direction = np.zeros((n, 2))
    has_out = out_w > 0
    has_in = (~has_out) & (in_w > 0)
    direction[has_out] = out_sum[has_out] / out_w[has_out, None]
    direction[has_in] = in_sum[has_in] / in_w[has_in, None]
    return xy, direction


def hourly_mean_intensity(timestamps, intensity):
    """Mean intensity [24, N] per hour of day, averaged over all days and the steps within the hour."""
    hours = timestamps.hour.to_numpy()
    out = np.full((24, intensity.shape[1]), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns -> NaN, fine
        for h in range(24):
            sel = hours == h
            if sel.any():
                out[h] = np.nanmean(intensity[sel], axis=0)
    return out


def smooth_to_grid(xy, vectors, cell=400.0, sigma=500.0, min_weight=0.3, pad=500.0):
    """Gaussian-kernel interpolation of per-sensor vectors onto a regular grid.

    vectors: [H, N, 2] (NaN = missing sensor). Returns grid_x [Gy, Gx], grid_y [Gy, Gx],
    grid_uv [H, Gy, Gx, 2] with NaN where total kernel weight < min_weight (no sensors nearby).
    """
    xs = np.arange(xy[:, 0].min() - pad, xy[:, 0].max() + pad + cell, cell)
    ys = np.arange(xy[:, 1].min() - pad, xy[:, 1].max() + pad + cell, cell)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    d2 = ((pts[:, None, :] - xy[None, :, :]) ** 2).sum(-1)  # [G, N]
    kernel = np.exp(-d2 / (2 * sigma ** 2))

    grid_uv = np.full((vectors.shape[0], len(pts), 2), np.nan)
    for h in range(vectors.shape[0]):
        ok = ~np.isnan(vectors[h]).any(axis=1)
        k = kernel * ok[None, :]
        wsum = k.sum(axis=1)
        v = np.nan_to_num(vectors[h])
        uv = (k @ v) / np.maximum(wsum, 1e-12)[:, None]
        uv[wsum < min_weight] = np.nan
        grid_uv[h] = uv
    return xs, ys, grid_uv.reshape(vectors.shape[0], *gx.shape, 2)


# plotting ------------------------------------------------------------

def draw_field(ax, positions, uv, vmax, ref_mag, arrow_len, edge_segments=None, small=False):
    """Draw one quiver field. Arrow length (metres) = |v| / ref_mag * arrow_len, colour = |v|."""
    mag = np.hypot(uv[:, 0], uv[:, 1])
    ok = ~np.isnan(mag)

    if edge_segments is not None:
        ax.add_collection(LineCollection(edge_segments, colors="0.8", linewidths=0.3, zorder=1))

    ax.quiver(
        positions[ok, 0], positions[ok, 1], uv[ok, 0], uv[ok, 1], mag[ok],
        angles="xy", scale_units="xy", scale=ref_mag / arrow_len,
        cmap="viridis", norm=Normalize(0, vmax),
        width=0.004 if small else 0.0028, headwidth=3.5, headlength=4, zorder=2,
    )
    ax.set_aspect("equal")
    ax.autoscale_view()
    if small:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("UTM x (m)")
        ax.set_ylabel("UTM y (m)")


def draw_smooth(ax, xs, ys, uv, vmax, ref_mag, arrow_len, segments=None, sensor_xy=None,
                quiver_step=6, small=False):
    """Draw an interpolated field uv [Gy, Gx, 2]: magnitude heat-map + streamlines + sparse arrows."""
    mask = np.isnan(uv).any(axis=-1)
    U = np.ma.array(np.nan_to_num(uv[..., 0]), mask=mask)
    V = np.ma.array(np.nan_to_num(uv[..., 1]), mask=mask)
    mag = np.ma.array(np.hypot(np.nan_to_num(uv[..., 0]), np.nan_to_num(uv[..., 1])), mask=mask)

    ax.pcolormesh(xs, ys, mag, cmap="YlOrRd", norm=Normalize(0, vmax), shading="nearest", zorder=0)
    if segments is not None:
        ax.add_collection(LineCollection(segments, colors="0.45", linewidths=0.25, alpha=0.5, zorder=1))
    if sensor_xy is not None and not small:
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=2, c="k", alpha=0.35, zorder=2, linewidths=0)

    linewidth = 0.3 + 1.7 * np.clip(mag.filled(0) / vmax, 0, 1)
    ax.streamplot(xs, ys, U, V, color="0.15", linewidth=linewidth, density=1.0 if small else 1.7,
                  arrowsize=0.7 if small else 1.0, zorder=3)

    s = slice(quiver_step // 2, None, quiver_step)
    qx, qy = np.meshgrid(xs[s], ys[s])
    qu, qv, qok = U.filled(0)[s, s], V.filled(0)[s, s], ~mask[s, s]
    ax.quiver(qx[qok], qy[qok], qu[qok], qv[qok], angles="xy", scale_units="xy",
              scale=ref_mag / arrow_len, color="k", width=0.0035 if small else 0.0022,
              headwidth=3.5, headlength=4, zorder=4)

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_aspect("equal")
    if small:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("UTM x (m)")
        ax.set_ylabel("UTM y (m)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/subgraph_normal",
                   help="folder with readings.npz, edges.csv and sensors.csv")
    p.add_argument("--out-dir", default=None, help="default: figures/vector_field_<mode>")
    p.add_argument("--mode", choices=["smooth", "sensors"], default="sensor",
                   help="smooth: interpolated continuous field; sensors: one arrow per sensor")
    p.add_argument("--arrow-length", type=float, default=300.0,
                   help="length in metres of an arrow with the reference (99th pct) magnitude")
    p.add_argument("--grid-cell", type=float, default=150.0, help="grid spacing in metres (smooth mode)")
    p.add_argument("--sigma", type=float, default=300.0, help="Gaussian kernel width in metres (smooth mode)")
    p.add_argument("--min-weight", type=float, default=0.05,
                   help="smooth mode: blank cells whose total kernel weight is below this "
                        "(0.15 = about 2 sigma from the nearest lone sensor)")
    p.add_argument("--quiver-step", type=int, default=1, help="smooth mode: arrow every N grid cells")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    out_dir = Path(args.out_dir or f"figures/vector_field_{args.mode}")

    sensor_ids, timestamps, intensity, edges, sensors = load_dataset(args.data_dir)
    print(f"loaded {len(sensor_ids)} sensors, {len(timestamps):,} steps, {len(edges):,} directed edges")

    xy, direction = sensor_directions(sensor_ids, sensors, edges)
    n_nodir = int((np.hypot(*direction.T) == 0).sum())
    if n_nodir:
        print(f"note: {n_nodir} sensors have no usable edge and get a zero vector")

    hourly = hourly_mean_intensity(timestamps, intensity)        # [24, N]
    vectors = hourly[:, :, None] * direction[None, :, :]          # [24, N, 2]

    if args.mode == "smooth":
        xs, ys, grid_uv = smooth_to_grid(xy, vectors, args.grid_cell, args.sigma, args.min_weight)
        fields = grid_uv                                          # [24, Gy, Gx, 2]
        print(f"interpolated onto a {len(xs)} x {len(ys)} grid ({args.grid_cell:.0f} m cells, "
              f"sigma {args.sigma:.0f} m)")
    else:
        fields = vectors                                          # [24, N, 2]

    mags = np.hypot(fields[..., 0], fields[..., 1])
    ref_mag = float(np.nanpercentile(mags, 99))
    vmax = float(np.nanmax(mags))
    print(f"reference magnitude (99th pct) {ref_mag:.0f}, max {vmax:.0f} veh/h")

    pos = dict(zip(map(int, sensor_ids), map(tuple, xy)))
    segments = [(pos[u], pos[v]) for u, v in zip(edges["source"], edges["target"])
                if u in pos and v in pos and u != v]

    def draw(ax, h, small):
        if args.mode == "smooth":
            draw_smooth(ax, xs, ys, fields[h], vmax, ref_mag, args.arrow_length,
                        segments, xy, args.quiver_step, small)
        else:
            draw_field(ax, xy, fields[h], vmax, ref_mag, args.arrow_length,
                       None if small else segments, small)

    out_dir.mkdir(parents=True, exist_ok=True)
    norm = Normalize(0, vmax)
    cmap = "YlOrRd" if args.mode == "smooth" else "viridis"

    for h in range(24):
        fig = Figure(figsize=(10, 9))
        ax = fig.subplots()
        draw(ax, h, small=False)
        ax.set_title(f"Traffic flow, {h:02d}:00-{h:02d}:59 (mean over all days, {args.mode})")
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8,
                     label="flow magnitude (veh/h)")
        fig.tight_layout()
        fig.savefig(out_dir / f"hour_{h:02d}.png", dpi=args.dpi)
        print(f"hour {h:02d} done")

    fig = Figure(figsize=(24, 17))
    axes = fig.subplots(4, 6)
    for h, ax in enumerate(axes.ravel()):
        draw(ax, h, small=True)
        ax.set_title(f"{h:02d}:00", fontsize=11)
    fig.suptitle(f"Hourly traffic flow vector fields ({args.mode}); colour and arrow length share one scale",
                 fontsize=16)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.6, pad=0.01,
                 label="flow magnitude (veh/h)")
    fig.savefig(out_dir / "overview.png", dpi=max(args.dpi // 2, 100))

    extra = dict(grid_x=xs, grid_y=ys, grid_uv=grid_uv) if args.mode == "smooth" else {}
    np.savez_compressed(out_dir / "vector_fields.npz", sensor_ids=sensor_ids, utm_xy=xy,
                        direction=direction, hourly_intensity=hourly, hourly_vectors=vectors, **extra)
    print(f"saved 24 hourly figures, overview.png and vector_fields.npz to {out_dir}")


if __name__ == "__main__":
    main()