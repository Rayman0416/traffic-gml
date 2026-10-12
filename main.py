import argparse
import networkx as nx
import numpy as np
import preprocess
from pathlib import Path
import Visualize
import create_masked_data

def main():
	# Parse CLI arguments
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--n-nodes", type=int, default=500, help="number of sensors in the subgraph region")
	parser.add_argument("--min-coverage", type=float, default=0.8, help="minimum fraction of valid training readings per sensor")
	parser.add_argument("--overwrite", action="store_true", help="re-convert the raw CSV")
	parser.add_argument("--max-bad-fraction", type=float, default=0.5, help="discard time steps where more than this fraction of sensors are NaN or 0")
	parser.add_argument("--max-nan-run", type=int, default=500, help="exclude sensors with more than this many consecutive NaN time steps")
	parser.add_argument("--plot-dir", default="figures", help="where to save the pictures")

	#experimental turn all zeros to NaN to remove them. Not sure if this is good. I think 0 are no good values
	parser.add_argument("--zeros-to-nan", action=argparse.BooleanOptionalAction, default=True, help="convert zero values to NaN during grid creation")
	args = parser.parse_args()


	WantToDoPreProcess = True #There must be a better way than this??
	WantToDoMasks = True


	if WantToDoPreProcess:
		# Convert the raw CSV to Parquet for faster reads/memory efficiency in later steps
		preprocess.convert_readings_to_parquet(overwrite=args.overwrite)
		sensors = preprocess.load_sensors()
		graph = preprocess.load_route_graph()

		coverage = preprocess.sensor_coverage()
		print(f"sensors with readings: {len(coverage):,}; "
			f"with >= {args.min_coverage:.0%} valid training readings: "
			f"{(coverage >= args.min_coverage).sum():,}")

		# Create a subgraph of 500 sensors with good training coverage and save it
		subgraph = preprocess.select_subgraph(graph, sensors, coverage, args.n_nodes, args.min_coverage, max_nan_run=args.max_nan_run)

		timestamps, intensity, _ = preprocess.build_grid(subgraph, zeros_to_nan=args.zeros_to_nan)
		intensity, n_spikes = preprocess.remove_spikes(timestamps, intensity) # Remove heavy outliers
		print(f"removed {n_spikes} spike readings (> 3x sensor's training 99th percentile)")

		n_before = len(timestamps)
		timestamps, intensity, n_outage = preprocess.remove_outage_steps(timestamps, intensity, args.max_bad_fraction)
		print(f"removed {n_outage:,} of {n_before:,} time steps "
			  f"(> {args.max_bad_fraction:.0%} of sensors NaN or 0)")

		valid = ~np.isnan(intensity)

		edges = preprocess.subgraph_edges(graph, subgraph)
		preprocess.save_subgraph(subgraph, timestamps, intensity, edges, sensors)

		# Info about the subgraph region
		und = preprocess.undirected_simple(graph, subgraph)
		train = timestamps < preprocess.TRAIN_END
		test = timestamps >= preprocess.VAL_END
		print(f"subgraph: {len(subgraph)} sensors, {len(edges):,} directed edges, "
			f"{und.number_of_edges():,} undirected edges, connected={nx.is_connected(und)}")
		print(f"grid: {intensity.shape[0]:,} time steps x {intensity.shape[1]} sensors; "
			f"valid readings: all {valid.mean():.1%}, train {valid[train].mean():.1%}, "
			f"test {valid[test].mean():.1%}")
		print(f"intensity (veh/h) median {np.nanmedian(intensity):.0f}, "
			f"99th pct {np.nanpercentile(intensity, 99):.0f}, max {np.nanmax(intensity):.0f}")
		print(f"saved to {preprocess.SUBGRAPH_DIR}")

		plot_dir = Path(args.plot_dir)
		Visualize.make_picture(intensity, timestamps, plot_dir / "grid_cleaned.png", sensor_ids=subgraph, time_bin=1,y_scale=1)

		n_comp = Visualize.draw_graph(sensors.loc[subgraph], edges, plot_dir / "graph.png")
		print(f"pictures saved to {plot_dir}, graph has {n_comp} connected component(s)")

	else:
		print("Skipping preprocessing. Loading pre-processed subgraph...")
		subgraph, timestamps, intensity, edges, sensors = preprocess.load_subgraph()


	if WantToDoMasks:
		print("Creating datasets...")
		stats, groundtruth, randomised, randomised_sine = create_masked_data.create_datasets(overwrite=True)

		plot_dir = Path(args.plot_dir)
		# Plot Groundtruth Normal
		Visualize.make_picture(groundtruth, timestamps, plot_dir / "grid_groundtruth_normal.png", sensor_ids=subgraph, time_bin=1, y_scale=1,)

		# Plot Randomised Normal
		Visualize.make_picture(randomised, timestamps, plot_dir / "grid_randomised_normal.png", sensor_ids=subgraph, time_bin=1, y_scale=1,)

		# Sine randomised
		Visualize.make_picture(randomised_sine, timestamps, plot_dir / "grid_randomised_sine.png", sensor_ids=subgraph, time_bin=1, y_scale=1,)

	else:
		print("Loading pre-existing masked datasets from disk...")
		groundtruth, randomised, randomised_sine = create_masked_data.load_datasets()

if __name__ == "__main__":
    main()