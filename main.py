import argparse
import networkx as nx
import numpy as np
import preprocess


def main():
	# Parse CLI arguments
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--n-nodes", type=int, default=500, help="number of sensors in the subgraph region")
	parser.add_argument("--min-coverage", type=float, default=0.8, help="minimum fraction of valid training readings per sensor")
	parser.add_argument("--overwrite", action="store_true", help="re-convert the raw CSV")
	args = parser.parse_args()

	# Convert the raw CSV to Parquet for faster reads/memory efficiency in later steps
	preprocess.convert_readings_to_parquet(overwrite=args.overwrite)
	sensors = preprocess.load_sensors()
	graph = preprocess.load_route_graph()

	coverage = preprocess.sensor_coverage()
	print(f"sensors with readings: {len(coverage):,}; "
		f"with >= {args.min_coverage:.0%} valid training readings: "
		f"{(coverage >= args.min_coverage).sum():,}")

	# Create a subgraph of 500 sensors with good training coverage and save it
	subgraph = preprocess.select_subgraph(graph, sensors, coverage, args.n_nodes, args.min_coverage)
	timestamps, intensity, _ = preprocess.build_grid(subgraph)
	intensity, n_spikes = preprocess.remove_spikes(timestamps, intensity) # Remove heavy outliers
	valid = ~np.isnan(intensity)
	print(f"removed {n_spikes} spike readings (> 3x sensor's training 99th percentile)")
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


if __name__ == "__main__":
    main()
