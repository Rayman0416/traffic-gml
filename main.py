# Run the preprocessing pipeline, then load and print the model input features.

# python main.py                       # run the full pipeline, then show the features
# python main.py --skip-preprocess     # only load and show the existing features
# python main.py --from windows        # any other flag is passed to preprocess.run

import argparse
import numpy as np
from preprocess import features, windows
from preprocess import run as pipeline


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--skip-preprocess", action="store_true", help="load the existing features without running the pipeline")
	args, pipeline_args = parser.parse_known_args()

	if not args.skip_preprocess:
		pipeline.main(pipeline_args)

	out_dir = pipeline.parse_args(pipeline_args).out_dir
	f = features.load(out_dir)
	T, N = f["intensity"].shape

	print(f"\nfeatures from {out_dir / 'features.npz'}")
	print(f"  {T:,} timesteps ({f['timestamps'][0]} .. {f['timestamps'][-1]}) x {N:,} sensors, history length {f['history_length']}")
	
	for ch in features.SENSOR_CHANNELS:
		v, m = f[ch], f[f"{ch}_mask"]
		print(f"{ch:9s} {v.shape} {v.dtype}, genuine {m.mean():.2%}, scaled mean {v[m].mean():+.3f}, std {v[m].std():.3f}")
	print(f"time features {f['time_features'].shape}: {', '.join(f['time_feature_names'])}")

	# Gather the inputs of a few training windows, as a model would read them.
	win = windows.load(out_dir)
	origins = win.loc[win["split"] == "train", "origin_index"].to_numpy()[:4]
	x = features.window_features(f, origins)
	print(f"\nbatch of {len(origins)} training windows")
	for k in ("intensity", "intensity_mask", "occupancy", "occupancy_mask", "time_features", "history_index"):
		print(f"{k:15s} {x[k].shape}")

	np.set_printoptions(precision=3, suppress=True)
	print(f"\nfirst window, origin {f['timestamps'][origins[0]]}, first 5 sensors {x['sensor_ids'][:5]}:")
	print(f"intensity [L, 5]:\n{x['intensity'][0, :, :5]}")
	print(f"intensity_mask [L, 5]:\n{x['intensity_mask'][0, :, :5]}")
	print(f"time_features: {x['time_features'][0]}")

if __name__ == "__main__":
	main()
