# GO-MO Madrid traffic: subgraph preprocessing

Builds a connected region of well-covered traffic sensors around central Madrid from the
GO-MO 2024 dataset, and saves its readings on a regular 15-minute grid together with the
road-graph edges between those sensors.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put the raw dataset files in `data/` (it is git-ignored):

```
data/
├── traffic_data_2024.csv   # raw readings
├── traffic_sensors.csv     # sensor metadata
└── routes-graph.pkl        # directed route graph between sensors
```

## Run

```bash
python main.py
```

Options:

| Flag             | Default | Meaning                                                        |
|------------------|---------|----------------------------------------------------------------|
| `--n-nodes`      | 500     | number of sensors in the subgraph region                       |
| `--min-coverage` | 0.8     | minimum fraction of valid training readings a sensor must have |
| `--overwrite`    | off     | re-convert the raw CSV to Parquet even if it already exists    |

For example, `python main.py --n-nodes 200 --min-coverage 0.9`.

The first run converts `traffic_data_2024.csv` to `data/traffic_2024.parquet`, which takes a
while. Later runs reuse the Parquet file. This is done for reading/memory efficiency for later steps.

## Pipeline

1. **Convert**: stream the raw CSV into a typed, zstd-compressed Parquet file.
2. **Coverage**: for each sensor, the fraction of 15-minute training slots (Jan–Aug) with a
   valid reading. A reading is valid if it is present, `read_error == "N"`, and its
   intensity was not interpolated.
3. **Select region**: keep sensors with coverage ≥ `--min-coverage`, then grow a connected
   region breadth-first from the sensor nearest Puerta del Sol until it has `--n-nodes` sensors.
4. **Grid**: place the region's valid `traffic_intensity` readings on a regular 15-minute
   grid for all of 2024 (`NaN` where missing).
5. **Remove spikes**: set readings above 3× the sensor's training 99th percentile to `NaN`.
6. **Save** to `data/subgraph/`.

Splits: train Jan–Aug, validation Sep–Oct, test Nov–Dec (`TRAIN_END` and `VAL_END` in
`preprocess.py`).

## Output

| File                         | Contents                                                                      |
|------------------------------|-------------------------------------------------------------------------------|
| `data/subgraph/readings.npz` | `sensor_ids` [N], `timestamps` [T], `intensity` [T, N] float32 (veh/h, `NaN` = missing) |
| `data/subgraph/edges.csv`    | directed edges: `source`, `target` (sensor ids), `distance` (m), `weight`      |
| `data/subgraph/sensors.csv`  | metadata for the selected sensors, indexed by `id`                            |

Load it back with:

```python
import numpy as np
from preprocess import load_subgraph

sensor_ids, timestamps, intensity, edges, sensors = load_subgraph()
valid = ~np.isnan(intensity)  # mask of real readings
```

Column `j` of `intensity` corresponds to `sensor_ids[j]`.
