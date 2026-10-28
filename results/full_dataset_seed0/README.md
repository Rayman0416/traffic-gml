# Full dataset, seed 0

`python rq1/compare.py --run --seeds 0 --data-dir data/forecast_dataset_full`, run 27–28 Oct 2026 on one
RTX 3060 Ti (8 GB, WSL). Full build: 4,143 sensors, 21 districts. Copied from `data/forecast_dataset_full/reports/`
and `figures/`.

- Every neural model: early stopping on val MAE (patience 25, min delta 0.01 veh/h, at most 1000 epochs),
  LR 1e-3 × 0.985 per epoch.
- `*_results.csv`: val and test MAE/RMSE per horizon, plus cost (parameters, epochs, time, peak GPU memory).
- `table1_main.tex`, `table2_cost.csv`, `table3_ablation.tex`, `table1_mae_by_horizon.png`: made by `compare.py`.
- `router_*.npz`: which districts each district picked on test, with weights and support.
- Full-city with district attention does not fit 256 origins per pass on 8 GB: it was trained with gradient
  accumulation over smaller chunks (same result, longer time per epoch). Full exchange as well. The chunk size
is only printed in the run log, not saved here.
