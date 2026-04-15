# Lab 3

Time-series forecasting and anomaly detection pipeline for `NAB/nyc_taxi`.

## Goal

This lab builds a reproducible pipeline that:

1. prepares the `NAB` time series and point-wise anomaly labels
2. trains a small `GRU` for next-step forecasting
3. detects anomalies from forecast residuals
4. evaluates `MAE`, `MAPE`, `precision`, `recall`, and `F1`
5. saves plots and JSON artifacts for the report

## Folder Structure

```text
lab3/
├── main.py
├── README.md
├── report.md
├── requirements.txt
├── dataset/
│   └── nab/
│       ├── data/realKnownCause/nyc_taxi.csv
│       └── labels/combined_windows.json
├── artifacts/
│   ├── prepared/
│   ├── baselines/
│   ├── models/
│   ├── forecasts/
│   ├── anomalies/
│   ├── eval/
│   └── plots/
├── docs/
│   └── superpowers/
├── src/
└── tests/
```

## Dataset Location

- Series: [`dataset/nab/data/realKnownCause/nyc_taxi.csv`](./dataset/nab/data/realKnownCause/nyc_taxi.csv)
- Labels: [`dataset/nab/labels/combined_windows.json`](./dataset/nab/labels/combined_windows.json)

## Command Order

Run all commands from the `lab3/` directory.

1. Prepare the labeled series

   ```bash
   ./.venv/bin/python3 main.py prepare-data
   ```

2. Run the persistence baseline

   ```bash
   ./.venv/bin/python3 main.py baseline
   ```

3. Train the GRU model

   ```bash
   ./.venv/bin/python3 main.py train
   ```

4. Detect anomalies from residuals

   ```bash
   ./.venv/bin/python3 main.py detect
   ```

5. Evaluate forecasting and anomaly metrics

   ```bash
   ./.venv/bin/python3 main.py evaluate
   ```

6. Generate the report plots

   ```bash
   ./.venv/bin/python3 main.py plot
   ```

7. Run the full pipeline

   ```bash
   ./.venv/bin/python3 main.py run-all
   ```

## Artifacts

- `artifacts/prepared/series.csv`
- `artifacts/prepared/prepared_metadata.json`
- `artifacts/baselines/baseline_predictions.csv`
- `artifacts/baselines/baseline_metrics.json`
- `artifacts/models/gru_best.pt`
- `artifacts/models/training_summary.json`
- `artifacts/forecasts/gru_predictions.csv`
- `artifacts/eval/forecast_metrics.json`
- `artifacts/anomalies/anomaly_predictions.csv`
- `artifacts/anomalies/threshold_summary.json`
- `artifacts/eval/anomaly_metrics.json`
- `artifacts/eval/error_analysis.json`
- `artifacts/plots/test_series_anomalies.png`
- `artifacts/plots/test_forecast_zoom.png`
- `artifacts/plots/test_residuals_threshold.png`
