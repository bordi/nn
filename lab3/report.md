# Lab 3 Report

## Theme and Goal

This lab implements a reproducible `NAB/nyc_taxi` pipeline for next-step time-series forecasting and anomaly detection. The goal is to forecast the series with a small `GRU`, turn forecast residuals into anomaly scores, and report both forecasting and anomaly-detection quality using saved artifacts.

## Dataset

The experiment uses the `NAB` dataset, specifically `realKnownCause/nyc_taxi.csv`, together with the official anomaly windows from `combined_windows.json`. The prepared input and labels are saved in [`artifacts/prepared/series.csv`](./artifacts/prepared/series.csv) and documented in [`artifacts/prepared/prepared_metadata.json`](./artifacts/prepared/prepared_metadata.json).

The prepared split is time-based:

- train: 6,192 rows
- val: 2,064 rows
- test: 2,064 rows

The metadata file records the boundaries, ratios, and labeling rule `start <= timestamp <= end`.

## Preprocessing and Split

Preprocessing follows a minimal cleaning contract:

- sort by `timestamp`
- keep the last record if duplicate timestamps appear
- do not resample or interpolate
- fail if timestamps or values are missing
- fail if the cadence differs from 30 minutes

This is captured in the prepared metadata and the saved series file. The split ratios are 60/20/20 by time, with no shuffling.

## Baseline

The baseline is a persistence forecast that predicts the next value as the last observed value in the input window. Its saved metrics are in [`artifacts/baselines/baseline_metrics.json`](./artifacts/baselines/baseline_metrics.json).

- val: `MAE=1311.4375`, `MAPE=11.6765`
- test: `MAE=1190.4797`, `MAPE=12.1645`

The baseline remains stronger than the one-epoch GRU run in this verification pass.

## GRU Forecasting

The GRU training summary is saved in [`artifacts/models/training_summary.json`](./artifacts/models/training_summary.json). In the required verification run with `--max-epochs 1`, the best checkpoint was found at epoch 1:

- best validation MAE, normalized: `0.270772`
- best validation MAE, original scale: `1850.753659`
- normalization stats: mean `15355.137274`, std `6835.095191`

The forecast evaluation is saved in [`artifacts/eval/forecast_metrics.json`](./artifacts/eval/forecast_metrics.json):

- val: `MAE=1850.7537`, `MAPE=16.3162`
- test: `MAE=1742.0928`, `MAPE=125.0912`

With only one epoch, the model underfits and does not beat the persistence baseline.

## Residual-Based Anomaly Detection

Anomaly detection uses `abs(target - prediction)` as the anomaly score and sets the threshold from the 95th percentile of normal validation residuals. The saved threshold metadata is in [`artifacts/anomalies/threshold_summary.json`](./artifacts/anomalies/threshold_summary.json):

- strategy: `validation_normal_residual_abs_percentile`
- percentile: `95`
- threshold value: `4862.170330`
- validation points used: `1857`

The anomaly evaluation is saved in [`artifacts/eval/anomaly_metrics.json`](./artifacts/eval/anomaly_metrics.json):

- precision: `0.1746`
- recall: `0.0177`
- F1: `0.0322`
- counts on test: `tp=11`, `fp=52`, `tn=1391`, `fn=610`

## FP/FN Analysis

The saved error analysis in [`artifacts/eval/error_analysis.json`](./artifacts/eval/error_analysis.json) shows two clear patterns:

- False positives cluster around `2014-12-20 17:30:00` to `2014-12-20 18:30:00`, where the model underpredicts a sharp rise in demand even though the timestamps are not labeled anomalous.
- False negatives cluster around `2014-12-23 11:30:00` to `2014-12-23 12:30:00`, where the series is anomalous but the residual stays below the threshold, so the detector misses the event.

This suggests the threshold is conservative and the short GRU training run smooths over abrupt local changes.

## Plots

The report uses three saved plot artifacts:

- [`artifacts/plots/test_series_anomalies.png`](./artifacts/plots/test_series_anomalies.png)
- [`artifacts/plots/test_forecast_zoom.png`](./artifacts/plots/test_forecast_zoom.png)
- [`artifacts/plots/test_residuals_threshold.png`](./artifacts/plots/test_residuals_threshold.png)

These plots show the test series with anomaly windows, the forecast vs. target zoom, and the residuals with the detection threshold.

## Conclusion

The pipeline is fully reproducible and saves all intermediate artifacts needed for inspection. The persistence baseline is still the strongest forecasting reference in the one-epoch verification run, while the GRU + residual-threshold detector demonstrates the complete forecasting-to-anomaly workflow. The main limitation is model quality under a very short training budget, which explains the low anomaly recall and the concentrated FP/FN patterns.
