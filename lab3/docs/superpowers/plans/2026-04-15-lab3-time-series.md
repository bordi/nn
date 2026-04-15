# Lab3 Time Series Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Побудувати `lab3/` як відтворюваний пайплайн для `NAB/nyc_taxi`, який готує часовий ряд, навчає `GRU` для `next-step forecast`, детектить аномалії через `forecast residuals`, обчислює `MAE/MAPE/precision/recall/F1` і зберігає графіки та артефакти для звіту.

**Architecture:** `lab3` наслідує стиль `lab2`: одна CLI-точка входу `main.py`, невеликі модулі в `src/`, чіткі артефакти в `artifacts/` і тести для чистих функцій. Пайплайн іде послідовно: `prepare-data -> baseline -> train -> detect -> evaluate -> plot`, а `run-all` просто виконує ці кроки по черзі з fail-fast поведінкою.

**Tech Stack:** Python, `numpy`, `torch`, `matplotlib`, `pytest`, стандартна бібліотека `csv/json/pathlib`

---

### Task 1: Створити каркас `lab3` і базову CLI-оболонку

**Files:**
- Create: `lab3/main.py`
- Create: `lab3/README.md`
- Create: `lab3/report.md`
- Create: `lab3/requirements.txt`
- Create: `lab3/.gitignore`
- Create: `lab3/src/__init__.py`
- Create: `lab3/src/config.py`
- Create: `lab3/src/utils.py`
- Create: `lab3/src/console.py`
- Test: `lab3/tests/test_config.py`

- [ ] **Step 1: Створити дерево директорій**

Створити такі папки:

```text
lab3/
lab3/src/
lab3/tests/
lab3/artifacts/prepared/
lab3/artifacts/baselines/
lab3/artifacts/models/
lab3/artifacts/forecasts/
lab3/artifacts/anomalies/
lab3/artifacts/eval/
lab3/artifacts/plots/
lab3/outputs/
```

- [ ] **Step 2: Додати базові файли проєкту**

Реалізувати:
- `requirements.txt` із мінімальними залежностями:
  - `numpy`
  - `torch`
  - `matplotlib`
  - `pytest`
- `.gitignore` для:
  - `__pycache__/`
  - `.pytest_cache/`
  - `.venv/`
  - усіх файлів у `artifacts/`, крім порожніх `.gitkeep` за потреби
- `src/config.py` з dataclass-конфігом:
  - шляхи до dataset/artifacts
  - `series_key = "realKnownCause/nyc_taxi.csv"`
  - `window_size = 48`
  - `horizon = 1`
  - `train_ratio = 0.6`
  - `val_ratio = 0.2`
  - `hidden_size = 64`
  - `learning_rate = 1e-3`
  - `batch_size = 64`
  - `max_epochs = 20`
  - `early_stopping_patience = 5`
  - `threshold_percentile = 95`
  - `seed = 42`
- `src/utils.py` з helpers для:
  - JSON read/write
  - CSV write
  - directory creation
  - seed setup
- `src/console.py` з компактними рендерами summary для команд

- [ ] **Step 3: Додати skeleton CLI**

У `main.py` створити subcommands:
- `prepare-data`
- `baseline`
- `train`
- `detect`
- `evaluate`
- `plot`
- `run-all`

Кожна команда на старті може викликати заглушку або мінімальну функцію з `src.*`, але help-текст і базовий аргумент-парсинг мають уже працювати.

- [ ] **Step 4: Написати перший smoke test**

У `tests/test_config.py` перевірити:
- `load_config()` повертає очікувані шляхи
- `ensure_directories()` створює всі потрібні artifact directories
- `horizon == 1`
- `threshold_percentile == 95`

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_config.py -v
```

Expected:
- `PASS`

- [ ] **Step 5: Перевірити форму CLI**

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
python3 main.py --help
```

Expected:
- у help-тексті видно всі 7 команд

- [ ] **Step 6: Commit**

```bash
git add lab3
git commit -m "feat: scaffold lab3 cli and config"
```

### Task 2: Реалізувати підготовку даних і point-wise anomaly labels

**Files:**
- Modify: `lab3/src/config.py`
- Modify: `lab3/src/utils.py`
- Create: `lab3/src/data.py`
- Modify: `lab3/main.py`
- Test: `lab3/tests/test_data.py`

- [ ] **Step 1: Написати failing tests для data preparation**

У `tests/test_data.py` покрити:
- читання `nyc_taxi.csv`
- сортування за `timestamp`
- перевірку регулярного кроку `30 хв`
- перетворення NAB windows у point-wise labels за правилом `start <= timestamp <= end`
- split `60/20/20` по часу
- збереження `split` як `train|val|test`

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_data.py -v
```

Expected:
- `FAIL`, бо `src/data.py` ще не реалізований

- [ ] **Step 2: Реалізувати читання та cleaning contract**

У `src/data.py` реалізувати функції:
- `load_series_csv(path)`
- `load_label_windows(path, series_key)`
- `validate_series(records)`
- `label_series_points(records, windows)`

Логіка:
- читати CSV через стандартну бібліотеку
- сортувати за `timestamp`
- при дублях `timestamp` лишати останній запис
- не виконувати ресемплінг
- якщо є пропуски або нерегулярний крок, кидати помилку

- [ ] **Step 3: Реалізувати time split і artifact export**

Додати функції:
- `split_series_by_time(records, train_ratio, val_ratio)`
- `save_prepared_series(...)`
- `save_prepared_metadata(...)`

Вихідні артефакти:
- `artifacts/prepared/series.csv`
- `artifacts/prepared/prepared_metadata.json`

`prepared_metadata.json` має містити:
- `series_key`
- `split_boundaries`
- `num_rows`
- `train_rows`
- `val_rows`
- `test_rows`
- `window_size`
- `horizon`
- `split_ratios`
- `labeling_rule`

- [ ] **Step 4: Підключити команду `prepare-data`**

`python3 main.py prepare-data` має:
- читати dataset
- будувати point-wise labels
- виконувати split
- зберігати prepared artifacts
- друкувати summary з кількістю точок і аномалій по split-ах

- [ ] **Step 5: Перевірити `prepare-data` end-to-end**

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
python3 main.py prepare-data
```

Expected:
- створені `artifacts/prepared/series.csv`
- створений `artifacts/prepared/prepared_metadata.json`
- у консолі видно counts для `train`, `val`, `test`

- [ ] **Step 6: Commit**

```bash
git add lab3
git commit -m "feat: prepare labeled time series data"
```

### Task 3: Реалізувати sliding windows і persistence baseline

**Files:**
- Create: `lab3/src/windows.py`
- Create: `lab3/src/baselines.py`
- Modify: `lab3/main.py`
- Test: `lab3/tests/test_windows.py`
- Test: `lab3/tests/test_baselines.py`

- [ ] **Step 1: Написати failing tests для windows**

У `tests/test_windows.py` покрити:
- формування `train` windows тільки з target у `train`
- формування `val` windows із target у `val` і контекстом із кінця `train`
- формування `test` windows із target у `test` і контекстом із кінця `val`
- виключення train-прикладів, де `target_is_anomaly = 1`
- збереження `timestamp` target-а для кожного прикладу

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_windows.py -v
```

Expected:
- `FAIL`

- [ ] **Step 2: Реалізувати генерацію forecasting windows**

У `src/windows.py` реалізувати:
- `build_train_windows(records, window_size, horizon)`
- `build_eval_windows(records, split_name, window_size, horizon)`
- `windows_to_arrays(window_records)`

Кожен window record має містити:
- `input_values`
- `target`
- `target_timestamp`
- `split`
- `target_is_anomaly`

- [ ] **Step 3: Написати failing tests для baseline**

У `tests/test_baselines.py` перевірити:
- прогноз baseline дорівнює останньому значенню у вікні
- residual і `residual_abs` рахуються коректно
- baseline повертає записи для `val` і `test`

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_baselines.py -v
```

Expected:
- `FAIL`

- [ ] **Step 4: Реалізувати baseline predictions і baseline metrics**

У `src/baselines.py` реалізувати:
- `run_persistence_baseline(window_records)`
- `compute_forecast_metrics(rows)`
- `save_baseline_predictions(...)`
- `save_baseline_metrics(...)`

Артефакти:
- `artifacts/baselines/baseline_predictions.csv`
- `artifacts/baselines/baseline_metrics.json`

- [ ] **Step 5: Підключити команду `baseline`**

Команда має:
- читати prepared artifacts
- будувати `val/test` windows
- рахувати baseline
- зберігати predictions/metrics
- друкувати `MAE/MAPE` для `val` і `test`

- [ ] **Step 6: Перевірити `baseline`**

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
python3 main.py baseline
```

Expected:
- створені `baseline_predictions.csv` і `baseline_metrics.json`
- у консолі видно baseline metrics

- [ ] **Step 7: Commit**

```bash
git add lab3
git commit -m "feat: add time series windows and persistence baseline"
```

### Task 4: Реалізувати `GRU` і train pipeline

**Files:**
- Create: `lab3/src/models.py`
- Create: `lab3/src/train.py`
- Modify: `lab3/src/config.py`
- Modify: `lab3/main.py`
- Test: `lab3/tests/test_models.py`
- Test: `lab3/tests/test_train.py`

- [ ] **Step 1: Написати failing tests для моделі**

У `tests/test_models.py` перевірити:
- `GRUForecastModel` приймає batch форми `[batch, seq_len, 1]`
- forward повертає tensor форми `[batch]` або `[batch, 1]`
- модель детерміністично створюється з фіксованим seed

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_models.py -v
```

Expected:
- `FAIL`

- [ ] **Step 2: Реалізувати модель**

У `src/models.py` реалізувати:
- `GRUForecastModel`
- helper `build_model(config)`

Дефолти:
- `input_size = 1`
- `hidden_size = 64`
- `num_layers = 1`
- `dropout = 0.0`

- [ ] **Step 3: Написати failing tests для training helpers**

У `tests/test_train.py` покрити:
- train-only normalisation statistics
- підготовку batch tensors
- вибір best checkpoint за найменшим `validation MAE`
- early stopping patience counter

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_train.py -v
```

Expected:
- `FAIL`

- [ ] **Step 4: Реалізувати training pipeline**

У `src/train.py` реалізувати:
- `fit_standardizer(train_values)`
- `apply_standardizer(values, mean, std)`
- `build_dataloaders(...)`
- `train_one_epoch(...)`
- `evaluate_model(...)`
- `train_model(...)`
- `save_checkpoint(...)`
- `save_training_summary(...)`

Правила:
- loss: `MSELoss`
- optimizer: `Adam`
- best checkpoint: найменший `validation MAE`
- summary містить hyperparameters, train/val history, best epoch, best validation MAE, normalisation stats

- [ ] **Step 5: Підключити команду `train`**

Додати CLI-override аргументи:
- `--max-epochs`
- `--batch-size`
- `--hidden-size`

Команда має створювати:
- `artifacts/models/gru_best.pt`
- `artifacts/models/training_summary.json`

- [ ] **Step 6: Smoke-перевірка training**

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
python3 main.py train --max-epochs 1
```

Expected:
- команда завершується без помилки
- checkpoint і training summary створені

- [ ] **Step 7: Commit**

```bash
git add lab3
git commit -m "feat: train gru forecasting model"
```

### Task 5: Реалізувати anomaly detection і evaluation

**Files:**
- Create: `lab3/src/anomaly.py`
- Create: `lab3/src/evaluate.py`
- Modify: `lab3/main.py`
- Modify: `lab3/src/console.py`
- Test: `lab3/tests/test_anomaly.py`
- Test: `lab3/tests/test_evaluate.py`

- [ ] **Step 1: Написати failing tests для thresholding**

У `tests/test_anomaly.py` покрити:
- threshold рахується тільки за `validation` точками з `is_anomaly = 0`
- використовується `95th percentile`
- `predicted_anomaly = 1`, якщо `residual_abs >= threshold`

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_anomaly.py -v
```

Expected:
- `FAIL`

- [ ] **Step 2: Реалізувати detect pipeline**

У `src/anomaly.py` реалізувати:
- `load_checkpoint_and_predict(...)`
- `build_prediction_rows(...)`
- `compute_threshold(...)`
- `flag_anomalies(...)`
- `save_forecast_rows(...)`
- `save_anomaly_rows(...)`
- `save_threshold_summary(...)`

Важливо:
- для `val/test` inference обов'язково повторно використати train-time normalisation statistics із `artifacts/models/training_summary.json`
- `gru_predictions.csv` має зберігати колонки контракту зі spec:
  - `timestamp`, `split`, `target`, `prediction`, `residual`, `residual_abs`, `is_anomaly`

Артефакти:
- `artifacts/forecasts/gru_predictions.csv`
- `artifacts/anomalies/anomaly_predictions.csv`
- `artifacts/anomalies/threshold_summary.json`

- [ ] **Step 3: Написати failing tests для evaluation**

У `tests/test_evaluate.py` перевірити:
- `MAE` і safe `MAPE` рахуються коректно
- `precision`, `recall`, `F1` рахуються point-wise
- anomaly `precision/recall/F1` рахуються тільки на `test` split
- error analysis правильно збирає `TP/FP/TN/FN`
- витягуються 3-5 прикладів `FP/FN`

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_evaluate.py -v
```

Expected:
- `FAIL`

- [ ] **Step 4: Реалізувати evaluation**

У `src/evaluate.py` реалізувати:
- `compute_mae(...)`
- `compute_mape(..., epsilon=1e-8)`
- `compute_precision_recall_f1(...)`
- `build_error_analysis(...)`
- `save_forecast_metrics(...)`
- `save_anomaly_metrics(...)`
- `save_error_analysis(...)`

Артефакти:
- `artifacts/eval/forecast_metrics.json`
- `artifacts/eval/anomaly_metrics.json`
- `artifacts/eval/error_analysis.json`

- [ ] **Step 5: Підключити команди `detect` і `evaluate`**

`python3 main.py detect` має:
- генерувати `val/test` прогнози через кращий checkpoint
- обчислювати threshold
- створювати anomaly flags

`python3 main.py evaluate` має:
- читати forecast/anomaly artifacts
- рахувати forecasting-метрики окремо для `val` і `test`
- рахувати anomaly `precision/recall/F1` тільки на `test`
- друкувати компактний summary

- [ ] **Step 6: Перевірити detect/evaluate end-to-end**

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
python3 main.py detect
python3 main.py evaluate
```

Expected:
- створені forecast/anomaly/eval artifacts
- у консолі видно `MAE`, `MAPE`, `precision`, `recall`, `F1`

- [ ] **Step 7: Commit**

```bash
git add lab3
git commit -m "feat: add anomaly detection and evaluation"
```

### Task 6: Реалізувати візуалізації і `run-all`

**Files:**
- Create: `lab3/src/plots.py`
- Modify: `lab3/main.py`
- Modify: `lab3/src/console.py`
- Test: `lab3/tests/test_plots.py`

- [ ] **Step 1: Написати failing tests для plot naming і artifact paths**

У `tests/test_plots.py` перевірити:
- генеруються рівно 3 ключові plot paths
- назви файлів стабільні:
  - `test_series_anomalies.png`
  - `test_forecast_zoom.png`
  - `test_residuals_threshold.png`

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests/test_plots.py -v
```

Expected:
- `FAIL`

- [ ] **Step 2: Реалізувати plot builders**

У `src/plots.py` реалізувати:
- `plot_test_series_with_anomalies(...)`
- `plot_test_forecast_zoom(...)`
- `plot_residuals_with_threshold(...)`
- `save_all_plots(...)`

Конвенція:
- true anomalies: shaded intervals
- predicted anomalies: point markers

- [ ] **Step 3: Підключити `plot`**

`python3 main.py plot` має створювати:
- `artifacts/plots/test_series_anomalies.png`
- `artifacts/plots/test_forecast_zoom.png`
- `artifacts/plots/test_residuals_threshold.png`

- [ ] **Step 4: Реалізувати `run-all`**

`run-all` має:
- послідовно викликати `prepare-data`, `baseline`, `train`, `detect`, `evaluate`, `plot`
- падати одразу, якщо будь-який етап завершився помилкою
- не покладатися на вручну створені проміжні артефакти

- [ ] **Step 5: Перевірити `plot` і `run-all`**

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
python3 main.py plot
python3 main.py run-all --max-epochs 1
```

Expected:
- створені всі 3 PNG-файли
- `run-all` проходить весь пайплайн на smoke-настройках

- [ ] **Step 6: Commit**

```bash
git add lab3
git commit -m "feat: add plots and full lab3 pipeline"
```

### Task 7: Завершити README, report і фінальну верифікацію

**Files:**
- Modify: `lab3/README.md`
- Modify: `lab3/report.md`
- Modify: `lab3/main.py`
- Modify: `lab3/src/console.py`

- [ ] **Step 1: Дописати `README.md`**

Включити:
- коротку постановку лабораторної
- структуру папок
- вимоги до датасету
- порядок запуску:
  - `prepare-data`
  - `baseline`
  - `train`
  - `detect`
  - `evaluate`
  - `plot`
  - `run-all`
- список артефактів

- [ ] **Step 2: Дописати `report.md`**

Структура:
- тема і мета
- опис датасету `NAB/nyc_taxi`
- preprocessing і split
- baseline
- `GRU` forecasting
- residual-based anomaly detection
- `MAE/MAPE/precision/recall/F1`
- короткий аналіз `FP/FN`
- посилання на 3 plot-файли
- висновок

У `report.md` посилатися на збережені JSON-метрики й plot artifacts, а не на разовий console output.

- [ ] **Step 3: Запустити всі unit tests**

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
pytest tests -v
```

Expected:
- `PASS`

- [ ] **Step 4: Запустити фінальний smoke pipeline**

Запуск:

```bash
cd /Users/artemb/Projects/uni/neuronetworks/lab3
python3 main.py run-all --max-epochs 1
```

Expected:
- усі artifact directories заповнені очікуваними файлами
- команда завершується без помилки

- [ ] **Step 5: Зафіксувати фінальний стан**

```bash
git add lab3
git commit -m "docs: finalize lab3 workflow and report"
```
