# Дизайн `lab3`: прогнозування часового ряду та детекція аномалій

## Мета

Побудувати `lab3/` як простий, відтворюваний і зрозумілий пайплайн для часових рядів, який:

- прогнозує часовий ряд у режимі `next-step forecast`
- знаходить аномалії через `forecast residuals`
- оцінює якість прогнозу та аномалій окремими метриками
- будує наочні графіки для захисту
- дозволяє коротко проаналізувати `FP` і `FN`

## Погоджені рішення

- Датасет: `NAB`
- Основний ряд: `realKnownCause/nyc_taxi.csv`
- Джерело даних у проєкті:
  - `lab3/dataset/nab/data/realKnownCause/nyc_taxi.csv`
  - `lab3/dataset/nab/labels/combined_windows.json`
- Задача прогнозування: `next-step forecast`
- Базовий baseline: `persistence forecast`
- Нейромережа: невелика `GRU`
- Детекція аномалій: поріг на `forecast residuals`
- Базова стратегія порогу: `95th percentile` на residuals нормальних точок `validation`
- Оцінка forecasting:
  - `MAE`
  - `MAPE`
- Оцінка anomalies:
  - `precision`
  - `recall`
  - `F1`
- Візуалізації:
  - часовий ряд із підсвіченими anomaly windows
  - фактичний ряд проти прогнозу
  - residual score з порогом
- Аналіз помилок:
  - короткий розбір `FP` і `FN` на тестовому відрізку

## Розглянуті підходи

### Варіант 1: `NAB` + `GRU` + `forecast residuals`

Переваги:

- у `NAB` уже є розмічені anomaly windows
- легко обґрунтувати `precision/recall/F1`
- forecasting і anomaly detection природно зв'язуються через одну модель
- сценарій добре пояснюється на захисті

Недоліки:

- forecasting тут радше службовий механізм для аномалій, а не велике load forecasting дослідження

### Варіант 2: `NAB` + autoencoder reconstruction error

Переваги:

- добре лягає на класичну anomaly detection постановку
- не вимагає явного прогнозу як основного сигналу

Недоліки:

- гірше покриває вимогу саме про forecasting pipeline
- важче пояснити зв'язок між прогнозом і виявленням аномалій

### Варіант 3: `UCI Electricity` + forecasting + синтетичні або слабко обґрунтовані аномалії

Переваги:

- дуже природний forecasting кейс
- енергоспоживання легко пояснювати предметно

Недоліки:

- немає готових аномальних міток
- `precision/recall/F1` для аномалій доведеться обґрунтовувати значно слабше

## Чому обрано саме цей дизайн

Для цієї лабораторної найважливіше не лише побудувати модель, а й отримати коректну оцінку аномалій із коротким аналізом `FP/FN`. Через це `NAB` є найкращим компромісом між простотою реалізації та якістю захисту. Вибір `forecast residuals` замість autoencoder робить pipeline ближчим до формулювання завдання: одна модель прогнозує майбутнє значення, а помилка прогнозу використовується як anomaly score.

`nyc_taxi.csv` є хорошим стартовим рядом, бо він достатньо простий для першої реалізації, має виразну часову структуру і вже прив'язаний до відомих anomaly windows у `NAB`.

## Цільовий сценарій використання

1. Користувач кладе `NAB`-дані в `lab3/dataset/nab/`.
2. Команда `prepare-data` читає ряд і labels, очищає дані та формує часові split-и.
3. Команда `baseline` рахує `persistence forecast` і зберігає базові метрики.
4. Команда `train` навчає `GRU` прогнозувати наступне значення.
5. Команда `detect` обчислює residuals, визначає threshold і будує anomaly flags.
6. Команда `evaluate` обчислює forecasting- і anomaly-метрики.
7. Команда `plot` зберігає графіки для звіту й захисту.
8. Команда `run-all` запускає весь пайплайн послідовно.

## Запропонована структура проєкту

```text
lab3/
├── main.py
├── README.md
├── report.md
├── requirements.txt
├── .gitignore
├── dataset/
│   └── nab/
│       ├── data/
│       │   └── realKnownCause/
│       │       └── nyc_taxi.csv
│       └── labels/
│           └── combined_windows.json
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
└── src/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── data.py
    ├── windows.py
    ├── baselines.py
    ├── models.py
    ├── train.py
    ├── anomaly.py
    ├── evaluate.py
    ├── plots.py
    └── console.py
```

## Команди `CLI`

- `prepare-data`
  - читає `nyc_taxi.csv`
  - приводить `timestamp` до datetime
  - сортує ряд за часом
  - додає бінарну `is_anomaly`-розмітку з `combined_windows.json`
  - перетворює anomaly windows у point-wise labels за timestamp-ами
  - виконує `train/val/test split` по часу
  - зберігає підготовлені таблиці та metadata

- `baseline`
  - обчислює `persistence forecast`
  - рахує `MAE` і `MAPE`
  - зберігає baseline predictions і summary

- `train`
  - нормалізує значення за статистиками train-частини
  - формує sliding windows
  - навчає `GRU`
  - використовує `MSELoss`
  - вибирає найкращий checkpoint за найменшим `validation MAE`
  - зберігає найкращий checkpoint і training summary

- `detect`
  - будує прогнози на `val` і `test`
  - перетворює `absolute residual` в anomaly score
  - визначає threshold на validation residuals
  - зберігає anomaly predictions і threshold metadata

- `evaluate`
  - рахує `MAE` і `MAPE` для forecasting
  - рахує `precision`, `recall`, `F1` для anomaly detection
  - збирає короткий список `FP` і `FN`

- `plot`
  - будує ключові графіки для звіту

- `run-all`
  - виконує повний пайплайн в одному запуску

## Дизайн даних

Початковий ряд зберігає:

- `timestamp`
- `value`

Після розмітки й підготовки кожен рядок має містити:

- `timestamp`
- `value`
- `is_anomaly`
- `split`

Після етапу forecasting на відповідних зрізах також зберігатимуться:

- `target`
- `prediction`
- `residual`
- `residual_abs`
- `predicted_anomaly`

Окремо має існувати metadata-файл із:

- обраним NAB series key
- часовими межами split-ів
- параметрами `window_size`, `horizon`
- статистиками нормалізації
- threshold strategy і threshold value

Для першої версії `horizon = 1`.

Канонічні джерела metadata по етапах:

- `artifacts/prepared/prepared_metadata.json`
  - series key, split boundaries, split ratios, `window_size`, `horizon`, labeling rule
- `artifacts/models/training_summary.json`
  - модельні гіперпараметри, train/val history, normalisation stats, best checkpoint info
- `artifacts/anomalies/threshold_summary.json`
  - threshold strategy, percentile, numeric threshold, validation points used

### Перетворення NAB windows у point-wise labels

Для першої версії anomaly evaluation виконується як point-wise бінарна класифікація по timestamp-ах, а не через оригінальний NAB scoring.

Правило розмітки:

- кожна anomaly window має `start` і `end`
- точка отримує `is_anomaly = 1`, якщо `start <= timestamp <= end`
- в іншому випадку `is_anomaly = 0`

Це правило використовується однаково в `prepare-data`, оцінюванні та візуалізаціях.

## Дизайн preprocessing

### Cleaning contract

Для `nyc_taxi.csv` у першій версії застосовується мінімальне очищення:

- відсортувати ряд за `timestamp`
- якщо з'являться дублікати `timestamp`, залишати останній запис
- не виконувати ресемплінг
- не інтерполювати пропуски
- якщо після читання виявлено пропущені значення в `timestamp` або `value`, команда `prepare-data` завершується з помилкою
- якщо часовий крок відрізняється від очікуваних `30 хвилин`, команда `prepare-data` завершується з помилкою

Для поточного завантаженого файлу очікується регулярний ряд без дублікатів і без пропусків.

### Time split

Split виконується тільки по часу, без перемішування. Для першої версії пропонується:

- `train`: перші `60%`
- `val`: наступні `20%`
- `test`: останні `20%`

Це просто реалізувати, легко пояснювати і достатньо для навчального сценарію.

### Правило меж split-ів

Навчальні приклади формуються так:

- `train` windows і цілі повністю лежать у `train`
- `val` прогнози мають target всередині `val`, але можуть використовувати історичний контекст з кінця `train`
- `test` прогнози мають target всередині `test`, але можуть використовувати історичний контекст з кінця `val`

Тобто split визначається за target timestamp, а не за всіма точками всередині вікна. Це дозволяє не втрачати перші прогнози на `val` і `test` та краще відповідає реальному сценарію forecasting, де під час inference доступна попередня історія.

### Правило навчальних прикладів

`GRU` навчається тільки на тих train-прикладах, де цільове значення є нормальним:

- приклад включається в навчання, якщо target timestamp має `is_anomaly = 0`
- приклад виключається з навчання, якщо target timestamp має `is_anomaly = 1`

Історичний контекст у вікні може містити аномальні точки. Ми не відкидаємо такі вікна додатково, якщо сам target є нормальним. Це зберігає простоту preprocessing і не змушує реалізацію складно фільтрувати кожне вікно по всій історії.

### Нормалізація

Нормалізація має обчислюватися тільки на `train` і потім застосовуватися до `val` і `test`. Для першої версії достатньо `z-score standardization`.

### Sliding windows

Вхід до моделі:

- послідовність довжини `window_size`

Ціль:

- наступне значення ряду на один крок вперед

Початково рекомендовано:

- `window_size = 48`

Для `nyc_taxi.csv` з кроком у 30 хвилин це покриває 24 години історії, що є розумним базовим контекстом.

## Дизайн baseline

`Persistence forecast` передбачає, що прогноз на наступний момент дорівнює останньому спостереженому значенню у вікні. Це обов'язковий baseline, бо:

- він дуже простий
- задає нижню межу для нейромережі
- добре підходить для часових рядів із сезонністю та локальною інерційністю

## Дизайн моделі

Для першої версії використовується невелика `GRU`.

Очікувана базова конфігурація:

- `input_size = 1`
- `hidden_size = 64`
- `num_layers = 1`
- `dropout = 0.0`
- один лінійний head до скаляра
- `optimizer = Adam`
- `learning_rate = 1e-3`
- `batch_size = 64`
- `max_epochs = 20`
- `early_stopping_patience = 5`

Чому `GRU`, а не складніший `Transformer`:

- менша реалізаційна складність
- достатньо для одномірного навчального кейсу
- легше стабільно натренувати на локальній машині

## Дизайн anomaly detection

Основний сигнал аномалії:

- `residual_abs = abs(target - prediction)`

Threshold визначається не на всіх validation residuals, а тільки на residuals тих точок `validation`, де:

- `is_anomaly = 0`

Для першої версії береться:

- `95th percentile`

Переваги такого підходу:

- проста реалізація
- легко обґрунтувати в звіті
- threshold не підглядає в `test`
- threshold менше спотворюється вже відомими аномаліями у `validation`

У metadata потрібно явно зберігати:

- threshold strategy
- percentile value
- numeric threshold

## Дизайн оцінювання

### Forecasting

Основні метрики:

- `MAE`
- `MAPE`

`MAPE` треба рахувати обережно. Якщо будуть дуже малі значення, слід використати безпечний знаменник з малим `epsilon`.

Для першої версії forecasting-метрики для `GRU` зберігаються окремо для `val` і `test`, так само як і для baseline. Основним результатом для звіту вважається `test`, а `val` використовується для model selection і sanity-check.

### Anomaly detection

Основні метрики:

- `precision`
- `recall`
- `F1`

Для першої версії оцінювання виконується на тестовому зрізі по точках часу з уже наявною бінарною розміткою `is_anomaly`. Це свідомо point-wise evaluation, а не NAB window-based scoring.

### Аналіз `FP/FN`

Окрім агрегованих метрик, потрібно сформувати короткий summary:

- кількість `FP`
- кількість `FN`
- 3-5 прикладів timestamp-ів або коротких інтервалів, де модель спрацювала хибно

## Контракт артефактів

### `artifacts/prepared/`

- `series.csv`
  - колонки: `timestamp`, `value`, `is_anomaly`, `split`
- `prepared_metadata.json`
  - поля:
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

`prepared_metadata.json` є канонічним metadata-файлом етапу підготовки даних. Наступні артефакти можуть мати власні stage-specific summary файли, але не повинні дублювати базові правила розмітки та split-ів як окреме джерело істини.

### `artifacts/baselines/`

- `baseline_predictions.csv`
  - колонки: `timestamp`, `split`, `target`, `prediction`, `residual`, `residual_abs`, `is_anomaly`
- `baseline_metrics.json`
  - forecasting-метрики baseline на `val` і `test`

### `artifacts/models/`

- `gru_best.pt`
- `training_summary.json`
  - поля:
    - модельні гіперпараметри
    - train/val history
    - best epoch
    - best validation MAE
    - нормалізаційні статистики

### `artifacts/forecasts/`

- `gru_predictions.csv`
  - колонки: `timestamp`, `split`, `target`, `prediction`, `residual`, `residual_abs`, `is_anomaly`

### `artifacts/anomalies/`

- `anomaly_predictions.csv`
  - колонки: `timestamp`, `split`, `target`, `prediction`, `residual_abs`, `is_anomaly`, `predicted_anomaly`
- `threshold_summary.json`
  - поля:
    - `threshold_strategy`
    - `threshold_percentile`
    - `threshold_value`
    - `validation_points_used`

### `artifacts/eval/`

- `forecast_metrics.json`
- `anomaly_metrics.json`
- `error_analysis.json`
  - кількості `TP`, `FP`, `TN`, `FN`
  - кілька прикладів `FP` і `FN`

## Дизайн візуалізацій

Для захисту достатньо трьох основних графіків:

1. Повний тестовий часовий ряд із виділеними true anomalies і predicted anomalies.
2. Фрагмент тестового ряду з фактичним значенням і прогнозом моделі.
3. Residual score over time з горизонтальною лінією threshold.

Конвенція відображення:

- true anomalies показуються як заштриховані часові інтервали
- predicted anomalies показуються як точкові маркери на відповідних timestamp-ах

Графіки мають зберігатися в `artifacts/plots/` у форматі `png`.

## Вимоги до відтворюваності

- фіксований `random seed`
- усі параметри зібрані в `config.py`
- артефакти мають стабільні шляхи
- `report.md` має посилатися на збережені артефакти й метрики, а не на “разові” виводи з консолі

## Поза межами першої версії

- multi-step forecasting
- порівняння `GRU` з `LSTM` або `1D-CNN`
- adaptive або rolling threshold
- автоенкодер на reconstruction error
- кілька NAB-рядів у межах одного run
