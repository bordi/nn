# Лабораторна 1: Edge Computer Vision на Food-101

Цей проєкт реалізує повний простий пайплайн для лабораторної роботи:
- класифікація зображень на `Food-101` через transfer learning
- покращення якості через аугментації та fine-tuning
- пояснення рішень моделі через `Grad-CAM`
- оптимізація через експорт у `ONNX`
- порівняння моделі у форматах `PyTorch` та `ONNX` за точністю, розміром і CPU latency

Проєкт спеціально написаний просто і модульно, щоб його було легко читати, запускати локально і пояснювати на захисті.

## Структура

- `main.py` - єдина точка входу з командами CLI
- `src/config.py` - шляхи, константи, device, seed
- `src/data.py` - завантаження `Food-101`, train/val/test split, transforms
- `src/model.py` - модель `MobileNetV3`
- `src/train.py` - baseline + improved training
- `src/evaluate.py` - accuracy, macro-F1, confusion summary
- `src/gradcam.py` - Grad-CAM для правильних і помилкових прикладів
- `src/export_onnx.py` - експорт у ONNX
- `src/benchmark.py` - порівняння `PyTorch` vs `ONNX`
- `artifacts/` - чекпоїнти, ONNX і JSON-підсумки
- `outputs/gradcam/` - згенеровані Grad-CAM зображення

## Підготовка середовища

Створіть віртуальне середовище:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Датасет уже очікується в:

```text
dataset/food-101/
```

Завантажити датасет можна тут:
[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

Важливо:
- папка `dataset/` має лежати локально в корені цієї лабораторної, тобто всередині `lab1/`
- у git датасет додавати не потрібно
- якщо ви ініціалізуєте репозиторій, варто додати `dataset/` у `.gitignore`, щоб не пушити великі файли

Очікувана структура:

```text
lab1/
├── dataset/
│   └── food-101/
│       ├── images/
│       └── meta/
├── src/
├── main.py
└── README.md
```

## Логіка навчання

Навчання складається з двох етапів:

0. `reset`
```bash
find artifacts/checkpoints artifacts/onnx outputs/gradcam -type f \( -name "*.pt" -o -name "*.json" -o -name "*.onnx" -o -name "*.png" \) -delete
find artifacts -maxdepth 1 -type f -name "*.json" -delete
```

1. `baseline`
   Тренується тільки classifier head, backbone заморожений.

2. `improved`
   Додаються аугментації, розморожуються останні блоки backbone, вмикається fine-tuning.

На кожному етапі зберігається найкращий чекпоїнт за `val_loss`.

## Основні команди

### 1. Навчання

```bash
python main.py train
```

Корисний швидкий режим для перевірки коду:

```bash
python main.py train --debug-samples 256 --baseline-epochs 1 --improved-epochs 1
```

Після навчання будуть створені:

- `artifacts/checkpoints/baseline_best.pt`
- `artifacts/checkpoints/improved_best.pt`
- `artifacts/checkpoints/training_summary.json`

### 2. Оцінювання

```bash
python main.py evaluate --checkpoint artifacts/checkpoints/improved_best.pt
```

Результат також збережеться в:

```text
artifacts/evaluation_summary.json
```

### 3. Grad-CAM

```bash
python main.py gradcam --checkpoint artifacts/checkpoints/improved_best.pt
```

За замовчуванням буде згенеровано:

- 5 правильних прикладів
- 5 помилкових прикладів

Ці значення тепер задаються централізовано в:

```text
src/config.py
```

Файли будуть у:

```text
outputs/gradcam/
```

### 4. Експорт у ONNX

```bash
python main.py export-onnx --checkpoint artifacts/checkpoints/improved_best.pt
```

Файл за замовчуванням:

```text
artifacts/onnx/improved_best.onnx
```

### 5. Benchmark

```bash
python main.py benchmark \
  --checkpoint artifacts/checkpoints/improved_best.pt \
  --onnx artifacts/onnx/improved_best.onnx
```

Скрипт:
- рахує `accuracy`
- рахує `macro-F1`
- вимірює розмір файлу
- вимірює `CPU latency`
- друкує Markdown-таблицю для звіту

## Приклад очікуваного порядку запуску

```bash
python main.py train
python main.py evaluate --checkpoint artifacts/checkpoints/baseline_best.pt
python main.py evaluate --checkpoint artifacts/checkpoints/improved_best.pt
python main.py gradcam --checkpoint artifacts/checkpoints/improved_best.pt
python main.py export-onnx --checkpoint artifacts/checkpoints/improved_best.pt
python main.py benchmark --checkpoint artifacts/checkpoints/improved_best.pt --onnx artifacts/onnx/improved_best.onnx
```

## Нотатки

- Для тренування на Mac буде автоматично вибрано `mps`, якщо він доступний.
- Для benchmark latency завжди міряється на `cpu`.
- Якщо повний запуск довгий, спочатку використовуйте `--debug-samples`.
