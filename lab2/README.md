# Лабораторна 2: `RAG`-помічник у `CLI`

У цій папці лежить робочий каркас `RAG`-асистента для `SQuAD 2.0`, а також усі відтворювані артефакти для корпусу, ембеддингів, індексу та оцінювання.

## Перед запуском

1. Поклади локальні файли датасету в `dataset/squad/`:
   - `train-v2.0.json`
   - `dev-v2.0.json`
2. Створи `lab2/.env.local` на основі `lab2/.env.example` і додай туди `OPENAI_API_KEY`, якщо плануєш запускати генерацію відповіді через `OpenAI`.
3. Встанови залежності з `requirements.txt`.

`OPENAI_API_KEY` потрібен для `ask` і `chat` у режимі з генерацією. Для `--no-llm` і для `eval-retrieval` ключ не потрібен.

## Порядок запуску

1. Підготуй корпус і evaluation queries:

```bash
cd lab2
python main.py prepare-corpus
```

Команда створює:
- `artifacts/corpus/documents.jsonl`
- `artifacts/corpus/chunks.jsonl`
- `artifacts/corpus/eval_queries.jsonl`

2. Побудуй індекс:

```bash
python main.py build-index
```

Команда створює:
- `artifacts/embeddings/chunk_embeddings.npy`
- `artifacts/embeddings/embeddings_metadata.json`
- `artifacts/index/faiss.index`
- `artifacts/index/index_metadata.json`

3. Перевір retrieval без LLM:

```bash
python main.py ask --query "What is SQuAD?" --top-k 5 --no-llm
```

4. Запусти повний однокроковий `ask` або інтерактивний `chat`:

```bash
python main.py ask --query "What is SQuAD?" --top-k 5
python main.py chat --top-k 5
```

5. Оціни retrieval:

```bash
python main.py eval-retrieval --top-k 5
```

Команда пише результати в:
- `artifacts/eval/retrieval_metrics.json`
- `artifacts/eval/retrieval_examples.jsonl`

6. Підготуй run для ручного оцінювання відповідей:

```bash
python main.py eval-answers --sample-size 20
```

Очікуваний результат етапу:
- окремий run у `artifacts/runs/`
- таблиця для ручної розмітки з `query`, `gold_reference`, `model_answer`, `used_chunk_ids`, `faithfulness_score`, `helpfulness_score`, `notes`

## Що де лежить

- `dataset/squad/` - локальний `SQuAD 2.0`
- `artifacts/corpus/` - згенерований корпус і evaluation queries
- `artifacts/embeddings/` - матриця ембеддингів і metadata
- `artifacts/index/` - `FAISS`-індекс і metadata
- `artifacts/eval/` - retrieval metrics і приклади помилок
- `artifacts/runs/` - run-артефакти для ручної оцінки відповідей
- `outputs/demos/` - демонстраційні виводи для захисту

## Коротка перевірка

```bash
python3 -m pytest tests/test_config.py -v
python3 main.py --help
```

Якщо бачиш зміну моделі ембеддингів або очищаєш `artifacts/`, повтори `prepare-corpus` і `build-index` перед `ask`, `chat` і `eval-retrieval`.
