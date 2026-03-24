# AnnotationAgent

Автоматическая разметка данных и контроль качества аннотаций. Gemini выполняет zero-shot классификацию батчами, измеряет self-consistency через N прогонов, генерирует спецификацию разметки и экспортирует в LabelStudio.

## Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| auto_label | `auto_label(df, task, spec_text=None, batch_size=20, consistency_sample=30, consistency_passes=3)` | Gemini zero-shot + N-pass consistency check |
| generate_spec | `generate_spec(df, task, topic_path)` -> str | Gemini пишет annotation_spec.md с классами, примерами, edge cases |
| check_quality | `check_quality(df_labeled, df_human=None)` -> QualityMetrics | Cohen's κ (авто↔человек или self-consistency), дистрибуция, confidence |
| export_to_labelstudio | `export_to_labelstudio(df, output_path)` -> list | LabelStudio JSON с предразметкой Gemini для проверки человеком |
| generate_report_notebook | `generate_report_notebook(df_labeled, metrics, spec_text, topic_path, task)` -> Path | Ноутбук: дистрибуция меток, confidence histogram, low-confidence примеры |

## CLI

```bash
# Разметка (интерактивный запрос числа строк)
python data_annotation/annotation_agent.py russian_toxic_comment \
  --task "toxic comment classification"

# Лимит строк явно
python data_annotation/annotation_agent.py russian_toxic_comment \
  --task "toxic comment classification" --rows 500

# Без генерации спека (быстрее)
python data_annotation/annotation_agent.py russian_toxic_comment \
  --task "toxic comment classification" --no-spec

# Сравнение с человеческой разметкой (Cohen's κ)
python data_annotation/annotation_agent.py russian_toxic_comment \
  --task "toxic comment classification" \
  --check-quality path/to/human_labels.csv \
  --human-col label

# Без аргументов — показывает все найденные темы
python data_annotation/annotation_agent.py
```

## Пайплайн

```
<topic>/data/clean/combined_clean.parquet   (приоритет)
<topic>/data/raw/combined.parquet           (fallback)
    |
    | -- интерактивный выбор числа строк --
    |
generate_spec() -> annotation_spec.md      (классы из датасета + описание)
    |
auto_label()   -> DataFrame(_label, _confidence, _reason)
    |              + N-pass consistency check на sample
    |
    | -- LLM ANNOTATION REPORT (CLI) --
    | -- label distribution + confidence stats --
    |
check_quality() -> QualityMetrics
    |
export_to_labelstudio() -> labelstudio_import.json
    |
generate_report_notebook() -> annotation_report.ipynb
    |
Сохраняет -> <topic>/data/labeled/auto_labeled.parquet
```

## Определение классов

Классы берутся **из данных**, не из спека:
1. Ищет колонку с ≤20 уникальными значениями (кроме `text`, `_source`, `_dataset_id`)
2. Нормализует: `0.0` → `"0"`, `1.0` → `"1"`
3. Если такой колонки нет — Gemini определяет классы по sample

## N-pass Consistency Check

```
  ┌─ Consistency (3 passes, 30 rows)
  │  Row agreement:   84.4%  (mean across rows)
  │  Full consensus:  16/30 rows (53.3%)
  │  Pairwise agree:
  │    pass 1 vs pass 2: 90.0%
  │    pass 1 vs pass 3: 86.7%
  │    pass 2 vs pass 3: 83.3%
  │  Mean pairwise:   86.7%
  │  Cohen's kappa:   0.000  ← 0 due to class imbalance (kappa paradox)
  └─
```

**Примечание**: Cohen's kappa = 0 при высоком agreement — нормально для несбалансированных данных (kappa paradox). Ориентируйтесь на Row agreement и Full consensus.

## LLM Annotation Report (CLI)

```
──────────────────────────────────────────────────
  LLM ANNOTATION REPORT
──────────────────────────────────────────────────
  Total labeled : 100

  Label distribution:
    1                        95  ( 95.0%)  ███████████████████
    0                         5  (  5.0%)  █

  Confidence:
    mean   0.941
    median 0.960
    min    0.720
    max    1.000
    low (<0.7)  3 rows  (3.0%)
──────────────────────────────────────────────────
```

## Формат LabelStudio

```json
[
  {
    "data": {"text": "текст комментария"},
    "predictions": [{
      "model_version": "AnnotationAgent-v1",
      "score": 0.98,
      "result": [{
        "type": "choices",
        "value": {"choices": ["1"]},
        "from_name": "label",
        "to_name": "text",
        "score": 0.98
      }]
    }],
    "annotations": []
  }
]
```

`predictions` — pre-annotation от Gemini, `annotations` — заполняет человек в LabelStudio.

## Структура папок

```
<topic>/                              <- папка темы (та же что в data_collection)
├── data/
│   ├── clean/
│   │   └── combined_clean.parquet    <- вход (от data_quality), приоритет
│   ├── raw/
│   │   └── combined.parquet          <- вход (fallback)
│   └── labeled/
│       ├── auto_labeled.parquet      <- выход: все колонки + _label, _confidence, _reason
│       └── labelstudio_import.json   <- для импорта в LabelStudio
├── notebooks/
│   └── annotation_report.ipynb       <- дистрибуция, confidence, примеры
└── annotation_spec.md                <- спецификация разметки (на русском)
```

## Env vars

```
GEMINI_API_KEY    обязательно
```

## Модель

Берётся из `config.yaml`:
```yaml
model: gemini-2.0-flash
```
При 503 (перегрузка) — автоматический retry с ожиданием 15/30/45/60/75 сек.