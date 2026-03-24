# AnnotationAgent

Автоматическая разметка данных и контроль качества аннотаций. Gemini выполняет zero-shot классификацию батчами, измеряет self-consistency (Cohen's κ), генерирует спецификацию разметки и экспортирует в LabelStudio.

## Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| auto_label | `auto_label(df, task, spec_text=None, batch_size=20)` -> DataFrame | Gemini zero-shot: батчевая разметка + self-consistency κ |
| generate_spec | `generate_spec(df, task, topic_path)` -> str | Gemini пишет annotation_spec.md с классами, примерами, edge cases |
| check_quality | `check_quality(df_labeled, df_human=None)` -> QualityMetrics | Cohen's κ (авто↔человек или self-consistency), дистрибуция, confidence |
| export_to_labelstudio | `export_to_labelstudio(df, output_path, task, text_col)` -> Path | LabelStudio JSON с предразметкой для проверки человеком |
| generate_report_notebook | `generate_report_notebook(df_labeled, topic_path, task)` -> Path | Ноутбук: дистрибуция меток, confidence histogram, low-confidence примеры |

## CLI

```bash
# Разметка + ноутбук
python data_annotation/annotation_agent.py toxic_comment \
  --task "классификация токсичных комментариев"

# С готовой спецификацией
python data_annotation/annotation_agent.py toxic_comment \
  --task "классификация токсичных комментариев" \
  --spec path/to/annotation_spec.md

# Сравнение с человеческой разметкой (Cohen's κ)
python data_annotation/annotation_agent.py toxic_comment \
  --task "классификация токсичных комментариев" \
  --check-quality path/to/human_labels.csv \
  --human-col label

# Без аргументов — показывает все найденные темы
python data_annotation/annotation_agent.py
```

## Пайплайн

```
<topic>/data/clean/combined_clean.parquet   (или raw/combined.parquet)
    |
generate_spec() -> annotation_spec.md      (классы, примеры, edge cases)
    |
auto_label()   -> DataFrame(_label, _confidence, _reason)
    |              + self-consistency κ (второй проход на sample)
check_quality() -> QualityMetrics (κ, agreement%, confidence mean/std)
    |
export_to_labelstudio() -> labelstudio_import.json
    |
generate_report_notebook() -> annotation_report.ipynb
    |
Сохраняет -> <topic>/data/labeled/auto_labeled.parquet
```

## QualityMetrics

```python
{
    'kappa':                   0.847,   # Cohen's κ
    'kappa_interpretation':    'almost perfect',
    'agreement_pct':           91.2,
    'confidence_mean':         0.83,
    'confidence_std':          0.12,
    'low_confidence_count':    47,      # confidence < 0.6
    'low_confidence_pct':      4.7,
    'label_distribution':      {'toxic': 4523, 'non-toxic': 45002},
    'label_distribution_pct':  {'toxic': 9.1, 'non-toxic': 90.9},
    'total_labeled':           49525,
}
```

## Интерпретация κ

| κ | Интерпретация |
|---|---|
| < 0.00 | poor (хуже случайного) |
| 0.00–0.20 | slight |
| 0.20–0.40 | fair |
| 0.40–0.60 | moderate |
| 0.60–0.80 | substantial |
| 0.80–1.00 | almost perfect |

## Структура папок

```
<topic>/                              <- папка темы (та же что в data_collection)
├── data/
│   ├── clean/
│   │   └── combined_clean.parquet    <- вход (от data_quality)
│   └── labeled/
│       ├── auto_labeled.parquet      <- выход
│       └── labelstudio_import.json   <- для импорта в LabelStudio
├── notebooks/
│   └── annotation_report.ipynb       <- дистрибуция, confidence, примеры
└── annotation_spec.md                <- спецификация разметки
```

## Формат LabelStudio

```json
[
  {
    "id": "uuid",
    "data": {"text": "...", "row_index": 0},
    "predictions": [{
      "model_version": "gemini-auto-label",
      "score": 0.92,
      "result": [{
        "type": "choices",
        "value": {"choices": ["toxic"]},
        "from_name": "label",
        "to_name": "text"
      }]
    }]
  }
]
```

## Env vars

```
GEMINI_API_KEY    обязательно — для auto_label, generate_spec
```
