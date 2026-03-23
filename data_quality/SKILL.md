# DataQualityAgent

Обнаружение и устранение проблем качества данных. Gemini предлагает 3 стратегии (строгая/средняя/мягкая) на основе реального отчёта, пользователь выбирает одну. Генерирует ноутбук с визуализациями и обоснованием.

## Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| detect_issues | `detect_issues(df, label_col=None)` -> QualityReport | Пропуски, дубликаты, выбросы (IQR/z-score), дисбаланс классов |
| fix | `fix(df, strategy)` -> DataFrame | Чистка по выбранной стратегии |
| compare | `compare(df_before, df_after)` -> DataFrame | Таблица Before/After/Delta/Change по всем метрикам |
| suggest_strategies | `suggest_strategies(report, task)` -> list[dict] | Gemini предлагает 3 стратегии под конкретные данные и задачу |
| explain_with_llm | `explain_with_llm(report, task)` -> str | Gemini обосновывает выбор стратегии |
| generate_report_notebook | `generate_report_notebook(df, report, topic_path, task, s1, s2)` -> Path | Ноутбук с визуализациями и сравнением стратегий |

## CLI

```bash
# Запуск на папке темы (находит data/raw/combined.parquet)
python data_quality/data_quality_agent.py toxic_comment

# С описанием задачи и Gemini-обоснованием
python data_quality/data_quality_agent.py toxic_comment \
  --task "классификация токсичных комментариев" \
  --explain

# Без аргументов — показывает все найденные темы
python data_quality/data_quality_agent.py
```

## Пайплайн

```
<topic>/data/raw/combined.parquet
    |
detect_issues() -> QualityReport (пропуски, дубли, выбросы, дисбаланс)
    |
Gemini смотрит на данные -> предлагает 3 стратегии (strict / medium / mild)
    |
Показывает сравнение Before/After для каждой стратегии
    |
Пользователь выбирает: 1 / 2 / 3
    |
[--explain] Gemini обосновывает выбор -> вставляется в ноутбук
    |
generate_report_notebook() -> quality_report.ipynb
    |
Сохраняет очищенные данные -> <topic>/data/clean/combined_clean.parquet
```

## Стратегии fix()

```python
agent.fix(df, strategy={
    'missing':    'median',      # median | mean | mode | drop | ffill | constant:<val>
    'duplicates': 'drop',        # drop | keep_first | keep_last | none
    'outliers':   'clip_iqr',    # clip_iqr | clip_zscore | drop | none
})
```

## Сравнение стратегий (вывод в CLI)

```
                Before   After  Delta   Change
Metric
Rows            541975  541930    -45    -0.0%
Missing values       0       0      0      n/a
Duplicates          45       0    -45  -100.0%
Outliers (IQR)    9586    9586      0    +0.0%
```

## QualityReport

```python
{
    'missing':    {'col': {'count': N, 'pct': 42.3}},
    'duplicates': 45,
    'outliers':   {'col': {'count': N, 'pct': 3.1, 'lower_bound': -1.2, 'upper_bound': 4.5}},
    'imbalance':  {'col': 'label', 'counts': {'0': 45000, '1': 5000}, 'ratio': 9.0},
    'summary':    {'missing_cells': N, 'missing_cols': N, 'outlier_cells': N, 'outlier_cols': N}
}
```

## Структура папок

```
<topic>/                              <- папка темы (та же что в data_collection)
├── data/
│   ├── raw/
│   │   └── combined.parquet          <- вход (от data_collection)
│   └── clean/
│       └── combined_clean.parquet    <- выход
└── notebooks/
    ├── eda.ipynb                      <- от data_collection
    └── quality_report.ipynb           <- Part 1: визуализации проблем
                                          Part 2: сравнение 2 стратегий
                                          Part 3: обоснование Gemini
```

## Env vars

```
GEMINI_API_KEY    обязательно — для suggest_strategies и explain_with_llm
```
