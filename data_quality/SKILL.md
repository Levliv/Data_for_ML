# DataQualityAgent

Обнаружение и устранение проблем качества данных. Принимает Parquet, возвращает очищенный Parquet в `data/clean/`.

## Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| detect_issues | `detect_issues(df, label_col=None)` -> QualityReport | Пропуски, дубликаты, выбросы (IQR), дисбаланс классов |
| fix | `fix(df, strategy)` -> DataFrame | Чистка по выбранной стратегии |
| compare | `compare(df_before, df_after)` -> DataFrame | Таблица метрик до/после с % изменением |
| explain_with_llm | `explain_with_llm(report, task)` -> str | Gemini объясняет проблемы и рекомендует стратегию |

## Стратегии fix()

```python
agent.fix(df, strategy={
    'missing':    'median',    # median | mean | mode | drop | ffill | constant:<val>
    'duplicates': 'drop',      # drop | keep_first | keep_last | none
    'outliers':   'clip_iqr',  # clip_iqr | clip_zscore | drop | none
})
```

## CLI

```bash
# Все файлы из data/raw/ (по умолчанию)
python agents/data_quality/data_quality_agent.py

# Конкретный файл
python agents/data_quality/data_quality_agent.py data/raw/huggingface/slug/data.parquet

# Кастомные стратегии
python agents/data_quality/data_quality_agent.py --strategy drop --outliers clip_zscore

# С объяснением от Gemini
python agents/data_quality/data_quality_agent.py --explain --task "генерация кода"

# Сохранить в конкретный файл
python agents/data_quality/data_quality_agent.py --output data/clean/result.parquet
```

## QualityReport

```python
{
    'missing':    {'col': {'count': N, 'pct': 42.3}},
    'duplicates': 0,
    'outliers':   {'col': {'count': N, 'pct': 3.1, 'lower_bound': ..., 'upper_bound': ...}},
    'imbalance':  {'col': 'label', 'counts': {...}, 'ratio': 2.5},
    'summary':    {'missing_cells': N, 'missing_cols': N, 'outlier_cells': N, 'outlier_cols': N}
}
```

## Структура данных

```
data/raw/<source>/<slug>/data.parquet      - исходник (не трогаем)
data/clean/<source>/<slug>/data.parquet    - очищенный (сохраняет структуру папок)
```