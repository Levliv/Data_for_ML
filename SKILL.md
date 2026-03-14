# ML for Data — Агенты

---

## Агент 1 — DataCollectionAgent

Поиск, ранжирование и скачивание датасетов. Gemini ранжирует кандидатов, генерирует скрипт скачивания и EDA-ноутбук. `config.yaml` обновляется автоматически после каждого поиска.

### Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| scrape | `scrape(url, selector)` → DataFrame | Скрапинг веб-страниц через CSS-селектор (UCI, Google Dataset Search) |
| fetch_api | `fetch_api(endpoint, params)` → DataFrame | REST API (Papers with Code, Zenodo) |
| load_dataset | `load_dataset(query, source='hf'|'kaggle')` → DataFrame | SDK платформ — HuggingFace Hub и Kaggle |
| merge | `merge(sources: list[DataFrame])` → DataFrame | pd.concat всех источников (outer join) |

### CLI

```bash
# Поиск + скачивание (интерактивный выбор)
python agents/data_collection_agent.py search "climate weather"

# Мониторинг новых датасетов (каждый час, Telegram-уведомления)
python agents/data_collection_agent.py monitor "climate weather"

# Скачать из сохранённого списка без повторного поиска
python agents/data_collection_agent.py download datasets_found.json
```

### Пайплайн search

```
HuggingFace + Kaggle + Papers with Code + Zenodo + UCI + Google
    ↓
Gemini: ранжирует, дедуплицирует, выставляет relevance_score 1–10
    ↓
Пользователь выбирает: 1,3,5 / all / q
    ↓
Gemini генерирует download_datasets.py
    ↓
Данные → data/raw/<source>/<slug>/data.parquet
    ↓
Gemini генерирует notebooks/eda.ipynb
    ↓
config.yaml обновляется: тема, источники, лимиты, last_search
```

### Выходные файлы

```
config.yaml                              — автообновляется после поиска
datasets_found.json                      — все кандидаты с оценками
download_datasets.py                     — сгенерированный скрипт
data/raw/<source>/<slug>/data.parquet    — данные в едином формате
notebooks/eda.ipynb                      — автогенерируется Gemini
```

### Формат данных

Все источники сохраняются в единый Parquet со служебными колонками:
- `_source` — источник (`huggingface`, `kaggle`, `zenodo`, …)
- `_dataset_id` — идентификатор датасета

Для HuggingFace показывается реальный размер файлов в MB/GB + количество строк.

---

## Агент 2 — DataQualityAgent

Обнаружение и устранение проблем качества данных. Принимает Parquet, возвращает очищенный Parquet в `data/clean/`.

### Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| detect_issues | `detect_issues(df, label_col=None)` → QualityReport | Пропуски, дубликаты, выбросы (IQR), дисбаланс классов |
| fix | `fix(df, strategy)` → DataFrame | Чистка по выбранной стратегии |
| compare | `compare(df_before, df_after)` → DataFrame | Таблица метрик до/после с % изменением |
| explain_with_llm | `explain_with_llm(report, task)` → str | Gemini объясняет проблемы и рекомендует стратегию |

### Стратегии fix()

```python
agent.fix(df, strategy={
    'missing':    'median',    # median | mean | mode | drop | ffill | constant:<val>
    'duplicates': 'drop',      # drop | keep_first | keep_last | none
    'outliers':   'clip_iqr',  # clip_iqr | clip_zscore | drop | none
})
```

### CLI

```bash
# Все файлы из data/raw/ (по умолчанию)
python agents/data_quality_agent.py

# Конкретный файл
python agents/data_quality_agent.py data/raw/huggingface/slug/data.parquet

# Кастомные стратегии
python agents/data_quality_agent.py --strategy drop --outliers clip_zscore

# С объяснением от Gemini
python agents/data_quality_agent.py --explain --task "генерация кода"

# Сохранить в конкретный файл
python agents/data_quality_agent.py --output data/clean/result.parquet
```

### QualityReport

```python
{
    'missing':    {'col': {'count': N, 'pct': 42.3}},
    'duplicates': 0,
    'outliers':   {'col': {'count': N, 'pct': 3.1, 'lower_bound': ..., 'upper_bound': ...}},
    'imbalance':  {'col': 'label', 'counts': {...}, 'ratio': 2.5},
    'summary':    {'missing_cells': N, 'missing_cols': N, 'outlier_cells': N, 'outlier_cols': N}
}
```

### Структура данных

```
data/raw/<source>/<slug>/data.parquet      ← исходник (не трогаем)
data/clean/<source>/<slug>/data.parquet    ← очищенный (сохраняет структуру папок)
```

---

## Ноутбуки

| Файл | Содержание |
|---|---|
| `notebooks/eda.ipynb` | Автогенерируется Gemini после скачивания датасетов |
| `notebooks/data_quality.ipynb` | 3 части: Детектив → Хирург → Аргумент + бонус LLM |
