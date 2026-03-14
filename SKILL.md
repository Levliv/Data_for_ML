# Skill: Dataset Search Agent

Агент для поиска, оценки, скачивания датасетов и автоматического EDA.
Использует Gemini для ранжирования, генерации скриптов скачивания, обновления конфига и генерации EDA-ноутбука.

## Команды

```bash
# Найти датасеты по теме
python agent.py search "climate weather"

# Мониторить новые датасеты (каждый час)
python agent.py monitor "climate weather"

# Скачать из сохранённого списка без поиска
python agent.py download datasets_found.json
```

## Установка

```bash
pip install -r requirements.txt

export GEMINI_API_KEY=AIza...
export KAGGLE_API_TOKEN=KGAT_...   # опционально
export TELEGRAM_BOT_TOKEN=...      # опционально
export TELEGRAM_CHAT_ID=...        # опционально
```

## Как работает search

```
1. Поиск на всех источниках
        HuggingFace Hub API  → до 50  (размер в MB/GB + кол-во строк)
        Kaggle API           → до 30
        Papers with Code     → варьируется
        Zenodo REST API      → до 20
        UCI ML Repository    → ограничено (JS-рендеринг)
        Google Dataset Search → JSON-LD скрапинг
              ↓
2. Gemini ранжирует и дедуплицирует (чанки по 50)
   Все кандидаты сохраняются — нерелевантные получают score 1–3
              ↓
3. Показывает датасеты с оценками, размером, лицензией
              ↓
4. Пользователь выбирает: 1,3,5 / all / q
              ↓
5. Gemini генерирует download_datasets.py
   Все датасеты сохраняются в едином формате DataFrame → data.parquet
              ↓
6. После скачивания Gemini генерирует notebooks/eda.ipynb
              ↓
7. config.yaml обновляется автоматически на основе результатов поиска
```

## Архитектура агента

### DataCollectionAgent

| Skill | Метод | Описание |
|---|---|---|
| 1 | `scrape(url, selector)` | Скрапинг веб-страниц (UCI, Google Dataset Search) |
| 2 | `fetch_api(endpoint, params)` | REST API (Papers with Code, Zenodo) |
| 3 | `load_dataset(query, source)` | SDK платформ (HuggingFace Hub, Kaggle) |
| 4 | `merge(sources)` | pd.concat всех источников (outer join) |

## Выходные файлы

```
config.yaml               — автообновляется после каждого поиска
datasets_found.json       — все найденные датасеты с оценками
download_datasets.py      — сгенерированный скрипт скачивания

data/raw/
  huggingface/<slug>/data.parquet
  kaggle/<slug>/data.parquet
  zenodo/<record-id>/data.parquet
  uci/<slug>/data.parquet

notebooks/
  eda.ipynb               — автогенерируется Gemini после скачивания
```

## Единый формат данных

Все датасеты сохраняются в Parquet с двумя служебными колонками:
- `_source` — источник (`huggingface`, `kaggle`, `zenodo`, ...)
- `_dataset_id` — идентификатор датасета

## config.yaml

Автоматически обновляется Gemini после каждого поиска:

```yaml
topic: "climate weather"
output_dir: "data/raw"
state_file: "known_datasets.json"
monitor_interval: 3600
model: "gemini-3.1-flash-lite-preview"

# автозаполняется после поиска:
last_search: "2026-03-14T21:30:00"
avg_relevance_score: 7.4
top_ml_tasks: [forecasting, regression, ...]
sources_stats: {huggingface: 42, zenodo: 11, ...}
preferred_sources: [huggingface, kaggle, zenodo]
search_limits: {huggingface: 50, kaggle: 30, ...}
```

## Мониторинг

```bash
python agent.py monitor "climate weather"
# → проверяет каждый час
# → уведомляет в Telegram если появились новые датасеты
# → обновляет datasets_found.json и config.yaml
```

## Размер датасетов HuggingFace

Показывается реальный размер файлов из API + количество строк:
```
huggingface | 2.3 MB | 10K<n<100K | cc-by-4.0
```
