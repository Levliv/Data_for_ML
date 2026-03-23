# DataCollectionAgent

Поиск, ранжирование и скачивание датасетов из нескольких источников. Gemini выбирает источники под тему, расширяет поисковые запросы, ранжирует кандидатов, генерирует скрипты скачивания и унификации, запускает EDA.

## Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| fetch_api | `fetch_api(endpoint, params, source)` -> DataFrame | REST API (Zenodo) |
| load_dataset | `load_dataset(query, source, limit, extra_terms)` -> DataFrame | HuggingFace Hub и Kaggle; поиск по нескольким синонимам с дедупликацией |
| fetch_generic | `fetch_generic(source, query)` -> DataFrame | Любой JSON API, предложенный Gemini |
| merge | `merge(sources)` -> DataFrame | pd.concat всех источников |

## CLI

```bash
# Поиск + скачивание (интерактивный выбор датасетов)
python agent.py search "toxic comment"

# Скачать из сохранённого списка без повторного поиска
python agent.py download datasets_found.json

# Превью скачанных данных
python agent.py preview toxic_comment -n 10
```

## Пайплайн search

```
Gemini думает: какие источники подходят для темы
    |
Gemini расширяет запрос: 5-7 синонимов для HuggingFace
    |
HuggingFace (по каждому синониму, дедупликация) + Kaggle + Zenodo + доп. API
    |
Gemini ранжирует: relevance_score >= 4, нерелевантные отбрасываются
    |
Пользователь выбирает: 1,3,5 / all / q
    |
Gemini генерирует download_datasets.py  ->  <topic>/data/raw/<name>.parquet
    |
Gemini генерирует unify_template.py     ->  <name>_unified.parquet + combined.parquet
    |
preview первых N строк combined.parquet
    |
EDA: 5 примеров -> Gemini пишет ноутбук -> exec() на всех данных -> вывод в CLI
    |
config.yaml обновляется: тема, источники, last_search
```

## Структура папок

```
<topic>/                              <- папка темы в корне проекта
├── data/
│   └── raw/
│       ├── dataset1.parquet              <- сырые данные
│       ├── dataset1_unified.parquet      <- единая схема
│       ├── dataset2.parquet
│       ├── dataset2_unified.parquet
│       └── combined.parquet              <- итоговый объединённый датасет
├── notebooks/
│   ├── eda.ipynb                         <- автогенерируется Gemini
│   └── plots/plot_01.png ...             <- графики из EDA
├── download_datasets.py                  <- сгенерирован Gemini
├── unify_template.py                     <- сгенерирован Gemini
├── datasets_found.json                   <- все кандидаты с оценками
└── config.yaml                           <- автообновляется после поиска
```

## Формат данных

Все источники сохраняются в единый Parquet со служебными колонками:
- `_source` — источник (`huggingface`, `kaggle`, `zenodo`, ...)
- `_dataset_id` — идентификатор датасета

Preview и EDA работают только с `combined.parquet`.

## EDA (data_collection/eda.py)

```bash
python data_collection/eda.py toxic_comment "toxic comment classification"
```

1. Берёт 5 строк из `combined.parquet` (без `_source`, `_dataset_id`)
2. Отправляет Gemini: тема + схема + примеры
3. Gemini сам решает что считать и пишет ноутбук
4. Ноутбук выполняется через `exec()` на всех данных
5. Все `print()` и markdown-ячейки выводятся в CLI

## Env vars

```
GEMINI_API_KEY    обязательно — aistudio.google.com
KAGGLE_USERNAME   опционально
KAGGLE_KEY        опционально
KAGGLE_API_TOKEN  опционально (альтернатива username+key)
```
