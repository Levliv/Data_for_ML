# DataCollectionAgent

Поиск, ранжирование и скачивание датасетов. Gemini ранжирует кандидатов, генерирует скрипт скачивания и EDA-ноутбук. `config.yaml` обновляется автоматически после каждого поиска.

## Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| scrape | `scrape(url, selector)` -> DataFrame | Скрапинг веб-страниц через CSS-селектор (UCI, Google Dataset Search) |
| fetch_api | `fetch_api(endpoint, params)` -> DataFrame | REST API (Papers with Code, Zenodo) |
| load_dataset | `load_dataset(query, source='hf'|'kaggle')` -> DataFrame | SDK платформ — HuggingFace Hub и Kaggle |
| merge | `merge(sources: list[DataFrame])` -> DataFrame | pd.concat всех источников |

## CLI

```bash
# Поиск + скачивание (интерактивный выбор)
python agents/data_collection/data_collection_agent.py search "climate weather"

# Скачать из сохранённого списка без повторного поиска
python agents/data_collection/data_collection_agent.py download datasets_found.json
```

## Пайплайн search

```
HuggingFace + Kaggle + Papers with Code + Zenodo + UCI + Google
    |
Gemini: ранжирует, дедуплицирует, выставляет relevance_score 1-10
    |
Пользователь выбирает: 1,3,5 / all / q
    |
Gemini генерирует download_datasets.py
    |
Данные -> data/raw/<source>/<slug>/data.parquet
    |
Gemini генерирует notebooks/eda.ipynb
    |
config.yaml обновляется: тема, источники, лимиты, last_search
```

## Выходные файлы

```
config.yaml                              - автообновляется после поиска
datasets_found.json                      - все кандидаты с оценками
download_datasets.py                     - сгенерированный скрипт
data/raw/<source>/<slug>/data.parquet    - данные в едином формате
notebooks/eda.ipynb                      - автогенерируется Gemini
```

## Формат данных

Все источники сохраняются в единый Parquet со служебными колонками:
- `_source` - источник (`huggingface`, `kaggle`, `zenodo`, ...)
- `_dataset_id` - идентификатор датасета

Для HuggingFace показывается реальный размер файлов в MB/GB + количество строк.