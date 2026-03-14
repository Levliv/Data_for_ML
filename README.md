# Dataset Search Agent

Агент для поиска, оценки и скачивания датасетов по заданной теме.
Использует Gemini для ранжирования и генерации скриптов скачивания.

## Структура

```
agents/data_collection_agent.py   — основной агент (DataCollectionAgent + CLI)
config.yaml                        — конфигурация источников и параметров
notebooks/eda.ipynb                — EDA и визуализации
data/raw/                          — собранные датасеты
```

## Установка

```bash
pip install google-genai huggingface_hub datasets requests beautifulsoup4 pandas pyyaml json-repair

export GEMINI_API_KEY=AIza...
export KAGGLE_API_TOKEN=KGAT_...   # опционально
```

## Команды

```bash
# Найти датасеты по теме
python agents/data_collection_agent.py search "climate weather"

# Мониторить новые датасеты (каждый час)
python agents/data_collection_agent.py monitor "climate weather"

# Скачать из сохранённого списка без поиска
python agents/data_collection_agent.py download datasets_found.json
```

Также работает через старый путь: `python agent.py search "..."`.

## Конфигурация (config.yaml)

```yaml
topic: "climate weather"        # тема по умолчанию
output_dir: "data/raw"          # куда сохранять датасеты
state_file: "known_datasets.json"
monitor_interval: 3600          # секунды между проверками
model: "gemini-3.1-flash-lite-preview"
```

## Как работает search

```
1. Параллельный поиск на всех источниках
        HuggingFace Hub API  → до 50
        Kaggle API           → до 30
        Papers with Code     → варьируется
        Zenodo REST API      → до 20
        UCI ML Repository    → ограничено (JS-рендеринг)
              ↓
2. Gemini ранжирует и дедуплицирует (чанки по 50)
              ↓
3. Показывает топ-N с оценками и описанием
              ↓
4. Пользователь выбирает: 1,3,5 / all / q
              ↓
5. Gemini генерирует download_datasets.py
              ↓
6. Запускает скрипт → data/raw/<source>/<dataset>/
```

## Мониторинг с Telegram

```bash
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
python agents/data_collection_agent.py monitor "climate weather"
```

## Выходные файлы

```
datasets_found.json       — все найденные датасеты с оценками
download_datasets.py      — сгенерированный скрипт скачивания

data/raw/
  huggingface/<slug>/
  kaggle/<owner>/<slug>/
  zenodo/<record-id>/
  uci/<n>/
```
