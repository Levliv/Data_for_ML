# Skill: Dataset Search Agent

Агент для поиска, оценки и скачивания датасетов по заданной теме.
Использует Gemini для ранжирования и генерации скриптов скачивания.

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
pip install google-genai huggingface_hub datasets requests beautifulsoup4 json-repair

export GEMINI_API_KEY=AIza...
export KAGGLE_API_TOKEN=KGAT_...   # опционально
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
6. Запускает скрипт → data/<source>/<dataset>/
```

## Выходные файлы

```
datasets_found.json       — все найденные датасеты с оценками
download_datasets.py      — сгенерированный скрипт скачивания
gemini_script_raw.log     — сырой ответ Gemini (для отладки)

data/
  huggingface/<slug>/
  kaggle/<owner>/<slug>/
  zenodo/<record-id>/
  uci/<n>/
```

## Мониторинг

```bash
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
python agent.py monitor "climate weather"
# → проверяет каждый час
# → уведомляет в Telegram если появились новые датасеты
```

## Настройки в коде

```python
TOPIC  = "climate weather"           # тема по умолчанию
MODEL  = "gemini-3.1-flash-lite-preview"
```
