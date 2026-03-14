# ML for Data — Dataset Agent

Два агента для сбора и контроля качества данных:
- **DataCollectionAgent** — поиск, ранжирование и скачивание датасетов
- **DataQualityAgent** — обнаружение и устранение проблем качества данных

## Структура

```
agents/
  data_collection_agent.py  — сбор данных (CLI + API)
  data_quality_agent.py     — контроль качества (API)
config.yaml                 — конфигурация (автообновляется после поиска)
notebooks/
  eda.ipynb                 — автогенерируется после скачивания
  data_quality.ipynb        — EDA качества данных (3 части + бонус)
data/raw/                   — скачанные датасеты в формате Parquet
requirements.txt
```

---

## Установка

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**Обязательные переменные окружения:**

```bash
export GEMINI_API_KEY=AIza...        # бесплатно: aistudio.google.com/app/apikey
```

**Опциональные:**

```bash
export KAGGLE_API_TOKEN=KGAT_...     # поиск на Kaggle
export TELEGRAM_BOT_TOKEN=...        # уведомления в Telegram (для monitor)
export TELEGRAM_CHAT_ID=...
```

---

## Агент 1 — Сбор данных (CLI)

### Поиск и скачивание

```bash
python agents/data_collection_agent.py search "climate weather"
```

Что произойдёт:
1. Поиск на HuggingFace, Kaggle, Papers with Code, Zenodo, UCI
2. Gemini ранжирует и дедуплицирует кандидатов
3. Выводит список с оценками — выбери номера через запятую, `all` или `q`
4. Gemini генерирует `download_datasets.py`, запускает его
5. Данные сохраняются в `data/raw/<source>/<slug>/data.parquet`
6. Gemini генерирует `notebooks/eda.ipynb`
7. `config.yaml` обновляется автоматически

### Скачать из сохранённого списка (без повторного поиска)

```bash
python agents/data_collection_agent.py download datasets_found.json
```

### Мониторинг новых датасетов

```bash
python agents/data_collection_agent.py monitor "climate weather"
# проверяет каждый час, уведомляет в Telegram при появлении новых
```

### Флаги

```bash
python agents/data_collection_agent.py search "тема" --output-dir data/raw
```

### Выходные файлы

```
config.yaml               — обновляется: тема, источники, рекомендованные лимиты
datasets_found.json       — все найденные датасеты с оценками релевантности
download_datasets.py      — сгенерированный скрипт скачивания

data/raw/
  huggingface/<slug>/data.parquet
  kaggle/<slug>/data.parquet
  zenodo/<record-id>/data.parquet

notebooks/eda.ipynb       — автогенерируется Gemini после скачивания
```

---

## Агент 2 — Качество данных (Python API + ноутбук)

### Запуск ноутбука

```bash
# Через VSCode: открой notebooks/data_quality.ipynb → Run All
# Через браузер:
jupyter notebook notebooks/data_quality.ipynb
```

Ноутбук работает с файлами из `data/raw/` автоматически.

### Использование агента напрямую

```python
import pandas as pd
from agents.data_quality_agent import DataQualityAgent

df = pd.read_parquet('data/raw/huggingface/<slug>/data.parquet')
agent = DataQualityAgent()

# Шаг 1 — обнаружить проблемы
report = agent.detect_issues(df)
print(report)
# {'missing': {...}, 'duplicates': N, 'outliers': {...}, 'imbalance': {...}}

# Шаг 2 — почистить (выбери стратегию)
df_clean = agent.fix(df, strategy={
    'missing':    'median',    # median | mean | mode | drop | ffill | constant:<val>
    'duplicates': 'drop',      # drop | keep_first | keep_last | none
    'outliers':   'clip_iqr',  # clip_iqr | clip_zscore | drop | none
})

# Шаг 3 — сравнить до/после
comparison = agent.compare(df, df_clean)
print(comparison)

# Бонус — Gemini объясняет проблемы и рекомендует стратегию
explanation = agent.explain_with_llm(report, task_description="code generation")
print(explanation)
```

### Что делает каждый скилл

| Скилл | Метод | Что обнаруживает / делает |
|---|---|---|
| detect | `detect_issues(df)` | пропуски, дубликаты, выбросы (IQR), дисбаланс классов |
| fix | `fix(df, strategy)` | чистка по выбранной стратегии |
| compare | `compare(df_before, df_after)` | таблица метрик до/после с % изменением |
| explain | `explain_with_llm(report, task)` | Gemini объясняет проблемы и рекомендует стратегию |

---

## Полный пайплайн

```bash
# 1. Найти и скачать датасеты
python agents/data_collection_agent.py search "programming questions"

# 2. Открыть ноутбук с EDA качества
jupyter notebook notebooks/data_quality.ipynb

# 3. Или запустить агент качества из Python напрямую
python - << 'EOF'
import pandas as pd
from pathlib import Path
from agents.data_quality_agent import DataQualityAgent

df = pd.read_parquet(next(Path('data/raw').rglob('data.parquet')))
agent = DataQualityAgent()
report = agent.detect_issues(df)
print(report)
df_clean = agent.fix(df, strategy={'missing': 'median', 'duplicates': 'drop', 'outliers': 'clip_iqr'})
agent.compare(df, df_clean)
EOF
```
