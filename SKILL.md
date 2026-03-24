# ML Pipeline Skill

Оркестрация полного ML-пайплайна: сбор данных → чистка → разметка → активное обучение.

Каждый шаг — отдельный агент. Claude запускает их последовательно, ждёт завершения и проверяет результат перед следующим шагом.

## КРИТИЧЕСКИ ВАЖНО: правила оркестрации

Claude **НЕ принимает решения за пользователя**. На каждом HITL checkpoint Claude:
1. Показывает результат агента (вывод, таблицы, метрики)
2. Формулирует конкретный вопрос
3. **ОСТАНАВЛИВАЕТСЯ и ждёт ответа**
4. Только после ответа продолжает

Если агент требует интерактивного ввода (выбор датасетов, выбор стратегии, параметры) — Claude **не пишет ответ за пользователя** и не использует pipe/echo для имитации ввода. Вместо этого Claude запускает агент в неинтерактивном режиме (через Python API), выводит результаты пользователю и ждёт его решения.

---

## Шаги пайплайна

```
STEP 1  data_collection   → <topic>/data/raw/combined.parquet
STEP 2  data_quality       → <topic>/data/clean/combined_clean.parquet
STEP 3  data_annotation    → <topic>/data/labeled/auto_labeled.parquet
STEP 4  active_learning    → <topic>/notebooks/plots/learning_curve.png
```

---

## STEP 1 — Сбор данных (DataCollectionAgent)

Claude запускает поиск, показывает найденные датасеты пользователю и **ждёт выбора**.

```python
# Claude запускает через Python API (не CLI):
from data_collection.data_collection_agent import full_search, rank_with_agent, display
candidates = full_search("<задача>")
datasets = rank_with_agent(candidates, topic="<задача>")
display(datasets)
```

**⛔ HITL CHECKPOINT 1 — Claude останавливается:**
> "Найдено N датасетов. Вот список с рейтингами:
> [таблица]
> Какие скачать? Введи номера (например: 1,3) или 'all'."

После ответа пользователя:
```python
from data_collection.data_collection_agent import generate_download_script, _run_pipeline
import subprocess
selected = [datasets[i] for i in chosen_indices]
script = generate_download_script(selected, output_dir)
subprocess.run([sys.executable, script])
_run_pipeline(output_dir, task)
```

**Проверка после скачивания:**
```bash
python -c "import pandas as pd; df = pd.read_parquet('<topic>/data/raw/combined.parquet'); print(df.shape, df.columns.tolist())"
```

---

## STEP 2 — Чистка данных (DataQualityAgent)

Claude запускает анализ и показывает все 3 стратегии пользователю. **Ждёт выбора.**

```python
from data_quality.data_quality_agent import DataQualityAgent
agent = DataQualityAgent()
df = pd.read_parquet('<topic>/data/raw/combined.parquet')
report = agent.detect_issues(df, label_col='<label_col>')
proposals = agent.suggest_strategies(report, task='<задача>')

# показать таблицы before/after для каждой стратегии:
for i, p in enumerate(proposals, 1):
    df_clean = agent.fix(df, p['strategy'], protected_cols=['<label_col>'])
    print(f"[{i}] {p['name']}: {p['reason']}")
    print(agent.compare(df, df_clean).to_string())
```

**⛔ HITL CHECKPOINT 2 — Claude останавливается:**
> "Gemini предлагает 3 стратегии очистки:
> [1] STRICT — ...
> [2] MEDIUM — ...
> [3] MILD — ...
> [таблицы Before/After]
> Какую стратегию выбрать?"

После ответа пользователя — применяет выбранную стратегию и сохраняет.

---

## STEP 3 — Разметка (AnnotationAgent)

```bash
python data_annotation/annotation_agent.py <topic> \
    --task "<задача>" \
    --rows 300
```

**Что происходит:**
1. Агент генерирует спецификацию разметки через Gemini
2. Размечает данные батчами по 20 (Gemini zero-shot)
3. Для каждого примера: метка + уверенность + обоснование
4. Проводит N-pass consistency check (3 прохода, считает agreement)
5. Сохраняет в `auto_labeled.parquet` и `labelstudio_import.json`

**Проверить результат:**
```bash
python -c "
import pandas as pd
df = pd.read_parquet('<topic>/data/labeled/auto_labeled.parquet')
print('Rows:', len(df))
print('Label dist:', df['_label'].value_counts().to_dict())
print('Confidence mean:', df['_confidence'].mean().round(3))
"
```

**⛔ HITL CHECKPOINT 3 — Claude показывает отчёт и останавливается:**
> "Разметка завершена. Confidence mean=X, kappa=Y.
> Найдено N примеров с confidence < 0.75:
> [примеры с текстом, текущей меткой и уверенностью]
> Хочешь исправить какие-то метки? (укажи номер строки и новую метку, или 'ok' если всё устраивает)"

---

## STEP 4 — Активное обучение (ActiveLearningAgent)

```bash
python active_learning/al_agent.py <topic> \
    --compare \
    --n-start 2000 \
    --iterations 8 \
    --batch-size 500
```

**Что происходит:**
1. Загружает данные (приоритет: clean → labeled → raw)
2. Стратифицированный split: 80% train / 20% test
3. Из train берёт N_START примеров как labeled, остальное — pool
4. Запускает entropy и random стратегии параллельно
5. Каждую итерацию: fit → evaluate → query → expand labeled
6. Строит кривые обучения в `learning_curve.png`
7. Показывает savings analysis: сколько примеров экономит entropy vs random
8. Генерирует ноутбук `al_experiment.ipynb` с результатами

**Проверить результат:**
```bash
ls <topic>/notebooks/plots/learning_curve.png
ls <topic>/notebooks/al_experiment.ipynb
```

**⛔ HITL CHECKPOINT 4 — Claude останавливается:**
> "AL завершён. Вот результаты:
> Entropy: F1=X, достигает финального качества на n=Y примерах
> Random:  F1=Z, финальный n=W
> Экономия: saved K примеров (P%)
> [ссылка на learning_curve.png]
> Результаты выглядят разумно?"

---

## Параметры

| Параметр | Описание | Пример |
|----------|----------|--------|
| `<topic>` | Папка темы (создаётся автоматически) | `russian_toxic_comment` |
| `<задача>` | Описание задачи для Gemini | `"toxic comment classification"` |
| `--label-col` | Колонка с метками в raw данных | `label` |
| `--rows` | Сколько строк размечать (annotation) | `300` |
| `--n-start` | Стартовый labeled pool для AL | `2000` |
| `--iterations` | Число AL итераций | `8` |
| `--batch-size` | Примеров за итерацию AL | `500` |

---

## Пример запуска (russian_toxic_comment)

```bash
# STEP 1
python data_collection/data_collection_agent.py search "toxic comment classification in Russian"

# STEP 2
python data_quality/data_quality_agent.py russian_toxic_comment \
    --task "toxic comment classification" --label-col label

# STEP 3
python data_annotation/annotation_agent.py russian_toxic_comment \
    --task "toxic comment classification" --rows 300

# STEP 4
python active_learning/al_agent.py russian_toxic_comment \
    --compare --n-start 2000 --iterations 8 --batch-size 500
```

---

## Структура папок

```
<topic>/
├── data/
│   ├── raw/
│   │   └── combined.parquet          ← STEP 1 output
│   ├── clean/
│   │   └── combined_clean.parquet    ← STEP 2 output
│   └── labeled/
│       ├── auto_labeled.parquet      ← STEP 3 output
│       └── labelstudio_import.json   ← для ручной разметки
├── notebooks/
│   ├── plots/
│   │   └── learning_curve.png        ← STEP 4 output
│   └── al_experiment.ipynb           ← STEP 4 output
├── annotation_spec.md                ← генерируется STEP 3
└── datasets_found.json               ← генерируется STEP 1
```

---

## HITL checkpoints

| Шаг | Точка проверки |
|-----|---------------|
| STEP 1 | Выбор датасетов из предложенных |
| STEP 2 | Выбор стратегии чистки (строгая / средняя / мягкая) |
| STEP 3 | Просмотр примеров с низким confidence (< 0.75) |
| STEP 4 | Оценка кривых обучения, savings analysis |

---

## Зависимости

```
google-generativeai
pandas, numpy, scikit-learn
huggingface_hub, kaggle
matplotlib
nbformat, nbclient
python-dotenv
```

Переменные окружения:
```bash
GEMINI_API_KEY=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```