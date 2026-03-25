Запусти полный ML-пайплайн по теме из $ARGUMENTS. Если тема не указана — спроси пользователя.

Главное правило: **не принимай решения за пользователя** — на каждом HITL checkpoint показывай результаты и жди ответа.

После каждого шага — выполни сгенерированный ноутбук и покажи outputs:
```python
from nb_display import execute_and_show
execute_and_show('<topic>/notebooks/<notebook>.ipynb')
```

---

## Шаги

### STEP 1 — Сбор данных

Запусти поиск через Python API:

```python
import sys; sys.path.insert(0, '.')
from data_collection.data_collection_agent import full_search, rank_with_agent, display
candidates = full_search("<тема>")
datasets = rank_with_agent(candidates, topic="<тема>")
display(datasets)
```

⛔ **ОСТАНОВИСЬ.** Покажи таблицу датасетов пользователю и спроси:
- Какие датасеты скачать?
- Как назвать папку темы (topic)?

После ответа — скачай через collection_agent и объедини:

```python
import sys, subprocess
sys.path.insert(0, '.')
from pathlib import Path
from data_collection.data_collection_agent import generate_download_script, _run_pipeline

selected = [datasets[i] for i in chosen_indices]  # индексы из ответа пользователя
topic_dir = Path('<topic>')          # корень темы, НЕ data/raw
topic_dir.mkdir(parents=True, exist_ok=True)

# generate_download_script возвращает строку с кодом, не путь
script_content = generate_download_script(selected, str(topic_dir))
script_path = topic_dir / 'download.py'
script_path.write_text(script_content)
subprocess.run([sys.executable, str(script_path)])  # ROOT=topic_dir → OUT=topic_dir/data/raw

# _run_pipeline ожидает топик-корень, сам добавляет /data/raw/
_run_pipeline(str(topic_dir), '<тема>')  # → unify → combined.parquet + EDA → eda.ipynb
```

Покажи outputs ноутбука:
```python
from nb_display import execute_and_show
execute_and_show('<topic>/notebooks/eda.ipynb')
```

---

### STEP 2 — Чистка данных

```python
from data_quality.data_quality_agent import DataQualityAgent
import pandas as pd
agent = DataQualityAgent()
df = pd.read_parquet('<topic>/data/raw/combined.parquet')
report = agent.detect_issues(df, label_col='<label_col>')
proposals = agent.suggest_strategies(report, task='<тема>')
for i, p in enumerate(proposals, 1):
    df_c = agent.fix(df, p['strategy'], protected_cols=['<label_col>'])
    print(f"[{i}] {p['name']}: {p.get('reason','')}")
    print(agent.compare(df, df_c).to_string())
```

⛔ **ОСТАНОВИСЬ.** Покажи три стратегии с таблицами Before/After и спроси:
- Какую стратегию выбрать (1/2/3)?

После ответа — примени, сохрани и сгенерируй отчёт:

```python
from pathlib import Path
chosen = proposals[<N-1>]  # N = номер стратегии пользователя
df_clean = agent.fix(df, chosen['strategy'], protected_cols=['<label_col>'])
Path('<topic>/data/clean').mkdir(parents=True, exist_ok=True)
df_clean.to_parquet('<topic>/data/clean/combined_clean.parquet', index=False)
agent.generate_report_notebook(df, report, Path('<topic>').resolve(), '<тема>', proposals[0]['strategy'], proposals[1]['strategy'])  # → quality_report.ipynb
```

Покажи outputs ноутбука:
```python
from nb_display import execute_and_show
execute_and_show('<topic>/notebooks/quality_report.ipynb')
```

---

### STEP 3 — Авторазметка

⛔ **ОСТАНОВИСЬ.** Спроси пользователя:
- Сколько примеров разметить? (по умолчанию 300; для нормального AL рекомендуется 2000–5000)

После ответа — запусти с указанным числом:

```bash
PYTHONPATH=. python data_annotation/annotation_agent.py <topic> \
    --task "<тема>" --rows <N>
```

Покажи outputs ноутбука:
```python
from nb_display import execute_and_show
execute_and_show('<topic>/notebooks/annotation_report.ipynb')
```

⛔ **ОСТАНОВИСЬ.** Покажи отчёт (label distribution, confidence mean, kappa) и спроси:
- Есть примеры с низким confidence — хочешь их исправить?

---

### STEP 4 — Активное обучение

⛔ **ОСТАНОВИСЬ.** Спроси пользователя:
- С какого числа размеченных примеров начать? (n_start; всего в датасете — посмотри через `wc` или shape)

Вычисли batch_size = round(n_start * 0.2), затем запусти:

```bash
PYTHONPATH=. python active_learning/al_agent.py <topic> \
    --parquet <topic>/data/labeled/auto_labeled.parquet \
    --label-col _label \
    --compare --n-start <n_start> --iterations 8 --batch-size <n_start * 0.2>
```

Покажи outputs ноутбука и график:
```python
from nb_display import execute_and_show
execute_and_show('<topic>/notebooks/al_experiment.ipynb')
```

⛔ **ОСТАНОВИСЬ.** Покажи savings analysis и путь к learning_curve.png. Спроси:
- Результаты выглядят разумно? Продолжить?

---

После завершения всех шагов — выведи итоговую сводку: количество строк на каждом этапе, финальные метрики AL.