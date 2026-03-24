# ActiveLearningAgent

Умный отбор данных для разметки. Обучает базовую модель (TF-IDF + LogReg), выбирает наиболее информативные примеры из пула, измеряет качество на каждой итерации, строит кривые обучения.

## Скиллы

| Скилл | Сигнатура | Описание |
|---|---|---|
| fit | `fit(labeled_df)` → Pipeline | TF-IDF + классификатор на размеченных данных |
| query | `query(pool_df, strategy, n)` → list[int] | Выбрать n информативных примеров из пула |
| evaluate | `evaluate(test_df, iteration, n_labeled)` → Metrics | accuracy + F1 на тест-сете |
| report | `report(history, output_path)` → Path | График quality vs n_labeled → PNG |
| run_cycle | `run_cycle(labeled_df, pool_df, test_df, ...)` → list[IterationResult] | Полный AL-цикл |
| compare | `compare(labeled_df, pool_df, test_df, strategies)` → dict | Сравнить несколько стратегий |

## Стратегии

| Стратегия | Принцип | Когда использовать |
|---|---|---|
| `entropy` | Максимальная энтропия предсказания | Основная стратегия AL |
| `margin` | Минимальная разница между топ-2 вероятностями | Альтернатива entropy |
| `random` | Случайный выбор | Baseline для сравнения |

## CLI

```bash
# Один запуск (entropy)
python active_learning/al_agent.py russian_toxic_comment \
    --strategy entropy --n-start 50 --iterations 5 --batch-size 20

# Сравнение стратегий (entropy vs margin vs random)
python active_learning/al_agent.py russian_toxic_comment --compare

# Другая модель
python active_learning/al_agent.py russian_toxic_comment --compare --model svm

# Без генерации ноутбука
python active_learning/al_agent.py russian_toxic_comment --compare --no-notebook
```

## Пайплайн

```
<topic>/data/clean/combined_clean.parquet   (приоритет)
<topic>/data/labeled/auto_labeled.parquet   (второй)
<topic>/data/raw/combined.parquet           (fallback)
    |
    | -- train/test split (80/20, stratified) --
    |
    | labeled_df (N_START=50)  +  pool_df (остаток train)
    |
    | -- AL цикл --
    |
    | iter 0: fit → evaluate → query(strategy) → move batch to labeled
    | iter 1: fit → evaluate → query → move
    | ...
    | iter N: fit → evaluate (финальная модель)
    |
compare() → savings analysis → learning_curve.png
    |
generate_experiment_notebook() → al_experiment.ipynb
```

## Пример вывода

```
  Strategy: entropy  |  start=50  iterations=5  batch=20
  ───────────────────────────────────────────────────────
  iter= 0  n_labeled=  50  accuracy=0.9823  f1=0.9731
  iter= 1  n_labeled=  70  accuracy=0.9851  f1=0.9764
  iter= 2  n_labeled=  90  accuracy=0.9867  f1=0.9789
  iter= 3  n_labeled= 110  accuracy=0.9874  f1=0.9801
  iter= 4  n_labeled= 130  accuracy=0.9881  f1=0.9812
  iter= 5  n_labeled= 150  accuracy=0.9885  f1=0.9818

============================================================
  SAVINGS ANALYSIS
============================================================
  Random baseline  n=150  acc=0.9871  f1=0.9798
  Entropy      reaches same F1 at n=90  →  saved 60 examples (40.0%)
  Margin       reaches same F1 at n=110 →  saved 40 examples (26.7%)
============================================================
```

## IterationResult

```python
@dataclass
class IterationResult:
    iteration:       int        # номер итерации (0 = старт)
    n_labeled:       int        # размер labeled set
    accuracy:        float
    f1:              float
    strategy:        str        # 'entropy' | 'margin' | 'random'
    queried_indices: list[int]  # iloc-индексы выбранных из пула
```

## Модели

| Модель | Флаг | Описание |
|---|---|---|
| LogisticRegression | `--model logreg` | По умолчанию, быстрая |
| LinearSVC | `--model svm` | Чуть медленнее, часто точнее |
| RandomForest | `--model rf` | Медленно, хорошо на нелинейных |

## Структура папок

```
<topic>/
└── notebooks/
    ├── learning_curve.png      <- кривые обучения (все стратегии)
    └── al_experiment.ipynb     <- полный эксперимент
```

## Зависимости

```
scikit-learn
numpy
pandas
matplotlib
nbformat
```