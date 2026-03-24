#!/usr/bin/env python3
"""
ActiveLearningAgent — умный отбор данных для разметки.

Skills:
  fit(labeled_df)                        -> Pipeline  (обучить модель)
  query(pool_df, strategy, n)            -> list[int] (отобрать n примеров)
  evaluate(test_df, iteration, n)        -> Metrics   (accuracy, F1 macro)
  report(history, output_path)           -> Path      (learning_curve.png)
  run_cycle(labeled_df, pool_df, ...)    -> list[IterationResult]
  compare(labeled_df, pool_df, ...)      -> dict      (entropy vs random)

Стратегии: 'entropy' | 'random'

Usage:
  python active_learning/al_agent.py russian_toxic_comment --compare
  python active_learning/al_agent.py russian_toxic_comment --strategy entropy

  from active_learning.al_agent import ActiveLearningAgent
  agent = ActiveLearningAgent()
  history = agent.run_cycle(labeled_df, pool_df, test_df, strategy='entropy')
  agent.report(history)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent.parent
AGENT_COLS = {"_source", "_dataset_id", "_label", "_confidence", "_reason"}

Strategy  = Literal["entropy", "random"]
ModelName = Literal["logreg", "svm", "rf", "rubert-tiny"]


# ------------------------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------------------------

@dataclass
class Metrics:
    accuracy:  float
    f1:        float
    n_labeled: int
    iteration: int

    def __str__(self) -> str:
        return (f"  iter={self.iteration:>2}  n_labeled={self.n_labeled:>5}"
                f"  accuracy={self.accuracy:.4f}  f1={self.f1:.4f}")


@dataclass
class IterationResult:
    iteration:       int
    n_labeled:       int
    accuracy:        float
    f1:              float
    strategy:        str
    queried_indices: list[int] = field(default_factory=list)


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def find_text_col(df: pd.DataFrame) -> str:
    preferred = ["text", "comment", "comment_text", "review", "sentence",
                 "body", "content", "message", "tweet", "post"]
    user_cols = [c for c in df.columns if c not in AGENT_COLS]
    for name in preferred:
        if name in user_cols:
            return name
    str_cols = [c for c in user_cols if df[c].dtype == object]
    if str_cols:
        return max(str_cols, key=lambda c: df[c].str.len().median())
    raise ValueError(f"No text column found in {list(df.columns)}")


def find_label_col(df: pd.DataFrame) -> str:
    preferred = ["label", "target", "class", "category", "toxic",
                 "_label", "sentiment"]
    user_cols = [c for c in df.columns if c not in (AGENT_COLS - {"_label"})]
    for name in preferred:
        if name in user_cols:
            return name
    for c in user_cols:
        if df[c].nunique() <= 20:
            return c
    raise ValueError(f"No label column found in {list(df.columns)}")


def norm_label(series: pd.Series) -> pd.Series:
    """Normalize float labels: 0.0 → '0', 1.0 → '1'."""
    def _n(v):
        s = str(v).strip()
        return s[:-2] if s.endswith(".0") else s
    return series.map(_n)


# ------------------------------------------------------------------------------
# AGENT
# ------------------------------------------------------------------------------

RUBERT_MODEL = "cointegrated/rubert-tiny2"


class RubertEncoder:
    """Encode Russian texts with rubert-tiny2, cache embeddings by text hash."""

    def __init__(self, model_name: str = RUBERT_MODEL, batch_size: int = 128):
        print(f"  Loading {model_name} ...")
        from transformers import AutoTokenizer, AutoModel
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model     = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}
        self._torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        uncached = [t for t in texts if t not in self._cache]
        for i in range(0, len(uncached), self._batch_size):
            batch = uncached[i : i + self._batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            with torch.no_grad():
                out = self._model(**enc)
            # mean pooling over token dimension
            embs = out.last_hidden_state.mean(dim=1).cpu().numpy()
            for text, emb in zip(batch, embs):
                self._cache[text] = emb
        return np.array([self._cache[t] for t in texts])


class ActiveLearningAgent:

    def __init__(self, model: ModelName = "logreg", random_state: int = 42):
        self.model_name    = model
        self.random_state  = random_state
        self._pipeline: Pipeline | None = None
        self._clf          = None        # used for rubert-tiny (no sklearn pipeline)
        self._encoder: RubertEncoder | None = None

    # -- skill 1: fit --

    def fit(self, labeled_df: pd.DataFrame) -> Pipeline:
        text_col  = find_text_col(labeled_df)
        label_col = find_label_col(labeled_df)
        X_raw = labeled_df[text_col].fillna("").astype(str).tolist()
        y     = norm_label(labeled_df[label_col]).tolist()

        if self.model_name == "rubert-tiny":
            if self._encoder is None:
                self._encoder = RubertEncoder()
            X = self._encoder.encode(X_raw)
            self._clf = LogisticRegression(
                max_iter=1000, random_state=self.random_state,
                C=1.0, class_weight="balanced"
            )
            self._clf.fit(X, y)
            self._pipeline = None
            return self._clf

        n_labeled = len(labeled_df)
        min_df = 1 if n_labeled < 500 else 2 if n_labeled < 5000 else 3
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=50_000, ngram_range=(1, 2),
                sublinear_tf=True, min_df=min_df,
            )),
            ("clf", self._build_clf()),
        ])
        pipeline.fit(X_raw, y)
        self._pipeline = pipeline
        return pipeline

    def _build_clf(self):
        rs = self.random_state
        if self.model_name == "logreg":
            return LogisticRegression(max_iter=1000, random_state=rs, C=1.0,
                                      class_weight="balanced")
        if self.model_name == "svm":
            return CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=rs,
                                                    class_weight="balanced"))
        if self.model_name == "rf":
            return RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1,
                                          class_weight="balanced")
        raise ValueError(f"Unknown model: {self.model_name}")

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.model_name == "rubert-tiny":
            X = self._encoder.encode(texts)
            return self._clf.predict_proba(X)
        return self._pipeline.predict_proba(texts)

    def _predict(self, texts: list[str]) -> np.ndarray:
        if self.model_name == "rubert-tiny":
            X = self._encoder.encode(texts)
            return self._clf.predict(X)
        return self._pipeline.predict(texts)

    # -- skill 2: query --

    def query(self, pool_df: pd.DataFrame, strategy: Strategy = "entropy",
              n: int = 20, candidate_sample: int | None = None) -> list[int]:
        """
        entropy — выбирает примеры с максимальной неопределённостью (Shannon entropy).
        random  — случайный выбор (baseline).
        candidate_sample — для больших пулов с BERT: сначала сэмплируем N кандидатов,
                           затем ищем самых неуверенных среди них (ускорение).
        Возвращает список iloc-индексов в исходном pool_df.
        """
        n = min(n, len(pool_df))

        if strategy == "random":
            rng = np.random.default_rng(self.random_state + len(pool_df))
            return rng.choice(len(pool_df), size=n, replace=False).tolist()

        if self._pipeline is None and self._clf is None:
            raise RuntimeError("Call fit() before query()")

        # для BERT-моделей по умолчанию ограничиваем кандидатов
        if candidate_sample is None and self.model_name == "rubert-tiny":
            candidate_sample = 10_000

        if candidate_sample and len(pool_df) > candidate_sample:
            rng = np.random.default_rng(self.random_state + len(pool_df))
            cand_iloc = rng.choice(len(pool_df), size=candidate_sample, replace=False)
            candidates = pool_df.iloc[cand_iloc]
        else:
            cand_iloc = np.arange(len(pool_df))
            candidates = pool_df

        text_col = find_text_col(candidates)
        X        = candidates[text_col].fillna("").astype(str).tolist()
        proba    = self._predict_proba(X)
        eps      = 1e-10
        scores   = -np.sum(proba * np.log(proba + eps), axis=1)
        top_n    = np.argsort(scores)[::-1][:n].tolist()
        return [int(cand_iloc[i]) for i in top_n]

    # -- skill 3: evaluate --

    def evaluate(self, test_df: pd.DataFrame, iteration: int = 0,
                 n_labeled: int = 0) -> Metrics:
        if self._pipeline is None and self._clf is None:
            raise RuntimeError("Call fit() before evaluate()")

        text_col  = find_text_col(test_df)
        label_col = find_label_col(test_df)

        X      = test_df[text_col].fillna("").astype(str).tolist()
        y_true = norm_label(test_df[label_col]).tolist()
        y_pred = self._predict(X)

        acc = round(float(accuracy_score(y_true, y_pred)), 4)
        f1  = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        return Metrics(accuracy=acc, f1=f1, n_labeled=n_labeled, iteration=iteration)

    # -- skill 4: report --

    def report(self, history: list[IterationResult], output_path: Path | None = None,
               title: str = "Active Learning: Learning Curve") -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        strategies = sorted({r.strategy for r in history})
        palette    = {"entropy": "#2196F3", "random": "#FF5722"}

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for strat in strategies:
            rows = sorted([r for r in history if r.strategy == strat],
                          key=lambda r: r.n_labeled)
            xs = [r.n_labeled for r in rows]
            c  = palette.get(strat, "gray")
            axes[0].plot(xs, [r.accuracy for r in rows], marker="o", color=c,
                         label=strat.capitalize(), linewidth=2)
            axes[1].plot(xs, [r.f1 for r in rows], marker="o", color=c,
                         label=strat.capitalize(), linewidth=2)

        for ax, ylabel in zip(axes, ["Accuracy", "F1 (macro)"]):
            ax.set_title(ylabel)
            ax.set_xlabel("N labeled")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.05)

        plt.tight_layout()

        if output_path is None:
            raise ValueError("output_path is required — pass topic_path / 'notebooks' / 'plots' / 'learning_curve.png'")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [ok] learning_curve.png -> {output_path.relative_to(ROOT)}")
        return output_path

    # -- run_cycle --

    def run_cycle(self, labeled_df: pd.DataFrame, pool_df: pd.DataFrame,
                  test_df: pd.DataFrame, strategy: Strategy = "entropy",
                  n_iterations: int = 5, batch_size: int = 20,
                  verbose: bool = True) -> list[IterationResult]:
        history: list[IterationResult] = []
        labeled = labeled_df.copy().reset_index(drop=True)
        pool    = pool_df.copy().reset_index(drop=True)

        if verbose:
            print(f"\n  Strategy: {strategy}  |  start={len(labeled)}  "
                  f"iterations={n_iterations}  batch={batch_size}")
            print(f"  {'─'*55}")

        for it in range(n_iterations + 1):
            self.fit(labeled)
            metrics = self.evaluate(test_df, iteration=it, n_labeled=len(labeled))
            if verbose:
                print(metrics)

            queried: list[int] = []
            if it < n_iterations and len(pool) > 0:
                queried     = self.query(pool, strategy=strategy, n=batch_size)
                new_samples = pool.iloc[queried]
                labeled     = pd.concat([labeled, new_samples], ignore_index=True)
                pool        = pool.drop(pool.index[queried]).reset_index(drop=True)

            history.append(IterationResult(
                iteration=it, n_labeled=metrics.n_labeled,
                accuracy=metrics.accuracy, f1=metrics.f1,
                strategy=strategy, queried_indices=queried,
            ))

        return history

    # -- compare --

    def compare(self, labeled_df: pd.DataFrame, pool_df: pd.DataFrame,
                test_df: pd.DataFrame, strategies: list[Strategy] | None = None,
                n_iterations: int = 5, batch_size: int = 20,
                output_path: Path | None = None) -> dict[str, list[IterationResult]]:
        if strategies is None:
            strategies = ["entropy", "random"]

        results: dict[str, list[IterationResult]] = {}
        all_history: list[IterationResult] = []

        for strat in strategies:
            print(f"\n{'='*60}")
            print(f"  Running strategy: {strat}")
            print(f"{'='*60}")
            clone = ActiveLearningAgent(model=self.model_name,
                                        random_state=self.random_state)
            h = clone.run_cycle(
                labeled_df=labeled_df.copy(), pool_df=pool_df.copy(),
                test_df=test_df, strategy=strat,
                n_iterations=n_iterations, batch_size=batch_size,
            )
            results[strat] = h
            all_history.extend(h)

        _print_savings(results)
        self.report(all_history, output_path=output_path)
        return results


# ------------------------------------------------------------------------------
# SAVINGS ANALYSIS
# ------------------------------------------------------------------------------

def _print_savings(results: dict[str, list[IterationResult]]) -> None:
    print(f"\n{'='*60}")
    print("  SAVINGS ANALYSIS")
    print(f"{'='*60}")

    if "random" not in results:
        print("  [!] No 'random' baseline to compare against.")
        return

    random_history = results["random"]
    target_f1  = random_history[-1].f1
    n_random   = random_history[-1].n_labeled

    print(f"  Random baseline  n={n_random}  "
          f"acc={random_history[-1].accuracy:.4f}  f1={target_f1:.4f}")
    print()

    for strat, history in results.items():
        if strat == "random":
            continue
        reached = next((r for r in history if r.f1 >= target_f1), None)
        if reached:
            saved = n_random - reached.n_labeled
            print(f"  {strat.capitalize():<12} reaches same F1 at n={reached.n_labeled}"
                  f"  →  saved {saved} examples  ({saved/n_random*100:.1f}%)")
        else:
            print(f"  {strat.capitalize():<12} never reached F1={target_f1:.4f} "
                  f"(final F1={history[-1].f1:.4f})")
    print(f"{'='*60}\n")


# ------------------------------------------------------------------------------
# NOTEBOOK GENERATION
# ------------------------------------------------------------------------------

def generate_experiment_notebook(topic_path: Path, parquet_path: Path, task: str,
                                  n_start: int = 2000, n_iterations: int = 8,
                                  batch_size: int = 500) -> Path:
    try:
        import nbformat
    except ImportError:
        print("  [!] nbformat not installed, skipping notebook generation")
        return Path()

    cells = []
    def md(src):   cells.append(nbformat.v4.new_markdown_cell(src))
    def code(src): cells.append(nbformat.v4.new_code_cell(src))

    md(f"# Active Learning Experiment\n**Task:** {task}  |  "
       f"**Dataset:** `{parquet_path.relative_to(ROOT)}`")

    code(f"""\
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '{str(ROOT)}')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pathlib import Path
from active_learning.al_agent import ActiveLearningAgent, find_label_col, norm_label

df = pd.read_parquet('{str(parquet_path)}')
print(f"Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
""")

    md("## Data Split\nTest 20%, остальное — pool. N_START стратифицированных → labeled.")

    code(f"""\
N_START    = {n_start}
ITERATIONS = {n_iterations}
BATCH_SIZE = {batch_size}

label_col = find_label_col(df)
df['_y'] = norm_label(df[label_col])
print("Label distribution:")
print(df['_y'].value_counts())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['_y'])
labeled_df, pool_df = train_test_split(train_df, train_size=N_START, random_state=42,
                                        stratify=train_df['_y'])
labeled_df = labeled_df.reset_index(drop=True)
pool_df    = pool_df.reset_index(drop=True)

print(f"\\nTrain: {{len(train_df)}}  |  Test: {{len(test_df)}}")
print(f"Labeled start: {{len(labeled_df)}}  |  Pool: {{len(pool_df)}}")
""")

    md("## AL Cycle: Entropy vs Random")

    code(f"""\
agent = ActiveLearningAgent(model='logreg')

results = agent.compare(
    labeled_df=labeled_df,
    pool_df=pool_df,
    test_df=test_df,
    strategies=['entropy', 'random'],
    n_iterations=ITERATIONS,
    batch_size=BATCH_SIZE,
    output_path=Path('{str(topic_path / "notebooks" / "plots" / "learning_curve.png")}'),
)
""")

    md("## Learning Curves")

    code(f"""\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Active Learning: Entropy vs Random', fontsize=13, fontweight='bold')
palette = {{'entropy': '#2196F3', 'random': '#FF5722'}}

for strat, history in results.items():
    rows = sorted(history, key=lambda r: r.n_labeled)
    xs = [r.n_labeled for r in rows]
    c  = palette[strat]
    axes[0].plot(xs, [r.accuracy for r in rows], marker='o', color=c, label=strat.capitalize(), lw=2)
    axes[1].plot(xs, [r.f1       for r in rows], marker='o', color=c, label=strat.capitalize(), lw=2)

for ax, title in zip(axes, ['Accuracy', 'F1 (macro)']):
    ax.set_title(title); ax.set_xlabel('N labeled')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

plt.tight_layout()
plots_dir = '{str(topic_path / "notebooks" / "plots")}'
import os; os.makedirs(plots_dir, exist_ok=True)
plt.savefig(f'{{plots_dir}}/learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()
""")

    md("## Results Table")

    code("""\
rows = [{'strategy': r.strategy, 'n_labeled': r.n_labeled,
          'accuracy': r.accuracy, 'f1': r.f1}
        for h in results.values() for r in h]
pd.DataFrame(rows).pivot_table(
    index='n_labeled', columns='strategy', values=['accuracy', 'f1']
).round(4)
""")

    md("## Savings Analysis")

    code("""\
from active_learning.al_agent import _print_savings
_print_savings(results)
""")

    nb = nbformat.v4.new_notebook(cells=cells)
    nb_path = topic_path / "notebooks" / "al_experiment.ipynb"
    nb_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(nb_path))
    print(f"  [ok] al_experiment.ipynb -> {nb_path.relative_to(ROOT)}")

    try:
        import nbclient
        print("  Executing notebook...")
        client = nbclient.NotebookClient(
            nb, timeout=600, kernel_name="python3",
            resources={"metadata": {"path": str(nb_path.parent)}},
        )
        client.execute()
        nbformat.write(nb, str(nb_path))
        print("  [ok] notebook executed and saved with outputs")
    except Exception as e:
        print(f"  [!] Could not execute notebook: {e}")

    return nb_path


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ActiveLearningAgent — умный отбор данных",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python active_learning/al_agent.py russian_toxic_comment --compare
  python active_learning/al_agent.py russian_toxic_comment --strategy entropy
        """,
    )
    parser.add_argument("topic_dir",     nargs="?", default=None)
    parser.add_argument("--strategy",    default="entropy", choices=["entropy", "random"])
    parser.add_argument("--model",       default="logreg",  choices=["logreg", "svm", "rf", "rubert-tiny"])
    parser.add_argument("--n-start",     type=int, default=None)
    parser.add_argument("--iterations",  type=int, default=None)
    parser.add_argument("--batch-size",  type=int, default=None)
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--compare",     action="store_true", help="entropy vs random")
    parser.add_argument("--no-notebook", action="store_true")
    parser.add_argument("--parquet",     default=None, help="explicit parquet path (overrides auto-detect)")
    parser.add_argument("--label-col",   default=None, help="override label column name")
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split

    # find topic
    if args.topic_dir:
        topic_path = (Path(args.topic_dir) if Path(args.topic_dir).is_absolute()
                      else ROOT / args.topic_dir)
    else:
        topics = sorted(ROOT.glob("*/data/raw/combined.parquet"))
        if not topics:
            print("[!] No topics found.")
            raise SystemExit(1)
        topic_path = topics[0].parent.parent.parent
        print(f"Using topic: {topic_path.name}")

    # find parquet — skip files with only 1 class
    pf = None
    candidates_list = (
        [Path(args.parquet) if Path(args.parquet).is_absolute() else ROOT / args.parquet]
        if args.parquet else [
            topic_path / "data" / "clean"  / "combined_clean.parquet",
            topic_path / "data" / "labeled" / "auto_labeled.parquet",
            topic_path / "data" / "raw"    / "combined.parquet",
        ]
    )
    for candidate in candidates_list:
        if not candidate.exists():
            continue
        _tmp = pd.read_parquet(candidate)
        try:
            n_cls = norm_label(_tmp[find_label_col(_tmp)]).nunique()
        except Exception:
            n_cls = 0
        if n_cls >= 2:
            pf = candidate
            break
        print(f"  [skip] {candidate.relative_to(ROOT)} — only {n_cls} class(es)")

    if pf is None:
        print(f"[!] No multi-class parquet found in {topic_path}")
        raise SystemExit(1)

    print(f"\nLoading: {pf.relative_to(ROOT)}")
    df = pd.read_parquet(pf)
    print(f"Shape: {df.shape}")

    label_col = args.label_col if args.label_col else find_label_col(df)
    print(f"Label column: {label_col}")
    df["_y"]  = norm_label(df[label_col])
    print(f"\nLabel distribution:\n{df['_y'].value_counts().to_string()}")

    # interactive params when defaults not overridden
    def _ask_int(prompt, default):
        try:
            ans = input(f"{prompt} [default={default}]: ").strip()
            return int(ans) if ans else default
        except (ValueError, EOFError):
            return default

    if args.n_start    is None: args.n_start    = _ask_int("  N start",        2000)
    if args.batch_size is None: args.batch_size = _ask_int("  Batch size",       500)
    if args.iterations is None: args.iterations = _ask_int("  Iterations",         8)
    print()

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42,
        stratify=df["_y"] if df["_y"].nunique() >= 2 else None,
    )
    labeled_df, pool_df = train_test_split(
        train_df, train_size=args.n_start, random_state=42,
        stratify=train_df["_y"] if train_df["_y"].nunique() >= 2 else None,
    )
    labeled_df = labeled_df.reset_index(drop=True)
    pool_df    = pool_df.reset_index(drop=True)

    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(f"Labeled start: {len(labeled_df)}  |  Pool: {len(pool_df)}")

    agent    = ActiveLearningAgent(model=args.model)
    png_path = topic_path / "notebooks" / "plots" / "learning_curve.png"

    if args.compare:
        agent.compare(labeled_df=labeled_df, pool_df=pool_df, test_df=test_df,
                      strategies=["entropy", "random"],
                      n_iterations=args.iterations, batch_size=args.batch_size,
                      output_path=png_path)
    else:
        history = agent.run_cycle(labeled_df=labeled_df, pool_df=pool_df, test_df=test_df,
                                  strategy=args.strategy, n_iterations=args.iterations,
                                  batch_size=args.batch_size)
        agent.report(history, output_path=png_path)

    if not args.no_notebook:
        generate_experiment_notebook(
            topic_path=topic_path, parquet_path=pf,
            task=topic_path.name.replace("_", " "),
            n_start=args.n_start, n_iterations=args.iterations, batch_size=args.batch_size,
        )

    print(f"\n[ok] Done. Results -> {topic_path / 'notebooks'}")


if __name__ == "__main__":
    main()