#!/usr/bin/env python3
"""
AnnotationAgent - Auto-labeling and annotation quality control.

Skills:
  auto_label(df, task, spec_text)   -> DataFrame with _label, _confidence, _reason
  generate_spec(df, task)           -> annotation_spec.md
  check_quality(df_labeled, df_human) -> QualityMetrics dict
  export_to_labelstudio(df)         -> labelstudio_import.json

Folder structure (mirrors data_collection / data_quality):
  <topic>/
  ├── data/
  │   └── labeled/
  │       ├── auto_labeled.parquet
  │       └── labelstudio_import.json
  ├── notebooks/
  │   └── annotation_report.ipynb
  └── annotation_spec.md

Usage:
  python data_annotation/annotation_agent.py toxic_comment \
      --task "toxic comment classification"

  python data_annotation/annotation_agent.py toxic_comment \
      --task "toxic comment classification" \
      --check-quality path/to/human_labels.csv \
      --human-col label

  from data_annotation.annotation_agent import AnnotationAgent
  agent = AnnotationAgent()
  df_labeled = agent.auto_label(df, task="sentiment classification")
  spec = agent.generate_spec(df, task="sentiment classification", topic_path=Path("."))
  metrics = agent.check_quality(df_labeled)
  agent.export_to_labelstudio(df_labeled, output_path=Path("labelstudio_import.json"))
"""

from __future__ import annotations

import json, os, time, re, uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
AGENT_COLS = {"_source", "_dataset_id"}


# ------------------------------------------------------------------------------
# GEMINI HELPER
# ------------------------------------------------------------------------------

def _call_gemini(prompt: str) -> str:
    import yaml
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("No GEMINI_API_KEY")
    model = yaml.safe_load((ROOT / "config.yaml").read_text()).get("model", "gemini-2.0-flash-lite")
    client = genai.Client(api_key=api_key)
    for attempt in range(5):
        try:
            return client.models.generate_content(model=model, contents=prompt).text.strip()
        except Exception as e:
            s = str(e)
            if "429" in s or "quota" in s.lower():
                wait = 60 * (attempt + 1)
                print(f"  [!] Rate limit, waiting {wait}s...")
                time.sleep(wait)
            elif "503" in s or "UNAVAILABLE" in s or "unavailable" in s.lower():
                wait = 15 * (attempt + 1)
                print(f"  [!] Gemini overloaded, waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini: too many retries")


def _strip_fences(text: str) -> str:
    if "```" not in text:
        return text
    m = re.search(r"```\w*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else re.sub(r"```\w*", "", text).strip()


# ------------------------------------------------------------------------------
# QUALITY METRICS
# ------------------------------------------------------------------------------

@dataclass
class QualityMetrics:
    kappa: float
    agreement_pct: float
    label_distribution: dict[str, int]
    label_distribution_pct: dict[str, float]
    confidence_mean: float
    confidence_std: float
    low_confidence_count: int      # confidence < 0.6
    low_confidence_pct: float
    total_labeled: int
    kappa_interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  ANNOTATION QUALITY METRICS",
            "=" * 60,
            f"  Total labeled:    {self.total_labeled}",
            f"  Cohen's kappa:    {self.kappa:.3f}",
            f"  Agreement:        {self.agreement_pct:.1f}%",
            f"  Confidence mean:  {self.confidence_mean:.3f} ± {self.confidence_std:.3f}",
            f"  Low confidence:   {self.low_confidence_count} ({self.low_confidence_pct:.1f}%)",
            "",
            "  Label distribution:",
        ]
        for label, count in sorted(self.label_distribution.items()):
            pct = self.label_distribution_pct.get(label, 0)
            bar = "#" * int(pct / 2)
            lines.append(f"    {label:20s} {count:6d} ({pct:5.1f}%)  {bar}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's kappa between two label sequences."""
    assert len(labels_a) == len(labels_b), "Label lists must have equal length"
    n = len(labels_a)
    if n == 0:
        return 0.0
    classes = sorted(set(labels_a) | set(labels_b))
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}

    # confusion matrix
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        if a in idx and b in idx:
            matrix[idx[a], idx[b]] += 1

    p_o = np.trace(matrix) / n
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_e = float(np.dot(row_sums, col_sums))

    if p_e == 1.0:
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def _interpret_kappa(k: float) -> str:
    if k < 0:      return "poor (worse than chance)"
    if k < 0.20:   return "slight"
    if k < 0.40:   return "fair"
    if k < 0.60:   return "moderate"
    if k < 0.80:   return "substantial"
    return "almost perfect"


# ------------------------------------------------------------------------------
# AGENT
# ------------------------------------------------------------------------------

class AnnotationAgent:

    def __init__(self, modality: str = "text"):
        self.modality = modality

    # -- skill 1: auto_label --

    def auto_label(
        self,
        df: pd.DataFrame,
        task: str,
        spec_text: str | None = None,
        batch_size: int = 20,
        consistency_sample: int = 30,
        consistency_passes: int = 3,
    ) -> pd.DataFrame:
        """
        Auto-label using Gemini zero-shot classification.
        Adds columns: _label, _confidence, _reason.
        Runs N passes on a sample to compute self-consistency kappa.
        """
        text_col = self._find_text_col(df)
        print(f"  Text column: '{text_col}'  |  rows: {len(df):,}")

        # derive classes from existing label column if present
        label_col_candidates = [c for c in df.columns if c not in AGENT_COLS and c != text_col
                                 and df[c].nunique() <= 20]
        classes = None
        for c in label_col_candidates:
            vals = sorted(df[c].dropna().unique().tolist())
            if vals:
                # normalize: 0.0 → "0", 1.0 → "1"
                classes = []
                for v in vals:
                    s = str(v)
                    classes.append(s[:-2] if s.endswith(".0") else s)
                print(f"  Classes from column '{c}': {classes}")
                break

        if not classes:
            print("  Detecting classes via Gemini...")
            classes = self._infer_classes(df[text_col].dropna().sample(min(10, len(df)), random_state=42).tolist(), task)
            print(f"  Detected classes: {classes}")

        # -- full labeling --
        print(f"  Labeling {len(df):,} rows in batches of {batch_size}...")
        all_results = self._label_in_batches(df[text_col].fillna("").tolist(), task, classes, batch_size)

        df_out = df.copy()
        df_out["_label"]      = [r.get("label", "unknown")  for r in all_results]
        df_out["_confidence"] = [float(r.get("confidence", 0.5)) for r in all_results]
        df_out["_reason"]     = [r.get("reason", "")         for r in all_results]

        # -- self-consistency: N passes on sample --
        sample_n = min(consistency_sample, len(df))
        sample_idx = df.sample(sample_n, random_state=42).index
        sample_texts = df_out.loc[sample_idx, text_col].fillna("").tolist()

        print(f"\n  Consistency check: {consistency_passes} passes on {sample_n} rows...")
        passes: list[list[str]] = [df_out.loc[sample_idx, "_label"].tolist()]  # pass 0 = main run
        for i in range(1, consistency_passes):
            print(f"    Pass {i+1}/{consistency_passes}...")
            res = self._label_in_batches(sample_texts, task, classes, batch_size)
            passes.append([r.get("label", "unknown") for r in res])

        def _norm(s: str) -> str:
            s = str(s).strip().lower()
            return s[:-2] if s.endswith(".0") else s

        # normalize all passes before any comparison
        norm_passes = [[_norm(lbl) for lbl in p] for p in passes]

        # pairwise kappa across all pass combinations
        from itertools import combinations
        kappas = [_cohens_kappa(norm_passes[a], norm_passes[b])
                  for a, b in combinations(range(len(norm_passes)), 2)]
        mean_kappa = round(float(np.mean(kappas)), 4)
        std_kappa  = round(float(np.std(kappas)), 4)

        # per-row agreement rate (fraction of passes that agree with majority)
        per_row_agree = []
        for i in range(sample_n):
            row_labels = [p[i] for p in norm_passes]
            majority = max(set(row_labels), key=row_labels.count)
            per_row_agree.append(row_labels.count(majority) / len(norm_passes))
        mean_row_agree = round(float(np.mean(per_row_agree)) * 100, 1)

        # rows with full agreement across all passes
        full_agree = sum(1 for a in per_row_agree if a == 1.0)

        # pairwise percent agreement (more robust than kappa for imbalanced data)
        pairwise_agree = []
        for a, b in combinations(range(len(norm_passes)), 2):
            ag = sum(x == y for x, y in zip(norm_passes[a], norm_passes[b])) / sample_n * 100
            pairwise_agree.append(ag)
        mean_pair_agree = round(float(np.mean(pairwise_agree)), 1)

        kappa_note = ""
        if mean_kappa == 0.0 and mean_row_agree > 50:
            kappa_note = "  ← 0 due to class imbalance (kappa paradox)"

        print(f"\n  ┌─ Consistency ({consistency_passes} passes, {sample_n} rows)")
        print(f"  │  Row agreement:   {mean_row_agree:.1f}%  (mean across rows)")
        print(f"  │  Full consensus:  {full_agree}/{sample_n} rows ({full_agree/sample_n*100:.1f}%)")
        print(f"  │  Pairwise agree:")
        for (a, b), ag in zip(combinations(range(len(norm_passes)), 2), pairwise_agree):
            print(f"  │    pass {a+1} vs pass {b+1}: {ag:.1f}%")
        print(f"  │  Mean pairwise:   {mean_pair_agree:.1f}%")
        print(f"  │  Cohen's kappa:   {mean_kappa:.3f}{kappa_note}")
        print(f"  └─")

        df_out.attrs["self_consistency_kappa"] = mean_kappa
        df_out.attrs["consistency_passes"]     = consistency_passes
        df_out.attrs["consistency_kappas"]     = kappas
        df_out.attrs["mean_row_agreement"]     = mean_row_agree
        df_out.attrs["classes"] = classes
        df_out.attrs["task"] = task
        return df_out

    def _find_text_col(self, df: pd.DataFrame) -> str:
        preferred = ["text", "comment", "comment_text", "review", "sentence",
                     "body", "content", "message", "tweet", "post"]
        user_cols = [c for c in df.columns if c not in AGENT_COLS]
        for name in preferred:
            if name in user_cols:
                return name
        # longest average string column
        str_cols = df[user_cols].select_dtypes(include="object").columns
        if len(str_cols) == 0:
            return user_cols[0]
        return max(str_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())

    def _infer_classes(self, samples: list[str], task: str) -> list[str]:
        prompt = f"""Task: "{task}"

Sample texts:
{json.dumps(samples[:10], ensure_ascii=False)}

What are the label classes for this task? Return a JSON array of class name strings only.
Example: ["positive", "negative", "neutral"]
Return ONLY the JSON array."""
        raw = _strip_fences(_call_gemini(prompt))
        return json.loads(raw)

    def _extract_classes_from_spec(self, spec_text: str) -> list[str] | None:
        m = re.search(r"##\s*Classes(.*?)(?=##|\Z)", spec_text, re.DOTALL | re.IGNORECASE)
        if not m:
            return None
        block = m.group(1)
        classes = re.findall(r"\*\*([^*]+)\*\*", block)
        return classes if classes else None

    def _label_in_batches(self, texts: list[str], task: str, classes: list[str],
                           batch_size: int) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._label_batch(batch, task, classes)
            results.extend(batch_results)
            if i + batch_size < len(texts):
                time.sleep(0.5)
        return results

    def _label_batch(self, texts: list[str], task: str, classes: list[str]) -> list[dict]:
        numbered = [{"id": i, "text": t[:500]} for i, t in enumerate(texts)]
        prompt = f"""Task: "{task}"
Classes: {classes}

Label each text. For each return:
{{"id": <same id>, "label": "<one of the classes>", "confidence": <0.0-1.0>, "reason": "<5 words max>"}}

Texts:
{json.dumps(numbered, ensure_ascii=False)}

Return a JSON array of {len(texts)} objects in the same order as input.
Return ONLY the JSON array, no markdown fences."""
        try:
            raw = _strip_fences(_call_gemini(prompt))
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == len(texts):
                return parsed
            # fallback: try to align by id
            by_id = {r.get("id", i): r for i, r in enumerate(parsed)}
            return [by_id.get(i, {"label": "unknown", "confidence": 0.0, "reason": ""})
                    for i in range(len(texts))]
        except Exception as e:
            print(f"  [!] Batch error: {e}")
            return [{"label": "unknown", "confidence": 0.0, "reason": ""}] * len(texts)

    # -- skill 2: generate_spec --

    def generate_spec(self, df: pd.DataFrame, task: str, topic_path: Path | None = None) -> str:
        """
        Generate annotation specification markdown.
        Saves to <topic>/annotation_spec.md and returns the text.
        """
        text_col = self._find_text_col(df)
        user_cols = [c for c in df.columns if c not in AGENT_COLS]
        sample_rows = (df[user_cols].sample(min(15, len(df)), random_state=42)
                       .fillna("").astype(str).to_dict(orient="records"))

        prompt = f"""You are an expert data annotation lead. Write a complete annotation specification.

Task: "{task}"
Dataset columns: {user_cols}
Sample rows:
{json.dumps(sample_rows, indent=2, ensure_ascii=False)}

Write a detailed annotation_spec.md with these sections:

# Annotation Specification: {{task name}}

## Task Description
What annotators must do, what the output is used for.

## Classes
For each class:
**ClassName**: precise definition, what to include, what to exclude

## Examples
For EACH class, provide 3+ real examples from the sample data:
### ClassName
- Example 1: "text" → label, reason
- Example 2: ...

## Edge Cases
List 5+ ambiguous or tricky cases with the recommended label and reasoning.

## Guidelines
Step-by-step annotation instructions for a non-expert.

## Quality Criteria
What makes a good annotation. Minimum confidence threshold.

Write in Russian. Be specific and use real examples from the sample data.
Return ONLY the markdown text, no fences."""

        spec_text = _call_gemini(prompt)

        if topic_path:
            spec_path = topic_path / "annotation_spec.md"
            spec_path.write_text(spec_text, encoding="utf-8")
            print(f"  [ok] annotation_spec.md -> {spec_path.relative_to(ROOT)}")

        return spec_text

    # -- skill 3: check_quality --

    def check_quality(
        self,
        df_labeled: pd.DataFrame,
        df_human: pd.DataFrame | None = None,
        label_col: str = "_label",
        human_col: str = "label",
        confidence_col: str = "_confidence",
        low_confidence_threshold: float = 0.6,
    ) -> QualityMetrics:
        """
        Compute quality metrics.
        - If df_human provided: Cohen's kappa between auto and human labels.
        - If not: uses self-consistency kappa from auto_label (stored in df.attrs).
        """
        labels = df_labeled[label_col].fillna("unknown").tolist()
        dist = df_labeled[label_col].value_counts().to_dict()
        total = len(labels)
        dist_pct = {k: round(v / total * 100, 1) for k, v in dist.items()}

        conf = df_labeled[confidence_col].astype(float) if confidence_col in df_labeled.columns \
               else pd.Series([0.5] * total)
        if conf.max() == 0.0:
            print(f"  [!] All confidence values are 0.0 — labeling likely failed silently. "
                  f"Check for batch errors above.")
        low_conf = (conf < low_confidence_threshold).sum()

        if df_human is not None:
            # align on index
            merged = df_labeled[[label_col]].join(df_human[[human_col]], how="inner")
            auto   = merged[label_col].tolist()
            human  = merged[human_col].tolist()
            kappa  = _cohens_kappa(auto, human)
            agree  = round(sum(a == h for a, h in zip(auto, human)) / len(auto) * 100, 1)
            print(f"  Comparing {len(auto)} rows (auto vs human)")
        else:
            kappa = float(df_labeled.attrs.get("self_consistency_kappa", 0.0))
            agree = round((kappa * (1 - 0) + 0) * 100, 1)  # approximate from kappa

        return QualityMetrics(
            kappa=kappa,
            agreement_pct=agree,
            label_distribution=dist,
            label_distribution_pct=dist_pct,
            confidence_mean=round(float(conf.mean()), 4),
            confidence_std=round(float(conf.std()), 4),
            low_confidence_count=int(low_conf),
            low_confidence_pct=round(low_conf / total * 100, 1),
            total_labeled=total,
            kappa_interpretation=_interpret_kappa(kappa),
        )

    # -- skill 4: export_to_labelstudio --

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        task_type: str = "classification",
        text_col: str | None = None,
        label_col: str = "_label",
        confidence_col: str = "_confidence",
        from_name: str = "label",
        to_name: str = "text",
        output_path: Path | None = None,
    ) -> list[dict]:
        """
        Export to LabelStudio JSON import format.
        Auto-labels appear as predictions (pre-annotations) for human review.
        """
        if text_col is None:
            text_col = self._find_text_col(df)
        classes = sorted(df[label_col].dropna().unique().tolist()) \
                  if label_col in df.columns else []

        tasks = []
        for _, row in df.iterrows():
            text = str(row.get(text_col, ""))
            label = str(row.get(label_col, "")) if label_col in df.columns else ""
            score = float(row.get(confidence_col, 0.5)) if confidence_col in df.columns else 0.5

            task: dict[str, Any] = {
                "data": {to_name: text},
                "predictions": [],
                "annotations": [],
            }

            if label and label not in ("unknown", "nan", ""):
                result_id = str(uuid.uuid4())[:8]
                if task_type == "classification":
                    result = {
                        "id": result_id,
                        "type": "choices",
                        "value": {"choices": [label]},
                        "from_name": from_name,
                        "to_name":   to_name,
                        "score": score,
                    }
                elif task_type == "ner":
                    result = {
                        "id": result_id,
                        "type": "labels",
                        "value": {"start": 0, "end": len(text), "labels": [label]},
                        "from_name": from_name,
                        "to_name":   to_name,
                        "score": score,
                    }
                else:
                    result = {
                        "id": result_id,
                        "type": "choices",
                        "value": {"choices": [label]},
                        "from_name": from_name,
                        "to_name":   to_name,
                        "score": score,
                    }

                task["predictions"] = [{
                    "model_version": "AnnotationAgent-v1",
                    "score": score,
                    "result": [result],
                }]

            tasks.append(task)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
            print(f"  [ok] labelstudio_import.json -> {output_path}  ({len(tasks)} tasks)")

        return tasks

    # -- report notebook --

    def generate_report_notebook(
        self,
        df_labeled: pd.DataFrame,
        metrics: QualityMetrics,
        spec_text: str,
        topic_path: Path,
        task: str,
    ) -> Path | None:
        try:
            import nbformat
        except ImportError:
            print("  [!] nbformat not installed"); return None

        parquet_path = str((topic_path / "data" / "labeled" / "auto_labeled.parquet").relative_to(ROOT))
        classes = list(metrics.label_distribution.keys())
        text_col = self._find_text_col(df_labeled)

        cells = [
            ("markdown",
             f"# Annotation Report: {task}\n\n"
             f"Total labeled: **{metrics.total_labeled:,}**  |  "
             f"Cohen's κ: **{metrics.kappa:.3f}** ({metrics.kappa_interpretation})  |  "
             f"Confidence: **{metrics.confidence_mean:.2f}**"),

            ("code",
             "import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n"
             "import sys; sys.path.insert(0, '.')\n"
             "from data_annotation.annotation_agent import AnnotationAgent, _cohens_kappa\n\n"
             f"df = pd.read_parquet('{parquet_path}')\n"
             "print(df.shape)\ndf[['_label','_confidence','_reason']].head(10)"),

            ("markdown", "## Label Distribution"),
            ("code",
             "vc = df['_label'].value_counts()\n"
             "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
             "vc.plot(kind='bar', ax=axes[0], color='steelblue')\n"
             "axes[0].set_title('Label counts'); axes[0].set_xlabel('')\n"
             "vc.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')\n"
             "axes[1].set_title('Label distribution')\n"
             "plt.tight_layout(); plt.show()\nprint(vc)"),

            ("markdown", "## Confidence Distribution"),
            ("code",
             "fig, ax = plt.subplots(figsize=(10, 3))\n"
             "df['_confidence'].hist(bins=20, ax=ax, color='steelblue', edgecolor='white')\n"
             "ax.axvline(0.6, color='red', linestyle='--', label='threshold 0.6')\n"
             "ax.set_title('Confidence Distribution'); ax.legend()\n"
             "plt.tight_layout(); plt.show()\n"
             f"print(f'Mean: {{df[\"_confidence\"].mean():.3f}}  "
             f"Low confidence (<0.6): {{(df[\"_confidence\"] < 0.6).sum()}}')"
             ),

            ("markdown", "## Low Confidence Samples (need human review)"),
            ("code",
             f"low = df[df['_confidence'] < 0.6][['{ text_col }','_label','_confidence','_reason']]\n"
             "print(f'Low confidence rows: {len(low)}')\nlow.head(20)"),

            ("markdown", "## Quality Metrics Summary"),
            ("code",
             f"print('Cohen kappa: {metrics.kappa:.3f} ({metrics.kappa_interpretation})')\n"
             f"print('Agreement:   {metrics.agreement_pct:.1f}%')\n"
             f"print('Confidence:  {metrics.confidence_mean:.3f} ± {metrics.confidence_std:.3f}')\n"
             f"print('Low conf:    {metrics.low_confidence_count} ({metrics.low_confidence_pct:.1f}%)')"),

            ("markdown",
             f"## Annotation Specification\n\n{spec_text[:3000]}{'...' if len(spec_text) > 3000 else ''}"),
        ]

        nb = nbformat.v4.new_notebook()
        for cell_type, source in cells:
            fn = nbformat.v4.new_markdown_cell if cell_type == "markdown" else nbformat.v4.new_code_cell
            nb.cells.append(fn(source))

        nb_path = topic_path / "notebooks" / "annotation_report.ipynb"
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        nb_path.write_text(nbformat.writes(nb))
        return nb_path


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AnnotationAgent — auto-labeling and quality control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_annotation/annotation_agent.py toxic_comment \\
      --task "toxic comment classification"

  python data_annotation/annotation_agent.py toxic_comment \\
      --task "toxic comment classification" --explain

  python data_annotation/annotation_agent.py toxic_comment \\
      --check-quality human_labels.csv --human-col label

  python data_annotation/annotation_agent.py toxic_comment \\
      --task "toxic comment classification" --rows 500
        """
    )
    parser.add_argument("topic_dir",      nargs="?", default=None,
                        help="Topic folder (e.g. toxic_comment)")
    parser.add_argument("--task",         default=None,
                        help="Annotation task description")
    parser.add_argument("--modality",     default="text",
                        choices=["text"], help="Data modality (default: text)")
    parser.add_argument("--rows",         type=int, default=None,
                        help="Limit number of rows to label (default: all)")
    parser.add_argument("--batch-size",   type=int, default=20,
                        help="Gemini batch size (default: 20)")
    parser.add_argument("--check-quality",dest="human_labels", default=None,
                        help="Path to CSV with human labels for kappa comparison")
    parser.add_argument("--human-col",    default="label",
                        help="Column name with human labels (default: label)")
    parser.add_argument("--no-spec",      action="store_true",
                        help="Skip spec generation")
    args = parser.parse_args()

    # -- find topic --
    if args.topic_dir:
        topic_path = Path(args.topic_dir) if Path(args.topic_dir).is_absolute() \
                     else ROOT / args.topic_dir
    else:
        topics = sorted(ROOT.glob("*/data/raw/combined.parquet"))
        if not topics:
            print("[!] No topics found. Run data_collection agent first.")
            raise SystemExit(1)
        print("Available topics:")
        for i, p in enumerate(topics, 1):
            print(f"  [{i}] {p.parent.parent.parent.name}")
        topic_path = topics[0].parent.parent.parent

    # -- find input parquet (prefer clean over raw) --
    clean = topic_path / "data" / "clean" / "combined_clean.parquet"
    raw   = topic_path / "data" / "raw"   / "combined.parquet"
    if clean.exists():
        pf = clean
    elif raw.exists():
        pf = raw
    else:
        candidates = list((topic_path / "data" / "raw").glob("*.parquet")) \
                     if (topic_path / "data" / "raw").exists() else []
        if not candidates:
            print(f"[!] No parquet files in {topic_path / 'data'}")
            raise SystemExit(1)
        pf = candidates[0]

    print(f"\nLoading: {pf.relative_to(ROOT)}")
    df = pd.read_parquet(pf)
    print(f"Total rows: {len(df)}")

    n_rows = args.rows
    if n_rows is None:
        try:
            ans = input(f"How many rows to label? [all / number, default=all]: ").strip()
            if ans.lower() not in ("", "all"):
                n_rows = int(ans)
        except (ValueError, EOFError):
            pass

    if n_rows:
        df = df.sample(min(n_rows, len(df)), random_state=42).reset_index(drop=True)
    print(f"Labeling: {len(df)} rows\n")

    task = args.task or topic_path.name.replace("_", " ")
    agent = AnnotationAgent(modality=args.modality)

    # -- generate spec --
    spec_text = ""
    if not args.no_spec:
        print("Generating annotation spec...")
        spec_text = agent.generate_spec(df, task, topic_path)
        print(f"  Spec length: {len(spec_text)} chars")

    # -- auto label --
    print(f"\nAuto-labeling ({args.modality})...")
    df_labeled = agent.auto_label(
        df, task=task, spec_text=spec_text or None,
        batch_size=args.batch_size,
    )

    # -- save labeled --
    labeled_dir = topic_path / "data" / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = labeled_dir / "auto_labeled.parquet"
    df_labeled.to_parquet(labeled_path, index=False)
    print(f"\n[ok] auto_labeled.parquet -> {labeled_path.relative_to(ROOT)}")

    # -- label distribution + confidence report --
    print(f"\n{'─'*50}")
    print("  LLM ANNOTATION REPORT")
    print(f"{'─'*50}")
    print(f"  Total labeled : {len(df_labeled)}")
    if "_label" in df_labeled.columns:
        dist = df_labeled["_label"].value_counts()
        print(f"\n  Label distribution:")
        for lbl, cnt in dist.items():
            pct = cnt / len(df_labeled) * 100
            bar = "█" * int(pct / 5)
            print(f"    {str(lbl):<25} {cnt:>5}  ({pct:5.1f}%)  {bar}")
    if "_confidence" in df_labeled.columns:
        conf = df_labeled["_confidence"]
        print(f"\n  Confidence:")
        print(f"    mean   {conf.mean():.3f}")
        print(f"    median {conf.median():.3f}")
        print(f"    min    {conf.min():.3f}")
        print(f"    max    {conf.max():.3f}")
        low = (conf < 0.7).sum()
        print(f"    low (<0.7)  {low} rows  ({low/len(df_labeled)*100:.1f}%)")
    print(f"{'─'*50}\n")

    # -- quality metrics --
    print("Computing quality metrics...")
    df_human = None
    if args.human_labels:
        hp = Path(args.human_labels)
        if hp.exists():
            df_human = pd.read_csv(hp)
            print(f"  Human labels loaded: {len(df_human)} rows")
        else:
            print(f"  [!] Human labels file not found: {hp}")

    metrics = agent.check_quality(df_labeled, df_human=df_human, human_col=args.human_col)
    print(metrics)

    # -- export to LabelStudio --
    print("Exporting to LabelStudio...")
    ls_path = labeled_dir / "labelstudio_import.json"
    agent.export_to_labelstudio(df_labeled, output_path=ls_path)

    # -- notebook --
    print("Generating annotation report notebook...")
    nb_path = agent.generate_report_notebook(df_labeled, metrics, spec_text, topic_path, task)
    if nb_path:
        print(f"[ok] notebook -> {nb_path.relative_to(ROOT)}")

    print(f"\n{'='*60}")
    print(f"  Done. Topic: {topic_path.name}")
    print(f"  Labeled rows:  {len(df_labeled):,}")
    print(f"  Cohen kappa:   {metrics.kappa:.3f} ({metrics.kappa_interpretation})")
    print(f"  LabelStudio:   {ls_path.relative_to(ROOT)}")
    print(f"  Spec:          {(topic_path / 'annotation_spec.md').relative_to(ROOT)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
