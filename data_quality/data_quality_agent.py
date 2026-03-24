#!/usr/bin/env python3
"""
DataQualityAgent - Data Detective

Detects and fixes data quality issues:
missing values, duplicates, outliers, class imbalance.

Folder structure (mirrors data_collection):
  <topic>/
  ├── data/
  │   ├── raw/
  │   │   └── combined.parquet        <- input
  │   └── clean/
  │       └── combined_clean.parquet  <- output
  └── notebooks/
      └── quality_report.ipynb        <- generated report

Usage:
  python data_quality/data_quality_agent.py toxic_comment
  python data_quality/data_quality_agent.py toxic_comment --task "toxic comment classification"
  python data_quality/data_quality_agent.py toxic_comment --strategy mean --outliers clip_zscore

  from data_quality.data_quality_agent import DataQualityAgent
  agent = DataQualityAgent()
  report = agent.detect_issues(df)
  df_clean = agent.fix(df, strategy={'missing': 'median', 'duplicates': 'drop', 'outliers': 'clip_iqr'})
  comparison = agent.compare(df, df_clean)
"""

from __future__ import annotations

import json, os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent


# ------------------------------------------------------------------------------
# REPORT TYPES
# ------------------------------------------------------------------------------

@dataclass
class QualityReport:
    missing:    dict[str, dict]
    duplicates: int
    outliers:   dict[str, dict]
    imbalance:  dict[str, Any]
    summary:    dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  QUALITY REPORT",
            "=" * 60,
            f"  Missing values:  {self.summary['missing_cells']} cells in {self.summary['missing_cols']} columns",
            f"  Duplicates:      {self.duplicates} rows",
            f"  Outliers (IQR):  {self.summary['outlier_cells']} values in {self.summary['outlier_cols']} columns",
        ]
        if self.imbalance:
            lines.append(f"  Class imbalance: ratio {self.imbalance.get('ratio', '?')} (max/min class)")
        lines.append("=" * 60)
        return "\n".join(lines)


# ------------------------------------------------------------------------------
# AGENT
# ------------------------------------------------------------------------------

class DataQualityAgent:

    # -- skill 1: detect --

    def detect_issues(self, df: pd.DataFrame, label_col: str | None = None) -> QualityReport:
        missing    = self._detect_missing(df)
        duplicates = self._detect_duplicates(df)
        outliers   = self._detect_outliers(df)
        imbalance  = self._detect_imbalance(df, label_col)
        summary = {
            "missing_cells":  sum(v["count"] for v in missing.values()),
            "missing_cols":   len(missing),
            "outlier_cells":  sum(v["count"] for v in outliers.values()),
            "outlier_cols":   len(outliers),
        }
        return QualityReport(missing=missing, duplicates=duplicates,
                             outliers=outliers, imbalance=imbalance, summary=summary)

    def _detect_missing(self, df: pd.DataFrame) -> dict:
        result = {}
        for col in df.columns:
            n = int(df[col].isna().sum())
            if n > 0:
                result[col] = {"count": n, "pct": round(n / len(df) * 100, 2)}
        return result

    def _detect_duplicates(self, df: pd.DataFrame) -> int:
        return int(df.duplicated().sum())

    def _detect_outliers(self, df: pd.DataFrame, method: str = "iqr") -> dict:
        result = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:
                low  = series.mean() - 3 * series.std()
                high = series.mean() + 3 * series.std()
            mask = (df[col] < low) | (df[col] > high)
            n = int(mask.sum())
            if n > 0:
                result[col] = {
                    "count": n, "pct": round(n / len(df) * 100, 2),
                    "lower_bound": round(float(low), 4),
                    "upper_bound": round(float(high), 4),
                }
        return result

    def _detect_imbalance(self, df: pd.DataFrame, label_col: str | None) -> dict:
        if label_col is None:
            candidates = [c for c in df.columns
                          if df[c].dtype == "object" and 2 <= df[c].nunique() <= 20]
            if not candidates:
                candidates = [c for c in df.select_dtypes(include=[np.number]).columns
                              if 2 <= df[c].nunique() <= 10]
            if not candidates:
                return {}
            label_col = candidates[0]
        counts = df[label_col].value_counts().to_dict()
        values = list(counts.values())
        ratio = round(max(values) / min(values), 2) if min(values) > 0 else float("inf")
        return {"col": label_col, "counts": counts, "ratio": ratio}

    # -- skill 2: fix --

    # колонки которые нельзя трогать при очистке
    TARGET_COLS = {"label", "target", "class", "category", "toxic",
                   "sentiment", "y", "split"}

    def fix(self, df: pd.DataFrame, strategy: dict,
            protected_cols: list[str] | None = None) -> pd.DataFrame:
        """
        strategy keys:
          missing:    'median' | 'mean' | 'mode' | 'drop' | 'ffill' | 'constant:<value>'
          duplicates: 'drop' | 'keep_first' | 'keep_last' | 'none'
          outliers:   'clip_iqr' | 'clip_zscore' | 'drop' | 'none'

        protected_cols: колонки-метки, которые не изменя��тся (label, target и т.п.)
        """
        # автоопределение защищённых колонок
        auto_protected = {c for c in df.columns
                          if c in self.TARGET_COLS or c.startswith("_")}
        if protected_cols:
            auto_protected.update(protected_cols)

        result = df.copy()
        if "duplicates" in strategy:
            result = self._fix_duplicates(result, strategy["duplicates"])
        if "missing" in strategy:
            result = self._fix_missing(result, strategy["missing"], auto_protected)
        if "outliers" in strategy:
            result = self._fix_outliers(result, strategy["outliers"], auto_protected)
        return result

    def _fix_missing(self, df: pd.DataFrame, strategy: str,
                     protected: set[str] | None = None) -> pd.DataFrame:
        protected = protected or set()
        if strategy == "drop":
            return df.dropna()
        if strategy == "ffill":
            return df.ffill()
        if strategy.startswith("constant:"):
            return df.fillna(strategy.split(":", 1)[1])
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in protected:
                continue
            if df[col].isna().any():
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
        for col in df.select_dtypes(include=["object"]).columns:
            if col in protected:
                continue
            if df[col].isna().any():
                fill = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(fill)
        return df

    def _fix_duplicates(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "none":
            return df
        keep = {"drop": "first", "keep_first": "first", "keep_last": "last"}.get(strategy, "first")
        return df.drop_duplicates(keep=keep)

    def _fix_outliers(self, df: pd.DataFrame, strategy: str,
                      protected: set[str] | None = None) -> pd.DataFrame:
        if strategy == "none":
            return df
        protected = protected or set()
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in protected:
                continue
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            low_iqr,  high_iqr  = q1 - 1.5 * iqr,            q3 + 1.5 * iqr
            low_z,    high_z    = series.mean() - 3 * series.std(), series.mean() + 3 * series.std()
            if strategy == "clip_iqr":
                df[col] = df[col].clip(low_iqr, high_iqr)
            elif strategy == "clip_zscore":
                df[col] = df[col].clip(low_z, high_z)
            elif strategy == "drop":
                mask = (df[col] >= low_iqr) & (df[col] <= high_iqr)
                df = df[mask | df[col].isna()]
        return df

    # -- skill 3: compare --

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                label: str = "") -> pd.DataFrame:
        """Return a before/after comparison table for all quality metrics."""
        def _outliers(df):
            total = 0
            for col in df.select_dtypes(include=[np.number]).columns:
                s = df[col].dropna()
                if len(s) < 4:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                total += int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
            return total

        rows = [
            ("Rows",           len(df_before),                     len(df_after)),
            ("Missing values", int(df_before.isna().sum().sum()),  int(df_after.isna().sum().sum())),
            ("Duplicates",     int(df_before.duplicated().sum()),  int(df_after.duplicated().sum())),
            ("Outliers (IQR)", _outliers(df_before),               _outliers(df_after)),
        ]
        records = []
        for metric, before, after in rows:
            delta = after - before
            pct   = f"{delta / before * 100:+.1f}%" if before != 0 else "n/a"
            records.append({"Metric": metric, "Before": before, "After": after,
                            "Delta": delta, "Change": pct})
        df_cmp = pd.DataFrame(records).set_index("Metric")
        if label:
            df_cmp.columns = pd.MultiIndex.from_tuples([(label, c) for c in df_cmp.columns])
        return df_cmp

    # -- suggest strategies --

    def suggest_strategies(self, report: QualityReport, task: str) -> list[dict]:
        """Ask Gemini to propose 3 strategies (strict/medium/mild) based on actual data."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return self._default_strategies()
        from google import genai
        import yaml
        model = yaml.safe_load((ROOT / "config.yaml").read_text()).get("model", "gemini-2.0-flash-lite")
        client = genai.Client(api_key=api_key)

        prompt = f"""You are a data quality expert. Propose 3 cleaning strategies for this dataset.

ML task: {task}

Quality report:
{json.dumps(report.to_dict(), indent=2, ensure_ascii=False)}

Available options:
  missing:    median | mean | mode | drop | ffill | constant:0
  duplicates: drop | keep_first | keep_last | none
  outliers:   clip_iqr | clip_zscore | drop | none

Return a JSON array of exactly 3 strategies, ordered strict -> medium -> mild:
[
  {{
    "name": "Strict",
    "label": "strict",
    "reason": "one sentence: when to use this",
    "strategy": {{"missing": "drop", "duplicates": "drop", "outliers": "drop"}}
  }},
  {{
    "name": "Medium",
    "label": "medium",
    "reason": "...",
    "strategy": {{"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"}}
  }},
  {{
    "name": "Mild",
    "label": "mild",
    "reason": "...",
    "strategy": {{"missing": "ffill", "duplicates": "none", "outliers": "none"}}
  }}
]

Base your choices on the actual issues found. If missing % is high, dropping rows may lose too much data.
If class imbalance exists, avoid drop strategies that worsen it.
Return ONLY the JSON array."""

        try:
            raw = client.models.generate_content(model=model, contents=prompt).text.strip()
            import re
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            return json.loads(m.group()) if m else self._default_strategies()
        except Exception as e:
            print(f"  [!] Gemini error: {e}")
            return self._default_strategies()

    def _default_strategies(self) -> list[dict]:
        return [
            {"name": "Strict", "label": "strict",
             "reason": "Maximum data quality, smaller dataset",
             "strategy": {"missing": "drop", "duplicates": "drop", "outliers": "drop"}},
            {"name": "Medium", "label": "medium",
             "reason": "Balance between quality and data retention",
             "strategy": {"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"}},
            {"name": "Mild",   "label": "mild",
             "reason": "Preserve maximum data, minimal intervention",
             "strategy": {"missing": "ffill", "duplicates": "none", "outliers": "none"}},
        ]

    # -- bonus: LLM explain --

    def explain_with_llm(self, report: QualityReport, task_description: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return "[!] No GEMINI_API_KEY"
        from google import genai
        import yaml
        model = yaml.safe_load((ROOT / "config.yaml").read_text()).get("model", "gemini-2.0-flash-lite")
        client = genai.Client(api_key=api_key)
        prompt = f"""You are a data quality expert. Analyze the report and justify the best cleaning strategy.

ML task: {task_description}

Quality report:
{json.dumps(report.to_dict(), indent=2, ensure_ascii=False)}

Answer in markdown with these sections:
## Found Issues
Explain each issue and its impact on the ML task.

## Recommended Strategy
Exact strategy dict for fix() with reasoning for each choice.

## Why This Strategy
Compare with alternatives. Why is this better for the ML task specifically?

## Critical vs Minor
What must be fixed vs what is optional.

Be specific, reference actual column names and numbers."""
        return client.models.generate_content(model=model, contents=prompt).text.strip()

    # -- report notebook --

    def generate_report_notebook(
        self,
        df: pd.DataFrame,
        report: QualityReport,
        topic_path: Path,
        task: str,
        strategy_1: dict,
        strategy_2: dict,
    ) -> Path:
        """
        Generate quality_report.ipynb with:
        - Part 1: visualizations of detected issues
        - Part 2: comparison of 2 cleaning strategies
        - Part 3: LLM justification markdown
        """
        try:
            import nbformat
        except ImportError:
            print("  [!] nbformat not installed: pip install nbformat"); return None

        parquet_path = str((topic_path / "data" / "raw" / "combined.parquet").relative_to(ROOT))
        num_cols = list(df.select_dtypes(include=[np.number]).columns[:6])
        imb_col  = report.imbalance.get("col", "")

        s1_label = f"Strategy 1: missing={strategy_1['missing']}, outliers={strategy_1['outliers']}"
        s2_label = f"Strategy 2: missing={strategy_2['missing']}, outliers={strategy_2['outliers']}"

        cells = [
            ("markdown", f"# Data Quality Report: {task}\n\n"
                         f"Dataset: `{parquet_path}`  \n"
                         f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns"),

            ("code",
             "import pandas as pd\nimport numpy as np\n"
             "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
             "from pathlib import Path\n"
             "import sys; sys.path.insert(0, str(Path('.')))\n"
             "from data_quality.data_quality_agent import DataQualityAgent\n\n"
             f"df = pd.read_parquet('{parquet_path}')\n"
             "print(f'Shape: {df.shape}')\ndf.head()"),

            ("markdown", "## Part 1: Detective — Issue Detection"),

            ("code",
             "agent = DataQualityAgent()\n"
             f"report = agent.detect_issues(df, label_col={repr(imb_col) if imb_col else 'None'})\n"
             "print(report)"),

            ("markdown", "### Missing Values"),
            ("code",
             "fig, ax = plt.subplots(figsize=(12, 4))\n"
             "sns.heatmap(df.isnull(), yticklabels=False, cbar=True, ax=ax, cmap='viridis')\n"
             "ax.set_title('Missing Values Heatmap')\n"
             "plt.tight_layout(); plt.show()\n\n"
             "missing = df.isnull().sum()\nmissing = missing[missing > 0]\n"
             "print('Missing per column:')\nprint(missing.sort_values(ascending=False))"),

            ("markdown", "### Outliers"),
            ("code",
             f"num_cols = {num_cols}\n"
             "if num_cols:\n"
             "    fig, axes = plt.subplots(1, len(num_cols), figsize=(4*len(num_cols), 4))\n"
             "    axes = [axes] if len(num_cols) == 1 else axes\n"
             "    for ax, col in zip(axes, num_cols):\n"
             "        df[col].dropna().plot(kind='box', ax=ax)\n"
             "        ax.set_title(col)\n"
             "    plt.suptitle('Outlier Detection (IQR)')\n"
             "    plt.tight_layout(); plt.show()"),

            ("markdown", "### Class Imbalance"),
            ("code",
             f"imb_col = {repr(imb_col)}\n"
             "if imb_col and imb_col in df.columns:\n"
             "    vc = df[imb_col].value_counts()\n"
             "    fig, ax = plt.subplots(figsize=(8, 4))\n"
             "    vc.plot(kind='bar', ax=ax, color='steelblue')\n"
             "    ax.set_title(f'Class Distribution: {imb_col}')\n"
             "    ax.set_xlabel(imb_col); ax.set_ylabel('Count')\n"
             "    plt.tight_layout(); plt.show()\n"
             "    print(vc)\n"
             "    print(f'Imbalance ratio: {vc.max()/vc.min():.2f}')\n"
             "else:\n"
             "    print('No categorical label column found for imbalance analysis')"),

            ("markdown", "## Part 2: Surgeon — Cleaning Strategies"),

            ("code",
             f"strategy_1 = {strategy_1}\n"
             f"strategy_2 = {strategy_2}\n\n"
             "df_clean_1 = agent.fix(df, strategy_1)\n"
             "df_clean_2 = agent.fix(df, strategy_2)\n\n"
             "cmp1 = agent.compare(df, df_clean_1)\n"
             "cmp2 = agent.compare(df, df_clean_2)\n\n"
             f"cmp1.columns = ['{s1_label[:30]}/' + c for c in cmp1.columns]\n"
             f"cmp2.columns = ['{s2_label[:30]}/' + c for c in cmp2.columns]\n\n"
             "comparison = pd.concat([cmp1, cmp2], axis=1)\n"
             "print('Strategy Comparison:')\nprint(comparison.to_string())"),

            ("markdown", "## Part 3: Justification — Why This Strategy?\n\n"
                         "_LLM analysis will be inserted here when running with --explain flag_"),
        ]

        nb = nbformat.v4.new_notebook()
        for cell_type, source in cells:
            fn = nbformat.v4.new_markdown_cell if cell_type == "markdown" else nbformat.v4.new_code_cell
            nb.cells.append(fn(source))

        nb_path = topic_path / "notebooks" / "quality_report.ipynb"
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        nb_path.write_text(nbformat.writes(nb))
        return nb_path


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DataQualityAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_quality/data_quality_agent.py toxic_comment
  python data_quality/data_quality_agent.py toxic_comment --task "toxic comment classification"
  python data_quality/data_quality_agent.py toxic_comment --strategy mean --outliers clip_zscore --explain
        """
    )
    parser.add_argument("topic_dir",    nargs="?",  default=None,
                        help="Topic folder (e.g. toxic_comment). Default: scan all */data/raw/combined.parquet")
    parser.add_argument("--label-col",  default=None)
    parser.add_argument("--explain",    action="store_true",
                        help="Ask Gemini to justify strategy choice")
    parser.add_argument("--task",       default="ML task",
                        help="ML task description for Gemini explanation")
    args = parser.parse_args()

    # -- find input file --
    if args.topic_dir:
        topic_path = Path(args.topic_dir) if Path(args.topic_dir).is_absolute() \
                     else ROOT / args.topic_dir
        pf = topic_path / "data" / "raw" / "combined.parquet"
        if not pf.exists():
            candidates = list((topic_path / "data" / "raw").glob("*.parquet")) \
                         if (topic_path / "data" / "raw").exists() else []
            if not candidates:
                print(f"[!] No parquet files in {topic_path / 'data' / 'raw'}")
                raise SystemExit(1)
            pf = candidates[0]
        topic_path_resolved = topic_path
    else:
        paths = sorted(ROOT.glob("*/data/raw/combined.parquet"))
        if not paths:
            paths = sorted(ROOT.glob("*/data/raw/*.parquet"))
        if not paths:
            print("[!] No parquet files found. Run data_collection agent first.")
            raise SystemExit(1)
        print("Available topics:")
        for i, p in enumerate(paths, 1):
            print(f"  [{i}] {p.relative_to(ROOT)}")
        print()
        pf = paths[0]
        topic_path_resolved = pf.parent.parent.parent  # .../topic/

    print(f"Loading: {pf.relative_to(ROOT)}")
    df = pd.read_parquet(pf)
    print(f"Shape: {df.shape}\n")

    agent = DataQualityAgent()

    # -- detect --
    report = agent.detect_issues(df, label_col=args.label_col)
    print(report)

    # -- Gemini suggests 3 strategies --
    print("\nGemini предлагает стратегии...")
    proposals = agent.suggest_strategies(report, args.task)

    # -- apply all 3 and show comparison --
    # label_col и служебные колонки (_source и т.п.) не трогаем
    protected = [args.label_col] if args.label_col else []
    cleaned = [agent.fix(df, p["strategy"], protected_cols=protected) for p in proposals]

    W = 65
    print("\n" + "=" * W)
    print("  STRATEGY COMPARISON")
    print("=" * W)

    for i, (p, df_c) in enumerate(zip(proposals, cleaned), 1):
        s = p["strategy"]
        print(f"\n  [{i}] {p['name'].upper()}  —  "
              f"missing={s['missing']}, outliers={s['outliers']}, duplicates={s['duplicates']}")
        print(f"  {p.get('reason', '')}")
        print(agent.compare(df, df_c).to_string())
        print()

    # -- human choice --
    print("\n" + "-" * W)
    print("  Выберите стратегию (1 / 2 / 3 / q):")
    choice = input("  > ").strip().lower()
    if choice in ("q", ""):
        print("Cancelled."); raise SystemExit(0)
    idx = (int(choice) - 1) if choice in ("1", "2", "3") else 1
    chosen = proposals[idx]
    df_clean = cleaned[idx]
    print(f"\n  Выбрано: [{idx+1}] {chosen['name']} — {chosen['strategy']}")

    strategy_1 = proposals[0]["strategy"]
    strategy_2 = proposals[1]["strategy"]

    # -- explain --
    justification = ""
    if args.explain:
        print("\nGemini обосновывает выбор...\n")
        justification = agent.explain_with_llm(report, args.task)
        print(justification)

    # -- notebook --
    nb_path = agent.generate_report_notebook(
        df, report, topic_path_resolved, args.task, strategy_1, strategy_2
    )
    if nb_path:
        if justification:
            try:
                import nbformat
                nb = nbformat.read(nb_path.open(), as_version=4)
                nb.cells[-1] = nbformat.v4.new_markdown_cell(
                    f"## Part 3: Justification — Why This Strategy?\n\n{justification}"
                )
                nb_path.write_text(nbformat.writes(nb))
            except Exception:
                pass
        print(f"\n[ok] notebook: {nb_path.relative_to(ROOT)}")

    # -- save clean --
    clean_dir = topic_path_resolved / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    out = clean_dir / "combined_clean.parquet"
    df_clean.to_parquet(out, index=False)
    print(f"[ok] clean data: {out.relative_to(ROOT)}  {df_clean.shape}")


if __name__ == "__main__":
    main()
