#!/usr/bin/env python3
"""
DataQualityAgent — Data Detective

Detects and fixes data quality issues:
missing values, duplicates, outliers, class imbalance.

Usage:
    from agents.data_quality_agent import DataQualityAgent

    agent = DataQualityAgent()
    report = agent.detect_issues(df)
    df_clean = agent.fix(df, strategy={'missing': 'median', 'duplicates': 'drop', 'outliers': 'clip_iqr'})
    comparison = agent.compare(df, df_clean)
    explanation = agent.explain_with_llm(report, task_description="...")  # bonus
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# REPORT TYPES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    missing: dict[str, dict]       # col → {count, pct}
    duplicates: int
    outliers: dict[str, dict]      # col → {count, pct, lower_bound, upper_bound}
    imbalance: dict[str, Any]      # {col, counts, ratio}
    summary: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            "═" * 60,
            "  QUALITY REPORT",
            "═" * 60,
            f"  Пропуски:      {self.summary['missing_cells']} ячеек в {self.summary['missing_cols']} колонках",
            f"  Дубликаты:     {self.duplicates} строк",
            f"  Выбросы:       {self.summary['outlier_cells']} значений в {self.summary['outlier_cols']} колонках",
        ]
        if self.imbalance:
            lines.append(f"  Дисбаланс:     ratio {self.imbalance.get('ratio', '?')} (max/min класс)")
        lines.append("═" * 60)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────────────────────────────────────

class DataQualityAgent:

    # ── skill 1: detect ───────────────────────────────────────────────────────

    def detect_issues(self, df: pd.DataFrame, label_col: str | None = None) -> QualityReport:
        """Detect missing values, duplicates, outliers and class imbalance."""
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
        return QualityReport(
            missing=missing,
            duplicates=duplicates,
            outliers=outliers,
            imbalance=imbalance,
            summary=summary,
        )

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
            else:  # z-score
                low  = series.mean() - 3 * series.std()
                high = series.mean() + 3 * series.std()
            mask = (df[col] < low) | (df[col] > high)
            n = int(mask.sum())
            if n > 0:
                result[col] = {
                    "count": n,
                    "pct": round(n / len(df) * 100, 2),
                    "lower_bound": round(float(low), 4),
                    "upper_bound": round(float(high), 4),
                }
        return result

    def _detect_imbalance(self, df: pd.DataFrame, label_col: str | None) -> dict:
        if label_col is None:
            candidates = [
                c for c in df.columns
                if df[c].dtype == "object" and 2 <= df[c].nunique() <= 20
            ]
            if not candidates:
                return {}
            label_col = candidates[0]
        counts = df[label_col].value_counts().to_dict()
        values = list(counts.values())
        ratio = round(max(values) / min(values), 2) if min(values) > 0 else float("inf")
        return {"col": label_col, "counts": counts, "ratio": ratio}

    # ── skill 2: fix ──────────────────────────────────────────────────────────

    def fix(self, df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        """
        Apply cleaning strategies to the DataFrame.

        strategy keys:
          missing:    'median' | 'mean' | 'mode' | 'drop' | 'ffill' | 'constant:<value>'
          duplicates: 'drop' | 'keep_first' | 'keep_last' | 'none'
          outliers:   'clip_iqr' | 'clip_zscore' | 'drop' | 'none'
        """
        result = df.copy()
        if "duplicates" in strategy:
            result = self._fix_duplicates(result, strategy["duplicates"])
        if "missing" in strategy:
            result = self._fix_missing(result, strategy["missing"])
        if "outliers" in strategy:
            result = self._fix_outliers(result, strategy["outliers"])
        return result

    def _fix_missing(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "drop":
            return df.dropna()
        if strategy == "ffill":
            return df.ffill()
        if strategy.startswith("constant:"):
            value = strategy.split(":", 1)[1]
            return df.fillna(value)
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                if strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
        for col in df.select_dtypes(include=["object"]).columns:
            if df[col].isna().any():
                fill = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(fill)
        return df

    def _fix_duplicates(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "none":
            return df
        keep = {"drop": "first", "keep_first": "first", "keep_last": "last"}.get(strategy, "first")
        return df.drop_duplicates(keep=keep)

    def _fix_outliers(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        if strategy == "none":
            return df
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            low_iqr,  high_iqr  = q1 - 1.5 * iqr,          q3 + 1.5 * iqr
            low_z,    high_z    = series.mean() - 3 * series.std(), series.mean() + 3 * series.std()
            if strategy == "clip_iqr":
                df[col] = df[col].clip(low_iqr, high_iqr)
            elif strategy == "clip_zscore":
                df[col] = df[col].clip(low_z, high_z)
            elif strategy == "drop":
                mask = (df[col] >= low_iqr) & (df[col] <= high_iqr)
                df = df[mask | df[col].isna()]
        return df

    # ── skill 3: compare ──────────────────────────────────────────────────────

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """Return a before/after comparison table for all quality metrics."""

        def count_outliers(df: pd.DataFrame) -> int:
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
            ("Строк",                  len(df_before),                       len(df_after)),
            ("Пропущенных значений",   int(df_before.isna().sum().sum()),    int(df_after.isna().sum().sum())),
            ("Дубликатов",             int(df_before.duplicated().sum()),    int(df_after.duplicated().sum())),
            ("Выбросов (IQR)",         count_outliers(df_before),            count_outliers(df_after)),
        ]

        records = []
        for metric, before, after in rows:
            delta = after - before
            pct   = f"{delta / before * 100:+.1f}%" if before != 0 else "—"
            records.append({"Метрика": metric, "До": before, "После": after, "Дельта": delta, "% изменение": pct})

        return pd.DataFrame(records).set_index("Метрика")

    # ── bonus: LLM explain ────────────────────────────────────────────────────

    def explain_with_llm(self, report: QualityReport, task_description: str) -> str:
        """Use Gemini to explain issues and recommend cleaning strategy."""
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return "⚠️  Нет GEMINI_API_KEY: export GEMINI_API_KEY=AIza..."

        client = genai.Client(api_key=api_key)

        prompt = f"""Ты — эксперт по качеству данных. Проанализируй отчёт и дай рекомендации.

ML-задача: {task_description}

Отчёт о качестве данных:
{json.dumps(report.to_dict(), indent=2, ensure_ascii=False)}

Ответь по структуре:
1. **Найденные проблемы** — объясни каждую проблему и её влияние на ML-задачу
2. **Рекомендуемая стратегия** — dict для метода fix() с обоснованием каждого выбора
3. **Критические vs минорные** — что нужно исправить обязательно, а что опционально

Будь конкретен и практичен. Ответ на русском."""

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
        )
        return response.text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="DataQualityAgent — обнаружение и устранение проблем качества данных",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python agents/data_quality_agent.py                          # все файлы из data/raw/
  python agents/data_quality_agent.py data/raw/huggingface/slug/data.parquet
  python agents/data_quality_agent.py data.parquet --strategy median --outliers clip_iqr
  python agents/data_quality_agent.py data.parquet --explain --task "классификация текста"
        """
    )
    parser.add_argument("file",            nargs="?", default=None, help="Путь к parquet-файлу (default: все файлы из data/raw/)")
    parser.add_argument("--strategy",      default="median",   choices=["median", "mean", "mode", "drop", "ffill"], help="Стратегия для пропусков (default: median)")
    parser.add_argument("--duplicates",    default="drop",     choices=["drop", "keep_first", "keep_last", "none"],  help="Стратегия для дубликатов (default: drop)")
    parser.add_argument("--outliers",      default="clip_iqr", choices=["clip_iqr", "clip_zscore", "drop", "none"],  help="Стратегия для выбросов (default: clip_iqr)")
    parser.add_argument("--label-col",     default=None,       help="Колонка с метками классов (для анализа дисбаланса)")
    parser.add_argument("--explain",       action="store_true", help="Запросить объяснение у Gemini")
    parser.add_argument("--task",          default="ML task",  help="Описание ML-задачи для Gemini (используется с --explain)")
    parser.add_argument("--output",        default=None,       help="Куда сохранить очищенный файл (default: <file>_clean.parquet)")
    args = parser.parse_args()

    data_raw = Path(__file__).parent.parent / "data" / "raw"
    if args.file is None:
        paths = list(data_raw.rglob("*.parquet"))
        if not paths:
            print(f"❌ Нет parquet-файлов в {data_raw}")
            raise SystemExit(1)
        print(f"📁 data/raw/ — найдено файлов: {len(paths)}")
        for i, p in enumerate(paths, 1):
            print(f"   [{i}] {p.relative_to(data_raw)}")
        print()
    else:
        path = Path(args.file)
        if not path.exists():
            print(f"❌ Файл не найден: {path}")
            raise SystemExit(1)
        paths = [path]

    dfs = []
    for p in paths:
        print(f"📂 Загружаю: {p}")
        dfs.append(pd.read_parquet(p))

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    print(f"   Shape: {df.shape}\n")

    agent = DataQualityAgent()

    # detect
    report = agent.detect_issues(df, label_col=args.label_col)
    print(report)

    # explain (optional)
    if args.explain:
        print("\n🤖 Gemini анализирует...\n")
        print(agent.explain_with_llm(report, args.task))

    # fix
    strategy = {
        "missing":    args.strategy,
        "duplicates": args.duplicates,
        "outliers":   args.outliers,
    }
    print(f"\n🔧 Применяю стратегию: {strategy}")
    df_clean = agent.fix(df, strategy=strategy)

    # compare
    print("\n📊 Сравнение до/после:")
    print(agent.compare(df, df_clean).to_string())

    # save
    data_clean = Path(__file__).parent.parent / "data" / "clean"
    if args.output:
        out = Path(args.output)
    else:
        rel = paths[0].relative_to(data_raw)
        out = data_clean / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(out, index=False)
    print(f"\n✅ Сохранено: {out}  {df_clean.shape}")


if __name__ == "__main__":
    main()
