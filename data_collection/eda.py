#!/usr/bin/env python3
"""
EDA notebook generator.
Scans a directory for parquet files, collects stats, asks Gemini to generate notebooks/eda.ipynb.

Usage:
  python data_collection/eda.py programming_questions
"""

import json, sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent


def _strip_fences(text: str) -> str:
    import re
    if "```" not in text:
        return text
    m = re.search(r"```\w*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else re.sub(r"```\w*", "", text).strip()


def _call_gemini(prompt: str) -> str:
    import os, time
    from google import genai
    import yaml
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = yaml.safe_load((ROOT / "config.yaml").read_text()).get("model", "gemini-2.0-flash-lite")
    for attempt in range(5):
        try:
            return client.models.generate_content(model=model, contents=prompt).text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  [!] Rate limit, жду {wait}с...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini: превышен лимит попыток")


def run_eda(output_dir: str) -> None:
    try:
        import nbformat
    except ImportError:
        print("  [!] nbformat не установлен: pip install nbformat"); return

    topic_path = Path(output_dir) if Path(output_dir).is_absolute() else ROOT / output_dir
    raw_dir = topic_path / "data" / "raw"
    combined = raw_dir / "combined.parquet"
    if combined.exists():
        parquet_files = [combined]
    elif raw_dir.exists():
        parquet_files = list(raw_dir.glob("*.parquet"))
    else:
        parquet_files = list(topic_path.glob("*.parquet"))
    if not parquet_files:
        print(f"  [!] Нет parquet-файлов в {topic_path}"); return

    print(f"\nГенерирую EDA для {len(parquet_files)} датасет(ов)...")
    stats = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            stats.append({
                "path": str(pf.relative_to(ROOT)), "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "head": df.head(2).fillna("").astype(str).to_dict(orient="records"),
                "missing": {c: int(n) for c, n in df.isnull().sum().items() if n > 0},
            })
        except Exception as e:
            print(f"  [!] {pf.name}: {e}")

    if not stats:
        return

    prompt = f"""You are a data scientist. Generate a Jupyter notebook for EDA of these datasets.

Dataset statistics:
{json.dumps(stats, indent=2, ensure_ascii=False)}

Return a JSON array of notebook cells. Each cell:
{{"cell_type": "markdown" | "code", "source": "cell content as string"}}

Notebook structure:
1. Markdown: title + dataset overview
2. Code: imports (pandas, matplotlib, seaborn, pathlib)
3. For each dataset:
   a. Markdown: dataset name and path
   b. Code: df = pd.read_parquet("<path>"); df.shape, df.dtypes, df.head()
   c. Code: df.describe(include="all")
   d. Code: missing values heatmap (seaborn.heatmap of df.isnull())
   e. Code: distribution plots for numeric columns (df.hist())
   f. Code: value_counts for top-3 categorical columns
4. Markdown: summary and next steps

Return ONLY the JSON array of cells, no markdown fences."""

    try:
        cells_data = json.loads(_strip_fences(_call_gemini(prompt)))
        nb = nbformat.v4.new_notebook()
        for c in cells_data:
            fn = nbformat.v4.new_markdown_cell if c.get("cell_type") == "markdown" else nbformat.v4.new_code_cell
            nb.cells.append(fn(c["source"]))
        nb_path = topic_path / "notebooks" / "eda.ipynb"
        nb_path.parent.mkdir(exist_ok=True)
        nb_path.write_text(nbformat.writes(nb))
        print(f"  [ok] {nb_path.relative_to(ROOT)} ({len(nb.cells)} ячеек)")
    except Exception as e:
        print(f"  [!] Ошибка генерации EDA: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python data_collection/eda.py <output_dir>")
        print(f"Example: python data_collection/eda.py programming_questions")
        sys.exit(1)
    run_eda(sys.argv[1])
