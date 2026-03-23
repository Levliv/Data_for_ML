#!/usr/bin/env python3
"""
EDA pipeline:
  1. Send topic + schema + 5 sample rows to Gemini
  2. Gemini writes a Jupyter notebook (decides what to compute)
  3. Execute notebook on full data
  4. Print outputs to CLI

Usage:
  python data_collection/eda.py <topic_dir> [topic description]
"""

import json, sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
AGENT_COLS = {"_source", "_dataset_id"}
W = 65


def _strip_fences(text: str) -> str:
    import re
    if "```" not in text:
        return text
    m = re.search(r"```\w*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else re.sub(r"```\w*", "", text).strip()


def _call_gemini(prompt: str) -> str:
    import os, time, yaml
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = yaml.safe_load((ROOT / "config.yaml").read_text()).get("model", "gemini-2.0-flash-lite")
    for attempt in range(5):
        try:
            return client.models.generate_content(model=model, contents=prompt).text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"  [!] Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini: too many retries")


def _ask_gemini_for_notebook(topic: str, schema: dict, sample: list, parquet_path: str) -> list:
    prompt = f"""You are a senior data scientist. ML task / topic: "{topic}"

You see only 5 sample rows and the column schema. Based on this, decide what statistics,
distributions, and plots are most valuable for understanding this dataset for the given task.

Column schema (name -> dtype):
{json.dumps(schema, indent=2)}

5 sample rows:
{json.dumps(sample, indent=2, ensure_ascii=False, default=str)}

Write a complete Jupyter notebook that:
1. Loads the full dataset: df = pd.read_parquet("{parquet_path}")
2. Computes the statistics YOU decide are most relevant for "{topic}"
3. Creates plots (matplotlib/seaborn) for the most important distributions
4. Prints key findings using print() so they appear in cell outputs
5. Ends with a markdown cell: "## Conclusions" with bullet-point findings

Rules:
- Use only: pandas, matplotlib, seaborn, numpy (standard libs)
- Every important number must be printed, not just plotted
- The conclusions markdown cell must summarize findings in plain text

Return a JSON array of cells:
[{{"cell_type": "markdown"|"code", "source": "cell content as string"}}]

Return ONLY the JSON array, no markdown fences."""

    raw = _strip_fences(_call_gemini(prompt))
    return json.loads(raw)


def _build_notebook(cells_data: list) -> "nbformat.NotebookNode":
    import nbformat
    nb = nbformat.v4.new_notebook()
    for c in cells_data:
        fn = nbformat.v4.new_markdown_cell if c.get("cell_type") == "markdown" else nbformat.v4.new_code_cell
        nb.cells.append(fn(c["source"]))
    return nb


def _execute_and_print(nb: "nbformat.NotebookNode", topic: str, nb_path: Path) -> None:
    """Execute notebook cells directly via exec(), print text outputs to CLI."""
    import io, matplotlib
    matplotlib.use("Agg")  # no GUI, save plots to files

    namespace: dict = {"__builtins__": __builtins__}
    plot_dir = nb_path.parent / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_count = [0]

    # patch plt.show() to save instead of display
    exec(
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import pandas as pd\n"
        "import numpy as np",
        namespace,
    )

    original_show = namespace["plt"].show

    def _save_show():
        plot_count[0] += 1
        path = plot_dir / f"plot_{plot_count[0]:02d}.png"
        namespace["plt"].savefig(path, bbox_inches="tight", dpi=100)
        namespace["plt"].clf()
        print(f"  [plot saved] {path.relative_to(ROOT)}")

    namespace["plt"].show = _save_show

    print(f"\n{'=' * W}")
    print(f"  EDA RESULTS: {topic.upper()}")
    print(f"{'=' * W}")

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            src = cell.source.strip()
            if src:
                print(f"\n{src}")
        elif cell.cell_type == "code":
            buf = io.StringIO()
            try:
                import contextlib
                with contextlib.redirect_stdout(buf):
                    exec(cell.source, namespace)
                out = buf.getvalue()
                if out.strip():
                    print(out.rstrip())
            except Exception as e:
                print(f"  [!] Cell error: {e}")

    print(f"\n{'=' * W}\n")


def run_eda(output_dir: str, topic: str = "") -> None:
    try:
        import nbformat
    except ImportError:
        print("  [!] nbformat not installed: pip install nbformat nbconvert"); return

    topic_path = Path(output_dir) if Path(output_dir).is_absolute() else ROOT / output_dir
    raw_dir = topic_path / "data" / "raw"
    combined = raw_dir / "combined.parquet"

    if combined.exists():
        pf = combined
    else:
        files = list(raw_dir.glob("*.parquet")) if raw_dir.exists() else list(topic_path.glob("*.parquet"))
        if not files:
            print(f"  [!] No parquet files in {topic_path}"); return
        pf = files[0]

    effective_topic = topic or topic_path.name
    print(f"\nEDA: {pf.name} | topic: {effective_topic}")

    try:
        df = pd.read_parquet(pf)
    except Exception as e:
        print(f"  [!] {e}"); return

    user_cols = [c for c in df.columns if c not in AGENT_COLS]
    df_u = df[user_cols]
    schema = {c: str(df_u[c].dtype) for c in user_cols}
    sample = (df_u.sample(min(5, len(df_u)), random_state=42)
              .fillna("").astype(str).to_dict(orient="records"))

    print("  Asking Gemini to design EDA...")
    try:
        cells_data = _ask_gemini_for_notebook(
            effective_topic, schema, sample, str(pf.relative_to(ROOT))
        )
    except Exception as e:
        print(f"  [!] Gemini error: {e}"); return

    nb = _build_notebook(cells_data)
    nb_path = topic_path / "notebooks" / "eda.ipynb"
    nb_path.parent.mkdir(exist_ok=True)
    nb_path.write_text(nbformat.writes(nb))
    print(f"  [ok] notebook saved: {nb_path.relative_to(ROOT)} ({len(nb.cells)} cells)")

    print("  Executing notebook on full data...")
    _execute_and_print(nb, effective_topic, nb_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_collection/eda.py <topic_dir> [topic description]")
        print("Example: python data_collection/eda.py toxic_comment 'toxic comment classification'")
        sys.exit(1)
    run_eda(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
