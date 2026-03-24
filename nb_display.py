"""Helper: execute a notebook and print all cell outputs to CLI."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def execute_and_show(nb_path: str | Path, rerun: bool = False) -> None:
    """
    Execute a notebook (if rerun=True or no outputs yet) and print all
    text/stream outputs to stdout. Images are reported as file paths.
    """
    nb_path = Path(nb_path).resolve()
    if not nb_path.exists():
        print(f"[!] Notebook not found: {nb_path}")
        return

    nb = json.loads(nb_path.read_text())

    # check if already executed
    has_outputs = any(
        cell.get("outputs") or cell.get("execution_count")
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )

    if rerun or not has_outputs:
        try:
            rel = nb_path.relative_to(ROOT)
        except ValueError:
            rel = nb_path
        print(f"Executing {rel} ...")

        # inject os.chdir(ROOT) as first cell so kernel runs from project root
        import nbformat as nbf
        chdir_cell = nbf.v4.new_code_cell(
            f"import os; os.chdir({str(ROOT)!r})"
        )
        chdir_cell.metadata["tags"] = ["injected-chdir"]
        nb_obj = nbf.reads(nb_path.read_text(), as_version=4)
        nb_obj.cells.insert(0, chdir_cell)
        nb_path.write_text(nbf.writes(nb_obj))

        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert",
             "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=300",
             str(nb_path)],
            capture_output=True, text=True,
        )

        # remove injected cell after execution
        nb_obj = nbf.reads(nb_path.read_text(), as_version=4)
        nb_obj.cells = [c for c in nb_obj.cells
                        if "injected-chdir" not in c.get("metadata", {}).get("tags", [])]
        nb_path.write_text(nbf.writes(nb_obj))

        if result.returncode != 0:
            print(f"  [!] Execution error:\n{result.stderr[-800:]}")
            return
        nb = json.loads(nb_path.read_text())

    print(f"\n{'='*65}")
    print(f"  NOTEBOOK: {nb_path.name}")
    print(f"{'='*65}\n")

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            src = "".join(cell["source"]).strip()
            if src:
                print(src)
                print()
            continue

        # code cell
        src = "".join(cell["source"]).strip()
        if src:
            print(f"[{cell.get('execution_count', '?')}] {src[:120]}{'...' if len(src) > 120 else ''}")

        for out in cell.get("outputs", []):
            otype = out.get("output_type", "")

            if otype in ("stream", "display_data", "execute_result"):
                text = out.get("text") or out.get("data", {}).get("text/plain", [])
                if isinstance(text, list):
                    text = "".join(text)
                if text.strip():
                    print(text.rstrip())

            # images saved inline as base64 — just note the existence
            if "image/png" in out.get("data", {}):
                print("  [image — see plots/ folder]")

            if otype == "error":
                print(f"  [ERROR] {out.get('ename')}: {out.get('evalue')}")

        print()

    print(f"{'='*65}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("notebook")
    p.add_argument("--rerun", action="store_true")
    args = p.parse_args()
    execute_and_show(args.notebook, rerun=args.rerun)
