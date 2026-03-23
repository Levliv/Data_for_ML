#!/usr/bin/env python3
"""
Dataset Search Agent - Gemini Edition
3 acquisition methods: open datasets, APIs, scraping.

Commands:
  python agents/data_collection_agent.py search "climate weather"
  python agents/data_collection_agent.py download datasets_found.json
"""

import json, sys, time, argparse, subprocess, requests, os, re
import yaml, pandas as pd
from collections import Counter
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import list_datasets
from google import genai
from google.genai import types

load_dotenv()

ROOT       = Path(__file__).parent.parent
_cfg       = yaml.safe_load((ROOT / "config.yaml").read_text())
TOPIC      = _cfg.get("topic", "climate weather")
OUTPUT_DIR = str(ROOT / _cfg.get("output_dir", "."))
MODEL      = _cfg.get("model", "gemini-3.1-flash-lite-preview")

GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")
KAGGLE_USER  = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY   = os.getenv("KAGGLE_KEY", "")
KAGGLE_TOKEN = os.getenv("KAGGLE_API_TOKEN", "")

if not GEMINI_KEY:
    print("[!] Нет GEMINI_API_KEY\n   export GEMINI_API_KEY=your_key")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_KEY)

SYSTEM_PROMPT = """You are a Dataset Curation Agent.

Given raw dataset metadata from multiple sources:
1. Deduplicate (same dataset may appear from multiple sources)
2. Rank by relevance to the user's topic
3. Add ML task labels and a short description

Output ONLY a valid JSON array, nothing else. No markdown, no explanation.

[
  {
    "id": "unique-id",
    "name": "Human readable name",
    "source": "huggingface | kaggle | zenodo | uci | google",
    "url": "https://...",
    "downloads": 0,
    "likes": 0,
    "size_category": "1M<n<10M",
    "license": "cc-by-4.0",
    "description": "2 sentences about what data and what ML tasks it suits.",
    "ml_tasks": ["forecasting", "regression"],
    "relevance_score": 9
  }
]

Sort by relevance_score descending.
Drop datasets with relevance_score below 4 — do not include irrelevant results."""


def _gemini(prompt: str, system: bool = True, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            kw = {"config": types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.1)} if system else {}
            return client.models.generate_content(model=MODEL, contents=prompt, **kw).text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 30 * (attempt + 1)
                print(f"\n  [!] Rate limit, жду {wait}с (попытка {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini: превышен лимит попыток")


def _strip_fences(text: str) -> str:
    if "```" not in text:
        return text
    m = re.search(r"```\w*\n?(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else re.sub(r"```\w*", "", text).strip()


# ------------------------------------------------------------------------------
# DATA COLLECTION AGENT
# ------------------------------------------------------------------------------

class DataCollectionAgent:

    def fetch_api(self, endpoint: str, params: dict, source: str = "") -> pd.DataFrame:
        _labels = {"zenodo": "Zenodo API", "uci": "UCI ML Repository API"}
        print(f"  [{_labels.get(source, source or endpoint)}]", end=" ", flush=True)
        try:
            data = requests.get(endpoint, params=params, timeout=15).json()
            rows = []
            if source == "uci":
                for ds in (data.get("datasets") or data if isinstance(data, list) else []):
                    ds_id = ds.get("id", ds.get("ucimlId", ""))
                    rows.append({
                        "id": f"uci:{ds_id}", "source": "uci",
                        "url": f"https://archive.ics.uci.edu/dataset/{ds_id}",
                        "downloads": ds.get("numHits", 0),
                        "likes": 0,
                        "size_category": f"{ds.get('numInstances', '?')} rows",
                        "license": ds.get("license", "unknown"),
                        "name": ds.get("name", ""),
                        "description_raw": (ds.get("abstract") or "")[:200],
                        "tags": ds.get("tasks", []),
                    })
            elif source == "zenodo":
                for rec in data.get("hits", {}).get("hits", []):
                    meta = rec.get("metadata", {})
                    size_bytes = sum(f.get("size", 0) for f in rec.get("files", []))
                    rows.append({
                        "id": f"zenodo:{rec.get('id')}", "source": "zenodo",
                        "url": rec.get("links", {}).get("html", ""),
                        "downloads": rec.get("stats", {}).get("downloads", 0),
                        "likes": rec.get("stats", {}).get("views", 0),
                        "size_category": f"{size_bytes // 1_000_000} MB",
                        "license": (meta.get("license") or {}).get("id", "unknown"),
                        "name": meta.get("title", ""),
                        "description_raw": (meta.get("description") or "")[:200],
                        "tags": meta.get("keywords", []),
                    })
            print(f"-> {len(rows)}")
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[!] {e}")
            return pd.DataFrame()

    def load_dataset(self, query: str, source: str = "hf", limit: int = 50,
                     extra_terms: list[str] = None) -> pd.DataFrame:
        if source == "hf":
            terms = [query] + (extra_terms or [])
            seen_ids: set = set()
            all_rows = []
            for term in terms:
                print(f"  [HuggingFace] '{term}'", end=" ", flush=True)
                try:
                    rows = []
                    for ds in list_datasets(search=term, limit=limit, sort="downloads", full=True):
                        if ds.id in seen_ids:
                            continue
                        seen_ids.add(ds.id)
                        tags = getattr(ds, "tags", []) or []
                        size_bytes = sum(getattr(s, "size", 0) or 0 for s in (getattr(ds, "siblings", []) or []))
                        size_mb = size_bytes / (1024 * 1024)
                        size_disk = f"{size_mb:.1f} MB" if 0 < size_mb < 1024 else (f"{size_mb/1024:.2f} GB" if size_mb >= 1024 else None)
                        row_count = next((t.replace("size_categories:", "") for t in tags if "size_categories:" in t), None)
                        parts = [p for p in [size_disk, row_count] if p]
                        rows.append({
                            "id": ds.id, "source": "huggingface",
                            "url": f"https://huggingface.co/datasets/{ds.id}",
                            "downloads": getattr(ds, "downloads", 0) or 0,
                            "likes": getattr(ds, "likes", 0) or 0,
                            "size_category": " | ".join(parts) if parts else "unknown",
                            "license": next((t.replace("license:", "") for t in tags if t.startswith("license:")), "unknown"),
                            "tags": tags,
                        })
                    print(f"-> {len(rows)}")
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"[!] {e}")
            return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

        elif source == "kaggle":
            print("  [Kaggle API]", end=" ", flush=True)
            if KAGGLE_TOKEN:
                headers, auth = {"Authorization": f"Bearer {KAGGLE_TOKEN}"}, None
            elif KAGGLE_USER and KAGGLE_KEY:
                headers, auth = {}, (KAGGLE_USER, KAGGLE_KEY)
            else:
                print("skipped (set KAGGLE_API_TOKEN or KAGGLE_USERNAME + KAGGLE_KEY)")
                return pd.DataFrame()
            try:
                resp = requests.get(
                    "https://www.kaggle.com/api/v1/datasets/list",
                    params={"search": query, "sortBy": "votes", "pageSize": limit},
                    headers=headers, auth=auth, timeout=15,
                )
                resp.raise_for_status()
                rows = []
                for ds in resp.json():
                    ref = (ds.get("ref") or f"{ds.get('ownerRef','')}/{ds.get('datasetSlug','')}").strip("/")
                    if not ref:
                        continue
                    rows.append({
                        "id": ref, "name": ds.get("title", ref), "source": "kaggle",
                        "url": f"https://www.kaggle.com/datasets/{ref}",
                        "downloads": ds.get("downloadCount", ds.get("totalDownloads", 0)),
                        "likes": ds.get("voteCount", ds.get("totalVotes", 0)),
                        "size_category": f"{ds.get('totalBytes', 0) // 1_000_000} MB",
                        "license": ds.get("licenseName", "unknown"),
                        "tags": [t["name"] for t in ds.get("tags", [])],
                    })
                print(f"-> {len(rows)}")
                return pd.DataFrame(rows)
            except Exception as e:
                print(f"[!] {e}")
                return pd.DataFrame()

        return pd.DataFrame()

    def fetch_generic(self, source: dict, query: str) -> pd.DataFrame:
        """Fetch any JSON API source suggested by Gemini."""
        name = source.get("name", "unknown")
        print(f"  [{name}]", end=" ", flush=True)
        try:
            url = source["search_url"].replace("{query}", requests.utils.quote(query))
            params = {k: (v.replace("{query}", query) if isinstance(v, str) else v)
                      for k, v in source.get("params", {}).items()}
            headers = source.get("headers", {})
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "html" in ct or (not resp.text.strip().startswith(("{", "["))):
                print(f"[!] {name} returned HTML (SPA), skipping")
                return pd.DataFrame()
            data = resp.json()

            # navigate to result list using dot-path e.g. "hits.hits"
            result_path = source.get("result_path", "")
            items = data
            for key in (result_path.split(".") if result_path else []):
                if isinstance(items, dict):
                    items = items.get(key, [])
            if not isinstance(items, list):
                items = [items] if isinstance(items, dict) else []

            field_map = source.get("field_map", {})
            rows = []
            for item in items:
                row = {
                    "source": source.get("name", "custom").lower().replace(" ", "_"),
                    "downloads": 0, "likes": 0, "size_category": "unknown",
                    "license": "unknown", "tags": [],
                }
                for our_field, their_field in field_map.items():
                    val = item.get(their_field, "")
                    if val:
                        row[our_field] = str(val)
                if "id" not in row:
                    row["id"] = f"{row['source']}:{hash(str(item)) % 100000}"
                rows.append(row)
            print(f"-> {len(rows)}")
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"[!] {e}")
            return pd.DataFrame()

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        non_empty = [df for df in sources if df is not None and not df.empty]
        return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


# ------------------------------------------------------------------------------
# SOURCE DISCOVERY + SEARCH
# ------------------------------------------------------------------------------

def _expand_query(query: str) -> list[str]:
    prompt = f"""Generate search query variations for finding datasets on HuggingFace Hub.

Original query: "{query}"

Return a JSON array of 5-7 short search terms that would find semantically related datasets
that might use different terminology, abbreviations, or domain-specific names.

Example: "toxic comment" -> ["hate speech", "offensive language", "abusive text", "online harassment", "content moderation"]

Return ONLY a JSON array of strings, no explanation."""
    try:
        raw = _strip_fences(_gemini(prompt, system=False))
        terms = json.loads(raw)
        terms = [query] + [t for t in terms if t.lower() != query.lower()]
        return terms[:7]
    except Exception:
        return [query]


def _think_sources(query: str) -> list[dict]:
    print("Gemini выбирает источники...")
    prompt = f"""You are a dataset discovery expert. For the ML topic: "{query}"

Think about which sources are most likely to have relevant datasets.
Consider: general repositories, domain-specific archives, government portals, academic data sources.

Return a JSON array of sources to search. Each source:
{{
  "name": "Human readable name",
  "type": "huggingface" | "kaggle" | "zenodo" | "api",
  "reason": "one sentence why this source is relevant for the topic"
}}

For type "api" (any other REST API with JSON response), also include:
{{
  "search_url": "https://... (use {{query}} as placeholder for search term)",
  "params": {{"key": "value or {{query}}"}},
  "headers": {{}},
  "result_path": "dot.path.to.results.array in JSON response",
  "field_map": {{
    "id": "their_id_field",
    "name": "their_name_field",
    "url": "their_url_field",
    "description_raw": "their_description_field"
  }}
}}

Rules:
- Only suggest sources with FREE public access (no auth required for basic search)
- Always include huggingface and zenodo
- Add domain-specific sources relevant to "{query}"
- Maximum 6 sources total

Return ONLY the JSON array, no markdown fences."""

    try:
        raw = _strip_fences(_gemini(prompt, system=False))
        sources = json.loads(raw)
        for s in sources:
            print(f"  + {s['name']} ({s['type']}) — {s.get('reason','')}")
        return sources
    except Exception as e:
        print(f"  [!] Gemini error, using defaults: {e}")
        return [
            {"name": "HuggingFace", "type": "huggingface"},
            {"name": "Zenodo",      "type": "zenodo"},
        ]


def full_search(query: str) -> list[dict]:
    print(f"\nТема: '{query}'\n")
    sources = _think_sources(query)

    print("\nGemini расширяет поисковые запросы для HuggingFace...")
    hf_terms = _expand_query(query)
    print(f"  Термины: {hf_terms}\n")

    agent = DataCollectionAgent()
    dfs = []

    for source in sources:
        stype = source.get("type")
        if stype == "huggingface":
            dfs.append(agent.load_dataset(query, source="hf", limit=30, extra_terms=hf_terms[1:]))
        elif stype == "kaggle":
            dfs.append(agent.load_dataset(query, source="kaggle", limit=30))
        elif stype == "zenodo":
            dfs.append(agent.fetch_api(
                "https://zenodo.org/api/records",
                {"q": query, "type": "dataset", "size": 20, "sort": "mostviewed"},
                source="zenodo"
            ))
        elif stype == "api":
            dfs.append(agent.fetch_generic(source, query))

    candidates = agent.merge(dfs).to_dict("records")
    print(f"\n   Всего кандидатов: {len(candidates)}")
    return candidates


def rank_with_agent(candidates: list[dict], topic: str) -> list[dict]:
    print(f"\nGemini ранжирует {len(candidates)} кандидатов...")
    CHUNK, all_ranked = 50, []
    for i in range(0, len(candidates), CHUNK):
        chunk = candidates[i:i + CHUNK]
        try:
            text = _gemini(f'Topic: "{topic}"\n\nDatasets:\n{json.dumps(chunk, ensure_ascii=False)}\n\nReturn ONLY JSON array.')
            start, end = text.find("["), text.rfind("]") + 1
            if start == -1 or end == 0:
                print("  [!] JSON массив не найден в ответе"); continue
            try:
                from json_repair import repair_json
                all_ranked.extend(json.loads(repair_json(text[start:end])))
            except ImportError:
                all_ranked.extend(json.loads(text[start:end]))
        except Exception as e:
            print(f"  [!] Ошибка в chunk {i//CHUNK + 1}: {e}")
    ranked = sorted(all_ranked, key=lambda x: x.get("relevance_score", 0), reverse=True)
    filtered = [d for d in ranked if d.get("relevance_score", 0) >= 4]
    dropped = len(ranked) - len(filtered)
    if dropped:
        print(f"  Отброшено нерелевантных: {dropped}")
    return filtered


# ------------------------------------------------------------------------------
# CONFIG + DOWNLOAD SCRIPT
# ------------------------------------------------------------------------------

def generate_config(query: str, ranked: list[dict], output_dir: str = None) -> None:
    print("\nGemini обновляет config.yaml...")
    cfg_path = Path(output_dir) / "config.yaml" if output_dir else ROOT / "config.yaml"
    sources_stats = dict(Counter(ds.get("source", "unknown") for ds in ranked))
    top_tasks = [t for t, _ in Counter(t for ds in ranked for t in ds.get("ml_tasks", [])).most_common(5)]
    avg_score = sum(ds.get("relevance_score", 0) for ds in ranked) / max(len(ranked), 1)
    top3 = [{"name": d.get("name", ""), "source": d.get("source", ""), "score": d.get("relevance_score", 0)} for d in ranked[:3]]
    current_cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    prompt = f"""You are a config generator for a dataset search agent.

User query: "{query}"
Search results summary:
- Total datasets found: {len(ranked)}
- Sources breakdown: {json.dumps(sources_stats)}
- Average relevance score: {avg_score:.1f}/10
- Top ML tasks: {top_tasks}
- Top 3 datasets: {json.dumps(top3)}

Current config:
{yaml.dump(current_cfg, allow_unicode=True)}

Generate an updated config.yaml. Rules:
- Keep: output_dir, state_file, monitor_interval, model (do not change these)
- Update: topic (use the query)
- Add: last_search (ISO timestamp {datetime.now().isoformat(timespec='seconds')}), sources_stats, avg_relevance_score, top_ml_tasks
- Add: preferred_sources (list of sources with >0 results, sorted by count descending)
- Add: search_limits per source based on what was productive (sources with 0 results -> limit 0)

Return ONLY valid YAML, no markdown fences, no explanation."""

    try:
        new_cfg = yaml.safe_load(_strip_fences(call_gemini_plain(prompt)))
        if not isinstance(new_cfg, dict):
            raise ValueError("Gemini вернул не словарь")
        for key in ("output_dir", "state_file", "monitor_interval", "model"):
            if key in current_cfg:
                new_cfg[key] = current_cfg[key]
        cfg_path.write_text(yaml.dump(new_cfg, allow_unicode=True, sort_keys=False))
        print(f"  [ok] {cfg_path} обновлён (тема: '{query}', источников: {len(sources_stats)})")
    except Exception as e:
        print(f"  [!] Не удалось обновить config.yaml: {e}")


def call_gemini_plain(prompt: str) -> str:
    return _gemini(prompt, system=False)


def _fallback_script(datasets: list[dict], rel_output_dir: str) -> str:
    lines = ["from pathlib import Path",
             "from datasets import load_dataset",
             "import os, requests, subprocess, pandas as pd",
             "ROOT = Path(__file__).parent",
             "OUT = ROOT / 'data' / 'raw'", "OUT.mkdir(parents=True, exist_ok=True)", ""]
    for ds in datasets:
        src, ds_id = ds.get("source", ""), ds.get("id", "")
        name = ds.get("name", ds_id).lower().replace(" ", "_").replace("/", "_").replace(":", "_")
        out_file = f"OUT / '{name}.parquet'"
        if src == "huggingface":
            lines += [f"print('Downloading {ds_id}...')", "try:",
                      f"    d = load_dataset('{ds_id}')",
                      "    frames = [d[s].to_pandas() for s in d]",
                      "    df = pd.concat(frames, ignore_index=True)",
                      f"    df['_source'] = 'huggingface'", f"    df['_dataset_id'] = '{ds_id}'",
                      f"    df.to_parquet({out_file}, index=False)",
                      f"    print(f'  [ok] {out_file}')",
                      "except Exception as e:", "    print(f'  [!] {e}')", ""]
        elif src == "kaggle":
            lines += [f"print('Downloading {ds_id} via kaggle CLI...')",
                      f"subprocess.run(['kaggle','datasets','download','-d','{ds_id}','-p',str(OUT),'--unzip'], check=False)", ""]
        else:
            url = ds.get("url", "")
            lines += [f"print('Manual download: {url}')", ""]
    lines += ["print('Done.')"]
    return "\n".join(lines)


def generate_download_script(datasets: list[dict], output_dir: str) -> str:
    print("\nGemini пишет скрипт скачивания...")
    rel_output_dir = str(Path(output_dir).relative_to(ROOT))

    prompt = f"""Write a Python download script for these datasets:

{json.dumps(datasets, indent=2, ensure_ascii=False)}

IMPORTANT: Use only relative paths. The script lives inside the topic directory.
Start the script with:
  from pathlib import Path
  ROOT = Path(__file__).parent   # = <topic>/
  OUT = ROOT / 'data' / 'raw'   # parquet files go into <topic>/data/raw/
  OUT.mkdir(parents=True, exist_ok=True)

UNIFIED OUTPUT FORMAT - all datasets must be saved as a single Parquet file per dataset:
  - Final file path: OUT / '<dataset_name>.parquet'
  - Use the dataset name (slug, lowercase, spaces->underscores) as the filename
  - Use pandas DataFrame as the unified intermediate format
  - Add a "_source" column with the source name (e.g. "huggingface", "kaggle", "zenodo")
  - Add a "_dataset_id" column with the dataset id string

Per-source download + conversion rules:

HuggingFace:
  - load_dataset(id) -> iterate all splits -> pd.concat all splits into one DataFrame -> save as parquet
  - Use: ds[split].to_pandas() for each split

Kaggle:
  - kaggle datasets download -d <id> -p <tmp_dir> --unzip
  - Find all .csv / .json / .parquet files in tmp_dir
  - pd.read_csv / pd.read_json / pd.read_parquet -> pd.concat -> save as parquet

Zenodo (EXACT REST API v2 structure):
  - Step 1: extract record_id from URL last path segment (e.g. "7734140")
  - Step 2: GET https://zenodo.org/api/records/<record_id>
  - Step 3: files = response.json()["files"]
  - Step 4: filename = file["key"], download_url = file["links"]["self"] + "/content"
  - Step 5: stream download to tmp file, then read into DataFrame based on extension
  - IMPORTANT: file["links"]["self"] = "https://zenodo.org/api/records/<id>/files/<key>"
  - DO NOT use file["links"]["content"] - that key does not exist at record level

UCI / others:
  - requests streaming download -> read into DataFrame -> save as parquet

General rules:
  - import pandas as pd, import pyarrow
  - Progress output and per-dataset error handling
  - Final summary: Successful / Failed
  - if __name__ == "__main__": guard

Return ONLY the Python script, no markdown fences."""

    script = _strip_fences(call_gemini_plain(prompt)).strip()
    if not script:
        print("  [!] Gemini вернул пустой скрипт, генерирую базовый...")
        script = _fallback_script(datasets, rel_output_dir)
    return script


def generate_unify_script(topic_dir: str, topic: str) -> str:
    print("\nGemini пишет скрипт унификации...")
    raw_dir = Path(topic_dir) / "data" / "raw"
    source_files = [f for f in raw_dir.glob("*.parquet")
                    if not f.stem.endswith("_unified") and f.name != "combined.parquet"]
    if not source_files:
        print("  [!] Нет исходных parquet-файлов для унификации")
        return ""

    schemas = []
    for pf in source_files:
        try:
            df = pd.read_parquet(pf)
            schemas.append({
                "file": pf.name,
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "shape": list(df.shape),
                "sample": df.head(1).fillna("").astype(str).to_dict(orient="records"),
            })
        except Exception as e:
            print(f"  [!] {pf.name}: {e}")

    if not schemas:
        return ""

    prompt = f"""Write a Python script that unifies multiple parquet files into a common schema.

Topic: "{topic}"
Files in data/raw/:
{json.dumps(schemas, indent=2, ensure_ascii=False)}

The script must:
1. Start with:
   from pathlib import Path
   import pandas as pd
   ROOT = Path(__file__).parent
   RAW = ROOT / 'data' / 'raw'

2. For each source file (skip files ending in _unified.parquet and combined.parquet):
   - Read the file
   - Map columns to a unified schema suited for the topic (e.g. 'text', 'label', keep '_source', '_dataset_id')
   - Drop columns that don't fit; rename those that do
   - Save as RAW / '<original_stem>_unified.parquet'

3. Glob all RAW / '*_unified.parquet', pd.concat, save as RAW / 'combined.parquet'
4. Print final stats: total rows, columns, value_counts of '_source'

Return ONLY the Python script, no markdown fences."""

    return _strip_fences(call_gemini_plain(prompt)).strip()


def _run_pipeline(output_dir: str, topic: str) -> None:
    unify_script = generate_unify_script(output_dir, topic)
    if unify_script:
        unify_path = Path(output_dir) / "unify_template.py"
        unify_path.write_text(unify_script)
        print(f"  {unify_path}")
        subprocess.run([sys.executable, str(unify_path)])
    cmd_preview(argparse.Namespace(dir=output_dir, n=5))
    run_eda(output_dir, topic)


# ------------------------------------------------------------------------------
# DISPLAY + INTERACTION
# ------------------------------------------------------------------------------

ICONS = {"huggingface": "[HF]", "kaggle": "[KGL]",
         "zenodo": "[ZEN]", "uci": "[UCI]", "google": "[GDS]"}

def display(datasets: list[dict]) -> None:
    print(f"\n{'='*65}")
    print(f"  Итого: {len(datasets)} датасетов")
    for src, n in sorted(Counter(ds.get("source", "?") for ds in datasets).items()):
        print(f"    {ICONS.get(src,'-')} {src}: {n}")
    print(f"{'='*65}\n")
    for i, ds in enumerate(datasets, 1):
        score = ds.get("relevance_score", 0)
        print(f"  [{i}] {ICONS.get(ds.get('source',''),'-')} {ds.get('name') or ds.get('id')}")
        print(f"      {ds.get('source')} | {ds.get('size_category','?')} | {ds.get('license','?')}")
        print(f"      dl:{ds.get('downloads',0):,}  likes:{ds.get('likes',0)}  tasks:{', '.join(ds.get('ml_tasks',[]))}")
        print(f"      {ds.get('description','')[:85]}...")
        print(f"      url: {ds.get('url','')}  score: {score}/10  {'#'*int(score)+'.'*(10-int(score))}\n")

def human_approval(datasets: list[dict]) -> list[dict]:
    print("-" * 65)
    print("  HUMAN IN THE LOOP - выберите датасеты для скачивания")
    print("  Форматы: 1,3,5 / all / q")
    choice = input("  Выбор: ").strip().lower()
    if choice in ("q", ""):   return []
    if choice == "all":       return datasets
    try:
        return [datasets[int(x.strip()) - 1] for x in choice.split(",") if x.strip().isdigit()]
    except IndexError:
        return datasets


from data_collection.eda import run_eda


# ------------------------------------------------------------------------------
# COMMANDS + CLI
# ------------------------------------------------------------------------------

def _topic_dir(base_dir: str, topic: str) -> str:
    path = Path(base_dir) / topic.strip().replace(" ", "_").lower()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def cmd_search(args) -> None:
    query = args.query or TOPIC
    output_dir = _topic_dir(args.output_dir, query)

    candidates = full_search(query)
    if not candidates:
        print("[!] Ничего не найдено."); sys.exit(1)

    datasets = rank_with_agent(candidates, topic=query)
    generate_config(query, datasets, output_dir)
    display(datasets)

    found_path = Path(output_dir) / "datasets_found.json"
    found_path.write_text(json.dumps(datasets, indent=2, ensure_ascii=False))
    print(f"  {found_path} ({len(datasets)} датасетов)\n")

    selected = human_approval(datasets)
    if not selected:
        sys.exit(0)
    script_path = Path(output_dir) / "download_datasets.py"
    script_path.write_text(generate_download_script(selected, output_dir))
    print(f"  {script_path}")

    print("\n  Запустить сейчас? [y/N]: ", end="", flush=True)
    if input().strip().lower() == "y":
        subprocess.run([sys.executable, str(script_path)])
        _run_pipeline(output_dir, query)
    else:
        print(f"  Запустите: python {script_path}")


def cmd_download(args) -> None:
    datasets = json.loads(Path(args.file).read_text())
    display(datasets)
    selected = human_approval(datasets)
    if not selected:
        sys.exit(0)
    output_dir = _topic_dir(OUTPUT_DIR, TOPIC)
    script_path = Path(output_dir) / "download_datasets.py"
    script_path.write_text(generate_download_script(selected, output_dir))
    subprocess.run([sys.executable, str(script_path)])
    _run_pipeline(output_dir, TOPIC)


def cmd_preview(args) -> None:
    raw_dir = Path(args.dir) / "data" / "raw"
    combined = raw_dir / "combined.parquet"
    if combined.exists():
        parquet_files = [combined]
    elif raw_dir.exists():
        parquet_files = list(raw_dir.glob("*.parquet"))
    else:
        parquet_files = list(Path(args.dir).glob("*.parquet"))
    if not parquet_files:
        print(f"[!] Нет parquet-файлов в {args.dir}"); sys.exit(1)
    for pf in parquet_files:
        print(f"\n{'='*65}")
        print(f"  {pf}")
        print(f"{'='*65}")
        try:
            import pyarrow.parquet as pq
            df = pq.read_table(pf).slice(0, args.n).to_pandas()
        except ImportError:
            df = pd.read_parquet(pf).head(args.n)
        print(df.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Agent - Gemini Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  search   "topic"              exhaustive search + interactive download
  download datasets_found.json  download from saved list
  preview  [dir]                show first N rows from downloaded parquet files

Env vars:
  GEMINI_API_KEY   required - free at aistudio.google.com
  KAGGLE_USERNAME  optional
  KAGGLE_KEY       optional
        """
    )
    sub = parser.add_subparsers(dest="command")
    sp = sub.add_parser("search")
    sp.add_argument("query", nargs="?", default=TOPIC)
    sp.add_argument("--output-dir", default=OUTPUT_DIR)
    sp.set_defaults(func=cmd_search)
    dp = sub.add_parser("download")
    dp.add_argument("file")
    dp.set_defaults(func=cmd_download)
    pp = sub.add_parser("preview")
    pp.add_argument("dir", nargs="?", default=OUTPUT_DIR)
    pp.add_argument("-n", type=int, default=5)
    pp.set_defaults(func=cmd_preview)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
