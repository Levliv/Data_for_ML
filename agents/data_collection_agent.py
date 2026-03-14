#!/usr/bin/env python3
"""
Dataset Search Agent — Gemini Edition
3 acquisition methods + monitoring for new datasets.

Setup:
  pip install google-genai huggingface_hub datasets requests beautifulsoup4 pandas pyyaml
  export GEMINI_API_KEY=...   # free at aistudio.google.com

Commands:
  python agents/data_collection_agent.py search "climate weather"
  python agents/data_collection_agent.py monitor "climate weather"
  python agents/data_collection_agent.py download datasets_found.json
"""

import json
import sys
import time
import argparse
import subprocess
import hashlib
import requests
import os
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from huggingface_hub import list_datasets
from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
_cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

TOPIC            = _cfg.get("topic", "climate weather")
OUTPUT_DIR       = str(ROOT / _cfg.get("output_dir", "data/raw"))
STATE_FILE       = str(ROOT / _cfg.get("state_file", "known_datasets.json"))
MONITOR_INTERVAL = _cfg.get("monitor_interval", 3600)
MODEL            = _cfg.get("model", "gemini-3.1-flash-lite-preview")

GEMINI_KEY     = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
KAGGLE_USER    = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY     = os.getenv("KAGGLE_KEY", "")
KAGGLE_TOKEN   = os.getenv("KAGGLE_API_TOKEN", "")

if not GEMINI_KEY:
    print("❌ Нет GEMINI_API_KEY")
    print("   1. Получи ключ бесплатно: https://aistudio.google.com/app/apikey")
    print("   2. export GEMINI_API_KEY=your_key")
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
    "source": "huggingface | kaggle | paperswithcode | zenodo | uci | google",
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
Include ALL datasets from the input — do not drop any. Assign relevance_score 1-3 to weakly related ones instead of excluding them."""


# ──────────────────────────────────────────────────────────────────────────────
# GEMINI CALL with retry on rate limit
# ──────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            )
            return response.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                wait = 30 * (attempt + 1)
                print(f"\n  ⏳ Rate limit, жду {wait}с (попытка {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini: превышен лимит попыток")


def call_gemini_plain(prompt: str) -> str:
    """Call without system prompt (for code generation)."""
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e):
                time.sleep(30 * (attempt + 1))
            else:
                raise
    raise RuntimeError("Gemini: превышен лимит попыток")


# ──────────────────────────────────────────────────────────────────────────────
# DATA COLLECTION AGENT
# ──────────────────────────────────────────────────────────────────────────────

class DataCollectionAgent:

    # ── skill 1 ───────────────────────────────────────────────────────────────

    def scrape(self, url: str, selector: str, params: dict = None, source: str = "") -> pd.DataFrame:
        """Fetch a web page and extract records via CSS selector."""
        print(f"  [{source} scrape]", end=" ", flush=True)
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (dataset-search-agent)"},
            )
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = []

            if selector.startswith("script"):
                # Google Dataset Search: JSON-LD embedded in <script> tags
                for script in soup.find_all("script", type="application/ld+json"):
                    try:
                        data = json.loads(script.string or "")
                        if isinstance(data, dict) and data.get("@type") == "Dataset":
                            rows.append({
                                "id": f"google:{hashlib.md5(data.get('name', '').encode()).hexdigest()[:8]}",
                                "source": "google",
                                "url": data.get("url", ""),
                                "downloads": 0,
                                "likes": 0,
                                "size_category": "unknown",
                                "license": str(data.get("license", "unknown")),
                                "name": data.get("name", ""),
                                "description_raw": str(data.get("description", ""))[:200],
                                "tags": [],
                            })
                    except Exception:
                        pass
            else:
                # Generic link selector (e.g. UCI "a[href*='/dataset/']")
                for link in soup.select(selector)[:20]:
                    name = link.get_text(strip=True)
                    href = link.get("href", "")
                    if name and href:
                        rows.append({
                            "id": f"uci:{href.split('/')[-1]}",
                            "source": "uci",
                            "url": f"https://archive.ics.uci.edu{href}" if href.startswith("/") else href,
                            "downloads": 0,
                            "likes": 0,
                            "size_category": "unknown",
                            "license": "open",
                            "name": name,
                            "tags": [],
                        })

            print(f"→ {len(rows)}")
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"⚠️  {e}")
            return pd.DataFrame()

    # ── skill 2 ───────────────────────────────────────────────────────────────

    def fetch_api(self, endpoint: str, params: dict, source: str = "") -> pd.DataFrame:
        """Call a REST API endpoint and normalise the response to a DataFrame."""
        _labels = {
            "paperswithcode": "Papers with Code API",
            "zenodo": "Zenodo scrape",
        }
        print(f"  [{_labels.get(source, source or endpoint)}]", end=" ", flush=True)
        try:
            resp = requests.get(endpoint, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = []

            if source == "paperswithcode":
                for ds in data.get("results", []):
                    rows.append({
                        "id": ds.get("url", ds.get("name", "")),
                        "source": "paperswithcode",
                        "url": "https://paperswithcode.com" + ds.get("url", ""),
                        "downloads": 0,
                        "likes": 0,
                        "size_category": "unknown",
                        "license": "unknown",
                        "name": ds.get("name", ""),
                        "description_raw": (ds.get("description") or "")[:200],
                        "tags": [],
                    })

            elif source == "zenodo":
                for rec in data.get("hits", {}).get("hits", []):
                    meta = rec.get("metadata", {})
                    size_bytes = sum(f.get("size", 0) for f in rec.get("files", []))
                    rows.append({
                        "id": f"zenodo:{rec.get('id')}",
                        "source": "zenodo",
                        "url": rec.get("links", {}).get("html", ""),
                        "downloads": rec.get("stats", {}).get("downloads", 0),
                        "likes": rec.get("stats", {}).get("views", 0),
                        "size_category": f"{size_bytes // 1_000_000} MB",
                        "license": (meta.get("license") or {}).get("id", "unknown"),
                        "name": meta.get("title", ""),
                        "description_raw": (meta.get("description") or "")[:200],
                        "tags": meta.get("keywords", []),
                    })

            print(f"→ {len(rows)}")
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"⚠️  {e}")
            return pd.DataFrame()

    # ── skill 3 ───────────────────────────────────────────────────────────────

    def load_dataset(self, query: str, source: str = "hf", limit: int = 50) -> pd.DataFrame:
        """Load dataset listing via platform SDK (HuggingFace Hub or Kaggle API)."""
        if source == "hf":
            print("  [HuggingFace Hub API]", end=" ", flush=True)
            try:
                results = list(list_datasets(search=query, limit=limit, sort="downloads", full=True))
                rows = []
                for ds in results:
                    tags = getattr(ds, "tags", []) or []
                    siblings = getattr(ds, "siblings", []) or []
                    size_bytes = sum(getattr(s, "size", 0) or 0 for s in siblings)
                    size_mb = size_bytes / (1024 * 1024)
                    size_disk = f"{size_mb:.1f} MB" if 0 < size_mb < 1024 else (f"{size_mb/1024:.2f} GB" if size_mb >= 1024 else None)
                    row_count = next((t.replace("size_categories:", "") for t in tags if "size_categories:" in t), None)
                    parts = [p for p in [size_disk, row_count] if p]
                    size_str = " | ".join(parts) if parts else "unknown"
                    rows.append({
                        "id": ds.id,
                        "source": "huggingface",
                        "url": f"https://huggingface.co/datasets/{ds.id}",
                        "downloads": getattr(ds, "downloads", 0) or 0,
                        "likes": getattr(ds, "likes", 0) or 0,
                        "size_category": size_str,
                        "license": next((t.replace("license:", "") for t in tags if t.startswith("license:")), "unknown"),
                        "tags": tags,
                    })
                print(f"→ {len(rows)}")
                return pd.DataFrame(rows)
            except Exception as e:
                print(f"⚠️  {e}")
                return pd.DataFrame()

        elif source == "kaggle":
            print("  [Kaggle API]", end=" ", flush=True)
            if KAGGLE_TOKEN:
                headers = {"Authorization": f"Bearer {KAGGLE_TOKEN}"}
                auth = None
            elif KAGGLE_USER and KAGGLE_KEY:
                headers = {}
                auth = (KAGGLE_USER, KAGGLE_KEY)
            else:
                print("skipped (set KAGGLE_API_TOKEN или KAGGLE_USERNAME + KAGGLE_KEY)")
                return pd.DataFrame()
            try:
                resp = requests.get(
                    "https://www.kaggle.com/api/v1/datasets/list",
                    params={"search": query, "sortBy": "votes", "pageSize": limit},
                    headers=headers,
                    auth=auth,
                    timeout=15,
                )
                resp.raise_for_status()
                rows = []
                for ds in resp.json():
                    ref = ds.get("ref") or f"{ds.get('ownerRef','')}/{ds.get('datasetSlug','')}"
                    ref = ref.strip("/")
                    if not ref:
                        continue
                    rows.append({
                        "id": ref,
                        "name": ds.get("title", ref),
                        "source": "kaggle",
                        "url": f"https://www.kaggle.com/datasets/{ref}",
                        "downloads": ds.get("downloadCount", ds.get("totalDownloads", 0)),
                        "likes": ds.get("voteCount", ds.get("totalVotes", 0)),
                        "size_category": f"{ds.get('totalBytes', 0) // 1_000_000} MB",
                        "license": ds.get("licenseName", "unknown"),
                        "tags": [t["name"] for t in ds.get("tags", [])],
                    })
                print(f"→ {len(rows)}")
                return pd.DataFrame(rows)
            except Exception as e:
                print(f"⚠️  {e}")
                return pd.DataFrame()

        return pd.DataFrame()

    # ── skill 4 ───────────────────────────────────────────────────────────────

    def merge(self, sources: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate DataFrames from multiple sources into one."""
        non_empty = [df for df in sources if df is not None and not df.empty]
        if not non_empty:
            return pd.DataFrame()
        return pd.concat(non_empty, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# FULL SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def full_search(query: str) -> list[dict]:
    print(f"\n🔍 Тема: '{query}'\n")
    agent = DataCollectionAgent()

    print("── Метод 1: Открытые датасеты ──────────────────────────")
    hf_df  = agent.load_dataset(query, source="hf", limit=50)
    kg_df  = agent.load_dataset(query, source="kaggle", limit=30)

    print("\n── Метод 2: API ────────────────────────────────────────")
    pwc_df = agent.fetch_api(
        "https://paperswithcode.com/api/v1/datasets/",
        {"q": query, "items_per_page": 30},
        source="paperswithcode",
    )

    print("\n── Метод 3: Скрапинг ───────────────────────────────────")
    zen_df = agent.fetch_api(
        "https://zenodo.org/api/records",
        {"q": query, "type": "dataset", "size": 20, "sort": "mostviewed"},
        source="zenodo",
    )
    uci_df = agent.scrape(
        "https://archive.ics.uci.edu/datasets",
        "a[href*='/dataset/']",
        params={"search": query},
        source="UCI ML Repository",
    )
    gds_df = agent.scrape(
        f"https://datasetsearch.research.google.com/search?query={requests.utils.quote(query)}",
        "script[type='application/ld+json']",
        source="Google Dataset Search",
    )

    merged = agent.merge([hf_df, kg_df, pwc_df, zen_df, uci_df, gds_df])
    candidates = merged.to_dict("records")
    print(f"\n   Всего кандидатов: {len(candidates)}")
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# AGENT RANKING
# ──────────────────────────────────────────────────────────────────────────────

def rank_with_agent(candidates: list[dict], topic: str) -> list[dict]:
    print(f"\n🤖 Gemini ранжирует {len(candidates)} кандидатов...")

    CHUNK = 50
    all_ranked = []

    for i in range(0, len(candidates), CHUNK):
        chunk = candidates[i:i + CHUNK]
        prompt = f'Topic: "{topic}"\n\nDatasets:\n{json.dumps(chunk, ensure_ascii=False)}\n\nReturn ONLY JSON array.'

        try:
            text = call_gemini(prompt)
            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                print(f"  ⚠️  JSON массив не найден в ответе")
                continue
            try:
                from json_repair import repair_json
                all_ranked.extend(json.loads(repair_json(text[start:end])))
            except ImportError:
                all_ranked.extend(json.loads(text[start:end]))
        except Exception as e:
            print(f"  ⚠️  Ошибка в chunk {i//CHUNK + 1}: {e}")

    return sorted(all_ranked, key=lambda x: x.get("relevance_score", 0), reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_config(query: str, ranked: list[dict]) -> None:
    """Ask Gemini to generate an updated config.yaml based on query + search results."""
    print("\n⚙️  Gemini обновляет config.yaml...")

    sources_stats = {}
    for ds in ranked:
        src = ds.get("source", "unknown")
        sources_stats[src] = sources_stats.get(src, 0) + 1

    avg_score = (
        sum(ds.get("relevance_score", 0) for ds in ranked) / len(ranked)
        if ranked else 0
    )
    top_tasks = {}
    for ds in ranked:
        for t in ds.get("ml_tasks", []):
            top_tasks[t] = top_tasks.get(t, 0) + 1
    top_tasks_sorted = sorted(top_tasks, key=top_tasks.get, reverse=True)[:5]

    current_cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    top3 = [{"name": d.get("name", ""), "source": d.get("source", ""), "score": d.get("relevance_score", 0)} for d in ranked[:3]]

    prompt = f"""You are a config generator for a dataset search agent.

User query: "{query}"
Search results summary:
- Total datasets found: {len(ranked)}
- Sources breakdown: {json.dumps(sources_stats)}
- Average relevance score: {avg_score:.1f}/10
- Top ML tasks: {top_tasks_sorted}
- Top 3 datasets: {json.dumps(top3)}

Current config:
{yaml.dump(current_cfg, allow_unicode=True)}

Generate an updated config.yaml. Rules:
- Keep: output_dir, state_file, monitor_interval, model (do not change these)
- Update: topic (use the query)
- Add: last_search (ISO timestamp {datetime.now().isoformat(timespec='seconds')}), sources_stats, avg_relevance_score, top_ml_tasks
- Add: preferred_sources (list of sources with >0 results, sorted by count descending)
- Add: search_limits per source based on what was productive (sources with 0 results → limit 0)

Return ONLY valid YAML, no markdown fences, no explanation."""

    try:
        raw = call_gemini_plain(prompt)

        # strip markdown fences if present
        if "```" in raw:
            import re
            m = re.search(r"```(?:yaml)?\n?(.*?)```", raw, re.DOTALL)
            raw = m.group(1).strip() if m else raw.replace("```", "").strip()

        new_cfg = yaml.safe_load(raw)
        if not isinstance(new_cfg, dict):
            raise ValueError("Gemini вернул не словарь")

        # safety: always preserve critical keys from current config
        for key in ("output_dir", "state_file", "monitor_interval", "model"):
            if key in current_cfg:
                new_cfg[key] = current_cfg[key]

        (ROOT / "config.yaml").write_text(
            yaml.dump(new_cfg, allow_unicode=True, sort_keys=False)
        )
        print(f"  ✅ config.yaml обновлён (тема: '{query}', источников: {len(sources_stats)})")

    except Exception as e:
        print(f"  ⚠️  Не удалось обновить config.yaml: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD SCRIPT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_script(datasets: list[dict], output_dir: str) -> str:
    lines = [
        "from datasets import load_dataset",
        "import os, requests, subprocess, pandas as pd",
        f"os.makedirs('{output_dir}', exist_ok=True)",
        "",
    ]
    for ds in datasets:
        src = ds.get("source", "")
        ds_id = ds.get("id", "")
        slug = ds_id.replace("/", "_").replace(":", "_")
        out = f"{output_dir}/{src}/{slug}"
        url = ds.get("url", "")
        if src == "huggingface":
            lines += [
                f"print('Downloading {ds_id}...')",
                f"try:",
                f"    os.makedirs('{out}', exist_ok=True)",
                f"    d = load_dataset('{ds_id}')",
                f"    frames = [d[s].to_pandas() for s in d]",
                f"    df = pd.concat(frames, ignore_index=True)",
                f"    df['_source'] = 'huggingface'",
                f"    df['_dataset_id'] = '{ds_id}'",
                f"    df.to_parquet('{out}/data.parquet', index=False)",
                f"    print('  ✅ saved to {out}/data.parquet')",
                f"except Exception as e:",
                f"    print(f'  ⚠️  {{e}}')",
                "",
            ]
        elif src == "kaggle":
            lines += [
                f"print('Downloading {ds_id} via kaggle CLI...')",
                f"os.makedirs('{out}', exist_ok=True)",
                f"subprocess.run(['kaggle', 'datasets', 'download', '-d', '{ds_id}', '-p', '{out}', '--unzip'], check=False)",
                "",
            ]
        else:
            lines += [
                f"print('Manual download: {url}')",
                "",
            ]
    lines += ["print('Done.')"]
    return "\n".join(lines)


def generate_download_script(datasets: list[dict], output_dir: str) -> str:
    print("\n⚙️  Gemini пишет скрипт скачивания...")

    prompt = f"""Write a Python download script for these datasets:

{json.dumps(datasets, indent=2, ensure_ascii=False)}

UNIFIED OUTPUT FORMAT — all datasets must be saved as a single Parquet file per dataset:
  - Final file path: {output_dir}/<source>/<slug>/data.parquet
  - Use pandas DataFrame as the unified intermediate format
  - Add a "_source" column with the source name (e.g. "huggingface", "kaggle", "zenodo")
  - Add a "_dataset_id" column with the dataset id string

Per-source download + conversion rules:

HuggingFace:
  - load_dataset(id) → iterate all splits → pd.concat all splits into one DataFrame → save as parquet
  - Use: ds[split].to_pandas() for each split

Kaggle:
  - kaggle datasets download -d <id> -p <tmp_dir> --unzip
  - Find all .csv / .json / .parquet files in tmp_dir
  - pd.read_csv / pd.read_json / pd.read_parquet → pd.concat → save as parquet

Zenodo (EXACT REST API v2 structure):
  - Step 1: extract record_id from URL last path segment (e.g. "7734140")
  - Step 2: GET https://zenodo.org/api/records/<record_id>
  - Step 3: files = response.json()["files"]
  - Step 4: filename = file["key"], download_url = file["links"]["self"] + "/content"
  - Step 5: stream download to tmp file, then read into DataFrame based on extension
  - IMPORTANT: file["links"]["self"] = "https://zenodo.org/api/records/<id>/files/<key>"
  - DO NOT use file["links"]["content"] — that key does not exist at record level

UCI / others:
  - requests streaming download → read into DataFrame → save as parquet

General rules:
  - import pandas as pd, import pyarrow
  - Progress output and per-dataset error handling
  - Final summary: Successful / Failed
  - if __name__ == "__main__": guard

Return ONLY the Python script, no markdown fences."""

    script = call_gemini_plain(prompt)

    import re
    FENCE = "```"
    if FENCE in script:
        m = re.search(FENCE + r"(?:python)?\n?(.*?)" + FENCE, script, re.DOTALL)
        if m:
            script = m.group(1).strip()
        else:
            script = script.replace(FENCE + "python", "").replace(FENCE, "").strip()
    script = script.strip()
    if not script:
        print("  ⚠️  Gemini вернул пустой скрипт, генерирую базовый...")
        script = _fallback_script(datasets, output_dir)

    return script


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

ICONS = {
    "huggingface": "🤗", "kaggle": "📊", "paperswithcode": "📄",
    "zenodo": "🔬", "uci": "🎓", "google": "🔍",
}

def display(datasets: list[dict]) -> None:
    print(f"\n{'═'*65}")
    print(f"  Итого: {len(datasets)} датасетов")
    sources = {}
    for ds in datasets:
        s = ds.get("source", "?")
        sources[s] = sources.get(s, 0) + 1
    for src, n in sorted(sources.items()):
        print(f"    {ICONS.get(src,'•')} {src}: {n}")
    print(f"{'═'*65}\n")

    for i, ds in enumerate(datasets, 1):
        score = ds.get("relevance_score", 0)
        bar = "█" * int(score) + "░" * (10 - int(score))
        icon = ICONS.get(ds.get("source", ""), "•")
        print(f"  [{i}] {icon} {ds.get('name') or ds.get('id')}")
        print(f"      {ds.get('source')} | {ds.get('size_category','?')} | {ds.get('license','?')}")
        print(f"      📥{ds.get('downloads',0):,}  ❤️{ds.get('likes',0)}  🎯{', '.join(ds.get('ml_tasks',[]))}")
        print(f"      {ds.get('description','')[:85]}...")
        print(f"      🔗 {ds.get('url','')}")
        print(f"      ⭐ {score}/10  {bar}\n")


# ──────────────────────────────────────────────────────────────────────────────
# HUMAN IN THE LOOP
# ──────────────────────────────────────────────────────────────────────────────

def human_approval(datasets: list[dict]) -> list[dict]:
    print("─" * 65)
    print("  HUMAN IN THE LOOP — выберите датасеты для скачивания")
    print("  Форматы: 1,3,5 / all / q")
    choice = input("  Выбор: ").strip().lower()
    if choice in ("q", ""):
        return []
    if choice == "all":
        return datasets
    try:
        return [datasets[int(x.strip()) - 1] for x in choice.split(",") if x.strip().isdigit()]
    except IndexError:
        return datasets


# ──────────────────────────────────────────────────────────────────────────────
# NOTIFICATIONS
# ──────────────────────────────────────────────────────────────────────────────

def notify(new_datasets: list[dict], topic: str) -> None:
    msg = f"🆕 Новые датасеты: {topic}\n\n"
    for ds in new_datasets[:10]:
        msg += f"• {ds.get('name') or ds.get('id')} [{ds.get('source')}]\n"
        msg += f"  {ds.get('url','')}\n"

    print("\n🔔 УВЕДОМЛЕНИЕ:\n" + msg)

    if TELEGRAM_TOKEN and TELEGRAM_CHAT:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT, "text": msg},
                timeout=10,
            )
            print("  📨 Telegram отправлен")
        except Exception as e:
            print(f"  ⚠️  Telegram: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# MONITOR
# ──────────────────────────────────────────────────────────────────────────────

def fingerprint(ds: dict) -> str:
    return hashlib.md5((ds.get("id", "") + ds.get("url", "")).encode()).hexdigest()

def load_state() -> set:
    if Path(STATE_FILE).exists():
        return set(json.loads(Path(STATE_FILE).read_text()))
    return set()

def save_state(fps: set) -> None:
    Path(STATE_FILE).write_text(json.dumps(list(fps)))

def cmd_monitor(args) -> None:
    query = args.query or TOPIC
    print(f"👁️  Мониторинг: '{query}'  |  интервал: {MONITOR_INTERVAL//60} мин")
    print(f"   Telegram: {'✅' if TELEGRAM_TOKEN else '❌ (не настроен)'}")
    print("   Ctrl+C для остановки\n")

    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Проверяю...")
        known = load_state()
        candidates = full_search(query)
        ranked = rank_with_agent(candidates, query)

        generate_config(query, ranked)
        current_fps = {fingerprint(ds) for ds in ranked}
        new_fps = current_fps - known

        found_path = ROOT / "datasets_found.json"
        if new_fps:
            new_ds = [ds for ds in ranked if fingerprint(ds) in new_fps]
            print(f"  🆕 Новых: {len(new_ds)}")
            notify(new_ds, query)
            save_state(known | current_fps)
            existing = json.loads(found_path.read_text()) if found_path.exists() else []
            found_path.write_text(json.dumps(existing + new_ds, indent=2, ensure_ascii=False))
        else:
            print("  Новых нет")

        print(f"  Следующая проверка через {MONITOR_INTERVAL//60} мин...")
        time.sleep(MONITOR_INTERVAL)


# ──────────────────────────────────────────────────────────────────────────────
# EDA
# ──────────────────────────────────────────────────────────────────────────────

def run_eda(output_dir: str) -> None:
    """Load all downloaded parquet files, collect stats, ask Gemini to generate EDA notebook."""
    try:
        import nbformat
    except ImportError:
        print("  ⚠️  nbformat не установлен: pip install nbformat")
        return

    parquet_files = list(Path(output_dir).rglob("*.parquet"))
    if not parquet_files:
        print("  ⚠️  Нет parquet-файлов в data/raw/ для EDA")
        return

    print(f"\n📊 Генерирую EDA для {len(parquet_files)} датасет(ов)...")

    stats = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            stats.append({
                "path": str(pf.relative_to(ROOT)),
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
                "head": df.head(3).fillna("").astype(str).to_dict(orient="records"),
                "describe": df.describe(include="all").fillna("").astype(str).to_dict(),
                "missing": {col: int(n) for col, n in df.isnull().sum().items() if n > 0},
            })
        except Exception as e:
            print(f"  ⚠️  {pf.name}: {e}")

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
        raw = call_gemini_plain(prompt)
        if "```" in raw:
            import re
            m = re.search(r"```(?:json)?\n?(.*?)```", raw, re.DOTALL)
            raw = m.group(1).strip() if m else raw.replace("```", "").strip()

        cells_data = json.loads(raw)

        nb = nbformat.v4.new_notebook()
        for c in cells_data:
            if c.get("cell_type") == "markdown":
                nb.cells.append(nbformat.v4.new_markdown_cell(c["source"]))
            else:
                nb.cells.append(nbformat.v4.new_code_cell(c["source"]))

        nb_path = ROOT / "notebooks" / "eda.ipynb"
        nb_path.parent.mkdir(exist_ok=True)
        nb_path.write_text(nbformat.writes(nb))
        print(f"  ✅ notebooks/eda.ipynb ({len(nb.cells)} ячеек)")

    except Exception as e:
        print(f"  ⚠️  Ошибка генерации EDA: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# COMMANDS
# ──────────────────────────────────────────────────────────────────────────────

def cmd_search(args) -> None:
    query = args.query or TOPIC
    candidates = full_search(query)
    if not candidates:
        print("❌ Ничего не найдено.")
        sys.exit(1)

    datasets = rank_with_agent(candidates, topic=query)
    generate_config(query, datasets)
    display(datasets)

    found_path = ROOT / "datasets_found.json"
    found_path.write_text(json.dumps(datasets, indent=2, ensure_ascii=False))
    print(f"  💾 datasets_found.json ({len(datasets)} датасетов)\n")

    selected = human_approval(datasets)
    if not selected:
        sys.exit(0)

    script = generate_download_script(selected, args.output_dir)
    script_path = ROOT / "download_datasets.py"
    script_path.write_text(script)
    print("  📄 download_datasets.py")

    print("\n  Запустить сейчас? [y/N]: ", end="", flush=True)
    if input().strip().lower() == "y":
        subprocess.run([sys.executable, str(script_path)])
        run_eda(args.output_dir)
    else:
        print("  Запустите: python download_datasets.py")


def cmd_download(args) -> None:
    datasets = json.loads(Path(args.file).read_text())
    display(datasets)
    selected = human_approval(datasets)
    if not selected:
        sys.exit(0)
    script = generate_download_script(selected, OUTPUT_DIR)
    script_path = ROOT / "download_datasets.py"
    script_path.write_text(script)
    subprocess.run([sys.executable, str(script_path)])
    run_eda(OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Agent — Gemini Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  search   "topic"              one-time exhaustive search
  monitor  "topic"              loop forever, alert on new datasets
  download datasets_found.json  download from saved list

Env vars:
  GEMINI_API_KEY       required  → free at aistudio.google.com
  KAGGLE_USERNAME      optional
  KAGGLE_KEY           optional
  TELEGRAM_BOT_TOKEN   optional
  TELEGRAM_CHAT_ID     optional
        """
    )
    sub = parser.add_subparsers(dest="command")

    sp = sub.add_parser("search")
    sp.add_argument("query", nargs="?", default=TOPIC)
    sp.add_argument("--output-dir", default=OUTPUT_DIR)
    sp.set_defaults(func=cmd_search)

    mp = sub.add_parser("monitor")
    mp.add_argument("query", nargs="?", default=TOPIC)
    mp.set_defaults(func=cmd_monitor)

    dp = sub.add_parser("download")
    dp.add_argument("file")
    dp.set_defaults(func=cmd_download)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
