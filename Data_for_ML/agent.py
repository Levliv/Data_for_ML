#!/usr/bin/env python3
"""
Dataset Search Agent — Gemini Edition
3 acquisition methods + monitoring for new datasets.

Setup:
  pip install google-genai huggingface_hub datasets requests beautifulsoup4
  export GEMINI_API_KEY=...   # free at aistudio.google.com

Commands:
  python agent.py search "climate weather"
  python agent.py monitor "climate weather"
  python agent.py download datasets_found.json
"""

import json
import sys
import time
import argparse
import subprocess
import hashlib
import requests
import os
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from huggingface_hub import list_datasets
from google import genai
from google.genai import types

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

TOPIC            = "climate weather"
OUTPUT_DIR       = "data"
STATE_FILE       = "known_datasets.json"
MONITOR_INTERVAL = 3600

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
MODEL  = "gemini-3.1-flash-lite-preview"

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

Sort by relevance_score descending. Include ALL relevant datasets."""


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
# METHOD 1: OPEN DATASETS
# ──────────────────────────────────────────────────────────────────────────────

def fetch_huggingface(query: str, limit: int = 50) -> list[dict]:
    print("  [HuggingFace Hub API]", end=" ", flush=True)
    try:
        results = list(list_datasets(search=query, limit=limit, sort="downloads", full=True))
        out = []
        for ds in results:
            tags = getattr(ds, "tags", []) or []
            out.append({
                "id": ds.id,
                "source": "huggingface",
                "url": f"https://huggingface.co/datasets/{ds.id}",
                "downloads": getattr(ds, "downloads", 0) or 0,
                "likes": getattr(ds, "likes", 0) or 0,
                "size_category": next((t.replace("size_categories:", "") for t in tags if "size_categories:" in t), "unknown"),
                "license": next((t.replace("license:", "") for t in tags if t.startswith("license:")), "unknown"),
                "tags": tags,
            })
        print(f"→ {len(out)}")
        return out
    except Exception as e:
        print(f"⚠️  {e}")
        return []


def fetch_kaggle(query: str, limit: int = 30) -> list[dict]:
    print("  [Kaggle API]", end=" ", flush=True)
    if KAGGLE_TOKEN:
        headers = {"Authorization": f"Bearer {KAGGLE_TOKEN}"}
        auth = None
    elif KAGGLE_USER and KAGGLE_KEY:
        headers = {}
        auth = (KAGGLE_USER, KAGGLE_KEY)
    else:
        print("skipped (set KAGGLE_API_TOKEN или KAGGLE_USERNAME + KAGGLE_KEY)")
        return []
    try:
        resp = requests.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={"search": query, "sortBy": "votes", "pageSize": limit},
            headers=headers,
            auth=auth,
            timeout=15,
        )
        resp.raise_for_status()
        out = []
        for ds in resp.json():
            ref = ds.get("ref") or f"{ds.get('ownerRef','')}/{ds.get('datasetSlug','')}"
            ref = ref.strip("/")
            if not ref:
                continue
            out.append({
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
        print(f"→ {len(out)}")
        return out
    except Exception as e:
        print(f"⚠️  {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 2: API
# ──────────────────────────────────────────────────────────────────────────────

def fetch_paperswithcode(query: str, limit: int = 30) -> list[dict]:
    print("  [Papers with Code API]", end=" ", flush=True)
    try:
        resp = requests.get(
            "https://paperswithcode.com/api/v1/datasets/",
            params={"q": query, "items_per_page": limit},
            timeout=15,
        )
        resp.raise_for_status()
        out = []
        for ds in resp.json().get("results", []):
            out.append({
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
        print(f"→ {len(out)}")
        return out
    except Exception as e:
        print(f"⚠️  {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 3: SCRAPING
# ──────────────────────────────────────────────────────────────────────────────

def scrape_zenodo(query: str, limit: int = 20) -> list[dict]:
    print("  [Zenodo scrape]", end=" ", flush=True)
    try:
        resp = requests.get(
            "https://zenodo.org/api/records",
            params={"q": query, "type": "dataset", "size": limit, "sort": "mostviewed"},
            timeout=15,
        )
        resp.raise_for_status()
        out = []
        for rec in resp.json().get("hits", {}).get("hits", []):
            meta = rec.get("metadata", {})
            size_bytes = sum(f.get("size", 0) for f in rec.get("files", []))
            out.append({
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
        print(f"→ {len(out)}")
        return out
    except Exception as e:
        print(f"⚠️  {e}")
        return []


def scrape_uci(query: str) -> list[dict]:
    print("  [UCI ML Repository scrape]", end=" ", flush=True)
    try:
        resp = requests.get(
            "https://archive.ics.uci.edu/datasets",
            params={"search": query},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (dataset-search-agent)"},
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        out = []
        for link in soup.select("a[href*='/dataset/']")[:20]:
            name = link.get_text(strip=True)
            href = link.get("href", "")
            if name and href:
                out.append({
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
        print(f"→ {len(out)}")
        return out
    except Exception as e:
        print(f"⚠️  {e}")
        return []


def scrape_google_dataset_search(query: str) -> list[dict]:
    print("  [Google Dataset Search scrape]", end=" ", flush=True)
    try:
        url = f"https://datasetsearch.research.google.com/search?query={requests.utils.quote(query)}"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        out = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, dict) and data.get("@type") == "Dataset":
                    out.append({
                        "id": f"google:{hashlib.md5(data.get('name','').encode()).hexdigest()[:8]}",
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
        print(f"→ {len(out)}")
        return out
    except Exception as e:
        print(f"⚠️  {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# FULL SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def full_search(query: str) -> list[dict]:
    print(f"\n🔍 Тема: '{query}'\n")
    candidates = []

    print("── Метод 1: Открытые датасеты ──────────────────────────")
    candidates += fetch_huggingface(query, limit=50)
    candidates += fetch_kaggle(query, limit=30)

    print("\n── Метод 2: API ────────────────────────────────────────")
    candidates += fetch_paperswithcode(query, limit=30)

    print("\n── Метод 3: Скрапинг ───────────────────────────────────")
    candidates += scrape_zenodo(query, limit=20)
    candidates += scrape_uci(query)
    candidates += scrape_google_dataset_search(query)

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
# DOWNLOAD SCRIPT GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_script(datasets: list[dict], output_dir: str) -> str:
    lines = [
        "from datasets import load_dataset",
        "import os, requests, subprocess",
        f"os.makedirs('{output_dir}', exist_ok=True)",
        "",
    ]
    for ds in datasets:
        src = ds.get("source", "")
        ds_id = ds.get("id", "")
        slug = ds_id.replace("/", "_").replace(":", "_")
        url = ds.get("url", "")
        if src == "huggingface":
            lines += [
                f"print('Downloading {ds_id}...')",
                f"try:",
                f"    d = load_dataset('{ds_id}')",
                f"    d.save_to_disk('{output_dir}/{slug}')",
                f"    print('  ✅ saved to {output_dir}/{slug}')",
                f"except Exception as e:",
                f"    print(f'  ⚠️  {{e}}')",
                "",
            ]
        elif src == "kaggle":
            lines += [
                f"print('Downloading {ds_id} via kaggle CLI...')",
                f"subprocess.run(['kaggle', 'datasets', 'download', '-d', '{ds_id}', '-p', '{output_dir}/kaggle/'], check=False)",
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

Requirements:
- HuggingFace: from datasets import load_dataset; ds.save_to_disk(...)
- Kaggle: kaggle datasets download -d <id> -p {output_dir}/kaggle/
- Zenodo: use requests library. EXACT Zenodo API structure:
  Step 1: GET https://zenodo.org/api/records/<record_id>
  Step 2: files = response.json()["files"]
  Step 3: each file: file["key"] = filename, file["links"]["content"] = download URL
  Step 4: stream download with iter_content(chunk_size=65536)
  DO NOT use file["name"] or file["links"]["self"] — these fields do not exist
- UCI / others: requests with streaming
- Subdirectory per source: {output_dir}/<source>/<slug>/
- Progress output, error handling per dataset, final summary
- if __name__ == "__main__": guard

Return ONLY the Python script, no markdown fences."""

    script = call_gemini_plain(prompt)

    # убрать markdown-обёртку любого вида
    import re
    FENCE = "```"
    if FENCE in script:
        import re as _re
        m = _re.search(FENCE + r"(?:python)?\n?(.*?)" + FENCE, script, _re.DOTALL)
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

        current_fps = {fingerprint(ds) for ds in ranked}
        new_fps = current_fps - known

        if new_fps:
            new_ds = [ds for ds in ranked if fingerprint(ds) in new_fps]
            print(f"  🆕 Новых: {len(new_ds)}")
            notify(new_ds, query)
            save_state(known | current_fps)
            existing = json.loads(Path("datasets_found.json").read_text()) if Path("datasets_found.json").exists() else []
            Path("datasets_found.json").write_text(json.dumps(existing + new_ds, indent=2, ensure_ascii=False))
        else:
            print("  Новых нет")

        print(f"  Следующая проверка через {MONITOR_INTERVAL//60} мин...")
        time.sleep(MONITOR_INTERVAL)


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
    display(datasets)

    Path("datasets_found.json").write_text(json.dumps(datasets, indent=2, ensure_ascii=False))
    print(f"  💾 datasets_found.json ({len(datasets)} датасетов)\n")

    selected = human_approval(datasets)
    if not selected:
        sys.exit(0)

    script = generate_download_script(selected, args.output_dir)
    Path("download_datasets.py").write_text(script)
    print("  📄 download_datasets.py")

    print("\n  Запустить сейчас? [y/N]: ", end="", flush=True)
    if input().strip().lower() == "y":
        subprocess.run([sys.executable, "download_datasets.py"])
    else:
        print("  Запустите: python download_datasets.py")


def cmd_download(args) -> None:
    datasets = json.loads(Path(args.file).read_text())
    display(datasets)
    selected = human_approval(datasets)
    if not selected:
        sys.exit(0)
    script = generate_download_script(selected, OUTPUT_DIR)
    Path("download_datasets.py").write_text(script)
    subprocess.run([sys.executable, "download_datasets.py"])


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
