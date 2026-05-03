"""
00_evaluation/fetch_wikipedia_references.py

Fetch human-written Wikipedia reference articles for evaluation.

These references are used for:
  - ROUGE-L
  - METEOR
  - human comparison

Run from project root:

    python 00_evaluation/fetch_wikipedia_references.py --titles Surtsey,Hawaii,Iceland

Or from cached RAG files in data/:

    python 00_evaluation/fetch_wikipedia_references.py --from-data

Output:
    00_evaluation/references/<Title>.txt
    00_evaluation/references/<Title>.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Optional

import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REFERENCES_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "references")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._()'-]+", "_", text)
    return text.strip("_")


def parse_csv(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def discover_titles_from_data(limit: Optional[int] = None) -> List[str]:
    titles = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith("_rag_context.json"):
            continue

        path = os.path.join(DATA_DIR, fname)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = (
                data.get("metadata", {}).get("resolved_entity_name")
                or data.get("blueprint_data", {}).get("island_name")
                or fname.replace("_rag_context.json", "").replace("_", " ")
            )

            titles.append(title)

        except Exception:
            title = fname.replace("_rag_context.json", "").replace("_", " ")
            titles.append(title)

    if limit:
        titles = titles[:limit]

    return titles


def fetch_wikipedia_page(title: str, lang: str = "en") -> dict:
    """
    Fetch page summary + plain text extract using the MediaWiki API.
    """
    api_url = f"https://{lang}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "titles": title,
        "prop": "extracts|info|pageprops",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "inprop": "url",
    }

    headers = {
        "User-Agent": "LLM-MA-Agent-Project/1.0 (student evaluation script)"
    }

    response = requests.get(api_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", [])

    if not pages:
        raise ValueError(f"No Wikipedia page found for title: {title}")

    page = pages[0]

    if page.get("missing"):
        raise ValueError(f"Wikipedia page missing for title: {title}")

    return {
        "requested_title": title,
        "resolved_title": page.get("title", title),
        "pageid": page.get("pageid"),
        "url": page.get("fullurl", ""),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "text": page.get("extract", ""),
    }


def save_reference(page: dict):
    os.makedirs(REFERENCES_DIR, exist_ok=True)

    title = page["resolved_title"]
    base = safe_name(title)

    txt_path = os.path.join(REFERENCES_DIR, f"{base}.txt")
    json_path = os.path.join(REFERENCES_DIR, f"{base}.json")

    text = page.get("text", "").strip()

    if not text:
        raise ValueError(f"Empty Wikipedia article for {title}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    payload = {
        "requested_title": page.get("requested_title"),
        "resolved_title": page.get("resolved_title"),
        "pageid": page.get("pageid"),
        "url": page.get("url"),
        "fetched_at": page.get("fetched_at"),
        "reference_article": text,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return txt_path, json_path


def main():
    parser = argparse.ArgumentParser(
        description="Fetch human-written Wikipedia reference articles."
    )

    parser.add_argument(
        "--titles",
        default=None,
        help="Comma-separated Wikipedia titles. Example: Surtsey,Hawaii,Iceland",
    )

    parser.add_argument(
        "--from-data",
        action="store_true",
        help="Infer titles from data/*_rag_context.json.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only fetch first N titles.",
    )

    parser.add_argument(
        "--lang",
        default="en",
        help="Wikipedia language code. Default: en.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between requests.",
    )

    args = parser.parse_args()

    titles = []

    if args.titles:
        titles.extend(parse_csv(args.titles))

    if args.from_data:
        titles.extend(discover_titles_from_data(limit=args.limit))

    # Deduplicate
    seen = set()
    titles = [t for t in titles if not (t in seen or seen.add(t))]

    if args.limit:
        titles = titles[:args.limit]

    if not titles:
        raise ValueError("No titles provided. Use --titles or --from-data.")

    print("\n" + "=" * 80)
    print("Fetching Wikipedia references")
    print("=" * 80)
    print(f"Titles: {len(titles)}")
    print(f"Output: {REFERENCES_DIR}")
    print("=" * 80 + "\n")

    for title in titles:
        print(f"Fetching: {title}")

        try:
            page = fetch_wikipedia_page(title, lang=args.lang)
            txt_path, json_path = save_reference(page)

            print(f" Saved TXT:  {txt_path}")
            print(f" Saved JSON: {json_path}")
            print(f"   Resolved title: {page['resolved_title']}")
            print(f"   URL: {page['url']}")

        except Exception as e:
            print(f" Failed for {title}: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\nDone.")


if __name__ == "__main__":
    main()