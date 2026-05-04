"""
00_evaluation/fetch_wikipedia_references.py

Fetch human-written Wikipedia reference articles for evaluation.

These references are used for:
  - ROUGE-L
  - METEOR
  - concept-based LLM evaluation

Run from project root:

    python 00_evaluation/fetch_wikipedia_references.py --final-10

Or from cached RAG files in data/:

    python 00_evaluation/fetch_wikipedia_references.py --from-data

Or manually:

    python 00_evaluation/fetch_wikipedia_references.py --titles Surtsey,Hawaii,Iceland

Output:
    00_evaluation/references/<Title>.txt
    00_evaluation/references/<Title>.json

The script saves multiple aliases for each reference, so names from:
  - generated result files
  - RAG context files
  - resolved Wikipedia titles

can all be matched later by full_evaluation.py.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REFERENCES_DIR = os.path.join(PROJECT_ROOT, "00_evaluation", "references")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ---------------------------------------------------------------------
# Final 10-island evaluation set
# ---------------------------------------------------------------------

FINAL_10_ISLANDS = [
    {
        "fetch_title": "Brecqhou",
        "aliases": ["Brecqhou"],
    },
    {
        "fetch_title": "Hashima Island",
        "aliases": ["Hashima", "Hashima Island", "Hashima_Island"],
    },
    {
        "fetch_title": "Howland Island",
        "aliases": ["Howland", "Howland Island", "Howland_Island"],
    },
    {
        "fetch_title": "Hunga Tonga–Hunga Haʻapai",
        "aliases": [
            "Hunga Tonga–Hunga Haʻapai",
            "Hunga_Tonga–Hunga_Ha_apai",
            "Hunga Tonga-Hunga Ha'apai",
            "Hunga_Tonga_Hunga_Ha_apai",
            "Hunga Tonga Hunga Haapai",
            "Hunga Tonga Hunga Ha apai",
        ],
    },
    {
        "fetch_title": "Jarvis Island",
        "aliases": ["Jarvis", "Jarvis Island", "Jarvis_Island"],
    },
    {
        "fetch_title": "Jethou",
        "aliases": ["Jethou"],
    },
    {
        "fetch_title": "Kavachi",
        "aliases": ["Kavachi"],
    },
    {
        "fetch_title": "Nishinoshima (Ogasawara)",
        "aliases": [
            "Nishinoshima (Ogasawara)",
            "Nishinoshima_(Ogasawara)",
            "Nishinoshima",
        ],
    },
    {
        "fetch_title": "Surtsey",
        "aliases": ["Surtsey"],
    },
    {
        "fetch_title": "Tromelin Island",
        "aliases": ["Tromelin", "Tromelin Island", "Tromelin_Island"],
    },
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_name(text: str) -> str:
    """
    Convert a title into a filesystem-safe name.

    Examples:
      Hashima Island -> Hashima_Island
      Nishinoshima (Ogasawara) -> Nishinoshima_(Ogasawara)
      Hunga Tonga–Hunga Haʻapai -> Hunga_Tonga_Hunga_Ha_apai
    """
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._()'-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def parse_csv(value: str) -> List[str]:
    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def add_common_aliases(name: str) -> List[str]:
    """
    Add useful aliases for island names.

    This helps when:
      output file says: Hashima
      RAG file says: Hashima_Island
      Wikipedia page says: Hashima Island
    """
    aliases = []

    if not name:
        return aliases

    name = str(name).strip()
    aliases.append(name)
    aliases.append(name.replace("_", " "))
    aliases.append(name.replace(" ", "_"))

    if name.endswith("_Island"):
        base = name.replace("_Island", "")
        aliases.extend([base, base.replace("_", " ")])

    if name.endswith(" Island"):
        base = name.replace(" Island", "")
        aliases.extend([base, base.replace(" ", "_")])

    if not name.endswith(" Island") and not name.endswith("_Island"):
        aliases.append(f"{name} Island")
        aliases.append(f"{name.replace(' ', '_')}_Island")

    return aliases


def dedupe_list(values: List[str]) -> List[str]:
    seen = set()
    result = []

    for value in values:
        value = str(value).strip()

        if not value:
            continue

        key = safe_name(value)

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


# ---------------------------------------------------------------------
# Discover from RAG context files
# ---------------------------------------------------------------------

def discover_records_from_data(limit: Optional[int] = None) -> List[Dict[str, object]]:
    """
    Infer Wikipedia titles from data/*_rag_context.json.

    The RAG filename is used as an alias, so references are saved with names
    matching the cached RAG files.
    """
    records = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith("_rag_context.json"):
            continue

        path = os.path.join(DATA_DIR, fname)

        rag_base = fname.replace("_rag_context.json", "")
        rag_title = rag_base.replace("_", " ")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = (
                data.get("metadata", {}).get("resolved_entity_name")
                or data.get("blueprint_data", {}).get("island_name")
                or rag_title
            )

        except Exception:
            title = rag_title

        aliases = []
        aliases.extend(add_common_aliases(title))
        aliases.extend(add_common_aliases(rag_title))
        aliases.extend(add_common_aliases(rag_base))

        records.append(
            {
                "fetch_title": title,
                "aliases": dedupe_list(aliases),
            }
        )

    if limit:
        records = records[:limit]

    return records


def deduplicate_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Deduplicate by fetch_title, while preserving aliases.
    """
    merged: Dict[str, Dict[str, object]] = {}

    for record in records:
        fetch_title = str(record.get("fetch_title", "")).strip()

        if not fetch_title:
            continue

        key = fetch_title.lower()

        if key not in merged:
            merged[key] = {
                "fetch_title": fetch_title,
                "aliases": [],
            }

        existing_aliases = merged[key].get("aliases", []) or []
        new_aliases = record.get("aliases", []) or []

        merged[key]["aliases"] = dedupe_list(existing_aliases + new_aliases)

    return list(merged.values())


# ---------------------------------------------------------------------
# Wikipedia fetching
# ---------------------------------------------------------------------

def fetch_wikipedia_page(title: str, lang: str = "en") -> dict:
    """
    Fetch plain-text Wikipedia article using the MediaWiki API.
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

    response = requests.get(
        api_url,
        params=params,
        headers=headers,
        timeout=30,
    )
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


def save_reference(page: dict, aliases: Optional[List[str]] = None) -> List[tuple]:
    """
    Save the reference article under all useful aliases.

    For example, for Hashima Island this may save:
      Hashima.txt
      Hashima.json
      Hashima_Island.txt
      Hashima_Island.json
    """
    os.makedirs(REFERENCES_DIR, exist_ok=True)

    text = page.get("text", "").strip()

    if not text:
        raise ValueError(f"Empty Wikipedia article for {page.get('resolved_title')}")

    names_to_save = []

    names_to_save.append(page.get("requested_title", ""))
    names_to_save.append(page.get("resolved_title", ""))

    for alias in aliases or []:
        names_to_save.append(alias)

    names_to_save = dedupe_list(names_to_save)

    saved_paths = []

    for name in names_to_save:
        base = safe_name(name)

        if not base:
            continue

        txt_path = os.path.join(REFERENCES_DIR, f"{base}.txt")
        json_path = os.path.join(REFERENCES_DIR, f"{base}.json")

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

        saved_paths.append((txt_path, json_path))

    return saved_paths


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch human-written Wikipedia reference articles."
    )

    parser.add_argument(
        "--final-10",
        action="store_true",
        help="Fetch references for the final 10-island evaluation set.",
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

    records = []

    if args.final_10:
        records.extend(FINAL_10_ISLANDS)

    if args.titles:
        for title in parse_csv(args.titles):
            aliases = add_common_aliases(title)

            records.append(
                {
                    "fetch_title": title,
                    "aliases": aliases,
                }
            )

    if args.from_data:
        records.extend(discover_records_from_data(limit=args.limit))

    records = deduplicate_records(records)

    if args.limit:
        records = records[:args.limit]

    if not records:
        raise ValueError("No titles provided. Use --final-10, --titles, or --from-data.")

    print("\n" + "=" * 80)
    print("Fetching Wikipedia references")
    print("=" * 80)
    print(f"Titles: {len(records)}")
    print(f"Output: {REFERENCES_DIR}")
    print("=" * 80 + "\n")

    for record in records:
        fetch_title = str(record.get("fetch_title", "")).strip()
        aliases = record.get("aliases", []) or []

        print(f"Fetching: {fetch_title}")
        print(f"  Aliases: {aliases}")

        try:
            page = fetch_wikipedia_page(fetch_title, lang=args.lang)

            saved_paths = save_reference(
                page=page,
                aliases=aliases,
            )

            print(f"  Resolved title: {page['resolved_title']}")
            print(f"  URL: {page['url']}")

            for txt_path, json_path in saved_paths:
                print(f"  Saved TXT:  {txt_path}")
                print(f"  Saved JSON: {json_path}")

        except Exception as e:
            print(f"  Failed for {fetch_title}: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\nDone.")


if __name__ == "__main__":
    main()