"""
Shared utility functions for evaluation scripts.
"""

import json
import os
import re
from typing import List, Optional, Tuple


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._()'-]+", "_", text)
    return text.strip("_")


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def word_count(text: str) -> int:
    return len(tokenize(text))


def strip_citations(text: str) -> str:
    """
    Remove numeric citation markers from text.

    Supports:
      [1]
      [1][2]
      [1, 2]
      [1,2,4]
      [1-3]
      [1–3]
    """
    text = str(text)
    text = re.sub(r"\[[0-9,\s\-–—]+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def split_sentences(text: str) -> List[str]:
    if not text:
        return []

    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"==.*?==", " ", text)

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(strip_citations(p).strip()) > 5]

def extract_citation_numbers(sentence: str) -> List[int]:
    """
    Extract citation numbers from citation markers.

    Supports:
      [1]
      [1][2]
      [1, 2]
      [1,2,4]
      [1-3]
      [1–3]
    """
    numbers = []

    # Capture bracket contents that look citation-like.
    bracket_contents = re.findall(r"\[([0-9,\s\-–—]+)\]", str(sentence))

    for content in bracket_contents:
        parts = re.split(r"\s*,\s*", content.strip())

        for part in parts:
            part = part.strip()

            # Handle ranges such as 1-3 or 1–3.
            range_match = re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", part)

            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))

                if start <= end:
                    numbers.extend(range(start, end + 1))
                else:
                    numbers.extend(range(start, end - 1, -1))

                continue

            # Handle single citation number.
            if re.fullmatch(r"\d+", part):
                numbers.append(int(part))

    return numbers


def normalize_url(url: str) -> str:
    if not url:
        return ""
    return str(url).strip().rstrip("/")


def clean_reference_article(text: str) -> str:
    """
    Removes Wikipedia sections that generated articles are not expected to reproduce.
    """
    if not text:
        return ""

    cutoff_headings = [
        "See also",
        "Notes",
        "Footnotes",
        "References",
        "Sources",
        "Bibliography",
        "Further reading",
        "External links",
    ]

    pattern = (
        r"\n==\s*("
        + "|".join(re.escape(h) for h in cutoff_headings)
        + r")\s*=="
    )

    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        text = text[:match.start()]

    return text.strip()


def normalize_result_file(path: str) -> List[dict]:
    """
    Supports both:

    Shape A:
      {
        "method0": {...},
        "method1": {...}
      }

    Shape B:
      {
        "metadata": {...},
        "result": {...}
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    if isinstance(data, dict) and "metadata" in data and "result" in data:
        metadata = data.get("metadata", {})
        result = data.get("result", {})

        rows.append(
            {
                "source_file": path,
                "input_json": metadata.get("input_json", ""),
                "island_name": metadata.get("island")
                or result.get("island_name")
                or "unknown",
                "method": metadata.get("method")
                or result.get("method")
                or "unknown",
                "generated_article": result.get("generated_article", ""),
                "sections": result.get("sections", []),
            }
        )

        return rows

    for method_name, result in data.items():
        if not isinstance(result, dict):
            continue

        if "generated_article" not in result:
            continue

        rows.append(
            {
                "source_file": path,
                "input_json": "",
                "island_name": result.get("island_name", "unknown"),
                "method": method_name,
                "generated_article": result.get("generated_article", ""),
                "sections": result.get("sections", []),
            }
        )

    return rows


def discover_input_files(input_path: Optional[str], input_dir: Optional[str]) -> List[str]:
    files = []

    if input_path:
        files.append(os.path.abspath(input_path))

    if input_dir:
        input_dir = os.path.abspath(input_dir)

        for fname in sorted(os.listdir(input_dir)):
            if fname.endswith(".json"):
                files.append(os.path.join(input_dir, fname))

    seen = set()
    unique = []

    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)

    return unique

def load_reference_article(
    island_name: str,
    references_dir: str,
) -> Tuple[Optional[str], str]:
    """
    Loads the human-written Wikipedia reference article.

    Supports aliases such as:
      Hashima -> Hashima.txt / Hashima_Island.txt
      Howland -> Howland.txt / Howland_Island.txt
      Jarvis -> Jarvis.txt / Jarvis_Island.txt
      Tromelin -> Tromelin.txt / Tromelin_Island.txt

    JSON reference can contain:
      reference_article
      article
      content
      text
      generated_article
    """
    alias_map = {
        "Hashima": ["Hashima", "Hashima Island", "Hashima_Island"],
        "Hashima Island": ["Hashima Island", "Hashima_Island", "Hashima"],

        "Howland": ["Howland", "Howland Island", "Howland_Island"],
        "Howland Island": ["Howland Island", "Howland_Island", "Howland"],

        "Jarvis": ["Jarvis", "Jarvis Island", "Jarvis_Island"],
        "Jarvis Island": ["Jarvis Island", "Jarvis_Island", "Jarvis"],

        "Tromelin": ["Tromelin", "Tromelin Island", "Tromelin_Island"],
        "Tromelin Island": ["Tromelin Island", "Tromelin_Island", "Tromelin"],

        "Nishinoshima": [
            "Nishinoshima",
            "Nishinoshima (Ogasawara)",
            "Nishinoshima_(Ogasawara)",
        ],
        "Nishinoshima (Ogasawara)": [
            "Nishinoshima (Ogasawara)",
            "Nishinoshima_(Ogasawara)",
            "Nishinoshima",
        ],

        "Hunga Tonga–Hunga Haʻapai": [
            "Hunga Tonga–Hunga Haʻapai",
            "Hunga_Tonga_Hunga_Ha_apai",
            "Hunga_Tonga_Hunga_Haapai",
            "Hunga_Tonga-Hunga_Ha'apai",
            "Hunga Tonga Hunga Haapai",
            "Hunga Tonga Hunga Ha apai",
        ],
    }

    candidates = []

    # Original name variants
    candidates.append(island_name)
    candidates.append(safe_name(island_name))
    candidates.append(island_name.replace(" ", "_"))

    # Alias variants
    for alias in alias_map.get(island_name, []):
        candidates.append(alias)
        candidates.append(safe_name(alias))
        candidates.append(alias.replace(" ", "_"))

    # Deduplicate while preserving order
    seen = set()
    candidates = [
        c for c in candidates
        if c and not (c in seen or seen.add(c))
    ]

    for base in candidates:
        txt_path = os.path.join(references_dir, f"{base}.txt")

        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return clean_reference_article(f.read()), txt_path

        json_path = os.path.join(references_dir, f"{base}.json")

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key in [
                "reference_article",
                "article",
                "content",
                "text",
                "generated_article",
            ]:
                if isinstance(data, dict) and data.get(key):
                    return clean_reference_article(str(data[key])), json_path

    return None, ""