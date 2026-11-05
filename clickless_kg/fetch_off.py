"""Download candidate products from the Open Food Facts v2 search API.

Responses are cached under data/raw/pages/, so re-running only fetches pages
that are missing. Commit data/raw/off_raw.jsonl to the repo once it is built:
from then on the catalog is rebuilt offline from that snapshot, which is what
makes the catalog reproducible for the whole team.

OFF asks API clients to send a descriptive User-Agent and limits search
requests to about 10 per minute; we sleep between requests to respect that.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from . import vocab

SEARCH_URL = "https://world.openfoodfacts.org/api/v2/search"
FIELDS = ",".join([
    "code", "product_name", "product_name_en", "brands", "categories_tags",
    "labels_tags", "allergens_tags", "traces_tags", "ingredients_text",
    "ingredients_text_en", "ingredients_analysis_tags", "ingredients_n",
    "states_tags", "countries_tags", "unique_scans_n",
])
USER_AGENT = os.environ.get("OFF_USER_AGENT", "ClickLessAI-CSE573-StudentProject/0.1")


def fetch_page(off_tag: str, page: int, page_size: int, country: str | None,
               cache_dir: Path, sleep_s: float) -> list[dict]:
    cache = cache_dir / f"{off_tag.replace(':', '_')}_p{page}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["products"]
    params = {"categories_tags": off_tag, "fields": FIELDS, "page": page,
              "page_size": page_size, "sort_by": "unique_scans_n", "json": 1}
    if country:
        params["countries_tags"] = country
    for attempt in range(4):
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=60,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            wait = sleep_s * (2 ** attempt)
            print(f"  {off_tag} p{page}: {e}; retrying in {wait:.0f}s")
            time.sleep(wait)
    else:
        print(f"  giving up on {off_tag} p{page}")
        return []
    cache.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(sleep_s)
    return data.get("products", [])


def fetch_all(out_path: Path, pages: int = 2, page_size: int = 100,
              country: str | None = "en:united-states", sleep_s: float = 7.0) -> int:
    cache_dir = out_path.parent / "pages"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}
    for spec in vocab.CATEGORIES:
        for off_tag in spec.off_tags:
            for page in range(1, pages + 1):
                products = fetch_page(off_tag, page, page_size, country, cache_dir, sleep_s)
                print(f"{spec.name:18s} {off_tag:40s} p{page}: {len(products)} products")
                for p in products:
                    if p.get("code"):
                        seen.setdefault(str(p["code"]), p)
                if len(products) < page_size:
                    break
    with out_path.open("w", encoding="utf-8") as f:
        for code in sorted(seen):
            f.write(json.dumps(seen[code], ensure_ascii=False) + "\n")
    return len(seen)
