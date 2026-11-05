"""Open Food Facts -> curated ClickLess catalog.

Input: raw OFF product records (dicts from the v2 API, the OFF JSONL dump, or
rows of the OFF CSV dump). Output: a list of catalog products with exactly the
fields the rest of the system needs.

Curation rules (each rejected record is counted under a reason in the report):
  1. has a barcode, a product name and English-ish ingredient text
  2. ingredients were analysed by OFF (so an empty allergen list means
     "no allergens", not "unknown")
  3. maps onto one of our catalog categories
  4. carries at least `min_tags` recognised dietary tags
  5. no dietary-tag / allergen contradiction (e.g. gluten_free + gluten)
  6. not a duplicate (same normalised name + brand)
Then a per-category quota is filled, most-scanned products first.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from . import vocab
from .allergens import infer_allergens
from .pricing import assign_price


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float):  # NaN from pandas
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _as_text(value) -> str:
    if value is None or isinstance(value, float):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_record(raw: dict) -> dict:
    """Bring an API/JSONL/CSV record into one shape."""
    allergens = raw.get("allergens_tags")
    if allergens is None:  # CSV dump stores the tags in the "allergens" column
        allergens = raw.get("allergens")
    try:
        ingredients_n = int(float(raw.get("ingredients_n") or 0))
    except (TypeError, ValueError):
        ingredients_n = 0
    try:
        scans = int(float(raw.get("unique_scans_n") or 0))
    except (TypeError, ValueError):
        scans = 0
    return {
        "code": _as_text(raw.get("code")),
        "name": _as_text(raw.get("product_name_en") or raw.get("product_name")),
        "brand": _as_text(_as_list(raw.get("brands"))[0] if _as_list(raw.get("brands")) else ""),
        "categories_tags": _as_list(raw.get("categories_tags")),
        "labels_tags": _as_list(raw.get("labels_tags")),
        "allergens_tags": _as_list(allergens),
        "traces_tags": _as_list(raw.get("traces_tags")),
        "analysis_tags": _as_list(raw.get("ingredients_analysis_tags")),
        "states_tags": _as_list(raw.get("states_tags")),
        "ingredients_text": _as_text(raw.get("ingredients_text_en") or raw.get("ingredients_text")),
        "ingredients_n": ingredients_n,
        "unique_scans_n": scans,
    }


def assign_category(categories_tags: Iterable[str]) -> str | None:
    """Most-specific match wins: see the comment above vocab.CATEGORIES."""
    tags = [t.lower() for t in categories_tags]
    tagset = set(tags)
    if tagset & set(vocab.GLOBAL_EXCLUDE_OFF_TAGS):
        return None
    best, best_pos = None, -1
    for spec in vocab.CATEGORIES:
        if tagset & set(spec.exclude_off_tags):
            continue
        positions = [i for i, t in enumerate(tags) if t in spec.off_tags]
        if positions and max(positions) > best_pos:
            best, best_pos = spec.name, max(positions)
    return best


def extract_dietary_tags(labels_tags, analysis_tags=(), use_analysis: bool = True) -> list[str]:
    found = {vocab.OFF_LABEL_TO_TAG[t] for t in labels_tags if t in vocab.OFF_LABEL_TO_TAG}
    if use_analysis:
        found |= {vocab.ANALYSIS_TAGS[t] for t in analysis_tags if t in vocab.ANALYSIS_TAGS}
    for tag in list(found):
        found.update(vocab.IMPLIED_TAGS.get(tag, ()))
    return sorted(found)


def extract_allergens(allergens_tags) -> list[str]:
    return sorted({vocab.OFF_ALLERGEN_TO_ALLERGEN[t] for t in allergens_tags
                   if t in vocab.OFF_ALLERGEN_TO_ALLERGEN})


def _ingredients_analysed(rec: dict) -> bool:
    return rec["ingredients_n"] > 0 or "en:ingredients-completed" in rec["states_tags"]


def _dedupe_key(rec: dict) -> str:
    return re.sub(r"[^a-z0-9]+", " ", f"{rec['name']} {rec['brand']}".lower()).strip()


DAIRY_CATEGORIES = {"milk", "cheese", "yogurt"}


def evaluate(rec: dict, min_tags: int = 1, use_analysis: bool = True) -> tuple[dict | None, str]:
    """Return (product, "ok") or (None, rejection_reason)."""
    if not rec["code"] or not rec["name"]:
        return None, "missing_code_or_name"
    if len(rec["ingredients_text"]) < 10:
        return None, "missing_ingredients"
    if not _ingredients_analysed(rec):
        return None, "ingredients_not_analysed"
    category = assign_category(rec["categories_tags"])
    if category is None:
        return None, "no_matching_category"
    for pattern, target in vocab.NAME_OVERRIDES.get(category, ()):
        if re.search(pattern, rec["name"], re.I):
            category = target
            break
    tags = extract_dietary_tags(rec["labels_tags"], rec["analysis_tags"], use_analysis)
    if len(tags) < min_tags:
        return None, "too_few_dietary_tags"
    off_allergens = extract_allergens(rec["allergens_tags"])
    inferred = set(infer_allergens(rec["ingredients_text"], rec["name"], vegan="vegan" in tags))
    if category in DAIRY_CATEGORIES and "vegan" not in tags:
        inferred.add("milk")
    allergens = sorted(set(off_allergens) | inferred)
    for tag in tags:
        if set(vocab.CONFLICTS.get(tag, ())) & set(allergens):
            return None, f"conflict_{tag}"
    return {
        "code": rec["code"],
        "name": rec["name"],
        "brand": rec["brand"],
        "category": category,
        "dietary_tags": tags,
        "allergens": allergens,
        "allergens_added": sorted(set(allergens) - set(off_allergens)),  # found by us, not OFF
        "traces": extract_allergens(rec["traces_tags"]),
        "ingredients_text": rec["ingredients_text"],
        "labels_raw": rec["labels_tags"],
        "unique_scans_n": rec["unique_scans_n"],
    }, "ok"


def _quotas(available: dict[str, int], target: int) -> dict[str, int]:
    """Spread `target` evenly over categories; leftover from short categories
    is redistributed to categories that still have candidates."""
    quotas = {c: 0 for c in available}
    remaining = target
    open_cats = [c for c in vocab.CATEGORY_NAMES if available.get(c, 0) > 0]
    while remaining > 0 and open_cats:
        share, extra = divmod(remaining, len(open_cats))
        next_open = []
        for i, c in enumerate(open_cats):
            want = share + (1 if i < extra else 0)
            give = min(want, available[c] - quotas[c])
            quotas[c] += give
            remaining -= give
            if quotas[c] < available[c]:
                next_open.append(c)
        if next_open == open_cats and share == 0 and extra == 0:
            break
        open_cats = next_open
    return quotas


def curate(raw_records: Iterable[dict], target: int = 400, seed: int = 573,
           min_tags: int = 1, use_analysis: bool = True) -> tuple[list[dict], dict]:
    reasons: Counter = Counter()
    unmapped_labels: Counter = Counter()
    by_category: dict[str, list[dict]] = {c: [] for c in vocab.CATEGORY_NAMES}
    seen_codes: set[str] = set()
    seen_keys: set[str] = set()
    total = 0

    for raw in raw_records:
        total += 1
        rec = normalize_record(raw)
        if rec["code"] in seen_codes:
            reasons["duplicate_code"] += 1
            continue
        seen_codes.add(rec["code"])
        unmapped_labels.update(t for t in rec["labels_tags"] if t not in vocab.OFF_LABEL_TO_TAG)
        product, reason = evaluate(rec, min_tags, use_analysis)
        reasons[reason] += 1
        if product is None:
            continue
        key = _dedupe_key(rec)
        if key in seen_keys:
            reasons["ok"] -= 1
            reasons["duplicate_name_brand"] += 1
            continue
        seen_keys.add(key)
        by_category[product["category"]].append(product)

    for items in by_category.values():
        items.sort(key=lambda p: (-p["unique_scans_n"], p["code"]))
    quotas = _quotas({c: len(v) for c, v in by_category.items()}, target)

    catalog: list[dict] = []
    for cat in vocab.CATEGORY_NAMES:
        for p in by_category[cat][:quotas.get(cat, 0)]:
            p = dict(p)
            p["price"] = assign_price(p["code"], cat, seed)
            catalog.append(p)
    catalog.sort(key=lambda p: (p["category"], p["code"]))

    report = {
        "raw_records": total,
        "eligible": sum(len(v) for v in by_category.values()),
        "selected": len(catalog),
        "target": target,
        "seed": seed,
        "rejections": dict(reasons.most_common()),
        "eligible_per_category": {c: len(v) for c, v in by_category.items()},
        "selected_per_category": dict(Counter(p["category"] for p in catalog)),
        "dietary_tag_counts": dict(Counter(t for p in catalog for t in p["dietary_tags"])),
        "allergen_counts": dict(Counter(a for p in catalog for a in p["allergens"])),
        "allergens_added_by_inference": dict(Counter(a for p in catalog for a in p["allergens_added"])),
        "products_with_added_allergens": sum(1 for p in catalog if p["allergens_added"]),
        "top_unmapped_label_tags": dict(unmapped_labels.most_common(40)),
    }
    return catalog, report
