"""Manual mapping of Instacart aisles to catalog LEAF categories (proposal Sec. II).

The Instacart 2017 release has 134 aisles in 21 departments. Each ordered
product is replaced by its aisle and each aisle by at most one catalog leaf
category (`vocab.CATEGORY_NAMES`). Aisles with no counterpart in the 400-product
Open Food Facts catalog (produce, meat, frozen, household, personal care,
alcohol, pets, ...) stay UNMAPPED and are dropped from the transactions; we
never force an aisle onto the nearest category.

Confidence levels (from the share of the aisle's products that belong to the
leaf, judged from product names):
  strong   most products in the aisle belong to the leaf
  partial  mixed aisle; the leaf is the plurality (flagged in the report)
  optional weak match; excluded by default, used only for a sensitivity run

Aisle ids/names follow aisles.csv of the Kaggle "instacart-market-basket-
analysis" files; `validate_against_aisles_csv` re-checks them after download.
"""
from __future__ import annotations

import csv
from pathlib import Path

from clickless_kg import vocab

STRONG, PARTIAL, OPTIONAL = "strong", "partial", "optional"

# aisle_id: (aisle name, leaf category, confidence, note)
AISLE_MAP: dict[int, tuple[str, str, str, str]] = {
    2: ("specialty cheeses", "cheese", STRONG, ""),
    21: ("packaged cheese", "cheese", STRONG, ""),
    108: ("other creams cheeses", "cheese", PARTIAL, "cottage/ricotta/cream cheese + sour cream"),
    84: ("milk", "milk", STRONG, "mostly cow milk; some kefir/flavoured milk"),
    91: ("soy lactosefree", "plant_milk", PARTIAL, "~72% plant milks; also lactose-free cow milk"),
    120: ("yogurt", "yogurt", STRONG, "includes drinkable yogurt"),
    88: ("spreads", "nut_butters", PARTIAL, "~37% nut/seed butters, rest jams/hummus; only nut-butter aisle"),
    121: ("cereal", "breakfast_cereal", STRONG, ""),
    57: ("granola", "breakfast_cereal", STRONG, "catalog puts en:granolas in breakfast_cereal"),
    130: ("hot cereal pancake mixes", "breakfast_cereal", PARTIAL, "oats yes, pancake mixes no"),
    3: ("energy granola bars", "snack_bars", STRONG, ""),
    48: ("breakfast bars pastries", "snack_bars", OPTIONAL, "~45% bars, rest toaster pastries"),
    107: ("chips pretzels", "chips", PARTIAL, "~65% chips; pretzels not in catalog chips"),
    78: ("crackers", "crackers", STRONG, ""),
    61: ("cookies cakes", "cookies", PARTIAL, "cookies + snack cakes"),
    45: ("candy chocolate", "chocolate_candy", STRONG, ""),
    117: ("nuts seeds dried fruit", "nuts_seeds", PARTIAL, "nuts/seeds + dried fruit"),
    125: ("trail mix snack mix", "nuts_seeds", STRONG, "catalog maps en:trail-mix to nuts_seeds"),
    112: ("bread", "bread", STRONG, ""),
    128: ("tortillas flat bread", "bread", STRONG, "catalog bread includes en:wraps"),
    93: ("breakfast bakery", "bread", PARTIAL, "bagels/English muffins + pastries"),
    43: ("buns rolls", "bread", OPTIONAL, ""),
    63: ("grains rice dried goods", "rice_grains", STRONG, ""),
    68: ("bulk grains rice dried goods", "rice_grains", STRONG, "small aisle"),
    131: ("dry pasta", "pasta", STRONG, ""),
    12: ("fresh pasta", "pasta", OPTIONAL, "fresh/filled pasta; catalog pasta is dry"),
    9: ("pasta sauce", "pasta_sauce", STRONG, ""),
    69: ("soup broth bouillon", "soup", STRONG, "catalog soup includes broths"),
    98: ("juice nectars", "juice", STRONG, ""),
    31: ("refrigerated", "juice", PARTIAL, "~48% juice; also kombucha, cold brew"),
    113: ("frozen juice", "juice", OPTIONAL, "frozen concentrate"),
    26: ("coffee", "coffee_tea", STRONG, ""),
    94: ("tea", "coffee_tea", STRONG, ""),
}

N_AISLES = 134


def mapping(include_optional: bool = False) -> dict[int, str]:
    """aisle_id -> leaf category (strong + partial; optional only if requested)."""
    return {aid: leaf for aid, (_, leaf, conf, _) in AISLE_MAP.items()
            if include_optional or conf != OPTIONAL}


def validate_mapping() -> list[str]:
    problems = []
    for aid, (name, leaf, conf, _) in AISLE_MAP.items():
        if not 1 <= aid <= N_AISLES:
            problems.append(f"aisle {aid}: id out of range")
        if leaf not in vocab.CATEGORY_NAMES:
            problems.append(f"aisle {aid} ({name}): {leaf!r} is not a leaf category")
        if conf not in (STRONG, PARTIAL, OPTIONAL):
            problems.append(f"aisle {aid}: bad confidence {conf!r}")
    missing = set(vocab.CATEGORY_NAMES) - set(mapping().values())
    if missing:
        problems.append(f"leaf categories without any aisle: {sorted(missing)}")
    return problems


def validate_against_aisles_csv(path: str | Path) -> list[str]:
    """Check ids/names against the downloaded aisles.csv."""
    with open(path, newline="", encoding="utf-8") as fh:
        names = {int(r["aisle_id"]): r["aisle"] for r in csv.DictReader(fh)}
    problems = []
    if len(names) != N_AISLES:
        problems.append(f"aisles.csv has {len(names)} aisles, expected {N_AISLES}")
    for aid, (name, *_rest) in AISLE_MAP.items():
        if names.get(aid) != name:
            problems.append(f"aisle {aid}: expected {name!r}, aisles.csv says {names.get(aid)!r}")
    return problems


def mapping_table() -> list[dict]:
    """Rows for the report (mapped and optional aisles only)."""
    return [{"aisle_id": aid, "aisle": name, "leaf": leaf, "confidence": conf, "note": note}
            for aid, (name, leaf, conf, note) in sorted(AISLE_MAP.items(), key=lambda kv: (kv[1][1], kv[0]))]
