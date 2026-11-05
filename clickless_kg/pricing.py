"""Synthetic, reproducible prices.

Each product's price is drawn uniformly from its category's range using a
random generator seeded by (global seed, barcode). Seeding per product rather
than drawing from one shared stream means adding or removing a product never
changes any other product's price, so the catalog stays reproducible even
when curation rules change.
"""
from __future__ import annotations

import random

from . import vocab


def assign_price(code: str, category: str, seed: int = 573) -> float:
    lo, hi = vocab.CATEGORY_BY_NAME[category].price_range
    rng = random.Random(f"{seed}:{code}")  # str seeds are hashed deterministically (sha512)
    return round(rng.uniform(lo, hi), 2)
