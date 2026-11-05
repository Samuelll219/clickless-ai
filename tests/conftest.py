"""Synthetic Open Food Facts-shaped records for tests.

These are NOT real products (barcodes start with 999). They mimic the field
layout of the OFF v2 API so curation can be tested without network access.
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clickless_kg import vocab  # noqa: E402


def off_record(code, name, categories, labels=(), allergens=(), brand="TestBrand",
               ingredients="water, sugar, salt, natural flavor", ingredients_n=4,
               analysis=(), traces=(), scans=10):
    return {
        "code": str(code), "product_name": name, "brands": brand,
        "categories_tags": list(categories), "labels_tags": list(labels),
        "allergens_tags": list(allergens), "traces_tags": list(traces),
        "ingredients_text": ingredients, "ingredients_n": ingredients_n,
        "ingredients_analysis_tags": list(analysis), "states_tags": [],
        "unique_scans_n": scans,
    }


def synthetic_dump(per_category=30, seed=0):
    """A few hundred valid-looking records spread over every catalog category."""
    rng = random.Random(seed)
    label_pool = [a[0] for a in vocab.DIETARY_TAGS.values()]
    allergen_pool = ["en:milk", "en:soybeans", "en:nuts", "en:peanuts", "en:eggs",
                     "en:sesame-seeds"]
    recs, code = [], 999_000_000_000
    for spec in vocab.CATEGORIES:
        for i in range(per_category):
            code += 1
            labels = rng.sample(label_pool, rng.randint(1, 3))
            allergens = rng.sample(allergen_pool, rng.randint(0, 2))
            if "en:vegan" in labels:
                allergens = [a for a in allergens if a not in ("en:milk", "en:eggs")]
            recs.append(off_record(code, f"Test product {code}", ["en:foods", spec.off_tags[0]],
                                   labels, allergens, brand=f"Brand{i % 7}",
                                   scans=rng.randint(0, 1000)))
    return recs


@pytest.fixture
def dump():
    return synthetic_dump()
