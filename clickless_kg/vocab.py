"""Controlled vocabularies for the ClickLess AI knowledge graph.

This module is the single source of truth for:
  * catalog categories (and how Open Food Facts category tags map onto them),
  * the per-category synthetic price ranges,
  * the dietary-tag and allergen vocabularies (and their OFF tag aliases).

Everything downstream (curation, Neo4j loading, the validator's vocabulary
check, the Ollama JSON schema, Vikas's aisle->category mapping) should import
names from here instead of hard-coding strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CategorySpec:
    name: str                      # canonical id used in the KG and in the constraint JSON
    display: str                   # human-readable name
    group: str                     # parent category (enables requests like "snacks")
    off_tags: tuple[str, ...]      # OFF categories_tags that map to this category
    price_range: tuple[float, float]  # USD, synthetic price drawn uniformly from this range
    exclude_off_tags: tuple[str, ...] = field(default_factory=tuple)


# A product is assigned to the category whose matching OFF tag is the MOST
# SPECIFIC one, i.e. appears last in its categories_tags (OFF lists a product's
# categories from general to specific). Example: Lay's chips that OFF also
# tagged "en:cheeses" end up in chips because "en:potato-crisps" comes later.
# Ties (same tag in two categories) go to the category listed first.
# exclude_off_tags removes a category from consideration for that product.
CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec("plant_milk", "Plant-based milk", "beverages",
                 ("en:plant-based-milk-alternatives", "en:plant-milks", "en:milk-substitutes",
                  "en:almond-milks", "en:oat-milks", "en:soy-milks", "en:rice-milks",
                  "en:coconut-milks", "en:oat-based-drinks"), (2.99, 6.49),
                 exclude_off_tags=("en:coconut-waters", "en:coffees", "en:creamer",
                                   "en:coffee-creamers")),
    CategorySpec("milk", "Milk", "dairy",
                 ("en:milks", "en:cow-milks", "en:whole-milks", "en:semi-skimmed-milks",
                  "en:skimmed-milks"), (2.49, 5.99),
                 exclude_off_tags=("en:plant-based-milk-alternatives", "en:plant-milks",
                                   "en:milk-substitutes")),
    CategorySpec("yogurt", "Yogurt", "dairy", ("en:yogurts", "en:greek-style-yogurts"), (1.29, 6.99)),
    CategorySpec("cheese", "Cheese", "dairy", ("en:cheeses",), (2.99, 9.99),
                 exclude_off_tags=("en:snacks", "en:rice-puffs")),
    CategorySpec("nut_butters", "Nut butters", "pantry",
                 ("en:nut-butters", "en:peanut-butters", "en:almond-butters",
                  "en:peanut-butter-powder"), (2.99, 9.99)),
    CategorySpec("breakfast_cereal", "Breakfast cereal", "breakfast",
                 ("en:breakfast-cereals", "en:granolas", "en:mueslis", "en:rolled-oats"),
                 (2.99, 6.99)),
    CategorySpec("snack_bars", "Snack bars", "snacks",
                 ("en:cereal-bars", "en:protein-bars", "en:snack-bars", "en:bars",
                  "en:energy-bars", "en:fruit-and-nut-bars", "en:chocolate-cereal-bars"),
                 (3.49, 9.99),
                 exclude_off_tags=("en:chocolates", "en:peanut-butter-powder")),
    CategorySpec("chips", "Chips", "snacks",
                 ("en:crisps", "en:chips", "en:potato-crisps", "en:tortilla-chips",
                  "en:corn-chips", "en:vegetable-crisps", "en:pita-chips",
                  "en:puffed-salty-snacks"), (2.49, 5.99)),
    CategorySpec("crackers", "Crackers", "snacks",
                 ("en:crackers", "en:crackers-appetizers", "en:crispbreads", "en:rice-cakes",
                  "en:rice-crackers"), (2.49, 5.99)),
    CategorySpec("cookies", "Cookies", "snacks",
                 ("en:biscuits", "en:cookies", "en:chocolate-chip-cookies", "en:sandwich-biscuits"),
                 (2.99, 6.99), exclude_off_tags=("en:fats", "en:spreads")),
    CategorySpec("chocolate_candy", "Chocolate & candy", "snacks",
                 ("en:chocolates", "en:candies", "en:confectioneries", "en:chocolate-candies",
                  "en:candy-chocolate-bars", "en:dark-chocolates", "en:milk-chocolates",
                  "en:gummies"), (1.99, 7.99)),
    CategorySpec("nuts_seeds", "Nuts & seeds", "snacks",
                 ("en:nuts", "en:nuts-and-their-products", "en:peanuts", "en:chia", "en:chia-seeds",
                  "en:flax-seeds", "en:hemp-seeds", "en:pumpkin-seeds", "en:sunflower-seeds",
                  "en:trail-mix", "en:mixed-nuts"), (3.99, 13.99),
                 exclude_off_tags=("en:nut-butters", "en:peanut-butters", "en:pulses",
                                   "en:peanut-butter-powder", "en:flours", "en:nut-flours")),
    CategorySpec("bread", "Bread", "bakery", ("en:breads", "en:sliced-breads", "en:bagel-breads",
                                               "en:wraps", "en:english-muffins"), (2.99, 6.99),
                 exclude_off_tags=("en:crispbreads", "en:crackers-appetizers")),
    CategorySpec("rice_grains", "Rice & grains", "pantry",
                 ("en:rices", "en:quinoa", "en:couscous", "en:bulgur", "en:wild-rices",
                  "en:long-grain-rices", "en:basmati-rices", "en:jasmine-rice", "en:brown-rices"),
                 (1.99, 7.99)),
    CategorySpec("pasta", "Pasta", "pantry", ("en:pastas", "en:dry-pastas", "en:spaghetti",
                                               "en:penne", "en:durum-wheat-pastas"), (1.29, 4.99),
                 exclude_off_tags=("en:noodles", "en:instant-noodles", "en:meals")),
    CategorySpec("pasta_sauce", "Pasta sauce", "pantry",
                 ("en:pasta-sauces", "en:tomato-sauces", "en:marinara-sauces",
                  "en:bolognese-sauces", "en:pestos", "en:alfredo-sauces"), (2.49, 7.99),
                 exclude_off_tags=("en:ketchup", "en:hot-sauces", "en:salsas", "en:tomato-purees")),
    CategorySpec("soup", "Soup", "pantry", ("en:soups", "en:broths", "en:bone-broths"), (1.99, 5.99)),
    CategorySpec("juice", "Juice", "beverages",
                 ("en:fruit-juices", "en:juices", "en:juices-and-nectars", "en:orange-juices",
                  "en:apple-juices", "en:smoothies", "en:coconut-waters"), (2.49, 7.49),
                 exclude_off_tags=("en:sodas", "en:carbonated-drinks", "en:lemonades",
                                   "en:flavored-waters", "en:soft-drinks", "en:squashes")),
    CategorySpec("coffee_tea", "Coffee & tea", "beverages",
                 ("en:coffees", "en:teas", "en:tea-bags", "en:ground-coffees", "en:instant-coffees",
                  "en:green-teas", "en:black-teas", "en:herbal-teas", "en:iced-teas"), (3.99, 14.99),
                 exclude_off_tags=("en:creamer", "en:coffee-creamers", "en:kombuchas",
                                   "en:kombucha")),
)

# OFF's "biscuits" means both cookies and crackers (British usage), and some bars
# are only tagged as biscuits, so for products that landed in `cookies` the
# product name decides between these three.
NAME_OVERRIDES: dict[str, tuple[tuple[str, str], ...]] = {
    "cookies": ((r"\bcrackers?\b|triscuit|wheat thins|water biscuits?", "crackers"),
                (r"\bbars?\b|macrobar", "snack_bars")),
}

# Products carrying any of these tags are never put in the catalog.
GLOBAL_EXCLUDE_OFF_TAGS: tuple[str, ...] = (
    "en:meats", "en:meats-and-their-products", "en:fishes", "en:seafood",
    "en:popcorn", "en:creamer", "en:coffee-creamers", "en:dietary-supplements",
    "en:baby-foods", "en:pet-food", "en:alcoholic-beverages",
)

CATEGORY_BY_NAME = {c.name: c for c in CATEGORIES}
CATEGORY_NAMES = tuple(c.name for c in CATEGORIES)
CATEGORY_GROUPS = tuple(sorted({c.group for c in CATEGORIES}))


# Dietary tags: canonical name -> OFF labels_tags aliases.
# OFF label taxonomy uses "no-gluten" for gluten-free, "no-lactose" for lactose-free, etc.
DIETARY_TAGS: dict[str, tuple[str, ...]] = {
    "gluten_free": ("en:no-gluten", "en:gluten-free", "en:certified-gluten-free",
                    "en:gluten-free-certification-organization"),
    "organic": ("en:organic", "en:usda-organic", "en:eu-organic", "en:certified-organic",
                "en:canada-organic", "fr:ab-agriculture-biologique"),
    "vegan": ("en:vegan", "en:certified-vegan", "en:vegan-action",
              "en:european-vegetarian-union-vegan"),
    "vegetarian": ("en:vegetarian", "en:european-vegetarian-union-vegetarian"),
    "non_gmo": ("en:no-gmos", "en:non-gmo-project", "en:non-gmo-project-verified",
                "en:non-gmo"),
    "lactose_free": ("en:no-lactose", "en:lactose-free"),
    "kosher": ("en:kosher", "en:orthodox-union-kosher", "en:ok-kosher"),
    "no_added_sugar": ("en:no-added-sugar", "en:no-sugar-added"),
    "fair_trade": ("en:fair-trade", "en:fairtrade-international", "en:fair-trade-usa"),
}
DIETARY_TAG_NAMES = tuple(DIETARY_TAGS)

# Tags implied by other tags (a vegan product is also vegetarian).
IMPLIED_TAGS: dict[str, tuple[str, ...]] = {"vegan": ("vegetarian",)}

# OFF ingredients_analysis_tags that we accept as evidence for a dietary tag.
# Only definitive values ("en:vegan"), never "en:maybe-vegan".
ANALYSIS_TAGS: dict[str, str] = {"en:vegan": "vegan", "en:vegetarian": "vegetarian"}


# Allergens: canonical name -> OFF allergens_tags aliases (EU-14 + sesame as in US FALCPA).
ALLERGENS: dict[str, tuple[str, ...]] = {
    "gluten": ("en:gluten",),
    "milk": ("en:milk",),
    "eggs": ("en:eggs",),
    "peanuts": ("en:peanuts",),
    "tree_nuts": ("en:nuts",),
    "soy": ("en:soybeans",),
    "fish": ("en:fish",),
    "shellfish": ("en:crustaceans", "en:molluscs"),
    "sesame": ("en:sesame-seeds",),
    "mustard": ("en:mustard",),
    "celery": ("en:celery",),
    "lupin": ("en:lupin",),
    "sulphites": ("en:sulphur-dioxide-and-sulphites",),
}
ALLERGEN_NAMES = tuple(ALLERGENS)

# Dietary tags that contradict an allergen. Products with such a conflict are
# dropped during curation because the data is internally inconsistent.
CONFLICTS: dict[str, tuple[str, ...]] = {
    "gluten_free": ("gluten",),
    "vegan": ("milk", "eggs", "fish", "shellfish"),
    "vegetarian": ("fish", "shellfish"),
    "lactose_free": (),  # lactose-free dairy still lists milk; not a conflict
}


def _invert(mapping: dict[str, tuple[str, ...]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for canonical, aliases in mapping.items():
        for a in aliases:
            out[a] = canonical
    return out


OFF_LABEL_TO_TAG = _invert(DIETARY_TAGS)
OFF_ALLERGEN_TO_ALLERGEN = _invert(ALLERGENS)


def categories_for_request(name: str) -> list[str]:
    """Expand a requested category (leaf or group) into leaf category names."""
    if name in CATEGORY_BY_NAME:
        return [name]
    return [c.name for c in CATEGORIES if c.group == name]


def vocabulary() -> dict:
    """Plain-dict export of the controlled vocabulary (written to data/vocab.json)."""
    return {
        "categories": [
            {"name": c.name, "display": c.display, "group": c.group,
             "price_range": list(c.price_range)} for c in CATEGORIES
        ],
        "category_groups": list(CATEGORY_GROUPS),
        "dietary_tags": list(DIETARY_TAG_NAMES),
        "allergens": list(ALLERGEN_NAMES),
    }


def constraint_json_schema() -> dict:
    """JSON schema for the LLM constraint extractor (Ollama `format=`).

    Field names follow the proposal: category, dietary_tags,
    excluded_allergens, max_price, quantity. Enums come from this vocabulary,
    so the validator and the KG can never disagree on valid values.
    """
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string",
                         "enum": list(CATEGORY_NAMES) + list(CATEGORY_GROUPS)},
            "dietary_tags": {"type": "array",
                             "items": {"type": "string", "enum": list(DIETARY_TAG_NAMES)}},
            "excluded_allergens": {"type": "array",
                                   "items": {"type": "string", "enum": list(ALLERGEN_NAMES)}},
            "max_price": {"type": ["number", "null"], "minimum": 0},
            "quantity": {"type": "integer", "minimum": 1},
        },
        "required": ["category", "dietary_tags", "excluded_allergens", "max_price", "quantity"],
        "additionalProperties": False,
    }
