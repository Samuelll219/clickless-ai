# ClickLess AI — Knowledge Graph module (Samuel)

Scope (from the proposal, Table IV): Open Food Facts curation, synthetic pricing,
Neo4j schema & population, Cypher retrieval template.

```
clickless_kg/
  vocab.py        controlled vocabulary: categories (+OFF tag mapping, price ranges, groups),
                  dietary tags, allergens; JSON schema for the LLM extractor
  fetch_off.py    OFF v2 search API download, cached per page
  curate.py       filtering, normalisation, de-duplication, per-category quotas
  allergens.py    allergen inference from ingredient text (fills OFF's gaps)
  pricing.py      reproducible synthetic prices (seed + barcode)
  catalog.py      load/validate catalog; pure-Python reference filter (= Cypher semantics)
  schema.cypher   constraints + price index
  load.py         idempotent Neo4j loader
  retrieval.py    RETRIEVAL_QUERY + KGRetriever (find_products, cheapest_in_category, vocabulary)
scripts/          build_catalog.py, load_kg.py, query_kg.py
tests/            unit tests + Neo4j integration test (Cypher vs reference filter)
```

## Run it

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 1. Build the catalog (~10 min the first time because of OFF's rate limit; cached afterwards)
python scripts/build_catalog.py --fetch
#    -> data/catalog.json, catalog.csv, curation_report.json, vocab.json, constraint_schema.json
#    Check curation_report.json: per-category counts, and "top_unmapped_label_tags"
#    shows OFF labels we may want to add to vocab.DIETARY_TAGS.

# 2. Start Neo4j and load
docker compose up -d                  # browser UI at http://localhost:7474 (neo4j / clickless)
python scripts/load_kg.py

# 3. Try queries
python scripts/query_kg.py snacks --tags gluten_free organic --max-price 15
python scripts/query_kg.py yogurt --exclude milk           # Neo4j
python scripts/query_kg.py yogurt --exclude milk --offline # same answer from catalog.json

# 4. Tests
pytest                                                     # unit tests
NEO4J_URI=bolt://localhost:7687 pytest tests/test_neo4j_integration.py -v
#   ^ loads a synthetic catalog; re-run scripts/load_kg.py afterwards
#   (or USE_REAL_CATALOG=1 to test against data/catalog.json)
```

Commit `data/raw/off_raw.jsonl` and `data/catalog.json` once built, so everyone
works from the same snapshot (OFF changes daily). `data/raw/pages/` can stay ignored.

## Knowledge graph

```
(:Product {code, name, brand, price, ingredients_text, traces})
  -[:BELONGS_TO]->(:Category {name, display, group, price_min, price_max})
  -[:HAS_TAG]->(:DietaryTag {name})
  -[:CONTAINS]->(:Allergen {name})
(:Category)-[:SUBCATEGORY_OF]->(:Category {is_group: true})   # e.g. chips -> snacks
```

Retrieval (`retrieval.RETRIEVAL_QUERY`) is the proposal's template plus three small
extensions: `max_price = null` means no limit; the category may be a group
(`snacks`, `dairy`, `beverages`, `pantry`, `breakfast`, `bakery`), which expands to
its leaf categories; and price ties are broken by barcode so results are deterministic.
An empty result means **no matching product**. Constraints are never relaxed.

## Curation rules

**Filters.** A product is kept only if:
- it has a barcode, a name and ingredient text;
- OFF has analysed its ingredients, so an empty allergen list means "none";
- it maps onto one catalog category;
- it has at least one recognised dietary tag;
- its tags and allergens don't contradict each other (e.g. `gluten_free` + `gluten`);
- it is not a duplicate.

Then the 400 slots are spread evenly over categories (leftover slots from short
categories go to the others), most-scanned products first.

**Categories.** OFF lists a product's category tags from general to specific,
and the most specific matching tag decides the category. That way, chips which
OFF also tagged `en:cheeses` still land in `chips`. Tags in
`GLOBAL_EXCLUDE_OFF_TAGS` (meat, popcorn, creamers, ...) drop the product.
OFF's `biscuits` covers both cookies and crackers, so for those the product
name decides (`NAME_OVERRIDES`).

**Allergens = OFF tags ∪ inferred** (`clickless_kg/allergens.py`). OFF often
leaves `allergens_tags` empty: in our catalog that happened for peanut butter,
parmesan and several milks. So allergens are also matched from the
ingredient text (EN/FR/ES keywords). The matcher ignores "may contain" and
"made in a facility" clauses, "X-free" claims, plant milks and butters,
buckwheat, and maltodextrin. Milk and eggs are inferred from ingredients only
(names like "vegan cream cheese" are misleading). Products labelled vegan get
no inferred animal allergens. Milk, cheese and yogurt always contain `milk`.
`catalog.csv` has an `allergens_added` column showing what inference added, and
`curation_report.json` counts it.

**Dietary tags.** `vegan` implies `vegetarian`. By default vegan and vegetarian
may also come from OFF's definitive ingredient analysis (`en:vegan`, never
`en:maybe-vegan`). Use `--no-analysis-tags` to rely on printed labels only.

**Prices.** Uniform in the category range, `random.Random(f"{seed}:{barcode}")`,
seed 573. Adding or removing products never changes other products' prices.

## Interface for teammates

**Soumya (extractor / validator / Config C).**
`data/constraint_schema.json` is the JSON schema to pass as Ollama's `format`. Its
enums come from `vocab.py`, so extractor, validator and KG share one vocabulary.
For the validator's vocabulary check use `vocab.CATEGORY_NAMES`,
`vocab.CATEGORY_GROUPS`, `vocab.DIETARY_TAG_NAMES` and `vocab.ALLERGEN_NAMES`, or
`KGRetriever.vocabulary()` to read them from the graph. Then:
```python
from clickless_kg.retrieval import KGRetriever
with KGRetriever.from_env() as kg:
    products = kg.find_products(validated_json)   # list of dicts, [] = no match
```
Each result has `code, name, brand, price, category, dietary_tags, allergens`
(enough to explain *why* it matched).
Note: "nut-free" should be extracted as `["peanuts", "tree_nuts"]`.

**Vikas (Apriori, TF-IDF, test prompts).**
Map Instacart aisles to the **leaf** names in `vocab.CATEGORY_NAMES`.
`data/catalog.json` / `catalog.csv` contains `ingredients_text` and tags for the
TF-IDF index. For ground truth, `catalog.acceptable_set(catalog, constraints)`
returns every acceptable product code; `[]` means the prompt is infeasible.

**Sujay (UI / cart).**
Product IDs are OFF barcodes (`code`, string). Price is a float in USD.

**Apriori add-ons.**
`kg.cheapest_in_category(category, dietary_tags, excluded_allergens)`.

## Sujay UI / cart — completed

Sujay's UI and cart implementation is complete on the `Sujay-ui` branch. Soumya's Config C pipeline is integrated directly into this branch, connecting the browser shopping flow to constraint extraction, validation, and Neo4j retrieval.

The UI work includes:

- A responsive shopping page with a natural-language request form.
- A local Python web server that calls `run_config_c()` and returns extracted constraints, matches, quantity, and no-match status.
- Product results with price, brand, category, dietary tags, and allergen details.
- A cart that adds products at the requested quantity, groups by OFF barcode string, and supports quantity changes and removal.
- USD subtotals calculated in cents, with cart contents saved in the browser.
- User-facing messages for no matches, invalid requests, and service errors.

### Run the shopping UI and cart

From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen2.5:7b
# Start Ollama with ollama serve if it isn't already running.
docker compose up -d
python scripts/load_kg.py
python -m clickless_ui
```

Open [ClickLess AI](http://127.0.0.1:8000). Enter a request, review the extracted
criteria, and choose a matching product to add at the extracted quantity. A
no-match response leaves the cart as it was. The price filter follows the KG's
per-unit USD limit. The cart groups items by exact OFF barcode, calculates
totals in cents, and saves quantities in this browser.
