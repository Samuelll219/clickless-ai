# ClickLess AI — Local LLM Pipeline

This module implements the local LLM components for ClickLess AI.

It is designed to integrate with:

- `clickless_kg/` for Samuel's Neo4j knowledge graph and deterministic retrieval.
- Vikas's TF-IDF candidate retrieval for Config B.
- The downstream UI/cart layer for final execution.

The LLM does not generate Cypher and does not directly modify the cart.

## Architecture

### Constraint extraction

Natural-language request:

    "Get me 2 organic gluten-free snacks under $15 without peanuts"

becomes:

    {
        "category": "snacks",
        "dietary_tags": ["organic", "gluten_free"],
        "excluded_allergens": ["peanuts"],
        "max_price": 15,
        "quantity": 2
    }

The schema and allowed vocabulary come directly from:

    clickless_kg.vocab.constraint_json_schema()

This keeps the LLM extractor, validator, and knowledge graph on the same
controlled vocabulary.

## Files

    clickless_llm/
        extractor.py
        validator.py
        pipeline.py
        config_b.py
        config_c.py
        evaluation.py

### extractor.py

Calls the local Ollama model using schema-constrained structured output.

Default model:

    qwen2.5:7b

The extractor converts natural-language shopping requests into the shared
constraint JSON format.

### validator.py

Validates:

- category
- dietary tags
- excluded allergens
- maximum price
- quantity
- required fields
- unexpected fields

Vocabulary values are imported from `clickless_kg.vocab` rather than
hard-coded separately.

### pipeline.py

Runs:

    extractor -> validator -> corrective retry

The default behavior allows one initial attempt plus up to two retries.

When validation fails, the validation error is included in the corrective
request so the LLM can repair its extraction.

### config_b.py

Implements the LLM portion of experimental Config B:

    TF-IDF -> top 20 candidates -> local LLM selection

Vikas's TF-IDF implementation should pass its candidate dictionaries into:

    from clickless_llm.config_b import select_from_candidates

    selected = select_from_candidates(
        user_request,
        candidates,
    )

Config B considers at most the first 20 candidates.

The selected product must come from the supplied candidate set. If the LLM
returns an unknown product code, the selection is rejected.

If no candidate satisfies the request:

    selected is None

Expected candidate fields include:

    code
    name
    brand
    price
    category
    dietary_tags
    allergens
    ingredients_text

Missing optional descriptive fields are handled safely.

### config_c.py

Implements the full constraint-aware Config C pipeline:

    user request
        -> Qwen constraint extraction
        -> validation / corrective retries
        -> Neo4j KG retrieval
        -> matching products

Usage:

    from clickless_llm.config_c import run_config_c

    result = run_config_c(
        "Get me an organic gluten-free snack under $15 without peanuts"
    )

The result has the form:

    {
        "request": "...",
        "constraints": {...},
        "products": [...],
        "status": "matched",
        "quantity": 1
    }

If no KG product satisfies all constraints:

    result["status"] == "no_match"
    result["products"] == []

Constraints are never silently relaxed.

## Setup

### Python environment

Python 3.11 is recommended.

Install repository dependencies:

    pip install -r requirements.txt

### Ollama

Install Ollama and make sure the server is running.

Download the project model:

    ollama pull qwen2.5:7b

Start Ollama if needed:

    ollama serve

The local API is expected at:

    http://127.0.0.1:11434

### Neo4j for Config C

Start the project Neo4j container:

    docker compose up -d

Load the catalog:

    python scripts/load_kg.py

The repository defaults are:

    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=clickless

Environment variables can override these values.

## Tests

Run the complete repository test suite:

    pytest -v

Tests marked `integration` require local services such as Ollama and/or
Neo4j.

Run tests without live integration tests:

    pytest -m "not integration" -v

Run only integration tests:

    pytest -m integration -v

The implementation was regression-tested together with the existing
knowledge-graph code.

## Constraint Extraction Evaluation

The labelled evaluation prompts are stored in:

    data/evaluation/constraint_prompts.json

Run the extraction evaluation from the repository root:

    PYTHONPATH=. python scripts/evaluate_constraints.py

Detailed predictions and metrics are written to:

    data/evaluation/constraint_results.json

Current evaluation on the included 20-prompt labelled set:

    Successful extractions:     20 / 20
    Exact-match accuracy:       95%
    Category accuracy:          100%
    Dietary-tag accuracy:       95%
    Allergen accuracy:          100%
    Maximum-price accuracy:     100%
    Quantity accuracy:          100%

The remaining observed extraction error is an occasional omission of the
`fair_trade` dietary tag. The evaluation result is retained rather than
further tuning specifically against the evaluation set.

## Integration Notes

### Vikas — TF-IDF / Config B

Provide the top-ranked TF-IDF candidates as a list of dictionaries:

    selected = select_from_candidates(
        user_request,
        top_20_candidates,
    )

The LLM selector will never intentionally return a product outside that
candidate set.

### Samuel — Knowledge Graph / Config C

Config C uses the existing interface:

    KGRetriever.find_products(constraints)

No Cypher generation is performed by the LLM.

### UI / Cart

Config C returns the requested quantity separately from the product
retrieval result.

For example:

    result["products"]
    result["quantity"]

The UI/cart layer can use these values for cart execution.

For Config B, `select_from_candidates()` returns the selected candidate
dictionary or `None`.

## Design Principles

1. The LLM has a narrow role.
2. Structured output is schema constrained.
3. Validation happens before deterministic retrieval.
4. User constraints are never silently relaxed.
5. The LLM never writes Cypher.
6. Config B cannot return an invented product to downstream execution.
7. Config C delegates product retrieval to the deterministic KG.
