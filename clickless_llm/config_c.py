"""Config C: LLM constraint extraction + deterministic KG retrieval.

Config C is the full ClickLess AI constraint-aware retrieval pipeline:

    natural-language request
        -> local LLM constraint extraction
        -> validation and corrective retries
        -> Neo4j knowledge-graph retrieval
        -> matching products or explicit no-match result

The LLM never writes Cypher and never directly selects a product.
"""

from __future__ import annotations

from typing import Any

from clickless_kg.retrieval import KGRetriever
from clickless_llm.extractor import DEFAULT_MODEL
from clickless_llm.pipeline import extract_and_validate


def run_config_c(
    user_request: str,
    *,
    limit: int = 5,
    max_retries: int = 2,
    model: str = DEFAULT_MODEL,
    kg: KGRetriever | None = None,
) -> dict[str, Any]:
    """Run the complete Config C pipeline.

    Parameters
    ----------
    user_request:
        Original natural-language grocery-shopping request.

    limit:
        Maximum number of matching products returned by the KG.

    max_retries:
        Number of corrective LLM retries after the initial extraction.

    model:
        Ollama model used for constraint extraction.

    kg:
        Optional existing KGRetriever. Supplying one is useful for
        applications that keep a Neo4j connection open. If omitted,
        Config C creates and closes its own retriever.

    Returns
    -------
    dict
        A structured result containing:

        request:
            The original user request.

        constraints:
            Validated constraint JSON produced by the LLM pipeline.

        products:
            Matching KG products ordered deterministically by Samuel's
            retriever.

        status:
            "matched" when products were found, otherwise "no_match".

        quantity:
            Requested quantity from the validated constraints.

    Notes
    -----
    An empty KG result is reported as "no_match". Config C never relaxes
    user constraints automatically.
    """

    if not user_request or not user_request.strip():
        raise ValueError("user_request must not be empty")

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be an integer greater than or equal to 1")

    constraints = extract_and_validate(
        user_request,
        max_retries=max_retries,
        model=model,
    )

    owns_kg = kg is None

    if owns_kg:
        kg = KGRetriever.from_env()

    try:
        products = kg.find_products(
            constraints,
            limit=limit,
        )
    finally:
        if owns_kg:
            kg.close()

    status = "matched" if products else "no_match"

    return {
        "request": user_request.strip(),
        "constraints": constraints,
        "products": products,
        "status": status,
        "quantity": constraints["quantity"],
    }
