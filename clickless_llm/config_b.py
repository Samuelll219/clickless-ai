"""Config B: local LLM product selection without the knowledge graph.

Config B receives candidate products from the TF-IDF retriever and asks
the local LLM to select one candidate. The LLM is never allowed to invent
a product outside the supplied candidate set.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from clickless_llm.extractor import DEFAULT_MODEL, OLLAMA_URL


class ConfigBError(RuntimeError):
    """Raised when Config B cannot produce a usable response."""


SELECTION_SYSTEM_PROMPT = """
You are the product-selection component for ClickLess AI Config B.

You receive:
1. The user's original grocery-shopping request.
2. Candidate products produced by TF-IDF retrieval.

Your job is to select one candidate only if it satisfies ALL explicit
constraints in the user's request.

Rules:
- Select ONLY from the supplied candidate products.
- Never invent a product, barcode, or product ID.
- Respect every explicit category constraint.
- Respect every explicit dietary constraint.
- Respect every explicit allergen restriction.
- Respect every explicit maximum price.
- Do not silently relax any constraint.
- A cheaper product is not acceptable if it violates another constraint.
- If multiple candidates satisfy every constraint, choose the best match.
- If NO supplied candidate satisfies EVERY explicit constraint, return
  {"code": null}.
- Returning null is always better than choosing an invalid product.
- Also return the number of units the user asked for in "quantity"
  (1 when no quantity is stated).
- Return only data matching the provided JSON schema.
""".strip()


SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {
            "type": ["string", "null"],
        },
        "quantity": {
            "type": "integer",
            "minimum": 1,
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


def _candidate_view(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Return fields useful to the LLM for product selection."""

    return {
        "code": str(candidate.get("code", "")),
        "name": candidate.get("name"),
        "brand": candidate.get("brand"),
        "price": candidate.get("price"),
        "category": candidate.get("category"),
        "dietary_tags": candidate.get("dietary_tags", []),
        "allergens": candidate.get("allergens", []),
        "ingredients_text": candidate.get(
            "ingredients_text",
            "",
        ),
    }


def select_from_candidates(
    user_request: str,
    candidates: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
    return_quantity: bool = False,
):
    """Select one product from TF-IDF candidates using the local LLM.

    The function considers at most the first 20 candidates.

    Returns the original selected candidate dictionary when the model
    selects a valid candidate.

    Returns None when:
    - the candidate list is empty,
    - the model determines no candidate satisfies the request, or
    - the model returns a product code outside the candidate set.

    A hallucinated product is therefore treated as an unsuccessful
    selection and can never reach cart execution.

    With return_quantity=True the result is (candidate_or_None, quantity),
    where quantity is the number of units the LLM read from the request
    (needed for the TSR quantity check and the cart update).
    """

    if not user_request or not user_request.strip():
        raise ValueError(
            "user_request must not be empty"
        )

    if not isinstance(candidates, list):
        raise TypeError(
            "candidates must be a list"
        )

    if not candidates:
        return (None, 1) if return_quantity else None

    # Config B is defined over the top-20 TF-IDF results.
    candidates = candidates[:20]

    candidate_views = [
        _candidate_view(candidate)
        for candidate in candidates
    ]

    user_message = {
        "request": user_request.strip(),
        "candidates": candidate_views,
    }

    payload = {
        "model": model,
        "stream": False,
        "format": SELECTION_SCHEMA,
        "messages": [
            {
                "role": "system",
                "content": SELECTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(user_message),
            },
        ],
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.RequestException as exc:
        raise ConfigBError(
            "Could not communicate with the local Ollama server."
        ) from exc

    try:
        content = response.json()["message"]["content"]
        selection = json.loads(content)
        selected_code = selection["code"]
        quantity = selection.get("quantity", 1)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            quantity = 1

    except (
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise ConfigBError(
            "Ollama returned an invalid Config B response."
        ) from exc

    # The model explicitly reports that none qualifies.
    if selected_code is None:
        return (None, quantity) if return_quantity else None

    selected_code = str(selected_code)

    candidate_by_code = {
        str(candidate.get("code")): candidate
        for candidate in candidates
    }

    # Critical hallucination guard:
    # an invented product must never leave Config B.
    #
    # Treat this as an unsuccessful selection instead of crashing,
    # because experimentally this is equivalent to Config B failing
    # to find an acceptable candidate.
    chosen = candidate_by_code.get(selected_code)
    return (chosen, quantity) if return_quantity else chosen
