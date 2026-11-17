"""Local LLM constraint extractor for ClickLess AI."""

from __future__ import annotations

import json
from typing import Any

import requests

from clickless_kg.vocab import constraint_json_schema


OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"


SYSTEM_PROMPT = """
You are the constraint extraction component of ClickLess AI.

Convert the user's grocery-shopping request into the provided JSON schema.
Preserve every constraint explicitly stated by the user.

CATEGORY RULES:
- Preserve the specificity of the user's category request.
- "snack" or "snacks" without a specific snack type -> "snacks".
- "snack bar", "granola bar", "protein bar", or "energy bar" -> "snack_bars".
- Never choose a more specific category unless the user clearly requests it.

DIETARY TAG RULES:
Map every explicitly requested dietary property:
- "gluten-free" or "gluten free" -> "gluten_free"
- "organic" -> "organic"
- "vegan" -> "vegan"
- "vegetarian" -> "vegetarian"
- "non-GMO" or "non GMO" -> "non_gmo"
- "lactose-free" or "lactose free" -> "lactose_free"
- "kosher" -> "kosher"
- "no added sugar" or "no sugar added" -> "no_added_sugar"
- "fair trade" -> "fair_trade"
- If multiple dietary properties are stated, include ALL of them.
- Do not omit one dietary tag because another dietary tag is present.
- If no dietary property is requested, dietary_tags must be [].

ALLERGEN RULES:
- excluded_allergens contains only allergens the user explicitly wants to avoid.
- If the user says "nut-free" or "nut free", exclude both "peanuts" and "tree_nuts".
- Never infer an allergen exclusion from a dietary tag or product category.
- Only add an allergen when the user explicitly asks to avoid it, states an allergy to it, or explicitly requests that allergen-free property.
- "gluten-free" is the dietary tag "gluten_free"; do not infer unrelated allergen exclusions.
- If no allergen restriction is specified, excluded_allergens must be [].

PRICE AND QUANTITY:
- Extract the user's maximum price when specified.
- If no maximum price is specified, max_price must be null.
- Extract the requested quantity when specified.
- If no quantity is specified, quantity must be 1.

GENERAL RULES:
- Extract only information stated or clearly implied by the user.
- Use only values permitted by the provided JSON schema.
- Never invent additional constraints.
- Never drop an explicitly stated constraint.
- Return only data matching the provided schema.
""".strip()


class ExtractionError(RuntimeError):
    """Raised when the local LLM cannot produce usable constraint JSON."""


def extract_constraints(
    user_request: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 120,
) -> dict[str, Any]:
    """Extract structured constraints from a shopping request."""

    if not user_request or not user_request.strip():
        raise ValueError("user_request must not be empty")

    schema = constraint_json_schema()

    payload = {
        "model": model,
        "stream": False,
        "format": schema,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request.strip()},
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
        raise ExtractionError(
            "Could not communicate with the local Ollama server."
        ) from exc

    try:
        content = response.json()["message"]["content"]
        extracted = json.loads(content)

    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ExtractionError(
            "Ollama returned an invalid structured response."
        ) from exc

    return extracted
