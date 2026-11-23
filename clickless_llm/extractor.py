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

Your task is extraction, not recommendation.
Preserve every constraint explicitly stated by the user.
Never add a constraint merely because it is common, plausible, healthy,
or associated with another constraint.

CATEGORY RULES:
- Preserve the specificity of the user's category request.
- "snack" or "snacks" without a specific snack type -> "snacks".
- "snack bar", "granola bar", "protein bar", or "energy bar" -> "snack_bars".
- "breakfast cereal", "breakfast cereals", "cereal", "granola", or "muesli"
  when requested as the product type -> "breakfast_cereal".
- "coffee", "coffees", "tea", "teas", or "coffee or tea" -> "coffee_tea".
- "plant-based milk", "plant milk", "non-dairy milk", "milk alternative",
  "oat milk", "almond milk", "soy milk", or "rice milk" -> "plant_milk"
  (never "milk"; "milk" means dairy milk only).
- "lactose-free milk", "whole milk", "2% milk", or plain "milk" is dairy
  milk -> "milk" (lactose-free cow milk is still "milk", not "plant_milk").
- Never choose a more specific or different category unless the user's
  product words support that category.
- Do not infer a category from dietary tags, allergens, price, or quantity.
- If the user explicitly names a valid specific product category, always use that specific category instead of its broader parent group.
- Explicit leaf categories take priority over category groups.
- For example, "chips" -> "chips", not "snacks"; "cookies" -> "cookies", not "snacks"; "yogurt" -> "yogurt", not "dairy".

DIETARY TAG RULES:
Include a dietary tag ONLY when the user explicitly requests the
corresponding dietary property.

Use these mappings:
- "gluten-free" or "gluten free" -> "gluten_free"
- "organic" -> "organic"
- "vegan" -> "vegan"
- "vegetarian" -> "vegetarian"
- "non-GMO", "non GMO", or "non gmo" -> "non_gmo"
- "lactose-free" or "lactose free" -> "lactose_free"
- "kosher" -> "kosher"
- "no added sugar" or "no sugar added" -> "no_added_sugar"
- "fair trade" or "fair-trade" -> "fair_trade"

IMPORTANT DIETARY RULES:
- If multiple dietary properties are explicitly stated, include ALL of them.
- Do not omit one dietary tag because another dietary tag is present.
- Never infer "no_added_sugar" merely because the request contains the
  word "no", "without", "free", or an allergen exclusion.
- "no_added_sugar" is allowed ONLY when the user explicitly says
  "no added sugar" or "no sugar added".
- "nut-free", "no peanuts", "no tree nuts", "without milk", and similar
  allergen restrictions are NOT "no_added_sugar".
- Never infer a dietary tag from the product category.
- If no dietary property is explicitly requested, dietary_tags must be [].

ALLERGEN RULES:
- excluded_allergens contains only allergens the user explicitly wants
  to avoid.
- "nut-free" or "nut free" -> exclude both "peanuts" and "tree_nuts".
- "no peanuts", "without peanuts", or a peanut allergy -> "peanuts".
- "no tree nuts", "without tree nuts", or a tree-nut allergy -> "tree_nuts".
- Never infer an allergen exclusion from a dietary tag or product category.
- Only add an allergen when the user explicitly asks to avoid it,
  states an allergy to it, or explicitly requests that allergen-free property.
- "gluten-free" maps to dietary tag "gluten_free".
- Never add "vegan" or any other tag because the product is plant-based.
- Do not infer unrelated allergen exclusions from "gluten-free".
- If no allergen restriction is specified, excluded_allergens must be [].

PRICE AND QUANTITY RULES:
- Extract the user's maximum price when specified.
- "under $X", "below $X", "at most $X", and similar phrases set max_price to X.
- If no maximum price is specified, max_price must be null.
- Extract the requested quantity when specified.
- If no quantity is specified, quantity must be 1.

GENERAL RULES:
- Extract only information stated or clearly implied by the user's words.
- Use only values permitted by the provided JSON schema.
- Never invent additional constraints.
- Never drop an explicitly stated constraint.
- Do not reinterpret one type of constraint as another.
- Pay attention to which word a negation such as "no" or "without" modifies.
- Return only data matching the provided JSON schema.
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
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_request.strip(),
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
        raise ExtractionError(
            "Could not communicate with the local Ollama server."
        ) from exc

    try:
        content = response.json()["message"]["content"]
        extracted = json.loads(content)

    except (
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise ExtractionError(
            "Ollama returned an invalid structured response."
        ) from exc

    return extracted
