"""End-to-end LLM extraction and validation pipeline for ClickLess AI."""

from __future__ import annotations

from typing import Any

from clickless_llm.extractor import extract_constraints, ExtractionError
from clickless_llm.validator import validate_constraints, ValidationError


class PipelineError(RuntimeError):
    """Raised when extraction remains invalid after all retries."""


def extract_and_validate(
    user_request: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Extract constraints and validate them, retrying invalid outputs.

    max_retries=2 means one initial attempt plus up to two retries.
    """

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            extracted = extract_constraints(user_request)
            return validate_constraints(extracted)

        except (ValidationError, ExtractionError) as exc:
            last_error = exc

            if attempt == max_retries:
                break

    raise PipelineError(
        f"Could not produce valid constraints after "
        f"{max_retries + 1} attempts. Last error: {last_error}"
    )
