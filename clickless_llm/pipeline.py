"""End-to-end LLM extraction and validation pipeline for ClickLess AI."""

from __future__ import annotations

from typing import Any

from clickless_llm.extractor import (
    DEFAULT_MODEL,
    ExtractionError,
    extract_constraints,
)
from clickless_llm.grounding import ground_constraints
from clickless_llm.validator import (
    ValidationError,
    validate_constraints,
)


class PipelineError(RuntimeError):
    """Raised when extraction remains invalid after all retries."""


def extract_and_validate(
    user_request: str,
    max_retries: int = 2,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Extract and validate constraints with corrective retries.

    The first attempt uses the original user request.

    If validation fails, the validation error is included in the next
    request so the LLM has explicit feedback about what must be corrected.

    max_retries=2 means:
        1 initial attempt + up to 2 corrective attempts.
    """

    if not user_request or not user_request.strip():
        raise ValueError("user_request must not be empty")

    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    original_request = user_request.strip()
    current_request = original_request
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            extracted = extract_constraints(
                current_request,
                model=model,
            )

            validated = validate_constraints(extracted)
            # Deterministic guard: drop tags the user never stated, fix milk type.
            return ground_constraints(original_request, validated)

        except ValidationError as exc:
            last_error = exc

            if attempt == max_retries:
                break

            current_request = (
                f"{original_request}\n\n"
                "Your previous constraint extraction was invalid.\n"
                f"Validation error: {exc}\n"
                "Extract the constraints again from the ORIGINAL "
                "shopping request.\n"
                "Correct the validation problem without inventing, "
                "removing, or relaxing any user constraint."
            )

        except ExtractionError as exc:
            last_error = exc

            if attempt == max_retries:
                break

            # Communication or malformed-response failures may be
            # transient, so retry the original request.
            current_request = original_request

    raise PipelineError(
        "Could not produce valid constraints after "
        f"{max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
