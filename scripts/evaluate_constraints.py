"""Run ClickLess AI constraint-extraction evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from clickless_llm.evaluation import evaluate_extractor
from clickless_llm.pipeline import extract_and_validate


DATASET_PATH = Path("data/evaluation/constraint_prompts.json")


def main() -> None:
    """Evaluate the real LLM pipeline on the labelled prompt set."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    print(
        f"Evaluating {len(dataset)} constraint prompts..."
    )
    print()

    predictions = []

    for index, example in enumerate(
        dataset,
        start=1,
    ):
        prompt = example["prompt"]
        expected = example["expected"]

        print(
            f"[{index}/{len(dataset)}] {prompt}"
        )

        try:
            predicted = extract_and_validate(prompt)

            predictions.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "predicted": predicted,
                }
            )

            print(
                f"  predicted: {predicted}"
            )

        except Exception as exc:
            predictions.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "predicted": None,
                    "error": str(exc),
                }
            )

            print(
                f"  ERROR: {exc}"
            )

    successful_pairs = [
        (
            row["predicted"],
            row["expected"],
        )
        for row in predictions
        if row["predicted"] is not None
    ]

    failed_count = (
        len(predictions)
        - len(successful_pairs)
    )

    metrics = evaluate_extractor(
        [
            {
                "prompt": row["prompt"],
                "expected": row["expected"],
            }
            for row in predictions
            if row["predicted"] is not None
        ],
        lambda prompt: next(
            row["predicted"]
            for row in predictions
            if row["prompt"] == prompt
        ),
    )

    # Extraction failures count as incorrect examples in the
    # overall exact-match metric.
    total = len(predictions)

    exact_matches = metrics["exact_matches"]

    overall_exact_accuracy = (
        exact_matches / total
        if total
        else 0.0
    )

    print()
    print("=" * 60)
    print("CLICKLESS AI CONSTRAINT EXTRACTION EVALUATION")
    print("=" * 60)

    print(
        f"Total prompts:          {total}"
    )

    print(
        f"Successful extractions: {len(successful_pairs)}"
    )

    print(
        f"Extraction failures:    {failed_count}"
    )

    print(
        f"Exact matches:          {exact_matches}"
    )

    print(
        "Exact-match accuracy:   "
        f"{overall_exact_accuracy:.2%}"
    )

    print()
    print("Per-field accuracy:")

    for field, accuracy in metrics[
        "field_accuracy"
    ].items():
        # If extraction itself failed, those examples should
        # count as incorrect for every field.
        adjusted_correct = metrics[
            "field_correct"
        ][field]

        adjusted_accuracy = (
            adjusted_correct / total
            if total
            else 0.0
        )

        print(
            f"  {field:20s} "
            f"{adjusted_accuracy:.2%}"
        )

    print("=" * 60)

    output_path = Path(
        "data/evaluation/constraint_results.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "metrics": {
                    "total": total,
                    "successful_extractions": len(
                        successful_pairs
                    ),
                    "extraction_failures": failed_count,
                    "exact_matches": exact_matches,
                    "exact_match_accuracy": (
                        overall_exact_accuracy
                    ),
                    "field_accuracy": {
                        field: (
                            metrics["field_correct"][field]
                            / total
                            if total
                            else 0.0
                        )
                        for field in metrics[
                            "field_correct"
                        ]
                    },
                },
                "predictions": predictions,
            },
            file,
            indent=2,
        )

    print()
    print(
        f"Detailed results written to {output_path}"
    )


if __name__ == "__main__":
    main()
