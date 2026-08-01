"""Evaluation framework runner for bill extraction models."""

from __future__ import annotations

import time
from typing import Any

from evaluation.cost import (
    calculate_cost_per_100_bills,
    calculate_cost_per_bill,
)
from evaluation.metrics import (
    calculate_average_time,
    calculate_field_accuracy,
    calculate_statistics,
)
from evaluation.scorer import BillEvaluator


def evaluate_model(
    extractor_name: str, extractor: Any, dataset_samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Evaluate a single model on all dataset samples with timing and field-level scoring.
    """
    evaluator = BillEvaluator(fuzzy_threshold=0.85)
    sample_scores = []
    extraction_times: list[float] = []
    failed_samples: list[dict[str, str]] = []

    for sample in dataset_samples:
        image_name = sample["image_name"]
        ground_truth = sample["ground_truth"]
        image_path = sample["image_path"]

        start_time = time.perf_counter()
        try:
            prediction = extractor.extract_bill(image_path)
            elapsed_time = time.perf_counter() - start_time
            extraction_times.append(elapsed_time)

            score = evaluator.score_sample(
                image_name, extractor_name, ground_truth, prediction
            )
            sample_scores.append(score)

        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            print(f"  ❌ {extractor_name} failed on {image_name}: {e}")
            failed_samples.append({
                "image_name": image_name,
                "error": str(e),
            })

    # Prepare sample score dicts for statistics and field calculations
    sample_dicts = [s.to_dict() for s in sample_scores]
    accuracy_scores = [s.overall_accuracy for s in sample_scores]

    stats = calculate_statistics(accuracy_scores)
    avg_time = calculate_average_time(extraction_times)
    field_acc = calculate_field_accuracy(sample_dicts)

    cost_per_bill = calculate_cost_per_bill(extractor_name)
    cost_per_100 = calculate_cost_per_100_bills(extractor_name)

    return {
        "model": extractor_name,
        "sample_count": len(sample_scores),
        "average_accuracy": stats["average_accuracy"],
        "highest_accuracy": stats["highest_accuracy"],
        "lowest_accuracy": stats["lowest_accuracy"],
        "average_extraction_time": avg_time,
        "field_accuracy": field_acc,
        "cost_per_bill": cost_per_bill,
        "cost_per_100_bills": cost_per_100,
        "failed_samples": failed_samples,
        "samples": sample_dicts,
    }