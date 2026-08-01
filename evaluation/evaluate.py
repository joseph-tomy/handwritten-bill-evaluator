"""Run evaluation across dataset and models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dataset.loader import load_dataset
from evaluation.scorer import BillEvaluator, SampleScore


def evaluate_model(
    extractor_name: str, extractor: Any, dataset_samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Evaluate a single model on all dataset samples.

    Returns a report with per-sample and aggregate accuracy.
    """
    evaluator = BillEvaluator(fuzzy_threshold=0.85)
    sample_scores = []

    for sample in dataset_samples:
        image_name = sample["image_name"]
        ground_truth = sample["ground_truth"]
        image_path = sample["image_path"]

        try:
            prediction = extractor.extract_bill(image_path)
        except Exception as e:
            print(f"  ❌ {extractor_name} failed on {image_name}: {e}")
            continue

        score = evaluator.score_sample(image_name, extractor_name, ground_truth, prediction)
        sample_scores.append(score)

    # Aggregate results
    if not sample_scores:
        return {
            "model": extractor_name,
            "sample_count": 0,
            "average_accuracy": 0.0,
            "samples": [],
        }

    avg_accuracy = sum(s.overall_accuracy for s in sample_scores) / len(sample_scores)

    return {
        "model": extractor_name,
        "sample_count": len(sample_scores),
        "average_accuracy": avg_accuracy,
        "samples": [s.to_dict() for s in sample_scores],
    }


def generate_report(results: dict[str, Any]) -> str:
    """Generate a readable text report from evaluation results."""
    lines = [
        "=" * 80,
        "BILL EXTRACTION EVALUATION REPORT",
        "=" * 80,
        "",
    ]

    models_summary = []
    for model_result in results["models"]:
        model_name = model_result["model"]
        avg_acc = model_result["average_accuracy"]
        sample_count = model_result["sample_count"]
        models_summary.append((model_name, avg_acc, sample_count))

        lines.append(f"\n📊 Model: {model_name}")
        lines.append(f"   Samples: {sample_count}")
        lines.append(f"   Average Accuracy: {avg_acc:.2%}")
        lines.append("")

        for sample in model_result["samples"]:
            image_name = sample["image_name"]
            overall = sample["overall_accuracy"]
            lines.append(f"   - {image_name}: {overall:.2%}")
            for field_score in sample["field_scores"]:
                field = field_score["field"]
                score = field_score["score"]
                match_type = field_score["match_type"]
                lines.append(f"      • {field}: {score:.2%} ({match_type})")

    # Summary table
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"{'Model':<20} {'Accuracy':<15} {'Samples':<10}")
    lines.append("-" * 45)
    for model_name, avg_acc, sample_count in models_summary:
        lines.append(f"{model_name:<20} {avg_acc:>13.2%} {sample_count:>9}")

    return "\n".join(lines)
