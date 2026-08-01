"""Metrics and calculation helper functions for model evaluation."""

from __future__ import annotations

from typing import Any


def calculate_average_accuracy(scores: list[float]) -> float:
    """Calculate average accuracy score across samples."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def calculate_average_time(times: list[float]) -> float:
    """Calculate average extraction time in seconds."""
    if not times:
        return 0.0
    return sum(times) / len(times)


def calculate_statistics(scores: list[float]) -> dict[str, float]:
    """Calculate min, max, and average accuracy statistics."""
    if not scores:
        return {
            "average_accuracy": 0.0,
            "highest_accuracy": 0.0,
            "lowest_accuracy": 0.0,
        }
    return {
        "average_accuracy": calculate_average_accuracy(scores),
        "highest_accuracy": max(scores),
        "lowest_accuracy": min(scores),
    }


def calculate_field_accuracy(samples: list[dict[str, Any]]) -> dict[str, float]:
    """
    Calculate average accuracy for each individual field across all successful samples.

    Fields tracked: vendor, bill_no, date, amount, currency, gst.
    """
    fields = ["vendor", "bill_no", "date", "amount", "currency", "gst"]
    field_totals: dict[str, float] = {field: 0.0 for field in fields}
    field_counts: dict[str, int] = {field: 0 for field in fields}

    for sample in samples:
        # Handle both raw dicts and SampleScore object dict conversions
        scores_list = sample.get("field_scores", [])
        for field_score in scores_list:
            if isinstance(field_score, dict):
                field_name = field_score.get("field")
                score = field_score.get("score", 0.0)
            else:
                field_name = getattr(field_score, "field", None)
                score = getattr(field_score, "score", 0.0)

            if field_name in field_totals:
                field_totals[field_name] += float(score)
                field_counts[field_name] += 1

    return {
        field: (field_totals[field] / field_counts[field] if field_counts[field] > 0 else 0.0)
        for field in fields
    }


def calculate_model_rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank models by balancing accuracy with estimated cost."""
    # Weight (per-second) applied to average extraction time when computing rank.
    # Smaller values make time less influential; default chosen conservatively.
    time_weight = 0.02

    def ranking_key(res: dict[str, Any]) -> tuple[float, float, float]:
        accuracy = float(res.get("average_accuracy", 0.0))
        cost_per_100 = float(res.get("cost_per_100_bills", 0.0))
        avg_time = float(res.get("average_extraction_time", 0.0))

        # Combine accuracy, cost, and time into a single adjusted score.
        # Higher is better. Cost and time are treated as penalties.
        adjusted_score = accuracy - (cost_per_100 * 0.5) - (avg_time * time_weight)
        # Tie-breakers: prefer higher accuracy, then lower cost.
        return (adjusted_score, accuracy, -cost_per_100)

    return sorted(results, key=ranking_key, reverse=True)