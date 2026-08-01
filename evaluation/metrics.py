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
        for field_score in sample.get("field_scores", []):
            field_name = field_score.get("field")
            score = field_score.get("score", 0.0)
            if field_name in field_totals:
                field_totals[field_name] += score
                field_counts[field_name] += 1

    return {
        field: (field_totals[field] / field_counts[field] if field_counts[field] > 0 else 0.0)
        for field in fields
    }


def calculate_model_rank(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank model evaluation results by highest average accuracy descending."""
    return sorted(
        results,
        key=lambda res: res.get("average_accuracy", 0.0),
        reverse=True,
    )