"""API cost estimation utilities for bill extraction models."""

from __future__ import annotations

# Estimated cost per bill (USD) per model type
MODEL_COST_PER_BILL: dict[str, float] = {
    "GeminiBillExtractor": 0.00015,
    "ClaudeBillExtractor": 0.0025,
    "OpenAIBillExtractor": 0.0015,
    "QwenVLExtractor": 0.0002,
    "LlamaVisionExtractor": 0.0002,
}

DEFAULT_COST_PER_BILL: float = 0.0005


def calculate_cost_per_bill(model_name: str) -> float:
    """Get estimated API cost per single bill extraction for a given model."""
    return MODEL_COST_PER_BILL.get(model_name, DEFAULT_COST_PER_BILL)


def calculate_cost_per_100_bills(model_name: str) -> float:
    """Get estimated API cost for extracting 100 bills."""
    return calculate_cost_per_bill(model_name) * 100.0


def estimate_dataset_cost(model_name: str, sample_count: int) -> float:
    """Calculate total estimated cost for a dataset given a sample count."""
    return calculate_cost_per_bill(model_name) * sample_count