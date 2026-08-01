"""Evaluation framework for comparing predictions against ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from difflib import SequenceMatcher
import json


@dataclass
class FieldScore:
    """Score for a single field comparison."""

    field: str
    ground_truth: Any
    prediction: Any
    score: float  # 0.0 to 1.0
    match_type: str  # "exact", "fuzzy", "missing", "extra", "none"


@dataclass
class SampleScore:
    """Score for a single bill sample."""

    image_name: str
    model: str
    field_scores: list[FieldScore]
    overall_accuracy: float  # Average of all field scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_name": self.image_name,
            "model": self.model,
            "overall_accuracy": self.overall_accuracy,
            "field_scores": [
                {
                    "field": fs.field,
                    "ground_truth": fs.ground_truth,
                    "prediction": fs.prediction,
                    "score": fs.score,
                    "match_type": fs.match_type,
                }
                for fs in self.field_scores
            ],
        }


class BillEvaluator:
    """Evaluates bill extraction accuracy against ground truth."""

    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Parameters
        ----------
        fuzzy_threshold:
            Similarity threshold for fuzzy matching (0.0 to 1.0).
            0.85 means 85% similarity counts as a match.
        """
        self.fuzzy_threshold = fuzzy_threshold

    def score_sample(
        self,
        image_name: str,
        model: str,
        ground_truth: dict[str, Any],
        prediction: dict[str, Any],
    ) -> SampleScore:
        """Compare prediction against ground truth for a single bill."""

        field_scores = []
        for field, expected_value in ground_truth.items():
            predicted_value = prediction.get(field)
            score = self._score_field(field, expected_value, predicted_value)
            field_scores.append(score)

        overall_accuracy = (
            sum(fs.score for fs in field_scores) / len(field_scores)
            if field_scores
            else 0.0
        )

        return SampleScore(
            image_name=image_name,
            model=model,
            field_scores=field_scores,
            overall_accuracy=overall_accuracy,
        )

    def _score_field(
        self, field: str, ground_truth: Any, prediction: Any
    ) -> FieldScore:
        """Score a single field. Returns FieldScore with match_type and score."""

        # Both None
        if ground_truth is None and prediction is None:
            return FieldScore(
                field=field,
                ground_truth=ground_truth,
                prediction=prediction,
                score=1.0,
                match_type="exact",
            )

        # Ground truth is None, but prediction exists
        if ground_truth is None and prediction is not None:
            return FieldScore(
                field=field,
                ground_truth=ground_truth,
                prediction=prediction,
                score=0.0,
                match_type="extra",
            )

        # Prediction is None, but ground truth exists
        if ground_truth is not None and prediction is None:
            return FieldScore(
                field=field,
                ground_truth=ground_truth,
                prediction=prediction,
                score=0.0,
                match_type="missing",
            )

        # Both have values — compare
        gt_str = str(ground_truth).strip().lower()
        pred_str = str(prediction).strip().lower()

        if gt_str == pred_str:
            return FieldScore(
                field=field,
                ground_truth=ground_truth,
                prediction=prediction,
                score=1.0,
                match_type="exact",
            )

        # Try fuzzy match
        similarity = self._fuzzy_match(gt_str, pred_str)
        if similarity >= self.fuzzy_threshold:
            return FieldScore(
                field=field,
                ground_truth=ground_truth,
                prediction=prediction,
                score=similarity,
                match_type="fuzzy",
            )

        # No match
        return FieldScore(
            field=field,
            ground_truth=ground_truth,
            prediction=prediction,
            score=0.0,
            match_type="none",
        )

    @staticmethod
    def _fuzzy_match(str1: str, str2: str) -> float:
        """Compute string similarity (0.0 to 1.0)."""
        return SequenceMatcher(None, str1, str2).ratio()
