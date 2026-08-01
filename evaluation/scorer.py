"""
Evaluation framework for comparing OCR/LLM predictions against ground truth.

This module provides data structures and evaluation logic to compare extracted
fields from handwritten or printed bills against ground truth data. Normalization
and domain-specific matching (dates, currencies, numerics, vendor names) are applied
to account for minor transcription and OCR variances.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# RapidFuzz fallback to difflib.SequenceMatcher
try:
    from rapidfuzz import fuzz

    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher

    HAS_RAPIDFUZZ = False

# Default field weightings reflecting real-world business importance
DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "vendor": 1.0,
    "bill_no": 0.8,
    "date": 1.2,
    "amount": 2.0,
    "currency": 1.0,
    "gst": 0.8,
}

# Equivalent currency representations mapped to a standardized code
CURRENCY_MAP: dict[str, str] = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupees": "INR",
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}

# Common vendor suffixes to ignore during vendor comparison
VENDOR_SUFFIXES: list[str] = [
    "pvt ltd",
    "ltd",
    "private limited",
    "limited",
    "medical store",
    "medical stores",
    "medicals",
    "medical",
    "agency",
    "agencies",
    "traders",
    "trader",
    "store",
    "stores",
    "enterprises",
    "co",
    "corp",
]

# Standard date formats for parsing handwritten/printed dates
DATE_FORMATS: list[str] = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d-%B-%Y",
    "%d %B %Y",
    "%B %d, %Y",
]


@dataclass
class FieldScore:
    """Score for a single field comparison."""

    field: str
    ground_truth: Any
    prediction: Any
    score: float  # 0.0 to 1.0
    match_type: str  # "exact", "numeric", "date", "normalized", "fuzzy", "missing", "extra", "none"


@dataclass
class SampleScore:
    """Score for a single bill sample."""

    image_name: str
    model: str
    field_scores: list[FieldScore]
    overall_accuracy: float  # Average or weighted average of all field scores

    def to_dict(self) -> dict[str, Any]:
        """Convert SampleScore object to dictionary representation."""
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
    """Evaluates bill extraction accuracy against ground truth using flexible rules."""

    def __init__(
        self,
        fuzzy_threshold: float = 0.85,
        use_weighted_scoring: bool = False,
        field_weights: dict[str, float] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        fuzzy_threshold:
            Similarity threshold for fuzzy matching (0.0 to 1.0).
            0.85 means 85% similarity counts as a match.
        use_weighted_scoring:
            Whether to compute overall accuracy using field weights.
        field_weights:
            Custom dictionary mapping field names to importance weights.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.use_weighted_scoring = use_weighted_scoring
        self.field_weights = field_weights or DEFAULT_FIELD_WEIGHTS

    def score_sample(
        self,
        image_name: str,
        model: str,
        ground_truth: dict[str, Any],
        prediction: dict[str, Any],
    ) -> SampleScore:
        """Compare prediction against ground truth for a single bill."""
        field_scores: list[FieldScore] = []

        for field, expected_value in ground_truth.items():
            predicted_value = prediction.get(field) if prediction else None
            score = self._score_field(field, expected_value, predicted_value)
            field_scores.append(score)

        if not field_scores:
            overall_accuracy = 0.0
        elif self.use_weighted_scoring:
            total_weighted_score = 0.0
            total_weight = 0.0
            for fs in field_scores:
                weight = self.field_weights.get(fs.field, 1.0)
                total_weighted_score += fs.score * weight
                total_weight += weight
            overall_accuracy = total_weighted_score / total_weight if total_weight > 0 else 0.0
        else:
            overall_accuracy = sum(fs.score for fs in field_scores) / len(field_scores)

        return SampleScore(
            image_name=image_name,
            model=model,
            field_scores=field_scores,
            overall_accuracy=overall_accuracy,
        )

    def _score_field(
        self, field: str, ground_truth: Any, prediction: Any
    ) -> FieldScore:
        """Score a single field using numeric, date, currency, and text normalization rules."""
        gt_is_empty = self._is_empty(ground_truth)
        pred_is_empty = self._is_empty(prediction)

        # Both empty
        if gt_is_empty and pred_is_empty:
            return FieldScore(field, ground_truth, prediction, 1.0, "exact")

        # Missing or extra
        if gt_is_empty and not pred_is_empty:
            return FieldScore(field, ground_truth, prediction, 0.0, "extra")
        if not gt_is_empty and pred_is_empty:
            return FieldScore(field, ground_truth, prediction, 0.0, "missing")

        # 1. Exact string match
        if str(ground_truth).strip() == str(prediction).strip():
            return FieldScore(field, ground_truth, prediction, 1.0, "exact")

        # 2. Numeric comparison (amount, gst, or general numeric fields)
        numeric_match = self._compare_numeric(ground_truth, prediction)
        if numeric_match is not None:
            score, match_type = numeric_match
            return FieldScore(field, ground_truth, prediction, score, match_type)

        # 3. Date comparison
        date_match = self._compare_date(ground_truth, prediction)
        if date_match is not None:
            score, match_type = date_match
            return FieldScore(field, ground_truth, prediction, score, match_type)

        # 4. Currency comparison
        if field.lower() == "currency":
            curr_gt = self._normalize_currency(str(ground_truth))
            curr_pred = self._normalize_currency(str(prediction))
            if curr_gt and curr_pred and curr_gt == curr_pred:
                return FieldScore(field, ground_truth, prediction, 1.0, "normalized")

        # 5. Text normalization and vendor matching
        text_match = self._compare_text(field, str(ground_truth), str(prediction))
        if text_match is not None:
            score, match_type = text_match
            return FieldScore(field, ground_truth, prediction, score, match_type)

        # 6. Fuzzy comparison fallback
        norm_gt = self._normalize_text(str(ground_truth))
        norm_pred = self._normalize_text(str(prediction))
        similarity = self._fuzzy_match(norm_gt, norm_pred)

        if similarity >= self.fuzzy_threshold:
            return FieldScore(field, ground_truth, prediction, similarity, "fuzzy")

        return FieldScore(field, ground_truth, prediction, 0.0, "none")

    # Private Helper Methods

    @staticmethod
    def _is_empty(val: Any) -> bool:
        """Check if value represents an empty/missing response."""
        if val is None:
            return True
        s = str(val).strip().lower()
        return s in ("", "none", "n/a", "na", "null", "undefined")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by stripping punctuation, extra spacing, and casing."""
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)  # Replace non-alphanumeric chars with space
        return re.sub(r"\s+", " ", text).strip()  # Normalize whitespace

    @staticmethod
    def _strip_vendor_suffixes(vendor_str: str) -> str:
        """Remove common business suffixes from vendor names."""
        norm = BillEvaluator._normalize_text(vendor_str)
        for suffix in VENDOR_SUFFIXES:
            pattern = r"\b" + re.escape(suffix) + r"\b"
            norm = re.sub(pattern, "", norm)
        return re.sub(r"\s+", " ", norm).strip()

    @staticmethod
    def _normalize_currency(val: str) -> str:
        """Map common currency symbols and strings to standardized currency codes."""
        cleaned = val.strip().lower()
        return CURRENCY_MAP.get(cleaned, cleaned.upper())

    @staticmethod
    def _normalize_date(date_str: str) -> str | None:
        """Attempt to parse date_str across formats and return standardized YYYY-MM-DD string."""
        s = str(date_str).strip()

        # Simple delimiter substitution for cases like 1-8-26 or 01.08.2026
        s_clean = re.sub(r"[\./]", "-", s)

        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
            try:
                dt = datetime.strptime(s_clean, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass

        return None

    @staticmethod
    def _compare_numeric(gt: Any, pred: Any) -> tuple[float, str] | None:
        """Compare two values as numbers, handling floating-point precision."""
        # Clean currency symbols and commas prior to numeric conversion
        gt_str = re.sub(r"[^\d.-]", "", str(gt))
        pred_str = re.sub(r"[^\d.-]", "", str(pred))

        if not gt_str or not pred_str:
            return None

        try:
            gt_num = float(gt_str)
            pred_num = float(pred_str)

            if abs(gt_num - pred_num) < 1e-4:
                return 1.0, "numeric"
        except ValueError:
            return None

        return None

    def _compare_date(self, gt: Any, pred: Any) -> tuple[float, str] | None:
        """Compare dates by normalizing them to standard ISO format."""
        gt_date = self._normalize_date(str(gt))
        pred_date = self._normalize_date(str(pred))

        if gt_date and pred_date and gt_date == pred_date:
            return 1.0, "date"

        return None

    def _compare_text(
        self, field: str, gt: str, pred: str
    ) -> tuple[float, str] | None:
        """Compare normalized text strings and vendor name variations."""
        norm_gt = self._normalize_text(gt)
        norm_pred = self._normalize_text(pred)

        if norm_gt and norm_pred and norm_gt == norm_pred:
            return 1.0, "normalized"

        if field.lower() == "vendor":
            clean_gt = self._strip_vendor_suffixes(gt)
            clean_pred = self._strip_vendor_suffixes(pred)
            if clean_gt and clean_pred and clean_gt == clean_pred:
                return 1.0, "normalized"

        return None

    @staticmethod
    def _fuzzy_match(str1: str, str2: str) -> float:
        """
        Compute similarity score (0.0 to 1.0) using RapidFuzz if available,
        falling back to difflib.SequenceMatcher.
        """
        if not str1 or not str2:
            return 0.0

        if HAS_RAPIDFUZZ:
            return fuzz.ratio(str1, str2) / 100.0

        return SequenceMatcher(None, str1, str2).ratio()