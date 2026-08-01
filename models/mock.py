"""Mock bill extractor for testing without API calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.gemini import BillExtraction


def _mock_responses() -> dict[str, dict[str, Any]]:
    """Factory for mock responses."""
    return {
        "bill1.jpg": {
            "vendor": "MURUGESH TRADERS",
            "bill_no": "835",
            "date": "2023-03-15",
            "amount": "49295.36",
            "currency": "INR",
            "gst": "28%",
        },
        "bill2.jpg": {
            "vendor": "OM SHIV MEDICOS",
            "bill_no": "3802",
            "date": "2020-11-2",
            "amount": "2059.48",
            "currency": "INR",
            "gst": None,
        },
    }


@dataclass(frozen=True)
class MockBillExtractor:
    """Mock extractor that returns predetermined responses based on image name."""

    model: str = "mock"
    mock_responses: dict[str, dict[str, Any]] = field(default_factory=_mock_responses)

    def extract_bill(self, image_path: Path | str) -> dict[str, Any]:
        """Return mock response based on image filename."""
        image_name = Path(image_path).name
        if image_name not in self.mock_responses:
            raise ValueError(f"No mock response for: {image_name}")
        return self.mock_responses[image_name]
