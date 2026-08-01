"""Gemini wrapper for handwritten bill extraction."""

from __future__ import annotations

from dataclasses import dataclass
import mimetypes
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class BillExtraction(BaseModel):
    """Structured bill fields returned by Gemini."""

    vendor: str | None = Field(default=None, description="Vendor or merchant name")
    bill_no: str | None = Field(default=None, description="Invoice or bill number")
    date: str | None = Field(default=None, description="Bill date in ISO-8601 format when possible")
    amount: str | None = Field(default=None, description="Final payable amount as a string")
    currency: str | None = Field(default=None, description="Currency code such as INR")
    gst: str | None = Field(default=None, description="GST or tax rate if present")


@dataclass(frozen=True)
class GeminiBillExtractor:
    """Thin wrapper around the Google Gen AI SDK for bill OCR extraction."""

    client: genai.Client
    model: str = "gemini-3.6-flash"

    @classmethod
    def from_env(cls, model: str = "gemini-3.6-flash") -> "GeminiBillExtractor":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        return cls(client=genai.Client(api_key=api_key), model=model)

    def extract_bill(self, image_path: Path | str) -> dict[str, Any]:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        mime_type = mimetypes.guess_type(image_file.name)[0] or "image/jpeg"
        image_bytes = image_file.read_bytes()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                "Extract the handwritten bill fields as structured JSON. "
                "Return only the fields vendor, bill_no, date, amount, currency, and gst. "
                "Use null for missing values and preserve the original bill text when uncertain.",
                image_part,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BillExtraction,
            ),
        )

        if response.parsed is not None:
            parsed = response.parsed
            if isinstance(parsed, BaseModel):
                return parsed.model_dump()
            return dict(parsed)

        return BillExtraction.model_validate_json(response.text or "{}").model_dump()