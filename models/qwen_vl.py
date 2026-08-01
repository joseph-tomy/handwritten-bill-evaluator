"""Qwen 2.5 VL wrapper for handwritten bill extraction using OpenRouter free tier."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class BillExtraction(BaseModel):
    """Structured bill fields returned by Qwen VL."""

    vendor: str | None = Field(default=None, description="Vendor or merchant name")
    bill_no: str | None = Field(default=None, description="Invoice or bill number")
    date: str | None = Field(default=None, description="Bill date in ISO-8601 format when possible")
    amount: str | None = Field(default=None, description="Final payable amount as a string")
    currency: str | None = Field(default=None, description="Currency code such as INR")
    gst: str | None = Field(default=None, description="GST or tax rate if present")


@dataclass(frozen=True)
class QwenVLExtractor:
    """Wrapper for Qwen 2.5 VL via OpenRouter API."""

    client: OpenAI
    model: str = "qwen/qwen2.5-vl-72b-instruct"

    @classmethod
    def from_env(
        cls, 
        model: str = "qwen/qwen2.5-vl-72b-instruct", 
        timeout: float = 60.0
    ) -> "QwenVLExtractor":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        
        # OpenRouter uses the standard OpenAI-compatible client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
        )
        return cls(client=client, model=model)

    def extract_bill(self, image_path: Path | str) -> dict[str, Any]:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        mime_type = mimetypes.guess_type(image_file.name)[0] or "image/jpeg"
        image_bytes = image_file.read_bytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_image}"

        # Prompt format matching Gemini and Llama Vision for a fair evaluation
        prompt_text = (
            "Extract the handwritten bill fields as structured JSON. "
            "Return only the fields vendor, bill_no, date, amount, currency, and gst. "
            "Use null for missing values and preserve the original bill text when uncertain."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=256,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            content = response.choices[0].message.content or "{}"
            raw_json = json.loads(content)
            
            # Validate output structure against Pydantic schema
            return BillExtraction.model_validate(raw_json).model_dump()
            
        except Exception as e:
            raise RuntimeError(f"Qwen 2.5 VL extraction failed: {e}") from e