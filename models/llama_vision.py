"""Llama 3.2 Vision wrapper for handwritten bill extraction using OpenRouter."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
import io

from PIL import Image, UnidentifiedImageError, ImageFile

from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class BillExtraction(BaseModel):
    """Structured bill fields returned by Llama Vision."""

    vendor: str | None = Field(default=None, description="Vendor or merchant name")
    bill_no: str | None = Field(default=None, description="Invoice or bill number")
    date: str | None = Field(default=None, description="Bill date in ISO-8601 format when possible")
    amount: str | None = Field(default=None, description="Final payable amount as a string")
    currency: str | None = Field(default=None, description="Currency code such as INR")
    gst: str | None = Field(default=None, description="GST or tax rate if present")


@dataclass(frozen=True)
class LlamaVisionExtractor:
    """Wrapper for Meta Llama Vision models via OpenRouter API."""

    client: OpenAI
    model: str = "meta-llama/llama-3.2-11b-vision-instruct:free"

    @classmethod
    def from_env(
        cls, 
        model: str = "meta-llama/llama-3.2-11b-vision-instruct:free", 
        timeout: float = 60.0
    ) -> "LlamaVisionExtractor":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")
        
        # OpenRouter client with identification headers to prevent routing 404s
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Bill Extraction Benchmark",
            },
        )
        return cls(client=client, model=model)

    def extract_bill(self, image_path: Path | str) -> dict[str, Any]:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")

        mime_type = mimetypes.guess_type(image_file.name)[0] or "image/jpeg"
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(image_file) as img:
                rgb = img.convert("RGB")
                buf = io.BytesIO()
                rgb.save(buf, format="JPEG")
                image_bytes = buf.getvalue()
        except Exception:
            image_bytes = image_file.read_bytes()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_image}"

        prompt_text = (
            "Extract the handwritten bill fields as structured JSON. "
            "Return only the fields vendor, bill_no, date, amount, currency, and gst. "
            "Use null for missing values and preserve the original bill text when uncertain."
        )

        # List of Llama vision endpoints to attempt if the primary is temporarily offline
        candidate_models = [
            self.model,
            "meta-llama/llama-3.2-11b-vision-instruct",
            "meta-llama/llama-3.2-90b-vision-instruct",
            "meta-llama/llama-4-scout",
        ]

        last_error = None
        for current_model in candidate_models:
            try:
                response = self.client.chat.completions.create(
                    model=current_model,
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
                try:
                    raw_json = json.loads(content)
                except Exception:
                    decoder = json.JSONDecoder()
                    try:
                        raw_json, _ = decoder.raw_decode(content)
                    except Exception:
                        # Heuristic fallback: extract simple "key": "value" pairs.
                        import re

                        pairs = re.findall(r'"([a-zA-Z0-9_]+)"\s*:\s*"([^\"]*)"', content)
                        if pairs:
                            raw_json = {k: v for k, v in pairs}
                        else:
                            raise RuntimeError(f"Failed to parse JSON response; content={content!r}")

                # Normalize types to strings for Pydantic validation
                def _normalize_value(v: Any) -> Any:
                    if v is None:
                        return None
                    if isinstance(v, dict):
                        parts = [f"{k}:{v[k]}" for k in sorted(v.keys()) if v[k] is not None]
                        return ", ".join(parts) if parts else None
                    if isinstance(v, (int, float)):
                        return str(v)
                    if isinstance(v, list):
                        return ", ".join(str(x) for x in v)
                    return str(v)

                normalized = {
                    key: _normalize_value(raw_json.get(key))
                    for key in ["vendor", "bill_no", "date", "amount", "currency", "gst"]
                }

                return BillExtraction.model_validate(normalized).model_dump()

            except Exception as e:
                last_error = e
                # Fall through to try the next candidate model
                continue

        raise RuntimeError(f"Llama Vision extraction failed across all endpoints: {last_error}") from last_error