"""Production pipeline for bill extraction and Zoho Books integration."""

from pathlib import Path
import json  # NEW: Read winner.json automatically

# NEW: Remove config.py dependency
# from config import WINNING_MODEL

from models.gemini import GeminiBillExtractor
from models.qwen_vl import QwenVLExtractor          # NEW
from models.llama_vision import LlamaVisionExtractor  # NEW

from zoho.books import (
    ZohoBooksClient,
    ZohoValidationError,
    ZohoAPIError,
    ZohoAuthError,
)


class ProductionPipeline:

    def __init__(self):

        # =====================================================
        # NEW: Read winning model from winner.json
        # =====================================================
        try:
            with open("winner.json", "r") as f:
                winner = json.load(f)

            winning_model = winner["winning_model"]

        except FileNotFoundError:
            # Fallback if winner.json doesn't exist
            print("winner.json not found. Using Gemini by default.")
            winning_model = "GeminiBillExtractor"

        # =====================================================
        # CHANGED: Select extractor automatically
        # =====================================================
        if winning_model == "GeminiBillExtractor":
            print("Using Gemini")
            self.extractor = GeminiBillExtractor.from_env()

        elif winning_model == "QwenVLExtractor":
            print("Using Qwen 2.5 VL")
            self.extractor = QwenVLExtractor.from_env()

        elif winning_model == "LlamaVisionExtractor":
            print("Using Llama 3.2 Vision")
            self.extractor = LlamaVisionExtractor.from_env()

        else:
            raise ValueError(
                f"Unsupported winning model: {winning_model}"
            )

        # Existing code (unchanged)
        self.zoho = ZohoBooksClient.from_env()

    # =====================================================
    # Extract bill
    # =====================================================
    def extract(self, image_path):

        image = Path(image_path)

        if not image.exists():
            raise FileNotFoundError(image)

        prediction = self.extractor.extract_bill(image)

        return prediction

    # =====================================================
    # Create Zoho Expense
    # =====================================================
    def create_expense(self, prediction):

        return self.zoho.create_expense(prediction)