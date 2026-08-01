from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.loader import load_dataset
from models.gemini import GeminiBillExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini extraction on bill images.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Path to the dataset directory. Defaults to dataset/ next to the script.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional path to a single bill image. If omitted, the first dataset sample is used.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser() if args.dataset_dir else None
    samples = load_dataset(dataset_dir)
    if not samples:
        raise SystemExit("No dataset samples were found.")

    image_path = Path(args.image).expanduser() if args.image else Path(samples[0]["image_path"])
    extractor = GeminiBillExtractor.from_env()
    prediction = extractor.extract_bill(image_path)

    print(json.dumps({"image_path": str(image_path), "prediction": prediction}, indent=2))


if __name__ == "__main__":
    main()
