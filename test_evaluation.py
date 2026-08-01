"""Quick test of evaluation framework with Gemini API."""

from pathlib import Path
from dataset.loader import load_dataset
from models.gemini import GeminiBillExtractor
from evaluation.evaluate import evaluate_model, generate_report


def main():
    print("📂 Loading dataset...")
    samples = load_dataset()
    print(f"✅ Loaded {len(samples)} samples")

    # Map ground_truth to image_name for evaluation
    dataset_with_images = []
    for sample in samples:
        image_path = sample["image_path"]
        image_name = Path(image_path).name
        dataset_with_images.append({
            "image_name": image_name,
            "image_path": image_path,
            "ground_truth": sample["ground_truth"],
        })

    print("\n🧪 Running evaluation with GeminiBillExtractor...")
    gemini_extractor = GeminiBillExtractor.from_env()
    results = evaluate_model("GeminiBillExtractor", gemini_extractor, dataset_with_images)

    print("\n📋 Evaluation Results:")
    print(generate_report({"models": [results]}))


if __name__ == "__main__":
    main()
