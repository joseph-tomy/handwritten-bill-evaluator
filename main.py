"""Evaluation framework runner for Gemini, Qwen 2.5 VL, and Llama 3.2 Vision APIs."""

from pathlib import Path
from datetime import datetime
import json

from dataset.loader import load_dataset
from evaluation.evaluate import evaluate_model
from evaluation.report import generate_text_report, save_reports
from models.gemini import GeminiBillExtractor
from models.llama_vision import LlamaVisionExtractor
from models.qwen_vl import QwenVLExtractor


def main() -> None:
    print("📂 Loading dataset...")
    samples = load_dataset()
    print(f"✅ Loaded {len(samples)} samples")

    # Map ground_truth to image_name for evaluation
    dataset_with_images = []

    for sample in samples:
        image_path = sample["image_path"]
        image_name = Path(image_path).name

        dataset_with_images.append(
            {
                "image_name": image_name,
                "image_path": image_path,
                "ground_truth": sample["ground_truth"],
            }
        )

    results_list = []

    # ==========================================================
    # 1. Gemini Evaluation
    # ==========================================================

    print("\n🧪 Running evaluation with GeminiBillExtractor...")

    try:
        gemini_extractor = GeminiBillExtractor.from_env()

        gemini_results = evaluate_model(
            "GeminiBillExtractor",
            gemini_extractor,
            dataset_with_images,
        )

        results_list.append(gemini_results)

    except Exception as e:
        print(f"⚠️ Gemini evaluation failed: {e}")

    # ==========================================================
    # 2. Qwen Evaluation
    # ==========================================================

    print("\n🧪 Running evaluation with QwenVLExtractor...")

    try:
        qwen_extractor = QwenVLExtractor.from_env()

        qwen_results = evaluate_model(
            "QwenVLExtractor",
            qwen_extractor,
            dataset_with_images,
        )

        results_list.append(qwen_results)

    except Exception as e:
        print(f"⚠️ Qwen 2.5 VL evaluation failed: {e}")

    # ==========================================================
    # 3. Llama Evaluation
    # ==========================================================

    print("\n🧪 Running evaluation with LlamaVisionExtractor...")

    try:
        llama_extractor = LlamaVisionExtractor.from_env()

        llama_results = evaluate_model(
            "LlamaVisionExtractor",
            llama_extractor,
            dataset_with_images,
        )

        results_list.append(llama_results)

    except Exception as e:
        print(f"⚠️ Llama 3.2 Vision evaluation failed: {e}")

    # ==========================================================
    # Check if at least one model succeeded
    # ==========================================================

    if not results_list:
        print("\n❌ No model completed successfully.")
        return

    full_results = {"models": results_list}

    # ==========================================================
    # 4. Print Report
    # ==========================================================

    print("\n📋 Evaluation Results:")

    print(generate_text_report(full_results))

    # ==========================================================
    # 5. Save Reports
    # ==========================================================

    save_reports(full_results)

    print(
        "\n💾 Evaluation reports saved to "
        "reports/evaluation_report.txt "
        "and reports/evaluation_report.json"
    )

    # ==========================================================
    # 6. Find Winning Model (NEW)
    # ==========================================================

    winner = max(
        full_results["models"],
        key=lambda model: model["average_accuracy"],
    )

    # ==========================================================
    # 7. Create winner.json (NEW)
    # ==========================================================

    winner_data = {
        "generated_at": datetime.now().isoformat(),
        "winning_model": winner["model"],
        "winning_accuracy": winner["average_accuracy"],
        "evaluated_models": [
            {
                "model": model["model"],
                "accuracy": model["average_accuracy"],
                "samples": model["sample_count"],
            }
            for model in full_results["models"]
        ],
    }

    with open("winner.json", "w") as file:
        json.dump(winner_data, file, indent=4)

    # ==========================================================
    # 8. Display Winner (NEW)
    # ==========================================================

    print("\n" + "=" * 50)
    print("🏆 WINNING MODEL")
    print("=" * 50)

    print(f"Model      : {winner['model']}")
    print(f"Accuracy   : {winner['average_accuracy']:.2%}")

    print("\n✅ winner.json created successfully.")


if __name__ == "__main__":
    main()
    