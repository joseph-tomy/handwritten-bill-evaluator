"""Quick test of evaluation framework with Gemini, Qwen 2.5 VL, and Llama 3.2 Vision APIs."""

from pathlib import Path

from dataset.loader import load_dataset
from evaluation.evaluate import evaluate_model
from evaluation.report import generate_text_report, save_reports
from models.gemini import GeminiBillExtractor
from models.llama_vision import LlamaVisionExtractor
from models.qwen_vl import QwenVLExtractor


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

    results_list = []

    # Test Gemini Extractor
    
    print("\n🧪 Running evaluation with GeminiBillExtractor...")
    try:
        gemini_extractor = GeminiBillExtractor.from_env()
        gemini_results = evaluate_model("GeminiBillExtractor", gemini_extractor, dataset_with_images)
        results_list.append(gemini_results)
    except Exception as e:
        print(f"⚠️ Gemini test failed: {e}")

    # Test Qwen 2.5 VL Extractor
    print("\n🧪 Running evaluation with QwenVLExtractor...")
    try:
        qwen_extractor = QwenVLExtractor.from_env()
        qwen_results = evaluate_model("QwenVLExtractor", qwen_extractor, dataset_with_images)
        results_list.append(qwen_results)
    except Exception as e:
        print(f"⚠️ Qwen 2.5 VL test failed: {e}")
    

    # Test Llama 3.2 Vision Extractor
    print("\n🧪 Running evaluation with LlamaVisionExtractor...")
    try:
        llama_extractor = LlamaVisionExtractor.from_env()
        llama_results = evaluate_model(
            "LlamaVisionExtractor", llama_extractor, dataset_with_images
        )
        results_list.append(llama_results)
    except Exception as e:
        print(f"⚠️ Llama 3.2 Vision test failed: {e}")

    full_results = {"models": results_list}

    # Display evaluation report in console
    print("\n📋 Evaluation Results:")
    print(generate_text_report(full_results))

    # Save reports automatically to reports/ (txt and json)
    save_reports(full_results)
    print("\n💾 Reports saved to reports/evaluation_report.txt and reports/evaluation_report.json")


if __name__ == "__main__":
    main()