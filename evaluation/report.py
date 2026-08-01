"""Report generation and export utilities for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.metrics import calculate_model_rank


def generate_text_report(results: dict[str, Any]) -> str:
    """Generate a detailed formatted text report from evaluation results."""
    lines = [
        "=" * 80,
        "BILL EXTRACTION EVALUATION REPORT",
        "=" * 80,
        "",
    ]

    models_list = results.get("models", [])

    for model_result in models_list:
        model_name = model_result.get("model", "Unknown")
        sample_count = model_result.get("sample_count", 0)
        avg_acc = model_result.get("average_accuracy", 0.0)
        high_acc = model_result.get("highest_accuracy", 0.0)
        low_acc = model_result.get("lowest_accuracy", 0.0)
        avg_time = model_result.get("average_extraction_time", 0.0)
        field_acc = model_result.get("field_accuracy", {})
        cost_bill = model_result.get("cost_per_bill", 0.0)
        cost_100 = model_result.get("cost_per_100_bills", 0.0)
        failed = model_result.get("failed_samples", [])

        lines.append(f"📊 Model: {model_name}")
        lines.append(f"   Samples: {sample_count}")
        lines.append(f"   Average Accuracy: {avg_acc:.2%}")
        lines.append(f"   Highest Accuracy: {high_acc:.2%}")
        lines.append(f"   Lowest Accuracy: {low_acc:.2%}")
        lines.append(f"   Average Extraction Time: {avg_time:.2f}s")
        lines.append("   Field Accuracy:")
        lines.append(f"      • Vendor Accuracy:   {field_acc.get('vendor', 0.0):.2%}")
        lines.append(f"      • Bill Number Accuracy: {field_acc.get('bill_no', 0.0):.2%}")
        lines.append(f"      • Date Accuracy:      {field_acc.get('date', 0.0):.2%}")
        lines.append(f"      • Amount Accuracy:    {field_acc.get('amount', 0.0):.2%}")
        lines.append(f"      • Currency Accuracy:  {field_acc.get('currency', 0.0):.2%}")
        lines.append(f"      • GST Accuracy:       {field_acc.get('gst', 0.0):.2%}")
        lines.append(f"   Estimated Cost/Bill: ${cost_bill:.5f}")
        lines.append(f"   Estimated Cost/100 Bills: ${cost_100:.3f}")
        
        if failed:
            lines.append(f"   Failed Samples ({len(failed)}):")
            for f in failed:
                lines.append(f"      ❌ {f['image_name']}: {f['error']}")
        else:
            lines.append("   Failed Samples: None")

        lines.append("-" * 80)

    # Ranking summary table
    lines.append("\n" + "=" * 80)
    lines.append("MODEL RANKING SUMMARY (Highest Average Accuracy)")
    lines.append("=" * 80)
    lines.append(
        f"{'Rank':<6} {'Model':<24} {'Accuracy':<12} {'Avg Time':<12} {'Cost/100':<10}"
    )
    lines.append("-" * 64)

    ranked_models = calculate_model_rank(models_list)
    for idx, model_res in enumerate(ranked_ranked := ranked_models, start=1):
        name = model_res.get("model", "Unknown")
        acc = model_res.get("average_accuracy", 0.0)
        t = model_res.get("average_extraction_time", 0.0)
        cost = model_res.get("cost_per_100_bills", 0.0)
        lines.append(f"{idx:<6} {name:<24} {acc:>10.2%} {t:>10.2f}s ${cost:>8.3f}")

    return "\n".join(lines)


def generate_json_report(results: dict[str, Any]) -> str:
    """Generate JSON structured string from evaluation results."""
    return json.dumps(results, indent=2)


def generate_report(results: dict[str, Any]) -> str:
    """Backwards-compatible report generation call (returns plain text report)."""
    return generate_text_report(results)


def save_reports(results: dict[str, Any], output_dir: str | Path = "reports") -> None:
    """
    Save evaluation results as text and JSON files into specified directory.

    Creates output directory automatically if it does not exist.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    text_content = generate_text_report(results)
    json_content = generate_json_report(results)

    (out_path / "evaluation_report.txt").write_text(text_content, encoding="utf-8")
    (out_path / "evaluation_report.json").write_text(json_content, encoding="utf-8")