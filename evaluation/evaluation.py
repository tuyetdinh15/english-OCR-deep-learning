"""Aggregate per-model test reports into a comparison table."""

import json
from pathlib import Path

MODEL_REPORTS = {
    "Custom CNN": "CustomCNN_model_test_report.json",
    "EfficientNetB7": "EfficientNetB7_model_test_report.json",
    "MobileNetV2": "MobileNetV2_model_test_report.json",
    "DenseNet121": "DenseNet121_model_test_report.json",
}


def evaluate_predictions(model, generator, mapping_inverse):
    from sklearn.metrics import classification_report

    predictions = model.predict(generator, verbose=0)
    predicted_labels = [mapping_inverse[int(prediction.argmax())] for prediction in predictions]
    actual_labels = [mapping_inverse[int(label)] for label in generator.classes]
    return classification_report(actual_labels, predicted_labels, output_dict=True)


def compare_results(results):
    import pandas as pd

    return pd.DataFrame(results)


def aggregate_reports(reports_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Load all saved per-model test reports and save comparison CSV and JSON."""
    reports = {}
    missing = []
    for model_name, filename in MODEL_REPORTS.items():
        report_path = reports_dir / filename
        if not report_path.is_file():
            missing.append(filename)
            continue
        with report_path.open(encoding="utf-8") as file:
            reports[model_name] = json.load(file)

    if missing:
        missing_list = "\n".join(f"- {filename}" for filename in missing)
        raise FileNotFoundError(
            "Chưa có báo cáo cho tất cả model. Hãy chạy các file test còn thiếu trong tests/:\n"
            f"{missing_list}"
        )

    rows = []
    for model_name, report in reports.items():
        macro = report["macro avg"]
        weighted = report["weighted avg"]
        rows.append({
            "model": model_name,
            "accuracy": report["accuracy"],
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1-score"],
            "weighted_precision": weighted["precision"],
            "weighted_recall": weighted["recall"],
            "weighted_f1": weighted["f1-score"],
            "test_samples": int(weighted["support"]),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "model_comparison.csv"
    reports_path = output_dir / "all_model_reports.json"
    comparison = compare_results(rows).sort_values("weighted_f1", ascending=False)
    comparison.to_csv(comparison_path, index=False)
    reports_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return comparison_path, reports_path


def main() -> None:
    reports_dir = Path(__file__).resolve().parent / "output"
    comparison_path, reports_path = aggregate_reports(reports_dir, reports_dir)
    import pandas as pd

    comparison = pd.read_csv(comparison_path)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nĐã lưu bảng so sánh: {comparison_path}")
    print(f"Đã lưu báo cáo chi tiết: {reports_path}")


if __name__ == "__main__":
    main()
