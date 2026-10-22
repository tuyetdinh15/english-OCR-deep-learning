"""Export OCR splits: ``python -m dataset.data_preprocessing.prepare_data``."""

import argparse
import json
from pathlib import Path

from config import IMAGE_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from .data_preprocessing import directory_to_dataframe, split_dataframe


def prepare_dataset(
    dataset_dir: str | Path = RAW_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    test_size: float = 0.1,
    validation_size: float = 0.1,
    random_state: int = 41,
) -> dict:
    """Validate images, split samples, and save manifests without loading TensorFlow.

    CSVs keep paths to original images. RGB conversion, resize, and train-only
    augmentation are performed by create_generators when training.
    """
    dataframe = directory_to_dataframe(dataset_dir)
    train, validation, test = split_dataframe(dataframe, test_size, validation_size, random_state)
    splits = {"train": train, "validation": validation, "test": test}
    class_indices = {label: index for index, label in enumerate(sorted(dataframe["label"].unique()))}
    summary = {
        "dataset_dir": str(Path(dataset_dir).expanduser().resolve()),
        "total_samples": len(dataframe),
        "num_classes": len(class_indices),
        "random_state": random_state,
        "split_ratios": {"train": 1 - test_size - validation_size,
                         "validation": validation_size, "test": test_size},
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "class_counts": {
            name: {label: int(count) for label, count in frame["label"].value_counts().sort_index().items()}
            for name, frame in splits.items()
        },
        "generator": {"image_size": list(IMAGE_SIZE), "color_mode": "rgb", "dtype": "int32",
                      "train_brightness_range": [0.0, 1.0], "rescale": None},
    }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8")
    for filename, content in (("class_indices.json", class_indices), ("summary.json", summary)):
        (destination / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare lowercase OCR data (train/test/validation = 80/10/10).")
    parser.add_argument("--dataset", type=Path, default=RAW_DATA_DIR, help="Raw class folders (default: dataset/raw)")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR, help="CSV/JSON destination (default: dataset/processed)")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test fraction of the whole dataset")
    parser.add_argument("--validation-size", type=float, default=0.1, help="Validation fraction of the whole dataset")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    try:
        summary = prepare_dataset(args.dataset, args.output, args.test_size, args.validation_size, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Prepared {summary['total_samples']} images across {summary['num_classes']} classes.")
    for name, count in summary["split_sizes"].items():
        print(f"{name}: {count} ({count / summary['total_samples']:.2%})")


if __name__ == "__main__":
    main()
