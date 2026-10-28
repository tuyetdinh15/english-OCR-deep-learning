"""Shared helpers for evaluating saved OCR models on the held-out test split."""

import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import BATCH_SIZE, IMAGE_SIZE, MODEL_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


class _CompatibleGlorotUniform(tf.keras.initializers.GlorotUniform):
    """Load GlorotUniform configs saved by Keras versions with axis fields."""

    def __init__(self, seed=None, input_axes=None, output_axes=None, **kwargs):
        if input_axes is not None or output_axes is not None:
            raise ValueError(
                "Chỉ hỗ trợ input_axes/output_axes=None trong cấu hình model đã lưu"
            )
        super().__init__(seed=seed, **kwargs)


def _load_saved_model(model_path: Path):
    """Load a .keras archive, adapting legacy initializer config in memory only."""
    with ZipFile(model_path) as archive:
        model_config = json.loads(archive.read("config.json"))

    adapted = False

    def adapt_initializer_config(value):
        nonlocal adapted
        if isinstance(value, dict):
            initializer_config = value.get("config")
            if (
                value.get("class_name") == "GlorotUniform"
                and isinstance(initializer_config, dict)
                and ("input_axes" in initializer_config or "output_axes" in initializer_config)
            ):
                if initializer_config.get("input_axes") is not None or initializer_config.get("output_axes") is not None:
                    raise ValueError("Model có input_axes/output_axes khác None, không thể tự chuyển đổi an toàn")
                initializer_config.pop("input_axes", None)
                initializer_config.pop("output_axes", None)
                value["registered_name"] = "ocr_compat>GlorotUniform"
                adapted = True
            for child in value.values():
                adapt_initializer_config(child)
        elif isinstance(value, list):
            for child in value:
                adapt_initializer_config(child)

    adapt_initializer_config(model_config)
    if not adapted:
        return load_model(model_path, compile=False)

    model = tf.keras.utils.deserialize_keras_object(
        model_config,
        custom_objects={"ocr_compat>GlorotUniform": _CompatibleGlorotUniform},
    )
    model.load_weights(model_path)
    return model


def _resolve_test_image_path(value: str) -> str:
    """Resolve manifest paths recorded on Windows or inside the Docker container."""
    original_path = Path(value)
    if original_path.is_file():
        return str(original_path.resolve())

    parts = PurePosixPath(value.replace("\\", "/")).parts
    raw_positions = [index for index, part in enumerate(parts) if part.lower() == "raw"]
    for index in reversed(raw_positions):
        candidate = RAW_DATA_DIR.joinpath(*parts[index + 1:])
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"Không tìm thấy ảnh test: {value}. Kiểm tra dataset/raw và đường dẫn trong test.csv."
    )


def run_model_test(model_filename: str, display_name: str) -> None:
    """Print and save the classification report for one saved model."""
    model_path = MODEL_DIR / model_filename
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model đã lưu: {model_path}")

    test_csv = PROCESSED_DATA_DIR / "test.csv"
    class_indices_path = PROCESSED_DATA_DIR / "class_indices.json"
    for required_path in (test_csv, class_indices_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu đánh giá: {required_path}")

    with class_indices_path.open(encoding="utf-8") as file:
        class_indices = json.load(file)
    class_names = [
        name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])
    ]

    test_frame = pd.read_csv(test_csv)
    if not {"image", "label"}.issubset(test_frame.columns):
        raise ValueError(f"{test_csv} cần có hai cột image và label")
    test_frame["image"] = test_frame["image"].map(_resolve_test_image_path)
    unknown_labels = set(test_frame["label"]) - set(class_names)
    if unknown_labels:
        raise ValueError(f"Nhãn trong test.csv không có trong class_indices.json: {unknown_labels}")

    generator = ImageDataGenerator(dtype="int32", fill_mode="nearest").flow_from_dataframe(
        test_frame,
        x_col="image",
        y_col="label",
        classes=class_names,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=min(BATCH_SIZE, 8),
        shuffle=False,
    )
    if generator.samples == 0:
        raise ValueError(f"Không có ảnh test trong {test_csv}")

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Đang đánh giá {display_name} trên {generator.samples} ảnh test...", flush=True)
    model = _load_saved_model(model_path)
    model.jit_compile = False
    inverse_mapping = {index: name for name, index in class_indices.items()}
    report = classification_report(
        [inverse_mapping[int(index)] for index in generator.classes],
        [
            inverse_mapping[int(prediction.argmax())]
            for prediction in model.predict(generator, verbose=1)
        ],
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(
        f"{display_name}: accuracy={report['accuracy']:.4f}, "
        f"macro F1={report['macro avg']['f1-score']:.4f}, "
        f"weighted F1={report['weighted avg']['f1-score']:.4f}"
    )
    for class_name in class_names:
        metrics = report[class_name]
        print(
            f"{class_name}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1-score']:.4f}"
        )

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{model_path.stem}_test_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Đã lưu báo cáo: {report_path}")
