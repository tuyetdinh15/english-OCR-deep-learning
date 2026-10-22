"""Prepare lowercase OCR samples and batch generators.

Training images stay on disk; resizing and brightness augmentation happen
inside the generators, just as in the reference notebook.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import ascii_lowercase
import warnings

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from config import BATCH_SIZE, IMAGE_SIZE


def directory_to_dataframe(path: str | Path) -> pd.DataFrame:
    """Read valid a-z images from class folders, including a dataset/ wrapper.

    Folder names such as ``a`` and ``a_lower`` both map to label ``a``.
    Non-image files are ignored; corrupt image files are skipped with a warning.
    Absolute paths allow the exported CSVs to be used from any working directory.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    class_dirs = _class_directories(root)
    if not class_dirs and (root / "dataset").is_dir():
        root = root / "dataset"
        class_dirs = _class_directories(root)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff"}
    rows = []
    invalid_images = []
    # Image validation is I/O-bound on the many small files in OCR-Dataset.
    # map() preserves input order, so worker scheduling cannot change the split.
    # Submit only one class at a time to bound pending tasks and memory use.
    with ThreadPoolExecutor(max_workers=16) as pool:
        for class_dir in class_dirs:
            label = class_dir.name.split("_")[0]
            candidates = [
                p for p in sorted(class_dir.iterdir())
                if p.suffix.lower() in extensions and p.is_file()
            ]
            for image_path, readable in zip(candidates, pool.map(_is_readable_image, candidates)):
                if not readable:
                    invalid_images.append(str(image_path))
                    continue
                rows.append({"image": str(image_path), "label": label})
    if invalid_images:
        warnings.warn(
            f"Skipped {len(invalid_images)} unreadable image(s): " + ", ".join(invalid_images[:3]),
            stacklevel=2,
        )
    if not rows:
        raise ValueError(f"No valid a-z images found in class directories under {root}")
    return pd.DataFrame(rows, columns=["image", "label"])


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        # Decode too: verify() alone does not detect every truncated JPEG.
        with Image.open(path) as image:
            image.convert("RGB").load()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return False
    return True


def _class_directories(root: Path) -> list[Path]:
    return [
        child for child in sorted(root.iterdir())
        if child.is_dir() and child.name.split("_")[0] in ascii_lowercase
        and len(child.name.split("_")[0]) == 1
    ]


def split_dataframe(dataframe: pd.DataFrame, test_size: float = 0.1, validation_size: float = 0.1, random_state: int = 41):
    """Return (train, validation, test), stratified by label with a fixed seed.

    Both sizes are fractions of the ORIGINAL dataset. Defaults give 80/10/10.
    Test and validation counts are rounded up to whole samples; train receives
    the remainder. Very small datasets may not support stratification.
    """
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive fractions with a sum below 1")
    test_count = math.ceil(len(dataframe) * test_size)
    validation_count = math.ceil(len(dataframe) * validation_size)
    train, test = train_test_split(dataframe, test_size=test_count, random_state=random_state, stratify=dataframe["label"])
    train, validation = train_test_split(train, test_size=validation_count, random_state=random_state, stratify=train["label"])
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def create_generators(training, validation, testing, batch_size: int = BATCH_SIZE):
    """Yield RGB 32x32 int32 batches with brightness augmentation on train only."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    classes = sorted(training["label"].unique())
    for name, frame in (("validation", validation), ("test", testing)):
        if not set(frame["label"]).issubset(classes):
            raise ValueError(f"{name} contains labels absent from the training set")
    train_generator = ImageDataGenerator(dtype="int32", brightness_range=[0.0, 1.0], fill_mode="nearest")
    evaluation_generator = ImageDataGenerator(dtype="int32", fill_mode="nearest")
    options = dict(x_col="image", y_col="label", batch_size=batch_size, target_size=IMAGE_SIZE,
                   color_mode="rgb", class_mode="categorical", classes=classes)
    return (
        train_generator.flow_from_dataframe(training, **options, seed=41),
        evaluation_generator.flow_from_dataframe(validation, **options, shuffle=False),
        evaluation_generator.flow_from_dataframe(testing, **options, shuffle=False),
    )
