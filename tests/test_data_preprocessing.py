import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from dataset.data_preprocessing import create_generators, directory_to_dataframe, split_dataframe


class SplitDataframeTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "image": [f"{label}/{i}.png" for label in "abcd" for i in range(100)],
            "label": [label for label in "abcd" for _ in range(100)],
        })

    def test_default_split_is_80_10_10_and_preserves_every_sample_once(self):
        train, validation, test = split_dataframe(self.frame)
        self.assertEqual([len(train), len(test), len(validation)], [320, 40, 40])
        combined = pd.concat([train, validation, test])
        self.assertEqual(set(combined.image), set(self.frame.image))
        self.assertTrue(combined.image.is_unique)
        for frame, count_per_label in ((train, 80), (validation, 10), (test, 10)):
            self.assertEqual(frame.label.value_counts().to_dict(), dict.fromkeys("abcd", count_per_label))

    def test_split_is_reproducible_and_seed_can_change_membership(self):
        first = split_dataframe(self.frame)
        second = split_dataframe(self.frame)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)
        changed = split_dataframe(self.frame, random_state=42)
        self.assertNotEqual(set(first[0].image), set(changed[0].image))

    def test_custom_split_sizes_are_fractions_of_the_whole_dataset(self):
        train, validation, test = split_dataframe(self.frame, test_size=0.2, validation_size=0.1)
        self.assertEqual([len(train), len(validation), len(test)], [280, 40, 80])

    def test_invalid_sizes_raise_clear_error(self):
        for test_size, validation_size in ((0, 0.1), (0.1, 0), (-0.1, 0.1), (0.6, 0.5)):
            with self.subTest(test_size=test_size, validation_size=validation_size):
                with self.assertRaisesRegex(ValueError, "test_size|validation_size"):
                    split_dataframe(self.frame, test_size=test_size, validation_size=validation_size)


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "input" / "dataset"
        for label in ("a_lower", "b_lower", "A_upper", "0", "é"):
            folder = self.dataset / label
            folder.mkdir(parents=True)
            for index in range(20):
                Image.new("RGB", (14, 18), color=(200, 100, 50)).save(folder / f"{index}.png")

    def test_scanner_accepts_wrapper_and_keeps_only_valid_english_lowercase_images(self):
        (self.dataset / "a_lower" / "notes.txt").write_text("not an image", encoding="utf-8")
        (self.dataset / "a_lower" / "broken.png").write_bytes(b"not a PNG")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = directory_to_dataframe(self.dataset.parent)
        self.assertEqual(len(frame), 40)
        self.assertEqual(set(frame.label), {"a", "b"})
        self.assertTrue(all(Path(path).is_absolute() for path in frame.image))
        self.assertTrue(any("broken.png" in str(item.message) for item in caught))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            direct = directory_to_dataframe(self.dataset)
        pd.testing.assert_frame_equal(frame, direct)

    def test_empty_dataset_fails_before_splitting(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "a-z"):
            directory_to_dataframe(empty)

    def test_cli_saves_usable_csv_mapping_and_summary_without_tensorflow(self):
        output = self.root / "prepared"
        command = [sys.executable, "-m", "dataset.data_preprocessing.prepare_data", "--dataset", str(self.dataset.parent),
                   "--output", str(output)]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        frames = [pd.read_csv(output / f"{name}.csv") for name in ("train", "validation", "test")]
        self.assertEqual([len(frame) for frame in frames], [32, 4, 4])
        combined = pd.concat(frames)
        self.assertTrue(combined.image.is_unique)
        self.assertEqual(len(combined), 40)
        for frame in frames:
            self.assertEqual(list(frame.columns), ["image", "label"])
            self.assertTrue(all(Path(path).is_file() for path in frame.image))
        mapping = json.loads((output / "class_indices.json").read_text(encoding="utf-8"))
        self.assertEqual(mapping, {"a": 0, "b": 1})
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["split_sizes"], {"train": 32, "validation": 4, "test": 4})
        snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
        rerun = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(rerun.returncode, 0, rerun.stderr)
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in output.iterdir()})

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_generators_match_saved_class_order_and_rgb_batch_format(self):
        frame = directory_to_dataframe(self.dataset)
        train, validation, test = split_dataframe(frame)
        generators = create_generators(train, validation, test, batch_size=4)
        for generator in generators:
            images, labels = generator[0]
            self.assertEqual(images.shape, (4, 32, 32, 3))
            self.assertEqual(images.dtype, np.int32)
            self.assertEqual(labels.shape, (4, 2))
            self.assertEqual(generator.class_indices, {"a": 0, "b": 1})
        self.assertTrue(generators[0].shuffle)
        self.assertFalse(generators[1].shuffle)
        self.assertFalse(generators[2].shuffle)


if __name__ == "__main__":
    unittest.main()
