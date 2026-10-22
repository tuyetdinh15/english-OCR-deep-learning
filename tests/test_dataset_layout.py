import importlib
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import unittest

import config


class DatasetLayoutTests(unittest.TestCase):
    def test_raw_and_processed_are_under_the_single_dataset_root(self):
        root = Path(__file__).resolve().parents[1] / "dataset"
        self.assertEqual(config.DATASET_DIR, root)
        self.assertEqual(getattr(config, "RAW_DATA_DIR", None), root / "raw")
        self.assertEqual(getattr(config, "PROCESSED_DATA_DIR", None), root / "processed")

    def test_preprocessing_package_uses_raw_and_processed_defaults(self):
        module_name = "dataset.data_preprocessing.prepare_data"
        self.assertIsNotNone(importlib.util.find_spec(module_name))
        module = importlib.import_module(module_name)
        parameters = inspect.signature(module.prepare_dataset).parameters
        self.assertEqual(parameters["dataset_dir"].default, config.RAW_DATA_DIR)
        self.assertEqual(parameters["output_dir"].default, config.PROCESSED_DATA_DIR)
        package = importlib.import_module("dataset.data_preprocessing")
        self.assertTrue(callable(package.directory_to_dataframe))
        self.assertTrue(callable(package.split_dataframe))
        self.assertTrue(callable(package.create_generators))

    def test_new_cli_entrypoint_is_available(self):
        result = subprocess.run(
            [sys.executable, "-m", "dataset.data_preprocessing.prepare_data", "--help"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dataset", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
