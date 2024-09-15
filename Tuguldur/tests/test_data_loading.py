from src.DatasetLoader import DatasetLoader
from src.helper import load_config

import unittest
from unittest.mock import patch
from pathlib import Path

class TestDatasetLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.CONFIG_FILE = "/home/nero/phasic_tonic/configs/dataset_loading.yaml"
        cls.patterns = load_config(cls.CONFIG_FILE)['patterns']
        cls.dataset_args = {
            "CBD": {"dir": "/home/nero/datasets/CBD", "pattern_set": "CBD"},
            "RGS": {"dir": "/home/nero/datasets/RGS14", "pattern_set": "RGS"},
            "OS": {"dir": "/home/nero/datasets/OSbasic", "pattern_set": "OS"}
        }
        cls.loader = DatasetLoader(cls.dataset_args, cls.CONFIG_FILE)

    def test_CBD(self):
        """
        Test that the CBD dataset is loaded correctly and has the expected length.
        """
        datasets = {"CBD": self.dataset_args['CBD']}
        result = self.loader.load_datasets(datasets)
        self.assertEqual(len(result), 170, "CBD dataset length mismatch")

    def test_RGS(self):
        """
        Test that the RGS dataset is loaded correctly and has the expected length.
        """
        datasets = {"RGS": self.dataset_args['RGS']}
        result = self.loader.load_datasets(datasets)
        self.assertEqual(len(result), 159, "RGS dataset length mismatch")

    def test_OS(self):
        """
        Test that the OS dataset is loaded correctly and has the expected length.
        """
        datasets = {"OS": self.dataset_args['OS']}
        result = self.loader.load_datasets(datasets)
        self.assertEqual(len(result), 210, "OS dataset length mismatch")

    def test_missing_files(self):
        """
        Test the handling of missing files in a directory.
        """
        with patch('src.DatasetLoader.process_directory', return_value={}):
            datasets = {"CBD": self.dataset_args['CBD']}
            result = self.loader.load_datasets(datasets)
            self.assertEqual(len(result), 0, "Should be empty due to missing files")

    def test_empty_directories(self):
        """
        Test the handling of empty directories.
        """
        with patch('os.walk', return_value=[("/fake/path", [], [])]):
            datasets = {"CBD": self.dataset_args['CBD']}
            result = self.loader.load_datasets(datasets)
            self.assertEqual(len(result), 0, "Should be empty due to empty directories")

    def test_incorrect_patterns(self):
        """
        Test the handling of incorrect patterns in the configuration.
        """
        incorrect_patterns = {
            "posttrial": r"incorrect_pattern",
            "hpc": r"incorrect_pattern",
            "states": r"incorrect_pattern"
        }
        with patch.object(self.loader, 'patterns', {"CBD": incorrect_patterns}):
            datasets = {"CBD": self.dataset_args['CBD']}
            result = self.loader.load_datasets(datasets)
            self.assertEqual(len(result), 0, "Should be empty due to incorrect patterns")

if __name__ == '__main__':
    unittest.main()
