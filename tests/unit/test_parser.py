import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.채점기.parser import (
    load_and_validate_score,
)


class TestScoreParser(unittest.TestCase):

    def create_score_files(self, directory, scores, metadata):
        score_path = (
            Path(directory)
            / "GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy"
        )

        np.save(score_path, np.asarray(scores, dtype=float))

        metadata_path = score_path.with_suffix(".meta.json")
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file)

        return score_path

    def test_valid_score_file(self):
        metadata = {
            "window_size": 5,
            "test_length": 5,
            "score_length": 5,
            "label_slice": [5, None],
        }

        with tempfile.TemporaryDirectory() as tmp:
            score_path = self.create_score_files(
                tmp,
                [0.1, 0.2, 0.3, 0.4, 0.5],
                metadata,
            )

            scores, file_info, loaded_metadata = (
                load_and_validate_score(score_path)
            )

            self.assertEqual(len(scores), 5)
            self.assertEqual(file_info["dataset"], "GHL")
            self.assertEqual(file_info["model"], "GDN")
            self.assertEqual(loaded_metadata["score_length"], 5)

    def test_score_length_mismatch(self):
        metadata = {
            "window_size": 5,
            "test_length": 5,
            "score_length": 10,
            "label_slice": [5, None],
        }

        with tempfile.TemporaryDirectory() as tmp:
            score_path = self.create_score_files(
                tmp,
                [0.1, 0.2, 0.3, 0.4, 0.5],
                metadata,
            )

            with self.assertRaises(ValueError):
                load_and_validate_score(score_path)


if __name__ == "__main__":
    unittest.main()
