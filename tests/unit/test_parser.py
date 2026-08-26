import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.채점기.parser import (
    load_and_validate_score,
)


class TestScoreParser(unittest.TestCase):

    def create_score_files(self, directory, scores, metadata, filename=None):
        score_path = (
            Path(directory)
            / (filename or "GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy")
        )

        np.save(score_path, np.asarray(scores, dtype=float))

        metadata_path = score_path.with_suffix(".meta.json")
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file)

        return score_path

    def test_valid_score_file(self):
        metadata = {
            "window_size": 5,
            "test_length": 10,
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
            "test_length": 10,
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

    def test_rejects_channels_and_testnorm_files(self):
        metadata = {
            "window_size": 5, "test_length": 10, "score_length": 5,
            "label_slice": [5, None],
        }
        invalid_names = (
            "GHL__03__GDN__t2__r010__s1__raw__trainnorm__channels.npy",
            "GHL__03__GDN__t2__r010__s1__raw__testnorm.npy",
        )
        with tempfile.TemporaryDirectory() as tmp:
            for filename in invalid_names:
                with self.subTest(filename=filename):
                    score_path = self.create_score_files(tmp, [0.1] * 5, metadata, filename)
                    with self.assertRaises(ValueError):
                        load_and_validate_score(score_path)

    def test_rejects_non_vector_and_nonfinite_scores(self):
        metadata = {
            "window_size": 5, "test_length": 10, "score_length": 5,
            "label_slice": [5, None],
        }
        invalid_scores = (np.ones((5, 2)), [0.1, 0.2, np.nan, 0.4, 0.5])
        with tempfile.TemporaryDirectory() as tmp:
            for index, scores in enumerate(invalid_scores):
                with self.subTest(scores=scores):
                    score_path = self.create_score_files(tmp, scores, metadata,
                                                         f"GHL__03__GDN__t2__r010__s{index}__raw__trainnorm.npy")
                    with self.assertRaises(ValueError):
                        load_and_validate_score(score_path)

    def test_rejects_invalid_metadata_type_and_label_alignment(self):
        invalid_metadata = (
            {"window_size": 5, "test_length": 10, "score_length": "5", "label_slice": [5, None]},
            {"window_size": 5, "test_length": 10, "score_length": 5, "label_slice": [6, None]},
            {"window_size": 5, "test_length": 10, "score_length": 5, "label_slice": [5, 11]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, metadata in enumerate(invalid_metadata):
                with self.subTest(metadata=metadata):
                    score_path = self.create_score_files(tmp, [0.1] * 5, metadata,
                                                         f"GHL__03__GDN__t2__r010__s{index}__raw__trainnorm.npy")
                    with self.assertRaises(ValueError):
                        load_and_validate_score(score_path)


if __name__ == "__main__":
    unittest.main()
