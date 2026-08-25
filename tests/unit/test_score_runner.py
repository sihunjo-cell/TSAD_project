import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.채점기.score_runner import (
    align_labels,
    average_precision,
    evaluate_score_file,
    f1_at_threshold,
    load_test_labels,
)


class TestScoreRunner(unittest.TestCase):
    def test_average_precision_handles_score_ties(self):
        score = np.array([0.9, 0.9, 0.2, 0.1])
        label = np.array([1, 0, 1, 0])
        # score=0.9의 동점 두 개는 하나의 threshold로 묶여 precision=1/2,
        # score=0.2에서 precision=2/3이므로 AP=1/2*1/2 + 1/2*2/3.
        self.assertAlmostEqual(average_precision(score, label), 7 / 12)

    def test_label_slice_must_match_score_length(self):
        with self.assertRaises(ValueError):
            align_labels(np.array([0, 1, 0]), [1, None], score_length=3)

    def test_f1_uses_fixed_threshold_without_point_adjust(self):
        self.assertAlmostEqual(
            f1_at_threshold(np.array([0.2, 0.8, 0.9]), np.array([0, 1, 0]), 0.5),
            2 / 3,
        )

    def test_hai_label_file_is_selected_by_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "label-test1.csv").write_text(
                "timestamp,label\n2023-01-01,0\n2023-01-02,1\n", encoding="utf-8"
            )
            np.testing.assert_array_equal(load_test_labels("HAI", 1, tmp), [0, 1])

    def test_evaluate_score_file_uses_metadata_label_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            score_path = Path(tmp, "GHL__01__GDN__t2__r010__s1__raw__trainnorm.npy")
            np.save(score_path, [0.1, 0.9, 0.2, 0.8])
            score_path.with_suffix(".meta.json").write_text(json.dumps({
                "window_size": 2,
                "test_length": 6,
                "score_length": 4,
                "label_slice": [2, None],
                "validation_threshold": 0.5,
                "validation_threshold_quantile": 0.99,
                "validation_score_count": 100,
                "validation_score_source": "raw_trainnorm_aggregated_validation",
            }), encoding="utf-8")

            row = evaluate_score_file(score_path, np.array([0, 0, 0, 1, 0, 1]))

            self.assertEqual(row["l_max"], 332)
            self.assertEqual(row["label_positive_count"], 2)
            self.assertEqual(row["n_thresholds"], 250)
            self.assertGreaterEqual(row["vus_pr"], 0.0)
            self.assertLessEqual(row["vus_pr"], 1.0)


if __name__ == "__main__":
    unittest.main()
