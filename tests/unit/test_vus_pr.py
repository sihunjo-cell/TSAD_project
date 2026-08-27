import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np

from src.채점기.vus_pr import (
    _calculate_ap,
    _calculate_ap_for_window,
    _extend_anomaly_ranges,
    _generate_thresholds,
    _get_anomaly_ranges,
    _predict_at_thresholds,
    vus_pr,
)


class TestVusPR(unittest.TestCase):

    def test_committed_official_tsb_ad_comparison_stays_equal(self):
        repository_root = Path(__file__).resolve().parents[2]
        report = json.loads((
            repository_root / "experiments" / "checks" / "reference_code"
            / "vus_pr" / "official_tsb_ad_comparison.json"
        ).read_text(encoding="utf-8"))
        evaluator_path = repository_root / "src" / "채점기" / "vus_pr.py"

        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["official"]["commit"],
            "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48",
        )
        self.assertEqual(report["official"]["version"], "opt")
        self.assertEqual(report["n_thresholds"], 250)
        self.assertEqual(
            report["evaluator_sha256"],
            hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        )
        for fixture in report["fixtures"]:
            current = vus_pr(
                np.asarray(fixture["score"], dtype=float),
                np.asarray(fixture["label"], dtype=int),
                l_max=fixture["l_max"],
                n_thresholds=report["n_thresholds"],
            )
            self.assertAlmostEqual(
                current, fixture["official_vus_pr"],
                delta=report["absolute_tolerance"],
            )

    def test_get_anomaly_ranges(self):
        label = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        self.assertEqual(_get_anomaly_ranges(label), [(2, 4), (7, 8)])

    def test_soft_buffer_uses_half_window_on_each_side(self):
        label = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        result = _extend_anomaly_ranges(label, window=4)
        expected = np.array([
            0.0, 0.0, math.sqrt(0.5), math.sqrt(0.75), 1.0,
            1.0, math.sqrt(0.75), math.sqrt(0.5), 0.0, 0.0,
        ])
        np.testing.assert_allclose(result, expected)

    def test_odd_window_has_no_buffer_when_half_window_is_zero(self):
        label = np.array([0, 0, 1, 1, 0])
        np.testing.assert_array_equal(_extend_anomaly_ranges(label, window=1), label)

    def test_thresholds_and_predictions_follow_score_rank(self):
        score = np.array([0.1, 0.9, 0.2, 0.8])
        thresholds = _generate_thresholds(score, 4)
        np.testing.assert_array_equal(thresholds, [0.9, 0.8, 0.2, 0.1])
        np.testing.assert_array_equal(
            _predict_at_thresholds(score, thresholds),
            [[False, True, False, False], [False, True, False, True],
             [False, True, True, True], [True, True, True, True]],
        )

    def test_window_zero_uses_range_existence_reward(self):
        score = np.array([0.1, 0.9, 0.2, 0.8])
        label = np.array([1, 0, 1, 0])
        result = _calculate_ap_for_window(score, label, window=0, n_thresholds=4)

        # One of two anomaly ranges is detected at the third threshold, then
        # both are detected at the fourth threshold.
        self.assertAlmostEqual(result, 11 / 24)

    def test_vus_pr_is_mean_of_window_range_aps(self):
        score = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.2, 0.1, 0.7, 0.3, 0.1])
        label = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
        expected = np.mean([
            _calculate_ap_for_window(score, label, window, n_thresholds=8, support_window=2)
            for window in range(3)
        ])
        self.assertAlmostEqual(vus_pr(score, label, l_max=2, n_thresholds=8), expected)

    def test_calculate_ap_requires_matching_lengths(self):
        with self.assertRaises(ValueError):
            _calculate_ap(np.array([1.0]), np.array([0.0, 1.0]))

    def test_rejects_invalid_metric_inputs(self):
        valid_score = np.array([0.1, 0.2, 0.3])
        valid_label = np.array([0, 1, 0])
        invalid_cases = (
            (np.array([0.1, np.nan, 0.3]), valid_label, 1, 3),
            (np.array([0.1, np.inf, 0.3]), valid_label, 1, 3),
            (valid_score, np.array([0, 0, 0]), 1, 3),
            (valid_score, np.array([0, 0.5, 1]), 1, 3),
            (valid_score, valid_label, -1, 3),
            (valid_score, valid_label, 1, 0),
        )
        for score, label, l_max, n_thresholds in invalid_cases:
            with self.subTest(score=score, label=label, l_max=l_max, n_thresholds=n_thresholds):
                with self.assertRaises(ValueError):
                    vus_pr(score, label, l_max=l_max, n_thresholds=n_thresholds)


if __name__ == "__main__":
    unittest.main()
