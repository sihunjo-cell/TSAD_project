import unittest
import numpy as np

from src.채점기.vus_pr import (
    _generate_thresholds,
    _predict_at_thresholds,
    _get_anomaly_ranges,
    _extend_anomaly_ranges,
    _calculate_precision_recall,
    _calculate_ap,
    _calculate_ap_for_window,
    vus_pr,
)


class TestVusPR(unittest.TestCase):

    def test_get_anomaly_ranges(self):
        label = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])

        result = _get_anomaly_ranges(label)

        self.assertEqual(result, [(2, 4), (7, 8)])

    def test_extend_anomaly_ranges(self):
        label = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])

        result = _extend_anomaly_ranges(label, 1)

        expected = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        np.testing.assert_array_equal(result, expected)

    def test_precision_recall(self):
        prediction = np.array([0, 0, 1, 1, 0])
        label = np.array([0, 0, 1, 0, 0])

        precision, recall = _calculate_precision_recall(
            prediction,
            label
        )

        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 1.0)

    def test_calculate_ap(self):
        precisions = np.array([1.0, 0.8, 0.6, 0.5])
        recalls = np.array([0.0, 0.2, 0.6, 1.0])

        ap = _calculate_ap(precisions, recalls)

        self.assertAlmostEqual(ap, 0.6)

    def test_calculate_ap_for_window(self):
        score = np.array([
            0.1, 0.2, 0.3, 0.8, 0.9,
            0.2, 0.1, 0.7, 0.3, 0.1
        ])

        label = np.array([
            0, 0, 0, 1, 1,
            0, 0, 1, 0, 0
        ])

        ap = _calculate_ap_for_window(
            score,
            label,
            0
        )

        self.assertGreaterEqual(ap, 0.0)
        self.assertLessEqual(ap, 1.0)

    def test_vus_pr(self):
        score = np.array([
            0.1, 0.2, 0.3, 0.8, 0.9,
            0.2, 0.1, 0.7, 0.3, 0.1
        ])

        label = np.array([
            0, 0, 0, 1, 1,
            0, 0, 1, 0, 0
        ])

        result = vus_pr(
            score,
            label,
            l_max=2
        )

        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


if __name__ == "__main__":
    unittest.main()