"""src/common/normalization.py 단위 테스트."""

import unittest

import numpy
import yaml

from src.common.normalization import apply_median_iqr, estimate_median_iqr


class TestEstimateMedianIqr(unittest.TestCase):
    def test_hand_computed_5x2(self):
        # 열0 = [1..5]: median 3, q75 4, q25 2 → iqr 2. 열1 = [10,20,30,40,50]: median 30, iqr 20.
        scores = numpy.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)
        median, iqr = estimate_median_iqr(scores)
        numpy.testing.assert_array_equal(median, [3.0, 30.0])
        numpy.testing.assert_array_equal(iqr, [2.0, 20.0])


class TestApplyMedianIqr(unittest.TestCase):
    def test_hand_computed_values(self):
        median = numpy.array([3.0, 30.0])
        iqr = numpy.array([2.0, 20.0])
        normalized = apply_median_iqr(numpy.array([[5.0, 10.0]]), median, iqr, epsilon=0.01)
        numpy.testing.assert_allclose(normalized, [[(5 - 3) / 2.01, (10 - 30) / 20.01]])

    def test_zero_iqr_stays_finite_with_epsilon(self):
        # 상수 채널에서도 epsilon이 0 나눗셈을 막는다.
        constant_scores = numpy.full((5, 1), 7.0)
        median, iqr = estimate_median_iqr(constant_scores)
        self.assertEqual(iqr[0], 0.0)
        normalized = apply_median_iqr(constant_scores, median, iqr, epsilon=0.01)
        self.assertTrue(numpy.isfinite(normalized).all())
        numpy.testing.assert_array_equal(normalized, numpy.zeros((5, 1)))

    def test_train_statistics_applied_to_test_split(self):
        # 학습·validation 통계를 테스트 구간에 적용한다.
        with open("configs/scoring_pipeline.yaml", encoding="utf-8") as config_file:
            epsilon = yaml.safe_load(config_file)["normalization"]["epsilon"]
        self.assertEqual(epsilon, 0.01)

        train_scores = numpy.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        test_scores = numpy.array([[3.0], [103.0]])
        median, iqr = estimate_median_iqr(train_scores)
        normalized_test = apply_median_iqr(test_scores, median, iqr, epsilon=epsilon)
        # (3-3)/2.01 = 0, (103-3)/2.01 ≈ 49.75 — 테스트 구간 통계는 어디에도 쓰이지 않는다.
        numpy.testing.assert_allclose(normalized_test, [[0.0], [100.0 / 2.01]])


if __name__ == "__main__":
    unittest.main()
