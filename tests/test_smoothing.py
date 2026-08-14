"""src/common/smoothing.py 단위 테스트.

기대값의 원본은 VERIFICATION.md의 "d-ailin 재현(후행 4-창)" 실측 출력 표다 —
직접 계산하지 않고 그대로 옮겨 적었다 (D-05: 단위 테스트로 그 출력과 대조).
"""

import unittest

import numpy

from src.common.smoothing import trailing_average_smoothing


class TestTrailingAverageSmoothing(unittest.TestCase):
    def setUp(self):
        # VERIFICATION.md 의혹 2의 입력: 10×3, 채널 0/1/2 = 0/100/200, t=5 채널2만 1000.
        self.spiked_scores = numpy.zeros((10, 3))
        self.spiked_scores[:, 1] = 100.0
        self.spiked_scores[:, 2] = 200.0
        self.spiked_scores[5, 2] = 1000.0

    def test_matches_verification_measured_output(self):
        # VERIFICATION.md "d-ailin 재현(후행 4-창, 처음 3개 시점은 0) 출력" 표 그대로.
        expected = numpy.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 100.0, 200.0],
            [0.0, 100.0, 200.0],
            [0.0, 100.0, 400.0],
            [0.0, 100.0, 400.0],
            [0.0, 100.0, 400.0],
            [0.0, 100.0, 400.0],
            [0.0, 100.0, 200.0],
        ])
        smoothed = trailing_average_smoothing(self.spiked_scores, window=4)
        numpy.testing.assert_array_equal(smoothed, expected)

    def test_constant_channel_preserved_after_warmup(self):
        smoothed = trailing_average_smoothing(self.spiked_scores, window=4)
        numpy.testing.assert_array_equal(smoothed[3:, 1], numpy.full(7, 100.0))

    def test_single_channel_input(self):
        single_channel = numpy.arange(10, dtype=float).reshape(10, 1)
        smoothed = trailing_average_smoothing(single_channel, window=4)
        self.assertEqual(smoothed.shape, (10, 1))
        numpy.testing.assert_array_equal(smoothed[:3, 0], numpy.zeros(3))
        # t=3: mean(0,1,2,3) = 1.5 — 후행 4-창의 정의 확인.
        self.assertEqual(smoothed[3, 0], 1.5)

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            trailing_average_smoothing(numpy.zeros(10))


if __name__ == "__main__":
    unittest.main()
