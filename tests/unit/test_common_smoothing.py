"""시간축 후행 smoothing의 단위 테스트."""

import unittest

import numpy

from src.common.smoothing import trailing_average_smoothing


class TestTrailingAverageSmoothing(unittest.TestCase):
    def setUp(self):
        # 한 채널의 단일 spike가 시간축으로만 퍼지는지 확인한다.
        self.spiked_scores = numpy.zeros((10, 3))
        self.spiked_scores[:, 1] = 100.0
        self.spiked_scores[:, 2] = 200.0
        self.spiked_scores[5, 2] = 1000.0

    def test_matches_verification_measured_output(self):
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
