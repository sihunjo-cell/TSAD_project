"""현재 비율 prefix 안의 시간순 80:20 분할을 검증한다."""

import unittest

import numpy

from src.data_split.split_ratio_prefix import (
    InsufficientPrefixError,
    compute_prefix_counts,
    split_ratio_prefix,
)


class TestComputePrefixCounts(unittest.TestCase):
    def test_uses_floor_for_available_and_fit_counts(self):
        self.assertEqual(compute_prefix_counts(500, 5), (25, 20, 5))
        self.assertEqual(compute_prefix_counts(19, 5), (0, 0, 0))
        self.assertEqual(compute_prefix_counts(101, 5), (5, 4, 1))

    def test_rejects_unregistered_ratio(self):
        with self.assertRaisesRegex(ValueError, "ratio"):
            compute_prefix_counts(500, 7)


class TestSplitRatioPrefix(unittest.TestCase):
    def test_never_reads_beyond_current_prefix(self):
        normal = numpy.arange(500 * 2).reshape(500, 2)

        fit, validation, split = split_ratio_prefix(normal, 5)

        self.assertEqual(fit.shape, (20, 2))
        self.assertEqual(validation.shape, (5, 2))
        numpy.testing.assert_array_equal(fit, normal[:20])
        numpy.testing.assert_array_equal(validation, normal[20:25])
        self.assertEqual(split, {
            "normal_range": (0, 500),
            "available_range": (0, 25),
            "fit_range": (0, 20),
            "validation_range": (20, 25),
            "unseen_range": (25, 500),
            "ratio_percent": 5,
        })

    def test_minima_are_checked_without_padding(self):
        with self.assertRaisesRegex(InsufficientPrefixError, "available=5"):
            split_ratio_prefix(
                numpy.zeros((101, 2)), 5,
                min_fit_length=5, min_validation_length=1,
            )

    def test_rejects_non_matrix_or_empty_input(self):
        for values in (numpy.array([]), numpy.array([1.0, 2.0])):
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    split_ratio_prefix(values, 5)


if __name__ == "__main__":
    unittest.main()
