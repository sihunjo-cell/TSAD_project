"""원격에서 최소 prefix 통계와 입력 비개입을 검사한다."""

import unittest
from unittest import mock

import numpy

from src.common.compute_prefix_features import compute_prefix_features, prefix_feature_identity


class TestPrefixFeatures(unittest.TestCase):
    def test_statistics_null_reasons_and_no_mutation(self):
        values = numpy.array([[1, 2, 7, numpy.nan], [2, 4, 7, numpy.inf],
                              [3, 6, 7, numpy.nan], [4, 8, 7, numpy.nan]],
                             dtype=numpy.float32)
        original = values.copy()
        random_state = numpy.random.get_state()
        summary, channels = compute_prefix_features(values, ["same", "same", "constant", "missing"])
        numpy.testing.assert_array_equal(values, original)
        after = numpy.random.get_state()
        self.assertEqual(random_state[0], after[0])
        numpy.testing.assert_array_equal(random_state[1], after[1])
        self.assertEqual(random_state[2:], after[2:])
        self.assertEqual(channels[0]["mean"], 2.5)
        self.assertAlmostEqual(channels[0]["std"], numpy.sqrt(1.25))
        self.assertEqual(channels[0]["median"], 2.5)
        self.assertEqual(channels[0]["acf_lag1"], 0.25)
        for field, expected in (("interquartile_range", 1.5),
                                ("difference_q90_iqr_ratio", 2 / 3),
                                ("median_shift_iqr_ratio", 4 / 3),
                                ("spectral_entropy", 0.9182958340544896)):
            self.assertAlmostEqual(channels[0][field], expected)
            self.assertIsNone(channels[0][field + "_reason"])
        self.assertEqual(channels[2]["interquartile_range"], 0.0)
        for field in ("difference_q90_iqr_ratio", "median_shift_iqr_ratio"):
            self.assertIsNone(channels[2][field])
            self.assertEqual(channels[2][field + "_reason"], "zero_interquartile_range")
        self.assertIsNone(channels[2]["spectral_entropy"])
        self.assertEqual(channels[2]["spectral_entropy_reason"], "constant_channel")
        self.assertIsNone(channels[3]["interquartile_range"])
        self.assertEqual(channels[3]["interquartile_range_reason"], "no_finite_values")
        self.assertEqual(channels[2]["acf_lag1_reason"], "constant_channel")
        self.assertIsNone(channels[3]["mean"])
        self.assertEqual(channels[3]["valid_value_count"], 0)
        self.assertIsNone(channels[3]["is_constant"])
        self.assertEqual(summary["constant_channel_count"], 1)
        self.assertEqual(summary["constant_evaluable_channel_count"], 3)
        self.assertEqual(summary["absolute_correlation_valid_pair_count"], 1)
        self.assertAlmostEqual(summary["absolute_correlation_median"], 1.0)
        for field, expected, valid_count in (("interquartile_range", 1.5, 3),
                                             ("difference_q90_iqr_ratio", 2 / 3, 2),
                                             ("median_shift_iqr_ratio", 4 / 3, 2),
                                             ("spectral_entropy", 0.9182958340544896, 2)):
            self.assertAlmostEqual(summary["channel_" + field + "_median"], expected)
            self.assertIsNone(summary["channel_" + field + "_median_reason"])
            self.assertEqual(summary["channel_" + field + "_valid_channel_count"], valid_count)

    def test_absolute_correlation_stays_within_unit_interval(self):
        for rows, expected in (
            ([[-1, -1], [1, 1]] * 3, 1.0),
            ([[-1, 1], [1, -1]] * 3, 1.0),
            ([[-1, -1], [0, 1], [1, 0]], 0.5),
        ):
            with self.subTest(rows=rows):
                summary, _ = compute_prefix_features(numpy.array(rows, dtype=numpy.float32), ["first", "second"])
                correlation = summary["absolute_correlation_median"]
                self.assertGreaterEqual(correlation, 0.0)
                self.assertLessEqual(correlation, 1.0)
                self.assertAlmostEqual(correlation, expected)
                self.assertIsNone(summary["absolute_correlation_median_reason"])
                self.assertEqual(summary["absolute_correlation_valid_pair_count"], 1)

    def test_short_partial_and_zero_iqr_nonconstant(self):
        summary, channels = compute_prefix_features(
            numpy.array([[1, 1], [1, numpy.nan], [1, 2], [1, 3], [2, 4]], dtype=numpy.float32),
            ["zero_iqr", "partial"],
        )
        self.assertEqual(channels[0]["is_constant"], 0)
        self.assertIsNotNone(channels[0]["acf_lag1"])
        self.assertEqual(channels[1]["acf_lag1_reason"], "nonfinite_input")
        self.assertEqual(channels[0]["interquartile_range"], 0.0)
        for field in ("difference_q90_iqr_ratio", "median_shift_iqr_ratio"):
            self.assertIsNone(channels[0][field])
            self.assertEqual(channels[0][field + "_reason"], "zero_interquartile_range")
        self.assertAlmostEqual(channels[0]["spectral_entropy"], 1.0)
        self.assertEqual(channels[1]["interquartile_range"], 1.5)
        for field in ("difference_q90_iqr_ratio", "median_shift_iqr_ratio", "spectral_entropy"):
            self.assertIsNone(channels[1][field])
            self.assertEqual(channels[1][field + "_reason"], "nonfinite_input")
        self.assertEqual(summary["absolute_correlation_valid_pair_count"], 0)
        self.assertIsNone(summary["absolute_correlation_median"])
        _, short = compute_prefix_features(numpy.array([[2]], dtype=numpy.float32), ["one"])
        self.assertEqual(short[0]["acf_lag1_reason"], "fewer_than_two_rows")
        self.assertEqual(short[0]["interquartile_range"], 0.0)
        self.assertIsNone(short[0]["interquartile_range_reason"])
        self.assertEqual(short[0]["difference_q90_iqr_ratio_reason"], "fewer_than_two_rows")
        for field in ("median_shift_iqr_ratio", "spectral_entropy"):
            self.assertIsNone(short[0][field])
            self.assertEqual(short[0][field + "_reason"], "fewer_than_four_rows")
        empty, _ = compute_prefix_features(numpy.empty((0, 1), dtype=numpy.float32), ["one"])
        self.assertIsNone(empty["constant_channel_count"])
        for field in ("interquartile_range", "difference_q90_iqr_ratio",
                      "median_shift_iqr_ratio", "spectral_entropy"):
            self.assertIsNone(empty["channel_" + field + "_median"])
            self.assertEqual(empty["channel_" + field + "_median_reason"], "no_valid_values")
            self.assertEqual(empty["channel_" + field + "_valid_channel_count"], 0)

    def test_difference_quantile_and_odd_half_split(self):
        _, two_rows = compute_prefix_features(numpy.array([[0], [2]], dtype=numpy.float32), ["one"])
        self.assertEqual(two_rows[0]["difference_q90_iqr_ratio"], 2.0)
        _, short = compute_prefix_features(numpy.array([[0], [1], [3]], dtype=numpy.float32), ["one"])
        self.assertAlmostEqual(short[0]["difference_q90_iqr_ratio"], 1.9 / 1.5)
        self.assertEqual(short[0]["median_shift_iqr_ratio_reason"], "fewer_than_four_rows")
        self.assertEqual(short[0]["spectral_entropy_reason"], "fewer_than_four_rows")
        _, channels = compute_prefix_features(
            numpy.array([[0], [2], [4], [6], [8]], dtype=numpy.float32), ["one"],
        )
        self.assertEqual(channels[0]["median_shift_iqr_ratio"], 1.25)

    def test_entropy_uses_undoubled_positive_bins_and_is_affine_invariant(self):
        values = numpy.array([[1, 1], [0, -1], [0, 1], [0, -1]], dtype=numpy.float32)
        _, channels = compute_prefix_features(values, ["impulse", "nyquist"])
        self.assertAlmostEqual(channels[0]["spectral_entropy"], 1.0)
        self.assertEqual(channels[1]["spectral_entropy"], 0.0)
        _, two_bins = compute_prefix_features(
            numpy.array([[1], [0], [0], [0], [1], [0], [0], [0]], dtype=numpy.float32), ["two_bins"],
        )
        self.assertAlmostEqual(two_bins[0]["spectral_entropy"], 0.5)
        _, transformed = compute_prefix_features(values * -4 + 13, ["impulse", "nyquist"])
        for original, shifted in zip(channels, transformed):
            self.assertEqual(shifted["interquartile_range"], 4 * original["interquartile_range"])
            for field in ("difference_q90_iqr_ratio", "median_shift_iqr_ratio", "spectral_entropy"):
                self.assertAlmostEqual(original[field], shifted[field])

    def test_nonfinite_spectrum_is_undefined(self):
        with mock.patch.object(numpy.fft, "rfft", return_value=numpy.array([0, numpy.nan, numpy.inf])):
            summary, channels = compute_prefix_features(
                numpy.array([[1], [2], [3], [4]], dtype=numpy.float32), ["one"],
            )
        self.assertIsNone(channels[0]["spectral_entropy"])
        self.assertEqual(channels[0]["spectral_entropy_reason"], "nonfinite_computation")
        self.assertEqual(summary["channel_spectral_entropy_valid_channel_count"], 0)

    def test_identity_changes_without_using_future_values(self):
        arguments = dict(csv_id="tuning/a.csv", source_sha256="a" * 64,
                         ordered_schema_sha256="b" * 64, q_percent=5,
                         source_start=0, source_end_exclusive=2)
        original = prefix_feature_identity(**arguments)
        self.assertEqual(original, prefix_feature_identity(**arguments))
        for field, value in (("source_sha256", "c" * 64), ("ordered_schema_sha256", "d" * 64),
                             ("q_percent", 10), ("source_end_exclusive", 3)):
            self.assertNotEqual(original["prefix_feature_id"],
                                prefix_feature_identity(**{**arguments, field: value})["prefix_feature_id"])
        values = numpy.array([[1], [3], [50]], dtype=numpy.float32)
        before = compute_prefix_features(values[:2], ["one"])
        values[2] = -999
        self.assertEqual(before, compute_prefix_features(values[:2], ["one"]))
        self.assertNotEqual(before, compute_prefix_features(values, ["one"]))


if __name__ == "__main__":
    unittest.main()
