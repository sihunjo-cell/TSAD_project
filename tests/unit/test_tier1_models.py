"""Tier 1 공식 동작과 누수 방지 계약을 검증한다."""

import unittest
import warnings

import numpy
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models.tier1 import (
    PCA_COMPONENTS,
    PCA_WINDOW,
    MWVAR_WINDOW,
    PcaLegacy,
    score_mwvar,
    score_sqdiff_last3,
)


def _official_pca_features(values):
    windows = sliding_window_view(
        values, window_shape=PCA_WINDOW, axis=0,
    ).reshape(len(values) - PCA_WINDOW + 1, -1)
    return numpy.nan_to_num(zscore(windows, axis=1, ddof=1))


class TestMwvar(unittest.TestCase):
    def test_matches_centered_sample_variance_and_fills_both_edges(self):
        time = numpy.arange(97, dtype=float)
        values = numpy.column_stack((time ** 2, 3 * time ** 2 + 2))
        first = numpy.var(values[:MWVAR_WINDOW], axis=0, ddof=1)
        second = numpy.var(values[1:], axis=0, ddof=1)
        expected = numpy.vstack((
            numpy.repeat(first[None, :], 49, axis=0),
            numpy.repeat(second[None, :], 48, axis=0),
        ))

        result = score_mwvar(values)

        numpy.testing.assert_allclose(result["scores"], expected)
        self.assertEqual(result["scores"].shape, values.shape)
        self.assertEqual(result["source_start"], 0)
        self.assertEqual(result["source_end_exclusive"], len(values))
        self.assertEqual(result["alignment"], "same_timestep")
        self.assertEqual(result["primitive"], "sample_variance")
        self.assertEqual(result["calibration_mode"], "none")
        self.assertEqual(result["native_source_start"], 48)
        self.assertEqual(result["native_source_end_exclusive"], 50)
        self.assertEqual(result["boundary_repeat"], {"left": 48, "right": 47})
        self.assertEqual(result["evaluation_mode"], "offline_noncausal")
        self.assertEqual(result["lookahead"], 47)
        self.assertEqual(result["maximum_effective_lookahead"], 95)
        self.assertEqual(result["normalization_scope"], "none")

    def test_rejects_series_shorter_than_official_window(self):
        with self.assertRaisesRegex(ValueError, "96"):
            score_mwvar(numpy.ones((MWVAR_WINDOW - 1, 2)))


class TestSqdiffLast3(unittest.TestCase):
    def test_matches_trailing_four_point_equation_and_fills_first_three(self):
        values = numpy.column_stack((
            numpy.arange(5, dtype=float),
            numpy.ones(5),
        ))
        expected = numpy.column_stack((
            [4.0, 4.0 / 9.0, 4.0 / 9.0, 4.0, 4.0],
            numpy.zeros(5),
        ))

        result = score_sqdiff_last3(values)

        numpy.testing.assert_allclose(result["scores"], expected)
        self.assertEqual(result["scores"].shape, values.shape)
        self.assertEqual(result["source_start"], 0)
        self.assertEqual(result["source_end_exclusive"], len(values))
        self.assertEqual(result["alignment"], "same_timestep")
        self.assertEqual(result["primitive"], "squared_difference")
        self.assertEqual(result["calibration_mode"], "none")
        self.assertEqual(result["native_source_start"], 3)
        self.assertEqual(result["native_source_end_exclusive"], 5)
        self.assertNotIn("boundary_repeat", result)
        self.assertEqual(
            result["boundary_policy"], "backfill_first_complete_rolling_mean",
        )
        self.assertNotEqual(result["scores"][0, 0], result["scores"][1, 0])
        self.assertEqual(result["evaluation_mode"], "offline_noncausal")
        self.assertEqual(result["lookahead"], 0)
        self.assertEqual(result["maximum_effective_lookahead"], 3)
        self.assertEqual(result["normalization_scope"], "none")

    def test_rejects_series_shorter_than_trailing_window(self):
        with self.assertRaisesRegex(ValueError, "4"):
            score_sqdiff_last3(numpy.ones((3, 2)))


class TestPcaLegacy(unittest.TestCase):
    def setUp(self):
        generator = numpy.random.default_rng(7)
        self.fit_values = generator.normal(size=(140, 3)).cumsum(axis=0)

    def test_uses_only_approved_components_and_keeps_window_as_recipe_field(self):
        self.assertEqual(PCA_COMPONENTS, (0.25, 0.5, 0.75, None))
        for component in PCA_COMPONENTS:
            with self.subTest(component=component):
                model = PcaLegacy(n_components=component)
                self.assertEqual(model.window, PCA_WINDOW)
                self.assertIs(model.fit(self.fit_values), model)

        with self.assertRaisesRegex(ValueError, "n_components"):
            PcaLegacy(n_components=0.4)

    def test_fit_state_does_not_change_when_scoring_validation_or_test(self):
        model = PcaLegacy(n_components=0.5).fit(self.fit_values)
        scaler_mean = model.scaler.mean_.copy()
        scaler_scale = model.scaler.scale_.copy()
        component_vectors = model.pca.components_.copy()
        pca_mean = model.pca.mean_.copy()

        model.score(numpy.full((104, 3), 1000.0))
        model.score(numpy.full((105, 3), -1000.0))

        numpy.testing.assert_array_equal(model.scaler.mean_, scaler_mean)
        numpy.testing.assert_array_equal(model.scaler.scale_, scaler_scale)
        numpy.testing.assert_array_equal(model.pca.components_, component_vectors)
        numpy.testing.assert_array_equal(model.pca.mean_, pca_mean)

    def test_fit_matches_official_row_zscore_then_fit_only_standard_scaler(self):
        original = self.fit_values.copy()
        reference_features = _official_pca_features(self.fit_values)
        reference_scaler = StandardScaler().fit(reference_features)
        reference_fit = reference_scaler.transform(reference_features)
        reference_pca = PCA(n_components=0.5, random_state=0).fit(reference_fit)

        model = PcaLegacy(n_components=0.5).fit(self.fit_values)

        numpy.testing.assert_array_equal(self.fit_values, original)
        numpy.testing.assert_allclose(model.scaler.mean_, reference_scaler.mean_)
        numpy.testing.assert_allclose(model.scaler.scale_, reference_scaler.scale_)
        numpy.testing.assert_allclose(model.pca.mean_, reference_pca.mean_)
        numpy.testing.assert_allclose(
            numpy.abs(model.pca.components_),
            numpy.abs(reference_pca.components_),
        )

    def test_matches_weighted_component_distance_and_center_padding(self):
        generator = numpy.random.default_rng(11)
        test_values = generator.normal(size=(105, 3)).cumsum(axis=0)
        model = PcaLegacy(n_components=0.5).fit(self.fit_values)
        fit_features = _official_pca_features(self.fit_values)
        test_features = _official_pca_features(test_values)
        reference_scaler = StandardScaler().fit(fit_features)
        reference_fit = reference_scaler.transform(fit_features)
        reference_pca = PCA(n_components=0.5, random_state=0).fit(reference_fit)
        raw_scores = numpy.sum(
            cdist(
                reference_scaler.transform(test_features),
                reference_pca.components_,
            ) / reference_pca.explained_variance_ratio_,
            axis=1,
        )
        expected = numpy.concatenate((
            numpy.repeat(raw_scores[0], 50),
            raw_scores,
            numpy.repeat(raw_scores[-1], 49),
        ))

        result = model.score(test_values)

        numpy.testing.assert_allclose(result["scores"], expected)
        self.assertEqual(result["scores"].shape, (len(test_values),))
        self.assertEqual(result["source_start"], 0)
        self.assertEqual(result["source_end_exclusive"], len(test_values))
        self.assertEqual(result["alignment"], "centered_window_edge_repeat")
        self.assertEqual(result["primitive"], "weighted_component_distance")
        self.assertEqual(result["calibration_mode"], "validation_median_iqr")
        self.assertEqual(result["native_source_start"], 50)
        self.assertEqual(result["native_source_end_exclusive"], 56)
        self.assertEqual(result["boundary_repeat"], {"left": 50, "right": 49})
        self.assertEqual(result["evaluation_mode"], "offline_noncausal")
        self.assertEqual(result["lookahead"], 49)
        self.assertEqual(result["maximum_effective_lookahead"], 99)
        self.assertEqual(result["normalization_scope"], "current_prefix_validation")

    def test_constant_score_window_is_zero_normalized_without_warning(self):
        model = PcaLegacy(n_components=0.5).fit(self.fit_values)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.score(numpy.ones((PCA_WINDOW, 3)))

        self.assertTrue(numpy.isfinite(result["scores"]).all())
        self.assertFalse(any("Precision loss" in str(item.message) for item in caught))

    def test_rejects_invalid_inputs_and_score_before_fit(self):
        model = PcaLegacy(n_components=0.5)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.score(numpy.ones((PCA_WINDOW, 3)))

        with self.assertRaisesRegex(ValueError, "101"):
            PcaLegacy(n_components=0.5).fit(
                numpy.ones((PCA_WINDOW - 1, 3)),
            )
        with self.assertRaisesRegex(ValueError, "101"):
            PcaLegacy(n_components=0.5).fit(
                numpy.ones((PCA_WINDOW, 3)),
            )
        fitted = PcaLegacy(n_components=0.5).fit(self.fit_values)
        with self.assertRaisesRegex(ValueError, "100"):
            fitted.score(numpy.ones((PCA_WINDOW - 1, 3)))

        invalid_values = (
            numpy.ones(3),
            numpy.empty((0, 3)),
            numpy.array([[1.0, numpy.nan, 2.0]]),
        )
        for values in invalid_values:
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    PcaLegacy(n_components=0.5).fit(values)


if __name__ == "__main__":
    unittest.main()
