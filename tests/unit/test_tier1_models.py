"""Tier 1 공식 동작과 누수 방지 계약을 검증한다."""

import io
import unittest
import warnings
from unittest.mock import patch

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
    score_mwvar96_sqdiff_centered5,
    score_mwvar96_sqdiff_last3,
    score_pca_official,
    score_sqdiff_centered5,
    score_sqdiff_last1,
    score_sqdiff_last3,
)


def _official_pca_features(values):
    windows = sliding_window_view(
        values, window_shape=PCA_WINDOW, axis=0,
    ).reshape(len(values) - PCA_WINDOW + 1, -1)
    return numpy.nan_to_num(zscore(windows, axis=1, ddof=1))


class TestMwvar(unittest.TestCase):
    def test_all_official_comparison_windows_keep_their_native_alignment(self):
        for window in (5, 10, 32, 50, 60, 64, 96, 100, 256, 512, 1024):
            with self.subTest(window=window):
                time = numpy.arange(window + 1, dtype=float)
                values = numpy.column_stack((time ** 2, 3 * time ** 2 + 2))
                original = values.copy()
                expected = numpy.vstack((
                    numpy.repeat(
                        numpy.var(values[:window], axis=0, ddof=1)[None, :],
                        window // 2 + 1, axis=0,
                    ),
                    numpy.repeat(
                        numpy.var(values[1:], axis=0, ddof=1)[None, :],
                        (window + 1) // 2, axis=0,
                    ),
                ))
                result = score_mwvar(values, window=window)
                numpy.testing.assert_allclose(result["scores"], expected)
                numpy.testing.assert_array_equal(values, original)
                self.assertEqual(result["native_source_start"], window // 2)
                self.assertEqual(result["lookahead"], (window - 1) // 2)
                self.assertEqual(result["maximum_effective_lookahead"], window - 1)

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

    def test_window64_changes_scores_and_native_boundaries(self):
        time = numpy.arange(65, dtype=float)
        values = numpy.column_stack((time ** 2, 3 * time ** 2 + 2))
        first = numpy.var(values[:64], axis=0, ddof=1)
        second = numpy.var(values[1:], axis=0, ddof=1)
        expected = numpy.vstack((
            numpy.repeat(first[None, :], 33, axis=0),
            numpy.repeat(second[None, :], 32, axis=0),
        ))
        result = score_mwvar(values, window=64)
        numpy.testing.assert_allclose(result["scores"], expected)
        self.assertEqual(result["native_source_start"], 32)
        self.assertEqual(result["native_source_end_exclusive"], 34)
        self.assertEqual(result["boundary_repeat"], {"left": 32, "right": 31})
        self.assertEqual(result["lookahead"], 31)
        self.assertEqual(result["maximum_effective_lookahead"], 63)
        with self.assertRaisesRegex(ValueError, "64"):
            score_mwvar(values[:63], window=64)


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


class TestSqdiffVariants(unittest.TestCase):
    def test_last1_is_a_separate_model_and_matches_first_difference(self):
        values = numpy.column_stack(([1.0, 3.0, 6.0, 10.0], numpy.ones(4)))
        original = values.copy()
        result = score_sqdiff_last1(values)
        numpy.testing.assert_allclose(
            result["scores"], numpy.column_stack(([4.0, 4.0, 9.0, 16.0], numpy.zeros(4))),
        )
        numpy.testing.assert_array_equal(values, original)
        self.assertEqual(result["native_source_start"], 1)
        self.assertEqual(result["native_source_end_exclusive"], 4)
        self.assertEqual(result["lookahead"], 0)
        self.assertEqual(result["maximum_effective_lookahead"], 1)
        self.assertEqual(result["normalization_scope"], "none")
        self.assertFalse(numpy.array_equal(result["scores"], score_sqdiff_last3(values)["scores"]))

    def test_centered5_excludes_current_point_and_backfills_the_mean(self):
        values = numpy.column_stack((numpy.arange(7, dtype=float) ** 2, numpy.ones(7)))
        original = values.copy()
        means = numpy.array([6.0, 6.0, 6.0, 11.0, 18.0, 18.0, 18.0])
        expected = numpy.column_stack((((values[:, 0] - means) * 5 / 4) ** 2, numpy.zeros(7)))
        result = score_sqdiff_centered5(values)
        numpy.testing.assert_allclose(result["scores"], expected)
        numpy.testing.assert_array_equal(values, original)
        self.assertEqual(result["native_source_start"], 2)
        self.assertEqual(result["native_source_end_exclusive"], 5)
        self.assertEqual(result["lookahead"], 2)
        self.assertEqual(result["maximum_effective_lookahead"], 4)
        self.assertEqual(result["boundary_policy"], "backfill_and_forward_fill_complete_rolling_mean")
        self.assertEqual(result["scores"].shape, values.shape)
        self.assertNotEqual(result["scores"][0, 0], result["scores"][1, 0])

    def test_variants_reject_short_or_nonfinite_inputs(self):
        for scorer, window in ((score_sqdiff_last1, 2), (score_sqdiff_centered5, 5)):
            with self.subTest(scorer=scorer.__name__):
                with self.assertRaisesRegex(ValueError, str(window)):
                    scorer(numpy.ones((window - 1, 2)))
                invalid = numpy.ones((window, 2))
                invalid[0, 0] = numpy.nan
                with self.assertRaisesRegex(ValueError, "finite"):
                    scorer(invalid)


class TestOneLinerEnsemble(unittest.TestCase):
    def test_matches_full_evaluation_component_minmax_then_max_per_channel(self):
        time = numpy.arange(100, dtype=float)
        values = numpy.column_stack((time ** 2, time ** 3 + 100 * numpy.sin(time)))
        original = values.copy()
        for scorer, component, component_name in (
            (score_mwvar96_sqdiff_last3, score_sqdiff_last3, "SQDIFF_LAST3"),
            (score_mwvar96_sqdiff_centered5, score_sqdiff_centered5, "SQDIFF_CENTERED5"),
        ):
            with self.subTest(scorer=scorer.__name__):
                variance = score_mwvar(values, window=96)["scores"]
                difference = component(values)["scores"]
                expected = numpy.maximum(
                    (variance - variance.min(axis=0)) / numpy.ptp(variance, axis=0),
                    (difference - difference.min(axis=0)) / numpy.ptp(difference, axis=0),
                )
                result = scorer(values)
                numpy.testing.assert_allclose(result["scores"], expected)
                numpy.testing.assert_array_equal(values, original)
                self.assertEqual(result["source_start"], 0)
                self.assertEqual(result["source_end_exclusive"], len(values))
                self.assertEqual(result["alignment"], "same_timestep")
                self.assertEqual(result["calibration_mode"], "none")
                self.assertEqual(result["evaluation_mode"], "offline_noncausal")
                self.assertEqual(result["maximum_effective_lookahead"], len(values) - 1)
                self.assertEqual(result["normalization_scope"], "full_evaluation_per_channel_score_minmax")
                for name, scores in (("MWVAR96", variance), (component_name, difference)):
                    self.assertIn(name, result["component_score_ranges"])
                    ranges = result["component_score_ranges"][name]
                    numpy.testing.assert_allclose(ranges["minimum"], scores.min(axis=0))
                    numpy.testing.assert_allclose(ranges["maximum"], scores.max(axis=0))

    def test_zero_range_rejects_undefined_official_minmax_without_silent_epsilon(self):
        for scorer in (score_mwvar96_sqdiff_last3, score_mwvar96_sqdiff_centered5):
            with self.subTest(scorer=scorer.__name__):
                with self.assertRaisesRegex(ValueError, "zero score range"):
                    scorer(numpy.ones((100, 2)))


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

    def test_checkpoint_roundtrip_restores_scores_without_refitting(self):
        import torch

        model = PcaLegacy(n_components=0.5).fit(self.fit_values)
        test_values = numpy.random.default_rng(11).normal(size=(105, 3))
        expected = model.score(test_values)
        checkpoint = model.checkpoint()
        model.scaler.mean_[:] += 1000
        model.pca.components_[:] = 0
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location="cpu", weights_only=False)

        with patch.object(StandardScaler, "fit", side_effect=AssertionError("refit")), \
                patch.object(PCA, "fit", side_effect=AssertionError("refit")):
            restored = PcaLegacy.from_checkpoint(saved)
            actual = restored.score(test_values)

        numpy.testing.assert_array_equal(actual.pop("scores"), expected.pop("scores"))
        self.assertEqual(actual, expected)
        self.assertIsNot(restored.scaler, saved["scaler"])
        self.assertIsNot(restored.pca, saved["pca"])
        numpy.testing.assert_array_equal(restored.pca.mean_, saved["pca"].mean_)

        saved["model_config"]["window"] = PCA_WINDOW + 1
        with self.assertRaisesRegex(ValueError, "window"):
            PcaLegacy.from_checkpoint(saved)

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
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.checkpoint()

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


class TestPcaOfficial(unittest.TestCase):
    def test_matches_official_full_evaluation_fit_score_for_both_input_scopes(self):
        generator = numpy.random.default_rng(53)
        for channel_count in (1, 3):
            for component in PCA_COMPONENTS:
                with self.subTest(channel_count=channel_count, component=component):
                    values = generator.normal(size=(115, channel_count))
                    original = values.copy()
                    windows = sliding_window_view(values, window_shape=100, axis=0).reshape(16, -1)
                    axis, degrees = (0, 0) if channel_count == 1 else (1, 1)
                    normalized = numpy.nan_to_num(
                        (windows - windows.mean(axis=axis, keepdims=True))
                        / windows.std(axis=axis, ddof=degrees, keepdims=True),
                    )
                    scaler = StandardScaler().fit(normalized)
                    standardized = scaler.transform(normalized)
                    nonzero = numpy.any(standardized != 0, axis=0)
                    fitted = standardized[:, nonzero]
                    pca = PCA(n_components=component, random_state=0).fit(fitted)
                    expected = numpy.sum(cdist(fitted, pca.components_) / pca.explained_variance_ratio_, axis=1)
                    expected = numpy.pad(expected, (50, 49), mode="edge")

                    with patch(
                        "src.models.tier1.pca_legacy.PCA_DISTANCE_CHUNK_ROWS", 3, create=True,
                    ), patch("src.models.tier1.pca_legacy.cdist", wraps=cdist) as distances:
                        result = score_pca_official(values, n_components=component)

                    numpy.testing.assert_allclose(result["scores"], expected)
                    self.assertEqual([len(call.args[0]) for call in distances.call_args_list],
                                     [3, 3, 3, 3, 3, 1])
                    numpy.testing.assert_array_equal(values, original)
                    self.assertEqual(result["fit_source"], "full_evaluation")
                    self.assertEqual(result["actual_fit_row_count"], len(values))
                    self.assertEqual(result["fit_source_range"], [0, len(values)])
                    self.assertEqual(result["normalization_scope"], "full_evaluation_window_standardscaler")
                    self.assertEqual(result["maximum_effective_lookahead"], len(values) - 1)
                    self.assertEqual(result["calibration_mode"], "none")
                    self.assertTrue(result["official_procedure"])
                    self.assertTrue(result["zero_pruning"])
                    self.assertEqual(result["pca_solver"]["initial"], pca._fit_svd_solver)
                    self.assertEqual(result["pca_solver"]["actual"], pca._fit_svd_solver)
                    checkpoint = result["checkpoint"]
                    numpy.testing.assert_allclose(checkpoint["scaler"].mean_, scaler.mean_)
                    numpy.testing.assert_array_equal(checkpoint["nonzero_window_features"], nonzero)
                    self.assertTrue(checkpoint["model_config"]["official_procedure"])
                    with self.assertRaisesRegex(ValueError, "official"):
                        PcaLegacy.from_checkpoint(checkpoint)

    def test_zero_covariance_weight_retries_full_svd_without_changing_score_formula(self):
        fit_inputs = []

        class ZeroCovariancePCA(PCA):
            def fit(self, values, y=None):
                fit_inputs.append(values.copy())
                super().fit(values, y)
                if self.svd_solver == "auto":
                    self._fit_svd_solver = "covariance_eigh"
                    self.explained_variance_ratio_[-1] = 0
                return self

        values = numpy.random.default_rng(54).normal(size=(115, 3))
        with patch("src.models.tier1.pca_legacy.PCA", ZeroCovariancePCA):
            result = score_pca_official(values)
        checkpoint = result["checkpoint"]
        self.assertEqual(len(fit_inputs), 2)
        numpy.testing.assert_array_equal(fit_inputs[0], fit_inputs[1])
        fitted = fit_inputs[0]
        reference = PCA(n_components=None, random_state=0, svd_solver="full").fit(fitted)
        expected = numpy.sum(cdist(fitted, reference.components_) / reference.explained_variance_ratio_, axis=1)
        numpy.testing.assert_allclose(result["scores"], numpy.pad(expected, (50, 49), mode="edge"))
        self.assertEqual(checkpoint["pca"].n_components_, reference.n_components_)
        self.assertEqual(result["pca_solver"], {
            "requested": "auto", "initial": "covariance_eigh", "actual": "full",
            "initial_zero_weight_count": 1,
        })
        self.assertEqual(checkpoint["pca_solver"], result["pca_solver"])

    def test_full_svd_with_zero_weight_still_rejects_undefined_score(self):
        class ZeroWeightPCA(PCA):
            def fit(self, values, y=None):
                super().fit(values, y)
                if self.svd_solver == "auto":
                    self._fit_svd_solver = "covariance_eigh"
                self.explained_variance_ratio_[-1] = 0
                return self

        values = numpy.random.default_rng(55).normal(size=(115, 3))
        with patch("src.models.tier1.pca_legacy.PCA", ZeroWeightPCA):
            with self.assertRaisesRegex(ValueError, "finite and nonzero"):
                score_pca_official(values)

    def test_official_zero_pruning_removes_zero_window_columns(self):
        values = numpy.column_stack((numpy.zeros(106), numpy.tile([-1.0, 1.0], 53)))
        result = score_pca_official(values, n_components=0.5)
        self.assertEqual(result["zero_pruned_window_feature_count"], 100)
        self.assertEqual(result["retained_window_feature_count"], 100)
        self.assertEqual(result["checkpoint"]["pca"].n_features_in_, 100)
        self.assertEqual(result["checkpoint"]["scaler"].n_features_in_, 200)

    def test_official_procedure_rejects_undefined_pruning_and_unofficial_switches(self):
        with self.assertRaisesRegex(ValueError, "zero-pruned"):
            score_pca_official(numpy.ones((110, 2)), n_components=0.5)
        with self.assertRaisesRegex(ValueError, "zero_pruning"):
            score_pca_official(numpy.ones((110, 2)), n_components=0.5, zero_pruning=False)
        with self.assertRaisesRegex(ValueError, "n_components"):
            score_pca_official(numpy.ones((110, 2)), n_components=0.4)


if __name__ == "__main__":
    unittest.main()
