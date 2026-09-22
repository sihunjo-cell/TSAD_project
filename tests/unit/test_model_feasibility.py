"""점수나 라벨을 보지 않는 모델별 정적 실행 가능성 판정을 검증한다."""

import unittest

from src.common import model_feasibility
from src.common.model_feasibility import assess_candidate


collect_fully_feasible_keys = getattr(
    model_feasibility, "collect_fully_feasible_keys", None,
)


class TestAssessCandidate(unittest.TestCase):
    def test_paano_requires_memory_and_one_full_pretext_step(self):
        params = {"patch_size": 32, "memory_fraction": 0.1, "neighbors": 3}
        memory_failed = assess_candidate(
            "PaAno", params, fit_length=60, validation_length=40,
            test_length=100, channel_count=19,
        )
        pretext_failed = assess_candidate(
            "PaAno", params, fit_length=63, validation_length=40,
            test_length=100, channel_count=19,
        )
        passed = assess_candidate(
            "PaAno", params, fit_length=64, validation_length=40,
            test_length=100, channel_count=19,
        )
        self.assertEqual(memory_failed["status"], "structurally_infeasible")
        self.assertIn("memory", memory_failed["reason"])
        self.assertEqual(pretext_failed["status"], "structurally_infeasible")
        self.assertIn("pretext", pretext_failed["reason"])
        self.assertEqual(passed["status"], "feasible")

    def test_gdn_checks_window_and_channel_topk_constraint(self):
        params = {"window": 5, "rho": 0.3}
        self.assertEqual(assess_candidate(
            "GDN", params, 6, 6, 6, 1,
        )["status"], "structurally_infeasible")
        feasible = assess_candidate("GDN", params, 6, 6, 6, 19)
        self.assertEqual(feasible["status"], "feasible")
        self.assertEqual(feasible["derived"]["topk"], 5)

    def test_gdn_preserves_fixed_topk_and_keeps_low_channel_alternative(self):
        official = assess_candidate("GDN", {"window": 5, "topk": 5}, 6, 6, 6, 3)
        self.assertEqual(official["status"], "structurally_infeasible")
        self.assertEqual(official["derived"]["topk"], 5)
        self.assertIn("topk 5 > channel count 3", official["reason"])
        for channel_count in (2, 3):
            connected = assess_candidate(
                "GDN", {"window": 5, "topk": 2}, 6, 6, 6, channel_count,
            )
            legacy = assess_candidate(
                "GDN", {"window": 5, "rho": 0.3}, 6, 6, 6, channel_count,
            )
            self.assertEqual(connected["status"], "feasible")
            self.assertEqual(connected["derived"]["topk"], 2)
            self.assertEqual(legacy["derived"]["topk"], 1)

    def test_training_free_and_zero_shot_ignore_fit_length(self):
        mwvar = assess_candidate("MWVAR", {"window": 96}, 0, 0, 96, 19)
        time_rcd = assess_candidate(
            "TimeRCD", {"context_length": 5000}, 0, 0, 1, 19,
        )
        self.assertEqual(mwvar["status"], "feasible")
        self.assertEqual(time_rcd["status"], "feasible")

    def test_mwvar_minimum_length_follows_each_registered_window(self):
        for window in (96, 64):
            with self.subTest(window=window):
                rejected = assess_candidate("MWVAR", {"window": window}, 0, 0, window - 1, 2)
                accepted = assess_candidate("MWVAR", {"window": window}, 0, 0, window, 2)
                self.assertEqual(rejected["status"], "structurally_infeasible")
                self.assertEqual(accepted["status"], "feasible")
                self.assertEqual(accepted["derived"]["native_continuous_score_count"], 1)

    def test_time_rcd_multi_checkpoint_requires_multivariate_input(self):
        result = assess_candidate(
            "TimeRCD",
            {"context_length": 5000, "checkpoint_variant": "multi"},
            fit_length=0, validation_length=0, test_length=1,
            channel_count=1,
        )
        self.assertEqual(result["status"], "structurally_infeasible")
        self.assertIn("multi", result["reason"])

    def test_tspulse_requires_three_context_lengths(self):
        params = {"context_length": 512, "aggregation_window": 96}
        rejected = assess_candidate("TSPulse", params, 0, 0, 1535, 19)
        accepted = assess_candidate("TSPulse", params, 0, 0, 1536, 19)
        self.assertEqual(rejected["status"], "structurally_infeasible")
        self.assertEqual(accepted["status"], "feasible")

    def test_pca_requires_two_fit_windows_and_one_validation_window(self):
        params = {"window": 100}
        self.assertEqual(assess_candidate(
            "PCA_LEGACY", params, 100, 100, 100, 3,
        )["status"], "structurally_infeasible")
        self.assertEqual(assess_candidate(
            "PCA_LEGACY", params, 101, 100, 100, 3,
        )["status"], "feasible")

    def test_paper_target_free_boundaries_and_legacy_tspulse_are_distinct(self):
        cases = [("SQDIFF_LAST1", {"window": 2}, 2),
                 ("SQDIFF_CENTERED5", {"window": 5}, 5),
                 ("MWVAR96_SQDIFF_LAST3", {"variance_window": 96}, 97),
                 ("MWVAR96_SQDIFF_CENTERED5", {"variance_window": 96}, 97),
                 ("PCA_LEGACY", {"window": 100}, 101)]
        cases.extend(("TSPulse", {"context_length": 512, "aggregation_window": window}, 513)
                     for window in (64, 96, 128))
        for model, parameters, minimum in cases:
            with self.subTest(model=model, parameters=parameters):
                self.assertEqual(assess_candidate(model, parameters, 0, 0, minimum - 1, 2,
                    official_protocol=True)["status"], "structurally_infeasible")
                self.assertEqual(assess_candidate(model, parameters, 0, 0, minimum, 2,
                    official_protocol=True)["status"], "feasible")
        self.assertEqual(assess_candidate("TSPulse", cases[-1][1], 0, 0, 513, 2)["status"],
                         "structurally_infeasible")

    def test_paper_gdn_uses_combined_session_window_holdout(self):
        parameters = {"window": 5, "topk": 5, "validation_ratio": .2}
        rejected = assess_candidate("GDN", parameters, 9, 0, 10, 5,
                                    full_prefix=True, official_protocol=True)
        accepted = assess_candidate("GDN", parameters, 17, 0, 10, 5,
            full_prefix=True, official_protocol=True, fit_session_lengths=(8, 9))
        self.assertEqual(rejected["status"], "structurally_infeasible")
        self.assertEqual(accepted["status"], "feasible")
        self.assertEqual(accepted["derived"]["fit_forecast_count"], 7)
        self.assertEqual(accepted["derived"]["internal_validation_window_count"], 1)
        self.assertEqual(accepted["derived"]["internal_training_window_count"], 6)

    def test_paper_paano_rejects_only_reached_singleton_batch(self):
        parameters = {"patch_size": 32, "memory_fraction": .1, "neighbors": 3,
                      "batch_size": 512, "iterations": 100}
        for patches, expected in ((513, "structurally_infeasible"),
                                  (514, "feasible"), (51201, "feasible")):
            result = assess_candidate("PaAno", parameters, patches + 31, 0, 64, 2,
                                      full_prefix=True, official_protocol=True)
            self.assertEqual(result["status"], expected)

    def test_official_paano_memory_keeps_source_minimum_and_fraction(self):
        parameters = {"patch_size": 2, "memory_fraction": .1, "neighbors": 3,
                      "batch_size": 512, "iterations": 100}
        for patches, memory_count in ((3, 2), (4, 3), (25, 24), (501, 500), (1000, 500), (6000, 600)):
            with self.subTest(patches=patches):
                result = assess_candidate("PaAno", parameters, patches + 1, 0, 1, 2,
                                          full_prefix=True, official_protocol=True)
                self.assertEqual(result["derived"]["memory_size"], memory_count)
                self.assertEqual(result["status"], "structurally_infeasible" if memory_count < 3 else "feasible")
                if memory_count < 3:
                    self.assertIn("memory", result["reason"])
        capped = assess_candidate("PaAno", {**parameters, "memory_fraction": 1}, 5, 0, 1, 2,
                                  full_prefix=True, official_protocol=True)
        self.assertEqual(capped["derived"]["memory_size"], 3)

    def test_paper_paano_counts_fitted_context_and_accepts_short_evaluation(self):
        parameters = {"patch_size": 32, "memory_fraction": .1, "neighbors": 3,
                      "batch_size": 512, "iterations": 100}
        for test_length in (1, 31, 32, 100):
            with self.subTest(test_length=test_length):
                result = assess_candidate("PaAno", parameters, 64, 0, test_length, 2,
                                          full_prefix=True, official_protocol=True)
                self.assertEqual(result["status"], "feasible")
                self.assertEqual(result["derived"]["test_patch_count"], test_length)
                self.assertEqual(result["derived"]["native_continuous_score_count"], test_length)
        self.assertEqual(assess_candidate("PaAno", parameters, 64, 0, 0, 2,
            full_prefix=True, official_protocol=True)["status"], "structurally_infeasible")
        legacy = assess_candidate("PaAno", parameters, 64, 0, 31, 2, full_prefix=True)
        self.assertEqual(legacy["status"], "structurally_infeasible")


class TestCollectFullyFeasibleKeys(unittest.TestCase):
    def setUp(self):
        self.required_series = ("series-a", "series-b")
        self.rows = [
            {
                "model": "GDN", "config_id": "c0123456789ab", "ratio": 20,
                "series": series, "status": "feasible",
            }
            for series in self.required_series
        ]

    def test_returns_only_candidates_feasible_for_every_required_series(self):
        self.assertTrue(callable(collect_fully_feasible_keys))
        incomplete = {
            "model": "GDN", "config_id": "c111111111111", "ratio": 20,
            "series": "series-a", "status": "feasible",
        }
        infeasible = [
            {
                "model": "PaAno", "config_id": "c222222222222", "ratio": 20,
                "series": series,
                "status": "structurally_infeasible" if series == "series-b" else "feasible",
            }
            for series in self.required_series
        ]

        result = collect_fully_feasible_keys(
            [*self.rows, incomplete, *infeasible], self.required_series,
        )

        self.assertIsInstance(result, frozenset)
        self.assertEqual(result, frozenset({("GDN", "c0123456789ab", 20)}))

    def test_rejects_duplicate_candidate_series_rows(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            collect_fully_feasible_keys([*self.rows, self.rows[0]], self.required_series)

    def test_rejects_unknown_status_and_series(self):
        mutations = (
            ({**self.rows[0], "status": "runtime_failed"}, "status"),
            ({**self.rows[0], "series": "series-c"}, "series"),
        )
        for row, message in mutations:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, message):
                    collect_fully_feasible_keys([row], self.required_series)

    def test_rejects_malformed_fields_and_ratio(self):
        mutations = (
            ({**self.rows[0], "runtime_seconds": 1.0}, "fields"),
            ({**self.rows[0], "config_id": "not-sealed"}, "config_id"),
            ({**self.rows[0], "ratio": 30}, "ratio"),
        )
        for row, message in mutations:
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, message):
                    collect_fully_feasible_keys([row], self.required_series)

    def test_rejects_empty_duplicate_or_malformed_required_series(self):
        invalid_collections = ((), ("series-a", "series-a"), ("", "series-b"))
        for required_series in invalid_collections:
            with self.subTest(required_series=required_series):
                with self.assertRaisesRegex(ValueError, "required_series"):
                    collect_fully_feasible_keys([], required_series)


if __name__ == "__main__":
    unittest.main()
