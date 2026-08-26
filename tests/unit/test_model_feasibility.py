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

    def test_alora_requires_at_least_one_pair_per_attention_head(self):
        params = {"window": 20, "heads": 8, "max_pairs": 512}
        infeasible = assess_candidate(
            "ALoRa", params, fit_length=20, validation_length=20,
            test_length=20, channel_count=4,
        )
        feasible = assess_candidate(
            "ALoRa", params, fit_length=20, validation_length=20,
            test_length=20, channel_count=5,
        )
        self.assertEqual(infeasible["status"], "structurally_infeasible")
        self.assertIn("pair count 6 < heads 8", infeasible["reason"])
        self.assertEqual(feasible["status"], "feasible")
        self.assertEqual(feasible["derived"]["pair_count"], 10)

    def test_training_free_and_zero_shot_ignore_fit_length(self):
        mwvar = assess_candidate("MWVAR", {"window": 96}, 0, 0, 96, 19)
        time_rcd = assess_candidate(
            "TimeRCD", {"context_length": 5000}, 0, 0, 1, 19,
        )
        self.assertEqual(mwvar["status"], "feasible")
        self.assertEqual(time_rcd["status"], "feasible")

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
            "model": "ALoRa", "config_id": "c111111111111", "ratio": 20,
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
