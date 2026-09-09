"""고정된 작은 입력으로 새 예산의 신원과 지원 집단을 검증한다."""

import hashlib
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

from src.common.equal_trial_budget import _canonical_bytes, _score_variants
from src.common.model_registry import validate_primary_hpo_seal
from tests.ghl_main.build_ratio_tuning_budget import build_full_prefix_budget
from tests.ghl_main.select_ratio_tuning import select_ratio_tuning_policies
from tests.unit.test_ratio_tuning_full_prefix import make_inputs, make_scores


class TestRatioTuningBudget(unittest.TestCase):
    def setUp(self):
        self.registry, self.entries, self.rows, self.summary = make_inputs()

    def build(self):
        return build_full_prefix_budget(self.registry, self.rows, self.entries, self.summary)

    def test_identity_is_stable_and_binds_support_and_source(self):
        original = deepcopy((self.registry, self.entries, self.rows, self.summary))
        budget = self.build()
        self.assertEqual(budget, self.build())
        self.assertEqual(original, (self.registry, self.entries, self.rows, self.summary))
        core = {key: value for key, value in budget.items() if key not in {"budget_id", "budget_sha256"}}
        digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
        self.assertEqual(budget["budget_sha256"], digest)
        self.assertEqual(budget["budget_id"], "b" + digest[:12])
        self.registry["models"]["PaAno"]["source_commit"] = "f" * 40
        self.assertNotEqual(budget["budget_id"], self.build()["budget_id"])

    def test_candidate_groups_partition_original_panel_and_include_unavailable(self):
        budget = self.build()
        for panel in budget["model_panels"]:
            for ratio in (5, 10, 20, 40, 60, 80, 100):
                groups = [group for group in panel["groups"] if group["ratio"] == ratio]
                members = [series for group in groups for series in group["series_ids"]]
                self.assertEqual(sorted(members), budget["series_ids"])
                self.assertEqual(len(members), len(set(members)))
        gdn = next(panel for panel in budget["model_panels"] if panel["model"] == "GDN")
        unavailable = next(group for group in gdn["groups"]
                           if group["ratio"] == 5 and not group["candidate_ids"])
        self.assertEqual(unavailable["series_ids"], [f"{index:02d}" for index in range(1, 7)])
        self.assertEqual(unavailable["feature_count_max"], 2)

    def test_changed_panel_and_unknown_or_duplicate_feasibility_are_rejected(self):
        for invalid in (self.rows + [self.rows[0]],
                        self.rows[:-1] + [{**self.rows[-1], "config_id": "unknown"}]):
            with self.assertRaises(ValueError):
                build_full_prefix_budget(self.registry, invalid, self.entries, self.summary)
        with self.assertRaises(ValueError):
            build_full_prefix_budget(self.registry, self.rows, self.entries[::-1], self.summary)
        self.registry["seeds"]["development"] = [1]
        with self.assertRaises(ValueError):
            self.build()

    def test_score_variant_seal_distinguishes_full_prefix_and_legacy(self):
        heads = ["time", "fft", "pred"]
        selection = {
            "primary_hpo_regime": "full_prefix_per_ratio",
            "primary_score_variants": {"TSPulse": heads},
            "diagnostic_score_variants": {"TSPulse": ["raw_max"]},
            "tspulse_prediction_aggregation_window": 96,
        }
        self.assertEqual(_score_variants(selection, "TSPulse"), (heads, ["raw_max"]))
        for window in (64, 96, 128):
            self.assertEqual(
                _score_variants(selection, "TSPulse", {"aggregation_window": window}),
                (heads, ["raw_max"]) if window == 96 else (["time", "fft"], ["pred", "raw_max"]),
            )
        for parameters in ({}, {"aggregation_window": True}, {"aggregation_window": 65}):
            with self.assertRaises(ValueError):
                _score_variants(selection, "TSPulse", parameters)
        with self.assertRaises(ValueError):
            _score_variants({**selection, "tspulse_prediction_aggregation_window": 64}, "TSPulse")
        for primary, diagnostic in (
            (heads[:-1], []), (heads + ["time"], []), (heads, ["time"]),
            (heads + ["unknown"], []), (["raw_max"], ["time", "fft", "pred"]),
        ):
            with self.subTest(primary=primary, diagnostic=diagnostic), self.assertRaises(ValueError):
                _score_variants({**selection, "primary_score_variants": {"TSPulse": primary},
                                 "diagnostic_score_variants": {"TSPulse": diagnostic}}, "TSPulse")
        for regime in ("equal_trial", "runtime_matched"):
            legacy = {**selection, "primary_hpo_regime": regime,
                      "primary_score_variants": {"TSPulse": ["raw_max"]},
                      "diagnostic_score_variants": {"TSPulse": ["time", "fft", "pred"]}}
            self.assertEqual(_score_variants(legacy, "TSPulse"), (["raw_max"], ["time", "fft", "pred"]))
            with self.assertRaises(ValueError):
                _score_variants({**selection, "primary_hpo_regime": regime}, "TSPulse")

    def test_paper_keeps_all_tspulse_heads_at_each_aggregation(self):
        heads = ["time", "fft", "pred", "ensemble"]
        selection = {"primary_hpo_regime": "full_prefix_per_ratio",
                     "primary_score_variants": {"TSPulse": heads}, "diagnostic_score_variants": {}}
        for window in (64, 96, 128):
            self.assertEqual(_score_variants(selection, "TSPulse", {"aggregation_window": window}),
                             (heads, []))
        with self.assertRaises(ValueError):
            _score_variants({**selection, "tspulse_prediction_aggregation_window": 96}, "TSPulse")
        for window in (True, 65, None):
            with self.assertRaises(ValueError):
                _score_variants(selection, "TSPulse", {"aggregation_window": window})

    def test_training_count_keeps_each_config_ratio_seed_and_series_independent(self):
        budget = self.build()
        training = [row for row in budget["execution_panel"] if row["tier"] == "t2"]
        self.assertEqual(budget["training_run_count"], sum(len(row["series_ids"]) for row in training))
        gdn = [row for row in training if row["model"] == "GDN"
               and row["physical_ratio"] == 5 and row["seed"] == 0]
        self.assertEqual(len({row["config_id"] for row in gdn}), 2)
        self.assertTrue(all("training_group_id" not in row for row in training))

    def test_tspulse_heads_share_inference_and_single_head_can_win(self):
        self.registry["selection"] = {
            "primary_hpo_regime": "full_prefix_per_ratio",
            "primary_score_variants": {"TSPulse": ["time", "fft", "pred"]},
            "diagnostic_score_variants": {"TSPulse": ["raw_max"]},
            "tspulse_prediction_aggregation_window": 96,
        }
        self.registry["models"] = {"TSPulse": {
            **self.registry["models"]["TimeRCD"], "execution_status": "ready",
            "candidates": [{"config_id": f"c{window:012x}",
                            "hyperparameters": {"aggregation_window": window}}
                           for window in (64, 96, 128)],
        }}
        self.rows = [{**row, "model": "TSPulse", "config_id": candidate["config_id"]}
                     for row in self.rows if row["model"] == "TimeRCD"
                     for candidate in self.registry["models"]["TSPulse"]["candidates"]]
        budget = self.build()
        validate_primary_hpo_seal(self.registry, budget=budget)
        self.assertEqual(budget["physical_execution_count"], 3)
        self.assertEqual(budget["physical_run_count"], 18 * 3)
        self.assertEqual(budget["expected_ledger_rows"], 18 * 7 * 7)
        for execution in budget["execution_panel"]:
            self.assertEqual(execution["physical_ratio"], 100)
            self.assertEqual(execution["logical_ratios"], [5, 10, 20, 40, 60, 80, 100])
            self.assertEqual("pred" in execution["primary_score_variants"],
                             execution["config_id"] == "c000000000060")
            self.assertIn("raw_max", execution["diagnostic_score_variants"])
        scores = make_scores(self.registry, self.entries, budget)
        for row in scores:
            row["vus_pr"] = .9 if row["score_variant"] == "time" and row["config_id"] == "c000000000060" else .2
        selection = select_ratio_tuning_policies(scores, self.registry, budget, "e" * 64)
        self.assertEqual(len(selection["model_ratio"]), 7)
        self.assertEqual(len(selection["tier_adaptive"]), 7)
        for policy in selection["model_ratio"] + selection["tier_adaptive"]:
            self.assertEqual((policy["config_id"], policy["score_variant"]), ("c000000000060", "time"))
        from tests.ghl_main import run_ratio_tuning as runner

        figure, axis = MagicMock(), MagicMock()
        with patch.object(runner.tuning, "_write_csv"), patch.object(
            runner.tuning.pyplot, "subplots", return_value=(figure, axis),
        ), patch.object(runner.tuning.pyplot, "close"):
            runner.write_full_prefix_reports({**selection, "tier_adaptive": []}, budget, "unused")
        self.assertEqual(axis.plot.call_count, 7)
        self.assertEqual({call.kwargs["label"] for call in axis.plot.call_args_list}, {
            f"{candidate['config_id']} / {variant} / 18 files"
            for candidate in self.registry["models"]["TSPulse"]["candidates"]
            for variant in (("time", "fft", "pred")
                            if candidate["hyperparameters"]["aggregation_window"] == 96 else ("time", "fft"))
        })
        self.registry["selection"].update(
            primary_hpo_regime="equal_trial", primary_score_variants={"TSPulse": ["raw_max"]},
            diagnostic_score_variants={"TSPulse": ["time", "fft", "pred"]},
        )
        legacy_variants = self.build()
        self.assertNotEqual(budget["budget_id"], legacy_variants["budget_id"])
        self.assertEqual(legacy_variants["physical_run_count"], 18 * 3)
        self.assertEqual(legacy_variants["expected_ledger_rows"], 18 * 3 * 7)
        with self.assertRaises(ValueError):
            validate_primary_hpo_seal(self.registry, budget=budget)


if __name__ == "__main__":
    unittest.main()
