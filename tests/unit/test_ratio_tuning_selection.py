"""비율별 설정 선택과 family holdout을 작은 원표로 검증한다."""

import csv
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.ghl_main import run_dev18_tuning as tuning
from tests.ghl_main.select_ratio_tuning import select_ratio_tuning_policies


class TestRatioTuningSelection(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "selection": {"tier_q_floor": {"t2": 10}},
            "models": {"PaAno": {
                "tier": "t2", "target_use": "fit_validation",
                "source_commit": "a" * 40, "source_checkpoint_sha256": "none",
                "candidates": [
                    {"config_id": "small", "hyperparameters": {"patch_size": 32}},
                    {"config_id": "large", "hyperparameters": {"patch_size": 96}},
                ],
            }},
        }
        legacy_budget = {
            "budget_id": "legacy", "primary_hpo_regime": "equal_trial",
            "tie_rule": {"tolerance": 1e-6},
            "model_panels": [{
                "model": "PaAno", "tier": "t2", "logical_ratios": [10, 40],
                "selected_config_ids": ["small"], "seeds": [0],
                "primary_score_variants": [""],
                "dev18_tier_representative_eligible": True,
            }],
        }
        self.budget = deepcopy(legacy_budget)
        self.budget.update(budget_id="ratio", legacy_budget=legacy_budget)
        self.budget["model_panels"][0].update(
            selected_config_ids=["small", "large"],
            selected_config_ids_by_ratio={"10": ["small"], "40": ["small", "large"]},
        )
        self.rows = [{
            "series": series, "family": family, "tier": "t2", "model": "PaAno",
            "config_id": config_id, "ratio": ratio, "seed": 0,
            "score_variant": "", "vus_pr": score, "status": "complete",
        } for series, family in (("01", "A"), ("02", "B"), ("03", "C"))
            for config_id, ratio, score in (
                ("small", 10, 0.8), ("small", 40, 0.2), ("large", 40, 0.9),
            )]

    def select(self):
        return select_ratio_tuning_policies(
            self.rows, self.registry, self.budget, evaluator_sha256="d" * 64,
        )

    def test_later_ratio_uses_config_unavailable_at_earlier_ratio(self):
        selection = self.select()
        policies = {row["ratio"]: row for row in selection["model_ratio"]}
        self.assertEqual(policies[10]["config_id"], "small")
        self.assertEqual(policies[40]["config_id"], "large")
        self.assertEqual(policies[40]["hyperparameters"], {"patch_size": 96})
        self.assertAlmostEqual(policies[40]["family_macro_vus_pr"], 0.9)
        adaptive = {row["ratio"]: row for row in selection["tier_adaptive"]}
        self.assertEqual(adaptive[5]["selection_status"], "unavailable")
        unavailable = next(row for row in selection["candidate_audit"] if row["ratio"] == 5)
        self.assertEqual(unavailable["eligibility"], "ratio_unsupported")
        self.assertEqual(adaptive[40]["config_id"], "large")
        self.assertAlmostEqual(adaptive[40]["selection_score"], 0.9)
        self.assertEqual(selection["model_fixed"][0]["config_id"], "small")
        self.assertAlmostEqual(selection["model_fixed"][0]["j_fixed"], 0.5)
        transition = next(row for row in selection["policy_transitions"]
                          if row["current_ratio"] == 40)
        self.assertEqual(transition["transition"], "switch")
        self.assertEqual(transition["previous_ratio"], 10)
        self.assertEqual(transition["previous_config_id"], "small")

    def test_holdout_family_cannot_choose_its_own_recipe(self):
        for row in self.rows:
            if row["family"] == "C" and row["ratio"] == 40:
                row["vus_pr"] = 0.99 if row["config_id"] == "small" else 0.1
        selection = self.select()
        folds = {row["holdout_family"]: row for row in selection["adaptive_lofo"]
                 if row["ratio"] == 40}
        self.assertEqual(folds["C"]["selected_config_id"], "large")
        self.assertAlmostEqual(folds["C"]["holdout_vus_pr"], 0.1)
        self.assertEqual(folds["A"]["selected_config_id"], "small")
        adaptive = next(row for row in selection["tier_adaptive"] if row["ratio"] == 40)
        self.assertEqual(adaptive["config_id"], "large")
        self.assertAlmostEqual(adaptive["selection_score"], 1 / 6)

class TestRatioTuningReports(unittest.TestCase):
    def test_conditional_report_keeps_separate_panels_and_ratio_winners(self):
        from tests.ghl_main.build_ratio_tuning_budget import build_full_prefix_budget
        from tests.ghl_main.run_ratio_tuning import write_full_prefix_reports
        from tests.unit.test_ratio_tuning_full_prefix import make_inputs, make_scores

        registry, entries, feasibility, summary = make_inputs()
        budget = build_full_prefix_budget(registry, feasibility, entries, summary)
        selection = select_ratio_tuning_policies(
            make_scores(registry, entries, budget), registry, budget, "d" * 64,
        )
        with TemporaryDirectory() as directory:
            output = Path(directory)
            support = {"points": [{"model": "PaAno", "ratio": 100, "series": "01",
                                   "available_count": 1000, "training_boundary": 1000,
                                   "feature_count": 2, "evaluation_rows": 200,
                                   "environment_id": "environment"}]}
            with patch.object(tuning.pyplot, "subplots", side_effect=AssertionError("plot disabled")):
                write_full_prefix_reports(selection, budget, output, write_plots=False, support_report=support)
            self.assertEqual(json.loads((output / "conditional_selection.json").read_text(encoding="utf-8")), selection)
            evidence = json.loads((output / "tuning_support.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["points"], support["points"])
            self.assertEqual(evidence["model_limits"][0]["complete_from_ratio"], 100)
            self.assertTrue((output / "model_support_limits.csv").exists())
            with (output / "model_ratio_policy.csv").open(encoding="utf-8", newline="") as source:
                policies = list(csv.DictReader(source))
            early = [row for row in policies if row["model"] == "PaAno" and int(row["ratio"]) == 5]
            self.assertEqual(sorted(len(json.loads(row["series_ids"])) for row in early), [6, 12])
            policies = {int(row["ratio"]): row for row in policies
                        if row["model"] == "PaAno" and int(row["ratio"]) in (60, 80)}
            self.assertNotEqual(policies[60]["config_id"], policies[80]["config_id"])
            with (output / "PaAno.csv").open(encoding="utf-8", newline="") as source:
                model_rows = list(csv.DictReader(source))
            self.assertTrue(all(row["group_id"] and row["series_ids"] for row in model_rows))
            self.assertTrue((output / "matched_model_comparison.csv").exists())
            self.assertTrue((output / "Tier2.csv").exists())
            self.assertFalse((output / "tier_fixed_policy.csv").exists())
            self.assertFalse(list(output.glob("*.png")))


if __name__ == "__main__":
    unittest.main()
