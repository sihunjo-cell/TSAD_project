"""관측하지 않은 규모와 불완전한 계획 경로를 추천 근거로 쓰지 않는다."""

import hashlib
import json
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from src.common.equal_trial_budget import _canonical_bytes, registry_space_sha256
from src.common.select_conditional_policy import recommend_conditional_candidates
from src.common.tuning_support import summarize_tuning_support


class ServiceSupportTests(unittest.TestCase):
    def test_finish_receipt_binds_both_support_and_selection_json(self):
        from tests.ghl_main import run_ratio_tuning as runner

        budget = {"experiment_mode": "full_prefix_v2", "budget_id": "budget",
                  "selection_rule_id": "rule", "series_ids": [], "input_manifest_sha256": "input",
                  "model_panels": [], "structural_exclusions": []}
        ledger = [{"evaluator_sha256": "evaluator", "ell_max_id": "ell"}]
        manifest = [{"primary_score": "true"}]
        selection = {field: [] for field in ("model_ratio", "tier_adaptive", "adaptive_lofo",
                                            "candidate_audit", "model_comparison")}
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root / "dev18_trial_score_ledger.csv").write_text("fixture ledger\n", encoding="utf-8")
            evidence = MagicMock()
            evidence.__enter__.return_value = evidence
            evidence.finalize.return_value = {"status": "complete"}
            stack.enter_context(patch.object(runner, "open_recommendation_evidence", return_value=evidence))
            for name in ("_require_clean_worktree", "_require_same_worktree", "_validate_primary_manifest_rows"):
                stack.enter_context(patch.object(runner.tuning, name))
            for name, value in (
                ("_git_head", "commit"), ("validate_vus_evidence", {"evaluator_sha256": "evaluator"}),
                ("_validate_ell_max", {"ell_max_id": "ell"}), ("load_trial_score_ledger", ledger),
                ("_reuse_trial_scores", {"score": ledger[0]}), ("build_conditional_membership_rows", []),
            ):
                stack.enter_context(patch.object(runner.tuning, name, return_value=value))
            stack.enter_context(patch.object(runner, "select_ratio_tuning_policies", return_value=selection))
            stack.enter_context(patch("src.common.load_final_membership.load_final_membership"))
            builder = stack.enter_context(patch("tests.ghl_main.build_tuning_support.build_tuning_support",
                                                return_value={"points": [], "service_status": "unvalidated"}))
            receipt = runner.finish_ratio_tuning({}, budget, manifest, [],
                                                 {"result": root, "checkpoint": root / "cache",
                                                  "manifest": root / "manifest.csv",
                                                  "recommendation": root / "recommendation_evidence"},
                                                 data_root=root, selection_only=True, write_plots=False)
            builder.assert_called_once_with(selection, {}, budget, ledger, manifest)
            for name in ("tuning_support.json", "conditional_selection.json"):
                self.assertEqual(receipt["outputs"][name], hashlib.sha256((root / name).read_bytes()).hexdigest())

    def test_observed_points_and_seals_control_candidates_and_plan_coverage(self):
        config = "c" + "1" * 12
        registry = {"selection": {}, "models": {"GDN": {
            "execution_status": "ready", "target_use": "fit_full_prefix",
            "candidates": [{"config_id": config, "hyperparameters": {"window": 5, "rho": .5}}],
        }}}
        shape = {"available_count": 100, "feature_count": 2, "training_boundary": 2000}
        policy = {"model": "GDN", "ratio": 5, "group_id": "group-five",
                  "candidate_ids": [config], "config_id": config, "score_variant": "",
                  "selection_status": "selected", "budget_id": "budget", "support": [shape]}
        selection = {"model_ratio": [policy]}
        point = {**shape, "model": "GDN", "ratio": 5, "group_id": "group-five",
                 "config_id": config, "score_variant": "", "series": "01",
                 "evaluation_rows": 100, "environment_id": "environment",
                 "execution_ids": ["execution"], "family_count": 2,
                 "validation_status": "family_lofo"}
        report = {"schema_version": 1, "budget_id": "budget",
                  "registry_space_sha256": registry_space_sha256(registry),
                  "selection_sha256": hashlib.sha256(_canonical_bytes(selection)).hexdigest(),
                  "observation_status": "complete", "service_status": "unvalidated",
                  "points": [point], "executions": {"execution": {"environment_id": "environment"}}}
        arguments = {"nrows": 100, "dfeatures": 2, "planned_rows": 2000,
                     "evaluation_rows": 100, "environment_id": "environment"}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tuning_support.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            arguments.update(support_path=path, support_sha256=digest)
            result = recommend_conditional_candidates(selection, registry, **arguments)
            self.assertEqual(result["status"], "candidates_for_validation")
            self.assertEqual(result["service_status"], "unvalidated")
            self.assertEqual(result["plan_observation_status"], "incomplete_grid")
            self.assertEqual(result["missing_plan_ratios"], [10, 20, 40, 60, 80, 100])
            self.assertFalse(result["profitability_optimal"])
            for update, reason in (
                ({"nrows": 1000, "planned_rows": 20000}, "joint_shape_not_observed"),
                ({"dfeatures": 3}, "joint_shape_not_observed"),
                ({"evaluation_rows": 101}, "joint_shape_not_observed"),
                ({"environment_id": "another"}, "joint_shape_not_observed"),
                ({"nrows": 150}, "registered_prefix_not_observed"),
                ({"support_path": None}, "tuning_support_required"),
                ({"evaluation_rows": None}, "evaluation_shape_and_environment_required"),
                ({"environment_id": None}, "evaluation_shape_and_environment_required"),
            ):
                result = recommend_conditional_candidates(selection, registry, **{**arguments, **update})
                self.assertEqual(result["status"], "unavailable")
                self.assertIn(reason, result["reasons"])
            for update in ({"support_sha256": "0" * 64},):
                with self.assertRaises(ValueError):
                    recommend_conditional_candidates(selection, registry, **{**arguments, **update})
            changed = deepcopy(selection)
            changed["model_ratio"][0]["budget_id"] = "other"
            with self.assertRaises(ValueError):
                recommend_conditional_candidates(changed, registry, **arguments)
            changed_registry = deepcopy(registry)
            changed_registry["models"]["GDN"]["candidates"][0]["hyperparameters"]["window"] = 6
            with self.assertRaises(ValueError):
                recommend_conditional_candidates(selection, changed_registry, **arguments)

    def test_caps_preserve_joint_context_and_missing_grid_points(self):
        points = [
            {"model": model, "series": "01", "ratio": ratio,
             "available_count": planned * ratio // 100, "training_boundary": planned,
             "feature_count": features, "evaluation_rows": 100, "environment_id": "environment",
             "config_id": "early" if ratio == 5 else "late", "score_variant": ""}
            for model, planned, features, ratios in (
                ("GDN", 2000, 2, [5, 10, 20, 40, 60, 80, 100]),
                ("PaAno", 4000, 4, [40, 60, 100]),
            ) for ratio in ratios
        ]
        summaries = summarize_tuning_support(points)
        first = next(row for row in summaries if row["model"] == "GDN")
        second = next(row for row in summaries if row["model"] == "PaAno")
        self.assertEqual(first["complete_from_ratio"], 5)
        self.assertEqual(second["complete_from_ratio"], 100)
        self.assertEqual(second["missing_ratios"], [5, 10, 20, 80])
        self.assertEqual(second["observed_max_planned_rows_at_same_columns"], 4000)
        self.assertEqual({(row["training_boundary"], row["feature_count"]) for row in summaries},
                         {(2000, 2), (4000, 4)})


if __name__ == "__main__":
    unittest.main()
