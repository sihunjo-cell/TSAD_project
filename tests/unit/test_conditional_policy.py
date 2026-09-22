"""학습 없이 조건부 후보와 기업 입력의 범위 제한을 확인한다."""

import unittest
from copy import deepcopy

from src.common.select_conditional_policy import (
    match_conditional_policy,
    recommend_conditional_candidates,
)


class TestConditionalPolicy(unittest.TestCase):
    def setUp(self):
        self.config_id = "c" + "1" * 12
        self.registry = {"models": {"GDN": {
            "execution_status": "ready", "target_use": "fit_full_prefix",
            "candidates": [{"config_id": self.config_id,
                            "hyperparameters": {"window": 5, "rho": 0.5}}],
        }}}
        self.selection = {"model_ratio": [{
            "model": "GDN", "ratio": 5, "group_id": "group-five",
            "candidate_ids": [self.config_id], "config_id": self.config_id,
            "selection_status": "selected", "hyperparameters": {"window": 5},
            "support": [
                {"available_count": 100, "feature_count": 2, "training_boundary": 2000},
                {"available_count": 200, "feature_count": 4, "training_boundary": 4000},
            ],
        }]}
        self.match_arguments = {
            "model": "GDN", "ratio": 5, "available_count": 150,
            "feature_count": 3, "test_length": 100, "training_boundary": 3000,
        }
        self.intake = {"nrows": 150, "dfeatures": 3, "planned_rows": 3000,
                       "max_rows": 5000, "max_columns": 10}

    def test_matches_candidate_signature_and_actual_size_support(self):
        arguments = {**self.match_arguments, "available_count": 100,
                     "feature_count": 2, "training_boundary": 2000}
        result = match_conditional_policy(self.selection, self.registry, **arguments)
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["policy"]["config_id"], self.config_id)
        self.assertEqual(result["support_status"], "within_dev_support")

    def test_non_pulse_family_mapping_cannot_reach_an_execution_match(self):
        self.selection["model_ratio"][0].update(
            score_variant="family_selected", score_variant_by_family={"family": "time"},
            score_variant_fallback="time",
        )
        with self.assertRaises(ValueError):
            match_conditional_policy(
                self.selection, self.registry, **self.match_arguments, allow_out_of_support=True,
            )

    def test_tier_signature_omits_empty_models_but_keeps_disabled_structural_candidates(self):
        self.registry["models"]["GDN"]["tier"] = "t2"
        self.registry["models"]["PaAno"] = {
            "tier": "t2", "target_use": "fit_full_prefix", "execution_status": "unavailable",
            "status_reason": "execution disabled",
            "candidates": [{"config_id": "paano", "hyperparameters": {
                "patch_size": 64, "memory_fraction": 0.01, "neighbors": 3,
            }}],
        }
        policy = {**deepcopy(self.selection["model_ratio"][0]), "tier": "t2",
                  "selected_model": "GDN", "candidate_ids_by_model": {"GDN": [self.config_id]}}
        self.selection["tier_adaptive"] = [policy]
        arguments = {**self.match_arguments, "available_count": 100, "feature_count": 2,
                     "training_boundary": 2000, "analysis_kind": "tier_adaptive"}
        self.assertEqual(match_conditional_policy(self.selection, self.registry, **arguments)["status"], "matched")
        self.registry["models"]["PaAno"]["candidates"][0]["hyperparameters"]["patch_size"] = 5
        self.assertEqual(match_conditional_policy(self.selection, self.registry, **arguments)["reasons"],
                         ["conditional_group_missing"])
        policy["candidate_ids_by_model"]["PaAno"] = ["paano"]
        self.assertEqual(match_conditional_policy(self.selection, self.registry, **arguments)["status"], "matched")

    def test_unobserved_joint_shape_is_external_even_inside_each_axis_range(self):
        for update in ({}, {"available_count": 100, "feature_count": 4,
                            "training_boundary": 2000}):
            with self.subTest(update=update):
                arguments = {**self.match_arguments, **update}
                result = match_conditional_policy(self.selection, self.registry, **arguments)
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(result["reasons"], ["joint_shape_not_observed"])
                external = match_conditional_policy(
                    self.selection, self.registry, **arguments, allow_out_of_support=True,
                )
                self.assertEqual(external["status"], "matched")
                self.assertEqual(external["support_status"], "out_of_dev_support")

    def test_missing_or_malformed_support_cannot_enable_external_execution(self):
        valid = self.selection["model_ratio"][0]["support"][0]
        for support in (None, [], "invalid", [None], [{}],
                        [valid, {**valid, "feature_count": True}],
                        [{**valid, "available_count": 2001}]):
            with self.subTest(support=support):
                selection = deepcopy(self.selection)
                selection["model_ratio"][0]["support"] = support
                result = match_conditional_policy(
                    selection, self.registry, **self.match_arguments, allow_out_of_support=True,
                )
                self.assertEqual(result["status"], "unavailable")
                self.assertEqual(result["support_status"], "unavailable")
                self.assertEqual(result["reasons"], ["conditional_group_support_missing"])

    def test_same_percentage_with_larger_absolute_pool_is_not_a_recommendation(self):
        arguments = {**self.match_arguments, "available_count": 1500, "training_boundary": 30000}
        result = match_conditional_policy(self.selection, self.registry, **arguments)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("available_count_out_of_dev_support", result["reasons"])
        self.assertIn("training_boundary_out_of_dev_support", result["reasons"])
        external = match_conditional_policy(
            self.selection, self.registry, **arguments, allow_out_of_support=True,
        )
        self.assertEqual(external["status"], "matched")
        self.assertEqual(external["support_status"], "out_of_dev_support")

    def test_multi_session_feasibility_override_can_exclude_an_aggregate_candidate(self):
        result = match_conditional_policy(
            self.selection, self.registry, **self.match_arguments, feasible_config_ids=(),
        )
        self.assertEqual(result["status"], "unavailable")
        with self.assertRaises(ValueError):
            match_conditional_policy(
                self.selection, self.registry, **self.match_arguments,
                feasible_config_ids=("unknown",),
            )

    def test_ambiguous_group_is_not_selected_by_performance(self):
        selection = deepcopy(self.selection)
        selection["model_ratio"].append(deepcopy(selection["model_ratio"][0]))
        result = match_conditional_policy(selection, self.registry, **self.match_arguments)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reasons"], ["conditional_group_ambiguous"])

    def test_intake_checks_optional_administrative_caps(self):
        for update in ({"max_rows": 2000}, {"max_columns": 2}):
            result = recommend_conditional_candidates(
                self.selection, self.registry, **{**self.intake, **update},
            )
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["reasons"], ["declared_capacity_exceeded"])

    def test_nearest_q_does_not_create_observed_support(self):
        result = recommend_conditional_candidates(
            self.selection, self.registry, **{**self.intake, "nrows": 225},
        )
        self.assertEqual(result["observed_ratio_percent"], 7.5)
        self.assertIsNone(result["normalized_q"])
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("registered_prefix_not_observed", result["reasons"])
        self.assertFalse(result["profitability_optimal"])
        self.assertFalse(result["domain_generalization_guaranteed"])
        self.assertIn("false_negative_cost", result["missing_inputs"])

    def test_under_five_percent_is_not_rounded_up(self):
        result = recommend_conditional_candidates(
            self.selection, self.registry, **{**self.intake, "nrows": 100},
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["normalized_q"])
        self.assertEqual(result["reasons"], ["below_minimum_supported_ratio"])


if __name__ == "__main__":
    unittest.main()
