"""등록된 shape와 합성 점수만으로 조건별 선택부터 최종 spec까지 확인한다."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from src.common.execution_identity import file_sha256
from src.common.load_final_membership import CONDITIONAL_FIELDS, load_final_membership
from src.common.model_feasibility import build_dev18_feasibility_rows, summarize_dev18_feasibility
from src.common.model_registry import load_model_registry_with_sha
from src.common.select_conditional_policy import recommend_conditional_candidates
from tests.ghl_main import run_dev18_tuning as tuning
from tests.ghl_main.build_ratio_tuning_budget import build_full_prefix_budget
from tests.ghl_main.run_registered_models import build_specs
from tests.ghl_main.select_ratio_tuning import select_ratio_tuning_policies


class FullPrefixPolicyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, registry_sha = load_model_registry_with_sha()
        cls.manifest_path = Path(__file__).resolve().parents[2] / "configs/input_manifest.yaml"
        cls.manifest = yaml.safe_load(cls.manifest_path.read_text(encoding="utf-8"))
        entries = cls.manifest["datasets"]["DEV18"]["files"]
        rows = build_dev18_feasibility_rows(
            cls.registry, entries, config_registry_sha256=registry_sha,
            input_manifest_sha256=file_sha256(cls.manifest_path), inventory_sha256="a" * 64,
        )
        evidence = {**summarize_dev18_feasibility(rows, cls.registry),
                    "input_manifest_sha256": file_sha256(cls.manifest_path),
                    "inventory_sha256": "a" * 64, "data_preprocessing_sha256": "b" * 64}
        cls.budget = build_full_prefix_budget(cls.registry, rows, entries, evidence)
        families = {entry["series"]: entry["family"] for entry in entries}
        parameters = {(model, candidate["config_id"]): candidate["hyperparameters"]
                      for model, details in cls.registry["models"].items()
                      for candidate in details["candidates"]}
        scores = []
        for execution in cls.budget["execution_panel"]:
            settings = parameters[execution["model"], execution["config_id"]]
            for series in execution["series_ids"]:
                for ratio in execution["logical_ratios"]:
                    preferred = (execution["model"] == "PaAno"
                                 and settings["patch_size"] == (64 if ratio == 60 else 96)
                                 and settings["learning_rate"] == .001)
                    for variant in execution["primary_score_variants"]:
                        score = .9 if preferred else .2
                        if execution["model"] == "GDN":
                            selected_tuple = (5, 30) if ratio == 60 else (15, 50)
                            if (settings["topk"], settings["epochs"]) == selected_tuple:
                                score = .8
                        if execution["model"] == "TSPulse":
                            selected_window = 64 if ratio == 60 else 96
                            selected_head = "pred" if families[series] == "SMD" else "ensemble"
                            if settings["aggregation_window"] == selected_window:
                                score = .9 if variant == selected_head else .6
                            elif variant in {"pred", "ensemble"}:
                                score = .99
                        scores.append({**execution, "series": series, "family": families[series],
                                       "ratio": ratio, "score_variant": variant,
                                       "vus_pr": score, "status": "complete"})
        cls.scores = scores
        cls.selection = select_ratio_tuning_policies(scores, cls.registry, cls.budget, "c" * 64)

    def test_development_specs_match_budget_and_share_tier3_heads(self):
        specs, panel_by_key = tuning._specs_for_budget(self.budget)
        self.assertEqual(len(specs), self.budget["physical_execution_count"])
        self.assertEqual(len(panel_by_key), len(specs))
        tier3 = [spec for spec in specs if spec["tier"] == "t3"]
        self.assertEqual(len(tier3), 4)
        for spec in tier3:
            self.assertEqual((spec["ratio"], spec["seed"]), (100, 0))
            expected = {"time", "fft", "pred", "ensemble"} if spec["model"] == "TSPulse" else {""}
            self.assertEqual(set(spec["score_variants"]), expected)
        tspulse = [policy for policy in self.selection["model_ratio"] if policy["model"] == "TSPulse"
                   and policy["selection_status"] == "selected"]
        self.assertEqual({policy["ratio"] for policy in tspulse}, {5, 10, 20, 40, 60, 80, 100})
        self.assertTrue(all(policy["score_variant"] == "family_selected"
                            and policy["hyperparameters"]["aggregation_window"] == (64 if policy["ratio"] == 60 else 96)
                            and policy["score_variant_by_family"]["SMD"] == "pred"
                            and policy["score_variant_fallback"] == "time" for policy in tspulse))
        tier3 = [policy for policy in self.selection["tier_adaptive"]
                 if policy["tier"] == "t3" and policy["selection_status"] == "selected"]
        self.assertTrue(all(policy["model"] == "TSPulse" and policy["score_variant"] == "family_selected"
                            and policy["hyperparameters"]["aggregation_window"] == (64 if policy["ratio"] == 60 else 96)
                            for policy in tier3))
        audit = [row for row in self.selection["candidate_audit"] if row["model"] == "TSPulse"
                 and row["analysis_kind"] == "model_ratio" and row["ratio"] == 100
                 and row["score_variant"] != "family_selected"]
        self.assertEqual(len(audit), 12)

    def test_lofo_refits_both_stages_without_holdout_family_scores(self):
        changed = [{**row, "vus_pr": (1. if self._window(row["config_id"]) == 128 else 0.)}
                   if row["model"] == "TSPulse" and row["family"] == "SMD" else row
                   for row in self.scores]
        selection = select_ratio_tuning_policies(changed, self.registry, self.budget, "c" * 64)
        original = {(row["analysis_kind"], row["group_id"], row["ratio"]): row
                    for row in self.selection["adaptive_lofo"] if row["holdout_family"] == "SMD"}
        for row in selection["adaptive_lofo"]:
            if row["holdout_family"] != "SMD":
                continue
            before = original[row["analysis_kind"], row["group_id"], row["ratio"]]
            self.assertEqual((row["model"], row["selected_config_id"], row["score_variant"]),
                             (before["model"], before["selected_config_id"], before["score_variant"]))
            if row["score_variant"] == "family_selected":
                self.assertNotIn("SMD", row["score_variant_by_family"])
                self.assertEqual(row["score_variant_fallback"], "time")
                self.assertAlmostEqual(before["holdout_vus_pr"], .6)
        for current, before in zip(selection["model_comparison"], self.selection["model_comparison"]):
            for fold, previous in zip(current["folds"], before["folds"]):
                if fold["holdout_family"] == "SMD":
                    for side in ("left", "right"):
                        self.assertEqual(fold[side + "_config_id"], previous[side + "_config_id"])
                        self.assertEqual(fold[side + "_score_variant"], previous[side + "_score_variant"])

    def _window(self, config_id):
        return next(candidate["hyperparameters"]["aggregation_window"]
                    for candidate in self.registry["models"]["TSPulse"]["candidates"]
                    if candidate["config_id"] == config_id)

    def test_registered_targets_keep_ratio_config_through_membership_and_final_specs(self):
        gdn_executions = [row for row in self.budget["execution_panel"] if row["model"] == "GDN"]
        gdn_configs = {candidate["config_id"] for candidate in self.registry["models"]["GDN"]["candidates"]}
        self.assertEqual(len(gdn_configs), 3)
        self.assertEqual({row["config_id"] for row in gdn_executions}, gdn_configs)
        for config in gdn_configs:
            self.assertEqual({(row["physical_ratio"], row["seed"]) for row in gdn_executions
                              if row["config_id"] == config},
                             {(ratio, seed) for ratio in (5, 10, 20, 40, 60, 80, 100) for seed in (0, 1, 2)})
        targets = tuning._load_final_target_shapes(self.manifest)
        self.assertEqual(len(targets), 27)
        self.assertEqual(sum(target["split_role"] == "ghl25_final" for target in targets), 25)
        membership = tuning.build_conditional_membership_rows(self.selection, self.registry, self.manifest)
        tier_count = len({model["tier"] for model in self.registry["models"].values()})
        self.assertEqual(len(membership), 27 * 7 * (len(self.registry["models"]) + tier_count))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "final_policy_membership.csv"
            tuning._write_csv(path, membership, CONDITIONAL_FIELDS)
            digest, loaded = load_final_membership(path, self.registry)
            diagnostic_rows = [dict(row) for row in membership]
            diagnostic = next(row for row in diagnostic_rows
                              if row["model"] == "TSPulse" and row["status"] == "runnable")
            diagnostic["score_variant"] = "raw_max"
            diagnostic_path = Path(directory) / "diagnostic_membership.csv"
            tuning._write_csv(diagnostic_path, diagnostic_rows, CONDITIONAL_FIELDS)
            with self.assertRaisesRegex(ValueError, "score_variant"):
                load_final_membership(diagnostic_path, self.registry)
            missing_group = [row for row in membership if not (
                row["series"] == "01" and row["split_role"] == "ghl25_final"
                and row["analysis_kind"] == "model_ratio" and row["model"] == "GDN")]
            damaged = Path(directory) / "missing_model.csv"
            tuning._write_csv(damaged, missing_group, CONDITIONAL_FIELDS)
            with self.assertRaisesRegex(ValueError, "집단 구성이 완전하지"):
                load_final_membership(damaged, self.registry)
            with patch("tests.ghl_main.run_registered_models.load_ghl_registered_inputs",
                       side_effect=AssertionError("raw inputs must not be read")):
                specs = build_specs("final", split_role="ghl25_final", series="01",
                                    final_policy_membership_path=path, input_manifest_path=self.manifest_path)
            self.assertTrue(specs)
            self.assertTrue(all(spec["series"] == "01" and spec["final_policy_membership_sha256"] == digest
                                for spec in specs))
            selected = {ratio: {spec["hyperparameters"]["patch_size"] for spec in specs
                                if spec["model"] == "PaAno" and spec["ratio"] == ratio}
                        for ratio in (60, 80)}
            self.assertEqual(selected, {60: {64}, 80: {96}})
            gdn_selected = {ratio: {
                (spec["hyperparameters"]["topk"], spec["hyperparameters"]["epochs"])
                for spec in specs if spec["model"] == "GDN" and spec["ratio"] == ratio
            } for ratio in (60, 80)}
            self.assertEqual(gdn_selected, {60: {(5, 30)}, 80: {(15, 50)}})
            pulse_specs = [spec for spec in specs if spec["model"] == "TSPulse"]
            self.assertTrue(pulse_specs)
            self.assertTrue(all(spec["hyperparameters"]["aggregation_window"] in (64, 96)
                                and spec["score_variants"] == ("time",) for spec in pulse_specs))
            target_rows = [row for row in loaded if row["split_role"] == "ghl25_final"
                           and row["series"] == "01" and row["analysis_kind"] == "model_ratio"]
            self.assertTrue(all(row["status"] == "runnable" for row in target_rows if row["model"] == "GDN"))
            self.assertTrue(any(row["support_status"] == "out_of_dev_support" for row in target_rows
                                if row["model"] == "PaAno"))

    def test_channel_and_size_support_are_enforced_without_globalizing_final_extrapolation(self):
        targets = [{"split_role": "ghl25_final", "series": series,
                    "training_lengths": [length], "test_length": 10000, "feature_count": channels}
                   for series, length, channels in (("01", 50000, 2), ("02", 50000, 19), ("03", 10, 1))]
        rows = tuning.build_conditional_membership_rows(self.selection, self.registry, targets=targets)
        model_rows = [row for row in rows if row["analysis_kind"] == "model_ratio"]
        low_channel = [row for row in model_rows if row["series"] == "01" and row["model"] == "GDN"]
        self.assertEqual(len(low_channel), 7)
        self.assertTrue(all(row["status"] == "unavailable" and not row["config_id"] for row in low_channel))
        gdn = [row for row in model_rows if row["series"] == "02" and row["model"] == "GDN"]
        self.assertTrue(all(row["status"] == "runnable" for row in gdn))
        unsupported = [row for row in model_rows if row["series"] == "03" and row["model"] == "PaAno"]
        self.assertTrue(all(row["status"] == "unavailable" for row in unsupported))
        final = next(row for row in model_rows if row["series"] == "02" and row["model"] == "PaAno"
                     and row["evaluation_ratio"] == 60)
        self.assertEqual((final["status"], final["support_status"]), ("runnable", "out_of_dev_support"))
        company = recommend_conditional_candidates(
            self.selection, self.registry, nrows=30000, dfeatures=19, planned_rows=50000,
            max_rows=1000000, max_columns=248, evaluation_rows=10000,
        )
        self.assertEqual(company["status"], "unavailable")
        self.assertFalse(company["candidates"])
        self.assertEqual(company["reasons"], ["tuning_support_required"])


if __name__ == "__main__":
    unittest.main()
