"""원래 점수의 재사용과 비율별 최종 실행 연결을 작은 입력으로 확인한다."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from tests.ghl_main import run_dev18_tuning as tuning
from src.common.load_final_membership import FIELDS, load_final_membership, build_final_execution_union
from tests.unit.test_final_membership import make_registry, make_membership_rows


class TestRatioTuningResume(unittest.TestCase):
    def test_preserved_diagnostic_must_exist_before_resuming(self):
        from tests.ghl_main.run_ratio_tuning import _validate_preserved_manifest

        panel = {"model": "TSPulse", "config_id": "c1", "physical_ratio": 100, "seed": 0,
                 "primary_score_variants": ["raw_max"], "diagnostic_score_variants": ["time"]}
        budget = {"budget_id": "old", "execution_panel": [panel]}
        row = {**panel, "series": "01", "status": "complete", "budget_id": "old",
               "score_variant": "raw_max"}
        with self.assertRaises(ValueError):
            _validate_preserved_manifest([row], budget, ["01"])
        _validate_preserved_manifest([row, {**row, "score_variant": "time"}], budget, ["01"])

    def test_reuses_only_same_score_and_evaluator_and_rejects_conflicting_values(self):
        manifest = {"series": "01", "model": "M", "config_id": "c1", "physical_ratio": 100,
                    "seed": 0, "score_variant": "", "score_file": "same.npy", "score_sha256": "a"}
        row = {**manifest, "ratio": 5, "status": "complete", "normalization": "trainnorm",
               "vus_pr": .7, "evaluator_sha256": "e", "ell_max_id": "l"}
        reused = tuning._reuse_trial_scores([manifest], [row, {**row, "ratio": 100}],
                                           evaluator_sha256="e", ell_max_id="l")
        self.assertEqual(reused[tuning._manifest_key(manifest)]["vus_pr"], .7)
        self.assertFalse(tuning._reuse_trial_scores([{**manifest, "score_sha256": "b"}], [row],
                                                    evaluator_sha256="e", ell_max_id="l"))
        with self.assertRaises(ValueError):
            tuning._reuse_trial_scores([manifest], [row, {**row, "vus_pr": .2}],
                                      evaluator_sha256="e", ell_max_id="l")
        with self.assertRaises(ValueError):
            tuning._reuse_trial_scores([manifest], [row], evaluator_sha256="new", ell_max_id="l")

    def test_accepts_parent_budget_only_for_parent_execution_keys(self):
        panel = {"model": "M", "config_id": "c1", "physical_ratio": 40,
                 "seed": 0, "primary_score_variants": [""]}
        row = {**panel, "series": "01", "score_variant": "", "primary_score": "true",
               "status": "complete", "budget_id": "old"}
        budget = {"budget_id": "new", "execution_panel": [panel],
                  "legacy_budget": {"budget_id": "old", "execution_panel": [panel]}}
        self.assertEqual(tuning._validate_primary_manifest_rows([row], budget, ["01"]), [row])
        changed = {**panel, "config_id": "c2"}
        with self.assertRaises(ValueError):
            tuning._validate_primary_manifest_rows([{**row, "config_id": "c2"}],
                                                   {**budget, "execution_panel": [changed]}, ["01"])

    def test_scoring_submits_only_missing_physical_scores(self):
        panel = {"model": "M", "config_id": "old", "physical_ratio": 40, "logical_ratios": [40],
                 "seed": 0, "primary_score_variants": [""], "diagnostic_score_variants": []}
        old = {**panel, "series": "01", "family": "A", "tier": "t2", "score_variant": "",
               "primary_score": "true", "status": "complete", "budget_id": "new",
               "score_file": "old.npy", "score_sha256": "a", "seed": "0", "physical_ratio": "40"}
        new = {**old, "config_id": "new", "score_file": "new.npy", "score_sha256": "b"}
        saved = {**old, "ratio": 40, "normalization": "trainnorm", "vus_pr": .7,
                 "evaluator_sha256": "e", "ell_max_id": "l"}
        budget = {"budget_id": "new", "budget_sha256": "b", "series_ids": ["01"],
                  "execution_panel": [panel, {**panel, "config_id": "new"}],
                  "primary_logical_score_row_count": 2}
        with patch.object(tuning, "_validate_ell_max", return_value={
            "ell_max_id": "l", "series": [{"series": "01", "l_max_samples": 3}],
        }), patch.object(tuning, "validate_vus_evidence", return_value={
            "evaluator_sha256": "e", "n_thresholds": 250,
        }), patch("tests.ghl_main.run_registered_models._verify_manifest_file"), \
                patch.object(tuning, "_load_dev18_labels", return_value=[0, 1]), \
                patch.object(tuning, "_score_primary_rows", return_value={
                    tuning._manifest_key(new): {"normalization": "trainnorm", "vus_pr": .8},
                }) as score:
            result = tuning.build_trial_score_ledger(
                [old, new], budget=budget, reused_ledger=[saved], project_commit="commit",
            )
        self.assertEqual([row["vus_pr"] for row in result], [.8, .7])
        self.assertEqual([task["manifest_row"]["score_file"] for task in score.call_args.args[0]],
                         ["new.npy"])

    def test_new_model_ratio_config_reaches_execution_union_without_changing_controls(self):
        registry = make_registry()
        rows = make_membership_rows(registry)
        model_rows = [dict(row, analysis_kind="model_ratio") for row in rows
                      if row["analysis_kind"] == "model_fixed"]
        model = registry["models"]["GDN"]
        new_config = "cnew"
        model["candidates"].append({"config_id": new_config})
        for row in model_rows:
            if row["model"] == "GDN" and row["evaluation_ratio"] == 100:
                row["config_id"] = new_config
        with TemporaryDirectory() as directory:
            path = Path(directory) / "membership.csv"
            tuning._write_csv(path, rows + model_rows, FIELDS)
            _, loaded = load_final_membership(path, registry)
        union = build_final_execution_union(loaded, "ghl25_final")
        self.assertIn(("GDN", new_config, 100), union)
        self.assertIn(("GDN", model["candidates"][0]["config_id"], 100), union)


if __name__ == "__main__":
    unittest.main()
