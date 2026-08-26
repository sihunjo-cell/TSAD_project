"""Dev18 단일 진입점의 선택·보고 계약을 검증한다."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.ghl_main.run_dev18_tuning import (
    _validate_bound_run_files,
    _validate_primary_manifest_rows,
    build_final_membership_rows,
    select_tuning_policies,
    validate_vus_evidence,
    write_selection_reports,
)


def _rows():
    rows = []
    scores = {
        ("M1", "c1", "A"): 0.8,
        ("M1", "c1", "B"): 0.7,
        ("M1", "c2", "A"): 0.9,
        ("M1", "c2", "B"): 0.8,
        ("M2", "c3", "A"): 0.6,
        ("M2", "c3", "B"): 0.65,
    }
    for (model, config_id, family), value in scores.items():
        for ratio in (5, 10):
            for seed in (0, 1):
                rows.append({
                    "series": f"{len(rows) + 1:02d}",
                    "family": family,
                    "tier": "t1",
                    "model": model,
                    "config_id": config_id,
                    "ratio": ratio,
                    "seed": seed,
                    "score_variant": "",
                    "vus_pr": value + seed * 0.01,
                    "status": "complete",
                })
    return rows


class TestDev18Tuning(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "selection": {
                "primary_hpo_regime": "equal_trial",
                "budget_id": "b123456789abc",
                "selection_rule_id": "tier_fixed_family_lofo_v2",
                "tier_q_floor": {"t1": 5},
            },
            "models": {
                "M1": {
                    "tier": "t1", "target_use": "training_free",
                    "source_commit": "a" * 40,
                    "source_checkpoint_sha256": "none",
                    "candidates": [
                        {"config_id": "c1", "hyperparameters": {"window": 1}},
                        {"config_id": "c2", "hyperparameters": {"window": 2}},
                    ],
                },
                "M2": {
                    "tier": "t1", "target_use": "fit_validation",
                    "source_commit": "b" * 40,
                    "source_checkpoint_sha256": "none",
                    "candidates": [
                        {"config_id": "c3", "hyperparameters": {"window": 3}},
                    ],
                },
                "M0": {
                    "tier": "t1", "target_use": "fit_validation",
                    "source_commit": "c" * 40,
                    "source_checkpoint_sha256": "none",
                    "candidates": [{"config_id": "c0", "hyperparameters": {}}],
                },
            },
        }
        self.budget = {
            "budget_id": "b123456789abc",
            "primary_hpo_regime": "equal_trial",
            "tie_rule": {"tolerance": 1e-6},
            "model_panels": [
                {
                    "model": "M1", "tier": "t1",
                    "logical_ratios": [5, 10],
                    "selected_config_ids": ["c1", "c2"],
                    "primary_score_variants": [""],
                    "support_status": "eligible_for_model_fixed_hpo",
                    "dev18_tier_representative_eligible": True,
                },
                {
                    "model": "M2", "tier": "t1",
                    "logical_ratios": [5, 10],
                    "selected_config_ids": ["c3"],
                    "primary_score_variants": [""],
                    "support_status": "eligible_for_model_fixed_hpo",
                    "dev18_tier_representative_eligible": True,
                },
                {
                    "model": "M0", "tier": "t1", "logical_ratios": [],
                    "selected_config_ids": [], "primary_score_variants": [""],
                    "support_status": "unavailable",
                    "dev18_tier_representative_eligible": False,
                },
            ],
        }

    def test_family_lofo_selects_model_then_full_panel_recipe(self):
        selection = select_tuning_policies(
            _rows(), self.registry, self.budget, evaluator_sha256="d" * 64,
        )

        model_rows = {row["model"]: row for row in selection["model_fixed"]}
        self.assertEqual(model_rows["M1"]["config_id"], "c2")
        self.assertEqual(model_rows["M0"]["selection_status"], "unavailable")
        self.assertEqual(selection["tier_fixed"][0]["selected_model"], "M1")
        self.assertIn("family-LOFO", selection["tier_fixed"][0]["selection_reason"])

    def test_membership_reuses_target_free_r100_and_marks_unsupported_rows(self):
        selection = select_tuning_policies(
            _rows(), self.registry, self.budget, evaluator_sha256="d" * 64,
        )
        rows = build_final_membership_rows(
            selection, self.registry, ratios=(5, 10),
            split_roles=("ghl25_final",),
        )

        m1 = [row for row in rows if row["analysis_kind"] == "model_fixed" and row["model"] == "M1"]
        self.assertEqual({row["physical_ratio"] for row in m1}, {100})
        m0 = [row for row in rows if row["model"] == "M0"]
        self.assertTrue(m0)
        self.assertTrue(all(row["status"] == "unavailable" for row in m0))

    def test_reports_use_simple_model_and_tier_names(self):
        selection = select_tuning_policies(
            _rows(), self.registry, self.budget, evaluator_sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_selection_reports(_rows(), selection, output)
            for name in (
                "M1.csv", "M1.png", "Tier1.csv", "Tier1.png",
                "M0.csv", "M0.png", "models.csv", "models.png",
                "selection.csv", "selection.png", "family_lofo.csv",
            ):
                self.assertTrue((output / name).is_file(), name)
            with (output / "selection.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["selected_model"], "M1")
            self.assertTrue(rows[0]["selection_reason"])
            with (output / "Tier1.csv").open(encoding="utf-8", newline="") as file:
                tier_rows = list(csv.DictReader(file))
            selected = next(row for row in tier_rows if row["selected_model"] == "True")
            self.assertTrue(selected["selected_hyperparameters"])
            self.assertTrue(selected["selection_reason"])

    def test_direct_file_cli_can_import_project_packages(self):
        script = Path(__file__).parents[1] / "ghl_main" / "run_dev18_tuning.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=script.parents[2],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_resume_snapshot_must_match_current_project_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "run_snapshot.json"
            snapshot.write_text(
                json.dumps({"project_commit": "a" * 40}), encoding="utf-8",
            )
            metadata = {"run_snapshot": {
                "file": snapshot.name,
                "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            }, "training_files": {}}
            with patch("tests.ghl_main.run_dev18_tuning.REPOSITORY_ROOT", root):
                _validate_bound_run_files(
                    metadata, expected_project_commit="a" * 40,
                )
                with self.assertRaisesRegex(ValueError, "commit"):
                    _validate_bound_run_files(
                        metadata, expected_project_commit="b" * 40,
                    )

    def test_primary_manifest_must_match_exact_budget_keys(self):
        budget = {
            "budget_id": "b123456789abc",
            "execution_panel": [{
                "model": "M1", "tier": "t1", "config_id": "c1",
                "physical_ratio": 100, "seed": 0,
                "primary_score_variants": [""],
                "diagnostic_score_variants": [],
            }],
        }
        row = {
            "series": "01", "family": "A", "tier": "t1", "model": "M1",
            "config_id": "c1", "physical_ratio": "100", "seed": "0",
            "score_variant": "", "primary_score": "true", "status": "complete",
            "budget_id": "b123456789abc",
        }
        _validate_primary_manifest_rows([row], budget, ("01",))
        with self.assertRaisesRegex(ValueError, "exact budget"):
            _validate_primary_manifest_rows(
                [{**row, "score_variant": "diagnostic"}], budget, ("01",),
            )

    def test_vus_evidence_is_bound_to_current_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = root / "vus_pr.py"
            evaluator.write_text("pass\n", encoding="utf-8")
            report = root / "report.json"
            report.write_text(json.dumps({
                "status": "passed",
                "official": {
                    "commit": "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48",
                    "version": "opt",
                },
                "n_thresholds": 250,
                "absolute_tolerance": 1e-12,
                "maximum_absolute_difference": 0.0,
                "evaluator_sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
            }), encoding="utf-8")

            validated = validate_vus_evidence(report, evaluator)
            self.assertEqual(validated["status"], "passed")
            evaluator.write_text("raise RuntimeError\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evaluator"):
                validate_vus_evidence(report, evaluator)


if __name__ == "__main__":
    unittest.main()
