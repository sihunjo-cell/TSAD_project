"""Dev18 단일 진입점의 선택·보고 계약을 검증한다."""

import csv
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.ghl_main import run_dev18_tuning
from tests.ghl_main.run_dev18_tuning import (
    _evidence_directory,
    _load_completion_receipt,
    _load_score_manifest,
    _replace_manifest_rows,
    _write_completion_receipt,
    _write_run_snapshot,
    _validate_bound_run_files,
    _validate_primary_manifest_rows,
    build_final_membership_rows,
    load_trial_score_ledger,
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

    def test_trial_score_ledger_loader_validates_sealed_logical_keys(self):
        header = [
            "series", "family", "tier", "model", "config_id", "ratio", "seed",
            "score_variant", "normalization", "vus_pr", "score_file", "score_sha256",
            "evaluator_sha256", "ell_max_id", "status", "status_reason",
        ]
        budget = {
            "budget_id": "b123456789abc", "seeds": [0, 1],
            "model_panels": [{
                "model": "M1", "tier": "t1", "logical_ratios": [5, 10],
                "selected_config_ids": ["c1"], "primary_score_variants": [""],
            }],
        }
        rows = [{
            "series": series, "family": family, "tier": "t1", "model": "M1",
            "config_id": "c1", "ratio": str(ratio), "seed": str(seed),
            "score_variant": "", "normalization": "trainnorm", "vus_pr": "0.5",
            "score_file": "score.npy", "score_sha256": "a" * 64,
            "evaluator_sha256": "b" * 64, "ell_max_id": "ell-v1",
            "status": "complete", "status_reason": "",
        } for series, family in (("01", "A"), ("02", "B"))
            for ratio in (5, 10) for seed in (0, 1)]

        def write_ledger(path, ledger_rows):
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=header)
                writer.writeheader()
                writer.writerows(ledger_rows)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.csv"
            write_ledger(path, rows)
            loaded = load_trial_score_ledger(path, budget)
            self.assertIsInstance(loaded[0]["ratio"], int)
            self.assertIsInstance(loaded[0]["seed"], int)
            self.assertIsInstance(loaded[0]["vus_pr"], float)

            write_ledger(path, rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_trial_score_ledger(path, budget)

            write_ledger(path, [{**row, "status": "failed"} for row in rows])
            with self.assertRaisesRegex(ValueError, "complete"):
                load_trial_score_ledger(path, budget)

            write_ledger(path, [{**row, "evaluator_sha256": "c" * 64} if index == 0 else row
                                for index, row in enumerate(rows)])
            with self.assertRaisesRegex(ValueError, "evaluator"):
                load_trial_score_ledger(path, budget)

            write_ledger(path, [{**row, "ratio": "40"} if index == 0 else row
                                for index, row in enumerate(rows)])
            with self.assertRaisesRegex(ValueError, "budget"):
                load_trial_score_ledger(path, budget)

            write_ledger(path, [{**row, "family": "C"} if index == 1 else row
                                for index, row in enumerate(rows)])
            with self.assertRaisesRegex(ValueError, "series.*family"):
                load_trial_score_ledger(path, budget)

            write_ledger(path, [{**row, "tier": "t2"} if index == 0 else row
                                for index, row in enumerate(rows)])
            with self.assertRaisesRegex(ValueError, "tier"):
                load_trial_score_ledger(path, budget)

            production_budget = {
                "budget_id": "b5367ad431093", "seeds": [0],
                "model_panels": [{
                    "model": "M1", "tier": "t1", "logical_ratios": [5],
                    "selected_config_ids": ["c1"], "primary_score_variants": [""],
                }],
            }
            production_rows = [
                {**rows[0], "series": f"{index:02d}", "family": f"F{min(index, 10)}"}
                for index in range(1, 19)
            ]
            write_ledger(path, production_rows)
            self.assertEqual(len(load_trial_score_ledger(path, production_budget)), 18)
            write_ledger(path, [{**row, "family": "F1"} for row in production_rows])
            with self.assertRaisesRegex(ValueError, "10개 family"):
                load_trial_score_ledger(path, production_budget)

    def test_ratio_adaptive_selection_uses_native_support_and_fixed_recipe(self):
        registry = {
            "selection": {"tier_q_floor": {"t1": 5, "t2": 5}},
            "models": {
                "PCA_LEGACY": {"tier": "t1", "target_use": "fit_validation",
                               "source_commit": "p" * 40, "source_checkpoint_sha256": "none",
                               "candidates": [{"config_id": "pca", "hyperparameters": {}}]},
                "MWVAR": {"tier": "t1", "target_use": "training_free",
                          "source_commit": "m" * 40, "source_checkpoint_sha256": "none",
                          "candidates": [{"config_id": "mw", "hyperparameters": {}}]},
                "ALoRa": {"tier": "t2", "target_use": "fit_validation",
                          "source_commit": "a" * 40, "source_checkpoint_sha256": "none",
                          "candidates": [{"config_id": "alora", "hyperparameters": {}}]},
                "Other": {"tier": "t2", "target_use": "fit_validation",
                          "source_commit": "o" * 40, "source_checkpoint_sha256": "none",
                          "candidates": [{"config_id": "other", "hyperparameters": {}}]},
                "GDN": {"tier": "t2", "target_use": "fit_validation",
                        "source_commit": "g" * 40, "source_checkpoint_sha256": "none",
                        "candidates": [
                            {"config_id": "gdn_q10", "hyperparameters": {"window": 10}},
                            {"config_id": "gdn_full", "hyperparameters": {"window": 20}},
                        ]},
                "PaAno": {"tier": "t2", "target_use": "fit_validation",
                          "source_commit": "n" * 40, "source_checkpoint_sha256": "none",
                          "candidates": [{"config_id": "paano", "hyperparameters": {}}]},
            },
        }
        budget = {
            "budget_id": "b123456789abc", "primary_hpo_regime": "equal_trial",
            "tie_rule": {"tolerance": 1e-6}, "model_panels": [
                {"model": "PCA_LEGACY", "tier": "t1", "logical_ratios": [100],
                 "selected_config_ids": ["pca"], "primary_score_variants": [""],
                 "dev18_tier_representative_eligible": False},
                {"model": "MWVAR", "tier": "t1", "logical_ratios": [5, 10, 40, 100],
                 "selected_config_ids": ["mw"], "primary_score_variants": [""],
                 "dev18_tier_representative_eligible": True},
                {"model": "ALoRa", "tier": "t2", "logical_ratios": [],
                 "selected_config_ids": [], "primary_score_variants": [""],
                 "dev18_tier_representative_eligible": False},
                {"model": "Other", "tier": "t2", "logical_ratios": [],
                 "selected_config_ids": [], "primary_score_variants": [""],
                 "dev18_tier_representative_eligible": False},
                {"model": "GDN", "tier": "t2", "logical_ratios": [10, 40],
                 "selected_config_ids": ["gdn_q10", "gdn_full"], "primary_score_variants": [""],
                 "dev18_tier_representative_eligible": True},
                {"model": "PaAno", "tier": "t2", "logical_ratios": [40, 60],
                 "selected_config_ids": ["paano"], "primary_score_variants": [""],
                 "dev18_tier_representative_eligible": False},
            ],
        }

        def scores(model, tier, config_id, values):
            return [{
                "series": series, "family": family, "tier": tier, "model": model,
                "config_id": config_id, "ratio": ratio, "seed": 0, "score_variant": "",
                "vus_pr": value, "status": "complete",
            } for ratio, value in values.items()
                for series, family in (("01", "A"), ("02", "B"))]

        rows = (
            scores("PCA_LEGACY", "t1", "pca", {100: 0.99})
            + scores("MWVAR", "t1", "mw", {5: 0.70, 10: 0.70, 40: 0.70, 100: 0.70})
            + scores("GDN", "t2", "gdn_q10", {10: 0.90, 40: 0.10})
            + scores("GDN", "t2", "gdn_full", {10: 0.60, 40: 0.95})
            + scores("PaAno", "t2", "paano", {40: 0.96, 60: 0.80})
        )
        selection = select_tuning_policies(
            rows, registry, budget, evaluator_sha256="d" * 64,
            structural_block_evidence={
                "ALoRa": {"heads": 8, "blocked_series": ["03", "07"]},
            },
        )
        adaptive = {
            (row["tier"], row["ratio"]): row
            for row in selection["tier_adaptive"]
        }

        self.assertEqual(adaptive[("t1", 100)]["selected_model"], "MWVAR")
        self.assertEqual(adaptive[("t2", 5)]["selection_status"], "unavailable")
        self.assertEqual(adaptive[("t2", 10)]["selected_model"], "GDN")
        self.assertEqual(adaptive[("t2", 40)]["selected_model"], "PaAno")
        self.assertNotIn("PCA_LEGACY", {
            row["selected_model"] for row in selection["tier_adaptive"]
        })
        fixed = {row["model"]: row for row in selection["model_fixed"]}
        self.assertEqual(fixed["GDN"]["config_id"], "gdn_full")
        self.assertTrue(all(
            row["config_id"] == fixed[row["selected_model"]]["config_id"]
            for row in selection["tier_adaptive"]
            if row["selection_status"] == "selected"
        ))
        audit = {
            (row["tier"], row["ratio"], row["model"]): row
            for row in selection["candidate_audit"]
        }
        self.assertEqual(
            set(audit[("t2", 10, "GDN")]),
            {
                "tier", "ratio", "model", "config_id", "score_variant", "eligibility",
                "family_lofo_vus_pr", "selected", "reason_code", "reason",
            },
        )
        self.assertEqual(audit[("t1", 100, "PCA_LEGACY")]["eligibility"], "reference_only")
        self.assertEqual(
            audit[("t2", 5, "ALoRa")]["eligibility"],
            "full_panel_config_unavailable",
        )
        self.assertIn("pair_count < heads 8", audit[("t2", 5, "ALoRa")]["reason"])
        self.assertIn("03, 07", audit[("t2", 5, "ALoRa")]["reason"])
        self.assertEqual(
            audit[("t2", 5, "Other")]["reason"],
            "18개 panel을 덮는 model-fixed config가 없다",
        )
        self.assertEqual(audit[("t2", 5, "GDN")]["eligibility"], "ratio_unsupported")
        self.assertEqual(
            audit[("t2", 40, "GDN")]["reason"],
            "tolerance와 결정적 동률 규칙을 적용해 미선택",
        )
        transitions = [
            row for row in selection["policy_transitions"] if row["tier"] == "t2"
        ]
        self.assertEqual(transitions[0]["transition"], "unavailable")
        self.assertEqual(transitions[1]["transition"], "initial")
        self.assertTrue(transitions[0]["transition_key"].startswith("tr"))
        self.assertEqual(transitions[-1]["current_ratio"], 100)

    def test_structural_block_evidence_summarizes_pair_count_series(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feasibility.csv"
            path.write_text(
                "model,status,series,status_reason,derived_json\n"
                "ALoRa,structurally_infeasible,03,pair count 1 < heads 8,\"{\"\"pair_count\"\":1}\"\n"
                "ALoRa,structurally_infeasible,04,validation length 1 < 20,\"{\"\"pair_count\"\":512}\"\n",
                encoding="utf-8",
            )
            evidence = run_dev18_tuning.load_structural_block_evidence(
                path, {"models": {"ALoRa": {"fixed": {"heads": 8}}}},
            )
        self.assertEqual(evidence["ALoRa"], {"heads": 8, "blocked_series": ["03"]})

    def test_checkpoint_validation_reads_fresh_runtime_evidence(self):
        with patch.object(
            run_dev18_tuning,
            "_read_json",
            side_effect=RuntimeError("captured"),
        ) as reader:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                run_dev18_tuning._validate_checkpoint_report("TimeRCD", {}, {})

        self.assertEqual(
            reader.call_args.args[0],
            run_dev18_tuning.REPOSITORY_ROOT
            / ".runtime" / "dev18_checkpoint_smoke"
            / "time_rcd" / "dev18_checkpoint_smoke.json",
        )

    def test_execution_priority_puts_small_work_before_expensive_models(self):
        specs = [
            {"model": "TSPulse", "ratio": 100, "seed": 0,
             "config_id": "tspulse", "hyperparameters": {}},
            {"model": "GDN", "ratio": 100, "seed": 2,
             "config_id": "gdn-large", "hyperparameters": {"embedding": 128}},
            {"model": "GDN", "ratio": 10, "seed": 0,
             "config_id": "gdn-large", "hyperparameters": {"embedding": 128}},
            {"model": "GDN", "ratio": 10, "seed": 0,
             "config_id": "gdn-small", "hyperparameters": {"embedding": 64}},
            {"model": "PaAno", "ratio": 40, "seed": 0,
             "config_id": "paano", "hyperparameters": {}},
            {"model": "TimeRCD", "ratio": 100, "seed": 0,
             "config_id": "timercd", "hyperparameters": {}},
            {"model": "PCA_LEGACY", "ratio": 100, "seed": 0,
             "config_id": "pca", "hyperparameters": {}},
            {"model": "MWVAR", "ratio": 100, "seed": 0,
             "config_id": "mwvar", "hyperparameters": {}},
            {"model": "SQDIFF_LAST3", "ratio": 100, "seed": 0,
             "config_id": "sqdiff", "hyperparameters": {}},
        ]
        ordered = sorted(specs, key=run_dev18_tuning._execution_priority)
        self.assertEqual(
            [(spec["model"], spec["ratio"], spec["hyperparameters"].get("embedding"))
             for spec in ordered],
            [
                ("SQDIFF_LAST3", 100, None),
                ("MWVAR", 100, None),
                ("PCA_LEGACY", 100, None),
                ("TimeRCD", 100, None),
                ("PaAno", 40, None),
                ("GDN", 10, 64),
                ("GDN", 10, 128),
                ("GDN", 100, 128),
                ("TSPulse", 100, None),
            ],
        )

    def test_series_priority_uses_manifest_workload(self):
        entries = [
            {"series": "01", "order": 1, "row_count": 100, "feature_count": 10},
            {"series": "02", "order": 2, "row_count": 10, "feature_count": 2},
            {"series": "03", "order": 3, "row_count": 5, "feature_count": 4},
        ]
        self.assertEqual(
            [entry["series"] for entry in sorted(
                entries, key=run_dev18_tuning._series_execution_priority,
            )],
            ["02", "03", "01"],
        )

    def test_series_evidence_directories_do_not_overwrite_each_other(self):
        base = Path("scores") / "tier2" / "GDN" / "config"
        self.assertEqual(_evidence_directory(base, 1), base / "series_01")
        self.assertEqual(_evidence_directory(base, 18), base / "series_18")
        self.assertNotEqual(
            _evidence_directory(base, 1), _evidence_directory(base, 18),
        )

    def test_completed_manifest_row_survives_restart(self):
        row = {
            "series": "01", "model": "GDN", "config_id": "gdn-small",
            "physical_ratio": "10", "seed": "0", "score_variant": "",
            "status": "complete", "budget_id": "b123456789abc",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score_manifest.csv"
            _replace_manifest_rows([], [row], path)

            loaded = _load_score_manifest(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["status"], "complete")
        self.assertEqual(loaded[0]["model"], "GDN")

    def test_interrupted_manifest_write_preserves_previous_progress(self):
        completed = {
            "series": "01", "model": "GDN", "config_id": "gdn-small",
            "physical_ratio": "10", "seed": "0", "score_variant": "",
            "status": "complete", "budget_id": "b123456789abc",
        }
        pending = {
            "series": "02", "model": "GDN", "config_id": "gdn-small",
            "physical_ratio": "10", "seed": "0", "score_variant": "",
            "status": "complete", "budget_id": "b123456789abc",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score_manifest.csv"
            rows = _replace_manifest_rows([], [completed], path)

            def interrupt_write(write_path, *_args):
                Path(write_path).write_text("partial", encoding="utf-8")
                raise KeyboardInterrupt

            with patch(
                "tests.ghl_main.run_dev18_tuning._write_csv",
                side_effect=interrupt_write,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    _replace_manifest_rows(rows, [pending], path)
            try:
                reloaded = _load_score_manifest(path)
            except ValueError:
                reloaded = []

        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0]["series"], "01")

    def test_completion_receipt_recovers_rows_before_manifest_update(self):
        spec = {
            "model": "TSPulse", "config_id": "c123456789abc",
            "ratio": 100, "seed": 0,
        }
        panel = {
            "primary_score_variants": ["raw_max"],
            "diagnostic_score_variants": ["time"],
        }
        rows = [{
            "series": "03", "model": "TSPulse",
            "config_id": "c123456789abc", "physical_ratio": 100,
            "seed": 0, "score_variant": variant, "status": "complete",
            "budget_id": "b123456789abc",
        } for variant in ("raw_max", "time")]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completion.json"
            _write_completion_receipt(path, rows)
            recovered = _load_completion_receipt(
                path, spec=spec, panel_row=panel, series=3,
                budget_id="b123456789abc",
            )

        self.assertEqual(recovered, rows)

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
        self.assertIn(
            "compatible_project_commit",
            inspect.signature(_validate_bound_run_files).parameters,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "run_snapshot.json"
            snapshot.write_text(
                json.dumps({
                    "project_commit": "a" * 40,
                    "spec": {"model": "MWVAR"},
                    "execution_policy": {},
                    "environment": {"runtime_snapshot": {"sha256": "c" * 64}},
                }), encoding="utf-8",
            )
            metadata = {"run_snapshot": {
                "file": snapshot.name,
                "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            }, "training_files": {}}
            with patch("tests.ghl_main.run_dev18_tuning.REPOSITORY_ROOT", root):
                _validate_bound_run_files(
                    metadata,
                    expected_project_commit="a" * 40,
                    expected_environment={
                        "runtime_snapshot": {"sha256": "c" * 64},
                    },
                )
                with self.assertRaisesRegex(ValueError, "commit"):
                    _validate_bound_run_files(
                        metadata, expected_project_commit="b" * 40,
                    )
                _validate_bound_run_files(
                    metadata,
                    expected_project_commit="b" * 40,
                    compatible_project_commit="a" * 40,
                )
                with self.assertRaisesRegex(ValueError, "commit"):
                    _validate_bound_run_files(
                        metadata,
                        expected_project_commit="b" * 40,
                        compatible_project_commit="c" * 40,
                    )
                with self.assertRaisesRegex(ValueError, "환경"):
                    _validate_bound_run_files(
                        metadata,
                        expected_project_commit="a" * 40,
                        expected_environment={
                            "runtime_snapshot": {"sha256": "d" * 64},
                        },
                    )

    def test_resume_source_is_limited_to_the_direct_recovery_commit(self):
        self.assertTrue(hasattr(run_dev18_tuning, "_compatible_resume_source"))
        source_commit = run_dev18_tuning.DEV18_RECOVERY_SOURCE_COMMIT
        parent_commit = getattr(
            run_dev18_tuning, "DEV18_RECOVERY_PARENT_COMMIT", None,
        )
        self.assertEqual(
            parent_commit, "6d5bcafda7a10fa6247f9cc32fe35d5061a286fa",
        )
        changed_files = "\n".join(sorted({
            "docs/lead/lightning_studio.md",
            "docs/lead/next_session.md",
            "docs/lead/plan_v5.md",
            "docs/lead/process_0_preverify.md",
            "tests/checks/check_dev18_resources.py",
            "tests/checks/reset_lightning_dev18.py",
            "tests/checks/run_lightning_dev18.py",
            "tests/ghl_main/run_dev18_tuning.py",
            "tests/unit/test_dev18_tuning.py",
            "tests/unit/test_lightning_dev18.py",
            "tests/unit/test_lightning_dev18_resource_tools.py",
        })) + "\n"
        with patch(
            "tests.ghl_main.run_dev18_tuning.subprocess.check_output",
            side_effect=[parent_commit + "\n", changed_files],
        ):
            self.assertEqual(
                run_dev18_tuning._compatible_resume_source(), source_commit,
            )
        with patch(
            "tests.ghl_main.run_dev18_tuning.subprocess.check_output",
            return_value="f" * 40 + "\n",
        ):
            self.assertIsNone(run_dev18_tuning._compatible_resume_source())
        with patch(
            "tests.ghl_main.run_dev18_tuning.subprocess.check_output",
            side_effect=[parent_commit + "\n", changed_files + "unexpected.py\n"],
        ):
            self.assertIsNone(run_dev18_tuning._compatible_resume_source())

    def test_only_exact_series13_gdn_oom_gets_one_recovery_attempt(self):
        self.assertTrue(hasattr(run_dev18_tuning, "_authorized_oom_recovery_row"))
        row = {
            "series": "13", "model": "GDN", "config_id": "c1168c94d4dfc",
            "physical_ratio": "10", "seed": "0", "score_variant": "",
            "status": "failed",
            "status_reason": "OutOfMemoryError: CUDA out of memory. Tried to allocate 3.82 GiB",
            "budget_id": "b5367ad431093", "retry_count": "3",
        }
        authorized = run_dev18_tuning._authorized_oom_recovery_row(
            [row],
            budget_id="b5367ad431093",
            compatible_project_commit=run_dev18_tuning.DEV18_RECOVERY_SOURCE_COMMIT,
        )
        self.assertEqual(authorized, row)
        for changed in (
            {**row, "series": "12"},
            {**row, "config_id": "cf1a967db6cfe"},
            {**row, "status_reason": "RuntimeError: unrelated"},
            {**row, "retry_count": "4"},
        ):
            with self.subTest(changed=changed):
                self.assertIsNone(run_dev18_tuning._authorized_oom_recovery_row(
                    [changed],
                    budget_id="b5367ad431093",
                    compatible_project_commit=run_dev18_tuning.DEV18_RECOVERY_SOURCE_COMMIT,
                ))
        self.assertIsNone(run_dev18_tuning._authorized_oom_recovery_row(
            [row],
            budget_id="b5367ad431093",
            compatible_project_commit=None,
        ))

    def test_oom_recovery_receipt_preserves_original_failure_idempotently(self):
        self.assertTrue(hasattr(run_dev18_tuning, "_write_oom_recovery_receipt"))
        row = {
            "series": "13", "model": "GDN", "config_id": "c1168c94d4dfc",
            "physical_ratio": "10", "seed": "0", "score_variant": "",
            "status": "failed", "status_reason": "OutOfMemoryError: CUDA out of memory.",
            "budget_id": "b5367ad431093", "retry_count": "3",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.json"
            run_dev18_tuning._write_oom_recovery_receipt(
                path, row, recovery_project_commit="b" * 40,
            )
            first = json.loads(path.read_text(encoding="utf-8"))
            run_dev18_tuning._write_oom_recovery_receipt(
                path, row, recovery_project_commit="b" * 40,
            )
            second = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first, second)
            self.assertEqual(first["original_failure"], row)
            self.assertEqual(
                first.get("failed_project_commit"),
                "6d5bcafda7a10fa6247f9cc32fe35d5061a286fa",
            )
            self.assertEqual(first["authorized_retry_count"], 4)
            with self.assertRaisesRegex(ValueError, "복구 영수증"):
                run_dev18_tuning._write_oom_recovery_receipt(
                    path, {**row, "seed": "1"},
                    recovery_project_commit="b" * 40,
                )

    def test_allocator_recovery_allows_exactly_retry_four(self):
        attempt_limit = getattr(
            run_dev18_tuning, "_authorized_attempt_limit", None,
        )
        self.assertIsNotNone(attempt_limit)
        if attempt_limit is None:
            return
        self.assertEqual(attempt_limit(3, recovery_row=None), 3)
        self.assertEqual(attempt_limit(3, recovery_row={"retry_count": "3"}), 5)

    def test_run_snapshot_records_and_recovery_checks_fixed_execution_policy(self):
        spec = {
            "model": "TimeRCD",
            "hyperparameters": {"context_length": 5000},
        }
        inputs = {
            "input_identity": {"sha256": "b" * 64},
            "source_ranges": {"test_sessions": ((1, 2),)},
        }
        expected_policy = {
            "context_length": 5000,
            "attention_query_chunk_size": 64,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "tests.ghl_main.run_dev18_tuning._git_head", return_value="a" * 40,
        ):
            root = Path(directory)
            snapshot_path = _write_run_snapshot(root, spec, inputs, {"device": "cuda"})
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["execution_policy"], expected_policy)

            tspulse_path = _write_run_snapshot(
                root / "tspulse",
                {
                    "model": "TSPulse",
                    "hyperparameters": {
                        "context_length": 512, "aggregation_window": 64,
                    },
                },
                inputs,
                {"device": "cuda"},
            )
            tspulse_snapshot = json.loads(tspulse_path.read_text(encoding="utf-8"))
            self.assertEqual(tspulse_snapshot["execution_policy"], {
                "context_length": 512,
                "aggregation_window": 64,
                "batch_size": 32,
            })

            snapshot["execution_policy"]["attention_query_chunk_size"] = 32
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            metadata = {
                "run_snapshot": {
                    "file": snapshot_path.name,
                    "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                },
                "training_files": {},
            }
            with patch("tests.ghl_main.run_dev18_tuning.REPOSITORY_ROOT", root):
                with self.assertRaisesRegex(ValueError, "execution policy"):
                    _validate_bound_run_files(metadata)

    def test_run_snapshot_accepts_json_equivalent_registered_spec(self):
        spec = {
            "model": "MWVAR",
            "hyperparameters": {},
            "score_variants": ("",),
        }
        inputs = {
            "input_identity": {"sha256": "b" * 64},
            "source_ranges": {"test_sessions": ((1, 2),)},
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "tests.ghl_main.run_dev18_tuning._git_head", return_value="a" * 40,
        ):
            root = Path(directory)
            snapshot_path = _write_run_snapshot(
                root, spec, inputs, {"device": "cuda"},
            )
            metadata = {
                "run_snapshot": {
                    "file": snapshot_path.name,
                    "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                },
                "training_files": {},
            }
            with patch("tests.ghl_main.run_dev18_tuning.REPOSITORY_ROOT", root):
                _validate_bound_run_files(metadata, expected_spec=spec)

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

    def test_cpu_scoring_worker_count_respects_cpu_memory_and_pending_work(self):
        gibibyte = 1024 ** 3
        resolve = run_dev18_tuning._resolve_score_workers

        self.assertEqual(resolve(
            0, pending_count=100, cpu_count=32,
            available_memory_bytes=10 * gibibyte,
        ), 8)
        self.assertEqual(resolve(
            12, pending_count=3, cpu_count=32,
            available_memory_bytes=10 * gibibyte,
        ), 3)
        with self.assertRaisesRegex(ValueError, "workers"):
            resolve(-1, pending_count=1)

    def test_cpu_postprocessing_keeps_static_gate_without_gpu_runtime(self):
        registry_sha256 = "a" * 64
        budget = {
            "budget_id": "b123456789abc", "physical_execution_count": 65,
            "primary_logical_score_row_count": 89, "seal_status": "sealed",
            "execution_readiness_status": "ready", "pending_execution_models": [],
            "attestation": {"config_registry_sha256": registry_sha256},
        }
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            (data_root / "tuning").mkdir()
            with (
                patch.object(
                    run_dev18_tuning, "load_model_registry_with_sha",
                    return_value=({}, registry_sha256),
                ),
                patch.object(run_dev18_tuning, "validate_primary_hpo_seal"),
                patch.object(
                    run_dev18_tuning, "_load_current_feasibility",
                    return_value=(None, None, {}),
                ),
                patch.object(
                    run_dev18_tuning, "build_equal_trial_budget",
                    return_value=budget,
                ),
                patch.object(run_dev18_tuning, "_read_json", return_value=budget),
                patch.object(
                    run_dev18_tuning, "_validate_ell_max",
                    return_value={"ell_max_id": "b" * 64},
                ),
                patch.object(
                    run_dev18_tuning, "validate_vus_evidence",
                    return_value={"evaluator_sha256": "c" * 64},
                ),
                patch.object(run_dev18_tuning, "verify_runtime_versions") as versions,
                patch.object(run_dev18_tuning, "_validate_checkpoint_report") as checkpoint,
                patch.object(
                    run_dev18_tuning, "collect_runtime_environment_identity",
                ) as runtime,
            ):
                result = run_dev18_tuning.prepare_tuning(
                    data_root=data_root, require_clean=False,
                    require_execution_environment=False,
                )

        self.assertFalse(result["execution_environment_verified"])
        self.assertEqual(result["checkpoint_models"], [])
        versions.assert_called_once()
        checkpoint.assert_not_called()
        runtime.assert_not_called()

    def test_vus_checkpoint_is_atomic_and_rejects_tampering(self):
        identity = {
            "schema_version": 1,
            "manifest_key": ["01", "M1", "c1", "100", "0", ""],
            "score_sha256": "a" * 64,
            "evaluator_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "score.json"
            run_dev18_tuning._write_score_checkpoint(path, identity, 0.625)
            self.assertEqual(
                run_dev18_tuning._load_score_checkpoint(path, identity), 0.625,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["vus_pr"] = 0.75
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                run_dev18_tuning._load_score_checkpoint(path, identity)

    def test_primary_score_reuses_checkpoint_after_revalidating_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_path = root / "score.npy"
            metadata_path = root / "score.meta.json"
            score_path.write_bytes(b"score")
            metadata_path.write_text("{}", encoding="utf-8")
            manifest_row = {
                "series": "01", "family": "A", "tier": "t1", "model": "M1",
                "config_id": "c1", "physical_ratio": "100", "seed": "0",
                "score_variant": "", "score_file": score_path.name,
                "score_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
                "metadata_file": metadata_path.name,
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            }
            task = {
                "manifest_row": manifest_row, "entry": {"family": "A"},
                "budget_id": "b123456789abc", "budget_sha256": "a" * 64,
                "input_manifest_sha256": "b" * 64,
                "environment_sha256": "e" * 64, "project_commit": "f" * 40,
                "evaluator_sha256": "c" * 64, "ell_max_id": "d" * 64,
                "l_max_samples": 1, "n_thresholds": 250,
                "checkpoint_directory": str(root / "checkpoints"),
            }
            scores = run_dev18_tuning.numpy.array([0.1, 0.9])
            labels = run_dev18_tuning.numpy.array([0, 1])
            info = {
                "dataset": "DEV18", "series": 1, "model": "M1", "tier": "t1",
                "ratio": 100, "seed": 0, "smoothing_kind": "raw",
                "norm_kind": "trainnorm", "channels": False,
            }
            metadata = {
                "dataset": "DEV18", "series": 1, "model": "M1", "tier": "t1",
                "ratio": 100, "seed": 0, "config_id": "c1",
                "score_variant": None, "label_slice": [0, 2],
            }
            run_dev18_tuning._initialize_score_worker({"01": labels})
            with (
                patch("tests.ghl_main.run_dev18_tuning.REPOSITORY_ROOT", root),
                patch(
                    "src.채점기.parser.load_and_validate_score",
                    return_value=(scores, info, metadata),
                ) as loader,
                patch("tests.ghl_main.run_dev18_tuning.vus_pr", return_value=0.625) as scorer,
            ):
                first = run_dev18_tuning._score_primary_row(task)
                second = run_dev18_tuning._score_primary_row(task)

            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(scorer.call_count, 1)
            self.assertEqual(loader.call_count, 2)

    def test_one_physical_vus_score_expands_to_all_logical_ratios(self):
        manifest_row = {
            "series": "01", "family": "A", "tier": "t1", "model": "M1",
            "config_id": "c1", "physical_ratio": "100", "seed": "0",
            "score_variant": "", "score_file": "score.npy",
            "score_sha256": "a" * 64,
        }
        rows = run_dev18_tuning._expand_primary_score(
            manifest_row,
            logical_ratios=(5, 10, 20),
            normalization="trainnorm",
            vus_pr_value=0.625,
            evaluator_sha256="b" * 64,
            ell_max_id="c" * 64,
        )

        self.assertEqual([row["ratio"] for row in rows], [5, 10, 20])
        self.assertEqual([row["vus_pr"] for row in rows], [0.625] * 3)

    def test_cpu_finish_cli_can_import_project_packages(self):
        script = Path(__file__).parents[1] / "checks" / "finish_lightning_dev18.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=script.parents[2],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
