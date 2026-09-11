"""Dev18 단일 진입점의 선택·보고 계약을 검증한다."""

import csv
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from tests.ghl_main import run_dev18_tuning
from tests.ghl_main.run_dev18_tuning import (
    _evidence_directory,
    _load_completion_receipt,
    _load_score_manifest,
    _draw_ratio_adaptive_selection,
    _replace_manifest_rows,
    _write_completion_receipt,
    _write_run_snapshot,
    _validate_bound_run_files,
    _validate_primary_manifest_rows,
    build_ratio_adaptive_plot_data,
    build_final_membership_rows,
    finish_selection_from_ledger,
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
    def test_conditional_membership_resolves_native_tspulse_head_by_target_family(self):
        policy = {"model": "TSPulse", "config_id": "c96", "group_id": "group",
                  "score_variant": "family_selected", "score_variant_by_family": {"GHL": "fft"},
                  "score_variant_fallback": "time"}
        registry = {"models": {"TSPulse": {"target_use": "strict_zero_shot"}}}
        for role, expected in (("ghl25_final", "fft"), ("train1_to_test1", "time"),
                               ("train1_train2_to_test2", "time")):
            with self.subTest(split_role=role):
                row = run_dev18_tuning._conditional_membership_row(
                    "model_ratio", {"split_role": role, "series": "01"}, 40,
                    "TSPulse", "t3", policy, "within_dev_support", True, "", registry,
                )
                self.assertEqual(row["score_variant"], expected)
                self.assertEqual(row["physical_ratio"], 100)
                self.assertEqual(row["evaluation_ratio"], 40)

    def test_csv_update_preserves_previous_file_on_write_or_sync_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.csv"
            previous = b"name\r\nprevious\r\n"
            path.write_bytes(previous)
            with self.assertRaises(ValueError):
                run_dev18_tuning._write_csv(path, [{"name": "new"}, {"unexpected": "bad"}], ["name"])
            self.assertEqual(path.read_bytes(), previous)
            with patch.object(run_dev18_tuning.os, "fsync", side_effect=OSError("sync failed")):
                with self.assertRaises(OSError):
                    run_dev18_tuning._write_csv(path, [{"name": "new"}])
            self.assertEqual(path.read_bytes(), previous)
            with patch.object(csv.DictWriter, "writerows", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_dev18_tuning._write_csv(path, [{"name": "new"}])
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            run_dev18_tuning._write_csv(path, [{"name": "new"}])
            with path.open(encoding="utf-8", newline="") as saved:
                self.assertEqual(list(csv.DictReader(saved)), [{"name": "new"}])
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_training_files_preserve_scaler_and_bind_its_size_and_hash(self):
        scaler_state = {
            "data_min": [1.0, 10.0], "data_max": [3.0, 14.0],
            "scale": [0.5, 0.25], "offset": [-0.5, -2.5], "sample_count": 20,
        }
        training_log = {
            "selected_iteration": 1, "iterations_completed": 2,
            "best_training_loss": 0.25,
            "loss_history": [{"iteration": 1, "total_loss": 0.25}, {"iteration": 2, "total_loss": 0.5}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(run_dev18_tuning, "REPOSITORY_ROOT", root):
                size, files = run_dev18_tuning._save_training_files(root, {
                    "checkpoint": None, "scaler_state": scaler_state,
                    "training_log": training_log,
                    "timing": {"fit_seconds": 1.0},
                })
            path = root / files["scaler_state"]["file"]
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), scaler_state)
            self.assertEqual(files["scaler_state"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(size, path.stat().st_size)
            self.assertEqual(files["scaler_state"]["bytes"], size)
            log_path = root / files["training_log"]["file"]
            self.assertEqual(json.loads(log_path.read_text(encoding="utf-8")), training_log)
            self.assertEqual(files["training_log"]["sha256"], hashlib.sha256(log_path.read_bytes()).hexdigest())
            self.assertEqual(files["training_log"]["bytes"], log_path.stat().st_size)
            self.assertEqual(list((root / "training").glob("*.tmp")), [])

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

        def load(path, current_budget=budget):
            return load_trial_score_ledger(
                path, current_budget,
                expected_evaluator_sha256="b" * 64,
                expected_ell_max_id="ell-v1",
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.csv"
            write_ledger(path, rows)
            loaded = load(path)
            self.assertIsInstance(loaded[0]["ratio"], int)
            self.assertIsInstance(loaded[0]["seed"], int)
            self.assertIsInstance(loaded[0]["vus_pr"], float)

            for invalid in ("nan", "inf", "-0.01", "1.01"):
                with self.subTest(vus_pr=invalid):
                    write_ledger(path, [{**rows[0], "vus_pr": invalid}, *rows[1:]])
                    with self.assertRaisesRegex(ValueError, "VUS-PR"):
                        load(path)

            write_ledger(path, [{**row, "evaluator_sha256": "c" * 64} for row in rows])
            with self.assertRaisesRegex(ValueError, "evaluator"):
                load(path)

            write_ledger(path, [{**row, "ell_max_id": "ell-v2"} for row in rows])
            with self.assertRaisesRegex(ValueError, "ell_max"):
                load(path)

            write_ledger(path, rows + [rows[0]])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load(path)

            write_ledger(path, [{**row, "status": "failed"} for row in rows])
            with self.assertRaisesRegex(ValueError, "complete"):
                load(path)

            write_ledger(path, [{**row, "evaluator_sha256": "c" * 64} if index == 0 else row
                                for index, row in enumerate(rows)])
            with self.assertRaisesRegex(ValueError, "evaluator"):
                load(path)

            write_ledger(path, [{**row, "ratio": "40"} if index == 0 else row
                                for index, row in enumerate(rows)])
            with self.assertRaisesRegex(ValueError, "budget"):
                load(path)

            write_ledger(path, [{**row, "family": "C"} if index == 1 else row
                                for index, row in enumerate(rows)])
            with self.assertRaisesRegex(ValueError, "series.*family"):
                load(path)

            write_ledger(path, [{**row, "tier": "t2"} if index == 0 else row
                                for index, row in enumerate(rows)])
            with self.assertRaisesRegex(ValueError, "tier"):
                load(path)

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
            self.assertEqual(len(load(path, production_budget)), 18)
            write_ledger(path, [{**row, "family": "F1"} for row in production_rows])
            with self.assertRaisesRegex(ValueError, "10개 family"):
                load(path, production_budget)

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

    def test_final_plot_data_contains_only_tier_paths_and_pca_reference(self):
        rows = [{
            "series": series, "family": family, "tier": "t1",
            "model": "PCA_LEGACY", "config_id": config_id, "ratio": 100,
            "seed": seed, "score_variant": "raw", "vus_pr": values[family],
            "status": "complete",
        } for config_id, values in (
            ("pca_selected", {"A": 0.1, "B": 0.467338}),
            ("pca_other", {"A": 0.9, "B": 0.9}),
        ) for series, family in (
            ("01", "A"), ("02", "B"), ("03", "B"), ("04", "B"),
        )
            for seed in (0, 1)]
        selection = {
            "model_fixed": [{
                "model": "PCA_LEGACY", "config_id": "pca_selected",
                "score_variant": "raw", "selection_status": "selected",
            }],
            "tier_adaptive": [
                {"tier": "t1", "ratio": 100, "selected_model": "MWVAR",
                 "selection_score": 0.41, "selection_status": "selected"},
                {"tier": "t2", "ratio": 5, "selected_model": "",
                 "selection_score": None, "selection_status": "unavailable"},
                {"tier": "t3", "ratio": 5, "selected_model": "TSPulse",
                 "selection_score": 0.52, "selection_status": "selected"},
                {"tier": "t1", "ratio": 5, "selected_model": "MWVAR",
                 "selection_score": 0.40, "selection_status": "selected"},
                {"tier": "t2", "ratio": 40, "selected_model": "PaAno",
                 "selection_score": 0.63, "selection_status": "selected"},
            ],
        }

        data = build_ratio_adaptive_plot_data(rows, selection)

        self.assertEqual(
            [(row["tier"], row["ratio"], row["model"])
             for row in data["points"]],
            [
                ("t1", 5, "MWVAR"), ("t1", 100, "MWVAR"),
                ("t2", 40, "PaAno"), ("t3", 5, "TSPulse"),
            ],
        )
        self.assertEqual(
            [row["selection_score"] for row in data["points"]],
            [0.40, 0.41, 0.63, 0.52],
        )
        self.assertEqual(data["unavailable"], [{"tier": "t2", "ratio": 5}])
        self.assertAlmostEqual(data["pca_reference_vus_pr"], 0.283669)

    def test_final_selection_plot_keeps_one_clean_axis(self):
        figure = run_dev18_tuning.matplotlib.figure.Figure()
        axis = figure.subplots()
        data = {
            "points": [
                {"tier": "t1", "ratio": 5, "model": "MWVAR",
                 "selection_score": 0.40},
                {"tier": "t1", "ratio": 10, "model": "MWVAR",
                 "selection_score": 0.42},
                {"tier": "t2", "ratio": 10, "model": "GDN",
                 "selection_score": 0.51},
                {"tier": "t2", "ratio": 20, "model": "GDN",
                 "selection_score": 0.53},
                {"tier": "t2", "ratio": 40, "model": "PaAno",
                 "selection_score": 0.55},
                {"tier": "t2", "ratio": 60, "model": "PaAno",
                 "selection_score": 0.54},
                {"tier": "t3", "ratio": 5, "model": "TSPulse",
                 "selection_score": 0.48},
                {"tier": "t3", "ratio": 10, "model": "TSPulse",
                 "selection_score": 0.48},
            ],
            "unavailable": [{"tier": "t2", "ratio": 5}],
            "pca_reference_vus_pr": 0.283669,
        }

        _draw_ratio_adaptive_selection(axis, data)

        lines = {line.get_label(): line for line in axis.lines}
        self.assertEqual(
            set(lines),
            {"Tier 1", "Tier 2", "Tier 3", "PCA q100 reference"},
        )
        self.assertEqual(lines["PCA q100 reference"].get_linestyle(), "--")
        self.assertEqual(list(axis.get_xticks()), [5, 10, 20, 40, 60, 80, 100])
        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(len(axis.tables), 0)
        self.assertTrue(any(line.get_visible() for line in axis.get_ygridlines()))
        self.assertFalse(any(line.get_visible() for line in axis.get_xgridlines()))
        annotations = {annotation.get_text(): annotation for annotation in axis.texts}
        self.assertEqual(
            set(annotations),
            {"MWVAR", "GDN", "PaAno", "TSPulse", "Tier 2 unavailable"},
        )
        self.assertEqual(len(axis.texts), 5)
        self.assertTrue(all(text.get_fontsize() == 7 for text in axis.texts))
        self.assertGreater(annotations["MWVAR"].get_position()[1], 0)
        self.assertLess(annotations["TSPulse"].get_position()[1], 0)
        for label in (
            axis.get_title(), axis.get_xlabel(), axis.get_ylabel(),
            *(text.get_text() for text in axis.get_legend().get_texts()),
        ):
            label.encode("ascii")

    def test_report_uses_clean_adaptive_selection_axis(self):
        rows = _rows() + [{
            "series": series, "family": family, "tier": "t1",
            "model": "PCA_LEGACY", "config_id": "pca", "ratio": 100,
            "seed": seed, "score_variant": "raw", "vus_pr": 0.283669,
            "status": "complete",
        } for series, family in (("01", "A"), ("02", "B")) for seed in (0, 1)]
        selection = select_tuning_policies(
            _rows(), self.registry, self.budget, evaluator_sha256="d" * 64,
        )
        selection["model_fixed"].append({
            "tier": "t1", "model": "PCA_LEGACY", "q_support": [100],
            "config_id": "pca", "hyperparameters": {}, "j_fixed": 0.283669,
            "score_variant": "raw", "hpo_regime": "equal_trial",
            "budget_id": "b123456789abc", "source_commit": "p" * 40,
            "source_checkpoint_sha256": "none", "selection_status": "selected",
            "selection_reason": "reference only",
        })
        captured = []

        def capture(figure, path, *_args, **_kwargs):
            if Path(path).name == "selection.png":
                captured.append(figure)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            run_dev18_tuning.matplotlib.figure.Figure, "savefig",
            autospec=True, side_effect=capture,
        ):
            write_selection_reports(rows, selection, Path(directory))

        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0].axes), 1)
        self.assertEqual(
            {line.get_label() for line in captured[0].axes[0].lines},
            {"Tier 1", "Tier 2", "Tier 3", "PCA q100 reference"},
        )

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
            previous = path.read_bytes()
            replacement_rows = [{**row, "retry_count": 1} for row in rows]
            for error_type in (OSError, KeyboardInterrupt):
                with self.subTest(error=error_type.__name__):
                    with patch.object(Path, "replace", side_effect=error_type):
                        with self.assertRaises(error_type):
                            _write_completion_receipt(path, replacement_rows)
                    self.assertEqual(path.read_bytes(), previous)
                    self.assertEqual(list(Path(directory).iterdir()), [path])
            _write_completion_receipt(path, replacement_rows)
            recovered = _load_completion_receipt(
                path, spec=spec, panel_row=panel, series=3,
                budget_id="b123456789abc",
            )

        self.assertEqual(recovered, replacement_rows)

    def test_metadata_update_cleans_interrupted_write_before_retry(self):
        spec = {"model": "MWVAR", "tier": "t1", "config_id": "c123456789abc",
                "ratio": 100, "seed": 0, "common_recipe": {}}
        panel = {"primary_score_variants": [""], "diagnostic_score_variants": []}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            snapshot_path = root / "run_snapshot.json"
            snapshot_path.write_text('{"project_commit":"sealed"}', encoding="utf-8")
            metadata_path = root / "score.meta.json"
            previous = b'{"model":"MWVAR"}'
            metadata_path.write_bytes(previous)
            score_path = root / "DEV18__01__MWVAR__t1__r100__s0__raw__trainnorm.npy"
            score_path.write_bytes(b"saved score")
            temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
            stack.enter_context(patch.object(run_dev18_tuning, "REPOSITORY_ROOT", root))
            for name, value in (("_require_same_worktree", None), ("_record_seed_state", None),
                                ("_save_training_files", (0, {})), ("_validate_bound_run_files", None)):
                stack.enter_context(patch.object(run_dev18_tuning, name, return_value=value))
            stack.enter_context(patch("src.common.execution_evidence.build_execution_evidence", return_value={}))
            stack.enter_context(patch("src.common.save_model_artifacts.save_execution_result", return_value={
                "metadata_path": str(metadata_path), "score_paths": [str(score_path)],
            }))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory", return_value=root))
            stack.enter_context(patch("tests.ghl_main.check_registered_outputs.check_registered_output"))

            def save_result():
                return run_dev18_tuning._save_run_result(
                    {"split": None, "timing": {}}, spec, panel,
                    {"family": "synthetic", "test_sessions": []}, series=1,
                    snapshot_path=snapshot_path, peak_memory_mb=None,
                    input_manifest_path=root / "manifest.yaml", retry_count=0, budget_id="sealed",
                    execution_attempt={"run_id": "attempt", "history_file": "attempt.json"},
                )

            original_replace = Path.replace
            for error_type in (OSError, KeyboardInterrupt):
                def interrupt_metadata_replace(source, target):
                    if source == temporary_metadata:
                        raise error_type
                    return original_replace(source, target)

                with self.subTest(error=error_type.__name__):
                    with patch.object(Path, "replace", autospec=True, side_effect=interrupt_metadata_replace):
                        with self.assertRaises(error_type):
                            save_result()
                    self.assertEqual(metadata_path.read_bytes(), previous)
                    self.assertFalse(temporary_metadata.exists())
                    self.assertFalse((root / "completion.json").exists())
            rows = save_result()
            self.assertEqual(json.loads((root / "completion.json").read_text(encoding="utf-8")), rows)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["execution_attempt"]["run_id"], "attempt")
            self.assertFalse(temporary_metadata.exists())

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
        saved_plots = set()

        def capture(_figure, path, *_args, **_kwargs):
            saved_plots.add(Path(path).name)

        plot_data = {
            "points": [{"tier": "t1", "ratio": 5, "model": "M1",
                        "selection_score": 0.8}],
            "unavailable": [], "pca_reference_vus_pr": 0.3,
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            run_dev18_tuning.matplotlib.figure.Figure, "savefig",
            autospec=True, side_effect=capture,
        ), patch.object(
            run_dev18_tuning, "build_ratio_adaptive_plot_data",
            return_value=plot_data,
        ):
            output = Path(directory)
            write_selection_reports(_rows(), selection, output)
            for name in (
                "M1.csv", "Tier1.csv", "M0.csv", "models.csv",
                "selection.csv", "family_lofo.csv",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue({
                "M1.png", "Tier1.png", "M0.png", "models.png", "selection.png",
            }.issubset(saved_plots))
            with (output / "selection.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["selected_model"], "M1")
            self.assertTrue(rows[0]["selection_reason"])
            with (output / "Tier1.csv").open(encoding="utf-8", newline="") as file:
                tier_rows = list(csv.DictReader(file))
            selected = next(row for row in tier_rows if row["selected_model"] == "True")
            self.assertTrue(selected["selected_hyperparameters"])
            self.assertTrue(selected["selection_reason"])

    def test_report_images_keep_only_compact_ascii_decision_information(self):
        selection = select_tuning_policies(
            _rows(), self.registry, self.budget, evaluator_sha256="d" * 64,
        )
        captured = {}

        def capture(figure, path, *_args, **_kwargs):
            captured[Path(path).name] = figure

        def visible_text(figure):
            values = []
            for axis in figure.axes:
                values.extend((axis.get_title(), axis.get_xlabel(), axis.get_ylabel()))
                values.extend(text.get_text() for text in axis.texts)
                values.extend(text.get_text() for text in axis.get_yticklabels())
                legend = axis.get_legend()
                if legend is not None:
                    values.extend(text.get_text() for text in legend.get_texts())
                for table in axis.tables:
                    values.extend(
                        cell.get_text().get_text()
                        for cell in table.get_celld().values()
                    )
            return [value for value in values if value]

        plot_data = {
            "points": [{"tier": "t1", "ratio": 5, "model": "M1",
                        "selection_score": 0.8}],
            "unavailable": [], "pca_reference_vus_pr": 0.3,
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            run_dev18_tuning.matplotlib.figure.Figure, "savefig",
            autospec=True, side_effect=capture,
        ), patch.object(
            run_dev18_tuning, "build_ratio_adaptive_plot_data",
            return_value=plot_data,
        ):
            output = Path(directory)
            write_selection_reports(_rows(), selection, output)
            model_csv = (output / "models.csv").read_text(encoding="utf-8")

        self.assertEqual(
            set(captured),
            {"M0.png", "M1.png", "M2.png", "Tier1.png", "models.png", "selection.png"},
        )
        for figure in captured.values():
            text = " ".join(visible_text(figure))
            text.encode("ascii")
            self.assertFalse({"c0", "c1", "c2", "c3"} & set(text.split()))
            self.assertNotIn("{", text)

        model_axis = captured["M1.png"].axes[0]
        self.assertEqual(
            {text.get_text() for text in model_axis.get_legend().get_texts()},
            {"Selected recipe", "Other tested recipe"},
        )
        self.assertIn("Same score reused across ratios", visible_text(captured["M1.png"]))
        self.assertLessEqual(model_axis.get_xlim()[0], 5)
        self.assertGreaterEqual(model_axis.get_xlim()[1], 100)
        self.assertIn(
            "No eligible recipe covers the full tuning panel.",
            visible_text(captured["M0.png"]),
        )

        tier_axis = captured["Tier1.png"].axes[0]
        self.assertEqual(len(tier_axis.lines), 0)
        self.assertEqual(tier_axis.get_xlabel(), "Family-LOFO VUS-PR")
        self.assertGreater(len(tier_axis.patches), 0)

        table = captured["models.png"].axes[0].tables[0]
        self.assertEqual(
            [table.get_celld()[(0, column)].get_text().get_text() for column in range(6)],
            ["Tier", "Model", "Status", "VUS-PR", "Prefix support", "Key point"],
        )
        self.assertIn(selection["model_fixed"][0]["selection_reason"], model_csv)

    def test_reports_separate_adaptive_audit_and_transition_tables(self):
        selection = select_tuning_policies(
            _rows(), self.registry, self.budget, evaluator_sha256="d" * 64,
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            run_dev18_tuning.pyplot, "subplots",
            side_effect=AssertionError("write_plots=False에서 figure를 만들면 안 된다"),
        ):
            output = Path(directory)
            write_selection_reports(_rows(), selection, output, write_plots=False)
            expected = {
                "ratio_adaptive_selection.csv",
                "tier_ratio_candidate_audit.csv",
                "tier_policy_transitions.csv",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            self.assertEqual(list(output.glob("*.png")), [])
            with (output / "tier_ratio_candidate_audit.csv").open(
                encoding="utf-8", newline="",
            ) as input_file:
                audit = list(csv.DictReader(input_file))
            self.assertEqual(
                set(audit[0]),
                {
                    "tier", "ratio", "model", "config_id", "score_variant",
                    "eligibility", "family_lofo_vus_pr", "selected",
                    "reason_code", "reason",
                },
            )

    def _prepare_selection_fixture(self):
        registry = json.loads(json.dumps(self.registry))
        registry["selection"].update(selection_status="ready", primary_score_variants={"TSPulse": ["raw_max"]})
        for model in registry["models"].values():
            model["execution_status"] = "ready"
        budget = {key: value for key, value in json.loads(json.dumps(self.budget)).items()
                  if key != "budget_id"}
        budget.update(seeds=[0, 1], selection_rule_id=registry["selection"]["selection_rule_id"],
                      registry_space_sha256=run_dev18_tuning.registry_space_sha256(registry))
        digest = hashlib.sha256(run_dev18_tuning._json(budget).encode("utf-8")).hexdigest()
        budget.update(budget_sha256=digest, budget_id="b" + digest[:12], seal_status="sealed",
                      execution_readiness_status="ready", pending_execution_models=[],
                      attestation={"config_registry_sha256": "f" * 64})
        registry["selection"]["budget_id"] = budget["budget_id"]
        evaluator, ell_max_id = "d" * 64, "ell-test"
        for name, value in (
            ("load_model_registry_with_sha", (registry, "f" * 64)), ("_read_json", budget),
            ("validate_vus_evidence", {"evaluator_sha256": evaluator}),
            ("_validate_ell_max", {"ell_max_id": ell_max_id}),
        ):
            self.enterContext(patch.object(run_dev18_tuning, name, return_value=value))
        return registry, "f" * 64, budget, evaluator, ell_max_id

    def test_selection_only_reuses_ledger_without_vus_or_score_arrays(self):
        registry, registry_sha256, budget, evaluator_sha256, ell_max_id = self._prepare_selection_fixture()
        scores = {"M1": .4, "M2": .2, "M0": .0}
        ledger_rows = []
        for series, family in (("01", "A"), ("02", "B")):
            for panel in budget["model_panels"]:
                for config_index, config_id in enumerate(panel["selected_config_ids"]):
                    for ratio in panel["logical_ratios"]:
                        for seed in panel.get("seeds", budget["seeds"]):
                            for variant in panel["primary_score_variants"]:
                                ledger_rows.append({
                                    "series": series, "family": family,
                                    "tier": panel["tier"], "model": panel["model"],
                                    "config_id": config_id, "ratio": ratio,
                                    "seed": seed, "score_variant": variant,
                                    "normalization": "trainnorm",
                                    "vus_pr": scores[panel["model"]] + config_index / 1000,
                                    "score_file": "unused.npy", "score_sha256": "a" * 64,
                                    "evaluator_sha256": evaluator_sha256,
                                    "ell_max_id": ell_max_id, "status": "complete",
                                    "status_reason": "",
                                })

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger_path = root / "dev18_trial_score_ledger.csv"
            run_dev18_tuning._write_csv(
                ledger_path, ledger_rows, run_dev18_tuning.TRIAL_SCORE_LEDGER_FIELDS,
            )
            ledger_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

            def save_placeholder(_figure, path, *_args, **_kwargs):
                Path(path).write_bytes(b"png placeholder")

            with (
                patch.object(
                    run_dev18_tuning, "load_model_registry_with_sha",
                    return_value=(registry, registry_sha256),
                ),
                patch.object(
                    run_dev18_tuning, "DEV18_RECOVERY_BUDGET_ID", "test-budget",
                ),
                patch.object(run_dev18_tuning, "_git_head", return_value="c" * 40),
                patch.object(
                    run_dev18_tuning, "_require_same_worktree",
                ) as worktree_guard,
                patch.object(
                    run_dev18_tuning, "build_trial_score_ledger",
                    side_effect=AssertionError("selection-only가 score를 다시 채점했다"),
                ) as ledger_builder,
                patch.object(
                    run_dev18_tuning, "vus_pr",
                    side_effect=AssertionError("selection-only가 VUS-PR을 호출했다"),
                ) as scorer,
                patch.object(
                    run_dev18_tuning, "_load_score_manifest",
                    side_effect=AssertionError("selection-only가 score manifest를 읽었다"),
                ) as manifest_loader,
                patch.object(
                    run_dev18_tuning.matplotlib.figure.Figure, "savefig",
                    autospec=True, side_effect=save_placeholder,
                ),
            ):
                result = finish_selection_from_ledger(
                    ledger_path, result_directory=root / "results",
                )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["ledger_sha256"], ledger_sha256)
            self.assertEqual(result["ledger_rows"], len(ledger_rows))
            self.assertEqual(result["membership_rows"], 105)
            self.assertEqual(len(result["selected_path"]), 7)
            self.assertEqual(hashlib.sha256(ledger_path.read_bytes()).hexdigest(), ledger_sha256)
            ledger_builder.assert_not_called()
            scorer.assert_not_called()
            manifest_loader.assert_not_called()
            self.assertEqual(worktree_guard.call_count, 2)
            output = root / "results"
            for name in (
                "model_fixed_policy.csv", "tier_fixed_policy.csv",
                "ratio_adaptive_selection.csv", "tier_ratio_candidate_audit.csv",
                "tier_policy_transitions.csv", "final_policy_membership.csv",
                "selection.png",
            ):
                self.assertTrue((output / name).is_file(), name)
            with (output / "final_policy_membership.csv").open(
                encoding="utf-8", newline="",
            ) as input_file:
                self.assertEqual(len(list(csv.DictReader(input_file))), 105)

    def test_selection_only_rejects_tampered_budget_before_output(self):
        registry, registry_sha256, budget, _, _ = self._prepare_selection_fixture()
        mutations = {
            "tie_rule": lambda changed: changed["tie_rule"].update(tolerance=0.5),
            "model_panels": lambda changed: changed["model_panels"][0].update(
                logical_ratios=[100],
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger_path = root / "dev18_trial_score_ledger.csv"
            run_dev18_tuning._write_csv(
                ledger_path, [], run_dev18_tuning.TRIAL_SCORE_LEDGER_FIELDS,
            )
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = json.loads(json.dumps(budget))
                    mutate(changed)
                    result_directory = root / name
                    with (
                        patch.object(
                            run_dev18_tuning, "load_model_registry_with_sha",
                            return_value=(registry, registry_sha256),
                        ),
                        patch.object(run_dev18_tuning, "_read_json", return_value=changed),
                        patch.object(run_dev18_tuning, "_git_head", return_value="c" * 40),
                        patch.object(run_dev18_tuning, "_require_same_worktree"),
                    ):
                        with self.assertRaisesRegex(ValueError, "budget.*봉인"):
                            finish_selection_from_ledger(
                                ledger_path, result_directory=result_directory,
                            )
                    self.assertFalse(result_directory.exists())

    def test_selection_only_rejects_invalid_ledger_before_output(self):
        _, _, budget, evaluator_sha256, ell_max_id = self._prepare_selection_fixture()
        panel = budget["model_panels"][0]
        row = {
            "series": "01", "family": "MSL", "tier": panel["tier"],
            "model": panel["model"], "config_id": panel["selected_config_ids"][0],
            "ratio": panel["logical_ratios"][0],
            "seed": panel.get("seeds", budget["seeds"])[0],
            "score_variant": panel["primary_score_variants"][0],
            "normalization": "trainnorm", "vus_pr": "0.5",
            "score_file": "unused.npy", "score_sha256": "a" * 64,
            "evaluator_sha256": evaluator_sha256, "ell_max_id": ell_max_id,
            "status": "complete", "status_reason": "",
        }
        cases = {
            "nonfinite": ({"vus_pr": "nan"}, "VUS-PR"),
            "outside": ({"vus_pr": "1.01"}, "VUS-PR"),
            "evaluator": ({"evaluator_sha256": "b" * 64}, "evaluator"),
            "ell_max": ({"ell_max_id": "ell-v2"}, "ell_max"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, (changes, message) in cases.items():
                with self.subTest(name=name):
                    ledger_path = root / f"{name}.csv"
                    run_dev18_tuning._write_csv(
                        ledger_path, [{**row, **changes}],
                        run_dev18_tuning.TRIAL_SCORE_LEDGER_FIELDS,
                    )
                    result_directory = root / f"{name}-results"
                    with patch.object(run_dev18_tuning, "_require_same_worktree"):
                        with self.assertRaisesRegex(ValueError, message):
                            finish_selection_from_ledger(
                                ledger_path, result_directory=result_directory,
                            )
                    self.assertFalse(result_directory.exists())

    def test_deployment_scenario_schema_has_no_cost_defaults(self):
        schema = json.loads(
            (Path(__file__).parents[2] / "configs" / "deployment_scenario.schema.json")
            .read_text(encoding="utf-8")
        )

        def assert_no_default(node):
            if isinstance(node, dict):
                self.assertNotIn("default", node)
                for value in node.values():
                    assert_no_default(value)
            elif isinstance(node, list):
                for value in node:
                    assert_no_default(value)

        assert_no_default(schema)
        self.assertEqual(schema["required"], ["transition_key"])
        expected_properties = {
            "transition_key", "currency", "analysis_horizon_hours",
            "observation_cost", "training_cost", "inference_cost",
            "memory_cost", "artifact_storage_cost", "false_alarm_cost",
            "miss_cost", "model_switch_cost", "validation_cost",
            "deployment_cost", "downtime_cost", "maximum_latency_seconds",
            "maximum_memory_mb", "maximum_artifact_bytes", "minimum_vus_pr",
        }
        self.assertEqual(set(schema["properties"]), expected_properties)
        for field in ("model_switch_cost", "validation_cost", "deployment_cost"):
            self.assertEqual(set(schema["properties"][field]["type"]), {"number", "null"})
            self.assertEqual(schema["properties"][field]["minimum"], 0)

    def test_direct_file_cli_can_import_project_packages(self):
        script = Path(__file__).parents[1] / "ghl_main" / "run_dev18_tuning.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=script.parents[2],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_selection_module_import_does_not_load_vus_evaluator(self):
        code = """
import builtins
real_import = builtins.__import__
def reject_evaluator(name, *args, **kwargs):
    if name == 'src.채점기.vus_pr':
        raise RuntimeError('selection import가 VUS evaluator를 읽었다')
    return real_import(name, *args, **kwargs)
builtins.__import__ = reject_evaluator
import tests.ghl_main.run_dev18_tuning
"""
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).parents[2],
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

    def test_scoring_failure_waits_for_running_worker_before_returning(self):
        from concurrent.futures import Future

        tasks = [{"estimated_cost": 1, "manifest_row": {
            "series": "01", "model": "M", "config_id": config,
            "physical_ratio": 100, "seed": 0, "score_variant": "",
        }} for config in ("c1", "c2")]
        for error in (RuntimeError("score failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                failed, running = Future(), Future()
                failed.set_exception(error)
                running.set_running_or_notify_cancel()

                def finish_workers(*, wait, cancel_futures=False):
                    if wait:
                        running.set_result("checkpoint saved")

                with patch.object(run_dev18_tuning, "_resolve_score_workers", return_value=2), \
                        patch.object(run_dev18_tuning.concurrent.futures, "ProcessPoolExecutor") as factory, \
                        patch.object(run_dev18_tuning.concurrent.futures, "as_completed", return_value=[failed]):
                    executor = factory.return_value
                    executor.submit.side_effect = [failed, running]
                    executor.shutdown.side_effect = finish_workers
                    with self.assertRaises(type(error)):
                        run_dev18_tuning._score_primary_rows(tasks, {}, workers=2)
                self.assertTrue(running.done())
                self.assertFalse(running.cancelled())
                self.assertEqual(running.result(), "checkpoint saved")
                executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

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
        for requested in (0, 32):
            self.assertEqual(resolve(
                requested, pending_count=100, cpu_count=32,
                available_memory_bytes=64 * gibibyte,
            ), 32)
        self.assertEqual(resolve(
            12, pending_count=100, cpu_count=32,
            available_memory_bytes=64 * gibibyte,
        ), 12)
        with patch.object(run_dev18_tuning, "available_cpu_count", return_value=16) as count, \
                patch.object(run_dev18_tuning, "_detect_available_memory_bytes", return_value=None):
            self.assertEqual(resolve(0, pending_count=100), 16)
            count.assert_called_once_with()
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

    def test_vus_checkpoint_reuses_approved_legacy_commit_without_rewriting_it(self):
        previous = {"project_commit": "a" * 40, "score_sha256": "score", "label_sha256": "labels",
                    "budget_id": "budget", "evaluator_sha256": "evaluator", "ell_max_id": "ell"}
        current = {**previous, "project_commit": "b" * 40}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_dev18_tuning, "_compatible_score_commits", return_value=(
                    "c" * 40, current["project_commit"], previous["project_commit"],
                )), patch("tests.checks.validate_resource_resume.resource_resume_compatible", return_value=True):
            original = run_dev18_tuning._score_checkpoint_path(directory, previous, normalize_commit=False)
            run_dev18_tuning._write_score_checkpoint(original, previous, 0.625)
            original_bytes = original.read_bytes()
            shared = run_dev18_tuning._score_checkpoint_path(directory, current)
            self.assertEqual(shared, run_dev18_tuning._score_checkpoint_path(directory, previous))
            self.assertEqual(run_dev18_tuning._load_score_checkpoint(shared, current), 0.625)
            self.assertEqual(original.read_bytes(), original_bytes)
            self.assertFalse(shared.exists())
            for field in ("score_sha256", "label_sha256", "budget_id", "evaluator_sha256", "ell_max_id"):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "checkpoint"):
                    run_dev18_tuning._load_score_checkpoint(original, {**current, field: "different"})
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(run_dev18_tuning, "_compatible_score_commits", return_value=(current["project_commit"],)), \
                patch("tests.checks.validate_resource_resume.resource_resume_compatible", return_value=False):
            path = Path(directory) / "unapproved.json"
            run_dev18_tuning._write_score_checkpoint(path, previous, 0.625)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                run_dev18_tuning._load_score_checkpoint(path, current)

    def test_vus_compatible_commit_search_is_cached_and_keeps_the_anchor(self):
        from tests.checks import validate_resource_resume as resume

        current, previous, unapproved = "a" * 40, "b" * 40, "c" * 40
        compatible = {resume.RESOURCE_RESUME_SOURCE, current, previous}
        run_dev18_tuning._compatible_score_commits.cache_clear()
        with patch.object(resume, "resource_resume_compatible", side_effect=lambda old, new, root: (
            old in compatible and new in compatible
        )), patch.object(run_dev18_tuning.subprocess, "check_output", return_value=f"{previous}\n{unapproved}\n") as git:
            for _ in range(2):
                self.assertEqual(run_dev18_tuning._compatible_score_commits(current, "repository"), (
                    resume.RESOURCE_RESUME_SOURCE, current, previous,
                ))
            git.assert_called_once()
        run_dev18_tuning._compatible_score_commits.cache_clear()

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

    def test_selection_only_cli_bypasses_scoring_lock_and_manifest(self):
        from tests.checks import finish_lightning_dev18

        required = (
            "model_fixed_policy.csv", "tier_fixed_policy.csv",
            "ratio_adaptive_selection.csv", "tier_ratio_candidate_audit.csv",
            "tier_policy_transitions.csv", "final_policy_membership.csv",
            "selection.png",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_directory = root / "results"
            result_directory.mkdir()
            for name in required:
                (result_directory / name).write_bytes(name.encode("ascii"))
            ledger_path = root / "dev18_trial_score_ledger.csv"
            ledger_path.write_text("sealed ledger", encoding="utf-8")
            selection_result = {
                "status": "complete", "budget_id": "b5367ad431093",
                "selection_rule_id": "tier_adaptive_family_lofo_v1",
                "project_commit": "c" * 40, "ledger_path": str(ledger_path),
                "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                "ledger_rows": 1602, "membership_rows": 294,
                "final_policy_membership_sha256": hashlib.sha256(
                    (result_directory / "final_policy_membership.csv").read_bytes()
                ).hexdigest(),
                "selected_path": [],
                "result_rows": {"final_policy_membership.csv": 294},
                "result_directory": str(result_directory),
            }
            lock_directory = root / "must_not_exist"
            with (
                patch.object(
                    sys, "argv",
                    [
                        str(Path(finish_lightning_dev18.__file__)),
                        "--selection-only", "--ledger", str(ledger_path),
                        "--result-directory", str(result_directory),
                    ],
                ),
                patch.object(
                    finish_lightning_dev18, "finish_selection_from_ledger",
                    return_value=selection_result,
                ) as selector,
                patch.object(
                    finish_lightning_dev18, "finish_tuning",
                    side_effect=AssertionError("selection-only가 기존 채점을 호출했다"),
                ) as finisher,
                patch.object(
                    finish_lightning_dev18, "_load_score_manifest",
                    side_effect=AssertionError("selection-only가 manifest를 읽었다"),
                ) as manifest_loader,
                patch.object(
                    finish_lightning_dev18, "DEFAULT_VUS_CHECKPOINT_DIRECTORY",
                    lock_directory,
                ),
                patch("builtins.print"),
            ):
                finish_lightning_dev18.main()

            selector.assert_called_once_with(
                ledger_path, result_directory=result_directory,
            )
            finisher.assert_not_called()
            manifest_loader.assert_not_called()
            self.assertFalse(lock_directory.exists())
            receipt_path = result_directory / "selection_complete.json"
            self.assertTrue(receipt_path.is_file())
            self.assertFalse((result_directory / "finish_complete.json").exists())
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["membership_rows"], 294)
            self.assertEqual(receipt["expected_result_files"], list(required))
            self.assertEqual(set(receipt["result_files_sha256"]), set(required))
            self.assertNotIn("score_manifest_sha256", receipt)

    def test_selection_only_cli_removes_stale_receipt_before_failed_run(self):
        from tests.checks import finish_lightning_dev18

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_directory = root / "results"
            result_directory.mkdir()
            receipt_path = result_directory / "selection_complete.json"
            receipt_path.write_text('{"status":"complete"}\n', encoding="utf-8")
            preserved_path = result_directory / "model_fixed_policy.csv"
            preserved_path.write_text("existing output\n", encoding="utf-8")
            ledger_path = root / "dev18_trial_score_ledger.csv"
            ledger_path.write_text("invalid ledger\n", encoding="utf-8")
            with (
                patch.object(
                    sys, "argv",
                    [
                        str(Path(finish_lightning_dev18.__file__)),
                        "--selection-only", "--ledger", str(ledger_path),
                        "--result-directory", str(result_directory),
                    ],
                ),
                patch.object(
                    finish_lightning_dev18, "finish_selection_from_ledger",
                    side_effect=ValueError("ledger rejected"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "ledger rejected"):
                    finish_lightning_dev18.main()

            self.assertFalse(receipt_path.exists())
            self.assertEqual(preserved_path.read_text(encoding="utf-8"), "existing output\n")

    def test_cpu_finish_cli_can_import_project_packages(self):
        script = Path(__file__).parents[1] / "checks" / "finish_lightning_dev18.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=script.parents[2],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
