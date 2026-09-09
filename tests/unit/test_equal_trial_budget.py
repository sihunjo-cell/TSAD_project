"""과거 equal-trial 계약은 현행 튜닝 풀과 분리한 작은 입력으로 검증한다."""

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

from src.common.equal_trial_budget import build_equal_trial_budget, registry_space_sha256
from src.common.execution_identity import file_sha256
from tests.ghl_main.build_dev18_budget import _load_current_feasibility


class TestEqualTrialBudget(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "selection": {"primary_hpo_regime": "equal_trial", "minimum_primary_ratio_count": 3,
                          "tier_q_floor": {"t1": 5, "t2": 5, "t3": 5},
                          "selection_rule_id": "legacy_fixture",
                          "primary_score_variants": {"TSPulse": ["raw_max"]},
                          "diagnostic_score_variants": {"TSPulse": ["time", "fft", "pred"]}},
            "seeds": {"development": [0, 1, 2]}, "models": {},
        }
        self.summary = {field: "a" * 64 for field in (
            "input_manifest_sha256", "inventory_sha256", "data_preprocessing_sha256",
            "feasibility_decision_sha256",
        )}
        self.summary["models"] = {}
        for model, tier, target_use, ratios, count, status in (
            ("MWVAR", "t1", "training_free", [5, 10, 20, 40, 60, 80, 100], 1, "eligible_for_model_fixed_hpo"),
            ("PCA_LEGACY", "t1", "fit_validation", [100], 4, "insufficient_ratio_support"),
            ("PaAno", "t2", "fit_validation", [40, 60, 80, 100], 2, "eligible_for_model_fixed_hpo"),
            ("GDN", "t2", "fit_validation", [], 0, "unavailable"),
            ("TimeRCD", "t3", "strict_zero_shot", [5, 10, 20, 40, 60, 80, 100], 1, "eligible_for_model_fixed_hpo"),
            ("TSPulse", "t3", "strict_zero_shot", [5, 10, 20, 40, 60, 80, 100], 3, "eligible_for_model_fixed_hpo"),
        ):
            candidates = [f"c{index:012d}" for index in range(1, count + 1)]
            self.registry["models"][model] = {
                "tier": tier, "target_use": target_use, "deterministic": tier != "t2",
                "execution_status": "ready", "source_commit": "b" * 40,
                "candidates": [{"config_id": config} for config in (candidates or ["c000000000001"])],
            }
            self.summary["models"][model] = {
                "tier": tier, "dev18_selection_support_status": status,
                "dev18_supported_ratios_at_or_above_q_floor": ratios,
                "dev18_common_config_ids_across_supported_ratios": candidates,
                "dev18_tier_representative_eligible": status == "eligible_for_model_fixed_hpo",
            }
        self.budget = build_equal_trial_budget(self.registry, self.summary)

    def test_seals_exact_model_panels_without_hiding_unavailable_models(self):
        panels = {panel["model"]: panel for panel in self.budget["model_panels"]}
        self.assertEqual(self.budget["tier_config_trial_count"], {"t1": 1, "t2": 2, "t3": 1})
        self.assertEqual(set(panels), set(self.registry["models"]))
        self.assertEqual(panels["PaAno"]["selected_config_ids"], ["c000000000001", "c000000000002"])
        self.assertEqual(panels["PCA_LEGACY"]["selected_config_ids"], ["c000000000001"])
        self.assertEqual(panels["PCA_LEGACY"]["selection_kind"], "canonical_fixed_without_hpo")
        self.assertEqual(panels["GDN"]["selection_kind"], "unavailable")
        self.assertEqual(panels["GDN"]["selected_config_ids"], [])

    def test_preserves_supported_curves_and_collapses_target_free_physical_runs(self):
        self.assertEqual(len(self.budget["execution_panel"]), 28)
        self.assertEqual(self.budget["primary_logical_score_row_count"], 46)
        mwvar = [row for row in self.budget["execution_panel"] if row["model"] == "MWVAR"]
        self.assertEqual(len(mwvar), 1)
        self.assertEqual(mwvar[0]["physical_ratio"], 100)
        self.assertEqual(mwvar[0]["logical_ratios"], [5, 10, 20, 40, 60, 80, 100])
        paano = [row for row in self.budget["execution_panel"] if row["model"] == "PaAno"]
        self.assertEqual(len(paano), 24)
        self.assertTrue(all(row["logical_ratios"] == [row["physical_ratio"]] for row in paano))

    def test_tspulse_has_one_presealed_primary_variant(self):
        panel = next(panel for panel in self.budget["model_panels"] if panel["model"] == "TSPulse")
        self.assertEqual(panel["primary_score_variants"], ["raw_max"])
        self.assertEqual(panel["diagnostic_score_variants"], ["time", "fft", "pred"])
        self.assertEqual(self.budget["fairness_rules"]["score_variant_rule"], "one_presealed_primary_variant_per_model")

    def test_seals_seed_failure_retry_and_tie_rules(self):
        self.assertEqual(self.budget["seeds"], {"deterministic": [0], "stochastic": [0, 1, 2]})
        self.assertEqual(self.budget["failure_rules"]["maximum_transient_retries"], 2)
        self.assertEqual(self.budget["failure_rules"]["maximum_total_attempts"], 3)
        self.assertFalse(self.budget["failure_rules"]["allow_partial_seed_mean"])
        self.assertFalse(self.budget["failure_rules"]["zero_fill_failed_trial"])
        self.assertFalse(self.budget["failure_rules"]["replacement_config"])
        self.assertEqual(self.budget["tie_rule"], {"tolerance": 1e-6, "order": ["model", "config_id", "score_variant"]})

    def test_budget_identity_ignores_only_mutable_seal_and_readiness_fields(self):
        changed = deepcopy(self.registry)
        changed["selection"].update(budget_id="b000000000000", selection_status="ready")
        changed["models"]["TimeRCD"].update(execution_status="pending_checkpoint_smoke", status_reason="unit test")
        self.assertEqual(registry_space_sha256(changed), registry_space_sha256(self.registry))
        self.assertEqual(build_equal_trial_budget(changed, self.summary)["budget_id"], self.budget["budget_id"])
        changed["models"]["MWVAR"]["source_commit"] = "c" * 40
        self.assertNotEqual(build_equal_trial_budget(changed, self.summary)["budget_id"], self.budget["budget_id"])

    def test_accepts_linux_checkout_of_windows_sealed_audit_text(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            feasibility_directory = repository_root / "feasibility"
            audit_root = (
                repository_root / "experiments" / "checks" / "datasets" / "dev18"
            )
            paths = {
                "input_manifest_sha256": repository_root / "configs/input_manifest.yaml",
                "inventory_sha256": audit_root / "logs/inventory.csv",
                "audit_snapshot_sha256": audit_root / "snapshots/audit.json",
                "ledger_sha256": feasibility_directory / "dev18_feasibility_ledger.csv",
                "builder_sha256": repository_root / "tests/ghl_main/build_dev18_feasibility.py",
                "feasibility_code_sha256": repository_root / "src/common/model_feasibility.py",
                "split_code_sha256": repository_root / "src/data_split/split_ratio_prefix.py",
                "data_preprocessing_sha256": repository_root / "configs/data_preprocessing.yaml",
            }
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"field,value\n1,2\n")

            summary = {
                "status": "complete_with_declared_static_unavailability",
                "row_count": 126, "config_count": 1,
                "labels_or_scores_read": False,
                "config_registry_sha256": "a" * 64,
                **{field: file_sha256(path) for field, path in paths.items()},
            }
            for field in ("inventory_sha256", "audit_snapshot_sha256"):
                summary[field] = hashlib.sha256(
                    paths[field].read_bytes().replace(b"\n", b"\r\n")
                ).hexdigest()
            summary_path = feasibility_directory / "dev18_feasibility_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            _, _, loaded = _load_current_feasibility(
                repository_root, feasibility_directory, "a" * 64, config_count=1,
            )
            for field in ("input_manifest_sha256", "inventory_sha256", "audit_snapshot_sha256"):
                original = paths[field].read_bytes()
                paths[field].write_bytes(b"changed\n")
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    _load_current_feasibility(repository_root, feasibility_directory, "a" * 64, config_count=1)
                paths[field].write_bytes(original)
            with self.assertRaisesRegex(ValueError, "핵심 계약"):
                _load_current_feasibility(repository_root, feasibility_directory, "a" * 64, config_count=2)

        self.assertEqual(loaded, summary)


if __name__ == "__main__":
    unittest.main()
