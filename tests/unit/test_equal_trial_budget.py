"""Dev18 equal-trial exact panel과 순환 없는 budget seal을 검증한다."""

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

from src.common.equal_trial_budget import (
    build_equal_trial_budget,
    registry_space_sha256,
)
from src.common.execution_identity import file_sha256, sealed_crlf_text_sha256
from src.common.model_feasibility import (
    build_dev18_feasibility_rows,
    summarize_dev18_feasibility,
)
from src.common.model_registry import load_model_registry_with_sha
from tests.ghl_main.build_dev18_budget import (
    _load_current_feasibility,
    build_dev18_budget_artifact,
)
from tests.ghl_main.build_dev18_feasibility import build_dev18_feasibility_artifacts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class TestEqualTrialBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest_bytes = (REPOSITORY_ROOT / "configs/input_manifest.yaml").read_bytes()
        cls.entries = yaml.safe_load(manifest_bytes.decode("utf-8"))["datasets"][
            "DEV18"
        ]["files"]
        cls.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        cls.registry, registry_sha = load_model_registry_with_sha()
        cls.registry_sha256 = registry_sha
        cls.summary = cls._build_summary(cls.registry)
        cls.budget = build_equal_trial_budget(cls.registry, cls.summary)

    @classmethod
    def _build_summary(cls, registry):
        rows = build_dev18_feasibility_rows(
            registry,
            cls.entries,
            config_registry_sha256=cls.registry_sha256,
            input_manifest_sha256=cls.manifest_sha256,
            inventory_sha256="a" * 64,
        )
        return {
            **summarize_dev18_feasibility(rows, registry),
            "input_manifest_sha256": cls.manifest_sha256,
            "inventory_sha256": "a" * 64,
            "data_preprocessing_sha256": "b" * 64,
        }

    def test_seals_exact_model_panels_without_hiding_unavailable_models(self):
        panels = {panel["model"]: panel for panel in self.budget["model_panels"]}
        self.assertEqual(self.budget["tier_config_trial_count"], {
            "t1": 1, "t2": 2, "t3": 1,
        })
        self.assertEqual(panels["MWVAR"]["selected_config_ids"], ["c43024819503c"])
        self.assertEqual(panels["SQDIFF_LAST3"]["selected_config_ids"], ["c472288b30428"])
        self.assertEqual(panels["PCA_LEGACY"]["selected_config_ids"], ["c24a495574d36"])
        self.assertEqual(panels["PCA_LEGACY"]["selection_kind"],
                         "canonical_fixed_without_hpo")
        self.assertEqual(panels["PaAno"]["selected_config_ids"], [
            "c486c198d51af", "c78142611e61b",
        ])
        self.assertEqual(panels["GDN"]["selected_config_ids"], [
            "cf1a967db6cfe", "c1168c94d4dfc",
        ])
        self.assertEqual(panels["ALoRa"]["selection_kind"], "unavailable")
        self.assertEqual(panels["ALoRa"]["selected_config_ids"], [])
        self.assertEqual(panels["TimeRCD"]["selected_config_ids"], ["c1c5aeea6f7d3"])
        self.assertEqual(panels["TSPulse"]["selected_config_ids"], ["c12c5e6196ea5"])

    def test_preserves_supported_curves_and_collapses_target_free_physical_runs(self):
        panels = {panel["model"]: panel for panel in self.budget["model_panels"]}
        self.assertEqual(panels["PaAno"]["logical_ratios"], [40, 60, 80, 100])
        self.assertEqual(panels["GDN"]["logical_ratios"], [10, 20, 40, 60, 80, 100])
        self.assertEqual(panels["PCA_LEGACY"]["logical_ratios"], [100])
        self.assertEqual(len(self.budget["execution_panel"]), 65)
        self.assertEqual(self.budget["primary_logical_score_row_count"], 89)
        mwvar = [
            row for row in self.budget["execution_panel"] if row["model"] == "MWVAR"
        ]
        self.assertEqual(len(mwvar), 1)
        self.assertEqual(mwvar[0]["physical_ratio"], 100)
        self.assertEqual(mwvar[0]["logical_ratios"], [5, 10, 20, 40, 60, 80, 100])

    def test_tspulse_has_one_presealed_primary_variant(self):
        panel = next(
            panel for panel in self.budget["model_panels"]
            if panel["model"] == "TSPulse"
        )
        self.assertEqual(panel["primary_score_variants"], ["raw_max"])
        self.assertEqual(panel["diagnostic_score_variants"], ["time", "fft", "pred"])
        self.assertEqual(self.budget["fairness_rules"]["score_variant_rule"],
                         "one_presealed_primary_variant_per_model")

    def test_seals_seed_failure_retry_and_tie_rules(self):
        self.assertEqual(self.budget["seeds"], {
            "deterministic": [0], "stochastic": [0, 1, 2],
        })
        self.assertEqual(self.budget["failure_rules"]["maximum_transient_retries"], 2)
        self.assertEqual(self.budget["failure_rules"]["maximum_total_attempts"], 3)
        self.assertFalse(self.budget["failure_rules"]["allow_partial_seed_mean"])
        self.assertFalse(self.budget["failure_rules"]["zero_fill_failed_trial"])
        self.assertFalse(self.budget["failure_rules"]["replacement_config"])
        self.assertEqual(self.budget["tie_rule"]["tolerance"], 1e-6)
        self.assertEqual(self.budget["tie_rule"]["order"], [
            "model", "config_id", "score_variant",
        ])

    def test_budget_identity_ignores_only_mutable_seal_and_readiness_fields(self):
        changed = deepcopy(self.registry)
        changed["selection"].update({
            "primary_hpo_regime": "equal_trial",
            "budget_id": "b000000000000",
            "selection_status": "ready",
        })
        changed["models"]["TimeRCD"]["execution_status"] = "pending_checkpoint_smoke"
        changed["models"]["TimeRCD"]["status_reason"] = "unit test"
        changed_summary = self._build_summary(changed)
        self.assertEqual(registry_space_sha256(changed), registry_space_sha256(self.registry))
        self.assertEqual(
            changed_summary["feasibility_decision_sha256"],
            self.summary["feasibility_decision_sha256"],
        )
        self.assertEqual(
            build_equal_trial_budget(changed, changed_summary)["budget_id"],
            self.budget["budget_id"],
        )

        changed["models"]["MWVAR"]["source_url"] += "?changed"
        self.assertNotEqual(registry_space_sha256(changed), registry_space_sha256(self.registry))
        self.assertNotEqual(
            build_equal_trial_budget(changed, self.summary)["budget_id"],
            self.budget["budget_id"],
        )

    def test_artifact_is_sealed_to_current_registry_and_feasibility_files(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            directory = Path(directory)
            feasibility = build_dev18_feasibility_artifacts(
                repository_root=REPOSITORY_ROOT,
                output_directory=directory / "feasibility",
            )
            result = build_dev18_budget_artifact(
                repository_root=REPOSITORY_ROOT,
                feasibility_directory=directory / "feasibility",
                output_path=directory / "budget.json",
            )
            budget_bytes = Path(result["budget_path"]).read_bytes()
            budget = yaml.safe_load(budget_bytes.decode("utf-8"))

        self.assertEqual(budget["seal_status"], "sealed")
        self.assertEqual(budget["budget_id"], self.registry["selection"]["budget_id"])
        self.assertEqual(budget["execution_readiness_status"], "ready")
        self.assertEqual(budget["pending_execution_models"], [])
        self.assertEqual(
            budget["attestation"]["feasibility_ledger_sha256"],
            feasibility["ledger_sha256"],
        )

    def test_artifact_rejects_stale_role_a_input_identities(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            directory = Path(directory)
            feasibility_directory = directory / "feasibility"
            build_dev18_feasibility_artifacts(
                repository_root=REPOSITORY_ROOT,
                output_directory=feasibility_directory,
            )
            targets = (
                (REPOSITORY_ROOT / "configs" / "input_manifest.yaml",
                 "input_manifest_sha256"),
                (REPOSITORY_ROOT / "experiments" / "checks" / "datasets"
                 / "dev18" / "logs" / "inventory.csv", "inventory_sha256"),
                (REPOSITORY_ROOT / "experiments" / "checks" / "datasets"
                 / "dev18" / "snapshots" / "audit.json",
                 "audit_snapshot_sha256"),
            )
            for target, field in targets:
                with self.subTest(field=field):
                    digest = (
                        sealed_crlf_text_sha256
                        if field in {"inventory_sha256", "audit_snapshot_sha256"}
                        else file_sha256
                    )

                    def changed_digest(path, *, _target=target):
                        return "0" * 64 if Path(path) == _target else digest(path)

                    with patch(
                        "tests.ghl_main.build_dev18_budget."
                        + ("sealed_crlf_text_sha256" if digest is sealed_crlf_text_sha256
                           else "file_sha256"),
                        side_effect=changed_digest,
                    ), self.assertRaisesRegex(ValueError, field):
                        build_dev18_budget_artifact(
                            repository_root=REPOSITORY_ROOT,
                            feasibility_directory=feasibility_directory,
                            output_path=directory / "budget.json",
                        )

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
                "row_count": 3276,
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
                repository_root, feasibility_directory, "a" * 64,
            )

        self.assertEqual(loaded, summary)


if __name__ == "__main__":
    unittest.main()
