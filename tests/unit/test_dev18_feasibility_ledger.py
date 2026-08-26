"""Dev18 정적 feasibility 원표와 support 계약을 검증한다."""

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.common.model_feasibility import (
    build_dev18_feasibility_rows,
    summarize_dev18_feasibility,
)
from src.common.model_registry import load_model_registry_with_sha
from tests.ghl_main.build_dev18_feasibility import (
    build_dev18_feasibility_artifacts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class TestDev18FeasibilityLedger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest_path = REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
        serialized = manifest_path.read_bytes()
        cls.entries = yaml.safe_load(serialized.decode("utf-8"))["datasets"][
            "DEV18"
        ]["files"]
        cls.manifest_sha256 = hashlib.sha256(serialized).hexdigest()
        cls.registry, cls.registry_sha256 = load_model_registry_with_sha()
        cls.rows = build_dev18_feasibility_rows(
            cls.registry,
            cls.entries,
            config_registry_sha256=cls.registry_sha256,
            input_manifest_sha256=cls.manifest_sha256,
            inventory_sha256="a" * 64,
        )

    def test_builds_all_26_config_ratio_series_combinations_in_sealed_order(self):
        self.assertEqual(len(self.rows), 26 * 7 * 18)
        first = self.rows[0]
        last = self.rows[-1]
        self.assertEqual(
            (first["model"], first["config_order"], first["logical_ratio"], first["series"]),
            ("MWVAR", 1, 5, "01"),
        )
        self.assertEqual(
            (last["model"], last["config_order"], last["logical_ratio"], last["series"]),
            ("TSPulse", 3, 100, "18"),
        )
        self.assertEqual(len({
            (row["model"], row["config_id"], row["logical_ratio"], row["series"])
            for row in self.rows
        }), len(self.rows))

    def test_target_free_rows_map_every_logical_ratio_to_one_physical_score(self):
        rows = [
            row for row in self.rows
            if row["model"] == "MWVAR" and row["series"] == "01"
        ]
        self.assertEqual([row["logical_ratio"] for row in rows], [5, 10, 20, 40, 60, 80, 100])
        self.assertEqual({row["physical_ratio"] for row in rows}, {100})
        self.assertEqual({row["uses_training_prefix"] for row in rows}, {False})
        self.assertEqual(
            [(row["available_count"], row["fit_count"], row["validation_count"])
             for row in rows[:2]],
            [(26, 20, 6), (53, 42, 11)],
        )
        self.assertEqual({row["status"] for row in rows}, {"feasible"})

    def test_support_counts_expose_structural_limits_without_changing_q_floor(self):
        summary = summarize_dev18_feasibility(self.rows, self.registry)
        expected = {
            "MWVAR": {5: 1, 10: 1, 20: 1, 40: 1, 60: 1, 80: 1, 100: 1},
            "SQDIFF_LAST3": {5: 1, 10: 1, 20: 1, 40: 1, 60: 1, 80: 1, 100: 1},
            "PCA_LEGACY": {5: 0, 10: 0, 20: 0, 40: 0, 60: 0, 80: 0, 100: 4},
            "PaAno": {5: 0, 10: 0, 20: 0, 40: 3, 60: 3, 80: 6, 100: 9},
            "ALoRa": {5: 0, 10: 0, 20: 0, 40: 0, 60: 0, 80: 0, 100: 0},
            "GDN": {5: 0, 10: 2, 20: 2, 40: 2, 60: 2, 80: 2, 100: 2},
            "TimeRCD": {5: 1, 10: 1, 20: 1, 40: 1, 60: 1, 80: 1, 100: 1},
            "TSPulse": {5: 3, 10: 3, 20: 3, 40: 3, 60: 3, 80: 3, 100: 3},
        }
        self.assertEqual(
            {
                model: {
                    int(ratio): count
                    for ratio, count in details["fully_feasible_config_count_by_ratio"].items()
                }
                for model, details in summary["models"].items()
            },
            expected,
        )
        self.assertEqual(summary["row_count"], 3276)
        self.assertEqual(summary["structurally_infeasible_row_count"], 684)
        self.assertEqual(summary["fully_feasible_logical_key_count"], 79)
        self.assertEqual(summary["fully_feasible_ledger_row_count"], 1422)
        self.assertEqual(summary["tiers"]["t1"]["dev18_tier_representative_models"], [
            "MWVAR", "SQDIFF_LAST3",
        ])
        self.assertEqual(summary["tiers"]["t2"]["dev18_tier_representative_models"], [
            "GDN",
        ])
        self.assertEqual(summary["models"]["PCA_LEGACY"]["dev18_selection_support_status"],
                         "insufficient_ratio_support")
        self.assertEqual(summary["models"]["ALoRa"]["dev18_selection_support_status"],
                         "unavailable")

    def test_summary_recomputes_every_decision_and_rejects_tampering(self):
        mutations = (
            (0, "physical_ratio", 5),
            (0, "available_count", 999),
            (0, "status", "forged"),
            (0, "derived_json", "{}"),
            (0, "config_registry_sha256", "0" * 64),
        )
        for index, field, value in mutations:
            rows = deepcopy(self.rows)
            rows[index][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "Dev18 feasibility",
            ):
                summarize_dev18_feasibility(rows, self.registry)

    def test_every_row_records_generation_and_continuous_score_counts(self):
        for row in self.rows:
            with self.subTest(model=row["model"], ratio=row["logical_ratio"]):
                derived = json.loads(row["derived_json"])
                self.assertTrue(any(key.endswith("_count") for key in derived))
                self.assertIn("native_continuous_score_count", derived)
        tspulse = [
            row for row in self.rows
            if row["model"] == "TSPulse"
            and row["logical_ratio"] == 5 and row["series"] == "01"
        ]
        self.assertEqual([
            json.loads(row["derived_json"])["native_continuous_score_count"]
            for row in tspulse
        ], [1356, 1340, 1324])

    def test_rejects_dev18_order_or_shape_drift(self):
        entries = deepcopy(self.entries)
        entries[0]["series"] = "02"
        with self.assertRaisesRegex(ValueError, "Dev18.*순서"):
            build_dev18_feasibility_rows(
                self.registry,
                entries,
                config_registry_sha256=self.registry_sha256,
                input_manifest_sha256=self.manifest_sha256,
                inventory_sha256="a" * 64,
            )

    def test_builder_seals_ledger_and_summary_to_the_approved_audit(self):
        with TemporaryDirectory() as directory:
            result = build_dev18_feasibility_artifacts(
                repository_root=REPOSITORY_ROOT,
                output_directory=Path(directory),
            )
            ledger_path = Path(result["ledger_path"])
            summary_path = Path(result["summary_path"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ledger_sha256 = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

        self.assertEqual(summary["row_count"], 3276)
        self.assertEqual(summary["status"], "complete_with_declared_static_unavailability")
        self.assertEqual(summary["ledger_sha256"], ledger_sha256)
        self.assertEqual(summary["input_manifest_sha256"], self.manifest_sha256)
        self.assertEqual(summary["config_registry_sha256"], self.registry_sha256)


if __name__ == "__main__":
    unittest.main()
