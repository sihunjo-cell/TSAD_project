"""최종 membership 한 파일의 실행 계약을 검증한다."""

import csv
import hashlib
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.common.load_final_membership import (
    CONDITIONAL_FIELDS,
    build_final_execution_union,
    load_final_membership,
)
from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from tests.ghl_main.run_dev18_tuning import build_final_membership_rows
from tests.ghl_main.run_registered_models import build_specs, load_registered_inputs


MEMBERSHIP_FIELDS = (
    "analysis_kind", "split_role", "tier", "model", "evaluation_ratio",
    "physical_ratio", "config_id", "score_variant", "status", "status_reason",
)
FINAL_SPLIT_ROLES = (
    "ghl25_final", "train1_to_test1", "train1_train2_to_test2",
)


def make_membership_rows(registry):
    rows = []
    for split_role in FINAL_SPLIT_ROLES:
        for model_name, model in registry["models"].items():
            for ratio in SUPPORTED_RATIO_PERCENTS:
                rows.append({
                    "analysis_kind": "model_fixed", "split_role": split_role,
                    "tier": model["tier"], "model": model_name,
                    "evaluation_ratio": ratio,
                    "physical_ratio": 100 if model["target_use"] in {
                        "training_free", "strict_zero_shot",
                    } else ratio,
                    "config_id": model["candidates"][0]["config_id"],
                    "score_variant": "raw_max" if model_name == "TSPulse" else "",
                    "status": "runnable", "status_reason": "",
                })
        tier_winners = {"t1": "TierOneWinner", "t2": "GDN", "t3": "TSPulse"}
        for tier, model_name in tier_winners.items():
            model = registry["models"][model_name]
            for ratio in SUPPORTED_RATIO_PERCENTS:
                rows.append({
                    "analysis_kind": "tier_fixed", "split_role": split_role,
                    "tier": tier, "model": model_name,
                    "evaluation_ratio": ratio,
                    "physical_ratio": 100 if model["target_use"] in {
                        "training_free", "strict_zero_shot",
                    } else ratio,
                    "config_id": model["candidates"][0]["config_id"],
                    "score_variant": "time" if model_name == "TSPulse" else "",
                    "status": "runnable", "status_reason": "",
                })
                unavailable = tier == "t2" and ratio == 5
                rows.append({
                    "analysis_kind": "tier_adaptive", "split_role": split_role,
                    "tier": tier, "model": model_name,
                    "evaluation_ratio": ratio,
                    "physical_ratio": "" if unavailable else (
                        100 if model["target_use"] in {
                            "training_free", "strict_zero_shot",
                        } else ratio
                    ),
                    "config_id": model["candidates"][0]["config_id"],
                    "score_variant": "raw_max" if model_name == "TSPulse" else "",
                    "status": "unavailable" if unavailable else "runnable",
                    "status_reason": (
                        "이 비율에서 실행 가능한 Tier 후보가 없다" if unavailable else ""
                    ),
                })
    return rows


def write_final_membership(directory, registry, *, mutate=None):
    rows = make_membership_rows(registry)
    if mutate is not None:
        rows = mutate(deepcopy(rows))
    path = Path(directory) / "final_policy_membership.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MEMBERSHIP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_registry():
    return {
        "models": {
            "TierOneWinner": {
                "tier": "t1", "execution_status": "ready",
                "target_use": "training_free",
                "candidates": [{"config_id": "c" + "1" * 12}],
            },
            "GDN": {
                "tier": "t2", "execution_status": "ready",
                "target_use": "fit_validation",
                "candidates": [{"config_id": "c" + "2" * 12}],
            },
            "TSPulse": {
                "tier": "t3", "execution_status": "ready",
                "target_use": "strict_zero_shot",
                "candidates": [{"config_id": "c" + "3" * 12}],
            },
        },
    }


class TestFinalMembership(unittest.TestCase):
    def setUp(self):
        self.registry = make_registry()

    def test_loads_file_hash_and_deduplicates_physical_execution(self):
        with TemporaryDirectory() as directory:
            path = write_final_membership(directory, self.registry)
            expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            membership_sha256, membership = load_final_membership(path, self.registry)

        self.assertEqual(membership_sha256, expected_sha256)
        self.assertEqual(
            {row["analysis_kind"] for row in membership},
            {"model_fixed", "tier_fixed", "tier_adaptive"},
        )
        union = build_final_execution_union(membership, "ghl25_final")
        self.assertEqual(union[("TierOneWinner", "c" + "1" * 12, 100)], ("",))
        self.assertEqual(
            union[("TSPulse", "c" + "3" * 12, 100)],
            ("raw_max", "time"),
        )

    def test_adaptive_policy_does_not_add_physical_execution(self):
        with TemporaryDirectory() as directory:
            path = write_final_membership(directory, self.registry)
            _, membership = load_final_membership(path, self.registry)

        with_adaptive = build_final_execution_union(membership, "ghl25_final")
        without_adaptive = build_final_execution_union(
            tuple(row for row in membership if row["analysis_kind"] != "tier_adaptive"),
            "ghl25_final",
        )
        self.assertEqual(with_adaptive, without_adaptive)

    def test_builder_uses_fixed_tier_placeholder_for_unavailable_adaptive_ratio(self):
        ratios = (5, 10)
        model_fixed = [{
            "tier": model["tier"], "model": model_name,
            "q_support": list(ratios),
            "config_id": model["candidates"][0]["config_id"],
            "score_variant": "raw_max" if model_name == "TSPulse" else "",
            "selection_status": "selected",
        } for model_name, model in self.registry["models"].items()]
        winners = {"t1": "TierOneWinner", "t2": "GDN", "t3": "TSPulse"}
        tier_fixed = [{
            "tier": tier, "selected_model": model_name,
            "selection_q_common": list(ratios),
            "config_id": self.registry["models"][model_name]["candidates"][0]["config_id"],
            "score_variant": "time" if model_name == "TSPulse" else "",
            "selection_status": "selected",
        } for tier, model_name in winners.items()]
        tier_adaptive = [{
            "tier": tier, "ratio": ratio,
            "selected_model": "" if tier == "t2" and ratio == 5 else model_name,
            "config_id": "" if tier == "t2" and ratio == 5 else (
                self.registry["models"][model_name]["candidates"][0]["config_id"]
            ),
            "score_variant": "raw_max" if model_name == "TSPulse" else "",
            "selection_status": (
                "unavailable" if tier == "t2" and ratio == 5 else "selected"
            ),
            "selection_reason": (
                "이 비율에서 실행 가능한 Tier 후보가 없다"
                if tier == "t2" and ratio == 5 else "selected"
            ),
        } for tier, model_name in winners.items() for ratio in ratios]

        rows = build_final_membership_rows(
            {"model_fixed": model_fixed, "tier_fixed": tier_fixed,
             "tier_adaptive": tier_adaptive},
            self.registry, ratios=ratios, split_roles=("ghl25_final",),
        )
        unavailable = next(
            row for row in rows
            if row["analysis_kind"] == "tier_adaptive"
            and row["tier"] == "t2" and row["evaluation_ratio"] == 5
        )
        self.assertEqual(unavailable["model"], "GDN")
        self.assertEqual(unavailable["config_id"], "c" + "2" * 12)
        self.assertEqual(unavailable["physical_ratio"], "")
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertEqual(
            unavailable["status_reason"],
            "이 비율에서 실행 가능한 Tier 후보가 없다",
        )

    def test_rejects_invalid_rows(self):
        def wrong_physical_ratio(rows):
            next(row for row in rows if row["model"] == "GDN")["physical_ratio"] = 100
            return rows

        def wrong_variant(rows):
            next(row for row in rows if row["model"] == "TSPulse")["score_variant"] = "other"
            return rows

        cases = {
            "duplicate": lambda rows: rows + [rows[0]],
            "wrong physical ratio": wrong_physical_ratio,
            "wrong score variant": wrong_variant,
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as directory:
                path = write_final_membership(directory, self.registry, mutate=mutate)
                with self.assertRaises(ValueError):
                    load_final_membership(path, self.registry)

    def test_requires_all_analysis_kinds_for_each_split(self):
        def remove_tier_adaptive(rows):
            return [
                row for row in rows
                if row["split_role"] != "ghl25_final"
                or row["analysis_kind"] != "tier_adaptive"
            ]

        with TemporaryDirectory() as directory:
            path = write_final_membership(
                directory, self.registry, mutate=remove_tier_adaptive,
            )
            with self.assertRaisesRegex(ValueError, "tier_adaptive"):
                load_final_membership(path, self.registry)

    def test_requires_all_splits_models_tiers_and_ratios(self):
        cases = {
            "split": lambda rows: [
                row for row in rows if row["split_role"] != "ghl25_final"
            ],
            "model": lambda rows: [
                row for row in rows
                if not (
                    row["analysis_kind"] == "model_fixed"
                    and row["split_role"] == "ghl25_final"
                    and row["model"] == "GDN"
                )
            ],
            "tier ratio": lambda rows: [
                row for row in rows
                if not (
                    row["analysis_kind"] == "tier_fixed"
                    and row["split_role"] == "ghl25_final"
                    and row["tier"] == "t2"
                    and row["evaluation_ratio"] == 20
                )
            ],
            "adaptive tier ratio": lambda rows: [
                row for row in rows
                if not (
                    row["analysis_kind"] == "tier_adaptive"
                    and row["split_role"] == "ghl25_final"
                    and row["tier"] == "t2"
                    and row["evaluation_ratio"] == 20
                )
            ],
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as directory:
                path = write_final_membership(directory, self.registry, mutate=mutate)
                with self.assertRaisesRegex(ValueError, "완전하지 않다"):
                    load_final_membership(path, self.registry)


class TestConditionalMembership(unittest.TestCase):
    def setUp(self):
        self.registry = make_registry()
        self.registry["models"]["GDN"]["target_use"] = "fit_full_prefix"
        self.registry["models"]["GDN"]["candidates"].append({"config_id": "c" + "4" * 12})

    def rows(self):
        return [{
            "analysis_kind": "model_ratio", "split_role": "ghl25_final",
            "tier": "t2", "model": "GDN", "evaluation_ratio": ratio,
            "physical_ratio": ratio, "config_id": "c" + digit * 12,
            "score_variant": "", "status": "runnable", "status_reason": "",
            "series": series, "group_id": "group-" + series,
            "support_status": "within_dev_support",
        } for series, digit in (("01", "2"), ("02", "4"))
            for ratio in SUPPORTED_RATIO_PERCENTS]

    def load_rows(self, directory, rows):
        path = Path(directory) / "conditional_membership.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CONDITIONAL_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path, load_final_membership(path, self.registry)[1]

    def test_requires_target_series_and_only_returns_its_configs(self):
        with TemporaryDirectory() as directory:
            _, membership = self.load_rows(directory, self.rows())
        with self.assertRaisesRegex(ValueError, "series"):
            build_final_execution_union(membership, "ghl25_final")
        union = build_final_execution_union(membership, "ghl25_final", series=2)
        self.assertEqual({key[1] for key in union}, {"c" + "4" * 12})
        with self.assertRaisesRegex(ValueError, "series"):
            build_final_execution_union(membership, "ghl25_final", series="03")

    def test_paper_tspulse_accepts_ensemble_and_rejects_legacy_raw_max(self):
        self.registry["common_recipe"] = {"methodology_revision": "paper_tuning_v4"}
        self.registry["models"] = {"TSPulse": self.registry["models"]["TSPulse"]}
        self.registry["models"]["TSPulse"]["candidates"][0]["hyperparameters"] = {"aggregation_window": 64}
        self.registry["selection"] = {"primary_hpo_regime": "full_prefix_per_ratio",
            "primary_score_variants": {"TSPulse": ["time", "fft", "pred", "ensemble"]},
            "diagnostic_score_variants": {}}
        config = self.registry["models"]["TSPulse"]["candidates"][0]["config_id"]
        rows = [{**row, "tier": "t3", "model": "TSPulse", "physical_ratio": 100,
                 "score_variant": "ensemble", "config_id": config, "analysis_kind": kind}
                for row in self.rows() if row["series"] == "01"
                for kind in ("model_ratio", "tier_adaptive")]
        with TemporaryDirectory() as directory:
            _, membership = self.load_rows(directory, rows)
            union = build_final_execution_union(membership, "ghl25_final", series="01")
            self.assertEqual(list(union.values()), [("ensemble",)])
            rows[0]["score_variant"] = "raw_max"
            with self.assertRaisesRegex(ValueError, "score_variant"):
                self.load_rows(directory, rows)

    def test_unavailable_can_omit_config_and_physical_ratio(self):
        rows = self.rows()
        rows[0].update(
            config_id="", physical_ratio="", status="unavailable",
            status_reason="실행 가능한 지원 그룹이 없다", group_id="",
            support_status="unavailable",
        )
        with TemporaryDirectory() as directory:
            _, membership = self.load_rows(directory, rows)
        union = build_final_execution_union(membership, "ghl25_final", series="01")
        self.assertEqual({key[2] for key in union}, set(SUPPORTED_RATIO_PERCENTS) - {5})

    def test_rejects_incomplete_q_and_missing_runnable_group(self):
        missing_group = self.rows()
        missing_group[0]["group_id"] = ""
        for rows in (self.rows()[1:], missing_group, self.rows() + [self.rows()[0]]):
            with self.subTest(rows=len(rows)), TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.load_rows(directory, rows)

    def test_out_of_support_is_allowed_for_hai_external_validation(self):
        rows = [row for row in self.rows() if row["series"] == "01"]
        for row in rows:
            row.update(split_role="train1_to_test1", support_status="out_of_dev_support")
        with TemporaryDirectory() as directory:
            _, membership = self.load_rows(directory, rows)
        self.assertEqual(len(build_final_execution_union(
            membership, "train1_to_test1", series="01",
        )), len(SUPPORTED_RATIO_PERCENTS))

    def test_specs_require_and_bind_the_conditional_series(self):
        registry = deepcopy(self.registry)
        registry["seeds"] = {"final": [0]}
        registry["common_recipe"] = {}
        registry["common_recipe_id"] = "test"
        for model in registry["models"].values():
            model.update(deterministic=True, source_commit=None,
                         source_checkpoint_sha256=None, preprocess_recipe={})
            for candidate in model["candidates"]:
                candidate["hyperparameters"] = {}
        with TemporaryDirectory() as directory:
            path, _ = self.load_rows(directory, self.rows())
            with patch("tests.ghl_main.run_registered_models.load_model_registry_with_sha",
                       return_value=(registry, "a" * 64)), patch(
                "tests.ghl_main.run_registered_models.validate_input_manifest_role",
                return_value="b" * 64,
            ):
                with self.assertRaisesRegex(ValueError, "series"):
                    build_specs("final", final_policy_membership_path=path)
                specs = build_specs("final", final_policy_membership_path=path, series=2)
        self.assertEqual({spec["series"] for spec in specs}, {"02"})
        self.assertEqual({spec["config_id"] for spec in specs}, {"c" + "4" * 12})

    def test_input_loader_rejects_a_different_csv_before_reading_it(self):
        spec = {
            "dataset_role": "final", "split_role": "ghl25_final",
            "input_manifest_sha256": "a" * 64,
            "final_policy_membership_sha256": "b" * 64, "series": "02",
        }
        with patch("tests.ghl_main.run_registered_models.load_input_manifest_role",
                   return_value=({}, "a" * 64)), patch(
            "tests.ghl_main.run_registered_models._verify_manifest_file",
        ) as verify:
            with self.assertRaisesRegex(ValueError, "series"):
                load_registered_inputs(spec=spec, csv_path="032_GHL_id_1_Sensor_tr_100_1st_120.csv")
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
