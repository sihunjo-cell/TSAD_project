"""완료 원표와 작은 실행 증거의 지원 지점 연결을 확인한다."""

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.common.equal_trial_budget import registry_space_sha256
from src.common.execution_evidence import (
    FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, FULL_PREFIX_STORAGE_SCHEMA_VERSION,
)
from src.common.tuning_support import load_tuning_support, resolve_policy_score_variant
from tests.ghl_main import run_dev18_tuning as tuning
from tests.ghl_main.build_tuning_support import build_tuning_support


def write_json(root, name, payload):
    serialized = json.dumps(payload).encode()
    (root / name).write_bytes(serialized)
    return {"file": name, "sha256": hashlib.sha256(serialized).hexdigest()}


def make_evidence(root, *, target_free=False):
    model, tier = ("MWVAR", "t1") if target_free else ("GDN", "t2")
    target_use = "training_free" if target_free else "fit_full_prefix"
    parameters = {"window": 5, **({"rho": .5} if not target_free else {})}
    details = {"target_use": target_use, "tier": tier, "source_commit": "a" * 40,
               "execution_status": "ready",
               "source_checkpoint_sha256": "none", "preprocess_recipe": {"calibration": "none"},
               "candidates": [{"config_id": "config", "hyperparameters": parameters}]}
    registry = {"models": {model: details}, "common_recipe": {"training_split": "full_prefix_v2"},
                "common_recipe_id": "recipe", "selection": {}}
    entry = {"series": "01", "family": "family", "name": "sample.csv", "source_directory": "tuning",
             "sha256": "a" * 64, "size_bytes": 400, "training_boundary": 100, "row_count": 140,
             "feature_count": 3}
    (root / "configs").mkdir()
    input_reference = write_json(root, "configs/input_manifest.yaml", {
        "roles": {"dev18_selection": {"dataset": "DEV18"}}, "datasets": {"DEV18": {"files": [entry]}},
    })
    ratios, seeds = ([5, 100], [0]) if target_free else ([100], [0, 1, 2])
    groups = []
    for ratio in ratios:
        groups.append({
            "model": model, "tier": tier, "ratio": ratio, "group_id": f"g{ratio}",
            "series_ids": ["01"], "candidate_ids": ["config"],
            "support": [{"series": "01", "family": "family", "name": "sample.csv",
                         "observed_row": ratio,
                         "training_boundary": 100, "feature_count": 3}],
        })
    budget = {
        "schema_version": 3, "experiment_mode": "full_prefix_v2", "budget_id": "budget",
        "selection_rule_id": "selection", "registry_space_sha256": registry_space_sha256(registry),
        "input_manifest_sha256": input_reference["sha256"], "series_ids": ["01"],
        "model_panels": [{"model": model, "tier": tier, "groups": groups}],
        "execution_panel": [{"model": model, "config_id": "config", "seed": seed,
                             "physical_ratio": 100, "logical_ratios": ratios,
                             "series_ids": ["01"], "primary_score_variants": [""]}
                            for seed in seeds],
    }
    selection = {"model_ratio": [{**deepcopy(group), "selection_status": "selected", "budget_id": "budget",
                                  "config_id": "config", "score_variant": "", "family_count": 1,
                                  "validation_status": "insufficient_families",
                                  "family_macro_vus_pr": .7, "family_lofo_vus_pr": None,
                                  "hyperparameters": parameters, "source_commit": "a" * 40,
                                  "source_checkpoint_sha256": "none"}
                                 for group in groups]}
    ledger, manifest = [], []
    for seed in seeds:
        identity = {"model": model, "tier": tier, "config_id": "config", "ratio": 100,
                    "seed": seed, "dataset_role": "development", "split_role": "dev18_selection",
                    "input_manifest_sha256": input_reference["sha256"], "target_use": target_use,
                    "common_recipe_id": "recipe", "config_registry_sha256": "c" * 64}
        verified = {field: entry[field] for field in ("name", "source_directory", "sha256", "size_bytes")}
        snapshot = write_json(root, f"snapshot_{seed}.json", {
            "storage_schema_version": FULL_PREFIX_STORAGE_SCHEMA_VERSION,
            "spec": {**identity, "series": "01", "series_input_sha256": entry["sha256"],
                     "hyperparameters": parameters, "source_commit": "a" * 40,
                     "source_checkpoint_sha256": "none", "checkpoint_config_sha256": None,
                     "checkpoint_revision": None, "preprocess_recipe": details["preprocess_recipe"],
                     "common_recipe": registry["common_recipe"], "common_recipe_id": "recipe"},
            "input_identity": {**verified, "dataset": "DEV18", "verified_files": [verified],
                               "input_manifest_sha256": input_reference["sha256"]},
            "source_ranges": {"source": "sample.csv", "source_directory": "tuning",
                              "normal_training": [0, 100], "test_sessions": [[100, 140]]},
            "environment": {"cuda_device": {"name": "test GPU"}},
        })
        sessions = [] if target_free else [{"observed_row": 100, "training_boundary": 100,
                                            "observed_duration_seconds": None, "duration_basis": "unavailable"}]
        metadata = write_json(root, f"metadata_{seed}.json", {
            "storage_schema_version": FULL_PREFIX_STORAGE_SCHEMA_VERSION,
            **identity, "dataset": "DEV18", "series": 1, "score_variant": None, "channel_count": 0,
            "run_snapshot": snapshot, "execution_evidence": {
                "measurement_protocol_id": FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
                "storage_schema_version": FULL_PREFIX_STORAGE_SCHEMA_VERSION,
                "resource_usage": {"status": "unavailable", "reason": "mock"},
                "execution_phase": "development_hpo", "status": "complete", "retry_count": 0,
                "training_sessions": sessions,
                "test_sessions": [{"observation_count": 40, "observed_duration_seconds": None,
                                   "duration_basis": "unavailable"}],
                "timing": {field: 0.0 for field in (
                    "split_preprocess_seconds", "model_setup_seconds", "training_seconds",
                    "test_inference_seconds", "calibration_inference_seconds")},
                "runtime_seconds": 0.0, "peak_memory_mb": 12.0, "model_artifact_bytes": 34,
            },
        })
        row = {"series": "01", "family": "family", "model": model, "tier": tier,
               "config_id": "config", "physical_ratio": "100", "seed": str(seed),
               "score_variant": "", "primary_score": "true", "status": "complete",
               "budget_id": "budget", "score_file": f"absent_score_{seed}.npy", "score_sha256": "a" * 64,
               "metadata_file": metadata["file"], "metadata_sha256": metadata["sha256"]}
        manifest.append(row)
        ledger.extend({**row, "ratio": ratio, "vus_pr": .7, "normalization": "trainnorm",
                       "evaluator_sha256": "e" * 64, "ell_max_id": "ell"} for ratio in ratios)
    return selection, registry, budget, ledger, manifest


def add_tier_selection(selection):
    selection["tier_adaptive"] = []
    for policy in selection["model_ratio"]:
        identity = {field: policy[field] for field in ("tier", "ratio", "series_ids")}
        identity["candidate_ids_by_model"] = {policy["model"]: policy["candidate_ids"]}
        tier_policy = {**deepcopy(policy), **identity, "analysis_kind": "tier_adaptive",
                       "selected_model": policy["model"],
                       "group_id": "g" + hashlib.sha256(tuning._json(identity).encode()).hexdigest()[:12]}
        for field in ("source_commit", "source_checkpoint_sha256"):
            tier_policy.pop(field)
        selection["tier_adaptive"].append(tier_policy)


class TestTuningSupport(unittest.TestCase):
    def test_family_head_resolution_rejects_malformed_policies(self):
        policy = {"model": "TSPulse", "score_variant": "family_selected",
                  "score_variant_by_family": {"family": "ensemble", "another": "fft"},
                  "score_variant_fallback": "time"}
        self.assertEqual(resolve_policy_score_variant(policy, "family"), "ensemble")
        self.assertEqual(resolve_policy_score_variant(policy, "another"), "fft")
        self.assertEqual(resolve_policy_score_variant(policy, "unseen"), "time")
        self.assertEqual(resolve_policy_score_variant({"model": "GDN"}, "family"), "")
        for head in ("time", "fft", "pred", "ensemble", "raw_max"):
            self.assertEqual(resolve_policy_score_variant(
                {"model": "TSPulse", "score_variant": head}, "family",
            ), head)
        for changes in (
            {"model": "GDN"}, {"score_variant_by_family": []},
            {"score_variant_by_family": {}}, {"score_variant_by_family": {"": "time"}},
            {"score_variant_by_family": {1: "time"}},
            {"score_variant_by_family": {"family": "unknown"}},
            {"score_variant_by_family": {"family": "raw_max"}},
            {"score_variant_by_family": {"family": ["time"]}},
            {"score_variant_fallback": ""}, {"score_variant_fallback": "fft"},
            {"score_variant_fallback": None}, {"score_variant": "time"},
            {"score_variant": "unknown", "score_variant_by_family": {}, "score_variant_fallback": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                resolve_policy_score_variant({**policy, **changes}, "family")

    def test_tier_selection_preserves_its_different_config_and_round_trips(self):
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            root = Path(directory)
            selection, registry, budget, ledger, manifest = make_evidence(root)
            parameters = {"window": 6, "rho": .5}
            registry["models"]["GDN"]["candidates"].append({
                "config_id": "tier-config", "hyperparameters": parameters,
            })
            budget["registry_space_sha256"] = registry_space_sha256(registry)
            budget["model_panels"][0]["groups"][0]["candidate_ids"].append("tier-config")
            selection["model_ratio"][0]["candidate_ids"].append("tier-config")
            budget["execution_panel"].extend(
                {**execution, "config_id": "tier-config"} for execution in list(budget["execution_panel"])
            )
            for row in list(manifest):
                metadata = json.loads((root / row["metadata_file"]).read_text())
                snapshot = json.loads((root / metadata["run_snapshot"]["file"]).read_text())
                snapshot["spec"].update(config_id="tier-config", hyperparameters=parameters)
                metadata.update(config_id="tier-config", run_snapshot=write_json(
                    root, f"tier_snapshot_{row['seed']}.json", snapshot,
                ))
                reference = write_json(root, f"tier_metadata_{row['seed']}.json", metadata)
                changed = {**row, "config_id": "tier-config", "metadata_file": reference["file"],
                           "metadata_sha256": reference["sha256"],
                           "score_file": f"absent_tier_score_{row['seed']}.npy"}
                manifest.append(changed)
                ledger.append({**next(trial for trial in ledger if trial["seed"] == row["seed"]), **changed})
            add_tier_selection(selection)
            selection["tier_adaptive"][0].update(config_id="tier-config", hyperparameters=parameters)
            report = build_tuning_support(selection, registry, budget, ledger, manifest)
            reference = write_json(root, "support.json", report)
            loaded = load_tuning_support(root / reference["file"], reference["sha256"], selection, registry)
            self.assertEqual({(point["analysis_kind"], point["config_id"]) for point in loaded["points"]},
                             {("model_ratio", "config"), ("tier_adaptive", "tier-config")})
            self.assertEqual(len(loaded["executions"]), 6)
            from src.common.select_conditional_policy import recommend_conditional_candidates

            result = recommend_conditional_candidates(
                selection, registry, nrows=100, dfeatures=3, planned_rows=100, evaluation_rows=40,
                support_path=root / reference["file"], support_sha256=reference["sha256"],
                environment_id=loaded["points"][0]["environment_id"],
            )
            self.assertEqual({(candidate["analysis_kind"], candidate["config_id"])
                              for candidate in result["candidates"]},
                             {("model_ratio", "config"), ("tier_adaptive", "tier-config")})
            report["points"][1]["analysis_kind"] = "model_ratio"
            reference = write_json(root, "support.json", report)
            with self.assertRaises(ValueError):
                load_tuning_support(root / reference["file"], reference["sha256"], selection, registry)

    def test_model_and_tier_policies_share_target_free_execution(self):
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            inputs = make_evidence(Path(directory), target_free=True)
            add_tier_selection(inputs[0])
            report = build_tuning_support(*inputs)
            self.assertEqual(len(report["points"]), 4)
            self.assertEqual({point["analysis_kind"] for point in report["points"]},
                             {"model_ratio", "tier_adaptive"})
            self.assertEqual(len(report["executions"]), 1)
            self.assertEqual(len({tuple(point["execution_ids"]) for point in report["points"]}), 1)
            inputs[0]["tier_adaptive"][0]["candidate_ids_by_model"] = {"MWVAR": []}
            with self.assertRaises(ValueError):
                build_tuning_support(*inputs)

    def test_observed_point_keeps_all_seeds_without_opening_score_arrays(self):
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            inputs = make_evidence(Path(directory))
            report = build_tuning_support(*inputs)
        point = report["points"][0]
        self.assertEqual((point["observed_row"], point["feature_count"], point["training_boundary"]), (100, 3, 100))
        self.assertEqual(point["evaluation_rows"], 40)
        self.assertEqual([row["seed"] for row in point["trial_vus_pr"]], [0, 1, 2])
        self.assertEqual(len(point["execution_ids"]), 3)
        self.assertEqual(len(report["executions"]), 3)
        self.assertEqual(report["service_status"], "unvalidated")
        self.assertEqual(report["resource_status"], "not_configured")
        self.assertTrue(all(execution["measurement_scope"] == "completed_single_execution"
                            for execution in report["executions"].values()))
        self.assertIsNone(point["selection_procedure_lofo_vus_pr"])

    def test_target_free_ratios_share_one_observed_execution(self):
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            report = build_tuning_support(*make_evidence(Path(directory), target_free=True))
        self.assertEqual([point["observed_row"] for point in report["points"]], [5, 100])
        self.assertEqual(len(report["executions"]), 1)
        self.assertEqual(report["points"][0]["execution_ids"], report["points"][1]["execution_ids"])
        execution = next(iter(report["executions"].values()))
        self.assertEqual(execution["execution_evidence"]["training_sessions"], [])

    def test_comparison_runtime_is_shared_without_replacing_measured_evidence(self):
        references = [{"run_id": "previous-run"}]
        comparison = {
            "actual_runtime_seconds": 0.0,
            "comparison_runtime_seconds": 7.0,
            "comparison_runtime_status": "estimated",
            "comparison_runtime_basis": "fixture_reference",
            "comparison_runtime_sources": references,
            "comparison_runtime_details": {"reference_count": 1},
        }
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            root = Path(directory)
            inputs = make_evidence(root, target_free=True)
            add_tier_selection(inputs[0])
            metadata_path = root / inputs[-1][0]["metadata_file"]
            metadata_before = metadata_path.read_bytes()
            with patch("tests.ghl_main.build_tuning_support.load_runtime_references",
                       return_value=references) as load_references, patch(
                "tests.ghl_main.build_tuning_support.compare_execution_runtime", return_value=comparison,
            ) as compare:
                report = build_tuning_support(*inputs)
            load_references.assert_called_once_with(
                root / "experiments/01_ghl_main/logs/run_history/model_attempts", "budget",
            )
            compare.assert_called_once_with(
                json.loads((root / "snapshot_0.json").read_text()), 0.0, references,
                repository_root=root,
            )
            self.assertEqual(metadata_path.read_bytes(), metadata_before)
        self.assertEqual(len(report["executions"]), 1)
        self.assertEqual(len({tuple(point["execution_ids"]) for point in report["points"]}), 1)
        execution = next(iter(report["executions"].values()))
        self.assertEqual({key: execution[key] for key in comparison}, comparison)
        self.assertEqual(execution["execution_evidence"]["runtime_seconds"], 0.0)

    def test_missing_seed_or_changed_score_identity_is_rejected(self):
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            inputs = make_evidence(Path(directory))
            for broken in (3, 4):
                invalid = deepcopy(inputs)
                invalid[broken].pop()
                with self.assertRaises(ValueError):
                    build_tuning_support(*invalid)
            inputs[4][0]["score_sha256"] = "b" * 64
            with self.assertRaises(ValueError):
                build_tuning_support(*inputs)

    def test_missing_or_tampered_small_evidence_is_rejected(self):
        for name, missing in (("metadata_0.json", True), ("metadata_0.json", False),
                              ("snapshot_0.json", True), ("snapshot_0.json", False)):
            with self.subTest(name=name, missing=missing), TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(tuning, "REPOSITORY_ROOT", root):
                    inputs = make_evidence(root)
                    if missing:
                        (root / name).unlink()
                    else:
                        (root / name).write_text("{}", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        build_tuning_support(*inputs)

    def test_support_shape_cannot_be_replaced_by_an_unobserved_point(self):
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            inputs = make_evidence(Path(directory))
            inputs[0]["model_ratio"][0]["support"][0]["feature_count"] = 99
            with self.assertRaises(ValueError):
                build_tuning_support(*inputs)

    def test_bound_metadata_must_keep_physical_identity_and_training_counts(self):
        for change in ("seed", "fit", "test_sessions", "environment", "common_recipe_id", "config_registry_sha256"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(tuning, "REPOSITORY_ROOT", root):
                    inputs = make_evidence(root)
                    metadata = json.loads((root / "metadata_0.json").read_text())
                    if change == "seed":
                        metadata["seed"] = 9
                    elif change == "fit":
                        metadata["execution_evidence"]["training_sessions"][0]["observed_row"] = 99
                    elif change == "test_sessions":
                        metadata["execution_evidence"]["test_sessions"] *= 2
                    elif change in ("common_recipe_id", "config_registry_sha256"):
                        metadata[change] = "other"
                    else:
                        snapshot = json.loads((root / "snapshot_0.json").read_text())
                        snapshot["environment"] = {"cuda_device": {"name": "another GPU"}}
                        metadata["run_snapshot"] = write_json(root, "snapshot_0.json", snapshot)
                    reference = write_json(root, "metadata_0.json", metadata)
                    inputs[4][0]["metadata_sha256"] = reference["sha256"]
                    with self.assertRaises(ValueError):
                        build_tuning_support(*inputs)

    def test_snapshot_must_bind_series_source_and_current_parameters(self):
        for change in ("series", "source_sha", "verified_file", "hyperparameters", "source_ranges"):
            with self.subTest(change=change), TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(tuning, "REPOSITORY_ROOT", root):
                    inputs = make_evidence(root)
                    snapshot = json.loads((root / "snapshot_0.json").read_text())
                    if change == "series":
                        snapshot["spec"]["series"] = "02"
                    elif change == "source_sha":
                        snapshot["spec"]["series_input_sha256"] = "b" * 64
                    elif change == "verified_file":
                        snapshot["input_identity"]["verified_files"][0]["name"] = "other.csv"
                    elif change == "hyperparameters":
                        snapshot["spec"]["hyperparameters"]["window"] = 99
                    else:
                        snapshot["source_ranges"]["test_sessions"] = [[100, 141]]
                    metadata = json.loads((root / "metadata_0.json").read_text())
                    metadata["run_snapshot"] = write_json(root, "snapshot_0.json", snapshot)
                    reference = write_json(root, "metadata_0.json", metadata)
                    inputs[4][0]["metadata_sha256"] = reference["sha256"]
                    with self.assertRaises(ValueError):
                        build_tuning_support(*inputs)

    def test_small_input_manifest_must_match_the_sealed_budget(self):
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            root = Path(directory)
            inputs = make_evidence(root)
            manifest = json.loads((root / "configs/input_manifest.yaml").read_text())
            manifest["datasets"]["DEV18"]["files"][0]["feature_count"] = 99
            write_json(root, "configs/input_manifest.yaml", manifest)
            with self.assertRaises(ValueError):
                build_tuning_support(*inputs)

    def test_current_registry_must_match_the_sealed_candidate_space(self):
        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            inputs = make_evidence(Path(directory))
            inputs[1]["common_recipe"]["training_split"] = "older_split"
            with self.assertRaises(ValueError):
                build_tuning_support(*inputs)

    def test_serialized_observations_reach_the_candidate_consumer(self):
        from src.common.select_conditional_policy import recommend_conditional_candidates
        from tests.ghl_main.run_ratio_tuning import _write_json

        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            root = Path(directory)
            inputs = make_evidence(root)
            report = build_tuning_support(*inputs)
            support_path = root / "support.json"
            _write_json(support_path, report)
            result = recommend_conditional_candidates(
                inputs[0], inputs[1], nrows=100, dfeatures=3, planned_rows=100, evaluation_rows=40,
                support_path=support_path, support_sha256=hashlib.sha256(support_path.read_bytes()).hexdigest(),
                environment_id=report["points"][0]["environment_id"],
            )
        self.assertEqual(result["status"], "candidates_for_validation")
        self.assertEqual(result["plan_observation_status"], "complete_grid")
        self.assertEqual([(candidate["model"], candidate["service_status"])
                          for candidate in result["candidates"]], [("GDN", "unvalidated")])

    def test_pulse_family_heads_share_physical_inference_and_reach_recommendations(self):
        from src.common.select_conditional_policy import recommend_conditional_candidates

        with TemporaryDirectory() as directory, patch.object(tuning, "REPOSITORY_ROOT", Path(directory)):
            root = Path(directory)
            selection, registry, budget, ledger, manifest = make_evidence(root, target_free=True)
            identity = {"model": "TSPulse", "tier": "t3", "target_use": "strict_zero_shot"}
            details = registry["models"].pop("MWVAR")
            details.update(tier="t3", target_use="strict_zero_shot")
            parameters = {"context_length": 16, "patch_size": 4, "heads": 4, "aggregation_window": 4}
            details["candidates"][0]["hyperparameters"] = parameters
            registry["models"]["TSPulse"] = details
            registry["common_recipe"]["methodology_revision"] = "paper_tuning_v4"
            budget["registry_space_sha256"] = registry_space_sha256(registry)
            for panel in budget["model_panels"]:
                panel.update(model="TSPulse", tier="t3")
                for group in panel["groups"]:
                    group.update(model="TSPulse", tier="t3")
            for policy in selection["model_ratio"]:
                policy.update(model="TSPulse", tier="t3", hyperparameters=parameters,
                              score_variant="family_selected", score_variant_fallback="time",
                              score_variant_by_family={"other": "fft"} if policy["ratio"] == 5
                              else {"family": "ensemble"})
            add_tier_selection(selection)
            variants = ["time", "fft", "pred", "ensemble"]
            budget["execution_panel"][0].update(model="TSPulse", primary_score_variants=variants)
            snapshot = json.loads((root / "snapshot_0.json").read_text())
            snapshot["spec"].update(identity, hyperparameters=parameters, common_recipe=registry["common_recipe"])
            snapshot_reference = write_json(root, "snapshot_0.json", snapshot)
            metadata = json.loads((root / "metadata_0.json").read_text())
            metadata.update(identity, run_snapshot=snapshot_reference)
            rows = []
            for variant in variants:
                metadata["score_variant"] = variant
                reference = write_json(root, f"metadata_{variant}.json", metadata)
                rows.append({**manifest[0], "model": "TSPulse", "tier": "t3", "score_variant": variant,
                             "metadata_file": reference["file"], "metadata_sha256": reference["sha256"],
                             "score_file": f"absent_{variant}.npy"})
            trials = [{**trial, **row, "ratio": trial["ratio"]} for row in rows for trial in ledger]
            report = build_tuning_support(selection, registry, budget, trials, rows)
            reference = write_json(root, "support.json", report)
            loaded = load_tuning_support(root / reference["file"], reference["sha256"], selection, registry)
            result = recommend_conditional_candidates(
                selection, registry, nrows=100, dfeatures=3, planned_rows=100, evaluation_rows=40,
                support_path=root / reference["file"], support_sha256=reference["sha256"],
                environment_id=loaded["points"][0]["environment_id"],
            )
            self.assertEqual({candidate["analysis_kind"] for candidate in result["candidates"]},
                             {"model_ratio", "tier_adaptive"})
            for candidate in result["candidates"]:
                self.assertEqual(candidate["score_variant"], "family_selected")
                self.assertEqual(candidate["score_variant_by_family"], {"family": "ensemble"})
                self.assertEqual(candidate["score_variant_fallback"], "time")
                self.assertEqual({point["score_variant"] for point in candidate["observations"]}, {"ensemble"})
            for changes in ({"score_variant": "time"}, {"family": "other", "score_variant": "time"},
                            {"family": None, "score_variant": "time"}):
                invalid = deepcopy(report)
                invalid["points"][1].update(changes)
                reference = write_json(root, "invalid_support.json", invalid)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    load_tuning_support(root / reference["file"], reference["sha256"], selection, registry)
            selection["model_ratio"][0]["score_variant_by_family"] = {"family": "unknown"}
            with self.assertRaises(ValueError):
                build_tuning_support(selection, registry, budget, trials, rows)
        self.assertEqual({point["score_variant"] for point in report["points"]}, {"time", "ensemble"})
        self.assertEqual(len(report["executions"]), 2)
        self.assertEqual(len({execution["physical_execution_id"] for execution in report["executions"].values()}), 1)


if __name__ == "__main__":
    unittest.main()
