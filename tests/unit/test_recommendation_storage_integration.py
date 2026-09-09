"""원격에서 합성 출력 한 건의 실제 저장·DB 검증·백업·인수를 잇는다."""

import csv
import json
import sqlite3
import tarfile
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy

from src.common.execution_identity import file_sha256
from src.common.model_registry import load_model_registry
from src.data_split.split_ratio_prefix import split_ratio_prefix
from tests.ghl_main import package_recommendation_handoff as packaging
from tests.ghl_main import run_dev18_tuning as tuning
from tests.ghl_main import store_recommendation_evidence as evidence
from tests.ghl_main.record_run_history import record_run_history
from tests.ghl_main.run_registered_models import build_output_directory
from tests.unit import test_recommendation_handoff as handoff_fixture
from tests.unit.test_recommendation_evidence import fixture, loader, write_source


class RecommendationStorageIntegrationTests(unittest.TestCase):
    def test_saved_output_passes_database_backup_and_handoff_without_mocked_verifiers(self):
        self._exercise_storage_pipeline(resume_after_resource_fix=False)

    def test_resource_fix_reuses_scores_database_and_handoff_without_rewriting_old_sources(self):
        self._exercise_storage_pipeline(resume_after_resource_fix=True)

    def _exercise_storage_pipeline(self, *, resume_after_resource_fix):
        registered = load_model_registry()
        model = registered["models"]["PaAno"]
        candidate = model["candidates"][0]
        registry = {"common_recipe": registered["common_recipe"],
                    "common_recipe_id": registered["common_recipe_id"],
                    "models": {"PaAno": {**model, "candidates": [candidate]}}}
        entries, _, _, _ = fixture()
        for entry in entries:
            entry.update(row_count=50, size_bytes=1)

        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            report, _ = handoff_fixture.RecommendationHandoffTests().prepare_files(root, model="PaAno")
            result_directory = Path(report["result_directory"])
            snapshot_directory = Path(report["artifacts"]["budget"]["file"]).parent
            store_directory = result_directory / "recommendation_evidence"
            # 기존 인수 fixture의 DB 자리표시자를 실제 SQLite로 교체한다.
            (store_directory / "recommendation.sqlite3").unlink()

            def write_json(path, value):
                path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
                return {"file": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}

            input_manifest = root / "configs/input_manifest.yaml"
            input_reference = write_json(input_manifest, {
                "roles": {"dev18_selection": {"dataset": "DEV18"}},
                "datasets": {"DEV18": {"files": entries}},
            })
            registry_reference = write_json(root / "configs/model_registry.yaml", registry)
            budget_path = Path(report["artifacts"]["budget"]["file"])
            artifact = json.loads(budget_path.read_text(encoding="utf-8"))
            budget = artifact["budget"]
            panel = budget["execution_panel"][0]
            panel.update(config_id=candidate["config_id"], physical_ratio=100, logical_ratios=[100])
            budget.update(input_manifest_sha256=input_reference["sha256"], expected_ledger_rows=1,
                          structural_exclusions=[])
            artifact["attestation"]["config_registry_sha256"] = registry_reference["sha256"]
            report["artifacts"]["budget"] = write_json(budget_path, artifact)
            ell_max, comparison = handoff_fixture.prepare_scoring_files(root, input_reference["sha256"])
            environment = report["recommendation_evidence"]["environment"]
            resource_path = root / "experiments/checks/reference_code/active_models/full_prefix_v2/resource_gate.json"
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            resource["input_manifest_sha256"] = input_reference["sha256"]
            resource["results"][0]["config_id"] = candidate["config_id"]
            probe_path = Path(resource["results"][0]["probe_history"]["file"])
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            probe["identity"]["input_manifest_sha256"] = input_reference["sha256"]
            probe["result"]["config_id"] = candidate["config_id"]
            resource["results"][0]["probe_history"] = write_json(probe_path, probe)
            write_json(resource_path, resource)

            entry = entries[0]
            spec = {**candidate, "model": "PaAno", "series": entry["series"], "ratio": 100, "seed": 0,
                    "dataset_role": "development", "split_role": "dev18_selection",
                    "input_manifest_sha256": input_reference["sha256"], "final_policy_membership_sha256": None,
                    "config_registry_sha256": registry_reference["sha256"], "series_input_sha256": entry["sha256"],
                    "common_recipe": registry["common_recipe"], "common_recipe_id": registry["common_recipe_id"],
                    "score_variants": ("",), **{field: model.get(field) for field in (
                        "tier", "target_use", "source_commit", "source_checkpoint_sha256",
                        "checkpoint_config_sha256", "checkpoint_revision", "preprocess_recipe")}}
            inputs = {**loader(entry), "family": entry["family"],
                      "test_sessions": (numpy.arange(20, dtype=numpy.float32).reshape(10, 2),),
                      "input_identity": {**{field: entry[field] for field in (
                          "name", "sha256", "source_directory", "size_bytes")},
                          "input_manifest_sha256": input_reference["sha256"]},
                      "source_ranges": {"source": entry["name"], "source_directory": entry["source_directory"],
                                        "normal_training": [0, 40], "test_sessions": [[40, 50]]}}
            _, _, split = split_ratio_prefix(inputs["normal_training"], 100, full_prefix=True)
            computed = {"split": split, "seed_state": {"seed": 0},
                        "checkpoint": {"synthetic_saved_state": [1.0, 2.0]},
                        "training_log": {"selected_iteration": 1, "loss_history": [{"iteration": 1, "loss": 0.25}]},
                        "effective_execution": {"backend": "cpu"},
                        "timing": {"split_preprocess_seconds": 0.1, "model_setup_seconds": 0.2,
                                   "training_seconds": 0.3, "calibration_inference_seconds": 0.0,
                                   "test_inference_seconds": 0.4},
                        "test_outputs": [{"scores": numpy.arange(10, dtype=float) / 10,
                            "source_start": 0, "source_end_exclusive": 10, "alignment": "same_timestep",
                            "primitive": "synthetic_distance", "calibration_mode": "none",
                            "normalization_scope": "none", "evaluation_mode": "offline_noncausal",
                            "lookahead": 0, "maximum_effective_lookahead": 0,
                            "native_postprocessing": True, "official_protocol": "paper_tuning_v4"}]}
            for module in (tuning, evidence, packaging):
                stack.enter_context(patch.object(module, "REPOSITORY_ROOT", root))
            stack.enter_context(patch.object(tuning, "_git_head", return_value=report["project_commit"]))
            stack.enter_context(patch.object(tuning, "_require_same_worktree"))
            stack.enter_context(patch("tests.ghl_main.check_registered_outputs.load_model_registry_with_sha",
                                      return_value=(registry, registry_reference["sha256"])))
            output_directory = build_output_directory(root / "experiments/01_ghl_main", spec)
            snapshot = tuning._write_run_snapshot(output_directory / "series_01", spec, inputs, environment)
            with record_run_history(root / "experiments/01_ghl_main/logs/run_history/model_attempts",
                                    identity={"kind": "model_attempt", "budget_id": budget["budget_id"],
                                              "model": "PaAno", "config_id": candidate["config_id"],
                                              "series": "01", "ratio": 100, "seed": 0, "attempt": 0}) as history:
                history["run_snapshot"] = tuning._read_json(snapshot)
                tuning._save_training_completion(snapshot.parent, {
                    **computed, "timing": {**computed["timing"], "test_inference_seconds": 0.0},
                }, history)
                checkpoint_reference = dict(history["training_complete"]["files"]["checkpoint"])
                with patch("torch.save", side_effect=AssertionError("completed checkpoint saved twice")):
                    rows = tuning._save_run_result(
                        computed, spec, panel, inputs, series=1, snapshot_path=snapshot, peak_memory_mb=None,
                        input_manifest_path=input_manifest, retry_count=0, budget_id=budget["budget_id"],
                        execution_attempt={"run_id": history["run_id"], "history_file": tuning._relative(history["history_file"])},
                        history=history,
                    )
                metadata = tuning._read_json(root / rows[0]["metadata_file"])
                self.assertEqual(metadata["training_files"]["checkpoint"], checkpoint_reference)
                self.assertEqual(tuning._read_json(root / metadata["training_files"]["timing"]["file"]), computed["timing"])
            saved_history = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
            saved_history.pop("result")
            saved_history["model_execution_complete"] = True
            write_json(Path(history["history_file"]), saved_history)
            (snapshot.parent / "completion.json").unlink()
            recovered = tuning._recover_saved_run_rows(
                spec=spec, panel_row=panel, series=1, family=entry["family"], budget_id=budget["budget_id"],
                computed={"attempt": 0, "run_id": history["run_id"], "history_file": history["history_file"]},
            )
            self.assertEqual(recovered, rows)
            manifest_path = Path(report["artifacts"]["manifest"]["file"])
            ledger_path = result_directory / "dev18_trial_score_ledger.csv"
            ledger = [{**rows[0], "ratio": 100, "normalization": "trainnorm", "vus_pr": 0.75,
                       "evaluator_sha256": comparison["evaluator_sha256"], "ell_max_id": ell_max["ell_max_id"]}]
            write_source(manifest_path, rows)
            write_source(ledger_path, ledger)
            identity = {"schema_version": evidence.SCHEMA_VERSION, "project_commit": report["project_commit"],
                        "storage_source_sha256": "d" * 64,
                        "budget_id": budget["budget_id"], "input_manifest_sha256": input_reference["sha256"],
                        "feasibility_ledger": {"file": tuning._relative(snapshot_directory / "dev18_feasibility_ledger.csv"),
                                               "sha256": file_sha256(snapshot_directory / "dev18_feasibility_ledger.csv")}}
            with evidence.RecommendationEvidence(store_directory, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                store.bind_environment(environment)
                store.prepare_features(loader)
                write_json(snapshot_directory / "recommendation_contract.json", store.contract)
                store.sync_results(rows, manifest_path=manifest_path)
                self.assertFalse(store.status()["scoring_complete"])
                store.sync_results(rows, ledger_rows=ledger, manifest_path=manifest_path, ledger_path=ledger_path)
                recommendation = store.finalize()
            if resume_after_resource_fix:
                from tests.checks import validate_resource_resume as resume

                original_commit = identity["project_commit"]
                current_commit = "9" * 40
                stack.enter_context(patch.object(resume, "_has_resource_resume_source",
                    side_effect=lambda commit, _root: commit in {original_commit, current_commit}))
                stack.enter_context(patch.object(resume, "committed_file_sha256", return_value="d" * 64))
                current_identity = {**identity, "project_commit": current_commit, "storage_source_sha256": "e" * 64}
                preserved_snapshot = snapshot.read_bytes()
                with evidence.RecommendationEvidence(store_directory, identity=current_identity, registry=registry,
                                                     budget=budget, entries=entries) as resumed:
                    self.assertEqual(resumed.identity["project_commit"], original_commit)
                    self.assertEqual(resumed.identity["storage_source_sha256"], "d" * 64)
                    resumed.bind_environment(environment)
                    resumed.prepare_features(lambda _entry: self.fail("cached prefixes must be reused"))
                    resumed.sync_results(rows, ledger_rows=ledger, manifest_path=manifest_path, ledger_path=ledger_path)
                    recommendation = resumed.finalize()
                self.assertTrue(tuning._completed_run(
                    rows, spec=spec, panel_row=panel, series=1, input_manifest_path=input_manifest,
                    expected_project_commit=current_commit, compatible_project_commit=None,
                    expected_environment=environment,
                ))
                self.assertEqual(snapshot.read_bytes(), preserved_snapshot)
                report["project_commit"] = current_commit
            self.assertTrue(recommendation["recommendation_complete"])
            with closing(sqlite3.connect(recommendation["database"]["file"])) as backup:
                self.assertEqual(backup.execute("SELECT count(*) FROM recommendation_inputs").fetchone()[0], 126)
                self.assertEqual(backup.execute("SELECT status,vus_pr,physical_execution_id FROM results").fetchone(),
                                 ("complete", 0.75, history["run_id"]))
            performance = next(item for item in recommendation["exports"] if "/performance/" in item["file"].replace("\\", "/"))
            with Path(performance["file"]).open(encoding="utf-8-sig", newline="") as source:
                exported = list(csv.DictReader(source))
            self.assertEqual([(row["status"], float(row["vus_pr"])) for row in exported], [("complete", 0.75)])

            costs_path = result_directory / "tuning_cost_history.json"
            costs = json.loads(costs_path.read_text(encoding="utf-8"))
            costs["model_attempt_wall"]["history_files"] = [{"file": history["history_file"],
                                                           "sha256": file_sha256(history["history_file"])}]
            write_json(costs_path, costs)
            report["recommendation_evidence"] = recommendation
            report["artifacts"]["manifest"] = {"file": str(manifest_path), "sha256": file_sha256(manifest_path)}
            report["outputs"] = {ledger_path.name: file_sha256(ledger_path)}
            write_json(result_directory / "selection_complete.json", report)
            handoff = packaging.package_recommendation_handoff(report)
            inventory = json.loads(Path(handoff["manifest"]["file"]).read_text(encoding="utf-8"))
            metadata = json.loads((root / rows[0]["metadata_file"]).read_text(encoding="utf-8"))
            checkpoint = root / metadata["training_files"]["checkpoint"]["file"]
            self.assertIn(str(checkpoint), {item["file"] for item in inventory["external_artifacts"]})
            with tarfile.open(handoff["archive"]["file"]) as archive:
                backup_name = Path(recommendation["database"]["file"]).relative_to(root).as_posix()
                self.assertEqual(archive.extractfile(backup_name).read(), Path(recommendation["database"]["file"]).read_bytes())
                log_name = metadata["training_files"]["training_log"]["file"]
                self.assertEqual(json.load(archive.extractfile(log_name)), computed["training_log"])
                ell_name = "experiments/01_ghl_main/snapshots/dev18_selection/dev18_ell_max.json"
                comparison_name = "experiments/checks/reference_code/vus_pr/official_tsb_ad_comparison.json"
                self.assertEqual(json.load(archive.extractfile(ell_name)), ell_max)
                self.assertEqual(json.load(archive.extractfile(comparison_name)), comparison)

            checkpoint.write_bytes(b"changed checkpoint")
            with evidence.RecommendationEvidence(store_directory, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as resumed:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    resumed.sync_results(rows, manifest_path=manifest_path)


if __name__ == "__main__":
    unittest.main()
