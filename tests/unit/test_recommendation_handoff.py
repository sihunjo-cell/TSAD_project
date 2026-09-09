"""원격에서 작은 합성 파일로 전달 묶음의 범위와 지문 검사를 확인한다."""

import csv
import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.common.execution_identity import file_sha256
from tests.ghl_main import package_recommendation_handoff as packaging
from tests.ghl_main import run_dev18_tuning as tuning


def prepare_scoring_files(root, input_manifest_sha256):
    source_root = Path(__file__).resolve().parents[2]
    for relative in ("src/채점기/vus_pr.py", "tests/ghl_main/build_dev18_ell_max.py"):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    ell_max = {
        "series_count": 18, "input_manifest_sha256": input_manifest_sha256,
        "generator_sha256": file_sha256(root / "tests/ghl_main/build_dev18_ell_max.py"),
        "series": [{"series": f"{index:02d}", "l_max_samples": 3} for index in range(1, 19)],
    }
    ell_max["ell_max_id"] = hashlib.sha256(tuning._json(ell_max).encode("utf-8")).hexdigest()
    comparison = {"status": "passed", "official": {"commit": tuning.OFFICIAL_TSB_AD_COMMIT, "version": "opt"},
                  "n_thresholds": 250, "evaluator_sha256": file_sha256(root / "src/채점기/vus_pr.py"),
                  "maximum_absolute_difference": 0.0, "absolute_tolerance": 1e-12}
    for relative, payload in (
        ("experiments/01_ghl_main/snapshots/dev18_selection/dev18_ell_max.json", ell_max),
        ("experiments/checks/reference_code/vus_pr/official_tsb_ad_comparison.json", comparison),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return ell_max, comparison


class RecommendationHandoffTests(unittest.TestCase):
    def test_failed_inference_training_evidence_is_retained_and_verified(self):
        for history_source, training_status in (
            ("training_complete", "complete"), ("training_complete", "saving"),
            ("result_files", "complete"), ("result_files", "saving"),
        ):
            with self.subTest(source=history_source, status=training_status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                report, _ = self.prepare_files(root)
                result = Path(report["result_directory"])
                training = root / "experiments/01_ghl_main/scores/dev18/tier2/PaAno/config/series_01/training_attempts/failed/training"
                training.mkdir(parents=True)
                files = {}
                for name, filename, payload in (("checkpoint", "checkpoint.ckpt", b"selected training state"),
                                                 ("training_log", "training_log.json", b'{"selected_iteration":2}')):
                    if training_status == "saving" and name == "training_log":
                        continue
                    path = training / filename
                    path.write_bytes(payload)
                    files[name] = {"file": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
                                   "bytes": path.stat().st_size}
                history = root / "experiments/01_ghl_main/logs/run_history/model_attempts/failed.json"
                history.write_text(json.dumps({"identity": {"budget_id": "new", "kind": "model_attempt"},
                    "run_id": "failed", "status": "failed", "model_execution_complete": False,
                    **({"training_complete": {"status": training_status, "files": files}}
                       if history_source == "training_complete" else {"result_files": files})}), encoding="utf-8")
                costs_path = result / "tuning_cost_history.json"
                costs = json.loads(costs_path.read_text(encoding="utf-8"))
                costs["model_attempt_wall"]["history_files"].append({"file": str(history), "sha256": file_sha256(history)})
                costs_path.write_text(json.dumps(costs), encoding="utf-8")
                with patch.object(packaging, "REPOSITORY_ROOT", root):
                    handoff = packaging.package_recommendation_handoff(report)
                    inventory = json.loads(Path(handoff["manifest"]["file"]).read_text(encoding="utf-8"))
                    self.assertIn(str(root / files["checkpoint"]["file"]),
                                  {item["file"] for item in inventory["external_artifacts"]})
                    checkpoint_item = next(item for item in inventory["external_artifacts"]
                                           if item["file"] == str(root / files["checkpoint"]["file"]))
                    self.assertEqual(checkpoint_item["kind"], f"{history_source}_checkpoint")
                    if "training_log" in files:
                        log_item = next(item for item in inventory["included_files"]
                                        if item["file"] == str(root / files["training_log"]["file"]))
                        with tarfile.open(handoff["archive"]["file"], "r:gz") as archive:
                            self.assertEqual(json.load(archive.extractfile(log_item["archive_name"])), {"selected_iteration": 2})
                    (root / files["checkpoint"]["file"]).write_bytes(b"changed state")
                    with self.assertRaisesRegex(ValueError, "지문"):
                        packaging.package_recommendation_handoff(report)

    def test_measured_pca_requires_original_probe_in_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, _ = self.prepare_files(root)
            gate_path = root / "experiments/checks/reference_code/active_models/full_prefix_v2/resource_gate.json"
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            row = gate["results"][0]
            row.pop("gpu_peak_bytes")
            row.update(measurement_kind="process_rss", actual_backend="cpu", ram_peak_bytes=600,
                       ram_total_bytes=1000, ram_peak_percent=60, maximum_memory_percent=80, wall_time_seconds=2,
                       pca_estimate={"estimated_ram_bytes": 900, "status": "requires_measurement"})
            probe_path = Path(row["probe_history"]["file"])
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            probe["result"] = {key: value for key, value in row.items() if key != "probe_history"}
            probe_path.write_text(json.dumps(probe), encoding="utf-8")
            row["probe_history"]["sha256"] = file_sha256(probe_path)
            gate_path.write_text(json.dumps(gate), encoding="utf-8")
            with patch.object(packaging, "REPOSITORY_ROOT", root):
                handoff = packaging.package_recommendation_handoff(report)
                inventory = json.loads(Path(handoff["manifest"]["file"]).read_text(encoding="utf-8"))
                self.assertIn(str(probe_path), {item["file"] for item in inventory["included_files"]})
                row.pop("probe_history")
                gate_path.write_text(json.dumps(gate), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "자원 probe 이력의 파일"):
                    packaging.package_recommendation_handoff(report)

    def prepare_files(self, root, model="PCA_LEGACY"):
        result = root / "experiments/01_ghl_main/results/dev18_tuning/full_prefix_v2"
        snapshot = root / "experiments/01_ghl_main/snapshots/dev18_selection/full_prefix_v2"
        evidence = result / "recommendation_evidence"
        variant = "time" if model == "TSPulse" else ""

        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
            return {"file": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}

        budget = {"budget_id": "new", "budget_sha256": "b" * 64, "experiment_mode": "full_prefix_v2", "execution_panel": [{
            "model": model, "config_id": "config", "physical_ratio": 5, "seed": 0,
            "series_ids": ["01"], "primary_score_variants": [variant], "diagnostic_score_variants": []}]}
        for field, name in (("input_manifest_sha256", "input_manifest.yaml"),
                            ("data_preprocessing_sha256", "data_preprocessing.yaml")):
            budget[field] = write(root / "configs" / name, {"sealed": name})["sha256"]
        registry = write(root / "configs/model_registry.yaml", {"registry": "sealed"})
        budget_reference = write(snapshot / "budget.json", {
            "budget": budget, "attestation": {"config_registry_sha256": registry["sha256"]}})
        identity = {"budget_id": "new", "project_commit": "source"}
        write(snapshot / "recommendation_contract.json", {"identity": identity})
        feasibility = snapshot / "dev18_feasibility_ledger.csv"
        feasibility.write_text(f"model,status\n{model},feasible\n", encoding="utf-8")
        write(snapshot / "dev18_feasibility_summary.json", {"ledger_sha256": file_sha256(feasibility)})
        backup = write(evidence / "recommendation_backup.sqlite3", {"consistent_backup": True})
        backup_path = evidence / f"recommendation_backup_{backup['sha256']}.sqlite3"
        Path(backup["file"]).replace(backup_path)
        backup["file"] = str(backup_path)
        write(evidence / "recommendation.sqlite3", {"live_database": "must not be copied"})
        exported = write(evidence / "exports/prefix_features.csv", "exported_rows")
        environment = {"runtime": "sealed"}
        recommendation = {"recommendation_complete": True, "identity": identity,
                          "environment": {**environment, "runtime_snapshot": {"identity": environment}},
                          "database": backup, "exports": [exported]}
        receipt = write(evidence / "recommendation_receipt.json", recommendation)
        recommendation.update(receipt_file=receipt["file"], receipt_sha256=receipt["sha256"])
        checks = root / "experiments/checks/reference_code/active_models/full_prefix_v2"
        resource_identity = {"project_commit": "source", "budget_id": "new",
                             "input_manifest_sha256": budget["input_manifest_sha256"],
                             "environment": environment}
        probe_result = {"model": model, "config_id": "config", "status": "passed", "gpu_peak_bytes": 123}
        probe = write(checks / "resource_probe_history/probe.json", {
            "identity": {**resource_identity, "kind": "resource_probe"}, "run_id": "probe",
            "status": "complete", "elapsed_seconds": 2.0, "result": probe_result})
        write(checks / "resource_gate.json", {**resource_identity, "status": "passed", "checked_models": [model],
              "probe_history_directory": str(checks / "resource_probe_history"),
              "results": [{**probe_result, "probe_history": {key: probe[key] for key in ("file", "sha256")}}]})
        command = write(root / "experiments/01_ghl_main/logs/run_history/commands/command.json",
                        {"identity": {"budget_id": "new"}, "status": "complete", "run_id": "command"})
        attempt = write(root / "experiments/01_ghl_main/logs/run_history/model_attempts/attempt.json",
                        {"identity": {"budget_id": "new"}, "status": "complete", "run_id": "attempt"})
        unassigned = write(root / "experiments/01_ghl_main/logs/run_history/commands/unassigned.json",
                           {"identity": {"budget_id": None, "kind": "tuning_command"},
                            "status": "failed", "run_id": "unassigned"})
        write(result / "tuning_cost_history.json", {"budget_id": "new",
              "command_wall": {"history_files": [command]}, "model_attempt_wall": {"history_files": [attempt]},
              "unassigned_command_wall": {"history_files": [unassigned]}})
        tier = "t3" if model in {"TimeRCD", "TSPulse"} else "t1"
        model_directory = root / f"experiments/01_ghl_main/scores/dev18/tier{tier[1:]}/{model}/config"
        raw = write(model_directory / f"DEV18__01__{model}__{tier}__r005__s0__raw__trainnorm.npy", "raw_score")
        write(model_directory / f"DEV18__01__{model}__{tier}__r005__s0__smoothed__trainnorm.npy", "smoothed_score")
        checkpoint = write(model_directory / "training/checkpoint.ckpt", "trained_parameters")
        scaler = write(model_directory / "training/scaler_state.json", {"scale": [1]})
        specification = {"model": model, "source_commit": "a" * 40, "source_checkpoint_sha256": "none"}
        if tier == "t3":
            pretrained = {name: write(root / f"pretrained/{model}/{name}.bin", f"original_{model}_{name}")
                          for name in ("checkpoint", "config")}
            specification.update(source_checkpoint_sha256=pretrained["checkpoint"]["sha256"],
                                 checkpoint_config_sha256=pretrained["config"]["sha256"])
            write(checks / "checkpoints" / {"TimeRCD": "time_rcd", "TSPulse": "tspulse"}[model]
                  / "dev18_checkpoint_smoke.json", {
                "status": "passed", "budget_id": budget["budget_id"], "budget_sha256": budget["budget_sha256"],
                "source": {"commit": specification["source_commit"]},
                **{name: {"path": item["file"], "sha256": item["sha256"], "bytes": item["bytes"]}
                   for name, item in pretrained.items()}})
        run_snapshot = write(model_directory / "series_01/run_snapshot.json", {
            "project_commit": "source", "storage_schema_version": "full_prefix_storage.v1",
            "environment": recommendation["environment"], "spec": specification})
        metadata = write(model_directory / "result.meta.json", {"model": model, "config_id": "config",
            "series": 1, "ratio": 5, "seed": 0, "score_variant": variant or None, "channel_count": 0,
            "run_snapshot": run_snapshot, "training_files": {"checkpoint": checkpoint, "scaler_state": scaler},
            "execution_attempt": {"run_id": "attempt", "history_file": attempt["file"]}})
        manifest = root / "experiments/01_ghl_main/logs/dev18_full_prefix_v2_manifest.csv"
        row = {"series": "01", "model": model, "config_id": "config", "physical_ratio": "5", "seed": "0",
               "score_variant": variant, "status": "complete", "budget_id": "new", "score_file": raw["file"],
               "score_sha256": raw["sha256"], "metadata_file": metadata["file"], "metadata_sha256": metadata["sha256"]}
        with manifest.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        ell_max, comparison = prepare_scoring_files(root, budget["input_manifest_sha256"])
        ledger_path = result / "dev18_trial_score_ledger.csv"
        ledger_row = {**row, "family": "synthetic", "tier": tier, "ratio": 5,
                      "normalization": "trainnorm", "vus_pr": 0.75, "status_reason": "",
                      "evaluator_sha256": comparison["evaluator_sha256"], "ell_max_id": ell_max["ell_max_id"]}
        with ledger_path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=tuning.TRIAL_SCORE_LEDGER_FIELDS)
            writer.writeheader()
            writer.writerow({field: ledger_row[field] for field in tuning.TRIAL_SCORE_LEDGER_FIELDS})
        write(result / "unrelated_old_result.csv", "unlisted_old_result")
        report = {"status": "ready_for_handoff", "experiment_mode": "full_prefix_v2", "budget_id": "new", "project_commit": "source",
                  "execution_status": "complete", "scoring_selection_status": "complete", "recommendation_status": "complete",
                  "result_directory": str(result), "recommendation_evidence": recommendation,
                  "history_file": command["file"], "run_id": "command", "outputs": {"dev18_trial_score_ledger.csv": file_sha256(ledger_path)},
                  "artifacts": {"budget": budget_reference, "manifest": {"file": str(manifest), "sha256": file_sha256(manifest)}}}
        write(result / "selection_complete.json", report)
        return report, Path(checkpoint["file"])

    def test_handoff_contains_backup_and_small_evidence_with_external_blob_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, checkpoint = self.prepare_files(root)
            with patch.object(packaging, "REPOSITORY_ROOT", root):
                handoff = packaging.package_recommendation_handoff(report)
            inventory = json.loads(Path(handoff["manifest"]["file"]).read_text(encoding="utf-8"))
            with tarfile.open(handoff["archive"]["file"]) as archive:
                names = archive.getnames()
                self.assertIn("handoff_manifest.json", names)
                self.assertIn("experiments/01_ghl_main/logs/run_history/commands/unassigned.json", names)
                for name in ("experiments/01_ghl_main/snapshots/dev18_selection/dev18_ell_max.json",
                             "experiments/checks/reference_code/vus_pr/official_tsb_ad_comparison.json"):
                    self.assertIn(name, names)
                    self.assertEqual(archive.extractfile(name).read(), (root / name).read_bytes())
                self.assertTrue(any(Path(name).name.startswith("recommendation_backup_") and name.endswith(".sqlite3") for name in names))
                self.assertFalse(any(name.endswith((".npy", ".ckpt", "scaler_state.json", "recommendation.sqlite3", "unrelated_old_result.csv")) for name in names))
            self.assertIn(str(checkpoint), {item["file"] for item in inventory["external_artifacts"]})
            self.assertEqual(len(inventory["external_artifacts"]), 4)
            self.assertEqual(file_sha256(handoff["archive"]["file"]), handoff["archive"]["sha256"])
            self.assertEqual(next(item["kind"] for item in inventory["included_files"]
                                  if Path(item["file"]).name == "unassigned.json"), "unassigned_command_history")

    def test_scoring_evidence_missing_or_different_from_ledger_blocks_handoff(self):
        for failure in ("missing_ell_max", "missing_comparison", "ell_max_id", "evaluator_sha256",
                        "input_manifest", "thresholds", "empty_ledger", "later_ledger_row"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                report, _ = self.prepare_files(root)
                result = Path(report["result_directory"])
                ell_path = root / "experiments/01_ghl_main/snapshots/dev18_selection/dev18_ell_max.json"
                comparison_path = root / "experiments/checks/reference_code/vus_pr/official_tsb_ad_comparison.json"
                if failure.startswith("missing"):
                    (ell_path if failure == "missing_ell_max" else comparison_path).unlink()
                elif failure in {"ell_max_id", "input_manifest"}:
                    snapshot = json.loads(ell_path.read_text(encoding="utf-8"))
                    snapshot.pop("ell_max_id")
                    if failure == "ell_max_id":
                        snapshot["series"][0]["l_max_samples"] += 1
                    else:
                        snapshot["input_manifest_sha256"] = "a" * 64
                    snapshot["ell_max_id"] = hashlib.sha256(tuning._json(snapshot).encode("utf-8")).hexdigest()
                    ell_path.write_text(json.dumps(snapshot), encoding="utf-8")
                elif failure in {"evaluator_sha256", "thresholds"}:
                    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                    if failure == "evaluator_sha256":
                        evaluator = root / "src/채점기/vus_pr.py"
                        evaluator.write_text("# different evaluator\n", encoding="utf-8")
                        comparison["evaluator_sha256"] = file_sha256(evaluator)
                    else:
                        comparison["n_thresholds"] = 249
                    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
                else:
                    ledger_path = result / "dev18_trial_score_ledger.csv"
                    with ledger_path.open(encoding="utf-8", newline="") as source:
                        rows = list(csv.DictReader(source))
                    with ledger_path.open("w", encoding="utf-8", newline="") as destination:
                        writer = csv.DictWriter(destination, fieldnames=tuning.TRIAL_SCORE_LEDGER_FIELDS)
                        writer.writeheader()
                        if failure == "later_ledger_row":
                            writer.writerows([*rows, {**rows[0], "series": "02", "ell_max_id": "a" * 64}])
                    report["outputs"][ledger_path.name] = file_sha256(ledger_path)
                    (result / "selection_complete.json").write_text(json.dumps(report), encoding="utf-8")
                with patch.object(packaging, "REPOSITORY_ROOT", root), self.assertRaises(ValueError):
                    packaging.package_recommendation_handoff(report)

    def test_tier3_snapshot_sources_include_smoke_and_pretrained_inventory(self):
        for model in ("TimeRCD", "TSPulse"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                report, _ = self.prepare_files(root, model=model)
                with patch.object(packaging, "REPOSITORY_ROOT", root):
                    handoff = packaging.package_recommendation_handoff(report)
                inventory = json.loads(Path(handoff["manifest"]["file"]).read_text(encoding="utf-8"))
                originals = {item["kind"]: item for item in inventory["external_artifacts"]
                             if item["kind"].startswith("pretrained_")}
                self.assertEqual(set(originals), {"pretrained_checkpoint", "pretrained_config"})
                for item in originals.values():
                    self.assertEqual(file_sha256(item["file"]), item["sha256"])
                    self.assertEqual(Path(item["file"]).stat().st_size, item["bytes"])
                with tarfile.open(handoff["archive"]["file"]) as archive:
                    self.assertTrue(any(name.endswith("dev18_checkpoint_smoke.json") for name in archive.getnames()))
                    self.assertFalse(any(name.startswith("pretrained/") for name in archive.getnames()))

    def test_native_channel_scores_are_included_in_external_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, _ = self.prepare_files(root, model="GDN")
            manifest_path = Path(report["artifacts"]["manifest"]["file"])
            with manifest_path.open(encoding="utf-8", newline="") as source:
                row = next(csv.DictReader(source))
            metadata_path = Path(row["metadata_file"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["native_channel_score_shape"] = [4, 2]
            channels = []
            raw = Path(row["score_file"])
            for variant in ("raw", "smoothed"):
                path = raw.with_name(raw.name.replace("__raw__trainnorm.npy", f"__{variant}__trainnorm__channels.npy"))
                path.write_bytes(b"native channels")
                channels.append(path)
            metadata["score_files"] = [{"file": str(path), "sha256": file_sha256(path),
                                        "bytes": path.stat().st_size} for path in channels]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            row["metadata_sha256"] = file_sha256(metadata_path)
            with manifest_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            report["artifacts"]["manifest"]["sha256"] = file_sha256(manifest_path)
            (Path(report["result_directory"]) / "selection_complete.json").write_text(json.dumps(report), encoding="utf-8")
            with patch.object(packaging, "REPOSITORY_ROOT", root):
                handoff = packaging.package_recommendation_handoff(report)
                inventory = json.loads(Path(handoff["manifest"]["file"]).read_text(encoding="utf-8"))
                self.assertTrue({str(path) for path in channels} <=
                                {item["file"] for item in inventory["external_artifacts"]})
                channels[0].write_bytes(b"changed native channels")
                with self.assertRaisesRegex(ValueError, "지문"):
                    packaging.package_recommendation_handoff(report)

    def test_pretrained_missing_or_changed_reference_blocks_handoff(self):
        for change in ("missing", "sha", "path", "bytes"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                report, _ = self.prepare_files(root, model="TSPulse")
                path = root / "experiments/checks/reference_code/active_models/full_prefix_v2/checkpoints/tspulse/dev18_checkpoint_smoke.json"
                if change == "missing":
                    path.unlink()
                elif change == "sha":
                    (root / "pretrained/TSPulse/checkpoint.bin").write_text("changed", encoding="utf-8")
                else:
                    source = json.loads(path.read_text(encoding="utf-8"))
                    source["checkpoint"][change] = "" if change == "path" else None
                    path.write_text(json.dumps(source), encoding="utf-8")
                with patch.object(packaging, "REPOSITORY_ROOT", root), self.assertRaises(ValueError):
                    packaging.package_recommendation_handoff(report)

    def test_tier3_missing_snapshot_checkpoint_identity_cannot_skip_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, _ = self.prepare_files(root, model="TSPulse")
            model_directory = root / "experiments/01_ghl_main/scores/dev18/tier3/TSPulse/config"
            snapshot_path = model_directory / "series_01/run_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            del snapshot["spec"]["source_checkpoint_sha256"]
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            metadata_path = model_directory / "result.meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["run_snapshot"]["sha256"] = file_sha256(snapshot_path)
            metadata["run_snapshot"]["bytes"] = snapshot_path.stat().st_size
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            manifest_path = Path(report["artifacts"]["manifest"]["file"])
            with manifest_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            rows[0]["metadata_sha256"] = file_sha256(metadata_path)
            with manifest_path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report["artifacts"]["manifest"]["sha256"] = file_sha256(manifest_path)
            (Path(report["result_directory"]) / "selection_complete.json").write_text(json.dumps(report), encoding="utf-8")
            with patch.object(packaging, "REPOSITORY_ROOT", root), self.assertRaisesRegex(ValueError, "snapshot"):
                packaging.package_recommendation_handoff(report)

    def test_resource_histories_preserve_reuse_failures_and_unknown_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, _ = self.prepare_files(root)
            checks = root / "experiments/checks/reference_code/active_models/full_prefix_v2"
            for name, budget, status in (("failed", "new", "failed"), ("interrupted", "new", "interrupted"),
                                         ("running", "new", "running"), ("other", "other", "complete")):
                (checks / f"resource_probe_history/{name}.json").write_text(json.dumps({
                    "identity": {"budget_id": budget, "kind": "resource_probe"}, "run_id": name,
                    "status": status, "elapsed_seconds": None if status == "running" else 1.0,
                }), encoding="utf-8")
            histories = {path.name: path.read_bytes() for path in (checks / "resource_probe_history").glob("*.json")}
            costs = Path(report["result_directory"]) / "tuning_cost_history.json"
            original_costs = costs.read_bytes()
            with patch.object(packaging, "REPOSITORY_ROOT", root):
                for _ in range(2):
                    handoff = packaging.package_recommendation_handoff(report)
                    with tarfile.open(handoff["archive"]["file"]) as archive:
                        names = archive.getnames()
                        for name in ("probe", "failed", "interrupted", "running"):
                            member = (checks / f"resource_probe_history/{name}.json").relative_to(root).as_posix()
                            self.assertEqual(names.count(member), 1)
                            self.assertEqual(archive.extractfile(member).read(), histories[name + ".json"])
                        self.assertNotIn((checks / "resource_probe_history/other.json").relative_to(root).as_posix(), names)
                        self.assertIn((checks / "resource_gate.json").relative_to(root).as_posix(), names)
            self.assertEqual(costs.read_bytes(), original_costs)

    def test_invalid_resource_report_or_probe_blocks_handoff(self):
        for change in ("missing", "failed", "budget", "results", "sha", "probe_budget", "probe_result", "probe_ref"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                report, _ = self.prepare_files(root)
                path = root / "experiments/checks/reference_code/active_models/full_prefix_v2/resource_gate.json"
                resource = json.loads(path.read_text(encoding="utf-8"))
                probe_path = Path(resource["results"][0]["probe_history"]["file"])
                if change == "missing":
                    path.unlink()
                else:
                    if change in {"failed", "budget"}:
                        resource["status" if change == "failed" else "budget_id"] = change
                    elif change == "results":
                        resource["results"] = [{"resource": "disk", "status": "passed"}]
                    elif change == "probe_ref":
                        resource["results"][0]["probe_history"]["sha256"] = ""
                    else:
                        probe = json.loads(probe_path.read_text(encoding="utf-8"))
                        if change == "probe_budget":
                            probe["identity"]["budget_id"] = "other"
                        else:
                            probe["result"]["gpu_peak_bytes"] = 456
                        probe_path.write_text(json.dumps(probe), encoding="utf-8")
                        if change != "sha":
                            resource["results"][0]["probe_history"]["sha256"] = file_sha256(probe_path)
                    path.write_text(json.dumps(resource), encoding="utf-8")
                with patch.object(packaging, "REPOSITORY_ROOT", root), self.assertRaises(ValueError):
                    packaging.package_recommendation_handoff(report)

    def test_unassigned_history_cannot_hide_another_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, _ = self.prepare_files(root)
            costs_path = Path(report["result_directory"]) / "tuning_cost_history.json"
            costs = json.loads(costs_path.read_text(encoding="utf-8"))
            item = costs["unassigned_command_wall"]["history_files"][0]
            Path(item["file"]).write_text(json.dumps({"identity": {"budget_id": "other"}}), encoding="utf-8")
            item["sha256"] = file_sha256(item["file"])
            costs_path.write_text(json.dumps(costs), encoding="utf-8")
            with patch.object(packaging, "REPOSITORY_ROOT", root), self.assertRaisesRegex(ValueError, "미배정"):
                packaging.package_recommendation_handoff(report)

    def test_handoff_retry_preserves_failure_cost_without_rewriting_command_history(self):
        for error, expected_status in ((OSError("disk full"), "failed"), (KeyboardInterrupt(), "interrupted")):
            with self.subTest(status=expected_status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                report, _ = self.prepare_files(root)
                command_content = Path(report["history_file"]).read_bytes()
                costs_path = Path(report["result_directory"]) / "tuning_cost_history.json"
                costs_content = costs_path.read_bytes()
                histories = Path(report["result_directory"]) / "handoff/run_history"
                with patch.object(packaging, "REPOSITORY_ROOT", root):
                    with patch.object(packaging.tarfile, "open", side_effect=error), self.assertRaises(type(error)):
                        packaging.package_recommendation_handoff(report)
                    failed_files = list(histories.glob("*.json"))
                    self.assertEqual(len(failed_files), 1)
                    failed_content = failed_files[0].read_bytes()
                    failed = json.loads(failed_content)
                    self.assertEqual(failed["status"], expected_status)
                    self.assertGreaterEqual(failed["elapsed_seconds"], 0)
                    self.assertEqual(failed["identity"]["budget_id"], "new")
                    self.assertEqual(failed["identity"]["cost_scope"], "handoff_archive_and_hash")
                    abandoned = histories / "abandoned.json"
                    abandoned.write_text(json.dumps({
                        "identity": {"budget_id": "new"}, "run_id": "abandoned",
                        "status": "running", "elapsed_seconds": None,
                    }), encoding="utf-8")
                    handoff = packaging.package_recommendation_handoff(report)
                current = handoff["history"]
                closed = json.loads(Path(current["file"]).read_text(encoding="utf-8"))
                self.assertEqual(closed["status"], "complete")
                self.assertGreaterEqual(closed["elapsed_seconds"], 0)
                for name in ("archive", "manifest"):
                    self.assertEqual(closed["artifacts"][name], {
                        "sha256": handoff[name]["sha256"], "bytes": handoff[name]["bytes"],
                    })
                self.assertEqual(file_sha256(current["file"]), current["sha256"])
                self.assertEqual(Path(current["file"]).stat().st_size, current["bytes"])
                with tarfile.open(handoff["archive"]["file"]) as archive:
                    self.assertIn(failed_files[0].relative_to(root).as_posix(), archive.getnames())
                    self.assertIn(abandoned.relative_to(root).as_posix(), archive.getnames())
                    self.assertNotIn(Path(current["file"]).relative_to(root).as_posix(), archive.getnames())
                self.assertEqual(failed_files[0].read_bytes(), failed_content)
                self.assertEqual(Path(report["history_file"]).read_bytes(), command_content)
                self.assertEqual(costs_path.read_bytes(), costs_content)

    def test_changed_saved_checkpoint_blocks_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            report, checkpoint = self.prepare_files(root)
            checkpoint.write_text("changed", encoding="utf-8")
            with patch.object(packaging, "REPOSITORY_ROOT", root), self.assertRaisesRegex(ValueError, "지문"):
                packaging.package_recommendation_handoff(report)
            self.assertFalse((Path(report["result_directory"]) / "handoff/recommendation_handoff.tar.gz").exists())

    def test_incomplete_recommendation_cannot_be_packaged(self):
        with self.assertRaisesRegex(ValueError, "완료"):
            packaging.package_recommendation_handoff({"status": "execution_complete", "experiment_mode": "full_prefix_v2"})


if __name__ == "__main__":
    unittest.main()
