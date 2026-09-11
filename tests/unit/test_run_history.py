"""실행 이력의 시작·종료·중단과 이전 영수증 보존을 확인한다."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

from tests.ghl_main.record_run_history import (
    preserve_run_receipt,
    record_run_history,
    save_run_history,
    load_attempt_counts,
    summarize_run_history,
    summarize_pca_compute,
    record_run_stage,
    hold_tuning_lock,
)


class TestRunHistory(unittest.TestCase):
    def test_pca_cpu_accounting_keeps_wall_time_and_missing_old_measurements(self):
        with TemporaryDirectory() as directory:
            for index, (model, status, wall, cpu) in enumerate((
                ("PCA_LEGACY", "complete", 10.0, 27.0),
                ("PCA_LEGACY", "interrupted", 2.0, 5.0),
                ("PCA_LEGACY", "complete", 70.0, None),
                ("GDN", "complete", 100.0, 100.0),
            )):
                record = {"run_id": str(index), "identity": {"model": model, "budget_id": "b1"},
                          "status": status, "model_execution_seconds": wall}
                if cpu is not None:
                    record["resource_usage"] = {"pca_compute": {"wall_seconds": wall, "cpu_core_seconds": cpu}}
                Path(directory, f"{index}.json").write_text(json.dumps(record), encoding="utf-8")
            summary = summarize_pca_compute(directory, "b1")
            self.assertEqual(summary["known_cpu_core_seconds"], 32.0)
            self.assertIsNone(summary["total_cpu_core_seconds"])
            self.assertEqual(summary["unmeasured_run_ids"], ["2"])
            self.assertEqual([row["wall_seconds"] for row in summary["runs"]], [10.0, 2.0, 70.0])
            self.assertEqual(summary["runs"][0]["single_core_equivalent_seconds"], 27.0)
            self.assertTrue(all(row["single_thread_wall_seconds"] is None for row in summary["runs"]))
            self.assertEqual(summarize_pca_compute(directory, "other")["runs"], [])

    def test_failed_preflight_is_preserved_before_retry(self):
        from tests.ghl_main.run_ratio_tuning import _pending_check

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "check.json"
            content = b'{"status":"failed","reason":"transient"}'
            path.write_bytes(content)
            with record_run_history(root / "history", identity={"budget_id": "b1"}) as history:
                self.assertTrue(_pending_check(path, history))
                self.assertEqual(Path(history["prior_failed_checks"][0]).read_bytes(), content)
            path.write_text('{"status":"passed"}', encoding="utf-8")
            self.assertFalse(_pending_check(path, history))

    def test_lock_rejects_overlap_and_releases_after_interrupt(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tuning.lock"
            with self.assertRaises(KeyboardInterrupt):
                with hold_tuning_lock(path):
                    with self.assertRaises(RuntimeError):
                        with hold_tuning_lock(path):
                            pass
                    raise KeyboardInterrupt()
            with hold_tuning_lock(path):
                pass

    def test_preflight_reuses_passed_checks_and_runs_only_missing_checkpoint(self):
        from tests.ghl_main import run_ratio_tuning as runner
        from tests.checks import run_model_smoke, run_checkpoint_smoke, check_dev18_resources
        from tests.checks import run_lightning_dev18

        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(run_model_smoke, "FULL_PREFIX_OUTPUT_PATH", root / "model.json"))
            stack.enter_context(patch.object(run_checkpoint_smoke, "FULL_PREFIX_OUTPUT_ROOT", root))
            stack.enter_context(patch.object(check_dev18_resources, "FULL_PREFIX_REPORT_PATH", root / "resource.json"))
            for path in (root / "model.json", root / "resource.json",
                         root / run_checkpoint_smoke.MODEL_DIRECTORIES["TimeRCD"] / "dev18_checkpoint_smoke.json"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"status":"passed"}', encoding="utf-8")
            for owner, name in ((runner.tuning, "_require_cuda_or_remote"),
                                (runner.tuning, "_require_clean_worktree"),
                                (runner.tuning, "_validate_checkpoint_report"),
                                (run_model_smoke, "validate_full_prefix_smoke"),
                                (check_dev18_resources, "validate_resource_report"),
                                (run_lightning_dev18, "seal_lightning_runtime")):
                stack.enter_context(patch.object(owner, name))
            stack.enter_context(patch("src.common.set_reproducible_seed.set_reproducible_seed"))
            model = stack.enter_context(patch.object(run_model_smoke, "run_full_prefix_smoke"))
            resource = stack.enter_context(patch.object(check_dev18_resources, "run_resource_check"))
            checkpoint = stack.enter_context(patch.object(run_checkpoint_smoke, "run_dev18_checkpoint_smoke",
                return_value={"models": {"TSPulse": {"status": "passed"}}}))
            stack.enter_context(patch.object(check_dev18_resources.shutil, "disk_usage", side_effect=[
                SimpleNamespace(total=1000, used=999, free=1),
                SimpleNamespace(total=1000, used=1000, free=0),
            ]))
            for free_bytes in (1, 0):
                with record_run_history(root / "history", identity={"budget_id": "b1"}) as history:
                    runner.prepare_tuning_environment({}, {"experiment_mode": "full_prefix_v2"},
                        {"budget": root / "budget.json", "result": root / "results"}, data_root=root, history=history)
                saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
                observation = saved["stages"]["resource_gate"]["disk_observation"]
                self.assertEqual(observation["free_bytes"], free_bytes)
                self.assertEqual(observation["policy"], "informational_only")
                self.assertEqual(saved["status"], "complete")
            self.assertEqual((root / "resource.json").read_text(encoding="utf-8"), '{"status":"passed"}')
            self.assertEqual(checkpoint.call_args.kwargs["models"], ["TSPulse"])
            model.assert_not_called()
            resource.assert_not_called()

    def test_hard_stop_counts_as_attempt_and_leaves_total_cost_unknown(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, kind, extra in (
                ("one", "model_attempt", {"config_id": "c1", "attempt": 0}),
                ("two", "model_attempt", {"config_id": "c2", "attempt": 0}),
                ("three", "model_attempt", {"config_id": "c3", "attempt": 1}),
            ):
                save_run_history({
                    "run_id": run_id, "history_file": str(root / f"{run_id}.json"),
                    "identity": {"kind": kind, "budget_id": "b1", "series": "01",
                                 "model": "M", "ratio": 5, "seed": 0, **extra},
                    "status": "running", "elapsed_seconds": None,
                })
            counts = load_attempt_counts(root, "b1")
            self.assertEqual(counts["01", "M", "c1", "5", "0"], 1)
            self.assertEqual(counts["01", "M", "c2", "5", "0"], 1)
            self.assertEqual(counts["01", "M", "c3", "5", "0"], 2)
            summary = summarize_run_history(root, "b1")
            self.assertIsNone(summary["total_elapsed_seconds"])
            self.assertEqual(summary["unknown_elapsed_run_ids"], ["one", "three", "two"])
            self.assertEqual(summary["known_elapsed_seconds"], 0)
            self.assertEqual(len(summary["history_files"]), 3)

    def test_stage_failure_is_durable_and_prior_stage_survives_resume(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(KeyboardInterrupt):
                with record_run_history(directory, identity={"budget_id": "b1"}) as history:
                    with record_run_stage(history, "prepare"):
                        pass
                    with record_run_stage(history, "score"):
                        raise KeyboardInterrupt()
            saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["stages"]["prepare"]["status"], "complete")
            self.assertEqual(saved["stages"]["score"]["status"], "interrupted")
            self.assertGreaterEqual(saved["elapsed_seconds_lower_bound"], 0)
            summary = summarize_run_history(directory, "b1")
            self.assertEqual(summary["total_elapsed_seconds"], saved["elapsed_seconds"])

    def test_start_is_saved_before_work_and_later_runs_keep_prior_history(self):
        with TemporaryDirectory() as directory:
            with record_run_history(directory, identity={"mode": "prepare", "budget_id": None}) as first:
                first_path = Path(first["history_file"])
                started = json.loads(first_path.read_text(encoding="utf-8"))
                self.assertEqual(started["status"], "running")
                self.assertIsNone(started["elapsed_seconds"])
                self.assertIsNone(started["finished_at"])
                first["identity"]["budget_id"] = "b1"
                save_run_history(first)
                self.assertEqual(json.loads(first_path.read_text(encoding="utf-8"))["identity"]["budget_id"], "b1")
            completed = first_path.read_bytes()
            with record_run_history(directory, identity={"mode": "execute-only", "budget_id": "b1"}) as second:
                self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertEqual(first_path.read_bytes(), completed)
            self.assertEqual(json.loads(completed)["status"], "complete")
            self.assertGreaterEqual(json.loads(completed)["elapsed_seconds"], 0)

    def test_failures_and_interrupts_are_saved_and_raised(self):
        with TemporaryDirectory() as directory:
            for error, status in ((RuntimeError("failure"), "failed"), (KeyboardInterrupt(), "interrupted")):
                with self.subTest(status=status), self.assertRaises(type(error)):
                    with record_run_history(directory, identity={"budget_id": "b1"}) as record:
                        raise error
                saved = json.loads(Path(record["history_file"]).read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], status)
                self.assertEqual(saved["error_type"], type(error).__name__)
                self.assertIsNotNone(saved["finished_at"])
                self.assertGreaterEqual(saved["elapsed_seconds"], 0)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_receipt_archive_keeps_original_bytes_when_current_receipt_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "execution_complete.json"
            previous = b'{"execution_seconds": 123.0}\n'
            receipt.write_bytes(previous)
            archived = preserve_run_receipt(receipt, root / "receipts")
            self.assertEqual(archived.read_bytes(), previous)
            self.assertEqual(preserve_run_receipt(receipt, root / "receipts"), archived)
            receipt.write_text('{"execution_seconds": 0.01}\n', encoding="utf-8")
            current_archive = preserve_run_receipt(receipt, root / "receipts")
            self.assertNotEqual(archived, current_archive)
            self.assertEqual(archived.read_bytes(), previous)

    def test_resumed_command_separates_plan_pending_work_and_execution_timing(self):
        from tests.ghl_main import run_ratio_tuning as runner

        panel = {"model": "M", "config_id": "c1", "physical_ratio": 5, "seed": 0,
                 "series_ids": ["01", "02"], "primary_score_variants": [""],
                 "diagnostic_score_variants": []}
        budget = {"budget_id": "b1", "budget_sha256": "b" * 64,
                  "experiment_mode": "full_prefix_v2", "series_ids": ["01", "02"],
                  "execution_panel": [panel], "physical_execution_count": 1,
                  "physical_run_count": 2, "expected_ledger_rows": 2,
                  "primary_logical_score_row_count": 2}
        completed = {**panel, "series": "01", "score_variant": "", "status": "complete",
                     "budget_id": "b1", "primary_score": "true"}
        measured = {"completion_check_seconds": .2, "model_attempt_seconds": .8}

        def execute_without_models(**arguments):
            arguments["execution_timing"].update(measured)
            return [completed, {**completed, "series": "02"}]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"result": root, "manifest": root / "manifest.csv", "budget": root / "budget.json",
                     "recommendation": root / "recommendation_evidence"}
            evidence = MagicMock()
            evidence.__enter__.return_value = evidence
            evidence.status.return_value = {"feature_complete": True}
            evidence.finalize.return_value = {"status": "incomplete", "execution_complete": True}
            paths["manifest"].write_text("mock manifest\n", encoding="utf-8")
            paths["budget"].write_text("{}\n", encoding="utf-8")
            arguments = SimpleNamespace(
                baseline_manifest=root / "old.csv", baseline_ledger=root / "old_ledger.csv",
                prepare=False, execute_only=True, finish_only=False, selection_only=False, data_root=root,
            )
            with patch.object(runner, "prepare_ratio_tuning", return_value=({}, budget, [], [], paths, {})), \
                    patch.object(runner, "open_recommendation_evidence", return_value=evidence), \
                    patch.object(runner.tuning, "_load_score_manifest", return_value=[completed]), \
                    patch.object(runner.tuning, "execute_panel", side_effect=execute_without_models), \
                    patch.object(runner, "prepare_tuning_environment"), \
                    patch("tests.checks.run_lightning_dev18.configure_cuda_environment"), \
                    record_run_history(root / "history", identity={"mode": "execute-only"}) as history:
                report = runner._run_ratio_tuning_command(arguments, history)
            self.assertEqual(report["planned_runs"], 2)
            self.assertEqual(report["pending_runs_before"], 1)
            self.assertEqual(report["pending_runs"], 1)
            self.assertNotIn("additional_runs", report)
            self.assertEqual(report["status"], "execution_complete")
            self.assertEqual(report["recommendation_status"], "pending")
            receipt = json.loads((root / "execution_complete.json").read_text(encoding="utf-8"))
            saved_history = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
            self.assertEqual(receipt["execution_timing"], measured)
            self.assertEqual(saved_history["execution_timing"], measured)
            self.assertEqual(saved_history["cost_scope"], "hpo_development")

            arguments.execute_only = False
            arguments.remote_cpu, arguments.workers, arguments.no_plots = False, 1, True
            with patch.object(runner, "prepare_ratio_tuning", return_value=({}, budget, [], [], paths, {})), \
                    patch.object(runner, "open_recommendation_evidence", return_value=evidence), \
                    patch.object(runner.tuning, "_load_score_manifest", return_value=[completed]), \
                    patch.object(runner.tuning, "execute_panel", side_effect=execute_without_models), \
                    patch.object(runner, "prepare_tuning_environment"), \
                    patch.object(runner.tuning, "_require_cuda_or_remote"), \
                    patch.object(runner, "finish_ratio_tuning", return_value={"ledger_rows": 2}) as finish, \
                    record_run_history(root / "history", identity={"mode": "run"}) as resumed:
                result = runner._run_ratio_tuning_command(arguments, resumed)
            finish.assert_called_once()
            self.assertEqual(result["completion_scope"], "selection")
            self.assertEqual(resumed["stages"]["execute"]["status"], "complete")
            self.assertEqual(resumed["stages"]["score_select_export"]["status"], "complete")
            self.assertTrue((root / "selection_complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
