"""모델을 실행하지 않고 조건별 배치의 실행 범위를 검사한다."""

import unittest
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from tests.ghl_main import run_dev18_tuning as tuning
from tests.ghl_main.run_dev18_tuning import _validate_primary_manifest_rows


class FullPrefixExecutionContractTests(unittest.TestCase):
    def test_pca_blas_recovery_preserves_attempts_and_allows_only_one_more(self):
        spec = {"model": "PCA_LEGACY", "tier": "t1", "config_id": "cb3ca230f385a",
                "ratio": 100, "seed": 0, "hyperparameters": {}}
        panel_key = (spec["model"], spec["config_id"], 100, 0)
        panel = {"primary_score_variants": [""], "diagnostic_score_variants": []}
        entry = {"series": "13", "row_count": 1000, "feature_count": 2, "order": 1}
        budget = {"budget_id": "b2f61f74691c6", "failure_rules": {"maximum_total_attempts": 3}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            histories = root / "experiments/01_ghl_main/logs/run_history/model_attempts"
            histories.mkdir(parents=True)
            preserved = {}
            for attempt in range(3):
                path = histories / f"prior_{attempt}.json"
                path.write_text(json.dumps({
                    "identity": {"kind": "model_attempt", "budget_id": budget["budget_id"],
                                 "series": "13", **{key: spec[key] for key in
                                 ("model", "config_id", "ratio", "seed")}, "attempt": attempt},
                    "status": "running",
                }), encoding="utf-8")
                preserved[path] = path.read_bytes()
            stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
            for name, value in (
                ("_require_cuda_or_remote", None), ("prepare_tuning", {"environment": {}}),
                ("_specs_for_budget", ([spec], {panel_key: panel})), ("_load_score_manifest", []),
                ("load_input_manifest_role", ({"datasets": {"DEV18": {"files": [entry]}}}, "sha")),
                ("_git_head", "repaired"), ("_compatible_resume_source", None), ("_completed_run", False),
                ("_load_completion_receipt", None),
            ):
                stack.enter_context(patch.object(tuning, name, return_value=value))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory",
                                      return_value=root / "scores"))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.load_registered_inputs",
                                      return_value={"family": "synthetic", "test_sessions": ()}))
            execute = stack.enter_context(patch.object(tuning, "_run_one_spec", side_effect=KeyboardInterrupt()))
            with self.assertRaises(KeyboardInterrupt):
                tuning.execute_panel(budget=budget, device="cpu", manifest_path=root / "manifest.csv")
            execute.assert_called_once()
            self.assertEqual(execute.call_args.kwargs["retry_count"], 3)
            added = set(histories.glob("*.json")) - preserved.keys()
            self.assertEqual(len(added), 1)
            recovered = json.loads(added.pop().read_text(encoding="utf-8"))
            self.assertEqual(recovered["identity"]["execution_recovery"]["maximum_total_attempts"], 4)
            with self.assertRaisesRegex(RuntimeError, "최대 시도 횟수"):
                tuning.execute_panel(budget=budget, device="cpu", manifest_path=root / "manifest.csv")
            execute.assert_called_once()
            self.assertEqual(budget["failure_rules"]["maximum_total_attempts"], 3)
            for path, content in preserved.items():
                self.assertEqual(path.read_bytes(), content)

    def test_pca_blas_recovery_does_not_extend_other_trials_or_repeat(self):
        key = ("13", "PCA_LEGACY", "cb3ca230f385a", "100", "0")
        arguments = {"budget_id": "b2f61f74691c6", "trial_key": key,
                     "attempts_used": 3, "maximum_attempts": 3}
        self.assertIsNotNone(tuning._pca_blas_recovery(**arguments))
        for changed in ({"budget_id": "other"}, {"attempts_used": 2}, {"attempts_used": 4},
                        {"trial_key": ("12", *key[1:])}, {"trial_key": (*key[:2], "other", *key[3:])},
                        {"trial_key": (key[0], "GDN", *key[2:])}, {"maximum_attempts": 4}):
            with self.subTest(changed=changed):
                self.assertIsNone(tuning._pca_blas_recovery(**{**arguments, **changed}))

    def test_pca_cpu_time_survives_success_and_interrupt_without_scaling_wall_time(self):
        spec = {"model": "PCA_LEGACY", "config_id": "c123456789abc", "ratio": 100, "seed": 0,
                "common_recipe": {"training_split": "full_prefix_v2"}}
        result = {"timing": {}, "test_outputs": ({"pca_parallelism": {"fit_blas_threads_requested": 8}},)}
        for error, stop_error in ((None, None), (KeyboardInterrupt(), None), (None, KeyboardInterrupt())):
            with self.subTest(model_error=error, stop_error=stop_error), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory).resolve()
                snapshot = root / "run_snapshot.json"
                snapshot.write_text("{}", encoding="utf-8")
                stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
                stack.enter_context(patch.object(tuning, "_write_run_snapshot", return_value=snapshot))
                stack.enter_context(patch.object(tuning, "_start_run_resources", return_value={}))
                stack.enter_context(patch.object(tuning, "_finish_run_resources", return_value={"gpu_allocated_peak_bytes": None}))
                stack.enter_context(patch.object(tuning, "_finish_failed_run_resources", return_value={}))
                stack.enter_context(patch.object(tuning, "start_process_memory_sampling", return_value=Mock(
                    stop=Mock(return_value={}, side_effect=stop_error),
                )))
                stack.enter_context(patch.object(tuning.time, "process_time", side_effect=[100.0, 127.0]))
                stack.enter_context(patch.object(tuning.tracemalloc, "start"))
                stack.enter_context(patch.object(tuning.tracemalloc, "stop"))
                stack.enter_context(patch.object(tuning.tracemalloc, "get_traced_memory", return_value=(0, 0)))
                stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory", return_value=root))
                stack.enter_context(patch("src.common.run_registered_model.execute_registered_model", return_value=result, side_effect=error))
                save_result = stack.enter_context(patch.object(tuning, "_save_run_result", return_value=[]))

                def run():
                    with tuning.record_run_history(root / "history", identity={"budget_id": "sealed"}) as history:
                        tuning._run_one_spec(spec, {}, {"test_sessions": ()}, series=1, device="cpu",
                                             environment={}, input_manifest_path=root / "manifest.yaml",
                                             retry_count=0, budget_id="sealed", history=history)
                    return history

                if error is None and stop_error is None:
                    history = run()
                    self.assertEqual(save_result.call_args.kwargs["resource_usage"]["pca_compute"]["cpu_core_seconds"], 27.0)
                else:
                    with self.assertRaises(KeyboardInterrupt):
                        run()
                    history = json.loads(next((root / "history").glob("*.json")).read_text(encoding="utf-8"))
                    save_result.assert_not_called()
                compute = history["resource_usage"]["pca_compute"]
                self.assertEqual(compute["cpu_core_seconds"], 27.0)
                self.assertEqual(compute["wall_seconds"], history["model_execution_seconds"])
                self.assertIsNone(compute["single_thread_wall_seconds"])

    def test_training_storage_failure_stays_blocked_after_restart_without_blocking_model_retries(self):
        spec = {"model": "PaAno", "tier": "t2", "config_id": "c123456789abc",
                "ratio": 40, "seed": 0, "hyperparameters": {}}
        panel_key = ("PaAno", spec["config_id"], 40, 0)
        panel = {"primary_score_variants": [""], "diagnostic_score_variants": []}
        entry = {"series": "01", "row_count": 1000, "feature_count": 2, "order": 1}
        budget = {"budget_id": "sealed", "failure_rules": {"maximum_total_attempts": 3}}
        trained = {"checkpoint": {"selected": 2}, "training_log": {"selected_iteration": 2},
                   "seed_state": {"seed": 0}, "timing": {"training_seconds": 1.25}}
        for failure in ("storage_error", "storage_interrupt", "storage_history_error", "training", "inference"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory).resolve()
                executions = []
                original_write = Path.write_text
                original_save_history = tuning.save_run_history

                def write_snapshot(directory, *args):
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "run_snapshot.json"
                    path.write_text('{"project_commit":"sealed"}', encoding="utf-8")
                    return path

                def fail_training_write(path, *args, **kwargs):
                    if path.name == ".training_log.json.tmp" and failure.startswith("storage"):
                        if failure == "storage_interrupt":
                            raise KeyboardInterrupt()
                        raise OSError("training evidence write failed")
                    return original_write(path, *args, **kwargs)

                def execute_model(*args, on_training_complete, **kwargs):
                    executions.append("executed")
                    if failure == "training":
                        raise RuntimeError("training failed")
                    on_training_complete(trained)
                    raise RuntimeError("inference failed")

                def fail_history_close(history):
                    if failure == "storage_history_error" and history["finished_at"] is not None:
                        raise OSError("closing history write failed")
                    return original_save_history(history)

                stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
                for name, value in (
                    ("_require_cuda_or_remote", None), ("prepare_tuning", {"environment": {}}),
                    ("_specs_for_budget", ([spec], {panel_key: panel})),
                    ("load_input_manifest_role", ({"datasets": {"DEV18": {"files": [entry]}}}, "sha")),
                    ("_git_head", "sealed"), ("_compatible_resume_source", None),
                    ("_completed_run", False), ("_require_same_worktree", None),
                ):
                    stack.enter_context(patch.object(tuning, name, return_value=value))
                stack.enter_context(patch.object(tuning, "_write_run_snapshot", side_effect=write_snapshot))
                stack.enter_context(patch.object(Path, "write_text", new=fail_training_write))
                stack.enter_context(patch("tests.ghl_main.record_run_history.save_run_history",
                                          side_effect=fail_history_close))
                stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory",
                                          return_value=root / "scores"))
                stack.enter_context(patch("tests.ghl_main.run_registered_models.load_registered_inputs",
                                          return_value={"family": "synthetic", "test_sessions": ()}))
                stack.enter_context(patch("src.common.run_registered_model.execute_registered_model",
                                          side_effect=execute_model))
                first_error = (KeyboardInterrupt if failure == "storage_interrupt" else
                               OSError if failure == "storage_history_error" else
                               tuning.RunResultPersistenceError if failure == "storage_error" else RuntimeError)
                arguments = {"budget": budget, "device": "cpu", "manifest_path": root / "manifest.csv"}
                with self.assertRaises(first_error):
                    tuning.execute_panel(**arguments)
                history_directory = root / "experiments/01_ghl_main/logs/run_history/model_attempts"
                before_restart = {path: path.read_bytes() for path in history_directory.glob("*.json")}
                if failure.startswith("storage"):
                    self.assertEqual(executions, ["executed"])
                    saved = json.loads(next(iter(before_restart.values())))
                    self.assertFalse(saved.get("model_execution_complete", False))
                    self.assertEqual(saved["training_complete"]["status"], "saving")
                    if failure == "storage_history_error":
                        self.assertEqual(saved["status"], "running")
                        self.assertEqual(saved["stages"]["save_training"]["status"], "failed")
                        self.assertNotIn("error_type", saved)
                    checkpoint = root / saved["training_complete"]["files"]["checkpoint"]["file"]
                    checkpoint_bytes = checkpoint.read_bytes()
                    with self.assertRaisesRegex(tuning.RunResultPersistenceError, "재학습"):
                        tuning.execute_panel(**arguments)
                    self.assertEqual(executions, ["executed"])
                    self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
                    self.assertEqual({path: path.read_bytes() for path in history_directory.glob("*.json")}, before_restart)
                else:
                    self.assertEqual(len(executions), 3)
                    self.assertEqual(len(before_restart), 3)

    def test_training_write_failure_keeps_each_prior_file_in_saved_history(self):
        trained = {"checkpoint": {"selected": 2}, "scaler_state": {"scale": [1]},
                   "training_log": {"selected_iteration": 2, "loss_history": [0.8, 0.4]},
                   "seed_state": {"seed": 0}, "timing": {"training_seconds": 1.25}}
        for filename, expected_files, error, expected_exception, status in (
            (".scaler_state.json.tmp", {"checkpoint"},
             OSError("disk full"), tuning.RunResultPersistenceError, "failed"),
            (".training_log.json.tmp", {"checkpoint", "scaler_state"},
             KeyboardInterrupt(), KeyboardInterrupt, "interrupted"),
            (".timing.json.tmp", {"checkpoint", "scaler_state", "training_log"},
             OSError("disk full"), tuning.RunResultPersistenceError, "failed"),
        ):
            with self.subTest(file=filename), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory).resolve()
                stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
                stack.enter_context(patch.object(tuning, "_require_same_worktree"))
                original_write = Path.write_text
                persisted_before_failure = []

                def fail_training_write(path, *args, **kwargs):
                    if path.name == filename:
                        persisted_before_failure.append(json.loads(
                            Path(history["history_file"]).read_text(encoding="utf-8")))
                        raise error
                    return original_write(path, *args, **kwargs)

                stack.enter_context(patch.object(Path, "write_text", new=fail_training_write))
                with self.assertRaises(expected_exception):
                    with tuning.record_run_history(root / "history", identity={"budget_id": "sealed"}) as history:
                        history["run_snapshot"] = {"project_commit": "sealed"}
                        tuning._save_training_completion(root / "scores", trained, history)
                before_failure = persisted_before_failure[0]["training_complete"]
                self.assertEqual(before_failure["status"], "saving")
                saved_files = before_failure.get("files", {})
                self.assertEqual(set(saved_files), expected_files)
                for reference in saved_files.values():
                    artifact = root / reference["file"]
                    self.assertEqual(tuning.file_sha256(artifact), reference["sha256"])
                    self.assertEqual(artifact.stat().st_size, reference["bytes"])
                saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], status)
                self.assertEqual(saved["training_complete"]["status"], "saving")
                self.assertEqual(saved["training_complete"]["files"], saved_files)
                if "training_log" in saved_files:
                    self.assertEqual(json.loads((root / saved_files["training_log"]["file"]).read_text(encoding="utf-8")),
                                     trained["training_log"])

    def test_result_write_failure_keeps_fitted_checkpoint_in_saved_history(self):
        result = {"checkpoint": {"components": [[1.0, 0.0]]}, "scaler_state": {"scale": [1]},
                  "seed_state": {"seed": 0}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            snapshot = root / "run_snapshot.json"
            snapshot.write_text('{"project_commit":"sealed"}', encoding="utf-8")
            stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
            stack.enter_context(patch.object(tuning, "_require_same_worktree"))
            original_write = Path.write_text
            persisted_before_failure = []

            def fail_scaler_write(path, *args, **kwargs):
                if path.name == ".scaler_state.json.tmp":
                    persisted_before_failure.append(json.loads(
                        Path(history["history_file"]).read_text(encoding="utf-8")))
                    raise OSError("scaler write failed")
                return original_write(path, *args, **kwargs)

            stack.enter_context(patch.object(Path, "write_text", new=fail_scaler_write))
            with self.assertRaisesRegex(OSError, "scaler write failed"):
                with tuning.record_run_history(root / "history", identity={"budget_id": "sealed"}) as history:
                    tuning._save_run_result(result, {"model": "PCA_LEGACY"}, {}, {}, series=1,
                        snapshot_path=snapshot, peak_memory_mb=None, input_manifest_path=root / "manifest.yaml",
                        retry_count=0, budget_id="sealed", history=history)
            files = persisted_before_failure[0].get("result_files", {})
            self.assertEqual(set(files), {"checkpoint"})
            checkpoint = root / files["checkpoint"]["file"]
            self.assertEqual(tuning.file_sha256(checkpoint), files["checkpoint"]["sha256"])
            self.assertEqual(checkpoint.stat().st_size, files["checkpoint"]["bytes"])
            saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["result_files"], files)
            self.assertNotIn("training_complete", saved)

    def test_inference_failure_preserves_training_files_for_each_attempt(self):
        spec = {"model": "PaAno", "config_id": "c123456789abc", "ratio": 40, "seed": 0}
        trained = {"checkpoint": {"selected": 2}, "scaler_state": {"scale": [1]},
                   "training_log": {"selected_iteration": 2, "loss_history": [0.8, 0.4]},
                   "seed_state": {"seed": 0}, "timing": {"training_seconds": 1.25}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            snapshot = root / "run_snapshot.json"
            snapshot.write_text('{"project_commit":"sealed"}', encoding="utf-8")
            stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
            stack.enter_context(patch.object(tuning, "_write_run_snapshot", return_value=snapshot))
            stack.enter_context(patch.object(tuning, "_require_same_worktree"))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory", return_value=root))

            def fail_inference(*args, on_training_complete, **kwargs):
                on_training_complete(trained)
                raise RuntimeError("inference failed after fit")

            stack.enter_context(patch("src.common.run_registered_model.execute_registered_model",
                                      side_effect=fail_inference))
            references = []
            for attempt in range(2):
                with self.assertRaisesRegex(RuntimeError, "inference failed after fit"):
                    with tuning.record_run_history(root / "history", identity={"budget_id": "sealed"}) as history:
                        tuning._run_one_spec(spec, {}, {"test_sessions": ()}, series=1, device="cpu",
                            environment={}, input_manifest_path=root / "manifest.yaml", retry_count=attempt,
                            budget_id="sealed", history=history)
                saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
                self.assertFalse(saved["model_execution_complete"])
                self.assertEqual(saved["training_complete"]["status"], "complete")
                self.assertEqual(saved["training_complete"]["timing"], trained["timing"])
                self.assertEqual(saved["training_complete"]["seed_state"], trained["seed_state"])
                files = saved["training_complete"]["files"]
                self.assertEqual(set(files), {"checkpoint", "scaler_state", "training_log", "timing"})
                for reference in files.values():
                    path = root / reference["file"]
                    self.assertEqual(tuning.file_sha256(path), reference["sha256"])
                    self.assertEqual(path.stat().st_size, reference["bytes"])
                self.assertEqual(json.loads((root / files["training_log"]["file"]).read_text(encoding="utf-8")),
                                 trained["training_log"])
                references.append(files)
            self.assertNotEqual(references[0]["checkpoint"]["file"], references[1]["checkpoint"]["file"])
            self.assertTrue((root / references[0]["checkpoint"]["file"]).is_file())

    def test_completed_panel_records_check_time_without_loading_or_running_model(self):
        spec = {"model": "PCA_LEGACY", "config_id": "c123456789abc", "ratio": 40,
                "seed": 0, "hyperparameters": {}}
        panel_key = (spec["model"], spec["config_id"], spec["ratio"], spec["seed"])
        entry = {"series": "01", "row_count": 1000, "feature_count": 2, "order": 1}
        execution_timing = {}
        with ExitStack() as stack:
            for name, value in (
                ("_require_cuda_or_remote", None), ("prepare_tuning", {"environment": {}}),
                ("_specs_for_budget", ([spec], {panel_key: {}})), ("_load_score_manifest", []),
                ("load_input_manifest_role", ({"datasets": {"DEV18": {"files": [entry]}}}, "sha")),
                ("_git_head", "sealed"), ("_compatible_resume_source", None), ("_completed_run", True),
            ):
                stack.enter_context(patch.object(tuning, name, return_value=value))
            stack.enter_context(patch.object(tuning.time, "perf_counter", side_effect=[10.0, 12.0]))
            load_inputs = stack.enter_context(patch(
                "tests.ghl_main.run_registered_models.load_registered_inputs",
                side_effect=AssertionError("completed run loaded input"),
            ))
            run_model = stack.enter_context(patch.object(
                tuning, "_run_one_spec", side_effect=AssertionError("completed run reran model"),
            ))
            self.assertEqual(tuning.execute_panel(
                budget={"budget_id": "sealed"}, device="cpu", execution_timing=execution_timing,
            ), [])
            load_inputs.assert_not_called()
            run_model.assert_not_called()
        self.assertEqual(execution_timing, {"completion_check_seconds": 2.0, "model_attempt_seconds": 0.0})

    def test_single_model_storage_failure_preserves_the_attempt_without_retraining(self):
        spec = {"model": "PCA_LEGACY", "tier": "t1", "config_id": "c123456789abc",
                "ratio": 40, "seed": 0, "hyperparameters": {}}
        panel_key = (spec["model"], spec["config_id"], 40, 0)
        panel = {"primary_score_variants": [""], "diagnostic_score_variants": []}
        entry = {"series": "01", "row_count": 1000, "feature_count": 2, "order": 1}
        budget = {"budget_id": "sealed", "failure_rules": {"maximum_total_attempts": 3}}
        execution_timing, executions, saves = {}, [], []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            manifest_path = root / "manifest.csv"

            def write_snapshot(directory, *args):
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "run_snapshot.json"
                path.write_text(json.dumps({"input_sha": "original"}), encoding="utf-8")
                return path

            def execute_model(*args, **kwargs):
                executions.append("executed")
                return {"timing": {"training_seconds": 1.25, "test_inference_seconds": 0.5,
                                   "accelerator": "cpu"}}

            def save_result(*args, **kwargs):
                saves.append("saved")
                (root / "partial_checkpoint").write_text(str(len(saves)), encoding="utf-8")
                raise OSError("storage interrupted after writing output")

            stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
            for name, value in (
                ("_require_cuda_or_remote", None), ("prepare_tuning", {"environment": {}}),
                ("_specs_for_budget", ([spec], {panel_key: panel})), ("_load_score_manifest", []),
                ("load_input_manifest_role", ({"datasets": {"DEV18": {"files": [entry]}}}, "sha")),
                ("_git_head", "sealed"), ("_compatible_resume_source", None), ("_completed_run", False),
            ):
                stack.enter_context(patch.object(tuning, name, return_value=value))
            stack.enter_context(patch.object(tuning, "_write_run_snapshot", side_effect=write_snapshot))
            stack.enter_context(patch.object(tuning, "_save_run_result", side_effect=save_result))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory",
                                      return_value=root / "scores"))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.load_registered_inputs",
                                      return_value={"family": "synthetic", "test_sessions": ()}))
            stack.enter_context(patch("src.common.run_registered_model.execute_registered_model",
                                      side_effect=execute_model))
            with self.assertRaisesRegex(RuntimeError, "결과 저장") as raised:
                tuning.execute_panel(budget=budget, device="cpu", manifest_path=manifest_path,
                                     execution_timing=execution_timing)
            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(executions, ["executed"])
            self.assertEqual((root / "partial_checkpoint").read_text(encoding="utf-8"), "1")
            self.assertFalse(manifest_path.exists())
            (root / "scores/series_01/run_snapshot.json").write_text(
                json.dumps({"input_sha": "overwritten"}), encoding="utf-8",
            )
            history_directory = root / "experiments/01_ghl_main/logs/run_history/model_attempts"
            histories = [json.loads(path.read_text(encoding="utf-8"))
                         for path in history_directory.glob("*.json")]
            self.assertEqual(len(histories), 1)
            history = histories[0]
            self.assertEqual(history["status"], "failed")
            self.assertTrue(history["model_execution_complete"])
            self.assertGreaterEqual(history["model_execution_seconds"], 0.0)
            self.assertEqual(history["run_snapshot_file"], "scores/series_01/run_snapshot.json")
            self.assertEqual(history["run_snapshot"], {"input_sha": "original"})
            self.assertEqual(history["model_timing"],
                             {"training_seconds": 1.25, "test_inference_seconds": 0.5, "accelerator": "cpu"})
            self.assertEqual(history["stages"]["save_result"]["status"], "failed")
            self.assertEqual(history["resource_usage"]["cpu_rss_peak_kind"], "sampled_lower_bound")
            self.assertTrue(history["resource_usage"]["cpu_rss_sampling_thread_stopped"])
            self.assertAlmostEqual(execution_timing["model_attempt_seconds"], history["elapsed_seconds"])
            self.assertEqual(tuning.load_attempt_counts(history_directory, "sealed"),
                             {("01", "PCA_LEGACY", "c123456789abc", "40", "0"): 1})
            snapshot_path = root / "scores/series_01/run_snapshot.json"
            preserved_snapshot = snapshot_path.read_bytes()
            preserved_history = Path(history["history_file"]).read_bytes()
            with self.assertRaisesRegex(tuning.RunResultPersistenceError, "재학습"):
                tuning.execute_panel(budget=budget, device="cpu", manifest_path=manifest_path)
            self.assertEqual(executions, ["executed"])
            self.assertEqual(saves, ["saved"])
            self.assertEqual((root / "partial_checkpoint").read_text(encoding="utf-8"), "1")
            self.assertEqual(snapshot_path.read_bytes(), preserved_snapshot)
            self.assertEqual(Path(history["history_file"]).read_bytes(), preserved_history)
            self.assertEqual(len(list(history_directory.glob("*.json"))), 1)

    def test_receipt_write_failure_recovers_saved_rows_on_the_next_command(self):
        spec = {"model": "MWVAR", "tier": "t1", "config_id": "c123456789abc",
                "ratio": 100, "seed": 0, "hyperparameters": {}, "common_recipe": {}}
        panel_key = (spec["model"], spec["config_id"], 100, 0)
        panel = {"primary_score_variants": [""], "diagnostic_score_variants": []}
        entry = {"series": "01", "row_count": 1000, "feature_count": 2, "order": 1}
        budget = {"budget_id": "sealed", "failure_rules": {"maximum_total_attempts": 3}}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            output = root / "scores"
            evidence = output / "series_01"
            evidence.mkdir(parents=True)
            snapshot_path = evidence / "run_snapshot.json"
            snapshot_path.write_text('{"project_commit":"sealed"}', encoding="utf-8")
            metadata_path = output / "DEV18__01__MWVAR__t1__r100__s0__raw__trainnorm.meta.json"
            metadata_path.write_text("{}", encoding="utf-8")
            score_path = output / "DEV18__01__MWVAR__t1__r100__s0__raw__trainnorm.npy"
            score_path.write_bytes(b"saved score")
            manifest_path = root / "manifest.csv"
            stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
            for name, value in (
                ("_require_cuda_or_remote", None), ("prepare_tuning", {"environment": {}}),
                ("_specs_for_budget", ([spec], {panel_key: panel})),
                ("load_input_manifest_role", ({"datasets": {"DEV18": {"files": [entry]}}}, "sha")),
                ("_git_head", "sealed"), ("_compatible_resume_source", None),
                ("_write_run_snapshot", snapshot_path), ("_require_same_worktree", None),
                ("_record_seed_state", None), ("_save_training_files", (0, {})),
                ("_validate_bound_run_files", None),
            ):
                stack.enter_context(patch.object(tuning, name, return_value=value))
            stack.enter_context(patch("src.common.execution_evidence.build_execution_evidence", return_value={}))
            stack.enter_context(patch("src.common.save_model_artifacts.save_execution_result", return_value={
                "metadata_path": str(metadata_path), "score_paths": [str(score_path)],
            }))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory", return_value=output))
            stack.enter_context(patch("tests.ghl_main.check_registered_outputs.check_registered_output"))
            load_inputs = stack.enter_context(patch(
                "tests.ghl_main.run_registered_models.load_registered_inputs",
                return_value={"family": "synthetic", "test_sessions": ()},
            ))
            execute = stack.enter_context(patch("src.common.run_registered_model.execute_registered_model",
                return_value={"timing": {}, "split": None}))
            with patch.object(tuning, "_write_completion_receipt", side_effect=OSError("receipt write failed")):
                with self.assertRaises(tuning.RunResultPersistenceError):
                    tuning.execute_panel(budget=budget, device="cpu", manifest_path=manifest_path)
            self.assertFalse(manifest_path.exists())
            self.assertFalse((evidence / "completion.json").exists())
            preserved = {path: path.read_bytes() for path in (score_path, metadata_path, snapshot_path)}
            histories = list((root / "experiments/01_ghl_main/logs/run_history/model_attempts").glob("*.json"))
            self.assertEqual(len(histories), 1)
            history_content = histories[0].read_bytes()
            saved_history = json.loads(history_content)
            saved_history.pop("result")
            histories[0].write_text(json.dumps(saved_history), encoding="utf-8")
            history_content = histories[0].read_bytes()
            score_path.write_bytes(b"changed after storage failure")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                tuning.execute_panel(budget=budget, device="cpu", manifest_path=manifest_path)
            self.assertFalse((evidence / "completion.json").exists())
            self.assertFalse(manifest_path.exists())
            self.assertEqual(execute.call_count, 1)
            score_path.write_bytes(preserved[score_path])
            completed_rows = tuning.execute_panel(budget=budget, device="cpu", manifest_path=manifest_path)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(load_inputs.call_count, 1)
            self.assertEqual(len(completed_rows), 1)
            self.assertEqual(completed_rows[0]["status"], "complete")
            self.assertEqual(json.loads((evidence / "completion.json").read_text(encoding="utf-8")), completed_rows)
            self.assertEqual(tuning._load_score_manifest(manifest_path)[0]["status"], "complete")
            self.assertEqual(histories[0].read_bytes(), history_content)
            for path, content in preserved.items():
                self.assertEqual(path.read_bytes(), content)

    def test_saved_outputs_from_another_attempt_cannot_recover_missing_receipt(self):
        spec = {"model": "MWVAR", "tier": "t1", "config_id": "c123456789abc",
                "ratio": 100, "seed": 0}
        panel = {"primary_score_variants": [""], "diagnostic_score_variants": []}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            raw = root / "DEV18__01__MWVAR__t1__r100__s0__raw__trainnorm.npy"
            raw.write_bytes(b"preserve")
            metadata = raw.with_suffix(".meta.json")
            metadata.write_text(json.dumps({
                "execution_attempt": {"run_id": "older", "history_file": "older.json"},
            }), encoding="utf-8")
            history = root / "attempt.json"
            history.write_text(json.dumps({"run_id": "current"}), encoding="utf-8")
            stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory", return_value=root))
            with self.assertRaisesRegex(ValueError, "시도"):
                tuning._recover_saved_run_rows(
                    spec=spec, panel_row=panel, series=1, family="synthetic", budget_id="sealed",
                    computed={"attempt": 0, "history_file": str(history)},
                )
            self.assertEqual(raw.read_bytes(), b"preserve")

    def test_memory_sampler_start_interruption_closes_python_measurement(self):
        spec = {"model": "GDN", "config_id": "c123456789abc", "ratio": 40, "seed": 0}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            snapshot = root / "run_snapshot.json"
            snapshot.write_text("{}", encoding="utf-8")
            stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
            stack.enter_context(patch.object(tuning, "_write_run_snapshot", return_value=snapshot))
            stack.enter_context(patch.object(tuning, "_start_run_resources", return_value={}))
            stack.enter_context(patch.object(tuning, "_finish_failed_run_resources", return_value={}))
            stack.enter_context(patch.object(tuning, "start_process_memory_sampling", side_effect=KeyboardInterrupt))
            stack.enter_context(patch.object(tuning.tracemalloc, "start"))
            stop = stack.enter_context(patch.object(tuning.tracemalloc, "stop"))
            execute = stack.enter_context(patch("src.common.run_registered_model.execute_registered_model"))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory", return_value=root))
            with self.assertRaises(KeyboardInterrupt):
                with tuning.record_run_history(root / "history", identity={"budget_id": "sealed"}) as history:
                    tuning._run_one_spec(
                        spec, {}, {}, series=1, device="cpu", environment={}, input_manifest_path=root / "manifest.yaml",
                        retry_count=0, budget_id="sealed", history=history,
                    )
            stop.assert_called_once()
            execute.assert_not_called()
            saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "interrupted")
            self.assertFalse(saved["model_execution_complete"])
            self.assertEqual(saved["resource_usage"]["cpu_rss_sampling_status"], "unavailable")

    def test_model_failure_keeps_available_resources_without_masking_the_error(self):
        spec = {"model": "GDN", "config_id": "c123456789abc", "ratio": 40, "seed": 0}
        memory = {"rss_bytes": 100, "process_lifetime_peak_bytes": 500, "reason": None}
        for measurement_fails in (False, True):
            with self.subTest(measurement_fails=measurement_fails), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory).resolve()
                snapshot_path = root / "run_snapshot.json"
                snapshot_path.write_text("{}", encoding="utf-8")
                stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
                stack.enter_context(patch.object(tuning, "_write_run_snapshot", return_value=snapshot_path))
                stack.enter_context(patch.object(tuning, "_read_process_memory", return_value=memory))
                stack.enter_context(patch("tests.ghl_main.run_registered_models.build_output_directory", return_value=root))
                for name, value in (("is_available", True), ("reset_peak_memory_stats", None),
                                    ("memory_allocated", 1000), ("memory_reserved", 2000),
                                    ("max_memory_reserved", 4000)):
                    stack.enter_context(patch("torch.cuda." + name, return_value=value))
                stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=3000,
                    side_effect=RuntimeError("device unavailable") if measurement_fails else None))
                error = RuntimeError("original model failure")
                stack.enter_context(patch("src.common.run_registered_model.execute_registered_model", side_effect=error))
                with self.assertRaises(RuntimeError) as raised:
                    with tuning.record_run_history(root / "history", identity={"budget_id": "sealed"}) as history:
                        tuning._run_one_spec(spec, {}, {"test_sessions": ()}, series=1, device="cuda",
                            environment={}, input_manifest_path=root / "manifest.yaml", retry_count=0,
                            budget_id="sealed", history=history)
                self.assertIs(raised.exception, error)
                saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
                self.assertFalse(saved["model_execution_complete"])
                usage = saved["resource_usage"]
                self.assertEqual(usage["cpu_rss_start_bytes"], 100)
                self.assertEqual(usage["cpu_rss_end_bytes"], 100)
                self.assertEqual(usage["gpu_allocated_start_bytes"], 1000)
                self.assertEqual(usage["gpu_reserved_peak_bytes"], 4000)
                self.assertGreater(usage["python_tracemalloc_peak_bytes"], 0)
                self.assertIsNone(usage["actual_backend"])
                self.assertEqual(usage["actual_backend_reason"], "execution_failed_before_result")
                self.assertEqual(usage["gpu_measurement_scope"], "requested_device_allocator_absolute_peak_since_reset")
                self.assertEqual(usage["gpu_measurement_reason"], "actual_backend_unknown_after_failure")
                if measurement_fails:
                    self.assertIsNone(usage["gpu_allocated_peak_bytes"])
                    self.assertIn("device unavailable", usage["measurement_errors"]["gpu_allocated_peak_bytes"])
                else:
                    self.assertEqual(usage["gpu_allocated_peak_bytes"], 3000)
                    self.assertEqual(usage["measurement_errors"], {})
                self.assertFalse(tuning.tracemalloc.is_tracing())

    def test_short_series_does_not_require_long_patch_execution(self):
        panel = {"model": "PaAno", "config_id": "c123456789abc",
                 "physical_ratio": 60, "seed": 0,
                 "primary_score_variants": [""], "series_ids": ["02"]}
        budget = {"budget_id": "b123456789abc", "execution_panel": [panel]}
        row = {"series": "02", "model": panel["model"],
               "config_id": panel["config_id"], "physical_ratio": 60,
               "seed": 0, "score_variant": "", "primary_score": "true",
               "status": "complete", "budget_id": budget["budget_id"]}
        self.assertEqual(_validate_primary_manifest_rows([row], budget, ["01", "02"]), [row])
        with self.assertRaises(ValueError):
            _validate_primary_manifest_rows([], budget, ["01", "02"])


if __name__ == "__main__":
    unittest.main()
