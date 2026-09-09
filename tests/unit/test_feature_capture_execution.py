"""원격에서 모델을 모킹해 feature 게이트와 저장 실패 재개를 확인한다."""

import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy

from tests.ghl_main import run_dev18_tuning as tuning


class FeatureCaptureExecutionTests(unittest.TestCase):
    @contextmanager
    def panel(self, store, *, model="PCA_LEGACY", revision=None):
        entries = [{"series": series, "sha256": series * 32, "row_count": 100,
                    "feature_count": 2, "order": index}
                   for index, series in enumerate(("01", "02"))]
        specs = [{"model": model, "tier": "t2" if model == "PaAno" else "t1", "config_id": f"c{index:012d}",
                  "ratio": 5, "seed": 0, "hyperparameters": {}, "target_use": "fit_full_prefix",
                  "common_recipe": {"training_split": "full_prefix_v2", "methodology_revision": revision}}
                 for index in range(2)]
        panels = [{"model": spec["model"], "config_id": spec["config_id"], "physical_ratio": 5,
                   "seed": 0, "primary_score_variants": [""], "diagnostic_score_variants": [],
                   "series_ids": ["02"]} for spec in specs]
        budget = {"experiment_mode": "full_prefix_v2", "budget_id": "sealed",
                  "execution_panel": panels, "failure_rules": {"maximum_total_attempts": 3}}
        inputs = {"family": "synthetic", "normal_training": numpy.arange(40).reshape(20, 2),
                  "test_sessions": (numpy.ones((10, 2)),)}
        events = []

        def execute(spec, panel, model_inputs, *, series, **kwargs):
            self.assertIs(model_inputs, inputs)
            events.append(("model", str(series)))
            return [{"series": f"{series:02d}", "family": "synthetic", "tier": spec["tier"],
                     "model": spec["model"], "config_id": spec["config_id"],
                     "physical_ratio": 5, "seed": 0, "score_variant": "", "primary_score": "true",
                     "status": "complete", "status_reason": "", "score_file": "saved.npy",
                     "score_sha256": "a" * 64, "metadata_file": "saved.json",
                     "metadata_sha256": "b" * 64, "budget_id": "sealed", "retry_count": 0}]

        def completed(rows, *, spec, series, **kwargs):
            return any(row["series"] == f"{series:02d}" and row["config_id"] == spec["config_id"]
                       and row["status"] == "complete" for row in rows)

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            stack.enter_context(patch.object(tuning, "REPOSITORY_ROOT", root))
            for name, value in (
                ("_require_cuda_or_remote", None), ("prepare_tuning", {"environment": {}}),
                ("_specs_for_budget", (specs, {(row["model"], row["config_id"], 5, 0): row for row in panels})),
                ("load_input_manifest_role", ({"datasets": {"DEV18": {"files": entries}}}, "sha")),
                ("load_model_registry_with_sha", ({}, "sha")), ("_git_head", "sealed"),
                ("_require_same_worktree", None), ("_load_completion_receipt", []),
            ):
                stack.enter_context(patch.object(tuning, name, return_value=value))
            stack.enter_context(patch.object(tuning, "_completed_run", side_effect=completed))
            stack.enter_context(patch.object(tuning, "_compatible_resume_source", side_effect=AssertionError("legacy resume")))
            run_model = stack.enter_context(patch.object(tuning, "_run_one_spec", side_effect=execute))
            load_model_inputs = stack.enter_context(patch(
                "tests.ghl_main.run_registered_models.load_registered_inputs", return_value=inputs,
            ))
            stack.enter_context(patch(
                "src.data_split.load_dev18_series.load_dev18_registered_inputs",
                side_effect=lambda entry, data_root: events.append(("feature", entry["series"])) or inputs,
            ))
            stack.enter_context(patch(
                "tests.ghl_main.store_recommendation_evidence.open_recommendation_evidence",
                side_effect=lambda *args, **kwargs: nullcontext(store),
            ))
            command = lambda: tuning.execute_panel(
                budget=budget, device="cpu", manifest_path=root / "manifest.csv",
                recommendation_directory=root / "evidence",
            )
            yield command, entries, events, run_model, load_model_inputs, inputs

    def test_all_csv_features_precede_models_and_excluded_csv_has_no_dummy(self):
        store = Mock()
        with self.panel(store) as (command, entries, events, run_model, load_inputs, inputs):
            store.prepare_features.side_effect = lambda loader: (
                [loader(entry) for entry in entries] and {"feature_complete": True}
            )
            before = inputs["normal_training"].copy()
            state = numpy.random.get_state()
            rows = command()
            self.assertEqual(events, [("feature", "01"), ("feature", "02"), ("model", "2"), ("model", "2")])
            self.assertEqual({row["series"] for row in rows}, {"02"})
            self.assertEqual(run_model.call_count, 2)
            load_inputs.assert_called_once()
            numpy.testing.assert_array_equal(inputs["normal_training"], before)
            after = numpy.random.get_state()
            self.assertEqual(state[0], after[0])
            numpy.testing.assert_array_equal(state[1], after[1])
            self.assertEqual(state[2:], after[2:])
            self.assertEqual(store.sync_results.call_count, 2)
            self.assertTrue(all("ledger_rows" not in call.kwargs for call in store.sync_results.call_args_list))

    def test_incomplete_features_block_model_batch(self):
        store = Mock()
        store.prepare_features.return_value = {"feature_complete": False}
        with self.panel(store) as (command, _, _, run_model, load_inputs, _):
            with self.assertRaisesRegex(ValueError, "입력 특징"):
                command()
            run_model.assert_not_called()
            load_inputs.assert_not_called()

    def test_database_failure_preserves_saved_result_and_resume_does_not_retrain(self):
        store = Mock()
        store.prepare_features.return_value = {"feature_complete": True}
        store.sync_results.side_effect = OSError("database commit failed")
        with self.panel(store) as (command, _, _, run_model, load_inputs, _):
            with self.assertRaisesRegex(OSError, "database commit"):
                command()
            self.assertEqual(run_model.call_count, 1)
            first_config = run_model.call_args.args[0]["config_id"]
            store.sync_results.side_effect = None
            rows = command()
            self.assertEqual(len(rows), 2)
            self.assertEqual([call.args[0]["config_id"] for call in run_model.call_args_list].count(first_config), 1)
            self.assertEqual(load_inputs.call_count, 2)

    def test_cpu_backend_does_not_use_unrelated_cuda_peak_as_model_memory(self):
        memory = {"rss_bytes": 100, "process_lifetime_peak_bytes": 500, "reason": None}
        start = {"requested_device": "cuda", "cpu_start": memory,
                 "gpu_allocated_start_bytes": 1000, "gpu_reserved_start_bytes": 2000}
        with patch.object(tuning, "_read_process_memory", return_value=memory), patch(
            "torch.cuda.max_memory_allocated", side_effect=AssertionError("CPU model sampled CUDA peak"),
        ):
            usage = tuning._finish_run_resources(
                start, {"effective_execution": {"backend": "cpu"}}, python_peak_bytes=25,
            )
        self.assertIsNone(usage["gpu_allocated_peak_bytes"])
        self.assertEqual(usage["cpu_peak_scope"], "process_lifetime")
        self.assertEqual(usage["cpu_rss_end_bytes"], 100)
        self.assertEqual(usage["python_tracemalloc_peak_bytes"], 25)

    def test_paper_training_runs_each_config_independently(self):
        store = Mock()
        with self.panel(store, model="PaAno", revision="paper_tuning_v4") as (
            command, entries, events, run_model, load_inputs, inputs,
        ):
            store.prepare_features.return_value = {"feature_complete": True}
            rows = command()
            self.assertEqual(len(rows), 2)
            self.assertEqual(run_model.call_count, 2)
            self.assertEqual(len({call.args[0]["config_id"] for call in run_model.call_args_list}), 2)

    def test_snapshot_uses_canonical_full_prefix_observation_counts(self):
        spec = {"model": "PCA_LEGACY", "target_use": "fit_full_prefix", "ratio": 5,
                "hyperparameters": {}, "common_recipe": {"training_split": "full_prefix_v2"}}
        inputs = {"normal_training": numpy.ones((100, 2)), "input_identity": {"sha256": "a" * 64},
                  "source_ranges": {"normal_training": (0, 100), "test_sessions": ((100, 120),)}}
        with tempfile.TemporaryDirectory() as directory, patch.object(tuning, "_git_head", return_value="sealed"):
            path = tuning._write_run_snapshot(Path(directory), spec, inputs, {})
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["training_boundary"], 100)
        self.assertEqual(snapshot["observed_row"], 5)
        self.assertEqual(snapshot["storage_schema_version"], "full_prefix_storage.v1")
        self.assertFalse({"available_count", "fit_count", "validation_count", "observed_prefix_count"} & snapshot.keys())


if __name__ == "__main__":
    unittest.main()
