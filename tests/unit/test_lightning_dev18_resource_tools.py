"""Lightning Dev18 자원 점검과 이전 실행 초기화 계약."""

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy

from tests.checks.check_dev18_resources import (
    _run_resource_probe,
    _run_model_probe,
    _system_memory_bytes,
    capacity_status,
    select_probe_cases,
    validate_resource_report,
    verify_input_files,
)
from tests.checks.reset_lightning_dev18 import reset_previous_run


class TestDev18ResourceCheck(unittest.TestCase):
    def test_pca_estimate_overflow_uses_isolated_measurement_and_preserves_estimate(self):
        from tests.checks import check_dev18_resources as resources

        specs = [{"model": "PCA_LEGACY", "config_id": name, "ratio": 100, "seed": 0,
                  "target_use": "training_free", "hyperparameters": {"window": 100, "n_components": components}}
                 for name, components in (("fraction", .25), ("all", None))]
        entries = [{"series": "01", "row_count": 300, "training_boundary": 100, "feature_count": 2},
                   {"series": "14", "row_count": 400000, "training_boundary": 28307, "feature_count": 17}]
        case = {"model": "PCA_LEGACY", "config_id": "all", "ratio": 100, "seed": 0, "series": "14"}
        with patch.object(resources, "_system_memory_bytes", return_value=32 * 1024 ** 3), patch.object(
            resources, "_run_resource_probe", return_value={**case, "status": "passed", "ram_peak_bytes": 20 * 1024 ** 3},
        ) as measure:
            estimate = resources._pca_static_check(specs, entries, 80)
            result = resources._check_pca_resources(
                specs, entries, 80, data_root=Path("data"), history_directory=Path("history"), identity={"budget_id": "b1"},
            )
        self.assertEqual(estimate["status"], "requires_measurement")
        self.assertEqual(result["status"], "passed")
        self.assertGreater(estimate["estimated_ram_bytes"], 32 * 1024 ** 3 * .8)
        self.assertEqual(measure.call_args.args[0], case)
        command = measure.call_args.args[1]
        self.assertEqual(command[command.index("--model") + 1], "PCA_LEGACY")
        self.assertEqual(measure.call_args.args[3]["pca_estimate"], estimate)
        with patch.object(resources, "_system_memory_bytes", return_value=64 * 1024 ** 3), patch.object(
            resources, "_run_resource_probe", side_effect=AssertionError("safe estimate must not run PCA"),
        ):
            result = resources._check_pca_resources(
                specs, entries, 80, data_root=Path("data"), history_directory=Path("history"), identity={},
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["measurement_kind"], "static_estimate")

    def test_pca_child_requires_remote_gate_and_measures_cpu_execution(self):
        from tests.checks import check_dev18_resources as resources

        case = {"model": "PCA_LEGACY", "config_id": "all", "ratio": 100, "seed": 0, "series": "14"}
        values = numpy.zeros((110, 2), dtype=numpy.float32)
        with patch("tests.checks.run_lightning_dev18.require_lightning_cuda", side_effect=RuntimeError("remote gate")), patch(
            "tests.ghl_main.run_registered_models.load_registered_inputs",
        ) as load:
            with self.assertRaisesRegex(RuntimeError, "remote gate"):
                resources.run_child_probe(model="PCA_LEGACY", config_id="all", series="14",
                                          data_root=Path("data"), maximum_memory_percent=80)
            load.assert_not_called()
        with patch("tests.checks.run_lightning_dev18.require_lightning_cuda"), patch.object(
            resources, "_find_case", return_value=(case, case),
        ), patch("src.common.set_reproducible_seed.set_reproducible_seed"), patch(
            "tests.ghl_main.run_registered_models.load_registered_inputs", return_value={"test_sessions": (values,)},
        ), patch("src.common.run_registered_model.execute_registered_model", return_value={}) as execute, patch.object(
            resources, "_maximum_rss_bytes", return_value=600,
        ), patch.object(resources, "_system_memory_bytes", return_value=1000):
            result = resources.run_child_probe(model="PCA_LEGACY", config_id="all", series="14",
                                              data_root=Path("data"), maximum_memory_percent=80)
            self.assertEqual(execute.call_args.kwargs["device"], "cpu")
            self.assertIs(execute.call_args.kwargs["test_sessions"][0], values)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["ram_peak_percent"], 60)
            self.assertEqual(result["measurement_kind"], "process_rss")
            execute.side_effect = MemoryError("allocation failed")
            failed = resources.run_child_probe(model="PCA_LEGACY", config_id="all", series="14",
                                              data_root=Path("data"), maximum_memory_percent=80)
            self.assertEqual(failed["status"], "failed")
            self.assertIn("MemoryError", failed["error"])

    def test_pca_capacity_evidence_rejects_unmeasured_or_inconsistent_success(self):
        from tests.checks.check_dev18_resources import _has_consistent_capacity_evidence

        static = {"model": "PCA_LEGACY", "measurement_kind": "static_estimate", "estimated_ram_bytes": 700,
                  "ram_total_bytes": 1000, "maximum_memory_percent": 80}
        measured = {**static, "measurement_kind": "process_rss", "estimated_ram_bytes": 900,
                    "ram_peak_bytes": 600, "ram_peak_percent": 60, "wall_time_seconds": 2, "actual_backend": "cpu"}
        for row in (static, measured):
            self.assertTrue(_has_consistent_capacity_evidence([row], {"PCA_LEGACY"}, 80))
        for row in ({**static, "estimated_ram_bytes": 900}, {**measured, "ram_peak_bytes": 800, "ram_peak_percent": 80},
                    {**measured, "ram_peak_percent": 1}, {**measured, "measurement_kind": "unknown"}):
            with self.subTest(row=row):
                self.assertFalse(_has_consistent_capacity_evidence([row], {"PCA_LEGACY"}, 80))
        self.assertFalse(_has_consistent_capacity_evidence([], {"PCA_LEGACY"}, 80))

    def test_pca_probe_reuse_and_report_keep_the_same_representative_and_estimate(self):
        from tests.checks import check_dev18_resources as resources

        case = {"model": "PCA_LEGACY", "config_id": "all", "ratio": 100, "seed": 0, "series": "14"}
        estimate = {**case, "status": "requires_measurement", "measurement_kind": "static_estimate",
                    "estimated_ram_bytes": 900, "ram_total_bytes": 1000, "maximum_memory_percent": 80}
        identity = {"budget_id": "b1", "environment": {"runtime": "sealed"}, "maximum_memory_percent": 80,
                    "ram_total_bytes": 1000, "pca_estimate": estimate}
        result = {**case, "status": "passed", "measurement_kind": "process_rss", "actual_backend": "cpu",
                  "ram_peak_bytes": 600, "ram_total_bytes": 1000, "ram_peak_percent": 60,
                  "maximum_memory_percent": 80, "wall_time_seconds": 2}
        with tempfile.TemporaryDirectory() as directory, patch.object(resources.subprocess, "Popen") as launch:
            process = launch.return_value.__enter__.return_value
            process.pid, process.returncode = 42, 0
            process.communicate.return_value = (resources.RESULT_PREFIX + json.dumps(result), "")
            first = resources._run_resource_probe(case, ["probe"], directory, identity)
            self.assertEqual(first["pca_estimate"], estimate)
            self.assertEqual(resources._run_resource_probe(case, ["probe"], directory, identity), first)
            self.assertEqual(launch.call_count, 1)
            history = json.loads(Path(first["probe_history"]["file"]).read_text(encoding="utf-8"))
            self.assertEqual(history["result"]["pca_estimate"], estimate)
            with patch.object(resources, "_load_plan", return_value=([], [])), patch.object(
                resources, "_pca_static_check", return_value=estimate,
            ):
                self.assertTrue(resources._has_required_pca_evidence([first], {"PCA_LEGACY"}, 80))
                for changed in ({"series": "01"}, {"config_id": "fraction"}, {"ratio": 5}, {"pca_estimate": {}},
                                {"probe_history": None}):
                    self.assertFalse(resources._has_required_pca_evidence([{**first, **changed}], {"PCA_LEGACY"}, 80))
            history["result"]["ram_peak_percent"] = 1
            Path(first["probe_history"]["file"]).write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "근거"):
                resources._run_resource_probe(case, ["probe"], directory, identity)

    def test_paper_tspulse_gate_accepts_native_head_and_batch_evidence(self):
        from tests.checks.check_dev18_resources import _has_required_tier3_evidence

        case = {"model": "TSPulse", "config_id": "pulse", "ratio": 100, "seed": 0, "series": "01"}
        spec = {**case, "common_recipe": {"methodology_revision": "paper_tuning_v4"}}
        arguments = {"batch_size": 128, "context_length": 512, "aggregation_window": 64}
        row = {**case, "execution_policy": {"status": "passed", **arguments, "official_protocol": True},
               "wall_time_seconds": 1, "gpu_peak_bytes": 100, "gpu_peak_percent": 1,
               "ram_peak_bytes": 100, "ram_peak_percent": 1,
               "equivalence": {"status": "passed", "reference_batch_size": 1, "registered_batch_size": 128,
                   "rtol": 1e-6, "atol": 1e-8, "head_maximum_absolute_differences": {
                       head: 0 for head in ("time", "fft", "pred", "ensemble")}}}
        with patch("tests.checks.check_dev18_resources._load_plan", return_value=([spec], [{"series": "01", "feature_count": 2}])), patch(
            "tests.checks.check_dev18_resources.select_probe_cases", return_value=[case],
        ), patch("src.common.run_registered_model.build_entrypoint_arguments", return_value=arguments):
            self.assertTrue(_has_required_tier3_evidence([row], {"TSPulse"}))
            differences = row["equivalence"]["head_maximum_absolute_differences"]
            differences["raw_max"] = differences.pop("ensemble")
            self.assertFalse(_has_required_tier3_evidence([row], {"TSPulse"}))

    def test_paper_resource_cases_keep_independent_gdn_configs_and_feasible_inputs(self):
        specs = [{"model": "GDN", "config_id": f"gdn{epoch}", "ratio": 100,
                  "seed": 0, "target_use": "fit_full_prefix",
                  "common_recipe": {"methodology_revision": "paper_tuning_v4"},
                  "hyperparameters": {"window": 20, "epochs": epoch, "validation_ratio": .1,
                                      "batch_size": 128, "embedding": 64, "topk": 5}}
                 for epoch in (3, 10)]
        entries = [{"series": "01", "row_count": 200, "training_boundary": 100, "feature_count": 2},
                   {"series": "02", "row_count": 200, "training_boundary": 100, "feature_count": 19}]
        cases = select_probe_cases(specs, entries)
        self.assertEqual({case["config_id"] for case in cases}, {"gdn3", "gdn10"})
        self.assertEqual({case["series"] for case in cases}, {"02"})

    def test_child_probe_sets_seed_before_loading_model_inputs(self):
        from tests.checks.check_dev18_resources import run_child_probe

        events = []
        def stop_at_input(**arguments):
            events.append("inputs")
            raise ValueError("stop before model execution")

        with patch("tests.checks.check_dev18_resources._find_case", return_value=({}, {"seed": 7})), patch(
            "src.common.set_reproducible_seed.set_reproducible_seed",
            side_effect=lambda seed: events.append(("seed", seed)),
        ), patch("tests.ghl_main.run_registered_models.load_registered_inputs", side_effect=stop_at_input):
            with self.assertRaisesRegex(ValueError, "stop before"):
                run_child_probe(model="GDN", config_id="c1", series="01", data_root=Path("."),
                                maximum_memory_percent=80)
        self.assertEqual(events, [("seed", 7), "inputs"])

    def test_probe_resume_preserves_completed_interrupted_and_changed_environment(self):
        case = {"model": "GDN", "config_id": "c1", "series": "01", "ratio": 100, "seed": 0}
        identity = {"budget_id": "b1", "environment": {"torch": "pinned"},
                    "maximum_memory_percent": 80, "ram_total_bytes": 1000}
        result = {**case, "status": "passed", "maximum_memory_percent": 80,
                  "gpu_peak_bytes": 10, "gpu_total_bytes": 1000, "gpu_peak_percent": 1.0,
                  "ram_peak_bytes": 20, "ram_total_bytes": 1000, "ram_peak_percent": 2.0}
        with tempfile.TemporaryDirectory() as directory, patch(
            "tests.checks.check_dev18_resources.subprocess.Popen",
        ) as launch:
            process = launch.return_value.__enter__.return_value
            process.pid, process.returncode = 42, 0
            process.communicate.return_value = (
                "DEV18_RESOURCE_RESULT=" + json.dumps(result), "",
            )
            first = _run_resource_probe(case, ["probe"], directory, identity)
            original = Path(first["probe_history"]["file"]).read_bytes()
            self.assertEqual(_run_resource_probe(case, ["probe"], directory, identity), first)
            self.assertEqual(launch.call_count, 1)
            process.communicate.side_effect = KeyboardInterrupt()
            changed = {**identity, "environment": {"torch": "changed"}}
            with self.assertRaises(KeyboardInterrupt):
                _run_resource_probe(case, ["probe"], directory, changed)
            process.kill.assert_called_once()
            process.wait.assert_called_once()
            records = [json.loads(path.read_text(encoding="utf-8")) for path in Path(directory).glob("*.json")]
            self.assertEqual(sorted(row["status"] for row in records), ["complete", "interrupted"])
            process.communicate.side_effect = None
            resumed = _run_resource_probe(case, ["probe"], directory, changed)
            self.assertNotEqual(resumed["probe_history"], first["probe_history"])
            self.assertEqual(Path(first["probe_history"]["file"]).read_bytes(), original)
            self.assertEqual(launch.call_count, 3)

    def test_full_prefix_probes_never_pass_validation_to_session_models(self):
        values = numpy.arange(500, dtype=numpy.float32).reshape(100, 5)
        for model in ("GDN",):
            captured = {}

            def entrypoint(fit_sessions, validation_sessions, test_sessions, **arguments):
                captured.update(fit=fit_sessions, validation=validation_sessions, tests=test_sessions, arguments=arguments)

            with self.subTest(model=model), patch(
                "src.common.run_registered_model.build_entrypoint_arguments",
                return_value={"window_size": 5, "batch_size": 4, "full_prefix": True},
            ), patch("src.common.run_registered_model.load_model_entrypoint", return_value=entrypoint):
                _run_model_probe(
                    {"model": model, "tier": "t2", "ratio": 100, "target_use": "fit_full_prefix"},
                    {"normal_training": values, "test_sessions": (values,)}, device="cuda",
                )
            self.assertEqual(captured["validation"], ())
            self.assertTrue(captured["arguments"]["full_prefix"])
            self.assertEqual(captured["arguments"]["epochs"], 1)

    def test_gdn_probe_runs_eight_consecutive_training_batches(self):
        fit = numpy.zeros((50, 3), dtype=numpy.float32)
        captured = {}

        def entrypoint(fit_sessions, validation_sessions, test_sessions, **arguments):
            captured["fit_length"] = len(fit_sessions[0])
            captured["validation_length"] = len(validation_sessions[0])
            captured["test_length"] = len(test_sessions[0])
            captured["arguments"] = arguments

        with patch(
            "src.common.run_registered_model.build_entrypoint_arguments",
            return_value={"window_size": 5, "batch_size": 4},
        ), patch(
            "src.common.run_registered_model.load_model_entrypoint",
            return_value=entrypoint,
        ), patch(
            "src.common.run_registered_model.prepare_session_inputs",
            return_value={
                "fit_sessions": (fit,),
                "validation_sessions": (fit,),
                "test_sessions": (fit,),
            },
        ):
            result = _run_model_probe(
                {"model": "GDN", "tier": "t2", "ratio": 100},
                {"normal_training": fit, "test_sessions": (fit,)},
                device="cuda",
            )

        self.assertEqual(captured["fit_length"], 37)
        self.assertEqual(captured["validation_length"], 37)
        self.assertEqual(captured["test_length"], 37)
        self.assertEqual(captured["arguments"]["epochs"], 1)
        self.assertEqual(
            result.get("probe_scope"),
            "exact maximum batch; eight consecutive training updates",
        )

    def test_official_gdn_probe_keeps_eight_full_training_batches_after_holdout(self):
        values = numpy.zeros((500, 3), dtype=numpy.float32)
        for validation_ratio in (0.1, 0.2):
            arguments = {"window_size": 5, "batch_size": 32, "validation_ratio": validation_ratio,
                         "official_procedure": True, "full_prefix": True}
            with self.subTest(validation_ratio=validation_ratio), patch(
                "src.common.run_registered_model.build_entrypoint_arguments", return_value=arguments,
            ), patch("src.common.run_registered_model.load_model_entrypoint") as load, patch(
                "src.common.run_registered_model.prepare_session_inputs", return_value={
                    "fit_sessions": (values,), "validation_sessions": (), "test_sessions": (values,),
                },
            ):
                result = _run_model_probe(
                    {"model": "GDN", "tier": "t2", "ratio": 100, "target_use": "fit_full_prefix",
                     "common_recipe": {"methodology_revision": "paper_tuning_v4"}},
                    {"normal_training": values, "test_sessions": (values,)}, device="cuda",
                )
            call = load.return_value.call_args
            windows = len(call.args[0][0]) - 5
            training_windows = windows - int(windows * validation_ratio)
            self.assertGreaterEqual(training_windows // 32, 8)
            self.assertLessEqual(training_windows, 8 * 32 + 1)
            self.assertEqual(call.args[1], ())
            self.assertEqual(call.kwargs["batch_size"], 32)
            self.assertEqual(call.kwargs["validation_ratio"], validation_ratio)
            self.assertEqual(result["training_full_batch_count"], training_windows // 32)

    def test_ram_capacity_respects_cgroup_limits_without_using_current_consumption(self):
        host_memory = 64 * 1024 ** 3
        paths = ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes")
        for limits, expected in (
            ({paths[0]: str(16 * 1024 ** 3)}, 16 * 1024 ** 3),
            ({paths[0]: "max", paths[1]: str(32 * 1024 ** 3)}, 32 * 1024 ** 3),
            ({paths[0]: "max", paths[1]: str(2 ** 63 - 4096)}, host_memory),
            ({paths[0]: "invalid", paths[1]: "0"}, host_memory),
        ):
            def read_limit(path, **arguments):
                if str(path) not in limits:
                    raise FileNotFoundError(path)
                return limits[str(path)]

            with self.subTest(limits=limits), patch(
                "tests.checks.check_dev18_resources.os.sysconf", create=True,
                side_effect=lambda name: {"SC_PAGE_SIZE": 4096,
                                          "SC_PHYS_PAGES": host_memory // 4096}[name],
            ), patch.object(Path, "read_text", autospec=True, side_effect=read_limit):
                self.assertEqual(_system_memory_bytes(), expected)

    def test_selects_highest_ratio_and_largest_model_specific_case(self):
        specs = [
            {
                "model": "GDN", "config_id": "gdn", "ratio": ratio,
                "seed": seed, "hyperparameters": {
                    "window": 5, "batch_size": 128, "embedding": 128,
                },
            }
            for ratio in (10, 100)
            for seed in (0, 1)
        ] + [{
            "model": "TimeRCD", "config_id": "time", "ratio": 100,
            "seed": 0, "hyperparameters": {"context_length": 5000},
        }, {
            "model": "TSPulse", "config_id": "tspulse", "ratio": 100,
            "seed": 0, "hyperparameters": {
                "context_length": 512, "patch_size": 8,
                "heads": ["time", "fft", "pred", "raw_max"],
                "aggregation_window": 64,
            },
        }]
        entries = [
            {
                "series": "01", "row_count": 1552,
                "training_boundary": 1000, "feature_count": 20,
            },
            {
                "series": "02", "row_count": 9000,
                "training_boundary": 3000, "feature_count": 19,
            },
        ]

        cases = select_probe_cases(specs, entries)

        self.assertEqual(
            [(case["model"], case["ratio"], case["seed"], case["series"])
             for case in cases],
            [
                ("GDN", 100, 0, "01"),
                ("TimeRCD", 100, 0, "02"),
                ("TSPulse", 100, 0, "01"),
            ],
        )

    def test_capacity_requires_configured_headroom(self):
        self.assertEqual(capacity_status(79, 100, 80), "passed")
        self.assertEqual(capacity_status(80, 100, 80), "failed")

    def test_disk_space_is_observed_without_blocking_the_resource_gate(self):
        from tests.checks import check_dev18_resources as resources

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(resources, "REPOSITORY_ROOT", root))
            for target in ("tests.checks.run_lightning_dev18.require_lightning_cuda",
                           "tests.ghl_main.run_dev18_tuning._require_clean_worktree",
                           "src.common.set_reproducible_seed.set_reproducible_seed"):
                stack.enter_context(patch(target))
            stack.enter_context(patch("tests.checks.seal_runtime_environment.collect_runtime_environment_identity",
                                      return_value={"cuda_device": {"name": "test"}}))
            stack.enter_context(patch.object(resources, "_git_head", return_value="a" * 40))
            stack.enter_context(patch("src.common.execution_identity.file_sha256", return_value="b" * 64))
            stack.enter_context(patch.object(resources, "_system_memory_bytes", return_value=1000))
            stack.enter_context(patch.object(resources, "_load_plan", return_value=([], [])))
            stack.enter_context(patch.object(resources, "_load_resource_budget",
                                      return_value={"budget_id": "b1", "experiment_mode": "full_prefix_v2"}))
            stack.enter_context(patch.object(resources, "verify_input_files", return_value={"status": "passed"}))
            stack.enter_context(patch.object(resources, "_check_pca_resources",
                                      return_value={"model": "PCA_LEGACY", "status": "passed"}))
            disk = stack.enter_context(patch.object(resources.shutil, "disk_usage"))
            for reading in (SimpleNamespace(total=1000, used=1000, free=0), OSError("disk query failed")):
                with self.subTest(reading=reading):
                    disk.side_effect = reading if isinstance(reading, Exception) else None
                    disk.return_value = reading
                    report = resources.run_resource_check(data_root=root, output_path=root / "resource.json")
                    saved = json.loads((root / "resource.json").read_text(encoding="utf-8"))
                    self.assertEqual(report["status"], "passed")
                    self.assertEqual(saved["disk_observation"], report["disk_observation"])
                    self.assertFalse(any(row.get("resource") == "disk" for row in report["results"]))
                    observation = report["disk_observation"]
                    self.assertEqual(observation["policy"], "informational_only")
                    if isinstance(reading, Exception):
                        self.assertEqual(observation["status"], "unavailable")
                        self.assertIsNone(observation["free_bytes"])
                        self.assertIn("disk query failed", observation["reason"])
                    else:
                        self.assertEqual(observation["status"], "observed")
                        self.assertEqual(observation["free_bytes"], 0)
                        self.assertEqual(observation["total_bytes"], 1000)

    def test_verifies_every_manifest_input_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tuning" / "series.csv"
            source.parent.mkdir()
            source.write_bytes(b"data")
            result = verify_input_files([{
                "series": "01",
                "source_directory": "tuning",
                "name": source.name,
                "size_bytes": 4,
                "sha256": "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7",
            }], root)
            self.assertEqual(result["file_count"], 1)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_input_files([{
                    "series": "01",
                    "source_directory": "tuning",
                    "name": source.name,
                    "size_bytes": 4,
                    "sha256": "0" * 64,
                }], root)

    def test_resource_report_is_bound_to_gpu_commit_and_exact_models(self):
        report = {
            "status": "passed",
            "project_commit": "a" * 40,
            "gate_code_sha256": "b" * 64,
            "input_manifest_sha256": "c" * 64,
            "budget_id": "b123456789abc",
            "pytorch_alloc_conf": "expandable_segments:True",
            "maximum_memory_percent": 80,
            "cuda_device": {"name": "NVIDIA L4", "total_memory_bytes": 24},
            "environment": {"packages": {"torch": "pinned"}}, "ram_total_bytes": 1000,
            "checked_models": ["GDN", "MWVAR"],
            "results": [
                {
                    "model": "GDN", "status": "passed",
                    "gpu_peak_bytes": 10, "gpu_total_bytes": 1000,
                    "gpu_peak_percent": 1.0,
                    "ram_peak_bytes": 20, "ram_total_bytes": 1000,
                    "ram_peak_percent": 2.0,
                    "maximum_memory_percent": 80,
                },
                {"model": "MWVAR", "status": "passed"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resource_gate.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            validated = validate_resource_report(
                path,
                project_commit="a" * 40,
                gate_code_sha256="b" * 64,
                input_manifest_sha256="c" * 64,
                budget_id="b123456789abc",
                cuda_device=report["cuda_device"],
                expected_models={"GDN", "MWVAR"},
                environment=report["environment"], ram_total_bytes=1000,
            )
            self.assertEqual(validated, report)
            for changed in ({"environment": {"packages": {"torch": "changed"}}},
                            {"ram_total_bytes": 500}):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "코드·입력·예산"):
                    validate_resource_report(path, **{
                        "project_commit": "a" * 40, "gate_code_sha256": "b" * 64,
                        "input_manifest_sha256": "c" * 64, "budget_id": "b123456789abc",
                        "cuda_device": report["cuda_device"], "expected_models": {"GDN", "MWVAR"},
                        "environment": report["environment"], "ram_total_bytes": 1000, **changed,
                    })
            path.write_text(json.dumps({
                **report, "pytorch_alloc_conf": "max_split_size_mb:64",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "코드·입력·예산"):
                validate_resource_report(
                    path,
                    project_commit="a" * 40,
                    gate_code_sha256="b" * 64,
                    input_manifest_sha256="c" * 64,
                    budget_id="b123456789abc",
                    cuda_device=report["cuda_device"],
                    expected_models={"GDN", "MWVAR"},
                    environment=report["environment"], ram_total_bytes=1000,
                )
            path.write_text(json.dumps({**report, "results": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "코드·입력·예산"):
                validate_resource_report(
                    path,
                    project_commit="a" * 40,
                    gate_code_sha256="b" * 64,
                    input_manifest_sha256="c" * 64,
                    budget_id="b123456789abc",
                    cuda_device=report["cuda_device"],
                    expected_models={"GDN", "MWVAR"},
                    environment=report["environment"], ram_total_bytes=1000,
                )

    def test_resource_report_rejects_missing_or_failed_tier3_evidence(self):
        time_rcd = {
            "model": "TimeRCD", "config_id": "c1c5aeea6f7d3",
            "ratio": 100, "seed": 0, "series": "13",
            "status": "passed", "wall_time_seconds": 1.0,
            "gpu_peak_bytes": 10, "gpu_total_bytes": 1000,
            "gpu_peak_percent": 1.0,
            "ram_peak_bytes": 20, "ram_total_bytes": 1000,
            "ram_peak_percent": 2.0, "maximum_memory_percent": 80,
            "execution_policy": {
                "status": "passed", "context_length": 5000,
                "attention_query_chunk_size": 64,
            },
        }
        tspulse = {
            "model": "TSPulse", "config_id": "c12c5e6196ea5",
            "ratio": 100, "seed": 0, "series": "13",
            "status": "passed", "wall_time_seconds": 1.0,
            "gpu_peak_bytes": 10, "gpu_total_bytes": 1000,
            "gpu_peak_percent": 1.0,
            "ram_peak_bytes": 20, "ram_total_bytes": 1000,
            "ram_peak_percent": 2.0, "maximum_memory_percent": 80,
            "execution_policy": {
                "status": "passed", "batch_size": 32,
                "context_length": 512, "aggregation_window": 64,
            },
            "equivalence": {
                "status": "passed", "reference_batch_size": 1,
                "registered_batch_size": 32,
                "rtol": 1e-6, "atol": 1e-8,
                "head_maximum_absolute_differences": {
                    "time": 0.0, "fft": 0.0, "pred": 0.0, "raw_max": 0.0,
                },
            },
        }
        report = {
            "status": "passed", "project_commit": "a" * 40,
            "gate_code_sha256": "b" * 64, "input_manifest_sha256": "c" * 64,
            "budget_id": "b123456789abc",
            "pytorch_alloc_conf": "expandable_segments:True",
            "maximum_memory_percent": 80,
            "cuda_device": {"name": "NVIDIA L4", "total_memory_bytes": 24},
            "environment": {"packages": {"torch": "pinned"}}, "ram_total_bytes": 1000,
            "checked_models": ["TimeRCD", "TSPulse"],
            "results": [time_rcd, tspulse],
        }
        validation_arguments = {
            "project_commit": "a" * 40,
            "gate_code_sha256": "b" * 64,
            "input_manifest_sha256": "c" * 64,
            "budget_id": "b123456789abc",
            "cuda_device": report["cuda_device"],
            "expected_models": {"TimeRCD", "TSPulse"},
            "environment": report["environment"], "ram_total_bytes": 1000,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resource_gate.json"
            for broken_tspulse in (
                {key: value for key, value in tspulse.items() if key != "equivalence"},
                {**tspulse, "equivalence": {**tspulse["equivalence"], "status": "failed"}},
            ):
                path.write_text(
                    json.dumps({**report, "results": [time_rcd, broken_tspulse]}),
                    encoding="utf-8",
                )
                with self.subTest(equivalence=broken_tspulse.get("equivalence")):
                    with self.assertRaisesRegex(ValueError, "코드·입력·예산"):
                        validate_resource_report(path, **validation_arguments)

            path.write_text(
                json.dumps({
                    **report,
                    "results": [
                        {key: value for key, value in time_rcd.items()
                         if key != "execution_policy"},
                        tspulse,
                    ],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "코드·입력·예산"):
                validate_resource_report(path, **validation_arguments)

            for name, mutate in (
                (
                    "short_time_rcd_context",
                    lambda rows: rows[0]["execution_policy"].update(context_length=1),
                ),
                (
                    "wrong_tspulse_context",
                    lambda rows: rows[1]["execution_policy"].update(context_length=513),
                ),
                (
                    "wrong_tspulse_aggregation",
                    lambda rows: rows[1]["execution_policy"].update(
                        aggregation_window=96,
                    ),
                ),
                (
                    "wrong_config_id",
                    lambda rows: rows[0].update(config_id="c000000000000"),
                ),
                (
                    "infinite_wall_time",
                    lambda rows: rows[0].update(wall_time_seconds=float("inf")),
                ),
                (
                    "infinite_peak",
                    lambda rows: rows[1].update(gpu_peak_bytes=float("inf")),
                ),
                (
                    "nan_peak",
                    lambda rows: rows[0].update(ram_peak_bytes=float("nan")),
                ),
                (
                    "boolean_wall_time",
                    lambda rows: rows[0].update(wall_time_seconds=True),
                ),
                (
                    "boolean_peak",
                    lambda rows: rows[1].update(gpu_peak_bytes=True),
                ),
                (
                    "zero_gpu_total",
                    lambda rows: rows[0].update(gpu_total_bytes=0),
                ),
                (
                    "row_threshold_mismatch",
                    lambda rows: rows[1].update(maximum_memory_percent=79),
                ),
                (
                    "gpu_percent_mismatch",
                    lambda rows: rows[0].update(gpu_peak_percent=2.0),
                ),
                (
                    "passed_at_eighty_percent",
                    lambda rows: rows[0].update(
                        gpu_peak_bytes=800, gpu_peak_percent=80.0,
                    ),
                ),
                (
                    "passed_at_ninety_nine_percent",
                    lambda rows: rows[1].update(
                        ram_peak_bytes=990, ram_peak_percent=99.0,
                    ),
                ),
            ):
                broken = copy.deepcopy(report)
                mutate(broken["results"])
                path.write_text(json.dumps(broken), encoding="utf-8")
                with self.subTest(invalid_evidence=name):
                    with self.assertRaisesRegex(ValueError, "코드·입력·예산"):
                        validate_resource_report(path, **validation_arguments)
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(
                validate_resource_report(path, **validation_arguments),
                report,
            )
            with self.assertRaisesRegex(ValueError, "GPU"):
                validate_resource_report(
                    path,
                    project_commit="a" * 40,
                    gate_code_sha256="b" * 64,
                    input_manifest_sha256="c" * 64,
                    budget_id="b123456789abc",
                    cuda_device={"name": "Tesla T4", "total_memory_bytes": 16},
                    expected_models={"GDN", "MWVAR"},
                )


class TestResetDev18Run(unittest.TestCase):
    def test_deletes_only_dev18_run_and_runtime_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            removable = (
                root / ".runtime" / "runtime.json",
                root / ".runtime" / ".runtime.json.tmp",
                root / ".runtime" / "dev18_resource_gate.json",
                root / ".runtime" / ".dev18_resource_gate.json.tmp",
                root / ".runtime" / "dev18_checkpoint_smoke"
                / "time_rcd" / "dev18_checkpoint_smoke.json",
                root / "experiments" / "01_ghl_main" / "scores" / "dev18" / "score.npy",
                root / "experiments" / "01_ghl_main" / "logs" / "dev18_score_manifest.csv",
                root / "experiments" / "01_ghl_main" / "logs" / "dev18_oom_recovery.json",
                root / "experiments" / "01_ghl_main" / "logs" / "dev18_allocator_recovery.json",
                root / "experiments" / "01_ghl_main" / "results" / "dev18_tuning" / "table.csv",
            )
            preserved = (
                root / ".runtime" / "lightning_dev18_input.zip",
                root / "experiments" / "01_ghl_main" / "snapshots"
                / "dev18_selection" / "dev18_budget_manifest.json",
                root / "experiments" / "01_ghl_main" / "scores" / "ghl25" / "score.npy",
            )
            for path in (*removable, *preserved):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")

            result = reset_previous_run(root, confirmation="DELETE_DEV18_RUN")

            self.assertTrue(result["deleted"])
            self.assertTrue(all(not path.exists() for path in removable))
            self.assertTrue(all(path.exists() for path in preserved))

    def test_rejects_missing_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "DELETE_DEV18_RUN"):
                reset_previous_run(Path(directory), confirmation="yes")


if __name__ == "__main__":
    unittest.main()
