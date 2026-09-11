"""채점 실패·재개에도 실제 병렬 수와 CPU 관측 근거를 보존한다."""

import json
import platform
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.ghl_main import run_dev18_tuning as tuning


class TestScoringEnvironment(unittest.TestCase):
    def test_pca_snapshot_records_cpu_observations_without_changing_the_sealed_environment(self):
        environment = {"packages": "sealed", "cuda_device": "L4"}
        observed_cpu = {"affinity_cpu_count": 16, "cgroup_cpu_quota": {"limit_cpu_count": 16}}
        spec = {"model": "PCA_LEGACY", "target_use": "training_free",
                "common_recipe": {"training_split": "full_prefix_v2", "methodology_revision": "paper_tuning_v4"}}
        with TemporaryDirectory() as directory, patch.object(tuning, "_git_head", return_value="commit"), \
                patch.object(tuning, "_collect_scoring_cpu_environment", return_value=observed_cpu), \
                patch("src.models.tier1.pca_legacy.PCA_FIT_BLAS_THREADS", 16), \
                patch("src.common.run_registered_model.build_registered_execution_policy", return_value={}):
            path = tuning._write_run_snapshot(Path(directory), spec, {
                "input_identity": {"sha256": "input"}, "source_ranges": {"test_sessions": [[100, 300]]},
            }, environment)
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(snapshot["execution_resources"], {
            "pca_fit_blas_threads_requested": 16, "pca_distance_workers_requested": 16,
            "cpu": observed_cpu,
        })
        self.assertEqual(snapshot["environment"], environment)
        self.assertEqual(snapshot["execution_policy"], {})

    def test_ledger_records_pending_workers_and_zero_after_all_scores_are_reused(self):
        panel = {"model": "M", "config_id": "c1", "physical_ratio": 40, "logical_ratios": [40],
                 "seed": 0, "primary_score_variants": [""], "diagnostic_score_variants": []}
        rows = [{**panel, "config_id": config, "series": "01", "family": "A", "tier": "t2",
                 "score_variant": "", "primary_score": "true", "status": "complete", "budget_id": "budget",
                 "score_file": f"{config}.npy", "score_sha256": config, "seed": "0", "physical_ratio": "40"}
                for config in ("c1", "c2")]
        ledger = [{**row, "ratio": 40, "normalization": "trainnorm", "vus_pr": .7,
                   "evaluator_sha256": "evaluator", "ell_max_id": "ell"} for row in rows]
        budget = {"budget_id": "budget", "budget_sha256": "budget-sha", "series_ids": ["01"],
                  "execution_panel": [panel, {**panel, "config_id": "c2"}], "primary_logical_score_row_count": 2}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/input_manifest.yaml").write_text(json.dumps({"datasets": {"DEV18": {"files": [{
                "series": "01", "row_count": 20, "training_boundary": 10,
                "source_directory": "unused", "name": "unused.csv",
            }]}}}), encoding="utf-8")
            (root / "configs/environment.yaml").write_text("{}", encoding="utf-8")
            with patch.object(tuning, "REPOSITORY_ROOT", root), \
                    patch.object(tuning, "_validate_ell_max", return_value={
                        "ell_max_id": "ell", "series": [{"series": "01", "l_max_samples": 3}]}), \
                    patch.object(tuning, "validate_vus_evidence", return_value={
                        "evaluator_sha256": "evaluator", "n_thresholds": 250}), \
                    patch("tests.ghl_main.run_registered_models._verify_manifest_file"), \
                    patch.object(tuning, "_load_dev18_labels", return_value=[0, 1]), \
                    patch.object(tuning, "_score_primary_row", return_value={
                        "manifest_key": ["01", "M", "c2", "40", "0", ""],
                        "normalization": "trainnorm", "vus_pr": .7, "reused": False}):
                for reused, expected_workers in ((ledger[:1], 1), (ledger, 0)):
                    with self.subTest(reused=len(reused)), \
                            tuning.record_run_history(root / "commands", identity={"kind": "tuning_command"}) as history:
                        result = tuning.build_trial_score_ledger(
                            rows, budget=budget, reused_ledger=reused, project_commit="commit",
                            workers=16, scoring_history=history,
                        )
                        self.assertEqual([row["vus_pr"] for row in result], [.7, .7])
                        saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
                        self.assertEqual(saved["scoring_environment"]["actual_workers"], expected_workers)
                        self.assertEqual(saved["scoring_environment"]["pending_physical_scores"], expected_workers)

    def test_resolved_workers_are_durable_before_dispatch_even_when_dispatch_fails(self):
        tasks = [{"estimated_cost": 1, "manifest_row": {
            "series": "01", "model": "M", "config_id": f"c{index}",
            "physical_ratio": 100, "seed": 0, "score_variant": "",
        }} for index in range(12)]
        for requested, memory, expected in ((0, 12, 10), (16, 5, 3), (1, 12, 1)):
            with self.subTest(requested=requested), TemporaryDirectory() as directory:
                def fail_after_reading_history(*arguments, **keywords):
                    saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
                    context = saved["scoring_environment"]
                    self.assertEqual(context["requested_workers"], requested)
                    self.assertEqual(context["actual_workers"], expected)
                    self.assertEqual(context["pending_physical_scores"], 12)
                    self.assertEqual(context["cpu"]["os_logical_cpu_count"], 32)
                    self.assertEqual(saved["status"], "running")
                    raise RuntimeError("dispatch failed")

                with patch.object(tuning.os, "sched_getaffinity", return_value=set(range(32)), create=True), \
                        patch.object(tuning.os, "cpu_count", return_value=32), \
                        patch.object(tuning, "available_cpu_count", return_value=32), \
                        patch.object(tuning, "_detect_available_memory_bytes", return_value=memory * 1024 ** 3), \
                        patch.object(tuning, "_score_primary_row", side_effect=fail_after_reading_history), \
                        patch.object(tuning.concurrent.futures, "ProcessPoolExecutor",
                                     side_effect=fail_after_reading_history):
                    with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                        with tuning.record_run_history(directory, identity={"kind": "tuning_command"}) as history:
                            tuning._score_primary_rows(tasks, {}, requested, scoring_history=history)
                saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["scoring_environment"]["actual_workers"], expected)

    def test_empty_scoring_records_zero_without_starting_workers(self):
        with TemporaryDirectory() as directory, \
                patch.object(tuning, "_initialize_score_worker", side_effect=AssertionError("worker started")), \
                patch.object(tuning.concurrent.futures, "ProcessPoolExecutor",
                             side_effect=AssertionError("pool started")):
            with tuning.record_run_history(directory, identity={"kind": "tuning_command"}) as history:
                self.assertEqual(tuning._score_primary_rows([], {}, 0, scoring_history=history), {})
                saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
                self.assertEqual(saved["scoring_environment"]["actual_workers"], 0)
                self.assertEqual(saved["scoring_environment"]["scoring_status"], "no_new_scoring")

    def test_missing_cpu_information_is_null_with_reasons(self):
        task = {"estimated_cost": 1, "manifest_row": {
            "series": "01", "model": "M", "config_id": "c1",
            "physical_ratio": 100, "seed": 0, "score_variant": "",
        }}
        result = {"manifest_key": ["01", "M", "c1", "100", "0", ""], "reused": False}
        with TemporaryDirectory() as directory:
            with tuning.record_run_history(directory, identity={"kind": "tuning_command"}) as history:
                with patch.object(tuning.Path, "read_text", side_effect=OSError("not exposed")), \
                        patch.object(tuning.os, "cpu_count", return_value=None), \
                        patch.object(tuning, "available_cpu_count", return_value=1), \
                        patch.object(tuning.os, "sched_getaffinity", side_effect=OSError("not exposed"), create=True), \
                        patch.object(platform, "processor", return_value=""), \
                        patch.object(tuning, "_score_primary_row", return_value=result):
                    tuning._score_primary_rows([task], {}, 0, scoring_history=history)
            saved = json.loads(Path(history["history_file"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["scoring_environment"]["actual_workers"], 1)
        context = saved["scoring_environment"]["cpu"]
        for field in ("cpu_model", "os_logical_cpu_count", "affinity_cpu_ids",
                      "affinity_cpu_count", "cgroup_cpu_quota"):
            self.assertIsNone(context[field])
            self.assertTrue(context["null_reasons"][field])

    def test_cpu_model_and_cgroup_quota_use_observed_files(self):
        cases = (
            ({"/sys/fs/cgroup/cpu.max": "250000 100000"}, 2, 2.5, False),
            ({"/sys/fs/cgroup/cpu.max": "max 100000"}, 2, None, True),
            ({"/proc/self/cgroup": "0::/group/worker\n",
              "/sys/fs/cgroup/cpu.max": "max 100000",
              "/sys/fs/cgroup/group/cpu.max": "200000 100000",
              "/sys/fs/cgroup/group/worker/cpu.max": "max 100000"}, 2, 2.0, False),
            ({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "400000",
              "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}, 1, 4.0, False),
            ({"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1",
              "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000"}, 1, None, True),
        )
        for files, version, limit, unlimited in cases:
            with self.subTest(version=version, unlimited=unlimited):
                files = {**files, "/proc/cpuinfo": "processor : 0\nmodel name : Test CPU\n"}

                def read_observed_file(path, *arguments, **keywords):
                    try:
                        return files[path.as_posix()]
                    except KeyError:
                        raise FileNotFoundError(str(path)) from None

                with patch.object(tuning.Path, "read_text", read_observed_file), \
                        patch.object(tuning.os, "cpu_count", return_value=32), \
                        patch.object(tuning.os, "sched_getaffinity", return_value={2, 3}, create=True):
                    context = tuning._collect_scoring_cpu_environment()
                self.assertEqual(context["cpu_model"], "Test CPU")
                self.assertEqual(context["affinity_cpu_ids"], [2, 3])
                self.assertEqual(context["affinity_cpu_count"], 2)
                quota = context["cgroup_cpu_quota"]
                self.assertEqual(quota["version"], version)
                self.assertEqual(quota["limit_cpu_count"], limit)
                self.assertEqual(quota["unlimited"], unlimited)


if __name__ == "__main__":
    unittest.main()
