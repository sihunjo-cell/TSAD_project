"""Lightning Dev18 자원 점검과 이전 실행 초기화 계약."""

import json
import tempfile
import unittest
from pathlib import Path

from tests.checks.check_dev18_resources import (
    capacity_status,
    disk_status,
    select_probe_cases,
    validate_resource_report,
    verify_input_files,
)
from tests.checks.reset_lightning_dev18 import reset_previous_run


class TestDev18ResourceCheck(unittest.TestCase):
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
        }]
        entries = [
            {
                "series": "01", "row_count": 6000,
                "training_boundary": 1000, "feature_count": 2,
            },
            {
                "series": "02", "row_count": 9000,
                "training_boundary": 3000, "feature_count": 20,
            },
        ]

        cases = select_probe_cases(specs, entries)

        self.assertEqual(
            [(case["model"], case["ratio"], case["seed"], case["series"])
             for case in cases],
            [("GDN", 100, 0, "02"), ("TimeRCD", 100, 0, "02")],
        )

    def test_capacity_requires_configured_headroom(self):
        self.assertEqual(capacity_status(79, 100, 80), "passed")
        self.assertEqual(capacity_status(80, 100, 80), "failed")
        self.assertEqual(disk_status(25, 25), "passed")
        self.assertEqual(disk_status(24, 25), "failed")

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
            "maximum_memory_percent": 80,
            "cuda_device": {"name": "NVIDIA L4", "total_memory_bytes": 24},
            "checked_models": ["GDN", "MWVAR"],
            "results": [
                {"model": "GDN", "status": "passed"},
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
            )
            self.assertEqual(validated, report)
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
                )
            path.write_text(json.dumps(report), encoding="utf-8")
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
                root / "experiments" / "01_ghl_main" / "scores" / "dev18" / "score.npy",
                root / "experiments" / "01_ghl_main" / "logs" / "dev18_score_manifest.csv",
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
