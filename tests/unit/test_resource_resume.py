"""원격 CPU에서 RAM 변동과 검사 수정 전후의 재개 연결을 검증한다."""

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.checks import check_dev18_resources as resources


class TestResourceResume(unittest.TestCase):
    def test_ram_changes_warn_without_rewriting_measured_evidence(self):
        previous_ram = 64919400448
        specs = [{"model": "PCA_LEGACY", "config_id": "all", "ratio": 100, "seed": 0,
                  "target_use": "training_free", "hyperparameters": {"window": 100, "n_components": None}}]
        entries = [{"series": "14", "row_count": 1000000, "training_boundary": 28307, "feature_count": 17}]
        with patch.object(resources, "_system_memory_bytes", return_value=previous_ram):
            estimate = resources._pca_static_check(specs, entries, 80)
        measured = {key: estimate[key] for key in ("model", "config_id", "series", "ratio", "seed")}
        measured.update(status="passed", measurement_kind="process_rss", actual_backend="cpu",
                        ram_peak_bytes=20 * 1024 ** 3, ram_total_bytes=previous_ram,
                        ram_peak_percent=round(20 * 1024 ** 3 * 100 / previous_ram, 2),
                        wall_time_seconds=2, maximum_memory_percent=80, pca_estimate=estimate,
                        probe_history={"file": "measured.json", "sha256": "f" * 64})
        gpu = {"model": "GDN", "status": "passed", "gpu_peak_bytes": 50, "gpu_total_bytes": 100,
               "gpu_peak_percent": 50, "ram_peak_bytes": 2 * 1024 ** 3, "ram_total_bytes": previous_ram,
               "ram_peak_percent": round(2 * 1024 ** 3 * 100 / previous_ram, 2), "maximum_memory_percent": 80}
        report = {"status": "passed", "project_commit": "a" * 40, "gate_code_sha256": "b" * 64,
                  "input_manifest_sha256": "c" * 64, "budget_id": "b1", "environment": {"runtime": "same"},
                  "cuda_device": {"name": "L4"}, "pytorch_alloc_conf": resources.PYTORCH_ALLOC_CONF_VALUE,
                  "maximum_memory_percent": 80, "ram_total_bytes": previous_ram,
                  "checked_models": ["PCA_LEGACY", "GDN"], "results": [measured, gpu]}
        arguments = {key: report[key] for key in ("project_commit", "gate_code_sha256", "input_manifest_sha256",
                                                 "budget_id", "environment", "cuda_device")}
        arguments["expected_models"] = set(report["checked_models"])
        with tempfile.TemporaryDirectory() as directory, patch.object(resources, "_load_plan", return_value=(specs, entries)):
            path = Path(directory) / "resource_gate.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            original = path.read_bytes()
            for current_ram in (previous_ram + 8192, previous_ram - 8192, 25 * 1024 ** 3):
                with self.subTest(current_ram=current_ram), patch.object(resources, "_system_memory_bytes", return_value=current_ram):
                    self.assertEqual(resources.validate_resource_report(path, **arguments), report)
                    self.assertEqual(path.read_bytes(), original)
            for change in ({"ram_peak_percent": 1}, {"pca_estimate": {**estimate, "estimated_ram_bytes": 1}}):
                broken = copy.deepcopy(report)
                broken["results"][0].update(change)
                path.write_text(json.dumps(broken), encoding="utf-8")
                with self.subTest(change=change), self.assertRaises(ValueError):
                    resources.validate_resource_report(path, **arguments, ram_total_bytes=previous_ram + 8192)

    def test_static_pca_evidence_keeps_its_original_capacity_basis(self):
        specs = [{"model": "PCA_LEGACY", "config_id": "all", "ratio": 100, "seed": 0,
                  "target_use": "training_free", "hyperparameters": {"window": 100, "n_components": None}}]
        entries = [{"series": "01", "row_count": 300, "training_boundary": 100, "feature_count": 2}]
        with patch.object(resources, "_system_memory_bytes", return_value=64919400448):
            row = resources._pca_static_check(specs, entries, 80)
        with patch.object(resources, "_load_plan", return_value=(specs, entries)), patch.object(
            resources, "_system_memory_bytes", return_value=64919408640,
        ):
            self.assertTrue(resources._has_required_pca_evidence([row], {"PCA_LEGACY"}, 80))

    def test_resume_accepts_only_reviewed_checker_changes(self):
        from tests.checks import validate_resource_resume as resume

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*arguments):
                return subprocess.check_output(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                                *arguments], cwd=root, text=True).strip()

            git("init", "--quiet")
            (root / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
            (root / "gate.py").write_text("STRICT_RAM = True\n", encoding="utf-8")
            git("add", ".")
            git("commit", "--quiet", "-m", "base")
            original = git("rev-parse", "HEAD")
            (root / "gate.py").write_text("STRICT_RAM = False\n", encoding="utf-8")
            git("add", ".")
            git("commit", "--quiet", "-m", "resource fix")
            repaired = git("rev-parse", "HEAD")
            approved_blob = git("rev-parse", "HEAD:gate.py")
            with patch.object(resume, "RESOURCE_RESUME_SOURCE", original), patch.object(
                resume, "RESOURCE_RESUME_BLOBS", {"gate.py": approved_blob},
            ):
                resume._has_resource_resume_source.cache_clear()
                try:
                    self.assertTrue(resume.resource_resume_compatible(original, repaired, root))
                    self.assertTrue(resume.resource_resume_compatible(repaired, original, root))
                    (root / "gate.py").write_text("STRICT_RAM = 'unreviewed'\n", encoding="utf-8")
                    git("add", ".")
                    git("commit", "--quiet", "-m", "other gate edit")
                    self.assertFalse(resume.resource_resume_compatible(original, git("rev-parse", "HEAD"), root))
                    (root / "gate.py").write_text("STRICT_RAM = False\n", encoding="utf-8")
                    (root / "model.py").write_text("MODEL = 2\n", encoding="utf-8")
                    git("add", ".")
                    git("commit", "--quiet", "-m", "model changed")
                    self.assertFalse(resume.resource_resume_compatible(original, git("rev-parse", "HEAD"), root))
                finally:
                    resume._has_resource_resume_source.cache_clear()

    def test_checked_in_repair_preserves_the_running_source(self):
        from src.common.execution_identity import file_sha256
        from tests.checks.validate_resource_resume import (
            RESOURCE_RESUME_SOURCE, REPOSITORY_ROOT, committed_file_sha256, resource_resume_compatible,
        )

        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
        self.assertTrue(resource_resume_compatible(RESOURCE_RESUME_SOURCE, current))
        report = {"project_commit": RESOURCE_RESUME_SOURCE,
                  "gate_code_sha256": committed_file_sha256(RESOURCE_RESUME_SOURCE, "tests/checks/check_dev18_resources.py"),
                  "input_manifest_sha256": "c" * 64, "budget_id": "b1", "environment": {"runtime": "same"},
                  "cuda_device": {"name": "L4"}, "pytorch_alloc_conf": resources.PYTORCH_ALLOC_CONF_VALUE,
                  "maximum_memory_percent": 80, "ram_total_bytes": 64919400448, "status": "passed",
                  "checked_models": ["MWVAR"], "results": [{"model": "MWVAR", "status": "passed"}]}
        arguments = {key: report[key] for key in ("input_manifest_sha256", "budget_id", "environment", "cuda_device")}
        arguments.update(project_commit=current, gate_code_sha256=file_sha256(Path(resources.__file__)),
                         expected_models={"MWVAR"}, ram_total_bytes=64919408640)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resource_gate.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(resources.validate_resource_report(path, **arguments), report)
            report["gate_code_sha256"] = "0" * 64
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "자원 검사 코드"):
                resources.validate_resource_report(path, **arguments)


if __name__ == "__main__":
    unittest.main()
