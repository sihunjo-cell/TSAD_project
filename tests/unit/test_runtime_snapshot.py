"""로컬 실행 환경 봉인의 생성·재개 계약을 검증한다."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestRuntimeSnapshot(unittest.TestCase):
    def _make_root(self, directory):
        root = Path(directory)
        (root / "src" / "models").mkdir(parents=True)
        (root / "configs").mkdir()
        (root / "src" / "models" / "requirements.txt").write_text(
            "torch==2.10.0\n", encoding="utf-8",
        )
        (root / "configs" / "environment.yaml").write_text(
            'python: "3.11.14"\n', encoding="utf-8",
        )
        return root

    def _functions(self):
        try:
            from tests.checks.seal_runtime_environment import (
                ensure_runtime_snapshot,
                validate_runtime_snapshot,
            )
        except ImportError as error:
            self.fail(f"runtime 봉인 기능이 없다: {error}")
        return ensure_runtime_snapshot, validate_runtime_snapshot

    def test_first_seal_embeds_exact_freeze_in_ignored_local_snapshot(self):
        ensure_runtime_snapshot, _ = self._functions()
        identity = {"python": "3.11.14", "torch": {"cuda_available": True}}

        with tempfile.TemporaryDirectory() as directory:
            root = self._make_root(directory)
            evidence = ensure_runtime_snapshot(
                repository_root=root,
                environment=identity,
                installed_packages=["numpy==2.3.2", "torch==2.10.0+cu126"],
                python_executable=sys.executable,
            )
            runtime_path = root / ".runtime" / "runtime.json"
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))

        self.assertEqual(runtime["identity"], identity)
        self.assertEqual(
            runtime["pip_freeze"],
            ["numpy==2.3.2", "torch==2.10.0+cu126"],
        )
        self.assertEqual(evidence["pip_freeze"], runtime["pip_freeze"])
        self.assertEqual(len(evidence["sha256"]), 64)

    def test_existing_local_seal_is_reused_when_environment_is_unchanged(self):
        ensure_runtime_snapshot, _ = self._functions()
        identity = {"python": "3.11.14", "torch": {"cuda_available": True}}

        with tempfile.TemporaryDirectory() as directory:
            root = self._make_root(directory)
            first = ensure_runtime_snapshot(
                repository_root=root,
                environment=identity,
                installed_packages=["torch==2.10.0+cu126"],
                python_executable=sys.executable,
            )
            second = ensure_runtime_snapshot(
                repository_root=root,
                environment=identity,
                installed_packages=["torch==2.10.0+cu126"],
                python_executable=sys.executable,
            )

        self.assertEqual(first, second)

    def test_resume_rejects_changed_packages_instead_of_resealing(self):
        ensure_runtime_snapshot, _ = self._functions()
        identity = {"python": "3.11.14", "torch": {"cuda_available": True}}

        with tempfile.TemporaryDirectory() as directory:
            root = self._make_root(directory)
            ensure_runtime_snapshot(
                repository_root=root,
                environment=identity,
                installed_packages=["torch==2.10.0+cu126"],
                python_executable=sys.executable,
            )
            with self.assertRaisesRegex(ValueError, "봉인.*다르"):
                ensure_runtime_snapshot(
                    repository_root=root,
                    environment=identity,
                    installed_packages=["torch==2.10.0+cu128"],
                    python_executable=sys.executable,
                )

    def test_first_seal_does_not_publish_a_partial_runtime_file(self):
        ensure_runtime_snapshot, _ = self._functions()
        identity = {"python": "3.11.14", "torch": {"cuda_available": True}}

        with tempfile.TemporaryDirectory() as directory:
            root = self._make_root(directory)
            with patch.object(Path, "replace", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    ensure_runtime_snapshot(
                        repository_root=root,
                        environment=identity,
                        installed_packages=["torch==2.10.0+cu126"],
                        python_executable=sys.executable,
                    )

            self.assertFalse((root / ".runtime" / "runtime.json").exists())

    def test_cuda_device_identity_records_gpu_kind_capability_and_driver(self):
        try:
            from tests.checks.seal_runtime_environment import (
                collect_cuda_device_identity,
            )
        except ImportError as error:
            self.fail(f"CUDA 장치 봉인 기능이 없다: {error}")
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            current_device=lambda: 0,
            get_device_name=lambda _index: "Tesla T4",
            get_device_capability=lambda _index: (7, 5),
            get_device_properties=lambda _index: SimpleNamespace(
                total_memory=16 * 1024 ** 3,
            ),
        ))

        identity = collect_cuda_device_identity(
            torch_module=fake_torch, driver_version="550.54.15",
        )

        self.assertEqual(identity["name"], "Tesla T4")
        self.assertEqual(identity["compute_capability"], [7, 5])
        self.assertEqual(identity["total_memory_bytes"], 16 * 1024 ** 3)
        self.assertEqual(identity["driver_version"], "550.54.15")

    def test_tracked_windows_snapshot_remains_a_valid_fallback(self):
        _, validate_runtime_snapshot = self._functions()
        identity = {"python": "3.11.14", "torch": {"cuda_available": False}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_directory = (
                root / "experiments" / "checks" / "reference_code" / "environment"
            )
            snapshot_directory.mkdir(parents=True)
            (root / "src" / "models").mkdir(parents=True)
            (root / "configs").mkdir()
            requirements = root / "src" / "models" / "requirements.txt"
            environment_config = root / "configs" / "environment.yaml"
            requirements.write_text("torch==2.10.0\n", encoding="utf-8")
            environment_config.write_text('python: "3.11.14"\n', encoding="utf-8")
            (snapshot_directory / "pip_freeze.txt").write_text(
                "torch==2.10.0\n", encoding="utf-8",
            )
            import hashlib

            runtime = {
                "environment": "tsad_models_311",
                "python_executable": str(Path(sys.executable).resolve()),
                "requirements": {
                    "file": "src/models/requirements.txt",
                    "sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
                },
                "environment_config": {
                    "file": "configs/environment.yaml",
                    "sha256": hashlib.sha256(environment_config.read_bytes()).hexdigest(),
                },
                "identity": identity,
                "pip_freeze": "pip_freeze.txt",
            }
            (snapshot_directory / "runtime.json").write_text(
                json.dumps(runtime), encoding="utf-8",
            )

            evidence = validate_runtime_snapshot(
                identity,
                repository_root=root,
                installed_packages=["torch==2.10.0"],
                python_executable=sys.executable,
            )

        self.assertEqual(evidence["pip_freeze"], ["torch==2.10.0"])


if __name__ == "__main__":
    unittest.main()
