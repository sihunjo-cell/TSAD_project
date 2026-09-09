"""실험이 잘못된 Git 상태에서 시작되지 않는지 검증한다."""

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.common import verify_run_context as context


class TestVerifyRunContext(unittest.TestCase):
    def test_one_liners_identity_uses_the_official_gitlab_url(self):
        self.assertEqual(
            context.UPSTREAM_SOURCE_COMMITS[
                "https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/"
                "publications/iclr2026_timeseriesfoundationmodelsad"
            ],
            "dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9",
        )
        self.assertNotIn("KU-Leuven-DTAI/One-Liners", context.UPSTREAM_SOURCE_COMMITS)

    def test_source_identity_covers_every_active_model_and_registry_contract(self):
        repository_root = Path(__file__).resolve().parents[2]

        identity = context.read_source_identity(repository_root)

        for relative_path in (
            "configs/model_registry.yaml",
            "src/models/tier1/mwvar.py",
            "src/models/tier1/one_liner_ensemble.py",
            "src/models/tier1/sqdiff.py",
            "src/models/tier1/sqdiff_last3.py",
            "src/models/tier1/pca_legacy.py",
            "src/models/tier2/paano/adapter.py",
            "src/models/tier2/gdn_official/adapter.py",
            "src/models/tier3/time_rcd.py",
            "src/models/tier3/tspulse.py",
        ):
            self.assertIn(relative_path, identity["local_model_sha256"])
        self.assertEqual(
            identity["upstream_source_commits"]["jinnnju/PaAno"],
            "d4c67116190efa4592dc6a8a157ced0def68b6af",
        )
        self.assertEqual(
            identity["upstream_source_commits"]["thu-sail-lab/Time-RCD"],
            "372bb980426b2f67007311c6f3165ab789c79bef",
        )
        self.assertNotIn("manigalati/usad", identity["upstream_source_commits"])
        for relative_path in (
            "src/models/tier2/AE/adapter.py",
            "src/models/tier2/LSTMAD/adapter.py",
            "src/models/tier2/USAD/adapter.py",
        ):
            self.assertNotIn(relative_path, identity["local_model_sha256"])

    def test_git_failure_is_an_error_not_a_placeholder_hash(self):
        with patch.object(
            context.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(128, ["git"]),
        ):
            with self.assertRaisesRegex(RuntimeError, "Git commit을 읽지 못했다"):
                context.read_git_hash(Path("broken-repository"))

    def test_dirty_project_is_rejected_before_training(self):
        with (
            patch.object(context, "read_source_identity", return_value={
                "project_commit": "a" * 40,
            }),
            patch.object(context, "run_git", return_value=" M src/model.py"),
        ):
            with self.assertRaisesRegex(RuntimeError, "TSAD 작업 트리가 clean하지 않다"):
                context.verify_run_context("project")

    def test_clean_project_returns_local_source_identity(self):
        identity = {
            "project_commit": "a" * 40,
            "local_model_sha256": {"src/model.py": "b" * 64},
            "upstream_source_commits": context.UPSTREAM_SOURCE_COMMITS,
        }
        with (
            patch.object(context, "read_source_identity", return_value=identity),
            patch.object(context, "run_git", return_value=""),
        ):
            self.assertEqual(context.verify_run_context("project"), identity)

    def test_runtime_version_mismatch_is_rejected(self):
        environment = {
            "python": "3.10.20",
            "packages": {"numpy": "1.26.4", "torch": "2.2.2"},
        }

        with (
            patch.object(context.platform, "python_version", return_value="3.10.20"),
            patch.object(context.metadata, "version", side_effect={
                "numpy": "2.0.0", "torch": "2.2.2+cpu",
            }.get),
        ):
            with self.assertRaisesRegex(RuntimeError, "numpy version"):
                context.verify_runtime_versions(environment)

    def test_python_compatibility_records_the_actual_patch_version(self):
        environment = {"python": ["3.11", "3.12"], "packages": {}}
        for version in ("3.11.0", "3.11.15", "3.12.11"):
            with self.subTest(version=version), patch.object(
                context.platform, "python_version", return_value=version,
            ):
                self.assertEqual(context.verify_runtime_versions(environment), {"python": version})
        for version in ("3.10.20", "3.13.0", "3.120.1"):
            with self.subTest(version=version), patch.object(
                context.platform, "python_version", return_value=version,
            ):
                with self.assertRaisesRegex(RuntimeError, "python version"):
                    context.verify_runtime_versions(environment)

    def test_runtime_reports_all_version_and_source_errors_together(self):
        environment = {
            "python": ["3.11", "3.12"],
            "packages": {"numpy": "2.3.2", "torch": "2.10.0"},
            "sources": {"Time-RCD": "git+https://github.com/thu-sail-lab/Time-RCD.git@pinned"},
        }
        with (
            patch.object(context.platform, "python_version", return_value="3.10.20"),
            patch.object(context.metadata, "version", side_effect=[
                "2.0.0", context.metadata.PackageNotFoundError("torch"),
            ]),
            patch.object(context.metadata, "distribution", side_effect=context.metadata.PackageNotFoundError("Time-RCD")),
        ):
            with self.assertRaises(RuntimeError) as raised:
                context.verify_runtime_versions(environment)
        for detail in ("python version", "numpy version", "설치되지 않았다: torch", "설치 원본을 읽지 못했다: Time-RCD"):
            self.assertIn(detail, str(raised.exception))

    def test_torch_build_suffix_does_not_change_public_version_contract(self):
        environment = {
            "python": "3.10.20",
            "packages": {"torch": "2.2.2"},
        }

        with (
            patch.object(context.platform, "python_version", return_value="3.10.20"),
            patch.object(context.metadata, "version", return_value="2.2.2+cu121"),
        ):
            self.assertEqual(
                context.verify_runtime_versions(environment),
                {"python": "3.10.20", "torch": "2.2.2+cu121"},
            )

    def test_pinned_package_source_commit_is_verified(self):
        environment = {
            "python": "3.10.20",
            "packages": {"pytorch-lightning": "2.6.2"},
            "sources": {
                "pytorch-lightning": (
                    "git+https://github.com/gonzachiar/pytorch-lightning.git@"
                    "834dbf3039ee82a2ac5e65eed25f9989222283c6"
                ),
            },
        }
        installed = SimpleNamespace(read_text=lambda _name: (
            '{"url":"https://pypi.org/project/pytorch-lightning/",'
            '"vcs_info":{"commit_id":"wrong"}}'
        ))
        with (
            patch.object(context.platform, "python_version", return_value="3.10.20"),
            patch.object(context.metadata, "version", return_value="2.6.2"),
            patch.object(context.metadata, "distribution", return_value=installed),
        ):
            with self.assertRaisesRegex(RuntimeError, "설치 원본"):
                context.verify_runtime_versions(environment)

    def test_uncommitted_source_change_during_run_is_rejected(self):
        expected = {
            "project_commit": "a" * 40,
            "local_model_sha256": {"src/model.py": "b" * 64},
            "upstream_source_commits": context.UPSTREAM_SOURCE_COMMITS,
        }
        with (
            patch.object(context, "read_source_identity", return_value=expected),
            patch.object(context, "run_git", return_value=" M src/model.py"),
        ):
            with self.assertRaisesRegex(RuntimeError, "실행 중 TSAD 작업 트리가 바뀌었다"):
                context.verify_source_identity_unchanged(expected, "project")


if __name__ == "__main__":
    unittest.main()
