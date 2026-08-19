"""실험이 잘못된 Git 상태에서 시작되지 않는지 검증한다."""

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.common import verify_run_context as context


class TestVerifyRunContext(unittest.TestCase):
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
