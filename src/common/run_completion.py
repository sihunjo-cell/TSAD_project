"""배치 실행의 최종 성공 표식과 snapshot 실행 신원을 읽고 쓴다."""

import json
from pathlib import Path


MARKER_NAME = "COMPLETE"
MARKER_CONTENT = "complete\n"


def completion_marker_path(run_dir) -> Path:
    return Path(run_dir) / MARKER_NAME


def clear_completion_marker(run_dir) -> None:
    completion_marker_path(run_dir).unlink(missing_ok=True)


def write_completion_marker(run_dir) -> None:
    completion_marker_path(run_dir).write_text(MARKER_CONTENT, encoding="ascii")


def has_completion_marker(run_dir) -> bool:
    try:
        return completion_marker_path(run_dir).read_text(encoding="ascii") == MARKER_CONTENT
    except (OSError, UnicodeError):
        return False


def snapshot_matches_source_identity(run_dir, expected_identity) -> bool:
    """완료 산출물이 현재 프로젝트·로컬 모델 소스에서 나온 것인지 확인한다."""
    if expected_identity is None:
        return True
    try:
        snapshot = json.loads(
            (Path(run_dir) / "snapshots" / "config_snapshot.json").read_text(
                encoding="utf-8",
            )
        )
        snapshot_identity = snapshot["config"]["source_identity"]
        snapshot_project = snapshot["git_commit_hash"]
        expected_project = expected_identity["project_commit"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return snapshot_project == expected_project and snapshot_identity == expected_identity
