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


def snapshot_matches_git_hashes(run_dir, expected_git_hashes) -> bool:
    """완료 산출물이 현재 봉인한 두 저장소 commit에서 나온 것인지 확인한다."""
    if expected_git_hashes is None:
        return True
    try:
        snapshot = json.loads(
            (Path(run_dir) / "snapshots" / "config_snapshot.json").read_text(
                encoding="utf-8",
            )
        )
        snapshot_hashes = snapshot["config"]["git_hashes"]
        snapshot_project = snapshot["git_commit_hash"]
        expected_project = expected_git_hashes["tsad_project"]
        expected_fork = expected_git_hashes["gragod_fork"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return False
    if not isinstance(snapshot_hashes, dict):
        return False
    return (
        snapshot_project == expected_project
        and snapshot_hashes.get("tsad_project") == expected_project
        and snapshot_hashes.get("gragod_fork") == expected_fork
    )
