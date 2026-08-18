"""학습 전후 Git 상태를 검증한다."""

import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path


EXPECTED_GRAGOD_COMMIT = "485e26b0c6b1d63f4f3531c8d05597db82e9db29"


def run_git(repository_dir, *arguments) -> str:
    repository_dir = Path(repository_dir).resolve()
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repository_dir}", *arguments],
            capture_output=True,
            text=True,
            cwd=repository_dir,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise RuntimeError(
            f"Git 명령에 실패했다: {repository_dir} {' '.join(arguments)}"
        ) from error


def read_git_hash(repository_dir) -> str:
    try:
        return run_git(repository_dir, "rev-parse", "HEAD")
    except RuntimeError as error:
        raise RuntimeError(f"Git commit을 읽지 못했다: {Path(repository_dir).resolve()}") from error


def verify_run_context(project_path, fork_path) -> dict[str, str]:
    repositories = (("TSAD", project_path), ("GraGOD", fork_path))
    hashes = {}
    for name, path in repositories:
        commit = read_git_hash(path)
        if run_git(path, "status", "--porcelain"):
            raise RuntimeError(f"{name} 작업 트리가 clean하지 않다: {Path(path).resolve()}")
        hashes["tsad_project" if name == "TSAD" else "gragod_fork"] = commit
    if hashes["gragod_fork"] != EXPECTED_GRAGOD_COMMIT:
        raise RuntimeError(
            "GraGOD commit이 고정값과 다르다: "
            f"{hashes['gragod_fork']} != {EXPECTED_GRAGOD_COMMIT}"
        )
    return hashes


def verify_git_hashes_unchanged(expected_hashes: dict[str, str], project_path, fork_path) -> None:
    current_hashes = {
        "tsad_project": read_git_hash(project_path),
        "gragod_fork": read_git_hash(fork_path),
    }
    if current_hashes != expected_hashes:
        raise RuntimeError(
            f"실행 중 Git commit이 바뀌었다: {expected_hashes} -> {current_hashes}"
        )
    for name, path in (("TSAD", project_path), ("GraGOD", fork_path)):
        if run_git(path, "status", "--porcelain"):
            raise RuntimeError(f"실행 중 {name} 작업 트리가 바뀌었다: {Path(path).resolve()}")


def verify_runtime_versions(environment: dict) -> dict[str, str]:
    actual_versions = {"python": platform.python_version()}
    if actual_versions["python"] != environment["python"]:
        raise RuntimeError(
            f"python version이 고정값과 다르다: "
            f"{actual_versions['python']} != {environment['python']}"
        )
    for distribution, expected in environment["packages"].items():
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"고정 package가 설치되지 않았다: {distribution}") from error
        comparable = actual.split("+", 1)[0] if distribution == "torch" else actual
        if comparable != expected:
            raise RuntimeError(
                f"{distribution} version이 고정값과 다르다: {actual} != {expected}"
            )
        actual_versions[distribution] = actual
    for distribution, expected_source in environment.get("sources", {}).items():
        expected_url, separator, expected_commit = expected_source.removeprefix("git+").rpartition("@")
        if not separator or not expected_url or not expected_commit:
            raise RuntimeError(f"고정 package 설치 원본 형식이 잘못됐다: {distribution}")
        try:
            direct_url = json.loads(
                metadata.distribution(distribution).read_text("direct_url.json") or ""
            )
            actual_url = direct_url["url"]
            actual_commit = direct_url["vcs_info"]["commit_id"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(f"고정 package 설치 원본을 읽지 못했다: {distribution}") from error
        if (actual_url, actual_commit) != (expected_url, expected_commit):
            raise RuntimeError(
                f"{distribution} 설치 원본이 고정값과 다르다: "
                f"{actual_url}@{actual_commit} != {expected_url}@{expected_commit}"
            )
        actual_versions[f"{distribution}_source"] = f"{actual_url}@{actual_commit}"
    return actual_versions
