"""학습 전후 Git 상태를 검증한다."""

import json
import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path


UPSTREAM_SOURCE_COMMITS = {
    "https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/publications/"
    "iclr2026_timeseriesfoundationmodelsad": (
        "dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9"
    ),
    "jinnnju/PaAno": "d4c67116190efa4592dc6a8a157ced0def68b6af",
    "d-ailin/GDN": "9853899da860682669a134e4af315d036aab4eca",
    "thu-sail-lab/Time-RCD": "372bb980426b2f67007311c6f3165ab789c79bef",
    "ibm-granite/granite-tsfm": "fe7a35697723e2a2f5246ae979474bfc554e26c0",
    "TheDatumOrg/TSB-AD": "6beac72e11d1155ade40870492c00d0d1cfdcaaf",
}

LOCAL_MODEL_FILES = (
    "configs/model_registry.yaml",
    "src/common/model_registry.py",
    "src/common/save_model_artifacts.py",
    "src/data_split/split_ratio_prefix.py",
    "src/models/tier1/mwvar.py",
    "src/models/tier1/one_liner_ensemble.py",
    "src/models/tier1/sqdiff.py",
    "src/models/tier1/sqdiff_last3.py",
    "src/models/tier1/pca_legacy.py",
    "src/models/tier2/paano/official.py",
    "src/models/tier2/paano/adapter.py",
    "src/models/tier2/gdn_official/official.py",
    "src/models/tier2/gdn_official/adapter.py",
    "src/models/tier3/time_rcd.py",
    "src/models/tier3/tspulse.py",
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_identity(project_path) -> dict:
    project_path = Path(project_path).resolve()
    local_hashes = {}
    for relative_path in LOCAL_MODEL_FILES:
        path = project_path / relative_path
        if not path.is_file():
            raise RuntimeError(f"로컬 모델 파일이 없다: {path}")
        local_hashes[relative_path] = _sha256(path)
    return {
        "project_commit": read_git_hash(project_path),
        "local_model_sha256": local_hashes,
        "upstream_source_commits": dict(UPSTREAM_SOURCE_COMMITS),
    }


def verify_run_context(project_path) -> dict:
    identity = read_source_identity(project_path)
    if run_git(project_path, "status", "--porcelain"):
        raise RuntimeError(f"TSAD 작업 트리가 clean하지 않다: {Path(project_path).resolve()}")
    return identity


def verify_source_identity_unchanged(expected_identity: dict, project_path) -> None:
    current_identity = read_source_identity(project_path)
    if current_identity != expected_identity:
        raise RuntimeError(
            f"실행 중 소스 신원이 바뀌었다: {expected_identity} -> {current_identity}"
        )
    if run_git(project_path, "status", "--porcelain"):
        raise RuntimeError(f"실행 중 TSAD 작업 트리가 바뀌었다: {Path(project_path).resolve()}")


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
