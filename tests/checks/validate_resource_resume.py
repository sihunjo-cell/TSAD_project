"""승인된 자원 검사·PCA 실행 자원 변경 전후의 Dev18 재개를 확인한다."""

import hashlib
import re
import subprocess
from functools import lru_cache
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_RESUME_SOURCE = "57e91eb8de2d130fab0fae437041beb8eccd73ee"
# 검토한 변경의 파일 내용만 허용하며 후보·점수식의 다른 변경은 거부한다.
RESOURCE_RESUME_BLOBS = {
    "tests/checks/check_dev18_resources.py": "efc502ea0f307d887527662294b60a19c355a9ef",
    "tests/ghl_main/run_dev18_tuning.py": (
        "f610da99a9ce6e922ec269909c32fbd08af7acf4", "693a26cd1d74c4cc8be013b42c737a58c191fc8c",
        "0154ee9a6b2201c911a25c861e5ed1e7f0ec7c90",
        "8fc66a26ecd267dd730d2aaf3305f06a8da076b9",
        "25497ebd3ef92eb257e14a2669bf8ee55fdc196b",
    ),
    "src/models/tier1/pca_legacy.py": (
        "f6a5a21ef1b8f01fa3c1b83d0e7c091af6ca15d0", "0470ab13ea5a1143a205fe990695143c94b0d781",
        "3e61552c10de8976c5cd157fffd332d480ee7bdd",
        "a3aa48f4473b21b7790b54e2694c9cc0903ebff3",
    ),
    "tests/ghl_main/record_run_history.py": "d55fa556f0dd0239bdc29272d99693086b0c4774",
    "tests/ghl_main/run_ratio_tuning.py": (
        "b3672e16931916dd5dbedf753f997adfb2398aa0", "0a827b4ece81129488c131b3c0605715d2220dba",
        "6b42ce921bf8f70d1bfbdd42c7a82d26da78f928",
    ),
    "tests/ghl_main/build_tuning_support.py": "dc542fb68d61249cdd6436c45374a35638fc9004",
    "tests/ghl_main/compare_execution_runtimes.py": (
        "57ecfbe46e2e690bbad08ce1085f778267ba4fd5", "c8308f114ba9018d763ed4c22a0d21b5f9d95a3d",
    ),
    "tests/ghl_main/store_recommendation_evidence.py": "2a5d13463d29c93c42c719ee1729fab7629856f7",
    "tests/ghl_main/package_recommendation_handoff.py": "b58df2e69334686e972c72847093b82ec8818f21",
}


@lru_cache(maxsize=32)
def _has_resource_resume_source(commit, repository_root):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return False
    if commit == RESOURCE_RESUME_SOURCE:
        return True
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", RESOURCE_RESUME_SOURCE, commit],
                       cwd=repository_root, check=True, capture_output=True)
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", RESOURCE_RESUME_SOURCE, commit], cwd=repository_root, text=True,
        ).splitlines()
        for path in changed:
            if path.startswith(("docs/", "tests/unit/")) or path == "tests/checks/validate_resource_resume.py":
                continue
            expected = RESOURCE_RESUME_BLOBS.get(path)
            accepted = (expected,) if isinstance(expected, str) else (expected or ())
            if subprocess.check_output(
                ["git", "rev-parse", f"{commit}:{path}"], cwd=repository_root, text=True,
            ).strip() not in accepted:
                return False
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def resource_resume_compatible(previous_commit, current_commit, repository_root=REPOSITORY_ROOT):
    if previous_commit == current_commit:
        return True
    return (_has_resource_resume_source(previous_commit, repository_root)
            and _has_resource_resume_source(current_commit, repository_root))


@lru_cache(maxsize=32)
def committed_file_sha256(commit, relative_path, repository_root=REPOSITORY_ROOT):
    content = subprocess.check_output(["git", "show", f"{commit}:{relative_path}"], cwd=repository_root)
    return hashlib.sha256(content).hexdigest()
