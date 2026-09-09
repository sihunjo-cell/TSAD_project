"""검사만 수정한 소스와 중단된 Dev18 실행의 호환성을 확인한다."""

import hashlib
import re
import subprocess
from functools import lru_cache
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_RESUME_SOURCE = "57e91eb8de2d130fab0fae437041beb8eccd73ee"
# 검토한 재개 연결 변경만 허용하며 다른 계산 코드 변경은 거부한다.
RESOURCE_RESUME_BLOBS = {
    "tests/checks/check_dev18_resources.py": "efc502ea0f307d887527662294b60a19c355a9ef",
    "tests/ghl_main/run_dev18_tuning.py": "f610da99a9ce6e922ec269909c32fbd08af7acf4",
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
            if expected is None or subprocess.check_output(
                ["git", "rev-parse", f"{commit}:{path}"], cwd=repository_root, text=True,
            ).strip() != expected:
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
