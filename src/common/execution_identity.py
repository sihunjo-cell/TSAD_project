"""등록 실행의 데이터 역할과 봉인 파일 신원을 검증한다."""

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

import yaml


EXECUTION_IDENTITY_FIELDS = (
    "dataset_role", "split_role", "input_manifest_sha256",
    "final_policy_membership_sha256",
)
SPLIT_DATASETS = {
    "dev18_selection": ("development", "DEV18"),
    "ghl25_final": ("final", "GHL"),
    "train1_to_test1": ("final", "HAI"),
    "train1_train2_to_test2": ("final", "HAI"),
}
HAI_ROLE_FILES = {
    "train1_to_test1": {
        "normal_training_files": ["hai-train1.csv"],
        "test_files": ["hai-test1.csv"],
    },
    "train1_train2_to_test2": {
        "normal_training_files": ["hai-train1.csv", "hai-train2.csv"],
        "test_files": ["hai-test2.csv"],
    },
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_execution_identity(values: Mapping) -> dict:
    """누락과 development의 유효한 null을 구분해 네 필드만 반환한다."""
    if not isinstance(values, Mapping):
        raise ValueError("execution identity는 mapping이어야 한다")
    for field in EXECUTION_IDENTITY_FIELDS:
        if field not in values:
            raise ValueError(f"execution identity 필드가 빠졌다: {field}")

    dataset_role = values["dataset_role"]
    split_role = values["split_role"]
    try:
        expected_role, _ = SPLIT_DATASETS[split_role]
    except KeyError as error:
        raise ValueError(f"실행할 수 없는 split_role이다: {split_role!r}") from error
    if dataset_role != expected_role:
        raise ValueError(
            f"dataset_role과 split_role이 다르다: {dataset_role!r}, {split_role!r}"
        )
    manifest_sha = values["input_manifest_sha256"]
    if not isinstance(manifest_sha, str) or not SHA256_PATTERN.fullmatch(manifest_sha):
        raise ValueError("input_manifest_sha256은 64자리 소문자 hex여야 한다")

    policy_sha = values["final_policy_membership_sha256"]
    if dataset_role == "development":
        if policy_sha is not None:
            raise ValueError("development final_policy_membership_sha256은 null이어야 한다")
    elif not isinstance(policy_sha, str) or not SHA256_PATTERN.fullmatch(policy_sha):
        raise ValueError("final_policy_membership_sha256은 64자리 소문자 hex여야 한다")
    return {field: values[field] for field in EXECUTION_IDENTITY_FIELDS}


def expected_dataset_for_identity(values: Mapping) -> str:
    """검증된 split role에 대응하는 논리 dataset 이름을 반환한다."""
    identity = validate_execution_identity(values)
    return SPLIT_DATASETS[identity["split_role"]][1]


def load_input_manifest_role(
    path, dataset_role: str, split_role: str,
) -> tuple[dict, str]:
    """manifest의 실행 role과 데이터 참조를 검사해 내용과 바이트 SHA를 반환한다."""
    manifest_path = Path(path)
    try:
        serialized = manifest_path.read_bytes()
        manifest = yaml.safe_load(serialized.decode("utf-8"))
        expected_role, expected_dataset = SPLIT_DATASETS[split_role]
        role = manifest["roles"][split_role]
        dataset_files = manifest["datasets"][expected_dataset]["files"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"입력 manifest에 실행 role이 없다: {split_role!r}") from error
    if not isinstance(role, dict):
        raise ValueError(f"입력 manifest의 실행 role은 mapping이어야 한다: {split_role!r}")
    if dataset_role != expected_role or role.get("dataset") != expected_dataset:
        raise ValueError("입력 manifest의 role과 dataset mapping이 다르다")
    if not isinstance(dataset_files, list):
        raise ValueError(f"입력 manifest의 {expected_dataset} files는 목록이어야 한다")

    expected_files = HAI_ROLE_FILES.get(split_role)
    if expected_files is not None:
        for field, filenames in expected_files.items():
            if role.get(field) != filenames:
                raise ValueError(f"입력 manifest의 {split_role}.{field}가 다르다")
        registered_names = {
            item.get("name") for item in dataset_files if isinstance(item, dict)
        }
        referenced_names = {
            name for filenames in expected_files.values() for name in filenames
        }
        if not referenced_names <= registered_names:
            raise ValueError("HAI 실행 role이 미등록 파일을 참조한다")
    return manifest, hashlib.sha256(serialized).hexdigest()


def validate_input_manifest_role(path, dataset_role: str, split_role: str) -> str:
    """manifest의 실행 role과 데이터 참조를 검사하고 바이트 SHA를 반환한다."""
    return load_input_manifest_role(path, dataset_role, split_role)[1]
