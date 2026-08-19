"""점수 파일과 metadata를 읽고 채점기 입력 계약을 검증한다."""

import json
from pathlib import Path

import numpy as np

from src.common.naming import parse_score_filename


REQUIRED_METADATA_FIELDS = (
    "window_size",
    "test_length",
    "score_length",
    "label_slice",
)


def load_score_file(score_path: str | Path) -> tuple[np.ndarray, dict]:
    """점수 배열과 파일명 정보를 읽고 기본 입력 계약을 검증한다.

    Parameters
    ----------
    score_path:
        .npy 점수 파일 경로.

    Returns
    -------
    scores:
        1차원 집계 점수 배열.
    file_info:
        score filename에서 파싱한 정보.

    Raises
    ------
    ValueError
        파일 형식이나 점수 배열이 규약과 다를 때.
    FileNotFoundError
        점수 파일이 존재하지 않을 때.
    """
    score_path = Path(score_path)

    if not score_path.is_file():
        raise FileNotFoundError(f"점수 파일이 없다: {score_path}")

    # 1. 파일명 규약 확인
    file_info = parse_score_filename(score_path.name)

    # 2. 본 채점 대상인지 확인
    if file_info["channels"]:
        raise ValueError(
            f"__channels 보조 배열은 본 채점 입력으로 사용할 수 없다: "
            f"{score_path.name}"
        )

    if file_info["norm_kind"] != "trainnorm":
        raise ValueError(
            f"본 채점은 trainnorm만 사용한다: {score_path.name}"
        )

    # 3. numpy 배열 읽기
    scores = np.asarray(np.load(score_path), dtype=float)

    # 4. 본 채점 점수는 1차원 집계본이어야 함
    if scores.ndim != 1:
        raise ValueError(
            f"본 채점 점수는 1차원 배열이어야 한다: "
            f"shape={scores.shape}, file={score_path.name}"
        )

    return scores, file_info


def load_score_metadata(score_path: str | Path) -> dict:
    """점수 파일과 대응하는 .meta.json을 읽고 metadata 계약을 검증한다."""
    score_path = Path(score_path)

    if not score_path.is_file():
        raise FileNotFoundError(f"점수 파일이 없다: {score_path}")

    metadata_path = score_path.with_suffix(".meta.json")

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"점수 파일의 metadata가 없다: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"metadata는 JSON object여야 한다: {metadata_path}"
        )

    # 필수 metadata 필드 확인
    missing_fields = [
        field
        for field in REQUIRED_METADATA_FIELDS
        if field not in metadata
    ]

    if missing_fields:
        raise ValueError(
            f"metadata에 필수 필드가 없다: {missing_fields}"
        )

    return metadata


def validate_score_metadata(
    scores: np.ndarray,
    metadata: dict,
) -> None:
    """점수 배열과 metadata의 길이/라벨 범위를 검증한다.

    문서와 실제 배열이 다르면 임의로 보정하지 않고 오류를 발생시킨다.
    """
    score_length = int(metadata["score_length"])

    if len(scores) != score_length:
        raise ValueError(
            "score_length가 실제 배열과 다르다: "
            f"metadata={score_length}, actual={len(scores)}"
        )

    window_size = int(metadata["window_size"])
    test_length = int(metadata["test_length"])

    if window_size < 0:
        raise ValueError(
            f"window_size는 음수일 수 없다: {window_size}"
        )

    if test_length < 0:
        raise ValueError(
            f"test_length는 음수일 수 없다: {test_length}"
        )

    label_slice = metadata["label_slice"]

    if (
        not isinstance(label_slice, list)
        or len(label_slice) != 2
    ):
        raise ValueError(
            "label_slice는 [start, end] 형식이어야 한다: "
            f"{label_slice!r}"
        )

    start, end = label_slice

    if not isinstance(start, int):
        raise ValueError(
            f"label_slice의 시작값은 정수여야 한다: {label_slice!r}"
        )

    if end is not None and not isinstance(end, int):
        raise ValueError(
            f"label_slice의 끝값은 정수 또는 null이어야 한다: "
            f"{label_slice!r}"
        )

    if start < 0:
        raise ValueError(
            f"label_slice 시작값은 음수일 수 없다: {label_slice!r}"
        )

    if end is not None and end < start:
        raise ValueError(
            f"label_slice 범위가 잘못되었다: {label_slice!r}"
        )


def load_and_validate_score(
    score_path: str | Path,
) -> tuple[np.ndarray, dict, dict]:
    """점수 배열, filename 정보, metadata를 모두 읽고 검증한다."""
    scores, file_info = load_score_file(score_path)
    metadata = load_score_metadata(score_path)

    validate_score_metadata(scores, metadata)

    return scores, file_info, metadata
