"""봉인한 Dev18 manifest 한 행을 이질 다변량 원시 입력으로 읽는다."""

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy
import pandas

from src.common.execution_identity import file_sha256


TRAIN_BOUNDARY_PATTERN = re.compile(r"_tr_(\d+)_")
SERIES_PATTERN = re.compile(r"^\d{2}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FIELDS = {
    "series", "family", "source_directory", "name", "training_boundary",
    "row_count", "feature_count", "feature_names_sha256", "size_bytes", "sha256",
}


def _validate_manifest_entry(manifest_entry) -> dict:
    if not isinstance(manifest_entry, Mapping):
        raise ValueError("Dev18 manifest entry는 mapping이어야 한다")
    if not REQUIRED_FIELDS <= manifest_entry.keys():
        raise ValueError("Dev18 manifest entry에 필수 필드가 없다")

    entry = dict(manifest_entry)
    if not isinstance(entry["series"], str) or not SERIES_PATTERN.fullmatch(entry["series"]):
        raise ValueError("Dev18 series는 두 자리 문자열이어야 한다")
    if not isinstance(entry["family"], str) or not entry["family"].strip():
        raise ValueError("Dev18 family는 비어 있지 않은 문자열이어야 한다")
    if (
        not isinstance(entry["source_directory"], str)
        or not entry["source_directory"].strip()
        or Path(entry["source_directory"]).is_absolute()
    ):
        raise ValueError("Dev18 source_directory는 상대 경로여야 한다")
    if (
        not isinstance(entry["name"], str)
        or not entry["name"]
        or Path(entry["name"]).name != entry["name"]
    ):
        raise ValueError("Dev18 name은 경로가 없는 파일명이어야 한다")
    if type(entry["training_boundary"]) is not int or entry["training_boundary"] <= 0:
        raise ValueError("Dev18 training_boundary는 양의 정수여야 한다")
    if type(entry["row_count"]) is not int or entry["row_count"] <= 0:
        raise ValueError("Dev18 row_count는 양의 정수여야 한다")
    if type(entry["feature_count"]) is not int or entry["feature_count"] <= 0:
        raise ValueError("Dev18 feature_count는 양의 정수여야 한다")
    if (
        not isinstance(entry["feature_names_sha256"], str)
        or not SHA256_PATTERN.fullmatch(entry["feature_names_sha256"])
    ):
        raise ValueError("Dev18 feature_names_sha256은 64자리 소문자 hex여야 한다")
    if type(entry["size_bytes"]) is not int or entry["size_bytes"] <= 0:
        raise ValueError("Dev18 size_bytes는 양의 정수여야 한다")
    if not isinstance(entry["sha256"], str) or not SHA256_PATTERN.fullmatch(entry["sha256"]):
        raise ValueError("Dev18 sha256은 64자리 소문자 hex여야 한다")

    boundary_match = TRAIN_BOUNDARY_PATTERN.search(entry["name"])
    if boundary_match is None:
        raise ValueError(f"파일명에서 tr_ 경계를 찾지 못했다: {entry['name']}")
    if int(boundary_match.group(1)) != entry["training_boundary"]:
        raise ValueError("파일명 경계와 manifest training_boundary가 다르다")
    return entry


def _resolve_input_path(entry: dict, data_root) -> Path:
    root = Path(data_root).resolve()
    path = (root / entry["source_directory"] / entry["name"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Dev18 입력 경로가 data root 밖을 가리킨다") from error
    if not path.is_file():
        raise FileNotFoundError(f"Dev18 입력 파일이 없다: {path}")
    return path


def load_dev18_registered_inputs(manifest_entry, data_root) -> dict:
    """Dev18 CSV의 label을 제외하고 정상 prefix와 test session을 반환한다."""
    entry = _validate_manifest_entry(manifest_entry)
    path = _resolve_input_path(entry, data_root)
    size_bytes = path.stat().st_size
    if size_bytes != entry["size_bytes"]:
        raise ValueError(
            f"입력 파일 크기가 manifest와 다르다: {path.name} "
            f"{size_bytes} != {entry['size_bytes']}"
        )
    sha256 = file_sha256(path)
    if sha256 != entry["sha256"]:
        raise ValueError(f"입력 파일 SHA-256이 manifest와 다르다: {path.name}")

    header = pandas.read_csv(path, nrows=0)
    if "Label" not in header.columns:
        raise ValueError(f"Label 열이 없다: {path.name}")
    feature_names = tuple(column for column in header.columns if column != "Label")
    if not feature_names:
        raise ValueError(f"feature 열이 없다: {path.name}")
    if len(feature_names) != entry["feature_count"]:
        raise ValueError(
            f"feature 수가 manifest와 다르다: {len(feature_names)} != "
            f"{entry['feature_count']}"
        )
    feature_names_sha256 = hashlib.sha256(json.dumps(
        list(feature_names), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    if feature_names_sha256 != entry["feature_names_sha256"]:
        raise ValueError("feature 순서가 manifest와 다르다")
    try:
        features = pandas.read_csv(path, usecols=feature_names).to_numpy(
            dtype=numpy.float32
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"feature 값은 숫자여야 한다: {path.name}") from error
    if not numpy.isfinite(features).all():
        raise ValueError(f"feature 값에 비유한 값이 있다: {path.name}")
    if len(features) != entry["row_count"]:
        raise ValueError(
            f"행 수가 manifest와 다르다: {len(features)} != {entry['row_count']}"
        )

    boundary = entry["training_boundary"]
    if not 0 < boundary < len(features):
        raise ValueError(f"tr_ 경계가 행 범위를 벗어났다: {boundary}/{len(features)}")
    return {
        "feature_names": feature_names,
        "normal_training": features[:boundary],
        "test_sessions": (features[boundary:],),
        "family": entry["family"],
        "series": entry["series"],
        "source_ranges": {
            "source": path.name,
            "source_directory": entry["source_directory"],
            "normal_training": (0, boundary),
            "test_sessions": ((boundary, len(features)),),
        },
        "input_identity": {
            "name": path.name,
            "source_directory": entry["source_directory"],
            "size_bytes": size_bytes,
            "sha256": sha256,
        },
    }
