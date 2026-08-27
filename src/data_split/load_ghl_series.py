"""GHL CSV 한 파일을 등록 실행기의 원시 입력으로 읽는다."""

import re
from pathlib import Path

import numpy
import pandas


TRAIN_BOUNDARY_PATTERN = re.compile(r"_tr_(\d+)_")
def load_ghl_registered_inputs(csv_path: str | Path) -> dict:
    """Registry executor에 원시 정상 training과 test feature만 넘긴다."""
    csv_path = Path(csv_path)
    boundary_match = TRAIN_BOUNDARY_PATTERN.search(csv_path.name)
    if boundary_match is None:
        raise ValueError(f"파일명에서 tr_ 경계를 찾지 못했다: {csv_path.name}")

    header = pandas.read_csv(csv_path, nrows=0)
    feature_names = tuple(column for column in header.columns if column != "Label")
    if len(feature_names) != 19:
        raise ValueError(f"GHL 센서 열은 19개여야 한다: {len(feature_names)}개")
    features = pandas.read_csv(
        csv_path, usecols=feature_names,
    ).to_numpy(dtype=numpy.float32)
    if not numpy.isfinite(features).all():
        raise ValueError(f"센서 값에 결측 또는 비유한 값이 있다: {csv_path.name}")

    train_boundary = int(boundary_match.group(1))
    if not 0 < train_boundary < len(features):
        raise ValueError(
            f"tr_ 경계가 행 범위를 벗어났다: {train_boundary}/{len(features)}"
        )
    return {
        "feature_names": feature_names,
        "normal_training": features[:train_boundary],
        "test_sessions": (features[train_boundary:],),
        "source_ranges": {
            "source": csv_path.name,
            "normal_training": (0, train_boundary),
            "test_sessions": ((train_boundary, len(features)),),
        },
    }
