"""이상 라벨 배열의 공통 계약을 검증한다."""

import numpy


def validate_binary_labels(labels, expected_length: int, source: str) -> numpy.ndarray:
    try:
        numeric_labels = numpy.asarray(labels, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"라벨은 0 또는 1이어야 한다: {source}") from error
    if (
        numeric_labels.ndim != 1
        or len(numeric_labels) != expected_length
        or not numpy.isfinite(numeric_labels).all()
        or not numpy.isin(numeric_labels, (0, 1)).all()
    ):
        raise ValueError(f"라벨은 길이가 맞는 유한한 0 또는 1이어야 한다: {source}")
    return numeric_labels.astype(numpy.int8)
