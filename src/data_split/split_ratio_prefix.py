"""현재 정상 prefix 안에서 fit과 validation을 시간순으로 나눈다."""

import numpy

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS


class InsufficientPrefixError(ValueError):
    """현재 prefix가 모델의 사전 길이 조건을 만족하지 않는다."""


def compute_prefix_counts(total_length: int, ratio_percent: int) -> tuple[int, int, int]:
    """`floor(N*q/100)` prefix와 그 안의 `floor(0.8*A)`를 계산한다."""
    if ratio_percent not in SUPPORTED_RATIO_PERCENTS:
        raise ValueError(
            f"ratio는 {SUPPORTED_RATIO_PERCENTS} 중 하나여야 한다: {ratio_percent!r}"
        )
    if not isinstance(total_length, int) or total_length < 0:
        raise ValueError(f"total_length는 0 이상의 정수여야 한다: {total_length!r}")
    available_count = total_length * ratio_percent // 100
    fit_count = available_count * 80 // 100
    return available_count, fit_count, available_count - fit_count


def split_ratio_prefix(
    normal_training: numpy.ndarray,
    ratio_percent: int,
    min_fit_length: int = 1,
    min_validation_length: int = 1,
) -> tuple[numpy.ndarray, numpy.ndarray, dict]:
    """현재 q-prefix만 사용해 앞 80% fit과 뒤 20% validation을 반환한다."""
    normal_training = numpy.asarray(normal_training)
    if normal_training.ndim != 2 or not len(normal_training):
        raise ValueError("normal_training은 비어 있지 않은 2차원 배열이어야 한다")
    if min_fit_length < 0 or min_validation_length < 0:
        raise ValueError("최소 길이는 0 이상이어야 한다")

    total_length = len(normal_training)
    available_count, fit_count, validation_count = compute_prefix_counts(
        total_length, ratio_percent,
    )
    if fit_count < min_fit_length or validation_count < min_validation_length:
        raise InsufficientPrefixError(
            f"prefix 길이 하한 미달: N={total_length}, ratio={ratio_percent}, "
            f"available={available_count}, fit={fit_count}, "
            f"validation={validation_count}, fit 하한={min_fit_length}, "
            f"validation 하한={min_validation_length}"
        )

    split = {
        "normal_range": (0, total_length),
        "available_range": (0, available_count),
        "fit_range": (0, fit_count),
        "validation_range": (fit_count, available_count),
        "unseen_range": (available_count, total_length),
        "ratio_percent": ratio_percent,
    }
    return normal_training[:fit_count], normal_training[fit_count:available_count], split
