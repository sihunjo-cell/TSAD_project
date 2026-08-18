"""학습 배열의 앞에서 `ceil(p*T)`개 시점을 남긴다.

이 모듈은 전처리를 마친 배열을 그대로 받아 길이와 비율만 계산한다.
"""

import math
from fractions import Fraction

import numpy

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS

# Fraction으로 계산해 부동소수점 오차가 ceil 경계를 바꾸지 않게 한다.
ALLOWED_RATIO_FRACTIONS = {
    percent / 100: Fraction(percent, 100) for percent in SUPPORTED_RATIO_PERCENTS
}


def compute_kept_length(total_length: int, ratio: float) -> int:
    """`ceil(ratio * total_length)`를 정확히 계산한다."""
    if ratio not in ALLOWED_RATIO_FRACTIONS:
        raise ValueError(
            f"ratio 는 {sorted(ALLOWED_RATIO_FRACTIONS)} 만 허용한다: {ratio!r}"
        )
    return math.ceil(ALLOWED_RATIO_FRACTIONS[ratio] * total_length)


def take_training_prefix(train_array: numpy.ndarray, ratio: float) -> tuple[numpy.ndarray, dict]:
    """시점 축(axis 0) 앞에서부터 ceil(p*T)개를 남긴다. (배열, 로그용 dict) 반환."""
    train_array = numpy.asarray(train_array)
    total_length = train_array.shape[0]
    kept_length = compute_kept_length(total_length, ratio)
    split_info = {
        "T": total_length,
        "ratio": ratio,
        "kept_length": kept_length,
        "kept_index_range": (0, kept_length),  # 반열림 [0, kept_length)
    }
    return train_array[:kept_length], split_info
