"""앞자르기 분할 — 비율 p 조건의 학습 데이터는 시점 인덱스 [0, ceil(p*T)) 구간.

근거: docs/plan_v4.md 7-1(213행)의 조작적 정의.
T의 정의(결측 처리 후인지, 안정화 구간 제거 후인지)는 Manifest 확정 사항(E1·E4)이다.
이 모듈은 T를 계산하거나 전처리를 하지 않고, 받은 배열을 진리로 취급한다.
"""

import math
from fractions import Fraction

import numpy

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS

# 허용 비율(계획서 7-1: 100/50/20/10/5%). Fraction 으로 두는 이유: float 곱의
# 이진 반올림 먼지가 ceil 경계를 넘는 사고(예: 0.05*T 가 정수보다 1ulp 크게 계산)를
# 원천 차단하기 위해 ceil(p*T)를 유리수 정확 연산으로 수행한다.
ALLOWED_RATIO_FRACTIONS = {
    percent / 100: Fraction(percent, 100) for percent in SUPPORTED_RATIO_PERCENTS
}


def compute_kept_length(total_length: int, ratio: float) -> int:
    """ceil(ratio * total_length) — 계획서 7-1. back_trim_split 도 이 함수를 공유한다."""
    if ratio not in ALLOWED_RATIO_FRACTIONS:
        raise ValueError(
            f"ratio 는 {sorted(ALLOWED_RATIO_FRACTIONS)} 만 허용한다: {ratio!r}"
        )
    return math.ceil(ALLOWED_RATIO_FRACTIONS[ratio] * total_length)


def front_trim_split(train_array: numpy.ndarray, ratio: float) -> tuple[numpy.ndarray, dict]:
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
