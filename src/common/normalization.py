"""채널별 점수를 `(score - median) / (|IQR| + epsilon)`으로 정규화한다.

통계 추정과 적용을 분리해 학습·validation 통계만 테스트 점수에 쓰도록 한다. epsilon은
설정에서 받아 기본값을 두지 않는다.
"""

import numpy


def estimate_median_iqr(scores: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """학습/검증 구간의 채널별 점수 (n_samples, n_features)에서 (median, iqr)을 추정한다."""
    scores = numpy.asarray(scores, dtype=float)
    median = numpy.median(scores, axis=0)
    iqr = numpy.percentile(scores, 75, axis=0) - numpy.percentile(scores, 25, axis=0)
    return median, iqr


def apply_median_iqr(
    scores: numpy.ndarray,
    median: numpy.ndarray,
    iqr: numpy.ndarray,
    epsilon: float,
) -> numpy.ndarray:
    """추정해 둔 `(median, iqr)`로 점수를 정규화한다."""
    scores = numpy.asarray(scores, dtype=float)
    return (scores - median) / (numpy.abs(iqr) + epsilon)
