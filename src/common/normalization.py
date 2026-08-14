"""채널별 median·IQR 정규화.

근거: DECISIONS D-06 — 통계는 학습/검증 구간에서만 추정해 테스트 점수에
적용한다(추정과 적용을 함수로 분리한 이유). D-07 — epsilon 값(1e-2)은
configs/scoring_pipeline.yaml 소유이므로 이 모듈에 기본값을 두지 않는다.
수식은 RECON [D]의 d-ailin/GDN(main, 9853899d) evaluate.py:58-60 방식:
(delta - median) / (|iqr| + epsilon).
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
    """추정해 둔 (median, iqr)로 점수를 정규화한다. epsilon 기본값 없음(D-07)."""
    scores = numpy.asarray(scores, dtype=float)
    return (scores - median) / (numpy.abs(iqr) + epsilon)
