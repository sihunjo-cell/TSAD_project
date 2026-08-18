"""채널별 점수에 시간축 후행 이동평균을 적용한다.

현재 시점을 포함한 창을 쓰고 처음 `window-1`개 시점은 0으로 둔다.
"""

import numpy


def trailing_average_smoothing(scores: numpy.ndarray, window: int = 4) -> numpy.ndarray:
    """(n_samples, n_features) 점수를 채널별 독립으로 시간축 후행 평균한다."""
    scores = numpy.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"입력은 (n_samples, n_features) 2차원이어야 한다: shape {scores.shape}")

    smoothed = numpy.zeros_like(scores)
    for time_index in range(window - 1, scores.shape[0]):
        smoothed[time_index] = scores[time_index - window + 1 : time_index + 1].mean(axis=0)
    return smoothed
