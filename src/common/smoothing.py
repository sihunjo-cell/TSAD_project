"""채널별 점수에 시간축 후행 이동평균을 적용한다.

현재 시점을 포함한 창을 쓰고 처음 `window-1`개 시점은 0으로 둔다.
"""

import numpy


def trailing_average_smoothing(scores: numpy.ndarray, window: int = 4) -> numpy.ndarray:
    """scalar 또는 채널 점수를 시간축 후행 평균한다."""
    scores = numpy.asarray(scores, dtype=float)
    if scores.ndim not in (1, 2):
        raise ValueError(f"입력은 1차원 또는 2차원이어야 한다: shape {scores.shape}")
    if window < 1:
        raise ValueError(f"window는 1 이상이어야 한다: {window}")

    smoothed = numpy.zeros_like(scores)
    for time_index in range(window - 1, scores.shape[0]):
        smoothed[time_index] = scores[time_index - window + 1 : time_index + 1].mean(axis=0)
    return smoothed
