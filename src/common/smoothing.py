"""시간축 후행 이동평균 smoothing.

근거: DECISIONS D-04·D-05 — d-ailin/GDN(main, 9853899d) evaluate.py:62-65 방식의
자체 구현. GraGOD smooth_scores 는 feature 축에 작용하므로 사용 금지
(VERIFICATION.md 의혹 2 = 참, d-ailin 대조 절 참조).

원본(evaluate.py:62-65, 1채널 기준):
    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])
즉 창 크기 4(현재 시점 포함 후행), 처음 window-1개 시점은 0.
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
