"""TSB-AD 호환 VUS-PR 독립 채점기.

``l_max``는 VUS의 최대 buffer window 폭이다. 각 window ``l``에서
원 이상 구간 양쪽에는 ``floor(l / 2)`` 만큼의 연속 감쇠 buffer가 생긴다.
"""

import numpy as np


def _generate_thresholds(score: np.ndarray, n_thresholds: int) -> np.ndarray:
    """TSB-AD와 같이 내림차순 score의 균등한 순위 위치를 고른다."""
    score_sorted = np.sort(score)[::-1]
    indices = np.linspace(0, len(score_sorted) - 1, n_thresholds).astype(int)
    return score_sorted[indices]


def _predict_at_thresholds(score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """각 threshold의 binary prediction을 ``(threshold, time)``으로 반환한다."""
    return np.asarray(thresholds)[:, None] <= np.asarray(score)[None, :]


def _get_anomaly_ranges(label: np.ndarray) -> list[tuple[int, int]]:
    """Binary label의 연속 이상 구간을 양 끝 포함 인덱스로 반환한다."""
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(label):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            ranges.append((start, index - 1))
            start = None
    if start is not None:
        ranges.append((start, len(label) - 1))
    return ranges


def _extend_anomaly_ranges(label: np.ndarray, window: int) -> np.ndarray:
    """TSB-AD VUS의 sqrt 감쇠 buffer가 있는 연속 label을 만든다.

    ``window``는 양쪽 buffer 폭이 아니라 전체 window 폭이다. 따라서 한쪽
    buffer는 ``window // 2`` 칸이며, 겹치는 buffer는 1을 넘지 않게 자른다.
    """
    extended = np.asarray(label, dtype=float).copy()
    if window == 0:
        return extended

    half_window = window // 2
    for start, end in _get_anomaly_ranges(np.asarray(label, dtype=int)):
        right = np.arange(end + 1, min(end + half_window + 1, len(label)))
        left = np.arange(max(start - half_window, 0), start)
        extended[right] += np.sqrt(1 - (right - end) / window)
        extended[left] += np.sqrt(1 - (start - left) / window)
    return np.minimum(extended, 1.0)


def _merged_buffer_ranges(
    length: int,
    original_ranges: list[tuple[int, int]],
    window: int,
) -> list[tuple[int, int]]:
    """겹치는 VUS buffer를 하나의 range로 합친다."""
    half_window = window // 2
    first_start, _ = original_ranges[0]
    range_start = max(first_start - half_window, 0)
    ranges: list[tuple[int, int]] = []

    for index in range(len(original_ranges) - 1):
        _, end = original_ranges[index]
        next_start, _ = original_ranges[index + 1]
        if end + half_window < next_start - half_window:
            ranges.append((range_start, end + half_window))
            range_start = next_start - half_window

    _, last_end = original_ranges[-1]
    ranges.append((range_start, min(last_end + half_window, length - 1)))
    return ranges


def _range_precision_recall(
    prediction: np.ndarray,
    original_label: np.ndarray,
    original_ranges: list[tuple[int, int]],
    support_ranges: list[tuple[int, int]],
    window: int,
) -> tuple[float, float]:
    """한 threshold에서 TSB-AD의 range precision/recall을 계산한다."""
    extended_label = _extend_anomaly_ranges(original_label, window)
    buffered_ranges = _merged_buffer_ranges(
        len(original_label), original_ranges, window,
    )
    adjusted_label = extended_label.copy()
    existence_count = 0

    # Buffer는 prediction이 있는 위치에서만 보너스로 작동한다.
    for start, end in buffered_ranges:
        adjusted_label[start:end + 1] *= prediction[start:end + 1]
        if prediction[start:end + 1].any():
            existence_count += 1

    # 실제 이상 구간은 항상 원래의 완전한 positive label로 유지한다.
    for start, end in original_ranges:
        adjusted_label[start:end + 1] = 1.0

    true_positive = 0.0
    adjusted_positive = 0.0
    for start, end in support_ranges:
        true_positive += np.dot(
            adjusted_label[start:end + 1], prediction[start:end + 1],
        )
        adjusted_positive += np.sum(adjusted_label[start:end + 1])

    predicted_positive = int(np.sum(prediction))
    precision = 0.0 if predicted_positive == 0 else true_positive / predicted_positive
    positive_mass = (float(np.sum(original_label)) + adjusted_positive) / 2
    recall = min(true_positive / positive_mass, 1.0)
    recall *= existence_count / len(buffered_ranges)
    return float(precision), float(recall)


def _calculate_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """TSB-AD와 같은 step-wise range PR 적분을 수행한다."""
    if len(precisions) != len(recalls):
        raise ValueError("precision과 recall의 길이가 같아야 합니다.")
    if len(precisions) == 0:
        return 0.0
    recalls_with_origin = np.concatenate(([0.0], np.asarray(recalls, dtype=float)))
    return float(np.dot(
        recalls_with_origin[1:] - recalls_with_origin[:-1],
        np.asarray(precisions, dtype=float),
    ))


def _calculate_ap_for_window(
    score: np.ndarray,
    label: np.ndarray,
    window: int,
    n_thresholds: int = 250,
    *,
    support_window: int | None = None,
    predictions: np.ndarray | None = None,
) -> float:
    """한 buffer window의 TSB-AD 호환 range AP를 계산한다."""
    if support_window is None:
        support_window = window
    original_ranges = _get_anomaly_ranges(label)
    support_ranges = _merged_buffer_ranges(
        len(label), original_ranges, support_window,
    )
    if predictions is None:
        thresholds = _generate_thresholds(score, n_thresholds)
        predictions = _predict_at_thresholds(score, thresholds)

    precisions = np.empty(n_thresholds, dtype=float)
    recalls = np.empty(n_thresholds, dtype=float)
    for index, prediction in enumerate(predictions):
        precisions[index], recalls[index] = _range_precision_recall(
            prediction, label, original_ranges, support_ranges, window,
        )
    return _calculate_ap(precisions, recalls)


def _validate_inputs(
    score: np.ndarray,
    label: np.ndarray,
    l_max: int,
    n_thresholds: int,
) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=float)
    raw_label = np.asarray(label)
    if score.ndim != 1 or raw_label.ndim != 1:
        raise ValueError("score와 label은 1차원 배열이어야 합니다.")
    if len(score) != len(raw_label):
        raise ValueError("score와 label의 길이가 같아야 합니다.")
    if len(score) == 0:
        raise ValueError("score와 label은 비어 있을 수 없습니다.")
    if not np.all(np.isfinite(score)):
        raise ValueError("score에는 NaN 또는 Inf를 넣을 수 없습니다.")
    if not np.all(np.isin(raw_label, [0, 1])):
        raise ValueError("label은 0과 1로 이루어진 binary label이어야 합니다.")
    if not np.any(raw_label == 1):
        raise ValueError("VUS-PR에는 최소 하나의 이상 label이 필요합니다.")
    if not isinstance(l_max, (int, np.integer)) or isinstance(l_max, bool) or l_max < 0:
        raise ValueError("l_max는 0 이상의 정수여야 합니다.")
    if (
        not isinstance(n_thresholds, (int, np.integer))
        or isinstance(n_thresholds, bool)
        or n_thresholds <= 0
    ):
        raise ValueError("n_thresholds는 0보다 큰 정수여야 합니다.")
    return score, raw_label.astype(int)


def vus_pr(
    score: np.ndarray,
    label: np.ndarray,
    l_max: int,
    n_thresholds: int = 250,
) -> float:
    """VUS-PR을 계산한다.

    ``l_max``는 최대 buffer window 폭이다. VUS는 0부터 ``l_max``까지의
    모든 정수 window에서 계산한 range AP의 산술평균이다.
    """
    score, label = _validate_inputs(score, label, l_max, n_thresholds)
    thresholds = _generate_thresholds(score, n_thresholds)
    predictions = _predict_at_thresholds(score, thresholds)
    aps = [
        _calculate_ap_for_window(
            score,
            label,
            window,
            n_thresholds,
            support_window=l_max,
            predictions=predictions,
        )
        for window in range(l_max + 1)
    ]
    return float(np.mean(aps))
