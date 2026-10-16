"""`vus_pr.vus_pr`와 같은 값을 내는 빠른 VUS-PR.

원본은 threshold마다 window의 확장 label과 buffer 구간을 다시 만든다. 확장 label을 만드는
`_get_anomaly_ranges`는 시점마다 도는 Python 반복이라, test가 길고 ℓ_max가 크면 이것이 시간을
거의 다 쓴다. 여기서는 threshold와 무관한 값(확장 label, buffer·support 구간, 예측 양성 수,
label 합)을 window마다 또는 한 번만 만들고, 나머지는 원본과 같은 배열에 같은 순서로 계산한다.
그래서 결과가 비트 단위로 같다. 원본 파일은 Dev18 평가 증거에 SHA로 묶여 있어 고치지 않는다.
"""

import numpy as np

from src.채점기.vus_pr import (
    _calculate_ap,
    _extend_anomaly_ranges,
    _generate_thresholds,
    _get_anomaly_ranges,
    _merged_buffer_ranges,
    _predict_at_thresholds,
    _validate_inputs,
)


def fast_vus_pr(score: np.ndarray, label: np.ndarray, l_max: int, n_thresholds: int = 250) -> float:
    score, label = _validate_inputs(score, label, l_max, n_thresholds)
    predictions = _predict_at_thresholds(score, _generate_thresholds(score, n_thresholds))
    original_ranges = _get_anomaly_ranges(label)
    support_ranges = _merged_buffer_ranges(len(label), original_ranges, l_max)
    predicted_positives = [int(np.sum(prediction)) for prediction in predictions]
    label_mass = float(np.sum(label))
    aps = []
    for window in range(l_max + 1):
        extended_label = _extend_anomaly_ranges(label, window)
        buffered_ranges = _merged_buffer_ranges(len(label), original_ranges, window)
        precisions = np.empty(n_thresholds, dtype=float)
        recalls = np.empty(n_thresholds, dtype=float)
        for index, prediction in enumerate(predictions):
            adjusted_label = extended_label.copy()
            existence_count = 0
            for start, end in buffered_ranges:
                adjusted_label[start:end + 1] *= prediction[start:end + 1]
                if prediction[start:end + 1].any():
                    existence_count += 1
            for start, end in original_ranges:
                adjusted_label[start:end + 1] = 1.0
            true_positive = 0.0
            adjusted_positive = 0.0
            for start, end in support_ranges:
                true_positive += np.dot(adjusted_label[start:end + 1], prediction[start:end + 1])
                adjusted_positive += np.sum(adjusted_label[start:end + 1])
            predicted_positive = predicted_positives[index]
            precision = 0.0 if predicted_positive == 0 else true_positive / predicted_positive
            recall = min(true_positive / ((label_mass + adjusted_positive) / 2), 1.0)
            recall *= existence_count / len(buffered_ranges)
            precisions[index], recalls[index] = float(precision), float(recall)
        aps.append(_calculate_ap(precisions, recalls))
    return float(np.mean(aps))
