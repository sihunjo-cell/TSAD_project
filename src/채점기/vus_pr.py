import numpy as np

def _generate_thresholds(score, n_thresholds):
    """score에서 평가에 사용할 threshold를 생성한다."""

    score_sorted = np.sort(score)[::-1]

    indices = np.linspace(
        0,
        len(score_sorted) - 1,
        n_thresholds
    ).astype(int)

    thresholds = score_sorted[indices]

    return thresholds

def _predict_at_thresholds(score, thresholds):
    """
    각 threshold에서 binary prediction 생성
    """
    score = np.asarray(score)
    thresholds = np.asarray(thresholds)

    predictions = thresholds[:, None] <= score[None, :]

    return predictions


def _extend_anomaly_ranges(label, l):
    """
    anomaly range를 양쪽으로 l만큼 확장
    """
    label = np.asarray(label, dtype=int).copy()

    ranges = _get_anomaly_ranges(label)

    extended = np.zeros_like(label)

    for start, end in ranges:
        new_start = max(0, start - l)
        new_end = min(len(label) - 1, end + l)

        extended[new_start:new_end + 1] = 1

    return extended

def _calculate_precision_recall(prediction, label):
    """
    prediction과 label을 이용해 Precision / Recall 계산
    """

    prediction = np.asarray(prediction, dtype=int)
    label = np.asarray(label, dtype=int)

    # 예측된 anomaly 개수
    predicted_anomaly = np.sum(prediction)

    # 실제 anomaly 개수
    actual_anomaly = np.sum(label)

    # True Positive
    true_positive = np.sum((prediction == 1) & (label == 1))

    # Precision
    if predicted_anomaly == 0:
        precision = 0.0
    else:
        precision = true_positive / predicted_anomaly

    # Recall
    if actual_anomaly == 0:
        recall = 0.0
    else:
        recall = true_positive / actual_anomaly

    return precision, recall
def _calculate_ap(precisions, recalls):
    """
    Precision-Recall curve에서 AP 계산
    """

    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)

    if len(precisions) != len(recalls):
        raise ValueError("precision과 recall의 길이가 같아야 합니다.")

    if len(precisions) == 0:
        return 0.0

    # recall 순서대로 정렬
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]

    # Precision envelope
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # Recall 변화량에 따른 면적 계산
    ap = np.sum(
        (recalls[1:] - recalls[:-1]) * precisions[1:]
    )

    return float(ap)

def _calculate_ap_for_window(score, label, l, n_thresholds=250):
    """
    하나의 range/window 길이 l에 대해
    모든 threshold의 Precision / Recall을 계산하고 AP를 반환
    """

    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)

    # l에 맞게 anomaly range 확장
    extended_label = _extend_anomaly_ranges(label, l)

    # threshold 생성
    thresholds = _generate_thresholds(score, n_thresholds)

    precisions = []
    recalls = []

    # threshold마다 prediction 생성
    predictions = _predict_at_thresholds(score, thresholds)

    for prediction in predictions:
        precision, recall = _calculate_precision_recall(
            prediction,
            extended_label
        )

        precisions.append(precision)
        recalls.append(recall)

    # Precision-Recall curve의 AP 계산
    ap = _calculate_ap(
        np.array(precisions),
        np.array(recalls)
    )

    return ap

def _get_anomaly_ranges(label):
    """binary label에서 anomaly 구간을 (start, end) 형태로 추출한다."""

    ranges = []
    start = None

    for i, value in enumerate(label):
        if value == 1 and start is None:
            # anomaly 시작
            start = i

        elif value == 0 and start is not None:
            # anomaly 종료
            ranges.append((start, i - 1))
            start = None

    # 마지막까지 anomaly인 경우
    if start is not None:
        ranges.append((start, len(label) - 1))

    return ranges
def vus_pr(score, label, l_max, n_thresholds=250):
    """
    VUS-PR 계산

    Parameters
    ----------
    score : array-like
        Anomaly score
    label : array-like
        Ground truth binary labels (0 or 1)
    l_max : int
        Maximum range/window size
    n_thresholds : int
        Number of thresholds

    Returns
    -------
    float
        VUS-PR score
    """

    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=int)

    # 입력 검증
    if len(score) != len(label):
        raise ValueError("score와 label의 길이가 같아야 합니다.")

    if len(score) == 0:
        raise ValueError("score와 label은 비어 있을 수 없습니다.")

    if not np.all(np.isin(label, [0, 1])):
        raise ValueError("label은 0과 1로 이루어진 binary label이어야 합니다.")

    if l_max < 0:
        raise ValueError("l_max는 0 이상이어야 합니다.")

    if n_thresholds <= 0:
        raise ValueError("n_thresholds는 0보다 커야 합니다.")


    aps = []

    for l in range(l_max + 1):
     ap = _calculate_ap_for_window(
         score,
         label,
         l,
         n_thresholds
       )
     aps.append(ap)

    vus_pr_score = np.mean(aps)

    return float(vus_pr_score)