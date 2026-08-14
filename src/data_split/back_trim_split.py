"""뒷자르기 분할 — 앞자르기와 같은 길이를 뒤에서부터 취하는 통제군.

근거: docs/plan_v4.md 7-1(216행) — 통제군 GHL × {100/20/5} 병행.
kept_length 는 front_trim_split.compute_kept_length 를 그대로 공유한다:
같은 T·p에서 두 방향이 같은 길이를 다른 위치에서 취해야 통제 비교가 성립한다.
T의 정의는 Manifest 확정 사항(E1·E4) — 이 모듈은 받은 배열을 진리로 취급한다.
"""

import numpy

from src.data_split.front_trim_split import compute_kept_length


def back_trim_split(train_array: numpy.ndarray, ratio: float) -> tuple[numpy.ndarray, dict]:
    """시점 축(axis 0) 뒤에서부터 ceil(p*T)개를 남긴다. (배열, 로그용 dict) 반환."""
    train_array = numpy.asarray(train_array)
    total_length = train_array.shape[0]
    kept_length = compute_kept_length(total_length, ratio)
    split_info = {
        "T": total_length,
        "ratio": ratio,
        "kept_length": kept_length,
        "kept_index_range": (total_length - kept_length, total_length),  # 반열림
    }
    return train_array[total_length - kept_length :], split_info
