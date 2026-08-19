"""전체 정상 학습 구간의 뒤쪽을 고정 validation으로 분리한다.

비율별 누적 구간을 만들기 전에 한 번만 나눈다. 시간 순서를 지키며 `val_fraction`과 최소
fit pool 길이는 호출자가 넘긴다. 입력이 너무 짧으면
임의로 늘리거나 버리지 않고 오류로 알린다.
"""

import math

import numpy


class InsufficientTrainLengthError(ValueError):
    """분할 후 train 길이가 호출자가 정한 하한보다 짧다."""


def validation_split(
    normal_training: numpy.ndarray,
    val_fraction: float,
    min_train_length: int,
    split_info: dict | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """전체 정상 구간의 뒤쪽 ceil(val_fraction * 길이)개를 validation으로 자른다.

    ceil 을 쓰는 이유: val_fraction > 0 이면 validation 이 반드시 1개 이상 존재해야
    early stopping(Loss/val 감시)이 성립한다.
    split_info: 예외 메시지에 원 길이와 요청 비율을 담기 위한 선택 인자.
    """
    normal_training = numpy.asarray(normal_training)
    normal_length = normal_training.shape[0]
    val_length = math.ceil(val_fraction * normal_length)
    train_length = normal_length - val_length

    if train_length < min_train_length:
        origin = split_info or {}
        raise InsufficientTrainLengthError(
            f"train 길이 하한 미달: T={origin.get('T', '미상')}, "
            f"ratio={origin.get('ratio', '미상')}, val_fraction={val_fraction}, "
            f"정상 구간 길이={normal_length}, fit pool 길이={train_length}, "
            f"val 길이={val_length}, 하한={min_train_length}"
        )

    return normal_training[:train_length], normal_training[train_length:]
