"""누적 학습 구간의 뒤쪽을 validation으로 분리한다.

시간 순서를 지키며 `val_fraction`과 최소 학습 길이는 호출자가 넘긴다. 입력이 너무 짧으면
임의로 늘리거나 버리지 않고 오류로 알린다.
"""

import math

import numpy


class InsufficientTrainLengthError(ValueError):
    """분할 후 train 길이가 호출자가 정한 하한보다 짧다."""


def validation_split(
    training_prefix: numpy.ndarray,
    val_fraction: float,
    min_train_length: int,
    split_info: dict | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """누적 구간의 뒤쪽 ceil(val_fraction * 길이)개를 validation으로 자른다.

    ceil 을 쓰는 이유: val_fraction > 0 이면 validation 이 반드시 1개 이상 존재해야
    early stopping(Loss/val 감시)이 성립한다.
    split_info: take_training_prefix가 반환한 dict — 예외 메시지에 T·ratio를
    담기 위한 선택 인자.
    """
    training_prefix = numpy.asarray(training_prefix)
    prefix_length = training_prefix.shape[0]
    val_length = math.ceil(val_fraction * prefix_length)
    train_length = prefix_length - val_length

    if train_length < min_train_length:
        origin = split_info or {}
        raise InsufficientTrainLengthError(
            f"train 길이 하한 미달: T={origin.get('T', '미상')}, "
            f"ratio={origin.get('ratio', '미상')}, val_fraction={val_fraction}, "
            f"누적 구간 길이={prefix_length}, train 길이={train_length}, "
            f"val 길이={val_length}, 하한={min_train_length}"
        )

    return training_prefix[:train_length], training_prefix[train_length:]
