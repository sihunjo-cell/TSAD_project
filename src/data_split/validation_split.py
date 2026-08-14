"""축소된 학습 구간 내부의 train/validation 분할.

근거: docs/plan_v4.md 7-2(218행) — validation 은 축소된 학습 구간 안에서만 자른다.
이 함수의 시그니처는 축소된 배열만 받으므로 원 학습 구간을 참조하는 leakage 가
구조적으로 불가능하다(5% 조건이 5%보다 많은 데이터를 보는 사고 차단).

분할 위치는 축소 구간의 뒤쪽 = validation. 시계열이므로 무작위 셔플 금지 —
시간 순서를 파괴하면 미래 정보가 학습 구간으로 새어 들어간다(leakage).

val_fraction 의 기본값을 코드에 두지 않는다 — 값(GraGOD 저자 기본 0.1, RECON [G])은
configs/gdn_hyperparams.yaml 의 val_size 소유이며 호출자가 읽어 넘긴다.

min_train_length 하한값 자체와 미달 시계열의 처리 규칙은 EDA(E1) 후 결정
사항이므로, 이 모듈은 판정·보고(예외)만 하고 처리하지 않는다.

T의 정의는 Manifest 확정 사항(E1·E4) — 이 모듈은 받은 배열을 진리로 취급한다.
"""

import math

import numpy


class InsufficientTrainLengthError(ValueError):
    """분할 후 train 길이가 하한 미달 — 처리 규칙은 EDA(E1) 후 결정, 여기서는 보고만."""


def validation_split(
    reduced_train_array: numpy.ndarray,
    val_fraction: float,
    min_train_length: int,
    split_info: dict | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """축소 구간의 뒤쪽 ceil(val_fraction * 길이)개를 validation 으로 자른다.

    ceil 을 쓰는 이유: val_fraction > 0 이면 validation 이 반드시 1개 이상 존재해야
    early stopping(Loss/val 감시)이 성립한다.
    split_info: front/back_trim_split 이 반환한 dict — 예외 메시지에 T·ratio 를
    담기 위한 선택 인자.
    """
    reduced_train_array = numpy.asarray(reduced_train_array)
    reduced_length = reduced_train_array.shape[0]
    val_length = math.ceil(val_fraction * reduced_length)
    train_length = reduced_length - val_length

    if train_length < min_train_length:
        origin = split_info or {}
        raise InsufficientTrainLengthError(
            f"train 길이 하한 미달: T={origin.get('T', '미상')}, "
            f"ratio={origin.get('ratio', '미상')}, val_fraction={val_fraction}, "
            f"축소 구간 길이={reduced_length}, train 길이={train_length}, "
            f"val 길이={val_length}, 하한={min_train_length}"
        )

    return reduced_train_array[:train_length], reduced_train_array[train_length:]
