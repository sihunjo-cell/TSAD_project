"""GHL CSV 한 파일을 GDN용 train·validation·test 세션으로 읽는다.

파일명의 `tr_` 값이 원 학습 경계이며 센서 19열만 모델 입력으로 쓴다.
"""

import re
from pathlib import Path

import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler

from src.common.experiment_config import load_dataset_ratios
from src.data_split.take_training_prefix import take_training_prefix
from src.data_split.validate_labels import validate_binary_labels
from src.data_split.validation_split import InsufficientTrainLengthError, validation_split


TRAIN_BOUNDARY_PATTERN = re.compile(r"_tr_(\d+)_")
ALLOWED_RATIO_PERCENTS = load_dataset_ratios("GHL")


def load_ghl_series(
    csv_path: str | Path,
    ratio_percent: int,
    val_fraction: float,
    min_train_length: int,
) -> dict:
    """GHL 학습 구간의 앞쪽 누적분을 남기고 train 부분에만 scaler를 fit한다."""
    if ratio_percent not in ALLOWED_RATIO_PERCENTS:
        raise ValueError(f"GHL ratio는 {ALLOWED_RATIO_PERCENTS}만 허용한다: {ratio_percent}")
    csv_path = Path(csv_path)
    boundary_match = TRAIN_BOUNDARY_PATTERN.search(csv_path.name)
    if boundary_match is None:
        raise ValueError(f"파일명에서 tr_ 경계를 찾지 못했다: {csv_path.name}")

    frame = pandas.read_csv(csv_path)
    if "Label" not in frame.columns:
        raise ValueError(f"Label 열이 없다: {csv_path.name}")
    feature_names = tuple(column for column in frame.columns if column != "Label")
    if len(feature_names) != 19:
        raise ValueError(f"GHL 센서 열은 19개여야 한다: {len(feature_names)}개")

    features = frame.loc[:, feature_names].to_numpy(dtype=numpy.float32)
    if not numpy.isfinite(features).all():
        raise ValueError(f"센서 값에 결측 또는 비유한 값이 있다: {csv_path.name}")
    labels = validate_binary_labels(frame["Label"], len(frame), csv_path.name)

    train_boundary = int(boundary_match.group(1))
    if not 0 < train_boundary < len(frame):
        raise ValueError(f"tr_ 경계가 행 범위를 벗어났다: {train_boundary}/{len(frame)}")

    fit_pool, validation_part = validation_split(
        features[:train_boundary],
        val_fraction=val_fraction,
        min_train_length=min_train_length,
        split_info={"T": train_boundary, "ratio": ratio_percent / 100},
    )
    train_part, split_info = take_training_prefix(fit_pool, ratio_percent / 100)
    if len(train_part) < min_train_length:
        raise InsufficientTrainLengthError(
            f"비율 적용 후 train 길이 하한 미달: fit pool={len(fit_pool)}, "
            f"ratio={ratio_percent}, train={len(train_part)}, 하한={min_train_length}"
        )
    test_part = features[train_boundary:]

    scaler = MinMaxScaler(copy=False).fit(train_part)
    train_part = scaler.transform(train_part)
    validation_part = scaler.transform(validation_part)
    test_part = scaler.transform(test_part)

    kept_start, kept_end = split_info["kept_index_range"]
    return {
        "feature_names": feature_names,
        "train_sessions": (train_part,),
        "validation_sessions": (validation_part,),
        "test_sessions": (test_part,),
        "test_labels": (labels[train_boundary:],),
        "session_splits": ({
            "source": csv_path.name,
            "original_train_range": (0, train_boundary),
            "fit_pool_range": (0, len(fit_pool)),
            "kept_train_range": (kept_start, kept_end),
            "train_range": (kept_start, kept_end),
            "validation_range": (len(fit_pool), train_boundary),
            "test_range": (train_boundary, len(frame)),
        },),
    }
