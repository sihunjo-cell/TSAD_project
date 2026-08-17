"""GHL CSV 한 파일을 GDN용 train·validation·test 세션으로 읽는다.

근거: TheDatumOrg/TSB-AD `Datasets/File_List/TSB-AD-M.csv:33-57`,
`docs/manifest_draft.md` GHL 절, D-20. 파일명의 `tr_` 값이 원 학습 경계이며
센서 19열만 모델 입력으로 쓴다.
"""

import re
from pathlib import Path

import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler

from src.common.experiment_config import load_dataset_ratios
from src.data_split.back_trim_split import back_trim_split
from src.data_split.front_trim_split import front_trim_split
from src.data_split.validate_labels import validate_binary_labels
from src.data_split.validation_split import validation_split


TRAIN_BOUNDARY_PATTERN = re.compile(r"_tr_(\d+)_")
ALLOWED_RATIO_PERCENTS = load_dataset_ratios("GHL")
BACK_TRIM_RATIO_PERCENTS = load_dataset_ratios("GHL", ratio_key="back_trim_ratios")


def load_ghl_series(
    csv_path: str | Path,
    ratio_percent: int,
    val_fraction: float,
    min_train_length: int,
    trim_direction: str = "front",
) -> dict:
    """GHL 한 시계열을 지정한 방향으로 자르고 train 부분에만 scaler를 fit한다."""
    if ratio_percent not in ALLOWED_RATIO_PERCENTS:
        raise ValueError(f"GHL ratio는 {ALLOWED_RATIO_PERCENTS}만 허용한다: {ratio_percent}")
    if trim_direction == "back" and ratio_percent not in BACK_TRIM_RATIO_PERCENTS:
        raise ValueError(
            f"GHL back trim ratio는 {BACK_TRIM_RATIO_PERCENTS}만 허용한다: {ratio_percent}"
        )
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

    splitters = {"front": front_trim_split, "back": back_trim_split}
    if trim_direction not in splitters:
        raise ValueError(f"trim_direction은 front 또는 back이어야 한다: {trim_direction}")
    reduced_train, split_info = splitters[trim_direction](
        features[:train_boundary], ratio_percent / 100,
    )
    train_part, validation_part = validation_split(
        reduced_train,
        val_fraction=val_fraction,
        min_train_length=min_train_length,
        split_info=split_info,
    )
    test_part = features[train_boundary:]

    scaler = MinMaxScaler(copy=False).fit(train_part)  # D-20; GraGOD data_processing.py:54-70
    train_part = scaler.transform(train_part)
    validation_part = scaler.transform(validation_part)
    test_part = scaler.transform(test_part)

    train_length = len(train_part)
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
            "kept_train_range": (kept_start, kept_end),
            "train_range": (kept_start, kept_start + train_length),
            "validation_range": (kept_start + train_length, kept_end),
            "test_range": (train_boundary, len(frame)),
        },),
    }
