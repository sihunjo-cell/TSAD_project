"""HAI 23.05의 train 4세션과 test 2세션을 GDN 입력으로 읽는다.

CSV 한 파일을 연속 세션 하나로 보며 파일 사이에는 window를 만들지 않는다.
timestamp를 빼면 센서 86열이다.
"""

from pathlib import Path

import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler

from src.common.experiment_config import load_dataset_ratios
from src.data_split.take_training_prefix import take_training_prefix
from src.data_split.validate_labels import validate_binary_labels
from src.data_split.validation_split import validation_split


TRAIN_FILENAMES = tuple(f"hai-train{session}.csv" for session in range(1, 5))
TEST_FILENAMES = tuple(f"hai-test{session}.csv" for session in range(1, 3))
LABEL_FILENAMES = tuple(f"label-test{session}.csv" for session in range(1, 3))
ALLOWED_RATIO_PERCENTS = load_dataset_ratios("HAI")

# 본 HAI 설계의 두 시간 조건. condition 1에는 train1, condition 2에는
# train1·train2만 사용할 수 있으며, 뒤 세션은 어떤 변환 통계에도 쓰지 않는다.
TEMPORAL_CONDITIONS = {
    1: {"train_filenames": ("hai-train1.csv",), "test_filename": "hai-test1.csv", "label_filename": "label-test1.csv"},
    2: {"train_filenames": ("hai-train1.csv", "hai-train2.csv"), "test_filename": "hai-test2.csv", "label_filename": "label-test2.csv"},
}


def read_feature_names(csv_path: Path) -> tuple[str, ...]:
    columns = tuple(pandas.read_csv(csv_path, nrows=0).columns)
    if not columns or columns[0] != "timestamp":
        raise ValueError(f"첫 열이 timestamp가 아니다: {csv_path.name}")
    feature_names = columns[1:]
    if len(feature_names) != 86:
        raise ValueError(f"HAI 센서 열은 86개여야 한다: {csv_path.name}={len(feature_names)}개")
    return feature_names


def read_feature_array(csv_path: Path, expected_names: tuple[str, ...]) -> numpy.ndarray:
    if read_feature_names(csv_path) != expected_names:
        raise ValueError(f"센서 열 또는 순서가 train1과 다르다: {csv_path.name}")
    frame = pandas.read_csv(
        csv_path,
        usecols=list(expected_names),
        dtype={name: numpy.float32 for name in expected_names},
    )
    features = frame.loc[:, expected_names].to_numpy(copy=False)
    if not numpy.isfinite(features).all():
        raise ValueError(f"센서 값에 결측 또는 비유한 값이 있다: {csv_path.name}")
    return features


def read_test_session(
    test_path: Path,
    label_path: Path,
    feature_names: tuple[str, ...],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    if read_feature_names(test_path) != feature_names:
        raise ValueError(f"센서 열 또는 순서가 train1과 다르다: {test_path.name}")
    test_frame = pandas.read_csv(
        test_path,
        dtype={"timestamp": "string", **{name: numpy.float32 for name in feature_names}},
    )
    label_frame = pandas.read_csv(label_path, dtype={"timestamp": "string"})
    if tuple(label_frame.columns) != ("timestamp", "label"):
        raise ValueError(f"라벨 열은 timestamp·label이어야 한다: {label_path.name}")
    if not test_frame["timestamp"].equals(label_frame["timestamp"]):
        raise ValueError(f"test와 label timestamp가 다르다: {test_path.name}")

    features = test_frame.loc[:, feature_names].to_numpy(copy=False)
    if not numpy.isfinite(features).all():
        raise ValueError(f"센서 값에 결측 또는 비유한 값이 있다: {test_path.name}")
    labels = validate_binary_labels(label_frame["label"], len(features), label_path.name)
    return features, labels


def load_hai_sessions(
    dataset_dir: str | Path,
    ratio_percent: int,
    val_fraction: float,
    min_train_length: int,
) -> dict:
    """세션별로 자른 뒤 네 train 부분에 scaler 하나를 fit한다."""
    if ratio_percent not in ALLOWED_RATIO_PERCENTS:
        raise ValueError(f"HAI ratio는 {ALLOWED_RATIO_PERCENTS}만 허용한다: {ratio_percent}")
    dataset_dir = Path(dataset_dir)
    feature_names = read_feature_names(dataset_dir / TRAIN_FILENAMES[0])
    train_sessions = []
    validation_sessions = []
    session_splits = []

    for filename in TRAIN_FILENAMES:
        features = read_feature_array(dataset_dir / filename, feature_names)
        training_prefix, split_info = take_training_prefix(features, ratio_percent / 100)
        train_part, validation_part = validation_split(
            training_prefix,
            val_fraction=val_fraction,
            min_train_length=min_train_length,
            split_info=split_info,
        )
        train_sessions.append(train_part)
        validation_sessions.append(validation_part)
        session_splits.append({
            "source": filename,
            "original_train_range": (0, len(features)),
            "kept_train_range": (0, split_info["kept_length"]),
            "train_range": (0, len(train_part)),
            "validation_range": (len(train_part), split_info["kept_length"]),
        })

    # partial_fit으로 네 세션을 복사 없이 같은 범위에 맞춘다.
    scaler = MinMaxScaler(copy=False)
    for train_part in train_sessions:
        scaler.partial_fit(train_part)
    for index, (train_part, validation_part) in enumerate(zip(train_sessions, validation_sessions)):
        train_sessions[index] = scaler.transform(train_part)
        validation_sessions[index] = scaler.transform(validation_part)

    test_sessions = []
    test_labels = []
    for test_filename, label_filename in zip(TEST_FILENAMES, LABEL_FILENAMES):
        test_part, labels = read_test_session(
            dataset_dir / test_filename,
            dataset_dir / label_filename,
            feature_names,
        )
        test_sessions.append(scaler.transform(test_part))
        test_labels.append(labels)

    return {
        "feature_names": feature_names,
        "train_sessions": tuple(train_sessions),
        "validation_sessions": tuple(validation_sessions),
        "test_sessions": tuple(test_sessions),
        "test_labels": tuple(test_labels),
        "session_splits": tuple(session_splits),
    }


def load_hai_temporal_condition(
    dataset_dir: str | Path,
    condition: int,
    val_fraction: float,
    min_train_length: int,
) -> dict:
    """HAI의 사전 고정 시간 조건 하나를 읽는다.

    condition 1은 train1→test1, condition 2는 train1+train2→test2다.
    각 사용 가능 train 세션의 마지막 20%만 정상 validation으로 분리하며,
    scaler는 해당 조건의 train 부분에만 fit한다.
    """
    if condition not in TEMPORAL_CONDITIONS:
        raise ValueError(f"HAI temporal condition은 {tuple(TEMPORAL_CONDITIONS)} 중 하나여야 한다.")
    dataset_dir = Path(dataset_dir)
    specification = TEMPORAL_CONDITIONS[condition]
    feature_names = read_feature_names(dataset_dir / TRAIN_FILENAMES[0])
    train_sessions, validation_sessions, session_splits = [], [], []
    for filename in specification["train_filenames"]:
        features = read_feature_array(dataset_dir / filename, feature_names)
        split_info = {"T": len(features), "condition": condition, "source": filename}
        train_part, validation_part = validation_split(
            features,
            val_fraction=val_fraction,
            min_train_length=min_train_length,
            split_info=split_info,
        )
        train_sessions.append(train_part)
        validation_sessions.append(validation_part)
        session_splits.append({
            "source": filename,
            "original_train_range": (0, len(features)),
            "train_range": (0, len(train_part)),
            "validation_range": (len(train_part), len(features)),
        })

    scaler = MinMaxScaler(copy=False)
    for train_part in train_sessions:
        scaler.partial_fit(train_part)
    train_sessions = [scaler.transform(part) for part in train_sessions]
    validation_sessions = [scaler.transform(part) for part in validation_sessions]
    test_part, test_label = read_test_session(
        dataset_dir / specification["test_filename"],
        dataset_dir / specification["label_filename"],
        feature_names,
    )
    return {
        "condition": condition,
        "feature_names": feature_names,
        "train_sessions": tuple(train_sessions),
        "validation_sessions": tuple(validation_sessions),
        "test_sessions": (scaler.transform(test_part),),
        "test_labels": (test_label,),
        "session_splits": tuple(session_splits),
    }
