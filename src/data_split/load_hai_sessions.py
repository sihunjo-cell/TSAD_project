"""HAI 23.05의 train 4세션과 test 2세션을 GDN 입력으로 읽는다.

근거: icsdataset/hai `README.md:229-264,284-294`, `docs/manifest_draft.md`
HAI 절, D-17·D-20·D-21. CSV 한 파일이 연속 세션 하나이며 파일 사이에는
window를 만들지 않는다. timestamp를 빼면 센서 86열이다.
"""

from pathlib import Path

import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler

from src.data_split.front_trim_split import front_trim_split
from src.data_split.validation_split import validation_split


TRAIN_FILENAMES = tuple(f"hai-train{session}.csv" for session in range(1, 5))
TEST_FILENAMES = tuple(f"hai-test{session}.csv" for session in range(1, 3))
LABEL_FILENAMES = tuple(f"label-test{session}.csv" for session in range(1, 3))
ALLOWED_RATIO_PERCENTS = (10, 100)


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
    return features, label_frame["label"].to_numpy(copy=True)


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
        reduced_train, split_info = front_trim_split(features, ratio_percent / 100)
        train_part, validation_part = validation_split(
            reduced_train,
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

    # partial_fit은 네 train 부분을 연결해 fit한 MinMaxScaler와 같은 min·max를 얻고
    # 큰 HAI 배열의 임시 연결 복사본을 만들지 않는다(D-20).
    scaler = MinMaxScaler(copy=False)
    for train_part in train_sessions:
        scaler.partial_fit(train_part)
    for train_part, validation_part in zip(train_sessions, validation_sessions):
        scaler.transform(train_part)
        scaler.transform(validation_part)

    test_sessions = []
    test_labels = []
    for test_filename, label_filename in zip(TEST_FILENAMES, LABEL_FILENAMES):
        test_part, labels = read_test_session(
            dataset_dir / test_filename,
            dataset_dir / label_filename,
            feature_names,
        )
        scaler.transform(test_part)
        test_sessions.append(test_part)
        test_labels.append(labels)

    return {
        "feature_names": feature_names,
        "train_sessions": tuple(train_sessions),
        "validation_sessions": tuple(validation_sessions),
        "test_sessions": tuple(test_sessions),
        "test_labels": tuple(test_labels),
        "session_splits": tuple(session_splits),
    }
