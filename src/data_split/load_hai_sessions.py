"""HAI 23.05의 봉인된 두 실행에 필요한 원시 세션을 읽는다."""

from pathlib import Path

import numpy
import pandas
REGISTERED_SPLITS = {
    "train1_to_test1": (("hai-train1.csv",), ("hai-test1.csv",)),
    "train1_train2_to_test2": (
        ("hai-train1.csv", "hai-train2.csv"),
        ("hai-test2.csv",),
    ),
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
    if not len(features):
        raise ValueError(f"센서 세션이 비어 있다: {csv_path.name}")
    if not numpy.isfinite(features).all():
        raise ValueError(f"센서 값에 결측 또는 비유한 값이 있다: {csv_path.name}")
    return features


def load_hai_registered_inputs(dataset_dir: str | Path, split_role: str) -> dict:
    """봉인한 HAI 실행에 필요한 원시 feature 세션만 읽는다."""
    try:
        train_filenames, test_filenames = REGISTERED_SPLITS[split_role]
    except KeyError as error:
        raise ValueError(f"지원하지 않는 HAI split_role이다: {split_role!r}") from error

    dataset_dir = Path(dataset_dir)
    feature_names = read_feature_names(dataset_dir / train_filenames[0])
    train_sessions = tuple(
        read_feature_array(dataset_dir / filename, feature_names)
        for filename in train_filenames
    )
    test_sessions = tuple(
        read_feature_array(dataset_dir / filename, feature_names)
        for filename in test_filenames
    )

    def ranges(filenames, sessions):
        return tuple(
            {"source": filename, "range": (0, len(values))}
            for filename, values in zip(filenames, sessions)
        )

    return {
        "feature_names": feature_names,
        "split_role": split_role,
        "normal_training_sessions": train_sessions,
        "test_sessions": test_sessions,
        "source_ranges": {
            "normal_training_sessions": ranges(train_filenames, train_sessions),
            "test_sessions": ranges(test_filenames, test_sessions),
        },
    }
