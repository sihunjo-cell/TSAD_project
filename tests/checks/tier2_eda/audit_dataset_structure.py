"""GHL과 HAI의 파일·세션 구조를 라벨 값 없이 점검한다."""

import re
from pathlib import Path

import numpy
import pandas


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GHL_PATTERN = re.compile(r"^(?P<archive>\d{3})_GHL_id_(?P<series>\d+)_Sensor_tr_(?P<train>\d+)_1st_(?P<first>\d+)\.csv$")


def resolve_dataset_dir(dataset: str) -> Path:
    name = "TSB-AD-M" if dataset == "GHL" else "HAI-23.05"
    direct = REPOSITORY_ROOT / name
    fallback = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / name
    for candidate in (direct, fallback):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{dataset} 데이터 폴더를 찾지 못했습니다.")


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as data_file:
        return max(sum(1 for _ in data_file) - 1, 0)


def inspect_dataset(dataset: str) -> tuple[list[dict], list[dict], tuple[str, ...]]:
    return inspect_ghl() if dataset == "GHL" else inspect_hai()


def inspect_ghl() -> tuple[list[dict], list[dict], tuple[str, ...]]:
    dataset_dir = resolve_dataset_dir("GHL")
    descriptors = []
    rows = []
    feature_names = None
    for path in sorted(dataset_dir.glob("*.csv")):
        match = GHL_PATTERN.fullmatch(path.name)
        if not match:
            continue
        parsed = {key: int(value) for key, value in match.groupdict().items()}
        columns = tuple(pandas.read_csv(path, nrows=0).columns)
        current_features = tuple(column for column in columns if column != "Label")
        if feature_names is None:
            feature_names = current_features
        elif set(current_features) != set(feature_names):
            raise ValueError(f"GHL 채널 구성이 다릅니다: {path.name}")
        total_length = count_csv_rows(path)
        descriptor = {
            "dataset": "GHL", "source": f"series_{parsed['series']:02d}", "path": path,
            "train_length": parsed["train"], "test_length": total_length - parsed["train"],
            "total_length": total_length, "series": parsed["series"],
            "feature_names": feature_names, "source_feature_names": current_features,
        }
        descriptors.append(descriptor)
        rows.append({
            "dataset": "GHL", "file_or_session": descriptor["source"], "ratio": "not_applicable",
            "stage": "normal", "file_name": path.name, "split": "train_and_test",
            "row_count": total_length, "feature_count": len(current_features),
            "feature_names": "|".join(current_features), "timestamp_column": "",
            "label_column": "Label", "sampling_interval_seconds": "not_available",
            "sampling_interval_status": "not_available", "irregular_gap_count": "not_available",
            "duplicate_timestamp_count": "not_available", "missing_value_count": numpy.nan,
            "nonfinite_value_count": numpy.nan, "train_start": 0, "train_end_exclusive": parsed["train"],
            "validation_start": "전체 정상 구간의 마지막 10%", "test_start": parsed["train"],
            "test_end_exclusive": total_length, "boundary_rule": "파일별 경계 유지",
            "normal_sample_count": parsed["train"], "anomaly_sample_count": numpy.nan,
            "label_values_used": False,
            "column_order_matches_reference": current_features == feature_names,
            "channel_set_matches_reference": set(current_features) == set(feature_names),
            "model_input_order_canonicalized": True,
            "input_reordered": current_features != feature_names,
            "timestamp_start": "not_available", "timestamp_end": "not_available",
            "timestamp_reversal_count": "not_available",
        })
    if len(descriptors) != 25 or feature_names is None:
        raise ValueError(f"GHL 파일은 25개여야 합니다: {len(descriptors)}개")
    return descriptors, rows, feature_names


def inspect_hai() -> tuple[list[dict], list[dict], tuple[str, ...]]:
    dataset_dir = resolve_dataset_dir("HAI")
    descriptors = []
    rows = []
    feature_names = None
    for kind, count in (("train", 4), ("test", 2)):
        for session in range(1, count + 1):
            path = dataset_dir / f"hai-{kind}{session}.csv"
            columns = tuple(pandas.read_csv(path, nrows=0).columns)
            if not columns or columns[0] != "timestamp":
                raise ValueError(f"HAI 첫 열이 timestamp가 아닙니다: {path.name}")
            current_features = columns[1:]
            if feature_names is None:
                feature_names = current_features
            elif current_features != feature_names:
                raise ValueError(f"HAI 채널 순서가 다릅니다: {path.name}")
            timestamp = pandas.to_datetime(pandas.read_csv(path, usecols=["timestamp"])["timestamp"], errors="coerce")
            if timestamp.isna().any():
                raise ValueError(f"HAI timestamp를 읽지 못했습니다: {path.name}")
            intervals = timestamp.diff().dt.total_seconds().iloc[1:]
            source = f"{kind}_{session}"
            if kind == "train":
                descriptors.append({
                    "dataset": "HAI", "source": source, "path": path,
                    "train_length": len(timestamp), "test_length": 0, "total_length": len(timestamp),
                    "session": session, "feature_names": current_features,
                })
            rows.append({
                "dataset": "HAI", "file_or_session": source, "ratio": "not_applicable", "stage": "normal",
                "file_name": path.name, "split": kind, "row_count": len(timestamp),
                "feature_count": len(current_features), "feature_names": "|".join(current_features),
                "timestamp_column": "timestamp", "label_column": "label-test 별도" if kind == "test" else "",
                "sampling_interval_seconds": float(intervals.mode().iloc[0]) if len(intervals) else numpy.nan,
                "sampling_interval_status": "1초 연속" if bool((intervals == 1.0).all()) else "불규칙",
                "irregular_gap_count": int((intervals != 1.0).sum()),
                "duplicate_timestamp_count": int(timestamp.duplicated().sum()),
                "timestamp_reversal_count": int((intervals < 0).sum()),
                "timestamp_start": timestamp.iloc[0].isoformat(),
                "timestamp_end": timestamp.iloc[-1].isoformat(),
                "missing_value_count": numpy.nan, "nonfinite_value_count": numpy.nan,
                "train_start": 0 if kind == "train" else "", "train_end_exclusive": len(timestamp) if kind == "train" else "",
                "validation_start": "전체 정상 구간의 마지막 10%" if kind == "train" else "", "test_start": 0 if kind == "test" else "",
                "test_end_exclusive": len(timestamp) if kind == "test" else "", "boundary_rule": "세션별 경계 유지",
                "normal_sample_count": len(timestamp) if kind == "train" else numpy.nan,
                "anomaly_sample_count": numpy.nan, "label_values_used": False,
                "column_order_matches_reference": current_features == feature_names,
                "channel_set_matches_reference": set(current_features) == set(feature_names),
                "model_input_order_canonicalized": True,
                "input_reordered": current_features != feature_names,
            })
    if len(descriptors) != 4 or feature_names is None or len(feature_names) != 86:
        raise ValueError("HAI는 86채널 학습 세션 4개여야 합니다.")
    return descriptors, rows, feature_names


def scan_feature_values(path: Path, feature_names: tuple[str, ...]) -> tuple[int, int]:
    missing_count = 0
    infinite_count = 0
    for chunk in pandas.read_csv(path, usecols=list(feature_names), chunksize=100_000):
        values = chunk.loc[:, feature_names].to_numpy(dtype=numpy.float64, copy=False)
        missing_count += int(numpy.isnan(values).sum())
        infinite_count += int(numpy.isinf(values).sum())
    return missing_count, infinite_count


def build_data_integrity_rows(
    dataset: str,
    structure_rows: list[dict],
    feature_names: tuple[str, ...],
) -> list[dict]:
    dataset_dir = resolve_dataset_dir(dataset)
    rows = []
    for structure in structure_rows:
        missing_count, infinite_count = scan_feature_values(
            dataset_dir / structure["file_name"], feature_names
        )
        has_timestamp = bool(structure["timestamp_column"])
        timestamp_valid = (
            not has_timestamp
            or (
                structure["duplicate_timestamp_count"] == 0
                and structure["timestamp_reversal_count"] == 0
            )
        )
        rows.append({
            "dataset": dataset,
            "file_or_session": structure["file_or_session"],
            "file_name": structure["file_name"],
            "split": structure["split"],
            "row_count": structure["row_count"],
            "feature_count": structure["feature_count"],
            "channel_order_matches_reference": structure["column_order_matches_reference"],
            "channel_set_matches_reference": structure["channel_set_matches_reference"],
            "model_input_order_canonicalized": structure["model_input_order_canonicalized"],
            "input_reordered": structure["input_reordered"],
            "timestamp_available": has_timestamp,
            "timestamp_start": structure["timestamp_start"],
            "timestamp_end": structure["timestamp_end"],
            "sampling_interval_seconds": structure["sampling_interval_seconds"],
            "irregular_gap_count": structure["irregular_gap_count"],
            "duplicate_timestamp_count": structure["duplicate_timestamp_count"],
            "timestamp_reversal_count": structure["timestamp_reversal_count"],
            "missing_value_count": missing_count,
            "infinite_value_count": infinite_count,
            "nonfinite_value_count": missing_count + infinite_count,
            "normal_train_end_exclusive": structure["train_end_exclusive"],
            "validation_rule": structure["validation_start"],
            "test_start": structure["test_start"],
            "boundary_rule": structure["boundary_rule"],
            "integrity_passed": (
                structure["channel_set_matches_reference"]
                and structure["model_input_order_canonicalized"]
                and timestamp_valid
                and missing_count == 0
                and infinite_count == 0
            ),
            "label_values_used": False,
        })
    return rows


def read_training_features(descriptor: dict, feature_names: tuple[str, ...]) -> numpy.ndarray:
    frame = pandas.read_csv(
        descriptor["path"], usecols=list(feature_names), nrows=descriptor["train_length"],
        dtype={name: numpy.float32 for name in feature_names},
    )
    return frame.loc[:, feature_names].to_numpy(copy=False)
