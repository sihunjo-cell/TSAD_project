"""GHL 입력의 구조·길이·정규화 위험을 다시 감사한다.

모델·분할·정규화 결정에는 학습 구간 통계만 쓴다. 테스트 라벨은 파일 무결성과 기술 통계에만
사용한다.
"""

import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy
import pandas

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.save_scores import snapshot_config
from src.data_split.take_training_prefix import compute_kept_length


EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments" / "checks" / "datasets" / "ghl"
DATA_DIR = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M"
LOGS_DIR = EXPERIMENT_DIR / "logs"

EXPECTED_ARCHIVE_INDICES = set(range(32, 57))
EXPECTED_SERIES = set(range(1, 26))
EXPECTED_FEATURE_COUNT = 19

RATIOS = (0.05, 0.10, 0.20, 0.50, 1.00)
VALIDATION_FRACTION = 0.1
# 현재 구현한 GDN의 잠정 window만 실행 가능성을 검사한다.
CURRENT_WINDOW_SIZES = (5,)

GHL_FILENAME_PATTERN = re.compile(
    r"^(?P<archive_index>\d{3})_GHL_id_(?P<series>\d+)_Sensor_"
    r"tr_(?P<train_boundary>\d+)_1st_(?P<first_anomaly>\d+)\.csv$"
)


def parse_ghl_filename(file_name: str) -> dict:
    """TSB-AD-M GHL 파일명에 든 series·학습 경계·첫 이상 위치를 읽는다."""
    match = GHL_FILENAME_PATTERN.fullmatch(Path(file_name).name)
    if not match:
        raise ValueError(f"GHL 파일명 규약 위반: {file_name}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    values["first_anomaly_index"] = values.pop("first_anomaly")
    return values


def build_ratio_channel_quality_rows(
    train_features: pandas.DataFrame,
    series: int,
    ratios=RATIOS,
) -> list[dict]:
    """앞쪽 비율별 학습 구간에서 채널 통계를 계산한다."""
    rows = []
    for ratio in ratios:
        kept_length = compute_kept_length(len(train_features), ratio)
        prefix_features = train_features.iloc[:kept_length]
        for channel in prefix_features.columns:
            values = prefix_features[channel]
            finite_values = values[numpy.isfinite(values.to_numpy())]
            first_quartile = float(finite_values.quantile(0.25))
            third_quartile = float(finite_values.quantile(0.75))
            interquartile_range = third_quartile - first_quartile
            standard_deviation = float(finite_values.std())
            rows.append({
                "series": series,
                "ratio": ratio,
                "kept_length": kept_length,
                "channel": channel,
                "missing_value_count": int(values.isna().sum()),
                "nonfinite_value_count": int((~numpy.isfinite(values.to_numpy())).sum()),
                "unique_value_count": int(finite_values.nunique()),
                "minimum": float(finite_values.min()),
                "first_quartile": first_quartile,
                "median": float(finite_values.median()),
                "third_quartile": third_quartile,
                "maximum": float(finite_values.max()),
                "interquartile_range": interquartile_range,
                "standard_deviation": standard_deviation,
                "iqr_zero": interquartile_range == 0.0,
                "standard_deviation_zero": standard_deviation == 0.0,
            })
    return rows


def analyze_series_frame(file_name: str, frame: pandas.DataFrame, parsed: dict) -> tuple[dict, list[dict]]:
    """시계열 하나의 구조·라벨 무결성과 학습 채널 통계를 계산한다."""
    if "Label" not in frame.columns:
        raise ValueError(f"Label 열 없음: {file_name}")

    feature_columns = [column for column in frame.columns if column != "Label"]
    if not feature_columns:
        raise ValueError(f"센서 열 없음: {file_name}")
    if any(not pandas.api.types.is_numeric_dtype(frame[column]) for column in frame.columns):
        raise ValueError(f"숫자형이 아닌 열 존재: {file_name}")

    train_boundary = parsed["train_boundary"]
    if not 0 < train_boundary < len(frame):
        raise ValueError(
            f"학습 경계 범위 오류: {file_name}, 경계={train_boundary}, 행={len(frame)}"
        )

    label = frame["Label"]
    label_values = set(label.dropna().unique().tolist())
    if label.isna().any() or not label_values.issubset({0, 1}):
        raise ValueError(f"Label은 결측 없는 0/1이어야 함: {file_name}, 값={label_values}")

    train_frame = frame.iloc[:train_boundary]
    test_frame = frame.iloc[train_boundary:]
    train_features = train_frame[feature_columns]
    test_features = test_frame[feature_columns]
    anomaly_positions = numpy.flatnonzero(label.to_numpy() == 1)
    observed_first_anomaly = int(anomaly_positions[0]) if anomaly_positions.size else None

    inventory = {
        "file_name": file_name,
        **parsed,
        "total_length": len(frame),
        "train_length": len(train_frame),
        "test_length": len(test_frame),
        "feature_count": len(feature_columns),
        "column_count_with_label": len(frame.columns),
        "train_missing_value_count": int(train_features.isna().to_numpy().sum()),
        "test_missing_value_count": int(test_features.isna().to_numpy().sum()),
        "train_anomaly_count": int(train_frame["Label"].sum()),
        "test_anomaly_count": int(test_frame["Label"].sum()),
        "test_anomaly_ratio": float(test_frame["Label"].mean()),
        "observed_first_anomaly_index": observed_first_anomaly,
        "first_anomaly_matches_filename": observed_first_anomaly
        == parsed["first_anomaly_index"],
    }

    channel_rows = build_ratio_channel_quality_rows(
        train_features, parsed["series"], ratios=(1.0,)
    )

    return inventory, channel_rows


def build_ratio_feasibility_rows(
    series: int,
    train_length: int,
    ratios=RATIOS,
    window_sizes=CURRENT_WINDOW_SIZES,
    validation_fraction: float = VALIDATION_FRACTION,
) -> list[dict]:
    """비율·현재 잠정 W마다 train/validation 윈도 수를 계산한다."""
    rows = []
    for ratio in ratios:
        kept_length = compute_kept_length(train_length, ratio)
        validation_length = math.ceil(validation_fraction * kept_length)
        model_train_length = kept_length - validation_length
        for window_size in window_sizes:
            train_window_count = model_train_length - window_size
            validation_window_count = validation_length - window_size
            normalization_sample_count = kept_length - window_size
            rows.append({
                "series": series,
                "ratio": ratio,
                "window_size": window_size,
                "original_train_length": train_length,
                "kept_length": kept_length,
                "model_train_length": model_train_length,
                "validation_length": validation_length,
                "train_window_count": train_window_count,
                "validation_window_count": validation_window_count,
                "normalization_sample_count": normalization_sample_count,
                "feasible": min(
                    train_window_count,
                    validation_window_count,
                    normalization_sample_count,
                ) > 0,
            })
    return rows


def run_ghl_preflight() -> None:
    files = sorted(path for path in DATA_DIR.glob("*.csv") if GHL_FILENAME_PATTERN.fullmatch(path.name))
    parsed_files = [(path, parse_ghl_filename(path.name)) for path in files]
    archive_indices = {parsed["archive_index"] for _, parsed in parsed_files}
    series_numbers = {parsed["series"] for _, parsed in parsed_files}
    if archive_indices != EXPECTED_ARCHIVE_INDICES or series_numbers != EXPECTED_SERIES:
        raise ValueError(
            f"GHL 25개 구성 불일치: archive={sorted(archive_indices)}, series={sorted(series_numbers)}"
        )

    inventory_rows = []
    channel_rows = []
    ratio_rows = []
    for position, (path, parsed) in enumerate(parsed_files, start=1):
        print(f"[{position:02d}/25] {path.name}")
        frame = pandas.read_csv(path, low_memory=False)
        inventory, _ = analyze_series_frame(path.name, frame, parsed)
        file_hash = hashlib.sha256()
        with path.open("rb") as data_file:
            for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
                file_hash.update(chunk)
        inventory["file_bytes"] = path.stat().st_size
        inventory["sha256"] = file_hash.hexdigest()
        inventory_rows.append(inventory)
        channel_rows.extend(build_ratio_channel_quality_rows(
            frame.iloc[:parsed["train_boundary"]].drop(columns="Label"),
            parsed["series"],
        ))
        ratio_rows.extend(build_ratio_feasibility_rows(parsed["series"], inventory["train_length"]))

    inventory_frame = pandas.DataFrame(inventory_rows).sort_values("series")
    channel_frame = pandas.DataFrame(channel_rows).sort_values(["series", "ratio", "channel"])
    ratio_frame = pandas.DataFrame(ratio_rows).sort_values(["series", "ratio", "window_size"])
    if set(inventory_frame["feature_count"]) != {EXPECTED_FEATURE_COUNT}:
        raise ValueError(
            f"GHL 채널 수 불일치: {sorted(set(inventory_frame['feature_count']))}"
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    inventory_frame.to_csv(LOGS_DIR / "inventory.csv", index=False)
    channel_frame.to_csv(LOGS_DIR / "train_channel_quality.csv", index=False)
    ratio_frame.to_csv(LOGS_DIR / "ratio_feasibility.csv", index=False)
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    snapshot_config(
        {
            "data_dir": str(DATA_DIR),
            "files": inventory_frame[["file_name", "sha256"]].to_dict("records"),
            "ratios": RATIOS,
            "validation_fraction": VALIDATION_FRACTION,
            "current_window_sizes": CURRENT_WINDOW_SIZES,
            "expected_feature_count": EXPECTED_FEATURE_COUNT,
        },
        git_hash,
        str(EXPERIMENT_DIR / "snapshots"),
    )


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_ghl_preflight()
