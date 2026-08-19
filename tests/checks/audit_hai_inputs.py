"""HAI 23.05 훈련 4세션의 누적 비율 조건과 전처리를 점검한다.

모델 설정에는 훈련 파일 통계만 사용한다.
"""

import hashlib
import math
import subprocess
import sys
from pathlib import Path

import numpy
import pandas

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.save_scores import snapshot_config
from src.common.experiment_config import load_dataset_ratios, load_validation_fraction
from src.data_split.take_training_prefix import compute_kept_length


EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments" / "checks" / "datasets" / "hai"
DATA_DIR = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "HAI-23.05"
LOGS_DIR = EXPERIMENT_DIR / "logs"
TRAIN_FILES = tuple(f"hai-train{session}.csv" for session in range(1, 5))
EXPECTED_FEATURE_COUNT = 86
RATIO_PERCENTS = load_dataset_ratios("HAI")
RATIOS = tuple(percent / 100 for percent in RATIO_PERCENTS)
VALIDATION_FRACTION = load_validation_fraction()
CURRENT_WINDOW_SIZES = (5,)


def analyze_training_session(file_name: str, frame: pandas.DataFrame) -> dict:
    """훈련 세션 하나의 스키마와 1초 시간 연속성을 검사한다."""
    if "timestamp" not in frame.columns:
        raise ValueError(f"timestamp 열 없음: {file_name}")

    timestamps = pandas.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"timestamp 파싱 실패: {file_name}")
    gap_count = int((timestamps.diff().dt.total_seconds().iloc[1:] != 1.0).sum())
    if gap_count:
        raise ValueError(f"1초 연속성 위반: {file_name}, gap={gap_count}")

    features = frame.drop(columns="timestamp")
    if features.empty:
        raise ValueError(f"센서 열 없음: {file_name}")
    if any(not pandas.api.types.is_numeric_dtype(features[column]) for column in features):
        raise ValueError(f"숫자형이 아닌 센서 열 존재: {file_name}")

    values = features.to_numpy()
    return {
        "file_name": file_name,
        "row_count": len(frame),
        "feature_count": len(features.columns),
        "first_timestamp": timestamps.iloc[0].isoformat(),
        "last_timestamp": timestamps.iloc[-1].isoformat(),
        "timestamp_gap_count": gap_count,
        "missing_value_count": int(features.isna().to_numpy().sum()),
        "nonfinite_value_count": int((~numpy.isfinite(values)).sum()),
    }


def build_ratio_feasibility_rows(
    session: int,
    train_length: int,
    ratios=RATIOS,
    window_sizes=CURRENT_WINDOW_SIZES,
    validation_fraction: float = VALIDATION_FRACTION,
) -> list[dict]:
    """세션별 앞쪽 누적분의 train·validation·정규화 표본 수를 센다."""
    rows = []
    for ratio in ratios:
        validation_length = math.ceil(validation_fraction * train_length)
        fit_pool_length = train_length - validation_length
        kept_length = compute_kept_length(fit_pool_length, ratio)
        model_train_length = kept_length
        for window_size in window_sizes:
            train_window_count = model_train_length - window_size
            validation_window_count = validation_length - window_size
            rows.append({
                "session": session,
                "ratio": ratio,
                "window_size": window_size,
                "original_train_length": train_length,
                "fit_pool_length": fit_pool_length,
                "kept_length": kept_length,
                "model_train_length": model_train_length,
                "validation_length": validation_length,
                "train_window_count": train_window_count,
                "validation_window_count": validation_window_count,
                "scaler_fit_observation_count": kept_length,
                "validation_transform_observation_count": validation_length,
                "feasible": min(
                    train_window_count,
                    validation_window_count,
                ) > 0,
            })
    return rows


def build_ratio_channel_activity_rows(
    train_features: pandas.DataFrame,
    session: int,
    ratios=RATIOS,
    validation_fraction: float = VALIDATION_FRACTION,
) -> list[dict]:
    """고정 validation 앞 fit pool의 비율별 누적 구간에서 채널 활동을 기록한다."""
    rows = []
    validation_length = math.ceil(len(train_features) * validation_fraction)
    fit_pool = train_features.iloc[: len(train_features) - validation_length]
    for ratio in ratios:
        kept_length = compute_kept_length(len(fit_pool), ratio)
        active_channels = fit_pool.iloc[:kept_length].nunique() > 1
        rows.extend(
            {
                "session": session,
                "ratio": ratio,
                "channel": channel,
                "active": bool(active),
            }
            for channel, active in active_channels.items()
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data_file:
        for chunk in iter(lambda: data_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_hai_preflight() -> None:
    inventory_rows = []
    feasibility_rows = []
    channel_activity_rows = []
    expected_columns = None

    for session, file_name in enumerate(TRAIN_FILES, start=1):
        path = DATA_DIR / file_name
        print(f"[{session}/4] {file_name}")
        frame = pandas.read_csv(path, low_memory=False)
        inventory = analyze_training_session(file_name, frame)
        features = frame.drop(columns="timestamp")
        if expected_columns is None:
            expected_columns = list(features.columns)
        elif list(features.columns) != expected_columns:
            raise ValueError(f"훈련 세션 채널 순서 불일치: {file_name}")
        if inventory["feature_count"] != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"HAI 채널 수 불일치: {file_name}, {inventory['feature_count']}"
            )

        inventory["session"] = session
        inventory["file_bytes"] = path.stat().st_size
        inventory["sha256"] = sha256_file(path)
        inventory_rows.append(inventory)
        feasibility_rows.extend(build_ratio_feasibility_rows(session, len(frame)))
        channel_activity_rows.extend(
            build_ratio_channel_activity_rows(features, session)
        )

    inventory_frame = pandas.DataFrame(inventory_rows).sort_values("session")
    feasibility_frame = pandas.DataFrame(feasibility_rows).sort_values(
        ["session", "ratio", "window_size"]
    )
    channel_activity_frame = pandas.DataFrame(channel_activity_rows).sort_values(
        ["session", "ratio", "channel"]
    )
    corpus_channel_activity_frame = (
        channel_activity_frame.groupby(["ratio", "channel"], as_index=False)["active"]
        .any()
        .sort_values(["ratio", "channel"])
    )
    infeasible_count = int((~feasibility_frame["feasible"]).sum())
    if infeasible_count:
        raise ValueError(f"학습 불가능한 비율·window 조건: {infeasible_count}개")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    inventory_frame.to_csv(LOGS_DIR / "inventory.csv", index=False)
    feasibility_frame.to_csv(LOGS_DIR / "ratio_feasibility.csv", index=False)
    channel_activity_frame.to_csv(
        LOGS_DIR / "ratio_channel_activity.csv", index=False
    )
    corpus_channel_activity_frame.to_csv(
        LOGS_DIR / "corpus_channel_activity.csv", index=False
    )

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
            "ratio_unit": "fraction_of_fit_pool_after_fixed_validation",
            "validation_split_order": "before_ratio",
            "ratio_base": "fit_pool",
            "scaler_fit_scope": "all_current_ratio_fit_subsets",
            "validation_fraction": VALIDATION_FRACTION,
            "current_window_sizes": CURRENT_WINDOW_SIZES,
            "expected_feature_count": EXPECTED_FEATURE_COUNT,
            "test_files_read": False,
            "metrics": [
                "channel_activity",
                "corpus_channel_activity",
            ],
        },
        git_hash,
        str(EXPERIMENT_DIR / "snapshots"),
    )

    corpus_counts = corpus_channel_activity_frame.groupby("ratio")["active"].sum()
    print("\n네 세션 합집합 활성 채널 수")
    print(corpus_counts.to_string())
    print(f"\n학습 불가능 조건: {infeasible_count}개")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_hai_preflight()
