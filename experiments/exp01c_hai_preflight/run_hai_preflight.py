"""HAI 23.05 훈련 4세션의 10% 조건·전처리 프리플라이트.

모델 설정 근거에는 훈련 파일만 쓴다. 공식 근거: icsdataset/hai README.md
229-264행(파일별 시간 연속성·HAI 23.05 데이터 포인트 86개 표), 284-294행
(timestamp 형식과 1초 간격 예시), 기술문서 PDF 2쪽(파일별 연속성·세션 시간).
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
from src.data_split.front_trim_split import compute_kept_length


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "HAI-23.05"
LOGS_DIR = EXPERIMENT_DIR / "logs"
TRAIN_FILES = tuple(f"hai-train{session}.csv" for session in range(1, 5))
EXPECTED_FEATURE_COUNT = 86
RATIOS = (0.10, 1.00)
VALIDATION_FRACTION = 0.1
SOURCE_WINDOW_CANDIDATES = (5, 15, 50, 55, 155)


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
    window_sizes=SOURCE_WINDOW_CANDIDATES,
    validation_fraction: float = VALIDATION_FRACTION,
) -> list[dict]:
    """세션별 앞자르기 뒤 train·validation·정규화 표본 수를 센다."""
    rows = []
    for ratio in ratios:
        kept_length = compute_kept_length(train_length, ratio)
        validation_length = math.ceil(validation_fraction * kept_length)
        model_train_length = kept_length - validation_length
        for window_size in window_sizes:
            train_window_count = model_train_length - window_size
            validation_window_count = validation_length - window_size
            # D-34: 1-step forecast 전체를 써서 정규화 표본은 L-W개다.
            normalization_sample_count = kept_length - window_size
            rows.append({
                "session": session,
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


def build_ratio_representativeness_rows(
    train_features: pandas.DataFrame,
    session: int,
    ratios=RATIOS,
) -> list[dict]:
    """각 세션 앞쪽 구간이 그 세션 전체의 채널 변화·선형 관계를 얼마나 담는지 잰다."""
    full_mean = train_features.mean()
    full_standard_deviation = train_features.std().where(train_features.nunique() > 1)
    full_first_quartile = train_features.quantile(0.25)
    full_third_quartile = train_features.quantile(0.75)
    pair_indices = numpy.triu_indices(len(train_features.columns), k=1)
    full_relationships = train_features.corr().to_numpy()[pair_indices]
    valid_full_relationships = numpy.isfinite(full_relationships)

    rows = []
    for ratio in ratios:
        kept_length = compute_kept_length(len(train_features), ratio)
        reduced_features = train_features.iloc[:kept_length]
        active_channels = reduced_features.nunique() > 1
        reduced_iqr = reduced_features.quantile(0.75) - reduced_features.quantile(0.25)
        covers_full_iqr = (
            (reduced_features.min() <= full_first_quartile)
            & (reduced_features.max() >= full_third_quartile)
        )
        standardized_mean_gap = (
            (reduced_features.mean() - full_mean).abs() / full_standard_deviation
        )

        reduced_relationships = reduced_features.corr().to_numpy()[pair_indices]
        relationship_similarity = numpy.zeros_like(full_relationships, dtype=float)
        comparable = valid_full_relationships & numpy.isfinite(reduced_relationships)
        relationship_similarity[comparable] = 1.0 - numpy.minimum(
            numpy.abs(reduced_relationships[comparable] - full_relationships[comparable]),
            1.0,
        )
        rows.append({
            "session": session,
            "ratio": ratio,
            "kept_length": kept_length,
            "active_channel_count": int(active_channels.sum()),
            "variable_iqr_channel_count": int((reduced_iqr > 0.0).sum()),
            "interquartile_support_coverage": float(covers_full_iqr.mean()),
            "median_standardized_mean_gap": float(standardized_mean_gap.median()),
            # Pearson은 GDN 성능 대리값이 아니라 10% 구간의 선형 구조 정찰치다.
            "relationship_similarity": float(
                relationship_similarity[valid_full_relationships].mean()
            ),
        })
    return rows


def build_ratio_channel_activity_rows(
    train_features: pandas.DataFrame,
    session: int,
    ratios=RATIOS,
) -> list[dict]:
    """비율별로 각 채널이 적어도 두 값을 관측했는지 기록한다."""
    rows = []
    for ratio in ratios:
        kept_length = compute_kept_length(len(train_features), ratio)
        active_channels = train_features.iloc[:kept_length].nunique() > 1
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
    representativeness_rows = []
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
        representativeness_rows.extend(
            build_ratio_representativeness_rows(features, session)
        )
        channel_activity_rows.extend(
            build_ratio_channel_activity_rows(features, session)
        )

    inventory_frame = pandas.DataFrame(inventory_rows).sort_values("session")
    feasibility_frame = pandas.DataFrame(feasibility_rows).sort_values(
        ["session", "ratio", "window_size"]
    )
    representativeness_frame = pandas.DataFrame(representativeness_rows).sort_values(
        ["session", "ratio"]
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
    representativeness_frame.to_csv(
        LOGS_DIR / "ratio_representativeness.csv", index=False
    )
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
            "validation_fraction": VALIDATION_FRACTION,
            "source_window_candidates": SOURCE_WINDOW_CANDIDATES,
            "expected_feature_count": EXPECTED_FEATURE_COUNT,
            "test_files_read": False,
            "metrics": [
                "active_channel_count",
                "variable_iqr_channel_count",
                "interquartile_support_coverage",
                "median_standardized_mean_gap",
                "relationship_similarity",
                "corpus_channel_activity",
            ],
        },
        git_hash,
        str(EXPERIMENT_DIR / "snapshots"),
    )

    ten_percent = representativeness_frame.query("ratio == 0.10")
    print("\n10% 훈련 구간 요약")
    print(ten_percent.to_string(index=False))
    corpus_counts = corpus_channel_activity_frame.groupby("ratio")["active"].sum()
    print("\n네 세션 합집합 활성 채널 수")
    print(corpus_counts.to_string())
    print(f"\n학습 불가능 조건: {infeasible_count}개")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_hai_preflight()
