"""GHL 실데이터 프리플라이트 — Manifest 대기 중 구조·길이·정규화 위험만 확인한다.

모델·분할·정규화 결정에는 학습 구간 통계만 쓴다(docs/role_A.md:41-44).
테스트 라벨은 파일 무결성과 기술 통계에만 쓰며 모델 설정 근거로 쓰지 않는다.
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
from src.data_split.front_trim_split import compute_kept_length


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M"
LOGS_DIR = EXPERIMENT_DIR / "logs"

# TheDatumOrg/TSB-AD, Datasets/File_List/TSB-AD-M.csv:33-57.
EXPECTED_ARCHIVE_INDICES = set(range(32, 57))
EXPECTED_SERIES = set(range(1, 26))
EXPECTED_FEATURE_COUNT = 19

RATIOS = (0.05, 0.10, 0.20, 0.50, 1.00)
RATIO_ROLES = {
    0.05: "스트레스 하한",
    0.10: "저데이터 기준점",
    0.20: "구조 회복 지점",
    0.50: "고데이터 대조",
    1.00: "전체 기준",
}
VALIDATION_FRACTION = 0.1  # configs/gdn_hyperparams.yaml train_params.val_size
# 근거 후보 전부를 계산하되 여기서 하나를 고르지 않는다: RECON.md:208-218.
SOURCE_WINDOW_CANDIDATES = (5, 15, 50, 55, 155)

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
        reduced_features = train_features.iloc[:kept_length]
        for channel in reduced_features.columns:
            values = reduced_features[channel]
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


def build_ratio_representativeness_rows(
    train_features: pandas.DataFrame,
    series: int,
    ratios=RATIOS,
) -> list[dict]:
    """비율별 앞쪽 구간이 학습 전체의 상태와 선형 관계를 얼마나 담는지 잰다."""
    full_mean = train_features.mean()
    full_standard_deviation = train_features.std().where(train_features.nunique() > 1)
    full_first_quartile = train_features.quantile(0.25)
    full_third_quartile = train_features.quantile(0.75)
    full_correlation = train_features.corr().to_numpy()
    pair_indices = numpy.triu_indices(len(train_features.columns), k=1)
    full_relationships = full_correlation[pair_indices]
    valid_full_relationships = numpy.isfinite(full_relationships)

    rows = []
    for ratio in ratios:
        kept_length = compute_kept_length(len(train_features), ratio)
        reduced_features = train_features.iloc[:kept_length]
        active_channels = reduced_features.nunique() > 1
        reduced_iqr = (
            reduced_features.quantile(0.75) - reduced_features.quantile(0.25)
        )
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
            numpy.abs(
                reduced_relationships[comparable] - full_relationships[comparable]
            ),
            1.0,
        )
        rows.append({
            "series": series,
            "ratio": ratio,
            "kept_length": kept_length,
            "active_channel_count": int(active_channels.sum()),
            "variable_iqr_channel_count": int((reduced_iqr > 0.0).sum()),
            "interquartile_support_coverage": float(covers_full_iqr.mean()),
            "median_standardized_mean_gap": float(standardized_mean_gap.median()),
            # Pearson은 관계 학습의 충분조건이 아니라 GDN 착수 전 선형 구조 검수치다.
            "relationship_similarity": float(
                relationship_similarity[valid_full_relationships].mean()
            ),
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
    window_sizes=SOURCE_WINDOW_CANDIDATES,
    validation_fraction: float = VALIDATION_FRACTION,
) -> list[dict]:
    """비율·후보 W마다 train/validation 윈도 수를 계산한다."""
    rows = []
    for ratio in ratios:
        kept_length = compute_kept_length(train_length, ratio)
        validation_length = math.ceil(validation_fraction * kept_length)
        model_train_length = kept_length - validation_length
        for window_size in window_sizes:
            # GraGOD, datasets/dataset.py:71-77 — window 수 = 구간 길이 - W.
            train_window_count = model_train_length - window_size
            validation_window_count = validation_length - window_size
            # D-34: 1-step forecast 전체를 써서 정규화 표본은 L-W개다.
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
    representativeness_rows = []
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
        representativeness_rows.extend(build_ratio_representativeness_rows(
            frame.iloc[:parsed["train_boundary"]].drop(columns="Label"),
            parsed["series"],
        ))
        ratio_rows.extend(build_ratio_feasibility_rows(parsed["series"], inventory["train_length"]))

    inventory_frame = pandas.DataFrame(inventory_rows).sort_values("series")
    channel_frame = pandas.DataFrame(channel_rows).sort_values(["series", "ratio", "channel"])
    ratio_frame = pandas.DataFrame(ratio_rows).sort_values(["series", "ratio", "window_size"])
    representativeness_frame = pandas.DataFrame(representativeness_rows).sort_values(
        ["series", "ratio"]
    )

    if set(inventory_frame["feature_count"]) != {EXPECTED_FEATURE_COUNT}:
        raise ValueError(
            f"GHL 채널 수 불일치: {sorted(set(inventory_frame['feature_count']))}"
        )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    inventory_frame.to_csv(LOGS_DIR / "inventory.csv", index=False)
    channel_frame.to_csv(LOGS_DIR / "train_channel_quality.csv", index=False)
    ratio_frame.to_csv(LOGS_DIR / "ratio_feasibility.csv", index=False)
    representativeness_frame.to_csv(
        LOGS_DIR / "ratio_representativeness.csv", index=False
    )

    full_train_channels = channel_frame[channel_frame["ratio"] == 1.0]
    five_percent_channels = channel_frame[channel_frame["ratio"] == 0.05]
    full_train_zero_iqr = full_train_channels[full_train_channels["iqr_zero"]]
    full_train_frozen = full_train_channels[full_train_channels["standard_deviation_zero"]]
    five_percent_zero_iqr = five_percent_channels[five_percent_channels["iqr_zero"]]
    five_percent_frozen = five_percent_channels[five_percent_channels["standard_deviation_zero"]]
    full_train_zero_iqr_names = ", ".join(sorted(full_train_zero_iqr["channel"].unique()))
    five_percent_frozen_names = ", ".join(sorted(five_percent_frozen["channel"].unique())) or "없음"
    infeasible_rows = ratio_frame[~ratio_frame["feasible"]]
    first_anomaly_mismatches = inventory_frame[~inventory_frame["first_anomaly_matches_filename"]]
    ratio_summary_lines = [
        "| 비율 | 보존 길이 중앙값 | 활성 채널 최소/중앙값 | IQR 변동 채널 중앙값 | 전체 IQR 범위 포괄률 중앙값 | 표준화 평균 차이 중앙값 | 관계 유사도 중앙값 | 역할 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for ratio, ratio_group in representativeness_frame.groupby("ratio", sort=True):
        ratio_summary_lines.append(
            f"| {ratio:.0%} | {ratio_group['kept_length'].median():,.0f} | "
            f"{ratio_group['active_channel_count'].min():.0f}/"
            f"{ratio_group['active_channel_count'].median():.0f} | "
            f"{ratio_group['variable_iqr_channel_count'].median():.0f} | "
            f"{ratio_group['interquartile_support_coverage'].median():.3f} | "
            f"{ratio_group['median_standardized_mean_gap'].median():.3f} | "
            f"{ratio_group['relationship_similarity'].median():.3f} | "
            f"{RATIO_ROLES[ratio]} |"
        )
    ratio_summary_table = "\n".join(ratio_summary_lines)
    report = f"""# exp01b — GHL 실데이터 프리플라이트

실행 목적은 강혁님의 공식 Manifest를 대신하는 것이 아니라, 시훈 담당 코드가 실제 GHL 구조에서 출발해도 되는지 확인하는 데 있다. 작업 원본 25개는 저장소 밖 공유 경로 `C:\\Users\\simon\\time_series\\shared_data\\TSAD_project\\TSB-AD-M`에 두고 분석 산출물만 이 실험 폴더에 저장했다.

## 근거

- 공식 목록: TheDatumOrg/TSB-AD `Datasets/File_List/TSB-AD-M.csv:33-57`의 GHL 25개.
- 학습 경계와 필요한 표: `docs/role_A.md:7-11, 19-29`.
- 모델 판단에 쓰는 통계 범위: `docs/role_A.md:41-44`에 따라 학습 구간만 사용.
- 윈도 수: GraGOD `datasets/dataset.py:71-77`의 `len(data) - window_size`.
- validation 비율: `configs/gdn_hyperparams.yaml`의 `train_params.val_size=0.1`.
- W 후보: `experiments/exp00_gragod_recon/RECON.md:208-218`에서 실제로 확인한 5, 15, 50, 55, 155. 여기서는 후보를 비교했을 뿐 하나를 고르지 않았다.
- 비율 축의 연구 목적과 앞자르기 정의: `docs/plan_v4.md:8, 29, 154-158, 213-215`.

## 확인 결과

- 파일 수: {len(inventory_frame)}개. archive index 032–056, series 01–25가 빠짐없이 한 번씩 존재한다.
- 스키마: 모든 파일이 센서 {EXPECTED_FEATURE_COUNT}개 + `Label` 1개로 같다.
- 학습 길이: 최솟값 {int(inventory_frame['train_length'].min()):,}, 최댓값 {int(inventory_frame['train_length'].max()):,}. 공식 파일명 경계와 실제 분할 길이가 전부 맞는다.
- 전체 길이: 최솟값 {int(inventory_frame['total_length'].min()):,}, 최댓값 {int(inventory_frame['total_length'].max()):,}.
- 결측값: 학습 {int(inventory_frame['train_missing_value_count'].sum()):,}개, 테스트 {int(inventory_frame['test_missing_value_count'].sum()):,}개.
- 학습 라벨 이상: {int(inventory_frame['train_anomaly_count'].sum()):,}개. 테스트 라벨은 구조 검수에만 사용했다.
- 파일명의 `1st_` 0-based 인덱스와 실제 첫 `Label=1` 인덱스 불일치: {len(first_anomaly_mismatches)}개.
- 학습 전체의 IQR=0 채널 행: {len(full_train_zero_iqr)} / {len(full_train_channels)}개. 표준편차까지 0인 완전 고정 행은 {len(full_train_frozen)}개다. IQR=0 채널은 {full_train_zero_iqr_names}이며, 희소하게 전환되는 제어 신호라서 삭제 대상으로 단정하지 않는다.
- 5% 구간의 IQR=0 채널 행: {len(five_percent_zero_iqr)} / {len(five_percent_channels)}개. 표준편차까지 0인 행은 {len(five_percent_frozen)}개이며 해당 채널은 {five_percent_frozen_names}이다. 비율마다 정규화 통계를 다시 추정해야 한다는 근거다.
- 25개 × 5비율 × W 후보 5개 = {len(ratio_frame)}조건 가운데 train·validation·정규화 표본이 1개라도 없는 조건: {len(infeasible_rows)}개.

## 비율 타당성

5·10·20·50·100%는 인접 조건이 각각 2배·2배·2.5배·2배라서 20배 범위를 거의 로그 간격으로 훑는다. 정확한 최적 비율이라는 뜻은 아니다. 저데이터 하한, 구조가 살아나는 구간, 전체 학습 사이를 적은 조건으로 나누는 실험 격자다.

대표성 검사는 테스트 구간과 모델 점수를 쓰지 않았다. 각 비율의 앞쪽 학습 구간에서 활성 채널 수와 IQR 변동 채널 수를 세고, 그 구간의 최솟값·최댓값이 학습 전체의 Q1–Q3 범위를 담는 채널 비율을 계산했다. 활성 채널은 서로 다른 값을 2개 이상 관측한 채널이다. `std > 0` 판정은 긴 정수 상수열에서도 부동소수점 오차를 낼 수 있어 쓰지 않았다. 표준화 평균 차이는 비율 구간과 학습 전체의 채널별 평균 차이를 학습 전체 표준편차로 나눈 뒤 변하는 채널의 중앙값을 취한 값이다. 관계 유사도는 채널 쌍의 Pearson 상관 차이를 0–1로 바꾼 정찰 지표다. 상관을 계산할 수 없는 고정 채널 쌍은 0점으로 뒀다. Pearson 값만으로 GDN의 관계 학습 가능성을 확정하지는 않는다.

{ratio_summary_table}

5%는 중앙값 2,500포인트와 충분한 윈도를 남기지만 활성 채널 중앙값이 13/19, 관계 유사도가 0.351이라 전체 구조를 대표하지 않는다. 저데이터 한계를 일부러 누르는 스트레스 하한으로는 합당하다. 10%는 최악 시계열의 활성 채널이 9개에서 13개로 늘고 표준화 평균 차이도 0.293에서 0.234로 줄지만, 관계 유사도는 0.383에 머문다. 10%도 충분 데이터가 아니라 저데이터 기준점으로 해석한다.

20%에서 활성 채널 중앙값은 처음 19/19가 되지만 최솟값은 13개다. 관계 유사도 중앙값은 0.814로 뛴다. GHL에서 채널 관계를 학습할 구조가 통상적으로 살아나는 첫 지점은 20%다. 50%는 관계 유사도 0.928인 고데이터 대조, 100%는 전체 기준으로 둔다. IQR 변동 채널 수가 5%의 8개에서 10%의 6개로 줄어드는 비단조 현상은 희소 제어 신호의 0 구간이 길어져 사분위수가 겹친 결과다. 그래서 IQR 하나로 비율을 고르지 않았다.

현재 가장 합당한 방향은 비율 격자를 유지하되 역할을 분리하는 것이다. 5%를 스트레스 하한, 10%를 저데이터 기준점, 20%를 구조 회복 지점, 50%를 고데이터 대조, 100%를 전체 기준으로 사전 등록한다. 5%나 10%를 운영에 충분한 학습량으로 해석하면 안 된다.

## 착수 판단

GHL 파일 로더와 배치 러너는 지금 작성해도 된다. 실제 입력 계약은 25개, 19채널, 마지막 열 `Label`, 파일명의 `tr_` 값을 경계로 쓰는 형태로 고정됐다. 가장 짧은 series 12도 5% 조건에서 W=155일 때 train 윈도 {int(ratio_frame.query("series == 12 and ratio == 0.05 and window_size == 155")['train_window_count'].iloc[0]):,}개, validation 윈도 {int(ratio_frame.query("series == 12 and ratio == 0.05 and window_size == 155")['validation_window_count'].iloc[0]):,}개, 정규화 표본 {int(ratio_frame.query("series == 12 and ratio == 0.05 and window_size == 155")['normalization_sample_count'].iloc[0]):,}개가 남는다. 비율별 정규화 통계는 해당 비율의 학습 구간에서 다시 추정하고, 5%·10%에서 멈춘 채널을 임의로 삭제하지 않는다.

강혁님의 산출물에서 데이터 버전·파일 경계·채널 구성이 다르거나, 가동 초기 안정화 구간 제거와 다운샘플링에 훈련 데이터 직접 근거가 있으면 치명적 변경으로 본다. 이때는 같은 스크립트를 다시 돌려 비율 역할까지 고친다. 현재는 D-20에 따라 다운샘플 배율 1과 초기 절단 0포인트로 고정했다. 이 분석 당시 미확정이던 W와 GDN 실행값은 이후 D-22에서 W=5와 공통 하이퍼파라미터로 확정했다.
"""
    (EXPERIMENT_DIR / "ANALYSIS.md").write_text(report, encoding="utf-8")

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
            "ratio_roles": RATIO_ROLES,
            "representativeness_metrics": [
                "active_channel_count",
                "variable_iqr_channel_count",
                "interquartile_support_coverage",
                "median_standardized_mean_gap",
                "relationship_similarity",
            ],
        },
        git_hash,
        str(EXPERIMENT_DIR / "snapshots"),
    )


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_ghl_preflight()
