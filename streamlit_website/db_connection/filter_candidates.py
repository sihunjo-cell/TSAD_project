"""등록 후보를 읽어 센서 구성으로 1차 축소하고, 현재 데이터의 feature를 계산한다."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.compute_prefix_features import EXTRACTOR_CONTRACT, compute_prefix_features
from src.common.model_registry import resolve_gdn_topk


DATABASE_PATH = Path(__file__).resolve().parents[2] / "experiments/tuning/results/recommendation.sqlite3"
STATUS_LABELS = {"eligible": "계획 후보", "infeasible": "고정 제약 제외", "unverified": "미검증"}
FEATURE_LABELS = {
    "observed_row": "확보한 행 수", "input_column": "센서 수",
    "channel_std_median": "채널 표준편차 중앙값", "channel_interquartile_range_median": "채널 IQR 중앙값",
    "channel_acf_lag1_median": "자기상관 중앙값 (lag 1)",
    "absolute_correlation_median": "채널 간 절대 상관 중앙값",
    "channel_difference_q90_iqr_ratio_median": "변화량 Q90 / IQR 중앙값",
    "channel_median_shift_iqr_ratio_median": "앞·뒤 중앙값 이동 / IQR 중앙값",
    "channel_spectral_entropy_median": "Spectral entropy 중앙값",
}
TARGET_USES = {"training_free", "fit_full_prefix", "strict_zero_shot"}
WINDOW_PARAMETERS = {
    "MWVAR": "window", "SQDIFF_LAST1": "window", "SQDIFF_LAST3": "window",
    "SQDIFF_CENTERED5": "window", "MWVAR96_SQDIFF_LAST3": "variance_window",
    "MWVAR96_SQDIFF_CENTERED5": "variance_window", "PCA_LEGACY": "window",
    "PaAno": "patch_size", "GDN": "window", "TimeRCD": "context_length", "TSPulse": "context_length",
}


def read_json_object(text):
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def load_candidates(database=DATABASE_PATH):
    """등록 recipe와 측정 장치를 읽는다. 실행 이력이나 성능으로 후보를 거르지 않는다."""
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        records = connection.execute(
            "SELECT config_id, model, settings_json FROM model_configs ORDER BY model, config_id"
        ).fetchall()
        devices = {config_id: (backend, peak) for config_id, backend, peak in connection.execute(
            "SELECT config_id, MAX(actual_backend), "
            "MAX(json_extract(execution_details_json, '$.resource_usage.gpu_reserved_peak_bytes')) "
            "FROM cost_executions GROUP BY config_id"
        )}
    candidates = []
    for config_id, model, serialized in records:
        if model.casefold() == "alora":
            continue
        settings = read_json_object(serialized)
        parameters = settings.get("hyperparameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        # DB 결과는 TSPulse만 head로 구분하고 나머지 모델은 빈 문자열이다.
        heads = parameters.get("heads") if model == "TSPulse" else None
        heads = heads if isinstance(heads, list) and heads else [""]
        backend, peak_bytes = devices.get(config_id, (None, None))
        for head in dict.fromkeys(heads):
            candidates.append({
                "model": model, "config_id": config_id, "head": head,
                "tier": settings.get("tier", ""), "target_use": settings.get("target_use", ""),
                "parameters": parameters, "recipe": settings.get("common_recipe", {}),
                "settings": settings, "backend": backend,
                "gpu_peak_gib": peak_bytes / 2 ** 30 if peak_bytes else None,
            })
    return candidates


def check_candidate(candidate, window, channel_count, gpu_available):
    """(status, reason). 데이터 길이와 값은 보지 않는다. 지금 짧아도 운영 중에 쌓이면 쓸 수 있다."""
    model, parameters, recipe = candidate["model"], candidate["parameters"], candidate["recipe"]
    if (not isinstance(recipe, dict) or not recipe.get("methodology_revision")
            or type(window) is not int or window < 1 or candidate["target_use"] not in TARGET_USES):
        return "unverified", "필수 입력 조건을 확인할 recipe 설정이 없음"
    if parameters.get("stride", 1) != 1:
        return "unverified", "등록 실행 코드의 stride=1 조건과 다름"
    if model == "PCA_LEGACY" and parameters.get("n_components") is None:
        return "infeasible", "성분 수를 정하지 않은 PCA라 실행 시간이 너무 김"
    if model == "GDN":
        if parameters.get("topk") is None and parameters.get("rho") is None:
            return "unverified", "GDN top-k 설정이 없음"
        try:
            resolve_gdn_topk(channel_count, rho=parameters.get("rho"), topk=parameters.get("topk"))
        except ValueError as error:
            if channel_count < 2:
                return "infeasible", "GDN에는 센서가 2개 이상 필요함"
            if type(parameters.get("topk")) is int and parameters["topk"] > channel_count:
                return "infeasible", f"top-k({parameters['topk']})가 센서 수({channel_count}개)보다 큼"
            return "infeasible", f"GDN top-k 설정을 쓸 수 없음 ({error})"
    if model == "TimeRCD":
        if parameters.get("checkpoint_variant") != "multi" or parameters.get("score_head") != "probability":
            return "unverified", "등록 실행 코드의 TimeRCD multi 입력 조건을 확인할 수 없음"
        if channel_count < 2:
            return "infeasible", "TimeRCD multi checkpoint에는 센서가 2개 이상 필요함"
    if model == "TSPulse":
        aggregation = parameters.get("aggregation_window")
        if parameters.get("patch_size") != 8 or candidate["head"] not in {"time", "fft", "pred", "ensemble"}:
            return "unverified", "지원하는 TSPulse patch·head 설정을 확인할 수 없음"
        if type(aggregation) is not int or aggregation < 2 or aggregation % 2 or aggregation > 2 * window:
            return "unverified", "TSPulse aggregation window 조건을 확인할 수 없음"
    if not gpu_available and candidate.get("backend") == "cuda":
        return "infeasible", "GPU에서만 잰 후보라 GPU가 없는 현장에서는 실행할 수 없음"
    return "eligible", "고정한 센서 구성과 모델의 필수 입력 조건을 충족함"


def filter_candidates(candidates, *, channel_count, gpu_available=True):
    """센서 수와 GPU 유무로 전체 계획의 후보 풀을 판정한다.

    GPU 메모리 최댓값은 채널이 많은 Dev18 파일에서 잰 값이라 판정에 쓰지 않는다.
    """
    if type(channel_count) is not int or channel_count < 1:
        raise ValueError("센서를 하나 이상 선택해 주세요.")
    rows = []
    for candidate in candidates:
        window = candidate["parameters"].get(WINDOW_PARAMETERS.get(candidate["model"]))
        status, reason = check_candidate(candidate, window, channel_count, gpu_available)
        stride = window if candidate["model"] == "TimeRCD" else candidate["parameters"].get("stride", 1)
        rows.append({**candidate, "window": window, "stride": stride, "status": status, "reason": reason})
    return rows


def profile_normal_data(frame, sensor_columns):
    if frame.empty or not sensor_columns:
        raise ValueError("정상 데이터와 센서 열을 선택해 주세요.")
    values = frame.loc[:, sensor_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    summary, channels = compute_prefix_features(values, sensor_columns)
    summary.update(observed_row=len(values), input_column=values.shape[1],
                   finite_value_fraction=float(np.isfinite(values).mean()),
                   extractor_version=EXTRACTOR_CONTRACT["version"])
    return summary, channels


def build_ml_input(frame, sensor_columns, rows, *, current_model, operating_conditions):
    """제출한 누적 prefix 전체와 계획 후보를 ML 단계에 넘긴다."""
    return {
        "input": {"row_count": len(frame), "channel_count": len(sensor_columns),
                  "sensor_columns": list(sensor_columns), "prefix_mode": "cumulative_full",
                  "sensor_configuration_fixed": True},
        "normal_prefix": frame.loc[:, sensor_columns].copy(),
        "current_model": current_model,
        "operating_conditions": operating_conditions,
        "candidate_scope": "fixed_structure_only",
        "candidates": [row for row in rows if row["status"] == "eligible"],
        "excluded_candidates": [row for row in rows if row["status"] != "eligible"],
    }
