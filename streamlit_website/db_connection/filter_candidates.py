"""고정된 센서 구성으로 계획 후보를 고르고 현재 feature는 별도로 계산한다."""

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
WINDOW_PARAMETERS = {
    "MWVAR": "window", "SQDIFF_LAST1": "window", "SQDIFF_LAST3": "window",
    "SQDIFF_CENTERED5": "window", "MWVAR96_SQDIFF_LAST3": "variance_window",
    "MWVAR96_SQDIFF_CENTERED5": "variance_window", "PCA_LEGACY": "window",
    "PaAno": "patch_size", "GDN": "window", "TimeRCD": "context_length", "TSPulse": "context_length",
}


def load_candidates(database=DATABASE_PATH):
    """실행 이력과 성능으로 후보를 거르지 않고 등록 recipe를 읽는다."""
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        records = connection.execute(
            "SELECT config_id, model, settings_json FROM model_configs ORDER BY model, config_id"
        ).fetchall()
    candidates = []
    for config_id, model, serialized in records:
        if model.casefold() == "alora":
            continue
        try:
            settings = json.loads(serialized)
        except (TypeError, json.JSONDecodeError):
            settings = {}
        if not isinstance(settings, dict):
            settings = {}
        parameters = settings.get("hyperparameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        heads = parameters.get("heads", [""]) if model == "TSPulse" else [parameters.get("score_head", "")]
        if not isinstance(heads, list) or not heads:
            heads = [""]
        for head in dict.fromkeys(heads):
            candidates.append({
                "model": model, "config_id": config_id, "head": head,
                "tier": settings.get("tier", ""), "target_use": settings.get("target_use", ""),
                "parameters": parameters, "recipe": settings.get("common_recipe", {}),
                "settings": settings,
            })
    return candidates


def profile_normal_data(frame, sensor_columns):
    if frame.empty or not sensor_columns:
        raise ValueError("정상 데이터와 센서 열을 선택해 주세요.")
    values = frame.loc[:, sensor_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    summary, channels = compute_prefix_features(values, sensor_columns)
    summary.update(observed_row=len(values), input_column=values.shape[1],
                   finite_value_fraction=float(np.isfinite(values).mean()),
                   extractor_version=EXTRACTOR_CONTRACT["version"])
    return summary, channels


def filter_candidates(candidates, *, channel_count):
    """전체 계획의 후보 풀만 판정한다. 현재 길이·통계·결측 여부는 받지 않는다."""
    if type(channel_count) is not int or channel_count < 1:
        raise ValueError("센서를 하나 이상 선택해 주세요.")
    rows = []
    for candidate in candidates:
        model = candidate["model"]
        parameters = candidate["parameters"]
        window = parameters.get(WINDOW_PARAMETERS.get(model))
        stride = parameters.get("stride", 1)
        row = {**candidate, "window": window,
               "stride": window if model == "TimeRCD" else stride,
               "status": "unverified", "reason": "모델의 필수 입력 조건을 확인할 recipe 설정이 없습니다."}
        rows.append(row)
        recipe = candidate["recipe"]
        if (not isinstance(recipe, dict) or not recipe.get("methodology_revision")
                or type(window) is not int or window < 1
                or candidate["target_use"] not in {"training_free", "fit_full_prefix", "strict_zero_shot"}):
            continue
        if stride != 1:
            row["reason"] = "등록 실행 코드의 stride=1 조건과 다릅니다."
            continue
        if model == "GDN":
            if parameters.get("topk") is None and parameters.get("rho") is None:
                row["reason"] = "GDN의 top-k 설정이 없습니다."
                continue
            try:
                resolve_gdn_topk(channel_count, rho=parameters.get("rho"), topk=parameters.get("topk"))
            except ValueError as error:
                row.update(status="infeasible", reason=str(error))
                continue
        if model == "TimeRCD":
            if parameters.get("checkpoint_variant") != "multi" or candidate["head"] != "probability":
                row["reason"] = "등록 실행 코드의 TimeRCD multi 입력 조건을 확인할 수 없습니다."
                continue
            if channel_count < 2:
                row.update(status="infeasible", reason="TimeRCD multi checkpoint에는 센서 2개 이상이 필요합니다.")
                continue
        if model == "TSPulse":
            aggregation = parameters.get("aggregation_window")
            if parameters.get("patch_size") != 8 or candidate["head"] not in {"time", "fft", "pred", "ensemble"}:
                row["reason"] = "지원하는 TSPulse patch·head 설정을 확인할 수 없습니다."
                continue
            if type(aggregation) is not int or aggregation < 2 or aggregation % 2 or aggregation > 2 * window:
                row["reason"] = "TSPulse의 aggregation window 조건을 확인할 수 없습니다."
                continue
        row.update(status="eligible", reason="고정된 센서 구성과 모델의 필수 입력 조건을 충족합니다.")
    return rows


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
    }
