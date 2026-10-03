"""과거 실행 기록으로 후보별 학습/추론 비용(초)을 추정한다.

비용 단위는 초다. 금액이 필요하면 `dp_input.py`가 장치(CPU/GPU)별 시간당 단가를 곱한다.
모든 모델에 같은 증가율을 적용하지 않고 후보마다 자신의 실행 기록만 쓴다.
여러 논리 q가 같은 물리 실행(run_id)을 재사용하면 한 번만 센다.

대부분의 모델은 처리한 셀 수(행 × 채널)에 비례하므로 셀당 중앙 단가를 쓴다. Dev18에서는
행 수와 채널 수가 반대로 움직여 행 수만 쓰는 선형회귀의 기울기가 음수가 되고, 현장 규모로
외삽하면 PCA·TimeRCD 추론비가 0초로 나왔다. PaAno만 행 수 선형회귀가 더 잘 맞았다.
학습 행 수는 그 구간의 행 수를 그대로 쓴다. 관측한 최대 행 수로 자르면 큰 현장의 학습비를
낮게 잡아 예산을 넘는 계획이 나온다. 관측이 없으면 추정 불가로 두고 0이나 임의 증가율로
채우지 않는다.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from streamlit_website.ml.config import (
    COST_SOURCE_CELL_RATE,
    COST_SOURCE_INSUFFICIENT,
    COST_SOURCE_REGRESSION,
    COST_SOURCE_SINGLE_OBSERVATION,
    DEFAULT_CONFIG,
    MLConfig,
)

ROW_LINEAR_COST_MODELS = frozenset({"PaAno"})


@dataclass(frozen=True)
class CostCurve:
    """한 candidate의 비용 곡선. `estimable=False`면 예측하지 않는다."""

    estimable: bool
    source: str
    observation_count: int
    per_cell: bool = False
    slope: float | None = None
    intercept: float = 0.0
    reason: str | None = None

    def predict(self, rows: int, channel_count: int = 1) -> float | None:
        """주어진 행 수(와 채널 수)에서의 예상 초."""
        if not self.estimable:
            return None
        units = rows * channel_count if self.per_cell else rows
        return max(0.0, self.slope * units + self.intercept)


def _distinct_run_observations(
    results_rows: list[dict], *, rows_field: str, seconds_field: str, per_cell: bool,
) -> list[tuple[int, float, float]]:
    """run_id로 물리 실행 재사용을 제거한 (행 수, 비용 단위 수, 초) 관측치."""
    seen_run_ids: set[str] = set()
    observations = []
    for row in results_rows:
        run_id, rows, seconds = row.get("run_id"), row.get(rows_field), row.get(seconds_field)
        channels = row.get("input_column") if per_cell else 1
        if run_id is None or rows is None or seconds is None or channels is None or run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        observations.append((int(rows), float(rows * channels), float(seconds)))
    return observations


def _fit_linear(observations: list[tuple[int, float]]) -> tuple[float, float]:
    """단순 최소제곱 선형회귀. 행 수가 모두 같으면 평균으로 수평선."""
    n = len(observations)
    mean_x = sum(x for x, _ in observations) / n
    mean_y = sum(y for _, y in observations) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in observations)
    if denominator == 0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in observations) / denominator
    return slope, mean_y - slope * mean_x


def build_cost_curve(
    results_rows: list[dict], *, rows_field: str, seconds_field: str, per_cell: bool = True,
    config: MLConfig = DEFAULT_CONFIG,
) -> CostCurve:
    """한 candidate의 학습 또는 추론 비용 곡선을 만든다.

    학습: rows_field="available_training_rows", seconds_field="training_seconds".
    추론: rows_field="test_observations", seconds_field="test_inference_seconds".
    """
    observations = _distinct_run_observations(
        results_rows, rows_field=rows_field, seconds_field=seconds_field, per_cell=per_cell,
    )
    rates = [seconds / units for _, units, seconds in observations if units > 0]
    if not observations or (per_cell and not rates):
        return CostCurve(False, COST_SOURCE_INSUFFICIENT, 0, reason="insufficient_historical_execution_data")
    if per_cell:
        return CostCurve(True, COST_SOURCE_CELL_RATE, len(observations), per_cell=True,
                         slope=statistics.median(rates))
    slope, intercept = _fit_linear([(rows, seconds) for rows, _, seconds in observations])
    distinct_rows = {rows for rows, _, _ in observations}
    source = (COST_SOURCE_REGRESSION if len(distinct_rows) >= config.minimum_cost_observations_for_regression
              else COST_SOURCE_SINGLE_OBSERVATION)
    return CostCurve(True, source, len(observations), slope=slope, intercept=intercept)


@dataclass(frozen=True)
class CandidateCostModel:
    """한 candidate의 학습 곡선과 추론 곡선을 함께 담는다."""

    training_curve: CostCurve
    inference_curve: CostCurve


def build_candidate_cost_model(
    results_rows: list[dict], *, model: str = "", config: MLConfig = DEFAULT_CONFIG,
) -> CandidateCostModel:
    per_cell = model not in ROW_LINEAR_COST_MODELS
    training = build_cost_curve(results_rows, rows_field="available_training_rows",
                                seconds_field="training_seconds", per_cell=per_cell, config=config)
    inference = build_cost_curve(results_rows, rows_field="test_observations",
                                 seconds_field="test_inference_seconds", per_cell=per_cell, config=config)
    return CandidateCostModel(training, inference)


def estimate_stage_cost(
    cost_model: CandidateCostModel, *, training_rows: int | None, inference_volume: float, needs_training: bool,
    channel_count: int = 1,
) -> tuple[float | None, float | None, float | None, str | None, str | None]:
    """한 stage의 (학습비용, 추론비용, 총비용, cost_estimation_source, exclusion_reason)을 만든다.

    `needs_training=False`면 (checkpoint 유지, 또는 학습이 필요 없는 후보) 학습
    비용은 0이 아니라 `None`으로 두고 "해당 없음"으로 취급한다.
    """
    exclusion_reason = None
    training_cost = None
    if needs_training:
        if training_rows is None:
            return None, None, None, None, "no_training_row_count_for_stage"
        curve = cost_model.training_curve
        training_cost = curve.predict(training_rows, channel_count)
        if training_cost is None:
            exclusion_reason = f"training_cost_{curve.reason}"

    inference_cost = cost_model.inference_curve.predict(inference_volume, channel_count)
    if inference_cost is None and exclusion_reason is None:
        exclusion_reason = f"inference_cost_{cost_model.inference_curve.reason}"

    if exclusion_reason is not None:
        return None, None, None, None, exclusion_reason

    total = (training_cost or 0.0) + inference_cost
    source_parts = [cost_model.training_curve.source if needs_training else None, cost_model.inference_curve.source]
    source = "+".join(part for part in source_parts if part)
    return training_cost, inference_cost, total, source, None
