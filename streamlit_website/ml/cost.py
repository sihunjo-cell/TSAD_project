"""과거 실행 기록으로 모델별 학습/추론 비용(초 단위)을 추정한다.

**초기 구현은 초(seconds) 단위 비용만 만든다.** 이 DB의 `l4_cost_inputs`
(GPU 시간당 단가)가 전부 NULL이라 금액으로 억지 변환하지 않는다
(Phase 2 데이터 조사에서 확인). 금액이 필요해지면 이 모듈이 반환하는 초 단위
비용에 단가를 곱하기만 하면 되도록 설계했다.

**모델마다 동일한 증가율을 적용하지 않는다.** candidate(=config_id+head)별로
자신의 과거 실행 기록만 사용해 (행 수 → 초) 회귀를 따로 만든다. 가장 단순하고
설명 가능한 방법을 우선한다는 원칙에 따라 **단순 선형회귀(least squares)** 를
쓴다 — 실제 DB를 확인한 결과 대부분의 candidate가 서로 다른 q(행 수)에서 여러
관측치를 갖고 있어 선형 추세를 잡기에 충분했고, 더 복잡한 모델(log-linear,
power-law 등)을 정당화할 근거가 부족했다.

training-free/zero-shot 모델은 여러 논리 q가 같은 물리 실행(run_id)을 재사용
한다 — 회귀에 같은 관측치를 중복으로 넣지 않도록 run_id로 중복 제거한다.

관측치가 부족한 candidate는 `estimable=False`로 표시하고 절대 0이나 임의의
증가율로 채우지 않는다 (원칙).
"""

from __future__ import annotations

from dataclasses import dataclass

from streamlit_website.ml.config import (
    COST_SOURCE_INSUFFICIENT,
    COST_SOURCE_REGRESSION,
    COST_SOURCE_SINGLE_OBSERVATION,
    DEFAULT_CONFIG,
    MLConfig,
)


@dataclass(frozen=True)
class CostCurve:
    """한 candidate의 (행 수 → 초) 비용 곡선. `estimable=False`면 예측하지 않는다."""

    estimable: bool
    source: str
    observation_count: int
    distinct_rows_count: int
    slope: float | None = None
    intercept: float | None = None
    flat_seconds: float | None = None  # 관측치가 1개뿐일 때만 사용
    reason: str | None = None

    def predict(self, rows: int) -> float | None:
        """주어진 행 수에서의 예상 초. 음수가 나오지 않도록 0으로 clamp한다."""
        if not self.estimable:
            return None
        if self.flat_seconds is not None:
            return self.flat_seconds
        value = self.slope * rows + self.intercept
        return max(0.0, value)


def _distinct_run_observations(
    results_rows: list[dict], *, rows_field: str, seconds_field: str,
) -> list[tuple[int, float]]:
    """run_id로 중복(물리 실행 재사용)을 제거한 (행 수, 초) 관측치 목록."""
    seen_run_ids: set[str] = set()
    observations = []
    for row in results_rows:
        run_id = row.get("run_id")
        rows = row.get(rows_field)
        seconds = row.get(seconds_field)
        if run_id is None or rows is None or seconds is None:
            continue
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        observations.append((int(rows), float(seconds)))
    return observations


def _fit_linear(observations: list[tuple[int, float]]) -> tuple[float, float]:
    """단순 최소제곱 선형회귀 (외부 의존성 없이 직접 계산)."""
    n = len(observations)
    mean_x = sum(x for x, _ in observations) / n
    mean_y = sum(y for _, y in observations) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in observations)
    denominator = sum((x - mean_x) ** 2 for x, _ in observations)
    if denominator == 0:
        # 모든 관측치의 행 수가 같다 — 기울기를 정의할 수 없으므로 평균으로 수평선.
        return 0.0, mean_y
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def build_cost_curve(
    results_rows: list[dict], *, rows_field: str, seconds_field: str, config: MLConfig = DEFAULT_CONFIG,
) -> CostCurve:
    """한 candidate의 학습 또는 추론 비용 곡선을 만든다.

    `rows_field`/`seconds_field`는 `db.load_results_for_candidate()` 행의 키다.
    학습: rows_field="available_training_rows", seconds_field="training_seconds".
    추론: rows_field="test_observations", seconds_field="test_inference_seconds".
    """
    observations = _distinct_run_observations(results_rows, rows_field=rows_field, seconds_field=seconds_field)
    distinct_rows = {rows for rows, _ in observations}
    if len(distinct_rows) >= config.minimum_cost_observations_for_regression:
        slope, intercept = _fit_linear(observations)
        return CostCurve(True, COST_SOURCE_REGRESSION, len(observations), len(distinct_rows),
                          slope=slope, intercept=intercept)
    if len(observations) >= 1:
        flat = sum(seconds for _, seconds in observations) / len(observations)
        return CostCurve(True, COST_SOURCE_SINGLE_OBSERVATION, len(observations), len(distinct_rows),
                          flat_seconds=flat)
    return CostCurve(False, COST_SOURCE_INSUFFICIENT, 0, 0,
                      reason="insufficient_historical_execution_data")


@dataclass(frozen=True)
class CandidateCostModel:
    """한 candidate의 학습 곡선과 추론 곡선을 함께 담는다."""

    training_curve: CostCurve
    inference_curve: CostCurve


def build_candidate_cost_model(results_rows: list[dict], *, config: MLConfig = DEFAULT_CONFIG) -> CandidateCostModel:
    training = build_cost_curve(
        results_rows, rows_field="available_training_rows", seconds_field="training_seconds", config=config,
    )
    inference = build_cost_curve(
        results_rows, rows_field="test_observations", seconds_field="test_inference_seconds", config=config,
    )
    return CandidateCostModel(training, inference)


def estimate_stage_cost(
    cost_model: CandidateCostModel, *, training_rows: int | None, inference_volume: float, needs_training: bool,
) -> tuple[float | None, float | None, float | None, str | None, str | None]:
    """한 stage의 (학습비용, 추론비용, 총비용, cost_estimation_source, exclusion_reason)을 만든다.

    `needs_training=False`면 (checkpoint 유지, 또는 학습이 필요 없는 후보) 학습
    비용은 0이 아니라 `None`으로 두고 "해당 없음"으로 취급한다 — 학습 안 함을
    "학습비 0으로 추정됨"과 구분한다.
    """
    exclusion_reason = None
    training_cost = None
    if needs_training:
        if training_rows is None:
            return None, None, None, None, "no_training_row_count_for_stage"
        training_cost = cost_model.training_curve.predict(training_rows)
        if training_cost is None:
            exclusion_reason = f"training_cost_{cost_model.training_curve.reason}"

    inference_cost = cost_model.inference_curve.predict(round(inference_volume))
    if inference_cost is None and exclusion_reason is None:
        exclusion_reason = f"inference_cost_{cost_model.inference_curve.reason}"

    if exclusion_reason is not None:
        return None, None, None, None, exclusion_reason

    total = (training_cost or 0.0) + inference_cost
    source_parts = [cost_model.training_curve.source if needs_training else None, cost_model.inference_curve.source]
    source = "+".join(part for part in source_parts if part)
    return training_cost, inference_cost, total, source, None
