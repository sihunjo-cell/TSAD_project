"""similarity + historical trajectory로 단계별 후보 성능을 추정한다.

각 단계(stage)는 `future_rows`(그 단계까지 누적될 행 수)를 갖는다. 이를 현재
행 수로 나눈 `growth_ratio`가 그 단계의 "성장 비율"이다. stage=0(지금)은
growth_ratio=1.0이므로, similarity로 찾은 과거 (series, q)에서 그 candidate의
관측값을 그대로 가중평균하는 것과 같아진다 — 즉 이 모듈은 현재 단계와 미래
단계를 같은 함수로 다루며, 미래 단계는 growth_ratio>1일 뿐이다
(`trajectory.map_growth_to_trajectory_q`).

provenance 규칙:
- `estimated_from_similarity`: similarity로 찾은 과거 사례(들)의 가중평균/trajectory
  값을 썼다 (stage=0 포함, DB 관측 범위 안의 미래 단계 포함).
- `extrapolated_held_constant`: 목표 성장 비율이 DB 관측 범위(q<=100)를 넘어서
  마지막 관측(q=100) 값을 그대로 유지한다고 "가정"했다. 이는 실측이 아니라 미래
  예측을 위한 가정임을 명시한다.
- `checkpoint_maintained`: 기존 checkpoint를 유지하는 경우, 재학습 없이 현재 단계의
  추정 성능을 그대로 유지한다고 가정한다 (`estimate_checkpoint_maintenance` 참고).
- 어느 similarity match에서도 그 candidate의 데이터를 얻지 못하면 `None`, provenance
  도 `None`, 그리고 상위 pipeline이 `exclusion_reason`을 채운다 (0으로 대체하지 않는다).

lower_bound는 그 단계 추정에 실제로 쓰인 관측값들의 최솟값이다 — 표본이 하나뿐이면
lower_bound == predicted_performance.
"""

from __future__ import annotations

from dataclasses import dataclass

from streamlit_website.ml.config import (
    PREDICTION_SOURCE_CHECKPOINT_MAINTAINED,
    PREDICTION_SOURCE_EXTRAPOLATED,
    PREDICTION_SOURCE_SIMILARITY,
)
from streamlit_website.ml.schemas import SimilarityMatch
from streamlit_website.ml.trajectory import get_series_trajectory, map_growth_to_trajectory_q, trajectory_point_at_q


@dataclass(frozen=True)
class PerformanceEstimate:
    predicted_performance: float | None
    performance_lower_bound: float | None
    prediction_source: str | None
    used_match_count: int


def estimate_stage_performance(
    results_rows: list[dict], matches: list[SimilarityMatch], *, future_rows: int, current_rows: int,
) -> PerformanceEstimate:
    """한 candidate의 한 stage 성능을, similarity match들의 trajectory로 추정한다.

    `results_rows`는 이 candidate(config_id, head) 전체의
    `db.load_results_for_candidate()` 결과다 (한 번 로드해 모든 stage/모든 match에
    재사용할 수 있도록 상위에서 넘긴다).
    """
    if current_rows <= 0:
        raise ValueError("current_rows는 1 이상이어야 한다")
    growth_ratio = max(1.0, future_rows / current_rows)

    contributions = []  # (weight, vus_pr, extrapolated)
    series_cache: dict[str, list] = {}
    for match in matches:
        points = series_cache.setdefault(match.series, get_series_trajectory(results_rows, match.series))
        if not points:
            continue  # 이 candidate는 이 series에 대한 결과가 전혀 없다
        target_q, extrapolated = map_growth_to_trajectory_q(match.q_percent, growth_ratio)
        point = trajectory_point_at_q(points, target_q)
        if point is None or point.vus_pr is None:
            # 목표 q 자체가 이 series의 trajectory에 없거나 complete 결과가 없다.
            # DB 관측 범위를 넘어 extrapolate하려던 경우, 마지막 관측(q=100)으로 대체한다.
            if extrapolated:
                point = trajectory_point_at_q(points, 100)
            if point is None or point.vus_pr is None:
                continue
        contributions.append((match.weight, point.vus_pr, extrapolated))

    if not contributions:
        return PerformanceEstimate(None, None, None, 0)

    weight_sum = sum(weight for weight, _, _ in contributions)
    predicted = sum(weight * value for weight, value, _ in contributions) / weight_sum
    lower_bound = min(value for _, value, _ in contributions)
    # 사용된 기여가 전부 extrapolate(마지막 관측 유지)였을 때만 그 사실을 provenance에 남긴다.
    # 일부만 extrapolate였다면 여전히 similarity 기반 추정이 주된 근거이므로 SIMILARITY로 남긴다.
    source = (
        PREDICTION_SOURCE_EXTRAPOLATED if all(extrapolated for _, _, extrapolated in contributions)
        else PREDICTION_SOURCE_SIMILARITY
    )
    return PerformanceEstimate(predicted, lower_bound, source, len(contributions))


def estimate_checkpoint_maintenance_performance(
    results_rows: list[dict], matches: list[SimilarityMatch], *, current_rows: int,
) -> PerformanceEstimate:
    """기존 checkpoint 유지 성능: 재학습 없이 "현재 단계"의 추정 성능을 그대로 유지한다.

    `MLInput.current_model`의 `last_trained_at`은 자유 텍스트라 구조화된 stage
    index로 변환할 수 없다 (Phase 1/2 조사에서 확인된 한계). 따라서 이 구현은
    "마지막 학습 시점의 추정 성능"을 "현재 단계(stage 0)에서 similarity로 추정한
    성능"으로 근사한다 — 이는 최근에 학습했을수록 정확하고, 오래 전에 학습했을수록
    낙관적일 수 있는 가정이다 (보고서의 "아직 해결하지 못한 가정"에 기록).
    """
    estimate = estimate_stage_performance(results_rows, matches, future_rows=current_rows, current_rows=current_rows)
    if estimate.predicted_performance is None:
        return estimate
    return PerformanceEstimate(
        estimate.predicted_performance, estimate.performance_lower_bound,
        PREDICTION_SOURCE_CHECKPOINT_MAINTAINED, estimate.used_match_count,
    )
