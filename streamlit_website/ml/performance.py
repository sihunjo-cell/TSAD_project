"""similarity + historical trajectory로 단계별 후보 성능을 추정한다.

각 단계(stage)는 DB 등록 비율 하나(`ratio_percent`)에 대응한다. 운영 기간 끝까지 모을
행 수를 100%로 보고 그 단계가 시작할 때 확보한 몫이다(`pipeline.build_future_stages`).
similarity로 찾은 과거 series마다 같은 비율에서 잰 결과를 거리 가중평균한다. DB가 그
비율에서만 성능을 쟀으므로 단계 안에서는 예측이 바뀌지 않는다.

provenance 규칙:
- `estimated_from_similarity`: 목표 비율에서 잰 결과를 하나 이상 썼다.
- `extrapolated_held_constant`: 모든 series에 목표 비율 결과가 없어 더 적은 데이터에서 잰
  마지막 값을 유지한다고 "가정"했다. 실측이 아니라 가정임을 명시한다.
- `checkpoint_maintained`: 기존 checkpoint를 유지하는 경우, 그 checkpoint를 학습한 비율의
  추정 성능을 그대로 유지한다고 가정한다.
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
from streamlit_website.ml.trajectory import get_series_trajectory, observed_point_up_to


@dataclass(frozen=True)
class PerformanceEstimate:
    predicted_performance: float | None
    performance_lower_bound: float | None
    prediction_source: str | None
    used_match_count: int


def estimate_stage_performance(
    results_rows: list[dict], matches: list[SimilarityMatch], *, ratio_percent: int,
) -> PerformanceEstimate:
    """한 candidate의 한 stage 성능을, similarity match들이 속한 series의 같은 비율 결과로 추정한다.

    `results_rows`는 이 candidate(config_id, head) 전체의
    `db.load_results_for_candidate()` 결과다 (한 번 로드해 모든 stage/모든 match에
    재사용할 수 있도록 상위에서 넘긴다).
    """
    contributions = []  # (weight, vus_pr, held_from_lower_ratio)
    series_cache: dict[str, list] = {}
    for match in matches:
        if match.series not in series_cache:
            series_cache[match.series] = get_series_trajectory(results_rows, match.series)
        point = observed_point_up_to(series_cache[match.series], ratio_percent)
        if point is None:
            continue  # 이 series에는 목표 비율 이하의 결과가 없다
        contributions.append((match.weight, point.vus_pr, point.q_percent < ratio_percent))

    if not contributions:
        return PerformanceEstimate(None, None, None, 0)

    weight_sum = sum(weight for weight, _, _ in contributions)
    predicted = sum(weight * value for weight, value, _ in contributions) / weight_sum
    lower_bound = min(value for _, value, _ in contributions)
    # 모든 기여가 더 적은 데이터의 값을 유지한 것일 때만 그 사실을 provenance에 남긴다.
    source = (
        PREDICTION_SOURCE_EXTRAPOLATED if all(held for _, _, held in contributions)
        else PREDICTION_SOURCE_SIMILARITY
    )
    return PerformanceEstimate(predicted, lower_bound, source, len(contributions))


def estimate_checkpoint_maintenance_performance(
    results_rows: list[dict], matches: list[SimilarityMatch], *, ratio_percent: int,
) -> PerformanceEstimate:
    """기존 checkpoint 유지 성능: 그 checkpoint를 학습한 비율의 추정 성능을 그대로 유지한다.

    학습 행 수를 모르면 호출자가 현재 행 수의 비율을 넘긴다. 오래전에 적은 데이터로
    학습했다면 PaAno·GDN처럼 학습량에 민감한 모델은 낙관적으로 추정된다.
    """
    estimate = estimate_stage_performance(results_rows, matches, ratio_percent=ratio_percent)
    if estimate.predicted_performance is None:
        return estimate
    return PerformanceEstimate(
        estimate.predicted_performance, estimate.performance_lower_bound,
        PREDICTION_SOURCE_CHECKPOINT_MAINTAINED, estimate.used_match_count,
    )
