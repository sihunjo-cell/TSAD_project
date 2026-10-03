"""구간별 후보를 DP로 넘길 후보와 제외 후보로 나눈다.

예산 안에서 성능을 최대화할 때는 싼 후보가 빠듯한 예산의 답이 되므로 예측 성능 순위로
자르지 않는다. 실행할 수 없거나 성능·비용을 추정하지 못한 후보만 사유와 함께 제외한다.
checkpoint 유지 선택지는 별도 트랙이라 여기로 오지 않는다.
"""

from __future__ import annotations

from streamlit_website.ml.schemas import CandidateEstimate


def select_stage_candidates(
    candidate_estimates: list[CandidateEstimate],
) -> tuple[list[CandidateEstimate], list[CandidateEstimate]]:
    """(DP 후보: 예측 성능 내림차순, 제외 후보)를 돌려준다."""
    usable = [c for c in candidate_estimates if c.feasible and c.exclusion_reason is None]
    excluded = [c for c in candidate_estimates if not (c.feasible and c.exclusion_reason is None)]
    usable.sort(key=lambda c: (-c.predicted_performance, c.estimated_total_cost_seconds, c.candidate_id))
    return usable, excluded
