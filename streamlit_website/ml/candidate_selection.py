"""stage별로 신규 도입/재학습 후보를 top-k로 줄인다.

**checkpoint 유지 선택지는 이 축소 대상이 아니다** — `pipeline.py`가 애초에 이
함수에 checkpoint maintenance 후보를 넘기지 않는다 (별도 트랙, 스키마도 다른
`CheckpointMaintenanceOption`).

순위 규칙 (feasibility/성능/비용을 계산한 결과를 바탕으로 하되, 정렬 기준 자체는
새로 정해야 했다 — README/요구사항이 "성능·비용·실행 가능성을 계산해 top-k를
고른다"고만 하고 정확한 공식은 정하지 않았으므로, 아래를 **가정**으로 명시한다):

1. feasible한 후보를 infeasible한 후보보다 우선한다.
2. feasible한 후보 중에서는 `predicted_performance`가 높은 순.
   (predicted_performance가 없는 후보 — 즉 similarity로도 추정할 수 없는 후보 —
   는 feasible해도 맨 뒤로 보낸다. 성능을 모르는 채로 상위에 두지 않는다.)
3. 성능이 같으면 `estimated_total_cost_seconds`가 낮은 순 (비용 정보가 없으면
   맨 뒤로 보낸다).

**infeasible 후보와, feasible하지만 성능/비용을 추정할 수 없는 후보(=
`exclusion_reason`이 채워진 후보)는 top-k 경쟁에 넣지 않는다** — DP가 성능·비용을
모르는 채로 후보를 고를 수는 없기 때문이다. 이런 후보는 k와 무관하게 항상
`stage_candidates_excluded_from_top_k`로 간다.

top-k에서 순위 경쟁을 벌이는 것은 "feasible하고 exclusion_reason이 없는"(=성능과
비용을 둘 다 추정할 수 있었던) 후보뿐이다. k를 넘어 탈락한 후보도 전부
`stage_candidates_excluded_from_top_k`에 보존된다 — 다음 ML 실행에서 다시 top-k
후보로 검토될 수 있어야 하기 때문이다 (원칙).
"""

from __future__ import annotations

from streamlit_website.ml.config import DEFAULT_CONFIG, MLConfig
from streamlit_website.ml.schemas import CandidateEstimate


def _is_usable(candidate: CandidateEstimate) -> bool:
    return candidate.feasible and candidate.exclusion_reason is None


def _rank_key(candidate: CandidateEstimate):
    # usable 후보는 predicted_performance/estimated_total_cost_seconds가 항상 채워져
    # 있어야 하지만(_is_usable 조건), 방어적으로 fallback을 남겨둔다.
    performance_rank = (
        -candidate.predicted_performance if candidate.predicted_performance is not None else float("inf")
    )
    cost_rank = (
        candidate.estimated_total_cost_seconds if candidate.estimated_total_cost_seconds is not None
        else float("inf")
    )
    return (performance_rank, cost_rank, candidate.candidate_id)


def select_stage_candidates(
    candidate_estimates: list[CandidateEstimate], *, config: MLConfig = DEFAULT_CONFIG,
) -> tuple[list[CandidateEstimate], list[CandidateEstimate]]:
    """한 stage의 후보 전체를 (top_k, excluded_from_top_k)로 나눈다.

    남은 usable 후보가 k보다 적으면 전부 top_k에 남는다.
    """
    usable = [c for c in candidate_estimates if _is_usable(c)]
    not_usable = [c for c in candidate_estimates if not _is_usable(c)]
    ranked = sorted(usable, key=_rank_key)
    top_k = ranked[: config.top_k_candidates]
    excluded = ranked[config.top_k_candidates :] + not_usable
    return top_k, excluded
