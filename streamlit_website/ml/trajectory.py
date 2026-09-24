"""유사한 과거 series의 이후 q trajectory를 연결한다.

과거 DB의 prefix 비율(q_percent)과 우리 미래 단계의 행 수 비율을 **그대로 1:1로
대응시키지 않는다** (원칙). 대신 "성장 비율"(future_rows / current_rows)을 계산해,
같은 series 안에서 그 비율만큼 행 수가 늘어나는 q를 찾는다. 한 series 안에서는
`observed_row = training_boundary * q_percent / 100`이므로 행 수 비율은 곧
q_percent 비율과 같다 — 따라서 "성장 비율 × anchor_q"에 가장 가까운(이상인) 등록
q를 고르는 것으로 계산할 수 있다. 이 성장 비율은 *우리 데이터*의 미래/현재
행 수에서 나오고, anchor_q는 similarity로 찾은 *과거* series의 q이므로, 서로
다른 series의 절대 q를 동일시하는 것이 아니다.

training-free 모델(target_use="training_free")은 여러 논리 q가 물리적으로 같은
실행(run_id)을 재사용한다 — `results → cost_result_links → cost_executions`의
`run_id`로 이를 구분한다. 이 모듈은 trajectory point마다 사용된 run_id 집합을
그대로 노출해서, 상위(performance.py/cost.py)가 "새로 관측된 지점"과 "이전과
같은 물리 실행을 재사용한 지점"을 구분할 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS


@dataclass(frozen=True)
class TrajectoryPoint:
    """한 series, 한 q_percent에서의 (여러 seed를 평균한) 관측값."""

    q_percent: int
    observed_row: int
    vus_pr: float | None  # complete 결과가 하나도 없으면 None
    complete_result_count: int  # 평균에 쓰인 complete 결과(seed) 수
    run_ids: frozenset[str]  # 이 지점의 물리 실행 id들 (cost 정보가 있는 것만)
    training_seconds: float | None
    test_inference_seconds: float | None
    available_training_rows: int | None
    test_observations: int | None
    is_new_physical_execution: bool
    """이 지점의 run_id 중 하나 이상이, 이 series에서 더 작은 q에서는 보지 못한
    새 물리 실행이면 True. training-free 모델이 이전 q의 실행을 그대로 재사용한
    지점이면 False (새 관측이 아니라 재사용)."""


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def get_series_trajectory(results_rows: list[dict], series: str) -> list[TrajectoryPoint]:
    """`db.load_results_for_candidate()`의 결과에서 한 series의 q trajectory를 뽑는다.

    같은 (series, q_percent)에 여러 seed가 있으면 vus_pr과 비용은 seed 평균을
    쓴다 (DB_column_def.txt의 "seed 평균 → family 평균" 관례와 일치). status가
    'complete'가 아닌 행은 vus_pr 평균에서 제외한다.
    """
    by_q: dict[int, list[dict]] = {}
    for row in results_rows:
        if row["series"] != series:
            continue
        by_q.setdefault(row["q_percent"], []).append(row)

    points = []
    seen_run_ids: set[str] = set()
    for q_percent in sorted(by_q):
        rows = by_q[q_percent]
        complete = [row for row in rows if row["status"] == "complete" and row["vus_pr"] is not None]
        run_ids = frozenset(row["run_id"] for row in rows if row.get("run_id"))
        is_new = bool(run_ids - seen_run_ids) if run_ids else False
        seen_run_ids |= run_ids
        points.append(TrajectoryPoint(
            q_percent=q_percent,
            observed_row=rows[0]["observed_row"],
            vus_pr=_mean([row["vus_pr"] for row in complete]),
            complete_result_count=len(complete),
            run_ids=run_ids,
            training_seconds=_mean([row["training_seconds"] for row in rows if row.get("training_seconds") is not None]),
            test_inference_seconds=_mean([row["test_inference_seconds"] for row in rows if row.get("test_inference_seconds") is not None]),
            available_training_rows=next((row["available_training_rows"] for row in rows if row.get("available_training_rows") is not None), None),
            test_observations=next((row["test_observations"] for row in rows if row.get("test_observations") is not None), None),
            is_new_physical_execution=is_new,
        ))
    return points


def map_growth_to_trajectory_q(anchor_q: int, growth_ratio: float) -> tuple[int, bool]:
    """anchor_q에서 growth_ratio만큼 행 수가 늘어난 지점에 대응하는 등록 q를 찾는다.

    Returns:
        (selected_q, extrapolated) — extrapolated=True면 목표 지점이 DB 관측
        범위(q<=100)를 넘어섰다는 뜻이며, 호출자가 마지막 관측(q=100)을 유지하는
        가정을 적용해야 한다.
    """
    if anchor_q not in SUPPORTED_RATIO_PERCENTS:
        raise ValueError(f"anchor_q는 등록된 q여야 한다: {anchor_q!r}")
    if growth_ratio < 1.0:
        raise ValueError("growth_ratio는 1.0 이상이어야 한다 (미래는 현재보다 작을 수 없다)")
    target = anchor_q * growth_ratio
    if target > 100:
        return 100, True
    for q in SUPPORTED_RATIO_PERCENTS:
        if q >= anchor_q and q >= target:
            return q, False
    return 100, False


def trajectory_point_at_q(points: list[TrajectoryPoint], q_percent: int) -> TrajectoryPoint | None:
    """trajectory에서 특정 q의 지점을 찾는다. 없으면 None (해당 series에 그 q가 없음)."""
    for point in points:
        if point.q_percent == q_percent:
            return point
    return None
