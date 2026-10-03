"""유사한 과거 series의 이후 q trajectory를 연결한다.

DB는 series마다 학습 구간의 5·10·20·40·60·80·100%에서만 성능을 쟀다. 미래 구간은
운영 기간 끝까지 모을 행 수를 100%로 본 같은 비율에 두므로(`pipeline.build_future_stages`),
과거 series의 같은 비율 결과를 그대로 읽는다. 같은 비율은 같은 행 수가 아니라
"모을 데이터 중 같은 몫"이라는 뜻이다.

training-free 모델(target_use="training_free")은 여러 논리 q가 물리적으로 같은
실행(run_id)을 재사용한다 — `results → cost_result_links → cost_executions`의
`run_id`로 이를 구분한다. 이 모듈은 trajectory point마다 사용된 run_id 집합을
그대로 노출해서, 상위(performance.py/cost.py)가 "새로 관측된 지점"과 "이전과
같은 물리 실행을 재사용한 지점"을 구분할 수 있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def observed_point_up_to(points: list[TrajectoryPoint], q_percent: int) -> TrajectoryPoint | None:
    """q_percent 이하에서 성능을 잰 가장 큰 q의 지점. 없으면 None.

    목표 비율에 결과가 없으면 그보다 적은 데이터에서 잰 마지막 값을 유지한다고 본다.
    더 작은 비율에도 결과가 없으면(예: 적은 데이터에서 실행 불가였던 PaAno) 추정하지 않는다.
    """
    observed = [point for point in points if point.q_percent <= q_percent and point.vus_pr is not None]
    return observed[-1] if observed else None
