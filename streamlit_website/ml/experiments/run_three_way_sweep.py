"""N x k x 관측%(observed_q_percent) 3중 교차 sweep — Q6(한계3)에서 제안된 히트맵용 재현 스크립트.

`run_holdout_experiments.py`의 개별 sweep들은 팀 피드백 문서의 "N과 관측%는 서로 다른
축이므로 섞지 않는다" 원칙에 따라 항상 한 축만 바꾸고 나머지는 고정했다. 이 스크립트는
그 원칙을 지키면서도 세 축을 모두 도는 교차표를 만든다 — N x k 히트맵을 관측%별로
따로 하나씩(q_percent를 "패싯"으로 고정) 만드는 방식이다: 한 히트맵 안에서는 N/k
두 축만 비교되고, 관측%는 히트맵 자체를 고르는 축이라 두 축을 섞지 않는다는 원칙과
충돌하지 않는다.

사용법:
  `python3 -m streamlit_website.ml.experiments.run_three_way_sweep`
  `python3 -m streamlit_website.ml.experiments.run_three_way_sweep <db_path> <csv_path>`

DB는 한 번만 읽고(다른 sweep 함수들과 동일한 캐시 재사용 패턴) N x k x q 모든 조합에
그대로 재사용한다 — 조합 수가 늘어나도 DB 재조회 비용이 늘지 않는다.
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

from streamlit_website.ml import db
from streamlit_website.ml.config import DEFAULT_CONFIG
from streamlit_website.ml.experiments.holdout_validation import evaluate_holdout
from streamlit_website.ml.experiments.run_holdout_experiments import _series_list

N_VALUES = (5, 10, 20, 45)
K_VALUES = (1, 3, 5, 10, 15)
OBSERVED_Q_PERCENT_VALUES = (5, 10, 20, 40, 60, 80)

DEFAULT_CSV_PATH = Path(__file__).parent / "output" / "three_way_sweep.csv"


def run_three_way_sweep(
    database=None, n_values=N_VALUES, k_values=K_VALUES, q_values=OBSERVED_Q_PERCENT_VALUES,
) -> list[dict]:
    """N x k x 관측% 모든 조합에서 series 전체 평균 Recall@N을 계산한다.

    한 series에 해당 관측%(q_percent) 행이 없으면(팀 피드백에서 요구한 것과 달리 이
    DB는 series마다 있는 q_percent가 다를 수 있음) 그 series는 건너뛴다 — 임의로
    대체하지 않는다(다른 sweep 함수들과 동일한 원칙).
    """
    series_list = _series_list(database)
    all_prefixes = db.load_historical_prefixes(database)
    results_by_candidate = db.load_all_results_grouped_by_candidate(database)
    candidate_definitions = db.load_candidate_definitions(database)
    channel_count_by_prefix = db.load_channel_count_by_prefix(database)

    rows = []
    for k in k_values:
        config = replace(DEFAULT_CONFIG, similarity_knn_k=k)
        for q in q_values:
            # q_percent는 series/DB 로드와 무관하게 evaluate_holdout 호출마다 바뀌므로,
            # N loop보다 바깥에서 한 번만 series별로 결과를 계산해 N값들에 재사용한다.
            per_series_by_n: dict[int, list[float]] = {n: [] for n in n_values}
            per_series_evaluated: dict[int, int] = {n: 0 for n in n_values}
            for series in series_list:
                for n in n_values:
                    try:
                        result = evaluate_holdout(
                            series, q, database=database, config=config, top_n=n,
                            _all_prefixes=all_prefixes, _results_by_candidate=results_by_candidate,
                            _candidate_definitions=candidate_definitions,
                            _channel_count_by_prefix=channel_count_by_prefix,
                        )
                    except ValueError:
                        continue
                    per_series_evaluated[n] += 1
                    if result.recall_at_n is not None:
                        per_series_by_n[n].append(result.recall_at_n)
            for n in n_values:
                recalls = per_series_by_n[n]
                rows.append({
                    "N": n, "k": k, "observed_q_percent": q,
                    "mean_recall_at_n": statistics.mean(recalls) if recalls else None,
                    "series_evaluated": per_series_evaluated[n],
                })
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    database = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CSV_PATH

    start = time.perf_counter()
    rows = run_three_way_sweep(database)
    elapsed = time.perf_counter() - start

    _write_csv(rows, csv_path)
    print(f"{len(rows)}개 조합(N x k x 관측%) 계산 완료, {elapsed:.2f}초 소요")
    print(f"CSV 저장 위치: {csv_path}")


if __name__ == "__main__":
    main()
