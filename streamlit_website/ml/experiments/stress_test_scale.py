"""팀 피드백 문서가 목표로 하는 미래/프로덕션 규모(~3,000개 후보)에서, 이 held-out
harness가 구조적으로 동작하고 ML 계산 시간이 감당할 만한지 **직접 실행해서** 확인하는
스트레스 테스트.

**왜 필요한가**: 실제 dev18 DB는 지금 45개 후보뿐이다(진짜 실측). 3,000개 규모에서의
Recall/Precision/축소율 "실측값"은 그 규모의 실제 실행 결과가 쌓이기 전까지는 만들
수 없다 — 없는 데이터를 만들어내지 않는다. 대신 이 스크립트는 **합성 DB**로 (1) 코드가
후보 수를 하드코딩하지 않고 실제로 3,000개 규모까지 그대로 동작하는지, (2) N을
50/100/200/500까지 올려도 ML 계산 시간이 어느 정도인지를 검증한다. 이건 "3,000
규모에서의 Recall@N이 얼마다"라는 결론이 아니라 "구조와 성능이 그 규모를 감당하는가"
에 대한 답이다 (RESULTS.md에 이 구분을 명시한다).

사용법: `python3 -m streamlit_website.ml.experiments.stress_test_scale`
"""

from __future__ import annotations

import random
import sqlite3
import tempfile
import time
from pathlib import Path

from streamlit_website.ml.experiments.holdout_validation import evaluate_holdout, sweep_n_sensitivity
from streamlit_website.ml.test_ml_pipeline import _create_schema

SERIES_COUNT = 18          # 실제 dev18과 동일한 series 수
CANDIDATE_COUNT = 3000     # 팀 피드백 문서가 목표로 하는 미래/프로덕션 규모
OBSERVED_Q_PERCENT = 20
N_VALUES = (50, 100, 200, 500)


def _build_synthetic_database(path: Path, *, series_count: int, candidate_count: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    with sqlite3.connect(path) as connection:
        _create_schema(connection)
        prefix_rows = []
        for series_index in range(series_count):
            series = f"s{series_index:03d}"
            for q_percent, observed_row in ((OBSERVED_Q_PERCENT, 200), (100, 1000)):
                prefix_feature_id = f"{series}-{q_percent}"
                prefix_rows.append((
                    prefix_feature_id, f"csv-{prefix_feature_id}", f"{prefix_feature_id}.csv", "famA", series,
                    q_percent, observed_row, observed_row,
                    rng.uniform(0.1, 2.0), rng.uniform(-0.5, 0.9), rng.uniform(0.0, 0.9),
                    rng.uniform(0.5, 3.0), rng.uniform(0.1, 1.0), rng.uniform(0.0, 0.5), rng.uniform(0.0, 1.0),
                ))
        connection.executemany(
            "INSERT INTO prefix_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", prefix_rows,
        )

        result_rows = []
        cost_link_rows = []
        cost_execution_rows = []
        for candidate_index in range(candidate_count):
            config_id = f"c{candidate_index:05d}"
            for series_index in range(series_count):
                series = f"s{series_index:03d}"
                # q=100(ground truth)만 채운다 — 예측에 필요한 anchor는 다른 series의 q=100이므로
                # 이것으로 충분하다(현재 관측 q=20 자체는 prefix_features에만 필요, results엔 불필요).
                result_rows.append((f"{series}-100", config_id, 0, "", 1, "complete", rng.uniform(0.3, 0.95)))
                # cost 데이터도 함께 채운다 — candidate_selection.py와 동일한 usability 기준
                # (feasible+performance+cost 모두 있어야 top-N 경쟁에 들어감)을 이 스트레스
                # 테스트에서도 만족시켜야, 실제 harness가 3,000개 규모에서 겪을 계산량을 그대로 반영한다.
                run_id = f"run-{config_id}-{series}"
                cost_link_rows.append((f"{series}-100", config_id, 0, "", run_id))
                cost_execution_rows.append(
                    (run_id, f"{series}-100", "complete", None, rng.uniform(1.0, 60.0),
                     None, None, None, None, None, 1000)
                )
        connection.executemany("INSERT INTO results VALUES (?,?,?,?,?,?,?)", result_rows)
        connection.executemany("INSERT INTO cost_result_links VALUES (?,?,?,?,?)", cost_link_rows)
        connection.executemany(
            "INSERT INTO cost_executions VALUES (?,?,?,?,?,?,?,?,?,?,?)", cost_execution_rows,
        )
        connection.commit()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        database = Path(tmp) / "stress.sqlite3"
        build_start = time.perf_counter()
        _build_synthetic_database(database, series_count=SERIES_COUNT, candidate_count=CANDIDATE_COUNT)
        print(f"합성 DB 생성: series={SERIES_COUNT}, candidates={CANDIDATE_COUNT} "
              f"({time.perf_counter() - build_start:.2f}s)")

        held_out_series = "s000"
        single_start = time.perf_counter()
        single = evaluate_holdout(held_out_series, OBSERVED_Q_PERCENT, database=database, top_n=100)
        single_elapsed = time.perf_counter() - single_start
        print(f"\n단일 evaluate_holdout(top_n=100): {single_elapsed:.2f}s "
              f"(total_candidates={single.total_candidates}, ml_runtime_seconds={single.ml_runtime_seconds:.2f}s)")

        sweep_start = time.perf_counter()
        results = sweep_n_sensitivity(held_out_series, OBSERVED_Q_PERCENT, list(N_VALUES), database=database)
        sweep_elapsed = time.perf_counter() - sweep_start
        print(f"\nN={N_VALUES} sweep 전체: {sweep_elapsed:.2f}s\n")
        print("N | recall_at_n | candidate_reduction_ratio | ml_runtime_seconds")
        print("--- | --- | --- | ---")
        for result in results:
            print(f"{result.top_n} | {result.recall_at_n:.3f} | "
                  f"{result.candidate_reduction_ratio:.3f} | {result.ml_runtime_seconds:.2f}")


if __name__ == "__main__":
    main()
