"""held-out validation을 실제 dev18 DB에 대해 돌려 N/관측%(observed_q_percent) 축을
각각 sweep하고 표/CSV로 출력하는 재현 스크립트.

사용법:
  `python3 -m streamlit_website.ml.experiments.run_holdout_experiments`
  `python3 -m streamlit_website.ml.experiments.run_holdout_experiments <db_path> <csv_dir>`

콘솔에는 지금까지처럼 마크다운 표를 출력하고, 추가로 각 sweep 결과를
`<csv_dir>`(기본값: `streamlit_website/ml/experiments/output/`) 아래 CSV 파일 4개로도
저장한다: `n_sensitivity.csv`, `observed_q_percent_sensitivity.csv`,
`k_sensitivity.csv`, `similarity_metric_comparison.csv`.

**N과 관측%(observed_q_percent)는 서로 다른 축이므로 항상 한쪽을 고정하고 다른
쪽만 바꾼다** (팀 피드백 문서의 "두 축을 섞지 않는다" 원칙). 이 스크립트는 그 둘을
분리된 두 표로 출력한다.

현재 실제 DB 규모(45 candidates, 18 series)에 맞춰 N=5/10/20/45(=전체)를 쓴다 —
피드백 문서가 예시로 든 50/100/200/500은 이 규모를 넘어서므로 쓰지 않는다
(3,000-후보 규모는 미래/프로덕션 목표치이며, 지금은 구조만 검증한다는 확정 사항).
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

from streamlit_website.ml import db
from streamlit_website.ml.config import SUPPORTED_SIMILARITY_METRICS
from streamlit_website.ml.experiments.holdout_validation import (
    compare_similarity_metrics, evaluate_holdout, sweep_k_sensitivity,
)

N_VALUES = (5, 10, 20, 45)
OBSERVED_Q_PERCENT_VALUES = (5, 10, 20, 40, 60, 80)
K_VALUES = (1, 3, 5, 10)
FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP = 20
FIXED_N_FOR_Q_SWEEP = 10
FIXED_N_FOR_K_SWEEP = 10
FIXED_N_FOR_METRIC_COMPARISON = 10


def _series_list(database=None) -> list[str]:
    return sorted({row["series"] for row in db.load_historical_prefixes(database)})


def run_n_sensitivity(database=None) -> list[dict]:
    series_list = _series_list(database)
    rows = []
    for n in N_VALUES:
        recalls, runtimes = [], []
        for series in series_list:
            result = evaluate_holdout(series, FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP, database=database, top_n=n)
            runtimes.append(result.ml_runtime_seconds)
            if result.recall_at_n is not None:
                recalls.append(result.recall_at_n)
        total_candidates = evaluate_holdout(
            series_list[0], FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP, database=database, top_n=n,
        ).total_candidates
        rows.append({
            "N": n,
            "mean_recall_at_n": statistics.mean(recalls) if recalls else None,
            "candidate_reduction_ratio": max(0.0, 1 - n / total_candidates) if total_candidates else None,
            "mean_ml_runtime_seconds": statistics.mean(runtimes),
            "series_evaluated": len(recalls),
        })
    return rows


def run_observed_percent_sensitivity(database=None) -> list[dict]:
    series_list = _series_list(database)
    rows = []
    for q in OBSERVED_Q_PERCENT_VALUES:
        recalls, runtimes = [], []
        for series in series_list:
            try:
                result = evaluate_holdout(series, q, database=database, top_n=FIXED_N_FOR_Q_SWEEP)
            except ValueError:
                continue  # 이 series에 해당 q_percent 관측이 없음 (건너뛴다, 임의 대체 없음)
            runtimes.append(result.ml_runtime_seconds)
            if result.recall_at_n is not None:
                recalls.append(result.recall_at_n)
        rows.append({
            "observed_q_percent": q,
            "mean_recall_at_n10": statistics.mean(recalls) if recalls else None,
            "mean_ml_runtime_seconds": statistics.mean(runtimes) if runtimes else None,
            "series_evaluated": len(recalls),
        })
    return rows


def run_k_sensitivity(database=None) -> list[dict]:
    series_list = _series_list(database)
    rows = []
    for k in K_VALUES:
        recalls, matches_found = [], []
        for series in series_list:
            result = sweep_k_sensitivity(
                series, FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP, [k], database=database, top_n=FIXED_N_FOR_K_SWEEP,
            )[0]
            matches_found.append(result.similarity_matches_found)
            if result.recall_at_n is not None:
                recalls.append(result.recall_at_n)
        rows.append({
            "k": k,
            "mean_recall_at_n10": statistics.mean(recalls) if recalls else None,
            "mean_matches_found": statistics.mean(matches_found),
            "series_evaluated": len(recalls),
        })
    return rows


def run_similarity_metric_comparison(database=None) -> list[dict]:
    series_list = _series_list(database)
    per_metric_recalls = {metric: [] for metric in SUPPORTED_SIMILARITY_METRICS}
    zero_match_counts = {metric: 0 for metric in SUPPORTED_SIMILARITY_METRICS}
    for series in series_list:
        comparison = compare_similarity_metrics(
            series, FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP, database=database, top_n=FIXED_N_FOR_METRIC_COMPARISON,
        )
        for metric, result in comparison.items():
            if result.similarity_matches_found == 0:
                zero_match_counts[metric] += 1
            elif result.recall_at_n is not None:
                per_metric_recalls[metric].append(result.recall_at_n)
    return [
        {
            "similarity_metric": metric,
            "mean_recall_at_n10": statistics.mean(recalls) if recalls else None,
            "series_with_zero_matches": zero_match_counts[metric],
            "series_evaluated": len(recalls),
        }
        for metric, recalls in per_metric_recalls.items()
    ]


DEFAULT_CSV_DIR = Path(__file__).parent / "output"


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("(결과 없음)")
        return
    headers = list(rows[0].keys())
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for row in rows:
        print(" | ".join(str(row[h]) for h in headers))


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    database = sys.argv[1] if len(sys.argv) > 1 else None
    csv_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CSV_DIR

    n_rows = run_n_sensitivity(database)
    print(f"## N-sensitivity (관측% 고정={FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP}%)\n")
    _print_table(n_rows)
    _write_csv(n_rows, csv_dir / "n_sensitivity.csv")

    q_rows = run_observed_percent_sensitivity(database)
    print(f"\n## 관측%(observed_q_percent)-sensitivity (N 고정={FIXED_N_FOR_Q_SWEEP})\n")
    _print_table(q_rows)
    _write_csv(q_rows, csv_dir / "observed_q_percent_sensitivity.csv")

    k_rows = run_k_sensitivity(database)
    print(f"\n## k-sensitivity (관측%={FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP}, N={FIXED_N_FOR_K_SWEEP} 고정)\n")
    _print_table(k_rows)
    _write_csv(k_rows, csv_dir / "k_sensitivity.csv")

    metric_rows = run_similarity_metric_comparison(database)
    print(f"\n## Euclidean vs Cosine (관측%={FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP}, N={FIXED_N_FOR_METRIC_COMPARISON} 고정)\n")
    _print_table(metric_rows)
    _write_csv(metric_rows, csv_dir / "similarity_metric_comparison.csv")

    print(f"\nCSV 저장 위치: {csv_dir}")


if __name__ == "__main__":
    main()
