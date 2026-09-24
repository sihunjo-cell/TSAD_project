"""held-out validation을 실제 dev18 DB에 대해 돌려 N/관측%(observed_q_percent) 축을
각각 sweep하고 표/CSV로 출력하는 재현 스크립트.

사용법:
  `python3 -m streamlit_website.ml.experiments.run_holdout_experiments`
  `python3 -m streamlit_website.ml.experiments.run_holdout_experiments <db_path> <csv_dir>`

콘솔에는 지금까지처럼 마크다운 표를 출력하고, 추가로 각 sweep 결과를
`<csv_dir>`(기본값: `streamlit_website/ml/experiments/output/`) 아래 CSV 파일 7개로도
저장한다: `n_sensitivity.csv`, `observed_q_percent_sensitivity.csv`,
`k_sensitivity.csv`, `similarity_metric_comparison.csv`,
`performance_only_validation_gap.csv`(Method 1/2, 2026-09-25 추가 — 아래 참고),
`candidate_diagnostics_detail.csv`/`candidate_diagnostics_summary.csv`(Q1 보완,
2026-09-25 추가 — held-out series x candidate 전체 조합에서 predicted_top_n 경쟁
탈락 사유를 모은 것. 상세는 `holdout_validation.py`의 "Q1 보완" 절 참고).

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


def run_performance_only_validation_gap(database=None) -> list[dict]:
    """"recall_at_n은 성능만 검증한다"는 팀 리뷰 지적 보완(Method 1/2)의 series별 실측치.

    N/관측%는 다른 sweep들의 기준값(N=10, 관측%=20)과 동일하게 고정한다 — 새 축을
    또 하나 섞지 않기 위함이다. `holdout_validation.py` 모듈 docstring의 Method 1/2
    설명을 그대로 따른다.
    """
    series_list = _series_list(database)
    rows = []
    for series in series_list:
        result = evaluate_holdout(
            series, FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP, database=database, top_n=FIXED_N_FOR_Q_SWEEP,
        )
        rows.append({
            "series": series,
            "recall_at_n": result.recall_at_n,
            "recall_policy_at_n": result.recall_policy_at_n,
            "actual_mean_cost_seconds_of_predicted_top_n": result.actual_mean_cost_seconds_of_predicted_top_n,
            "actual_mean_cost_seconds_of_performance_only_ground_truth_top_n": (
                result.actual_mean_cost_seconds_of_performance_only_ground_truth_top_n
            ),
        })
    return rows


REASON_MEANING = {
    "predicted_performance_missing": "유사 과거 사례에서 해당 후보의 성능 근거가 부족함",
    "estimated_cost_missing": "실행시간 관측치가 부족함",
    "infeasible": "데이터 규모·채널 수 등의 구조 조건을 충족하지 못함",
    "usable": "predicted_top_n 경쟁에 실제로 들어감",
}


def run_candidate_diagnostics(database=None) -> tuple[list[dict], list[dict]]:
    """Q1 보완: leave-series-out 18개 series x 전체 candidate 조합에서 각 후보가 왜
    `predicted_top_n` 경쟁에서 빠졌는지(또는 안 빠졌는지)를 모은다.

    N/관측%는 다른 sweep들의 기준값(N=10, 관측%=20)과 동일하게 고정한다 — 진단
    자체는 top_n과 무관하게 후보 루프에서 계산되므로(`holdout_validation.py`
    참고) N값이 결과를 바꾸지 않지만, 다른 표들과 같은 기준으로 재현하기 위해
    고정값을 그대로 쓴다.

    반환값은 (detail_rows, summary_rows) — detail은 series x candidate 한 줄씩,
    summary는 사유별 발생 횟수/비율/의미.
    """
    series_list = _series_list(database)
    detail_rows = []
    reason_counts: dict[str, int] = {}
    for series in series_list:
        result = evaluate_holdout(
            series, FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP, database=database, top_n=FIXED_N_FOR_Q_SWEEP,
        )
        for candidate_id, diagnosis in result.candidate_diagnostics.items():
            detail_rows.append({
                "held_out_series": series,
                "candidate_id": candidate_id,
                "predicted_performance": diagnosis["predicted_performance"],
                "estimated_cost": diagnosis["estimated_cost"],
                "feasible": diagnosis["feasible"],
                "reason": diagnosis["reason"],
            })
            reason_counts[diagnosis["reason"]] = reason_counts.get(diagnosis["reason"], 0) + 1

    total = len(detail_rows)
    summary_rows = [
        {
            "제외 사유": reason,
            "발생 횟수": count,
            "비율": f"{count / total * 100:.1f}%" if total else "0.0%",
            "의미": REASON_MEANING.get(reason, ""),
        }
        for reason, count in sorted(reason_counts.items(), key=lambda item: -item[1])
    ]
    return detail_rows, summary_rows


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
    # utf-8-sig: 한글 헤더/값(제외 사유, 의미 등)이 있는 CSV를 윈도우 엑셀에서 열 때
    # BOM 없는 utf-8은 인코딩을 잘못 추측해 글자가 깨진다 — BOM을 붙여 방지한다.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
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

    gap_rows = run_performance_only_validation_gap(database)
    print(
        f"\n## 성능만 검증 지적 보완: Method 1/2 (관측%={FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP}, "
        f"N={FIXED_N_FOR_Q_SWEEP} 고정)\n"
    )
    _print_table(gap_rows)
    _write_csv(gap_rows, csv_dir / "performance_only_validation_gap.csv")
    recall_vals = [r["recall_at_n"] for r in gap_rows if r["recall_at_n"] is not None]
    recall_policy_vals = [r["recall_policy_at_n"] for r in gap_rows if r["recall_policy_at_n"] is not None]
    pred_costs = [
        r["actual_mean_cost_seconds_of_predicted_top_n"] for r in gap_rows
        if r["actual_mean_cost_seconds_of_predicted_top_n"] is not None
    ]
    gt_costs = [
        r["actual_mean_cost_seconds_of_performance_only_ground_truth_top_n"] for r in gap_rows
        if r["actual_mean_cost_seconds_of_performance_only_ground_truth_top_n"] is not None
    ]
    if recall_vals and recall_policy_vals:
        print(
            f"\nmean recall_at_n={statistics.mean(recall_vals):.4f} vs "
            f"mean recall_policy_at_n={statistics.mean(recall_policy_vals):.4f}"
        )
    if pred_costs and gt_costs:
        cheaper = sum(1 for r in gap_rows if (
            r["actual_mean_cost_seconds_of_predicted_top_n"] is not None
            and r["actual_mean_cost_seconds_of_performance_only_ground_truth_top_n"] is not None
            and r["actual_mean_cost_seconds_of_predicted_top_n"]
            < r["actual_mean_cost_seconds_of_performance_only_ground_truth_top_n"]
        ))
        print(
            f"mean actual cost: predicted_top_n={statistics.mean(pred_costs):.2f}s vs "
            f"performance_only_ground_truth_top_n={statistics.mean(gt_costs):.2f}s "
            f"(predicted_top_n이 더 싼 series: {cheaper}/{len(gap_rows)})"
        )

    detail_rows, summary_rows = run_candidate_diagnostics(database)
    print(
        f"\n## Q1 보완: 후보별 제외 사유 (관측%={FIXED_OBSERVED_Q_PERCENT_FOR_N_SWEEP}, "
        f"N={FIXED_N_FOR_Q_SWEEP} 고정, {len(_series_list(database))}개 series x 전체 candidate)\n"
    )
    _print_table(summary_rows)
    _write_csv(detail_rows, csv_dir / "candidate_diagnostics_detail.csv")
    _write_csv(summary_rows, csv_dir / "candidate_diagnostics_summary.csv")

    print(f"\nCSV 저장 위치: {csv_dir}")


if __name__ == "__main__":
    main()
