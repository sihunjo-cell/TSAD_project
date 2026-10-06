"""웹사이트 추천을 GHL25에서 Dev18 롤링 백테스트와 같은 방식으로 점검한다.

GHL 파일마다 학습 구간의 5%에서 운영을 시작해 100일 뒤 100%에 닿는다고 본다. 추론량은 test 구간
20개 분량을 기간에 고르게 나눈다. 데이터가 5·10·20·40·60·80%에 닿을 때마다 그 prefix로 웹사이트와
같은 ML·DP를 다시 돌리고, 다음 비율까지 계획의 첫 구간을 실행한다. 실현값은 배치한 후보의 GHL 실측
VUS-PR(seed 평균)을 기간으로 가중한 값이다.

비용은 웹사이트의 추정(DB 실측 기반)을 정답으로 둔다. GHL을 돌린 장치는 DB와 달라 실행 시간을 견줄 수
없으므로, 이 점검은 성능 예측과 경로 선택만 본다. 예산은 남은 몫을 다음 재계획에 넘긴다.

전략은 웹사이트 추천(빠듯한 예산 규칙 포함), 규칙 없는 DP, ML 기준 가장 싼 계획, 실측 VUS-PR로 푼 사후 최적이다.
예산은 파일마다 가장 싼 계획 비용의 1.5·10·100배와 제한 없음이다. p는 파일 단위 Wilcoxon이다.

    python -m tests.ghl_main.evaluate_ghl_service
"""

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy
import pandas
from scipy.stats import wilcoxon

from src.data_split.load_ghl_series import TRAIN_BOUNDARY_PATTERN
from streamlit_website.DP.planner import optimize_plan
from streamlit_website.DP.recommend import cheapest_plan, recommend
from streamlit_website.db_connection.filter_candidates import (
    build_ml_input, filter_candidates, load_candidates, profile_normal_data,
)
from streamlit_website.ml.db import DEFAULT_DATABASE_PATH
from streamlit_website.ml.dp_input import build_dp_input
from streamlit_website.ml.pipeline import run_ml_pipeline
from tests.ghl_main.run_ghl_candidates import (
    DATA_DIRECTORY, EXPERIMENT_DIRECTORY, SERVICE_RATIOS, ghl_file_names, ghl_series, read_rows, run_paths,
)


OPERATING_DAYS = 100
INFERENCE_SEGMENTS = 20
BUDGET_MULTIPLES = (1.5, 10, 100, None)
OUTPUT_DIRECTORY = EXPERIMENT_DIRECTORY / "results" / "service_backtest"
STRATEGIES = {
    "website": lambda payload: recommend(payload, optimize_plan(payload))[0],
    "dp": optimize_plan,
    "cheapest": lambda payload: cheapest_plan(payload, optimize_plan(payload)),
}


def load_actuals(series) -> dict:
    """{(후보, 비율): seed 평균 VUS-PR}. 학습 없는 후보는 한 번 잰 값을 모든 비율에 쓴다."""
    runs_path, vus_path = run_paths(series)
    vus = {row["score_file"]: float(row["vus_pr"]) for row in read_rows(vus_path)}
    measured = defaultdict(list)
    for row in read_rows(runs_path):
        if row["status"] != "complete" or row["score_file"] not in vus:
            continue
        ratios = (int(row["ratio"]),) if row["target_use"] == "fit_full_prefix" else SERVICE_RATIOS
        for ratio in ratios:
            measured[f"{row['config_id']}::{row['score_variant']}", ratio].append(vus[row["score_file"]])
    return {key: float(numpy.mean(values)) for key, values in measured.items()}


def simulate(plan_input, choose, budget, actual, *, segments) -> dict:
    """비율마다 다시 계획하고 첫 구간을 실행한다. segments는 (비율, 시작일, 끝일)이다.

    비용은 계획 경로가 쓴 웹사이트 추정이고, 성능은 실측이다.
    """
    current, spent, realized, trainings, missing, steps = None, 0.0, 0.0, 0, 0, []
    for ratio, start, end in segments:
        payload = {**plan_input(ratio, current), "budget": None if budget is None else max(budget - spent, 0.0)}
        plan = choose(payload)
        if not plan or plan["status"] not in ("ok", "over_budget"):
            raise RuntimeError(f"q{ratio}에서 계획을 만들지 못했다: {plan and plan.get('reason')}")
        stages = payload["stages"]
        if stages[0]["ratio_percent"] != ratio or (
                len(stages) > 1 and not math.isclose(stages[1]["elapsed_days"], end - start, rel_tol=1e-9)):
            raise ValueError(f"q{ratio}: 계획 구간이 재계산 시점과 맞지 않는다")
        step = plan["path"][0]
        if step["action"] != "keep":
            current, trainings = (step["candidate_id"], ratio), trainings + 1
        vus = actual.get(current)
        missing += vus is None
        spent += step["training_cost"] + step["inference_cost"]
        realized += (vus or 0.0) * (end - start) / OPERATING_DAYS
        steps.append(f"q{ratio}:{step['action']}:{step['candidate_id']}")
    return {"vus_pr": realized, "spent_seconds": spent, "trainings": trainings,
            "missing_actual": missing, "path": " | ".join(steps)}


def oracle_payload(base, actual, budget) -> dict:
    """첫 계획 시점의 DP 입력에서 예측 VUS-PR 자리에 실측값을 넣는다. 비용은 그대로다."""
    ratios = [stage["ratio_percent"] for stage in base["stages"]]
    training = [{**option, "predicted_performance": actual[option["candidate_id"], ratios[option["stage"]]]}
                for option in base["training_options"]
                if (option["candidate_id"], ratios[option["stage"]]) in actual]
    kept = {(option["stage"], option["candidate_id"]) for option in training}
    return {**base, "budget": budget, "training_options": training,
            "inference_options": [row for row in base["inference_options"]
                                  if (row["trained_stage"], row["candidate_id"]) in kept]}


def evaluate_file(series, csv_file, frame, actual, database) -> dict:
    """한 파일의 전략·예산별 실현값과, 비율별 ML 예측 대 실측 VUS-PR을 만든다."""
    sensors = [column for column in frame.columns if column != "Label"]
    boundary = int(TRAIN_BOUNDARY_PATTERN.search(csv_file).group(1))
    test_rows = len(frame) - boundary
    pool = filter_candidates(load_candidates(database), channel_count=len(sensors), gpu_available=True)
    models = {f"{row['config_id']}::{row['head']}": row["model"] for row in pool}
    start_rows = SERVICE_RATIOS[0] * boundary // 100
    conditions = {
        "collection_rows_per_second": (boundary - start_rows) / (OPERATING_DAYS * 86400),
        "inference_rows_per_day": INFERENCE_SEGMENTS * test_rows / OPERATING_DAYS,
        "inference_batch_length": test_rows, "budget": None,
        "cpu_hourly_cost": 0.0, "gpu_hourly_cost": 0.0, "gpu_memory_gib": 24.0,
    }
    rows_per_day = conditions["collection_rows_per_second"] * 86400
    starts = [(ratio * boundary // 100 - start_rows) / rows_per_day for ratio in SERVICE_RATIOS]
    segments = list(zip(SERVICE_RATIOS, starts, starts[1:] + [OPERATING_DAYS]))
    elapsed = dict(zip(SERVICE_RATIOS, starts))
    features, payloads = {}, {}

    def plan_input(ratio, current):
        """데이터가 ratio%일 때 웹사이트가 DP에 넘기는 표. current는 (후보, 학습 비율)이다."""
        if (ratio, current) not in payloads:
            prefix = frame.iloc[:ratio * boundary // 100]
            if ratio not in features:
                summary, channels = profile_normal_data(prefix, sensors)
                features[ratio] = {"summary": summary, "channels": channels}
            current_model = {"candidate_id": None, "checkpoint": None, "trained_rows": None, "last_trained_at": None}
            if current:
                current_model.update(candidate_id=current[0], checkpoint=f"{current[0]}@q{current[1]}",
                                     trained_rows=current[1] * boundary // 100)
            ml_input = build_ml_input(prefix, sensors, pool, current_model=current_model, operating_conditions={
                **conditions, "operating_days": OPERATING_DAYS - elapsed[ratio]})
            ml_input["current_features"] = features[ratio]
            payloads[ratio, current] = build_dp_input(run_ml_pipeline(ml_input, db_path=database), ml_input)
        return payloads[ratio, current]

    base = plan_input(SERVICE_RATIOS[0], None)
    cheapest_cost = optimize_plan(oracle_payload(base, actual, None))["frontier"][0]["total_cost"]
    plans = []
    for multiple in BUDGET_MULTIPLES:
        budget = None if multiple is None else multiple * cheapest_cost
        results = {name: simulate(plan_input, choose, budget, actual, segments=segments)
                   for name, choose in STRATEGIES.items()}
        oracle = optimize_plan(oracle_payload(base, actual, budget))
        results["oracle"] = {
            "vus_pr": oracle["timeline_performance"], "spent_seconds": oracle["total_cost"],
            "trainings": oracle["trainings"], "missing_actual": 0,
            "path": " | ".join(f"q{SERVICE_RATIOS[step['stage']]}:{step['action']}:{step['candidate_id']}"
                               for step in oracle["path"]),
        }
        plans += [{"series": series, "csv_file": csv_file, "budget_multiple": multiple or "unlimited",
                   "budget_seconds": "" if budget is None else budget, "strategy": name, **result,
                   "over_budget": budget is not None and result["spent_seconds"] > budget * (1 + 1e-9)}
                  for name, result in results.items()]

    predictions = [{
        "series": series, "ratio": ratio, "candidate_id": option["candidate_id"],
        "model": models[option["candidate_id"]], "predicted_vus_pr": option["predicted_performance"],
        "actual_vus_pr": actual.get((option["candidate_id"], ratio)),
    } for ratio in SERVICE_RATIOS for option in plan_input(ratio, None)["training_options"] if option["stage"] == 0]
    missing = [f"{row['config_id']}::{row['head']}@q{ratio}" for row in pool if row["status"] == "eligible"
               for ratio in SERVICE_RATIOS if (f"{row['config_id']}::{row['head']}", ratio) not in actual]
    return {"plans": plans, "predictions": predictions, "missing": missing}


def paired_p_value(rows, reference) -> float | str:
    differences = [rows[series]["vus_pr"] - reference[series]["vus_pr"] for series in rows if series in reference]
    if not differences:
        return ""
    return 1.0 if not any(differences) else float(wilcoxon(differences).pvalue)


def summarize_plans(plan_rows) -> list[dict]:
    by_key = defaultdict(dict)
    for row in plan_rows:
        by_key[row["budget_multiple"], row["strategy"]][row["series"]] = row
    return [{
        "budget_multiple": multiple, "strategy": strategy, "files": len(rows),
        "mean_vus_pr": float(numpy.mean([row["vus_pr"] for row in rows.values()])),
        "median_vus_pr": float(numpy.median([row["vus_pr"] for row in rows.values()])),
        "mean_spent_seconds": float(numpy.mean([row["spent_seconds"] for row in rows.values()])),
        "over_budget_files": sum(row["over_budget"] for row in rows.values()),
        **{f"p_vs_{other}": "" if other == strategy else paired_p_value(rows, by_key.get((multiple, other), {}))
           for other in ("cheapest", "dp")},
    } for (multiple, strategy), rows in by_key.items()]


def summarize_predictions(prediction_rows) -> dict:
    """새 현장에서 ML 성능 예측이 얼마나 맞았는지: VUS-PR 절대 오차와 예측 1위 손실."""
    measured = [row for row in prediction_rows if row["actual_vus_pr"] is not None]
    groups = defaultdict(list)
    for row in measured:
        groups[row["series"], row["ratio"]].append(row)
    top1_losses = [max(row["actual_vus_pr"] for row in rows)
                   - max(rows, key=lambda row: row["predicted_vus_pr"])["actual_vus_pr"]
                   for rows in groups.values()]
    return {
        "rows": len(measured),
        "vus_pr_mean_absolute_error": float(numpy.mean(
            [abs(row["predicted_vus_pr"] - row["actual_vus_pr"]) for row in measured])) if measured else None,
        "top1_loss_mean": float(numpy.mean(top1_losses)) if top1_losses else None,
    }


def write_csv(path, rows) -> None:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_text(stream.getvalue(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--data-directory", type=Path, default=DATA_DIRECTORY)
    arguments = parser.parse_args()
    if not arguments.db.is_file():
        raise SystemExit(f"recommendation DB가 없다: {arguments.db}")
    plan_rows, prediction_rows = [], []
    for csv_name in ghl_file_names():
        series = ghl_series(csv_name)
        actual = load_actuals(series)
        if not actual:
            print(f"경고: GHL {series}에 채점 결과가 없어 건너뛴다", flush=True)
            continue
        result = evaluate_file(series, csv_name, pandas.read_csv(arguments.data_directory / csv_name),
                               actual, arguments.db)
        plan_rows += result["plans"]
        prediction_rows += result["predictions"]
        note = f", 실측이 없는 후보·비율 {len(result['missing'])}건 (예: {result['missing'][0]})" if result["missing"] else ""
        print(f"GHL {series}: 점검 끝{note}", flush=True)
    if not plan_rows:
        raise SystemExit("점검할 파일이 없다")
    summary = summarize_plans(plan_rows)
    prediction_summary = summarize_predictions(prediction_rows)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIRECTORY / "plans.csv", plan_rows)
    write_csv(OUTPUT_DIRECTORY / "predictions.csv", prediction_rows)
    write_csv(OUTPUT_DIRECTORY / "summary.csv", summary)
    (OUTPUT_DIRECTORY / "prediction_summary.json").write_text(
        json.dumps(prediction_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pandas.DataFrame(summary).to_string(index=False))
    print(json.dumps(prediction_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
