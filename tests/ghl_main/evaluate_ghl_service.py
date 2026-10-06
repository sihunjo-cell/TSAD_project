"""웹사이트 추천을 GHL에서 Dev18 롤링 백테스트와 같은 방식으로 점검한다.

GHL 파일마다 학습 구간의 5%에서 운영을 시작해 100일 뒤 100%에 닿는다고 본다. 추론량은 test 구간
20개 분량을 기간에 고르게 나눈다. 데이터가 5·10·20·40·60·80%에 닿을 때마다 그 prefix로 웹사이트와
같은 ML·DP를 다시 돌리고, 다음 비율까지 계획의 첫 구간을 실행한다. 실현값은 배치한 후보의 GHL 실측
VUS-PR(seed 평균)을 기간으로 가중한 값이다. 비용은 웹사이트의 추정(DB 실측 기반)을 정답으로 둔다.
GHL을 돌린 장치는 DB와 달라 실행 시간을 견줄 수 없으므로 이 점검은 성능 예측과 경로 선택만 본다.

전략
- website: 웹사이트 추천(유사도 ML + DP + 빠듯한 예산 규칙)
- dp: 규칙 없는 DP
- cheapest: ML 기준 가장 싼 계획
- no_similarity: 예측 자리에 Dev18 series의 같은 비율 VUS-PR 동일 가중 평균을 넣고 웹사이트와 같이 고른 계획
- random: 비율마다 후보를 무작위로 골라 그 비율에서 학습할 때의 기대값(예산은 보지 않는다)
- oracle: 실측 VUS-PR로 푼 사후 최적
예산은 파일마다 실측 기준 가장 싼 계획 비용의 1.5·10·100배와 제한 없음이다. p는 파일 단위 Wilcoxon이다.

파일은 모든 실행이 끝 상태(성공 또는 실패)에 닿고 성공한 점수를 모두 채점했을 때 점검한다. 늘 실패하는
후보가 있는 파일도 빼지 않는다. 계획이 실패한 후보를 고르면 그 구간 VUS-PR을 0으로 세고 missing_actual로 드러낸다.

멈춤 규칙은 결과를 보기 전에 정했다. 점검한 파일이 10·15·20·25개일 때만 보고, 모든 예산에서 웹사이트 추천과
cheapest·no_similarity·dp의 파일별 차이 평균의 95% 신뢰구간(t) 반폭이 0.03 이하이면 멈추고, 아니면 25개까지
돈다. 빠듯한 예산에서는 앞의 셋이 같은 가장 싼 경로를 골라 차이가 0이므로, 그 예산에서 웹사이트 규칙이 맞았는지는
dp와의 비교가 정한다. 10개는 Dev18 롤링 백테스트의 표본 단위 수(family 10개)다. 0.03은 ml/README.md에 p와 함께 적힌 유의
차이 0.04(데이터 증가에 따른 성능 향상을 빼면 예산 무제한에서 0.04 낮음, p=0.008)보다 작다.
유의성이 아니라 추정 폭으로 멈추므로 p값은 멈춘 뒤 한 번만 보고한다. random은 예산을 보지 않으므로 예산
제한이 없는 줄에만 둔다.

    python -m tests.ghl_main.evaluate_ghl_service --file-index 0 --seeds 3   # 끝난 파일 하나 점검
    python -m tests.ghl_main.evaluate_ghl_service --stop-check               # 멈추면 0, 아직이면 10
    python -m tests.ghl_main.evaluate_ghl_service --summarize --seeds 3      # 끝난 파일을 모두 점검하고 요약
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
from scipy.stats import spearmanr, t, wilcoxon

from src.common.execution_identity import file_sha256
from src.data_split.load_ghl_series import TRAIN_BOUNDARY_PATTERN
from streamlit_website.DP.planner import optimize_plan
from streamlit_website.DP.recommend import cheapest_plan, recommend
from streamlit_website.db_connection.filter_candidates import (
    build_ml_input, filter_candidates, load_candidates, profile_normal_data,
)
from streamlit_website.ml import db
from streamlit_website.ml.dp_input import build_dp_input
from streamlit_website.ml.performance import estimate_stage_performance
from streamlit_website.ml.pipeline import run_ml_pipeline
from streamlit_website.ml.schemas import SimilarityMatch
from tests.ghl_main.run_ghl_candidates import (
    DATA_DIRECTORY, EXPERIMENT_DIRECTORY, GHL_SENSOR_COUNT, SERVICE_RATIOS,
    build_ghl_specs, ghl_file_names, ghl_series, read_rows, require_clean_commit, run_key, run_paths,
)


OPERATING_DAYS = 100
INFERENCE_SEGMENTS = 20
BUDGET_MULTIPLES = (1.5, 10, 100, None)
BASELINES = ("cheapest", "no_similarity", "random")
STOPPING_BASELINES = ("cheapest", "no_similarity", "dp")
STOPPING_CHECKPOINTS = (10, 15, 20, 25)
CONFIDENCE_HALF_WIDTH = 0.03
NOT_YET = 10
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


def load_dev18_means(database) -> dict:
    """{(후보, 비율): 모든 과거 prefix를 같은 가중치로 둔 웹사이트 성능 추정}.

    웹사이트 추정 함수를 그대로 쓰고 유사도 가중만 뺀다(Dev18 백테스트의 '유사도 없음, 과거 prefix 동일 가중').
    """
    matches = [SimilarityMatch(prefix_feature_id=row["prefix_feature_id"], series=row["series"],
                               csv_file=row["csv_file"], family=row["family"], q_percent=row["q_percent"],
                               observed_row=row["observed_row"], distance=0.0, weight=1.0, features_used=())
               for row in db.load_historical_prefixes(database)]
    means = {}
    for (config_id, head), rows in db.load_all_results_grouped_by_candidate(database).items():
        for ratio in SERVICE_RATIOS:
            estimate = estimate_stage_performance(rows, matches, ratio_percent=ratio).predicted_performance
            if estimate is not None:
                means[f"{config_id}::{head}", ratio] = estimate
    return means


def substitute_predictions(payload, values, current=None) -> dict:
    """예측 VUS-PR 자리에 다른 값을 넣은 DP 입력. 값이 없는 학습 선택지는 빼고 비용은 그대로 둔다.

    current는 기존 checkpoint의 (후보, 학습 비율)이다. 그 유지 성능도 같은 출처의 값으로 바꾸고, 값이 없으면
    유지 선택지를 뺀다.
    """
    ratios = [stage["ratio_percent"] for stage in payload["stages"]]
    training = [{**option, "predicted_performance": values[option["candidate_id"], ratios[option["stage"]]]}
                for option in payload["training_options"]
                if (option["candidate_id"], ratios[option["stage"]]) in values]
    kept = {(option["stage"], option["candidate_id"]) for option in training}
    checkpoint = payload["current_checkpoint"]
    if checkpoint is not None:
        value = values.get((checkpoint["candidate_id"], current[1]))
        checkpoint = None if value is None else {**checkpoint, "predicted_performance": value}
    return {**payload, "training_options": training, "current_checkpoint": checkpoint,
            "inference_options": [row for row in payload["inference_options"]
                                  if (row["trained_stage"] == -1 and checkpoint is not None)
                                  or (row["trained_stage"], row["candidate_id"]) in kept]}


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


def random_choice(base, actual) -> dict:
    """비율마다 실행할 수 있는 후보를 무작위로 골라 그 비율에서 학습할 때의 기대값."""
    total = sum(stage["weight"] for stage in base["stages"])
    vus = spent = 0.0
    for stage in base["stages"]:
        ratio = stage["ratio_percent"]
        options = [option for option in base["training_options"]
                   if option["stage"] == stage["stage"] and (option["candidate_id"], ratio) in actual]
        inference = {row["candidate_id"]: row["inference_cost"] for row in base["inference_options"]
                     if row["stage"] == row["trained_stage"] == stage["stage"]}
        vus += stage["weight"] / total * numpy.mean([actual[option["candidate_id"], ratio] for option in options])
        spent += numpy.mean([option["training_cost"] + inference[option["candidate_id"]] for option in options])
    return {"vus_pr": float(vus), "spent_seconds": float(spent), "trainings": len(base["stages"]),
            "missing_actual": 0, "path": "비율마다 무작위 후보"}


def evaluate_file(series, csv_file, frame, actual, database, pool, means) -> dict:
    """한 파일의 전략·예산별 실현값과, 비율별 ML 예측 대 실측 VUS-PR을 만든다."""
    sensors = [column for column in frame.columns if column != "Label"]
    boundary = int(TRAIN_BOUNDARY_PATTERN.search(csv_file).group(1))
    test_rows = len(frame) - boundary
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

    def without_similarity(ratio, current):
        return substitute_predictions(plan_input(ratio, current), means, current)

    base = plan_input(SERVICE_RATIOS[0], None)
    measured = substitute_predictions(base, actual)
    cheapest_cost = optimize_plan(measured)["frontier"][0]["total_cost"]
    plans = []
    for multiple in BUDGET_MULTIPLES:
        budget = None if multiple is None else multiple * cheapest_cost
        results = {name: simulate(plan_input, choose, budget, actual, segments=segments)
                   for name, choose in STRATEGIES.items()}
        results["no_similarity"] = simulate(without_similarity, STRATEGIES["website"], budget, actual,
                                            segments=segments)
        if multiple is None:
            results["random"] = random_choice(base, actual)
        oracle = optimize_plan({**measured, "budget": budget})
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
        "dev18_mean_vus_pr": means.get((option["candidate_id"], ratio)),
        "actual_vus_pr": actual.get((option["candidate_id"], ratio)),
    } for ratio in SERVICE_RATIOS for option in plan_input(ratio, None)["training_options"] if option["stage"] == 0]
    return {"plans": plans, "predictions": predictions}


def paired_p_value(rows, reference) -> float | str:
    differences = [rows[series]["vus_pr"] - reference[series]["vus_pr"] for series in rows if series in reference]
    if not differences:
        return ""
    return 1.0 if not any(differences) else float(wilcoxon(differences).pvalue)


def group_plans(plan_rows) -> dict:
    by_key = defaultdict(dict)
    for row in plan_rows:
        by_key[row["budget_multiple"], row["strategy"]][row["series"]] = row
    return by_key


def summarize_plans(plan_rows) -> list[dict]:
    by_key = group_plans(plan_rows)
    return [{
        "budget_multiple": multiple, "strategy": strategy, "files": len(rows),
        "mean_vus_pr": float(numpy.mean([row["vus_pr"] for row in rows.values()])),
        "median_vus_pr": float(numpy.median([row["vus_pr"] for row in rows.values()])),
        "mean_spent_seconds": float(numpy.mean([row["spent_seconds"] for row in rows.values()])),
        "over_budget_files": sum(row["over_budget"] for row in rows.values()),
        "missing_actual": sum(row["missing_actual"] for row in rows.values()),
        **{f"p_vs_{other}": "" if other == strategy else paired_p_value(rows, by_key.get((multiple, other), {}))
           for other in (*BASELINES, "dp")},
    } for (multiple, strategy), rows in by_key.items()]


def stopping_status(plan_rows) -> dict:
    """정해 둔 파일 수에서만, 웹사이트 추천과 주 기준선의 파일별 차이 평균의 95% 신뢰구간 반폭으로 멈춤을 정한다."""
    by_key = group_plans(plan_rows)
    files = len({row["series"] for row in plan_rows})
    half_widths = {}
    for multiple in dict.fromkeys(row["budget_multiple"] for row in plan_rows):
        website = by_key[multiple, "website"]
        for baseline in STOPPING_BASELINES:
            differences = [website[series]["vus_pr"] - by_key[multiple, baseline][series]["vus_pr"]
                           for series in website]
            half_widths[f"{multiple}:{baseline}"] = (
                float(t.ppf(0.975, len(differences) - 1) * numpy.std(differences, ddof=1)
                      / math.sqrt(len(differences))) if len(differences) > 1 else math.inf)
    return {"files": files, "checkpoints": STOPPING_CHECKPOINTS, "half_width_limit": CONFIDENCE_HALF_WIDTH,
            "half_widths": half_widths,
            "stop": files in STOPPING_CHECKPOINTS
            and max(half_widths.values(), default=math.inf) <= CONFIDENCE_HALF_WIDTH}


def summarize_predictions(prediction_rows) -> dict:
    """새 현장에서 ML 성능 예측이 얼마나 맞았는지. DP는 절대값보다 후보 순위에 기대므로 순위 상관을 같이 본다."""
    measured = [row for row in prediction_rows if row["actual_vus_pr"] is not None]
    groups = defaultdict(list)
    for row in measured:
        groups[row["series"], row["ratio"]].append(row)
    top1_losses, correlations, baseline_correlations = [], [], []
    for rows in groups.values():
        actual = [row["actual_vus_pr"] for row in rows]
        top1_losses.append(max(actual) - max(rows, key=lambda row: row["predicted_vus_pr"])["actual_vus_pr"])
        if len(rows) > 2:
            correlations.append(spearmanr([row["predicted_vus_pr"] for row in rows], actual).statistic)
        known = [row for row in rows if row["dev18_mean_vus_pr"] is not None]
        if len(known) > 2:
            baseline_correlations.append(spearmanr([row["dev18_mean_vus_pr"] for row in known],
                                                   [row["actual_vus_pr"] for row in known]).statistic)

    def finite_mean(values):
        values = [value for value in values if numpy.isfinite(value)]
        return float(numpy.mean(values)) if values else None

    return {
        "rows": len(measured),
        "vus_pr_mean_absolute_error": finite_mean(
            [abs(row["predicted_vus_pr"] - row["actual_vus_pr"]) for row in measured]),
        "top1_loss_mean": finite_mean(top1_losses),
        "spearman_mean": finite_mean(correlations),
        "spearman_mean_without_similarity": finite_mean(baseline_correlations),
    }


def csv_text(rows) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_text(path, text) -> None:
    temporary = Path(path).with_name(f".{Path(path).name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_pool(database) -> list[dict]:
    return filter_candidates(load_candidates(database), channel_count=GHL_SENSOR_COUNT, gpu_available=True)


def evaluate_index(file_index, database, data_directory, pool, means, specs, identity) -> bool:
    """모든 실행이 끝 상태에 닿고 성공한 점수를 모두 채점했으면 그 파일을 점검해 저장하고 True를 돌려준다.

    저장한 점검이 같은 commit·DB로 실행·채점 기록보다 나중에 만든 것이면 다시 하지 않는다.
    """
    csv_name = ghl_file_names()[file_index]
    series = ghl_series(csv_name)
    runs_path, vus_path = run_paths(series)
    runs = read_rows(runs_path)
    scored = {row["score_file"] for row in read_rows(vus_path)}
    unfinished = {run_key(spec) for spec in specs} - {run_key(row) for row in runs}
    unscored = [row for row in runs if row["status"] == "complete" and row["score_file"] not in scored]
    if unfinished or unscored:
        print(f"GHL {series}: 남은 실행 {len(unfinished)}건, 미채점 {len(unscored)}건이라 아직 점검하지 않는다", flush=True)
        return False
    output = OUTPUT_DIRECTORY / f"GHL_{series}.json"
    if (output.exists()
            and output.stat().st_mtime >= max(path.stat().st_mtime for path in (runs_path, vus_path))
            and json.loads(output.read_text(encoding="utf-8")).get("identity") == identity):
        return True
    result = {**evaluate_file(series, csv_name, pandas.read_csv(data_directory / csv_name), load_actuals(series),
                              database, pool, means), "identity": identity}
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_text(output, json.dumps(result, ensure_ascii=False) + "\n")
    failed = len({run_key(row) for row in runs if row["status"] == "failed"}
                 - {run_key(row) for row in runs if row["status"] == "complete"})
    print(f"GHL {series}: 점검 끝 (실패한 실행 {failed}건 포함)", flush=True)
    return True


def load_results(identity) -> list[dict]:
    """지금 commit·DB로 만든 점검 결과만 읽는다. 다른 코드로 만든 결과와 섞지 않는다."""
    results = (json.loads(path.read_text(encoding="utf-8")) for path in sorted(OUTPUT_DIRECTORY.glob("GHL_*.json")))
    return [result for result in results if result.get("identity") == identity]


def summarize(database, data_directory, specs, identity) -> None:
    pool, means = load_pool(database), load_dev18_means(database)
    # 실행을 시작했는데 점검하지 못한 파일만 빠진 파일로 남긴다. 아직 손대지 않은 파일은 세지 않는다.
    skipped = [ghl_series(csv_name) for index, csv_name in enumerate(ghl_file_names())
               if run_paths(ghl_series(csv_name))[0].exists()
               and not evaluate_index(index, database, data_directory, pool, means, specs, identity)]
    results = load_results(identity)
    if not results:
        raise SystemExit("점검한 파일이 없다")
    plan_rows = [row for result in results for row in result["plans"]]
    prediction_rows = [row for result in results for row in result["predictions"]]
    summary = summarize_plans(plan_rows)
    stopping = {**stopping_status(plan_rows), "skipped_series": skipped}
    prediction_summary = summarize_predictions(prediction_rows)
    write_text(OUTPUT_DIRECTORY / "plans.csv", csv_text(plan_rows))
    write_text(OUTPUT_DIRECTORY / "predictions.csv", csv_text(prediction_rows))
    write_text(OUTPUT_DIRECTORY / "summary.csv", csv_text(summary))
    write_text(OUTPUT_DIRECTORY / "prediction_summary.json", json.dumps(prediction_summary, ensure_ascii=False, indent=2) + "\n")
    write_text(OUTPUT_DIRECTORY / "stopping.json", json.dumps(stopping, ensure_ascii=False, indent=2) + "\n")
    print(pandas.DataFrame(summary).to_string(index=False))
    print(json.dumps({"prediction": prediction_summary, "stopping": stopping}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file-index", type=int, choices=range(25))
    mode.add_argument("--stop-check", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", help="실행할 때 준 --seeds와 같게 준다")
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DATABASE_PATH)
    parser.add_argument("--data-directory", type=Path, default=DATA_DIRECTORY)
    arguments = parser.parse_args()
    if not arguments.db.is_file():
        raise SystemExit(f"recommendation DB가 없다: {arguments.db}")
    identity = {"project_commit": require_clean_commit(), "database_sha256": file_sha256(arguments.db)}
    if arguments.stop_check:
        results = load_results(identity)
        stopping = stopping_status([row for result in results for row in result["plans"]])
        print(json.dumps(stopping, ensure_ascii=False), flush=True)
        raise SystemExit(0 if stopping["stop"] else NOT_YET)
    specs = build_ghl_specs(arguments.seeds)
    if arguments.summarize:
        summarize(arguments.db, arguments.data_directory, specs, identity)
        return
    # 실행기가 방금 끝낸 파일이므로 점검할 수 없으면 채점이 실패한 것이다. 작업을 멈춰 드러낸다.
    if not evaluate_index(arguments.file_index, arguments.db, arguments.data_directory,
                          load_pool(arguments.db), load_dev18_means(arguments.db), specs, identity):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
