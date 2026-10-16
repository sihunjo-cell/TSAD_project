"""웹사이트 추천을 GHL·Dev18 밖 TSB-AD-M 파일에서 GHL과 같은 롤링 백테스트로 점검한다.

판별력 기준·지표·주 판정은 실행 전에 tests/tsb_extension/README.md에 적고 commit했다. 이 파일은 그 규칙을 계산만 한다.

재계획 시점·운영 조건·전략(website/dp/cheapest/no_similarity/random/oracle)·예산(가장 싼 계획 비용의 1.5·10·100배와
제한 없음)·실현값 계산은 tests/ghl_main/evaluate_ghl_service.py의 evaluate_file을 그대로 쓴다. 다른 점은 이렇다.
- 후보 풀을 파일마다 그 센서 수로 만든다. 실행기도 같은 함수로 후보를 정하므로, 풀과 실행 목록이 다르면 멈춘다.
  다르면 웹사이트가 실행하지 않은 후보를 골라 그 구간을 0으로 센다.
- 실측은 그 파일 실행 목록(--seeds와 run_tsb_candidates.file_seeds)의 기록만 쓰고, 점검과 요약은 seed 조합마다
  다른 폴더(service_backtest/seeds_*)에 둔다. GHL과 견줄 때는 seed 3 요약을 쓴다.
- 이상 비율이 파일마다 달라 VUS-PR의 바닥도 다르다. 균등 난수 점수의 VUS-PR(우연 수준)을 실측 채점과 같은
  test 구간·ℓ_max·threshold로 재서 파일 정보와 계획 행에 남긴다. 배수는 바닥이 낮은 파일(Genesis)에서 크게 부풀어
  파일 사이 평균에는 쓰지 않는다.
- GHL의 멈춤 규칙은 쓰지 않고, 모두 돈 뒤 요약한다. 요약은 같은 commit·DB의 GHL 점검 10개를 family 하나로 같은 표에
  넣는다.

요약 산출물
- files.csv: 파일마다 판별력((후보, 비율) 실측 중 우연 수준 2배 초과 몫)·판별 가능 여부·순위 보조 지표. GHL 포함.
- family_summary.csv: family·예산·전략마다 파일 평균 VUS-PR, 우연 대비 배수, 사후 최적 격차 회수율.
  scope는 judgeable(판별 가능 파일만, 주 판정)과 all(모든 파일)이다.
- verdict.json: 주 비교 세 개의 family 승·무·패와 단측 부호 검정 p, family 중앙값, 순위 보조 지표.
- ranking.csv: (파일, 비율)마다 후보 순위 Spearman ρ(유사도·유사도 없음)와 예측 1위 후보의 실제 순위.
- cost_ranks.csv: (파일, 비율, 장치, 비용 종류)마다 웹사이트 비용 추정과 실측 시간의 Spearman ρ.
- plans.csv·predictions.csv·summary.csv·status.json: 파일 단위 원값과 실행 상태(모든 TSB 파일).

    python -m tests.tsb_extension.evaluate_tsb_service --file-index 0 --seeds 3 4 5   # 끝난 파일 하나 점검
    python -m tests.tsb_extension.evaluate_tsb_service --summarize --seeds 3 4 5      # 끝난 파일을 모두 점검하고 요약
    python -m tests.tsb_extension.evaluate_tsb_service --summarize --seeds 3          # GHL과 견주는 seed 3 요약
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy
import pandas
from scipy.stats import binomtest, rankdata, spearmanr

from src.common.execution_identity import file_sha256
from src.채점기.fast_vus_pr import fast_vus_pr
from streamlit_website.db_connection.filter_candidates import filter_candidates, load_candidates
from streamlit_website.ml import db
from tests.ghl_main import evaluate_ghl_service as ghl
from tests.ghl_main.evaluate_ghl_service import (
    cost_ranks, csv_text, evaluate_file, load_cost_models, load_dev18_means, summarize_cost_ranks, summarize_plans,
    summarize_predictions, write_text,
)
from tests.ghl_main.report_ghl_service import chance_levels
from tests.ghl_main.run_ghl_candidates import N_THRESHOLDS, SERVICE_RATIOS, read_rows, require_clean_commit, run_key
from tests.tsb_extension.run_tsb_candidates import (
    DATA_ROOT, EXPERIMENT_DIRECTORY, build_tsb_specs, file_seeds, run_paths, tsb_files,
)


BASE_DIRECTORY = EXPERIMENT_DIRECTORY / "results" / "service_backtest"
CHANCE_SEEDS = 5
# 판별력 기준. 평가자만 쓰는 채점 규칙이고 서비스 기능이 아니다.
JUDGEABLE_MULTIPLE, JUDGEABLE_SHARE = 2, 0.10
# 주 비교 (예산 배수, 기준선). 판별 가능 family 중 웹사이트가 이긴 수로 단측 부호 검정을 한다.
PRIMARY_COMPARISONS = (("unlimited", "no_similarity"), (100, "no_similarity"), ("unlimited", "random"))


def output_directory(seeds) -> Path:
    return BASE_DIRECTORY / ("seeds_" + "_".join(map(str, seeds)) if seeds else "seeds_all")


def file_specs(entry, seeds) -> list[dict]:
    return build_tsb_specs(entry["feature_count"], file_seeds(entry["series"], seeds))


def file_pool(entry, database) -> list[dict]:
    return filter_candidates(load_candidates(database), channel_count=entry["feature_count"], gpu_available=True)


def load_actuals(series, keys) -> dict:
    """{(후보, 비율): seed 평균 VUS-PR}. keys(그 파일 실행 목록)에 든 실행만 쓴다. 학습 없는 후보는 한 번 잰 값을
    모든 비율에 쓴다."""
    runs_path, vus_path = run_paths(series)
    vus = {row["score_file"]: float(row["vus_pr"]) for row in read_rows(vus_path)}
    measured = defaultdict(list)
    for row in read_rows(runs_path):
        if row["status"] != "complete" or row["score_file"] not in vus or run_key(row) not in keys:
            continue
        ratios = (int(row["ratio"]),) if row["target_use"] == "fit_full_prefix" else SERVICE_RATIOS
        for ratio in ratios:
            measured[f"{row['config_id']}::{row['score_variant']}", ratio].append(vus[row["score_file"]])
    return {key: float(numpy.mean(values)) for key, values in measured.items()}


def chance_level(labels, l_max, series) -> float:
    """균등 난수 점수의 VUS-PR 평균. 실측 채점과 같은 test 구간·ℓ_max·threshold를 쓴다."""
    return float(numpy.mean([
        fast_vus_pr(numpy.random.default_rng([int(series), seed]).random(len(labels)), labels, l_max,
                    n_thresholds=N_THRESHOLDS)
        for seed in range(CHANCE_SEEDS)]))


def evaluate_index(file_index, database, data_root, means, seeds, identity) -> bool:
    """모든 실행이 끝 상태에 닿고 성공한 점수를 모두 채점했으면 그 파일을 점검해 저장하고 True를 돌려준다.

    저장한 점검이 같은 commit·DB로 실행·채점 기록보다 나중에 만든 것이면 다시 하지 않는다.
    """
    entry = tsb_files()[file_index]
    series = entry["series"]
    specs = file_specs(entry, seeds)
    runs_path, vus_path = run_paths(series)
    runs = read_rows(runs_path)
    scored = {row["score_file"] for row in read_rows(vus_path)}
    unfinished = {run_key(spec) for spec in specs} - {run_key(row) for row in runs}
    unscored = [row for row in runs if row["status"] == "complete" and row["score_file"] not in scored]
    if unfinished or unscored:
        print(f"TSB {series}: 남은 실행 {len(unfinished)}건, 미채점 {len(unscored)}건이라 아직 점검하지 않는다", flush=True)
        return False
    output = output_directory(seeds) / f"TSB_{series}.json"
    if (output.exists()
            and output.stat().st_mtime >= max(path.stat().st_mtime for path in (runs_path, vus_path))
            and json.loads(output.read_text(encoding="utf-8")).get("identity") == identity):
        return True
    pool = file_pool(entry, database)
    offered = {(row["config_id"], row["head"]) for row in pool if row["status"] == "eligible"}
    executed = {(spec["config_id"], variant) for spec in specs for variant in spec["score_variants"]}
    if offered != executed:
        raise SystemExit(f"TSB {series}: 웹사이트 후보 풀과 실행 목록이 다르다 {sorted(offered ^ executed)}")

    frame = pandas.read_csv(Path(data_root) / entry["source_directory"] / entry["name"])
    labels = frame["Label"].to_numpy(dtype=int)[entry["training_boundary"]:]
    l_max = int(read_rows(vus_path)[0]["l_max_samples"])
    chance = chance_level(labels, l_max, series)
    actual = load_actuals(series, {run_key(spec) for spec in specs})
    result = evaluate_file(series, entry["name"], frame, actual, database, pool, means)
    for row in result["plans"]:
        row.update(family=entry["family"], chance_vus_pr=chance, chance_multiple=row["vus_pr"] / chance)
    failed = ({run_key(row) for row in runs if row["status"] == "failed"}
              - {run_key(row) for row in runs if row["status"] == "complete"})
    result["file"] = {
        "series": series, "family": entry["family"], "csv_file": entry["name"], "sensors": entry["feature_count"],
        "training_boundary": entry["training_boundary"], "test_rows": len(labels),
        "prevalence": float(labels.mean()), "l_max_samples": l_max, "chance_vus_pr": chance,
        "candidates": len(offered), "runs": len(specs), "failed_runs": len(failed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text(output, json.dumps({**result, "identity": identity}, ensure_ascii=False) + "\n")
    print(f"TSB {series}: 점검 끝 (실패한 실행 {len(failed)}건 포함, 우연 수준 VUS-PR {chance:.4f})", flush=True)
    return True


def load_results(directory, identity) -> list[dict]:
    """지금 commit·DB로 만든 점검 결과만 읽는다. 다른 코드로 만든 결과와 섞지 않는다."""
    results = (json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("TSB_*.json")))
    return [result for result in results if result.get("identity") == identity]


def ghl_sources(identity) -> list[tuple]:
    """같은 commit·DB의 GHL 점검을 (dataset, 파일 정보, 점검 결과)로 읽는다. 10개 파일이 family 하나다."""
    results = ghl.load_results(identity)
    if not results:
        raise SystemExit("같은 commit·DB의 GHL 점검이 없다. 먼저 "
                         "python -m tests.ghl_main.evaluate_ghl_service --summarize --seeds 3을 돌린다")
    files = {result["plans"][0]["series"]: result["plans"][0]["csv_file"] for result in results}
    chance = chance_levels(files, CHANCE_SEEDS, workers=4)
    sources = []
    for result in results:
        series = result["plans"][0]["series"]
        level = float(chance.loc[series, "chance"])
        for row in result["plans"]:
            row.update(family="GHL", chance_vus_pr=level, chance_multiple=row["vus_pr"] / level)
        sources.append(("GHL", {"series": series, "family": "GHL", "csv_file": files[series],
                                "prevalence": float(chance.loc[series, "prevalence"]), "chance_vus_pr": level},
                        result))
    return sources


def ranking_rows(family, series, prediction_rows) -> list[dict]:
    """(파일, 비율)마다 예측과 실측의 후보 순위 일치. 1위 순위는 예측 1위 후보의 실제 순위다(1이 가장 좋다)."""
    groups = defaultdict(list)
    for row in prediction_rows:
        if row["actual_vus_pr"] is not None:
            groups[row["ratio"]].append(row)
    rows = []
    for ratio, group in groups.items():
        actual = numpy.array([row["actual_vus_pr"] for row in group])
        ranks = rankdata(-actual, method="min")
        result = {"family": family, "series": series, "ratio": ratio, "candidates": len(group)}
        for name, column in (("similarity", "predicted_vus_pr"), ("no_similarity", "dev18_mean_vus_pr")):
            known = [index for index, row in enumerate(group) if row[column] is not None]
            predicted = [group[index][column] for index in known]
            result[f"spearman_{name}"] = float(spearmanr(predicted, actual[known]).statistic)
            result[f"top1_rank_{name}"] = int(ranks[known[int(numpy.argmax(predicted))]])
        result["spearman_difference"] = result["spearman_similarity"] - result["spearman_no_similarity"]
        rows.append(result)
    return rows


def ranking_summary(rows) -> dict:
    return {
        "groups": len(rows),
        "spearman_difference_mean": float(numpy.nanmean([row["spearman_difference"] for row in rows])),
        "top1_rank_similarity_median": float(numpy.median([row["top1_rank_similarity"] for row in rows])),
        "top1_rank_no_similarity_median": float(numpy.median([row["top1_rank_no_similarity"] for row in rows])),
    } if rows else {"groups": 0}


def family_rows(plan_rows, judgeable) -> list[dict]:
    """family·예산·전략마다 파일을 평균한 VUS-PR과 사후 최적 격차 회수율. 같은 설비의 SWaT 두 파일, GHL 10개 파일이
    각각 한 단위가 된다. 회수율 (전략 − cheapest) / (oracle − cheapest)는 family 안의 비라 데이터 사이 VUS-PR 수준
    차이에 영향받지 않는다. 우연 대비 배수는 family 평균 VUS-PR / family 평균 우연 수준이다."""
    rows = []
    for scope in ("judgeable", "all"):
        groups = defaultdict(list)
        for row in plan_rows:
            if scope == "all" or judgeable[row["family"], row["series"]]:
                groups[row["family"], row["budget_multiple"], row["strategy"]].append(row)
        means = {key: float(numpy.mean([row["vus_pr"] for row in members])) for key, members in groups.items()}
        for (family, multiple, strategy), members in groups.items():
            value = means[family, multiple, strategy]
            floor, ceiling = means[family, multiple, "cheapest"], means[family, multiple, "oracle"]
            rows.append({
                "scope": scope, "family": family, "budget_multiple": multiple, "strategy": strategy,
                "files": len(members), "vus_pr": value,
                "chance_multiple": value / float(numpy.mean([row["chance_vus_pr"] for row in members])),
                "oracle_gap_closed": (value - floor) / (ceiling - floor) if ceiling > floor else None,
                "missing_actual": sum(row["missing_actual"] for row in members),
            })
    return rows


def primary_comparisons(rows) -> list[dict]:
    """판별 가능 family로 주 비교마다 승·무·패와 단측 부호 검정 p. 무승부는 검정에서 뺀다."""
    values = {(row["family"], row["budget_multiple"], row["strategy"]): row["vus_pr"]
              for row in rows if row["scope"] == "judgeable"}
    families = sorted({family for family, _, _ in values})
    results = []
    for multiple, baseline in PRIMARY_COMPARISONS:
        differences = {family: values[family, multiple, "website"] - values[family, multiple, baseline]
                       for family in families}
        wins = sum(value > 0 for value in differences.values())
        losses = sum(value < 0 for value in differences.values())
        results.append({
            "budget_multiple": multiple, "baseline": baseline, "families": len(families),
            "wins": wins, "ties": len(families) - wins - losses, "losses": losses,
            "sign_test_p": float(binomtest(wins, wins + losses, alternative="greater").pvalue) if wins + losses else None,
            "website_minus_baseline": differences,
        })
    return results


def family_medians(rows) -> list[dict]:
    """판별 가능 family의 우연 대비 배수와 회수율 중앙값. 평균은 우연 수준이 낮은 family에 끌려가 쓰지 않는다."""
    groups = defaultdict(list)
    for row in rows:
        if row["scope"] == "judgeable":
            groups[row["budget_multiple"], row["strategy"]].append(row)
    return [{
        "budget_multiple": multiple, "strategy": strategy, "families": len(members),
        "median_chance_multiple": float(numpy.median([row["chance_multiple"] for row in members])),
        "median_oracle_gap_closed": float(numpy.median(closed)) if (closed := [
            row["oracle_gap_closed"] for row in members if row["oracle_gap_closed"] is not None]) else None,
    } for (multiple, strategy), members in groups.items()]


def summarize(database, data_root, seeds, identity) -> None:
    directory = output_directory(seeds)
    means = load_dev18_means(database)
    # 실행을 시작했는데 점검하지 못한 파일만 빠진 파일로 남긴다. 아직 손대지 않은 파일은 세지 않는다.
    skipped = [entry["series"] for index, entry in enumerate(tsb_files())
               if run_paths(entry["series"])[0].exists()
               and not evaluate_index(index, database, data_root, means, seeds, identity)]
    results = load_results(directory, identity)
    if not results:
        raise SystemExit("점검한 파일이 없다")
    sources = [("TSB", result["file"], result) for result in results] + ghl_sources(identity)

    files, ranking = [], []
    for dataset, file, result in sources:
        rows = ranking_rows(file["family"], file["series"], result["predictions"])
        actual = numpy.array([row["actual_vus_pr"] for row in result["predictions"] if row["actual_vus_pr"] is not None])
        share = float(numpy.mean(actual > JUDGEABLE_MULTIPLE * file["chance_vus_pr"]))
        summary = {key: value for key, value in ranking_summary(rows).items() if key != "groups"}
        files.append({"dataset": dataset, **file, "judgeable_share": share, "judgeable": share >= JUDGEABLE_SHARE,
                      **summary})
        ranking += [{"dataset": dataset, **row} for row in rows]
    judgeable = {(file["family"], file["series"]): file["judgeable"] for file in files}
    families = family_rows([row for _, _, result in sources for row in result["plans"]], judgeable)
    verdict = {
        "seeds": seeds,
        "judgeable_files": [f"{family} {series}" for (family, series), flag in judgeable.items() if flag],
        "comparisons": primary_comparisons(families),
        "medians": family_medians(families),
        "ranking": {
            "judgeable": ranking_summary([row for row in ranking if judgeable[row["family"], row["series"]]]),
            "all": ranking_summary(ranking),
        },
    }

    entries = {entry["series"]: entry for entry in tsb_files()}
    cost_models, cost_rows = load_cost_models(database), []
    for result in results:
        entry = entries[result["file"]["series"]]
        keys = {run_key(spec) for spec in file_specs(entry, seeds)}
        runs = [row for row in read_rows(run_paths(entry["series"])[0]) if run_key(row) in keys]
        cost_rows += [{"family": entry["family"], **row} for row in cost_ranks(
            entry["series"], runs, file_pool(entry, database), cost_models, boundary=entry["training_boundary"],
            test_rows=result["file"]["test_rows"], sensors=entry["feature_count"])]

    plan_rows = [row for result in results for row in result["plans"]]
    prediction_rows = [row for result in results for row in result["predictions"]]
    status = {"files": len(results), "families": len({result["file"]["family"] for result in results}),
              "skipped_series": skipped, "prediction": summarize_predictions(prediction_rows),
              "cost_ranks": summarize_cost_ranks(cost_rows)}
    write_text(directory / "files.csv", pandas.DataFrame(files).to_csv(index=False))
    write_text(directory / "family_summary.csv", csv_text(families))
    write_text(directory / "ranking.csv", csv_text(ranking))
    write_text(directory / "cost_ranks.csv", csv_text(cost_rows))
    write_text(directory / "plans.csv", csv_text(plan_rows))
    write_text(directory / "predictions.csv", csv_text(prediction_rows))
    write_text(directory / "summary.csv", csv_text(summarize_plans(plan_rows)))
    write_text(directory / "verdict.json", json.dumps(verdict, ensure_ascii=False, indent=2) + "\n")
    write_text(directory / "status.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(pandas.DataFrame(files)[["dataset", "family", "series", "chance_vus_pr", "judgeable_share", "judgeable"]]
          .to_string(index=False))
    print(pandas.DataFrame([row for row in families if row["scope"] == "judgeable"]).to_string(index=False))
    print(json.dumps({"verdict": verdict, "status": status}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file-index", type=int, choices=range(len(tsb_files())))
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", help="실행할 때 준 --seeds와 같게 준다")
    parser.add_argument("--db", type=Path, default=db.DEFAULT_DATABASE_PATH)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    arguments = parser.parse_args()
    if not arguments.db.is_file():
        raise SystemExit(f"recommendation DB가 없다: {arguments.db}")
    identity = {"project_commit": require_clean_commit(), "database_sha256": file_sha256(arguments.db)}
    if arguments.summarize:
        summarize(arguments.db, arguments.data_root, arguments.seeds, identity)
        return
    # 실행기가 방금 끝낸 파일이므로 점검할 수 없으면 채점이 실패한 것이다. 작업을 멈춰 드러낸다.
    if not evaluate_index(arguments.file_index, arguments.db, arguments.data_root,
                          load_dev18_means(arguments.db), arguments.seeds, identity):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
