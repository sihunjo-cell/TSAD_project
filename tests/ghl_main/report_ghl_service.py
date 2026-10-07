"""GHL 서비스 백테스트 결과를 보고서용 표·그림으로 정리한다.

`evaluate_ghl_service.py`가 남긴 plans.csv·predictions.csv·cost_ranks.csv·stopping.json·prediction_summary.json만 읽고
계획을 다시 풀지 않는다. 먼저 각 경로를 실측 VUS-PR로 다시 더해 plans.csv의 실현값과 맞는지 확인한다.

통계 단위는 GHL 파일이고, 전략 비교는 같은 파일끼리 짝지은 차이(웹사이트 − 기준선)로 본다.
- 평균 차이와 95% t 신뢰구간: 멈춤 규칙(반폭 0.03 이하)에 쓴 값과 같다.
- Wilcoxon 부호순위 p: summary.csv와 같은 계산(차이 0은 뺀다). 웹사이트 대 기준선 비교 전체에 Holm 보정을 붙인다.
- 부호 뒤집기 순열 p: 평균 차이의 정확 검정이다. 분포를 가정하지 않아 파일이 적을 때 Wilcoxon을 보완한다.
- 파일 하나 빼기 범위: 한 파일이 평균 차이를 끌고 가는지 본다.
전략별 수준(평균)에는 파일 단위 bootstrap 95% 구간을 붙인다.

VUS-PR의 바닥은 이상 비율과 ℓ_max에 따라 달라서, 파일마다 균등 난수 점수의 VUS-PR(우연 수준)을 실측 채점과
같은 test 구간·ℓ_max·threshold로 재고 실현값을 그 배수로도 본다. 처음 한 번 계산해 chance_vus_pr.csv에 둔다.
Dev18 백테스트 수치는 코드가 남아 있지 않아 streamlit_website/ml/README.md의 표를 옮겨 적는다.

    python -m tests.ghl_main.report_ghl_service               # service_backtest/report/에 표·그림 생성
    python -m tests.ghl_main.report_ghl_service --workers 8   # 우연 수준 계산 프로세스 수
"""

import argparse
import json
import math
import multiprocessing
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import pandas
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.stats import rankdata, spearmanr, t, wilcoxon

from src.data_split.load_ghl_series import TRAIN_BOUNDARY_PATTERN
from src.채점기.fast_vus_pr import fast_vus_pr
from tests.ghl_main.evaluate_ghl_service import (
    CONFIDENCE_HALF_WIDTH, OPERATING_DAYS, OUTPUT_DIRECTORY, STOPPING_BASELINES,
)
from tests.ghl_main.run_ghl_candidates import (
    DATA_DIRECTORY, N_THRESHOLDS, SERVICE_RATIOS, append_row, read_rows, run_paths,
)


REPORT_DIRECTORY = OUTPUT_DIRECTORY / "report"
CHANCE_PATH = OUTPUT_DIRECTORY / "chance_vus_pr.csv"
CHANCE_FIELDS = ("series", "seed", "prevalence", "l_max_samples", "vus_pr")
BUDGETS = ("1.5", "10", "100", "unlimited")
CONTRASTS = ("cheapest", "no_similarity", "dp", "random")
BOOTSTRAP_DRAWS = 10_000
STEP_PATTERN = re.compile(r"q(\d+):(adopt|keep|retrain|switch):(.+)")
# streamlit_website/ml/README.md '예산 계획 백테스트 (2026-09-25)' 표. 웹사이트는 'DB 비율 구간 (현재)' 열이다.
DEV18_BACKTEST = {
    "1.5": {"cheapest": 0.263, "website": 0.249, "oracle": 0.429},
    "10": {"cheapest": 0.263, "website": 0.243, "oracle": 0.474},
    "100": {"cheapest": 0.263, "website": 0.266, "oracle": 0.561},
    "unlimited": {"cheapest": 0.263, "website": 0.424, "oracle": 0.673},
}

LABEL = {"website": "Website", "dp": "DP only", "no_similarity": "No similarity", "cheapest": "Cheapest",
         "random": "Random pick", "oracle": "Oracle", "Chance": "Chance"}
KOREAN = {"website": "웹사이트 추천", "dp": "DP만", "no_similarity": "유사도 없음", "cheapest": "가장 싼 계획",
          "random": "무작위", "oracle": "사후 최적"}
BUDGET_LABEL = {"1.5": r"$1.5\times$", "10": r"$10\times$", "100": r"$100\times$", "unlimited": "No limit"}
BUDGET_KOREAN = {"1.5": "1.5배", "10": "10배", "100": "100배", "unlimited": "제한 없음"}
# 같은 전략은 모든 그림에서 같은 색이다. 모델 색은 전략 색과 겹치지 않게 PuOr에서 고른다.
COLOR = {"website": "#B2182B", "dp": "#2166AC", "no_similarity": "#1B7837", "cheapest": "#8C8C8C",
         "random": "#525252", "oracle": "#222222", "Chance": "#8C8C8C"}
MODEL_COLOR = {"GDN": "#E08214", "PaAno": "#8073AC"}
INK, GRAY, LIGHT = "#222222", "#8C8C8C", "#D9D9D9"
DOUBLE_COLUMN, SINGLE_COLUMN = 7.0, 3.4


def apply_style() -> None:
    """논문 인쇄 폭(7.0/3.4 in)에 8.5pt serif로 조판한다. 슬라이드에 키우면 약 15pt가 된다."""
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#333333",
        "axes.labelcolor": INK, "xtick.color": "#333333", "ytick.color": "#333333",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "axes.titlepad": 3, "axes.labelpad": 2,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ---------------------------------------------------------------- 입력

def load_backtest() -> dict:
    plans = pandas.read_csv(OUTPUT_DIRECTORY / "plans.csv", dtype={"series": str, "budget_multiple": str})
    predictions = pandas.read_csv(OUTPUT_DIRECTORY / "predictions.csv", dtype={"series": str})
    return {
        "plans": plans, "predictions": predictions,
        "cost_ranks": pandas.read_csv(OUTPUT_DIRECTORY / "cost_ranks.csv", dtype={"series": str}),
        "stopping": json.loads((OUTPUT_DIRECTORY / "stopping.json").read_text(encoding="utf-8")),
        "prediction_summary": json.loads((OUTPUT_DIRECTORY / "prediction_summary.json").read_text(encoding="utf-8")),
        "files": dict(plans[["series", "csv_file"]].drop_duplicates().itertuples(index=False)),
    }


def test_labels(csv_file) -> numpy.ndarray:
    boundary = int(TRAIN_BOUNDARY_PATTERN.search(csv_file).group(1))
    return pandas.read_csv(DATA_DIRECTORY / csv_file, usecols=["Label"])["Label"].to_numpy(dtype=int)[boundary:]


def _random_score_vus(task) -> dict:
    series, seed, labels, l_max = task
    score = numpy.random.default_rng([int(series), seed]).random(len(labels))
    return {"series": series, "seed": seed, "prevalence": float(labels.mean()), "l_max_samples": l_max,
            "vus_pr": fast_vus_pr(score, labels, l_max, n_thresholds=N_THRESHOLDS)}


def chance_levels(files, seeds, workers) -> pandas.DataFrame:
    """파일마다 균등 난수 점수의 VUS-PR 평균. 없는 (파일, seed)만 계산해 덧붙인다."""
    done = {(row["series"], int(row["seed"])) for row in read_rows(CHANCE_PATH)}
    tasks = []
    for series, csv_file in files.items():
        missing = [seed for seed in range(seeds) if (series, seed) not in done]
        if missing:
            labels = test_labels(csv_file)
            l_max = int(read_rows(run_paths(series)[1])[0]["l_max_samples"])
            tasks += [(series, seed, labels, l_max) for seed in missing]
    if tasks:
        print(f"우연 수준 VUS-PR {len(tasks)}건을 계산한다 (건당 약 10초, 프로세스 {workers}개)", flush=True)
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for row in pool.imap_unordered(_random_score_vus, tasks):
                append_row(CHANCE_PATH, CHANCE_FIELDS, row)
    frame = pandas.read_csv(CHANCE_PATH, dtype={"series": str})
    frame = frame[frame["series"].isin(files) & (frame["seed"] < seeds)]
    return frame.groupby("series").agg(prevalence=("prevalence", "first"), chance=("vus_pr", "mean"),
                                       chance_sd=("vus_pr", "std"), seeds=("seed", "count"))


def operating_segments(csv_file) -> list[tuple]:
    """evaluate_file과 같은 (비율, 시작일, 끝일). 5%에서 시작해 100일 뒤 학습 구간 끝에 닿는다."""
    boundary = int(TRAIN_BOUNDARY_PATTERN.search(csv_file).group(1))
    start_rows = SERVICE_RATIOS[0] * boundary // 100
    rows_per_day = (boundary - start_rows) / OPERATING_DAYS
    starts = [(ratio * boundary // 100 - start_rows) / rows_per_day for ratio in SERVICE_RATIOS]
    return list(zip(SERVICE_RATIOS, starts, starts[1:] + [OPERATING_DAYS]))


def deployed_segments(plans, predictions) -> pandas.DataFrame:
    """경로를 구간마다 배치된 모델과 그 실측 VUS-PR로 펼치고, 다시 더한 값이 plans.csv와 같은지 확인한다."""
    actual = {(row.series, row.ratio, row.candidate_id): row.actual_vus_pr
              for row in predictions.itertuples() if not pandas.isna(row.actual_vus_pr)}
    model = dict(zip(predictions["candidate_id"], predictions["model"]))
    rows = []
    for plan in plans[plans["strategy"] != "random"].itertuples():
        current = None
        segments = operating_segments(plan.csv_file)
        steps = plan.path.split(" | ")
        if len(steps) != len(segments):
            raise SystemExit(f"GHL {plan.series} {plan.strategy}: 경로 단계 수가 구간 수와 다르다")
        realized = 0.0
        for (ratio, start, end), text in zip(segments, steps):
            step_ratio, action, candidate = STEP_PATTERN.fullmatch(text).groups()
            if int(step_ratio) != ratio:
                raise SystemExit(f"GHL {plan.series} {plan.strategy}: q{step_ratio} 단계가 q{ratio} 구간에 있다")
            if action != "keep":
                current = (candidate, ratio)
            vus = actual[plan.series, current[1], current[0]]
            realized += vus * (end - start) / OPERATING_DAYS
            rows.append({"budget": plan.budget_multiple, "strategy": plan.strategy, "series": plan.series,
                         "ratio": ratio, "start_day": start, "end_day": end, "action": action,
                         "candidate_id": current[0], "trained_ratio": current[1], "model": model[current[0]],
                         "vus_pr": vus})
        if not math.isclose(realized, plan.vus_pr, rel_tol=1e-9, abs_tol=1e-12):
            raise SystemExit(f"GHL {plan.series} {plan.budget_multiple} {plan.strategy}: 다시 더한 실현값 "
                             f"{realized}이 plans.csv의 {plan.vus_pr}와 다르다")
    return pandas.DataFrame(rows)


# ---------------------------------------------------------------- 통계

def bootstrap_interval(values, seed=0) -> tuple[float, float]:
    values = numpy.asarray(values, dtype=float)
    draws = numpy.random.default_rng(seed).integers(0, len(values), (BOOTSTRAP_DRAWS, len(values)))
    return tuple(float(value) for value in numpy.percentile(values[draws].mean(axis=1), [2.5, 97.5]))


def sign_flip_p_value(differences) -> float:
    """평균 차이가 0이라는 가설의 양측 정확 검정. 파일이 16개를 넘으면 무작위 부호 20만 개로 근사한다."""
    differences = numpy.asarray(differences, dtype=float)
    if not differences.any():
        return 1.0
    count = len(differences)
    if count <= 16:
        signs = ((numpy.arange(2 ** count)[:, None] >> numpy.arange(count)) & 1) * 2 - 1
    else:
        signs = numpy.random.default_rng(0).choice((-1, 1), size=(200_000, count))
    flipped = numpy.abs((signs * differences).mean(axis=1))
    return float(numpy.mean(flipped >= abs(differences.mean()) - 1e-15))


def paired_contrast(differences) -> dict:
    differences = numpy.asarray(differences, dtype=float)
    count = len(differences)
    mean = float(differences.mean())
    half = float(t.ppf(0.975, count - 1) * differences.std(ddof=1) / math.sqrt(count)) if count > 1 else math.inf
    nonzero = differences[differences != 0]
    ranks = rankdata(numpy.abs(nonzero))
    leave_one_out = (differences.sum() - differences) / (count - 1)
    return {
        "files": count, "mean_difference": mean, "ci_low": mean - half, "ci_high": mean + half, "half_width": half,
        # summary.csv의 paired_p_value와 같은 계산이다.
        "wilcoxon_p": 1.0 if not nonzero.size else float(wilcoxon(differences).pvalue),
        "sign_flip_p": sign_flip_p_value(differences),
        "rank_biserial": float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
        if nonzero.size else 0.0,
        "wins": int((differences > 0).sum()), "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
        "leave_one_out_min": float(leave_one_out.min()), "leave_one_out_max": float(leave_one_out.max()),
    }


def holm(p_values) -> list[float]:
    p_values = numpy.asarray(p_values, dtype=float)
    adjusted, running = numpy.empty_like(p_values), 0.0
    for rank, index in enumerate(numpy.argsort(p_values)):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def by_file(plans, budget, strategy) -> pandas.Series:
    rows = plans[(plans["budget_multiple"] == budget) & (plans["strategy"] == strategy)]
    return rows.set_index("series")["vus_pr"].sort_index()


# ---------------------------------------------------------------- 표

def main_table(plans, chance) -> pandas.DataFrame:
    rows = []
    for budget in BUDGETS:
        means = plans[plans["budget_multiple"] == budget].groupby("strategy")["vus_pr"].mean()
        for strategy in LABEL:
            if strategy not in means:
                continue
            selected = plans[(plans["budget_multiple"] == budget) & (plans["strategy"] == strategy)]
            values = selected.set_index("series")["vus_pr"].sort_index()
            low, high = bootstrap_interval(values)
            lift = values / chance["chance"].reindex(values.index)
            rows.append({
                "budget": budget, "strategy": strategy, "files": len(values),
                "mean_vus_pr": values.mean(), "bootstrap_low": low, "bootstrap_high": high,
                "median_vus_pr": values.median(), "mean_chance_multiple": lift.mean(),
                "files_above_chance": int((lift > 1).sum()),
                # 가장 싼 계획에서 사후 최적까지의 평균 격차 중 이 전략이 채운 몫
                "oracle_gap_closed": (means[strategy] - means["cheapest"]) / (means["oracle"] - means["cheapest"]),
                "mean_spent_seconds": selected["spent_seconds"].mean(),
                "median_spent_seconds": selected["spent_seconds"].median(),
                "mean_trainings": selected["trainings"].mean(),
                "over_budget_files": int(selected["over_budget"].sum()),
            })
    return pandas.DataFrame(rows)


def contrast_table(plans) -> pandas.DataFrame:
    rows = []
    for budget in BUDGETS:
        website = by_file(plans, budget, "website")
        for baseline in CONTRASTS:
            reference = by_file(plans, budget, baseline)
            if reference.empty:
                continue
            rows.append({"budget": budget, "baseline": baseline,
                         **paired_contrast((website - reference.reindex(website.index)).to_numpy())})
    frame = pandas.DataFrame(rows)
    frame["wilcoxon_p_holm"] = holm(frame["wilcoxon_p"])
    frame["sign_flip_p_holm"] = holm(frame["sign_flip_p"])
    return frame


def prediction_tables(predictions, summary) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """ML 성능 예측의 정확도. 유사도 예측(웹사이트)과 Dev18 동일 가중 평균(유사도 없음)을 같은 행에서 견준다."""
    measured = predictions.dropna(subset=["actual_vus_pr"])
    groups = []
    for (series, ratio), rows in measured.groupby(["series", "ratio"]):
        known = rows.dropna(subset=["dev18_mean_vus_pr"])
        best = rows["actual_vus_pr"].max()
        groups.append({
            "series": series, "ratio": ratio, "candidates": len(rows),
            "abs_error_similarity": (rows["predicted_vus_pr"] - rows["actual_vus_pr"]).abs().mean(),
            "abs_error_no_similarity": (known["dev18_mean_vus_pr"] - known["actual_vus_pr"]).abs().mean(),
            "top1_loss_similarity": best - rows.loc[rows["predicted_vus_pr"].idxmax(), "actual_vus_pr"],
            "top1_loss_no_similarity": best - known.loc[known["dev18_mean_vus_pr"].idxmax(), "actual_vus_pr"],
            "spearman_similarity": spearmanr(rows["predicted_vus_pr"], rows["actual_vus_pr"]).statistic,
            "spearman_no_similarity": spearmanr(known["dev18_mean_vus_pr"], known["actual_vus_pr"]).statistic,
        })
    groups = pandas.DataFrame(groups)
    files = groups.groupby("series").mean(numeric_only=True)

    # 후보마다 파일·비율 평균. DP가 고르는 단위가 후보라 모델 평균보다 이쪽이 선택을 설명한다.
    candidates = measured.groupby(["candidate_id", "model"]).agg(
        dev18_mean=("dev18_mean_vus_pr", "mean"), predicted=("predicted_vus_pr", "mean"),
        actual=("actual_vus_pr", "mean"), actual_median=("actual_vus_pr", "median")).reset_index()
    for column, key in (("predicted_vus_pr", "similarity"), ("dev18_mean_vus_pr", "no_similarity")):
        top = measured.loc[measured.dropna(subset=[column]).groupby(["series", "ratio"])[column].idxmax()]
        candidates[f"top1_{key}"] = candidates["candidate_id"].map(top["candidate_id"].value_counts()).fillna(0)
    models = measured.groupby("model")[["dev18_mean_vus_pr", "predicted_vus_pr", "actual_vus_pr"]].mean()

    metrics = []
    for name, key in (("VUS-PR 절대 오차", "abs_error"), ("예측 1위 손실", "top1_loss"),
                      ("후보 순위 Spearman ρ", "spearman")):
        similarity, baseline = files[f"{key}_similarity"], files[f"{key}_no_similarity"]
        metrics.append({"metric": name, "similarity": groups[f"{key}_similarity"].mean(),
                        "no_similarity": groups[f"{key}_no_similarity"].mean(),
                        "files_better_with_similarity": int(((similarity < baseline) if key != "spearman"
                                                             else (similarity > baseline)).sum()),
                        "wilcoxon_p": float(wilcoxon(similarity - baseline).pvalue)})
    metrics += [
        {"metric": f"후보 평균의 순위 Spearman ρ ({len(candidates)}개)",
         "similarity": spearmanr(candidates["predicted"], candidates["actual"]).statistic,
         "no_similarity": spearmanr(candidates["dev18_mean"], candidates["actual"]).statistic,
         "files_better_with_similarity": None, "wilcoxon_p": None},
        {"metric": f"모델 평균의 순위 Spearman ρ ({len(models)}개)",
         "similarity": spearmanr(models["predicted_vus_pr"], models["actual_vus_pr"]).statistic,
         "no_similarity": spearmanr(models["dev18_mean_vus_pr"], models["actual_vus_pr"]).statistic,
         "files_better_with_similarity": None, "wilcoxon_p": None},
    ]
    metrics = pandas.DataFrame(metrics)
    # evaluate_ghl_service.summarize_predictions와 같은 값이어야 한다.
    checks = {"vus_pr_mean_absolute_error": (measured["predicted_vus_pr"] - measured["actual_vus_pr"]).abs().mean(),
              "top1_loss_mean": groups["top1_loss_similarity"].mean(),
              "spearman_mean": groups["spearman_similarity"].mean(),
              "spearman_mean_without_similarity": groups["spearman_no_similarity"].mean()}
    for key, value in checks.items():
        if not math.isclose(value, summary[key], rel_tol=1e-9):
            raise SystemExit(f"{key} 재계산 {value}이 prediction_summary.json의 {summary[key]}와 다르다")
    return metrics, candidates.sort_values("predicted", ascending=False), groups


def learning_table(predictions, chance) -> pandas.DataFrame:
    """학습 비율에 따라 실측이 달라지는 모델(PaAno·GDN)의 비율별 실측·예측. 꼬리를 보려고 우연 수준 2배를 넘은 몫을 센다."""
    rows = predictions[predictions["model"].isin(MODEL_COLOR)].dropna(subset=["actual_vus_pr"]).copy()
    rows["above_twice_chance"] = rows["actual_vus_pr"] > 2 * rows["series"].map(chance["chance"])
    return rows.groupby(["model", "ratio"]).agg(
        runs=("actual_vus_pr", "size"), actual_median=("actual_vus_pr", "median"),
        actual_mean=("actual_vus_pr", "mean"), actual_max=("actual_vus_pr", "max"),
        share_above_twice_chance=("above_twice_chance", "mean"),
        predicted_mean=("predicted_vus_pr", "mean"), dev18_mean=("dev18_mean_vus_pr", "mean")).reset_index()


def path_table(segments) -> pandas.DataFrame:
    """전략·예산마다 파일들이 받은 경로가 (설정 단위로) 몇 가지인지와 가장 흔한 경로."""
    rows = []
    for (budget, strategy), group in segments.groupby(["budget", "strategy"], sort=False):
        paths = group.sort_values("ratio").groupby("series").apply(
            lambda frame: " → ".join(
                f"q{row.ratio} {row.model}·{row.candidate_id[:7]}"
                + ("" if row.action == "keep" else f" ({row.action})")
                for row in frame.itertuples()), include_groups=False)
        common = paths.value_counts()
        rows.append({"budget": budget, "strategy": strategy, "distinct_paths": len(common),
                     "files_on_common_path": int(common.iloc[0]), "common_path": common.index[0]})
    order = {strategy: index for index, strategy in enumerate(LABEL)}
    frame = pandas.DataFrame(rows)
    return frame.sort_values(["budget", "strategy"], key=lambda column: column.map(
        {budget: index for index, budget in enumerate(BUDGETS)} if column.name == "budget" else order))


def stopping_table(contrasts, stopping) -> pandas.DataFrame:
    rows = contrasts[contrasts["baseline"].isin(STOPPING_BASELINES)][["budget", "baseline", "half_width"]].copy()
    recorded = [stopping["half_widths"][f"{budget}:{baseline}"] for budget, baseline
                in zip(rows["budget"], rows["baseline"])]
    if not numpy.allclose(rows["half_width"], recorded, rtol=1e-9, atol=1e-15):
        raise SystemExit("다시 계산한 신뢰구간 반폭이 stopping.json과 다르다")
    rows["limit"] = CONFIDENCE_HALF_WIDTH
    rows["within_limit"] = rows["half_width"] <= CONFIDENCE_HALF_WIDTH
    return rows


def cost_table(cost_ranks) -> pandas.DataFrame:
    """장치·비용 종류마다 (파일, 비율) 묶음에서 잰 웹사이트 비용 추정 대 GHL 실측 시간의 Spearman ρ."""
    return cost_ranks.dropna(subset=["spearman"]).groupby(["device", "kind"]).agg(
        groups=("spearman", "size"), candidates_min=("candidates", "min"), candidates_max=("candidates", "max"),
        spearman_median=("spearman", "median"), spearman_mean=("spearman", "mean"),
        share_positive=("spearman", lambda values: (values > 0).mean())).reset_index()


def dev18_table(main) -> pandas.DataFrame:
    rows = []
    for budget in BUDGETS:
        ghl = main[main["budget"] == budget].set_index("strategy")["mean_vus_pr"]
        dev18 = DEV18_BACKTEST[budget]
        rows.append({
            "budget": budget, **{f"dev18_{key}": value for key, value in dev18.items()},
            "dev18_gap_closed": (dev18["website"] - dev18["cheapest"]) / (dev18["oracle"] - dev18["cheapest"]),
            "ghl_cheapest": ghl["cheapest"], "ghl_website": ghl["website"], "ghl_oracle": ghl["oracle"],
            "ghl_gap_closed": (ghl["website"] - ghl["cheapest"]) / (ghl["oracle"] - ghl["cheapest"]),
        })
    return pandas.DataFrame(rows)


# ---------------------------------------------------------------- 표 쓰기

def number(value, digits=4, sign=False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace("-", "−")


def p_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


NUMERIC_CELL = re.compile(r"^[−+\-<]?[\d.]")


def markdown(frame) -> str:
    """숫자 열은 오른쪽, 글자 열은 왼쪽으로 맞춘 Markdown 표."""
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join(" ---: " if all(NUMERIC_CELL.match(str(value)) for value in frame[column]) else " --- "
                          for column in frame.columns) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def write_table(name, raw, formatted, note) -> None:
    raw.to_csv(REPORT_DIRECTORY / "tables" / f"{name}.csv", index=False)
    (REPORT_DIRECTORY / "tables" / f"{name}.md").write_text(f"{markdown(formatted)}\n{note}\n", encoding="utf-8")


def write_tables(tables, chance) -> None:
    main = tables["main"]
    write_table("table1_main", main, pandas.DataFrame({
        "예산": main["budget"].map(BUDGET_KOREAN), "전략": main["strategy"].map(KOREAN),
        "평균 VUS-PR [95% 구간]": [f"{number(row.mean_vus_pr)} [{number(row.bootstrap_low)}, {number(row.bootstrap_high)}]"
                                for row in main.itertuples()],
        "중앙값": main["median_vus_pr"].map(number),
        "우연 대비 (배)": main["mean_chance_multiple"].map(lambda value: number(value, 2)),
        "우연 초과 파일": main["files_above_chance"].astype(str) + "/" + main["files"].astype(str),
        "사후 최적 격차 회수율": main["oracle_gap_closed"].map(lambda value: number(value, 2)),
        "평균 비용 (초)": main["mean_spent_seconds"].map(lambda value: number(value, 1)),
        "학습 횟수": main["mean_trainings"].map(lambda value: number(value, 1)),
        "예산 초과 파일": main["over_budget_files"],
    }), f"GHL 파일 {main['files'].max()}개. 구간은 파일 단위 bootstrap({BOOTSTRAP_DRAWS:,}회) 백분위 95%다. "
        f"우연 수준은 파일마다 균등 난수 점수의 VUS-PR(seed {int(chance['seeds'].min())}개 평균, "
        f"평균 {number(chance['chance'].mean())})이고 우연 대비는 파일별 배수의 평균이다. 격차 회수율은 "
        "(전략 − 가장 싼 계획) / (사후 최적 − 가장 싼 계획)이다. 비용은 웹사이트의 추정이며 GHL에서 재지 않았다.")

    contrasts = tables["contrasts"]
    write_table("table2_contrasts", contrasts, pandas.DataFrame({
        "예산": contrasts["budget"].map(BUDGET_KOREAN), "비교 기준": contrasts["baseline"].map(KOREAN),
        "평균 차이 [95% CI]": [f"{number(row.mean_difference, sign=True)} "
                            f"[{number(row.ci_low, sign=True)}, {number(row.ci_high, sign=True)}]"
                            for row in contrasts.itertuples()],
        "승/무/패": [f"{row.wins}/{row.ties}/{row.losses}" for row in contrasts.itertuples()],
        "Wilcoxon p": contrasts["wilcoxon_p"].map(p_text), "Holm p": contrasts["wilcoxon_p_holm"].map(p_text),
        "순열 p": contrasts["sign_flip_p"].map(p_text),
        "r (rank-biserial)": contrasts["rank_biserial"].map(lambda value: number(value, 2, sign=True)),
        "파일 하나 빼기 범위": [f"{number(row.leave_one_out_min, sign=True)} ~ {number(row.leave_one_out_max, sign=True)}"
                         for row in contrasts.itertuples()],
    }), "차이는 웹사이트 − 비교 기준이고 양수면 웹사이트가 높다. CI는 파일 단위 t 구간(멈춤 규칙과 같은 값)이다. "
        f"Holm은 이 표의 비교 {len(contrasts)}개에 붙였다. 순열 p는 부호 뒤집기 정확 검정이다. "
        "무작위는 예산을 보지 않아 제한 없음에서만 비교한다.")

    metrics = tables["prediction"]
    write_table("table3_prediction", metrics, pandas.DataFrame({
        "지표": metrics["metric"],
        "유사도 (웹사이트)": metrics["similarity"].map(lambda value: number(value, 3)),
        "유사도 없음 (Dev18 평균)": metrics["no_similarity"].map(lambda value: number(value, 3)),
        "유사도가 나은 파일": metrics["files_better_with_similarity"].map(
            lambda value: "–" if pandas.isna(value) else f"{int(value)}/10"),
        "Wilcoxon p (파일 단위)": metrics["wilcoxon_p"].map(p_text),
    }), "후보·비율마다 예측과 GHL 실측을 견준다. 1위 손실과 Spearman은 (파일, 비율) 60묶음 평균이고 p는 파일별 평균 "
        "10쌍의 Wilcoxon이다. 아래 두 줄은 후보·모델마다 파일·비율을 평균한 뒤의 순위 상관이다. Dev18 family 제외 "
        "검증(k=30)은 절대 오차 0.310, 예측 1위 손실 0.221이었다(streamlit_website/ml/README.md).")

    candidates = tables["candidates"].head(10)
    groups = len(tables["groups"])
    write_table("table4_top_candidates", tables["candidates"], pandas.DataFrame({
        "후보": candidates["candidate_id"].str.removesuffix("::"), "모델": candidates["model"],
        "웹사이트 예측": candidates["predicted"].map(lambda value: number(value, 3)),
        "Dev18 평균": candidates["dev18_mean"].map(lambda value: number(value, 3)),
        "GHL 실측 평균": candidates["actual"].map(number), "GHL 실측 중앙값": candidates["actual_median"].map(number),
        "예측 1위 횟수 (유사도 / 없음)": [f"{int(row.top1_similarity)} / {int(row.top1_no_similarity)}"
                                     for row in candidates.itertuples()],
    }), f"웹사이트 예측 상위 10개 후보(csv에 전체). 값은 파일·비율 평균이고, 1위 횟수는 (파일, 비율) {groups}묶음 중 "
        "그 예측에서 1위였던 수다. 그림 5의 근거 표다.")

    learning = tables["learning"]
    write_table("table5_learning", learning, pandas.DataFrame({
        "모델": learning["model"], "학습 비율 (%)": learning["ratio"], "실행 수": learning["runs"],
        "실측 중앙값": learning["actual_median"].map(number), "실측 평균": learning["actual_mean"].map(number),
        "실측 최대": learning["actual_max"].map(lambda value: number(value, 3)),
        "우연 2배 초과": learning["share_above_twice_chance"].map(lambda value: f"{value:.0%}"),
        "웹사이트 예측 평균": learning["predicted_mean"].map(lambda value: number(value, 3)),
    }), "실행 수는 (설정 × 파일) 중 성공한 수다. 우연 2배 초과는 그 파일 우연 수준의 2배를 넘은 실행의 몫이다. "
        "PaAno·GDN은 seed 3 하나로 돌렸다.")

    paths = tables["paths"]
    write_table("table6_paths", paths, pandas.DataFrame({
        "예산": paths["budget"].map(BUDGET_KOREAN), "전략": paths["strategy"].map(KOREAN),
        "서로 다른 경로": paths["distinct_paths"],
        "가장 흔한 경로를 받은 파일": paths["files_on_common_path"],
        "가장 흔한 경로": paths["common_path"],
    }), "q는 계획을 다시 세운 시점의 데이터 비율이고, · 뒤는 설정 ID 앞 7자리다. 괄호 없는 단계는 유지(keep)다. "
        "경로 수는 설정까지 같아야 같은 경로로 센다.")

    stopping = tables["stopping"]
    write_table("table7_stopping", stopping, pandas.DataFrame({
        "예산": stopping["budget"].map(BUDGET_KOREAN), "비교 기준": stopping["baseline"].map(KOREAN),
        "95% CI 반폭": stopping["half_width"].map(number), "기준": number(CONFIDENCE_HALF_WIDTH, 2),
        "충족": stopping["within_limit"].map({True: "예", False: "아니오"}),
    }), "결과를 보기 전에 정한 규칙: 파일 10·15·20·25개에서만 보고, 모든 반폭이 0.03 이하면 멈춘다.")

    dev18 = tables["dev18"]
    write_table("table8_dev18_vs_ghl", dev18, pandas.DataFrame({
        "예산": dev18["budget"].map(BUDGET_KOREAN),
        "Dev18 가장 싼": dev18["dev18_cheapest"].map(lambda value: number(value, 3)),
        "Dev18 웹사이트": dev18["dev18_website"].map(lambda value: number(value, 3)),
        "Dev18 사후 최적": dev18["dev18_oracle"].map(lambda value: number(value, 3)),
        "Dev18 회수율": dev18["dev18_gap_closed"].map(lambda value: number(value, 2)),
        "GHL 가장 싼": dev18["ghl_cheapest"].map(number), "GHL 웹사이트": dev18["ghl_website"].map(number),
        "GHL 사후 최적": dev18["ghl_oracle"].map(number),
        "GHL 회수율": dev18["ghl_gap_closed"].map(lambda value: number(value, 2)),
    }), "Dev18은 family 10개 평균이며 streamlit_website/ml/README.md의 표를 옮겼다(코드 없음). 두 데이터의 "
        "VUS-PR 수준이 달라 회수율 (웹사이트 − 가장 싼) / (사후 최적 − 가장 싼)으로 견준다. 사후 최적은 PaAno·GDN "
        "단일 seed의 우연한 고점을 고를 수 있어 높게 나올 수 있으므로 회수율은 보수적인 값이다.")

    costs = tables["costs"]
    write_table("table9_cost_ranks", costs, pandas.DataFrame({
        "장치": costs["device"].map({"cpu": "CPU", "cuda": "GPU"}),
        "비용": costs["kind"].map({"training": "학습", "inference": "추론", "runtime": "총 실행"}),
        "(파일, 비율) 묶음": costs["groups"],
        "후보 수": costs["candidates_min"].astype(str) + "~" + costs["candidates_max"].astype(str),
        "Spearman ρ 중앙값": costs["spearman_median"].map(lambda value: number(value, 3)),
        "평균": costs["spearman_mean"].map(lambda value: number(value, 3)),
        "ρ > 0 몫": costs["share_positive"].map(lambda value: f"{value:.0%}"),
    }), "웹사이트 비용 곡선(Dev18 DB 실측)을 GHL 실행과 같은 작업량(학습 = 그 비율의 prefix 행, 추론 = test 구간 행)에 "
        "넣은 추정과, GHL 실행의 .meta.json 실측(training_seconds·test_inference_seconds·runtime_seconds)의 후보 순위 "
        "상관이다. 총 실행은 학습 + 추론 추정을 runtime_seconds와 견준다. 장치가 DB와 달라 절대값은 견주지 않는다. "
        "head는 한 실행을 나눠 쓰므로 설정마다 한 번 센다. GHL에서 학습하는 CPU 후보는 없다.")


# ---------------------------------------------------------------- 그림

def save(fig, name) -> None:
    fig.savefig(REPORT_DIRECTORY / "figures" / f"{name}.pdf")
    fig.savefig(REPORT_DIRECTORY / "figures" / f"{name}.png", dpi=400)
    plt.close(fig)


def spread(values, gap, log=False) -> list[float]:
    """끝 라벨이 겹치지 않게 아래에서부터 최소 간격 gap만큼 벌린 y 위치. log면 log10 공간에서 벌린다."""
    positions = numpy.log10(values) if log else numpy.asarray(values, dtype=float)
    order = numpy.argsort(positions, kind="stable")
    placed = positions.copy()
    for previous, index in zip(order, order[1:]):
        placed[index] = max(placed[index], placed[previous] + gap)
    return (10 ** placed if log else placed).tolist()


def end_labels(ax, x, ends, gap, log=False) -> None:
    names = list(ends)
    for name, y in zip(names, spread([ends[name] for name in names], gap, log=log)):
        ax.text(x, y, LABEL.get(name, name), color=COLOR.get(name, GRAY), va="center", fontsize=7)


def chance_band(ax, chance, right=1.0) -> None:
    """파일별 우연 수준의 범위(띠)와 평균(점선). right(축 비율)에서 끊어 끝 라벨 자리는 비운다."""
    ax.axhspan(chance["chance"].min(), chance["chance"].max(), xmax=right, color=LIGHT, alpha=0.55, lw=0, zorder=0)
    ax.axhline(chance["chance"].mean(), xmax=right, color=GRAY, lw=0.6, ls=(0, (2, 2)), zorder=0)


def log_ticks(ax, ticks) -> None:
    low, high = ax.get_ylim()
    ticks = [tick for tick in ticks if low <= tick <= high]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{tick:g}" for tick in ticks])


def figure_budget(main, chance) -> None:
    """예산별 전략의 평균 실현 VUS-PR. 100배까지는 모든 전략이 우연 수준 아래에 머문다."""
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    last = len(BUDGETS) - 1
    left, right = -0.35, last + 1.4
    chance_band(ax, chance, right=(last + 0.3 - left) / (right - left))
    offsets = {"oracle": -0.12, "dp": -0.06, "website": 0.0, "no_similarity": 0.06, "cheapest": 0.12}
    ends = {}
    for strategy in ("cheapest", "no_similarity", "dp", "oracle", "website"):
        rows = main[main["strategy"] == strategy].set_index("budget").loc[list(BUDGETS)]
        ax.errorbar(numpy.arange(len(BUDGETS)) + offsets[strategy], rows["mean_vus_pr"],
                    yerr=[rows["mean_vus_pr"] - rows["bootstrap_low"], rows["bootstrap_high"] - rows["mean_vus_pr"]],
                    color=COLOR[strategy], lw=1.5 if strategy == "website" else 0.9, elinewidth=0.6, capsize=0,
                    ls=(0, (4, 2)) if strategy == "oracle" else "-", marker="o", ms=3, zorder=3)
        ends[strategy] = rows["mean_vus_pr"].iloc[-1]
    random = main[main["strategy"] == "random"].iloc[0]
    ax.errorbar([last + 0.2], [random["mean_vus_pr"]],
                yerr=[[random["mean_vus_pr"] - random["bootstrap_low"]],
                      [random["bootstrap_high"] - random["mean_vus_pr"]]],
                color=COLOR["random"], marker="s", mfc="white", mew=0.8, ms=3.2, elinewidth=0.6, capsize=0, zorder=3)
    ends["random"] = random["mean_vus_pr"]
    ends["Chance"] = chance["chance"].mean()
    end_labels(ax, last + 0.38, ends, 0.075, log=True)
    ax.set_yscale("log")
    ax.set_ylim(min(main["bootstrap_low"].min(), chance["chance"].min()) * 0.85, main["bootstrap_high"].max() * 1.3)
    log_ticks(ax, (0.005, 0.01, 0.02, 0.05, 0.1, 0.2))
    ax.set_xticks(range(len(BUDGETS)), [BUDGET_LABEL[budget] for budget in BUDGETS])
    ax.set_xlim(left, right)
    ax.spines["bottom"].set_bounds(left, last + 0.3)
    ax.set_xlabel("Budget (multiple of the cheapest plan's cost)")
    ax.set_ylabel("Realized VUS-PR (mean of files)")
    save(fig, "fig1_budget_vus_pr")


def figure_differences(plans) -> None:
    """파일별 웹사이트 − 기준선 차이와 평균의 95% t 구간."""
    baselines = ("cheapest", "no_similarity", "dp")
    differences = {(budget, baseline): (by_file(plans, budget, "website") - by_file(plans, budget, baseline)).to_numpy()
                   for budget in BUDGETS for baseline in baselines}
    contrasts = {key: paired_contrast(values) for key, values in differences.items()}
    # 신뢰구간은 모두 보이게 두고, 그보다 먼 점만 축 끝에 화살표와 값으로 남긴다.
    low = min(-0.006, min(contrast["ci_low"] for contrast in contrasts.values()) * 1.15)
    high = max(0.02, max(contrast["ci_high"] for contrast in contrasts.values()) * 1.08)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.0), sharey=True, gridspec_kw={"wspace": 0.12})
    rng = numpy.random.default_rng(0)
    for ax, baseline in zip(axes, baselines):
        ax.axvline(0, color=INK, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for row, budget in enumerate(BUDGETS):
            y = len(BUDGETS) - 1 - row
            values = differences[budget, baseline]
            jitter = y + 0.13 + rng.uniform(-0.1, 0.1, len(values))
            inside = (values >= low) & (values <= high)
            ax.scatter(values[inside], jitter[inside], s=7, color=GRAY, lw=0, zorder=2)
            for beyond, edge, marker, align in ((values > high, high, ">", "right"), (values < low, low, "<", "left")):
                if beyond.any():
                    ax.plot(edge, y + 0.13, marker, color=GRAY, ms=3.2, clip_on=False, zorder=2)
                    ax.text(edge, y + 0.3, ", ".join(number(value, 3, sign=True) for value in sorted(values[beyond])),
                            fontsize=5.8, color=GRAY, ha=align, va="bottom")
            contrast = contrasts[budget, baseline]
            ax.plot([contrast["ci_low"], contrast["ci_high"]], [y - 0.2] * 2, color=COLOR["website"], lw=1.1,
                    solid_capstyle="butt", zorder=3)
            ax.plot(contrast["mean_difference"], y - 0.2, "D", color=COLOR["website"], ms=3, zorder=4)
        ax.set_xlim(low, high)
        ax.set_title(f"vs {LABEL[baseline]}", color=INK if baseline == "cheapest" else COLOR[baseline])
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda value, _: "0" if abs(value) < 1e-12 else number(value, 2, sign=True)))
    axes[0].set_yticks(range(len(BUDGETS)), [BUDGET_LABEL[budget] for budget in reversed(BUDGETS)])
    axes[0].set_ylim(-0.6, len(BUDGETS) - 0.35)
    axes[1].set_xlabel(r"Website $-$ baseline (VUS-PR per file; diamond: mean, bar: 95% CI)")
    save(fig, "fig2_paired_differences")


def figure_timeline(segments, chance) -> None:
    """운영 기간 동안 배치된 모델의 실측 VUS-PR(파일 평균). 선 아래 면적이 실현값이다."""
    strategies = ("cheapest", "no_similarity", "oracle", "website")
    budgets = (("100", "(a)"), ("unlimited", "(b)"))
    means = {(budget, strategy): segments[(segments["budget"] == budget) & (segments["strategy"] == strategy)]
             .groupby("ratio").agg(vus=("vus_pr", "mean"), start=("start_day", "mean"), end=("end_day", "mean"))
             for budget, _ in budgets for strategy in strategies}
    ymax = max(frame["vus"].max() for frame in means.values()) * 1.1
    width = OPERATING_DAYS * 1.3
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (budget, tag) in zip(axes, budgets):
        labelled = ax is axes[-1]
        chance_band(ax, chance, right=OPERATING_DAYS / width if labelled else 1.0)
        for strategy in strategies:
            frame = means[budget, strategy]
            edges = numpy.r_[frame["start"].to_numpy(), frame["end"].iloc[-1]]
            ax.stairs(frame["vus"].to_numpy(), edges, baseline=None, color=COLOR[strategy],
                      lw=1.5 if strategy == "website" else 0.9, ls=(0, (4, 2)) if strategy == "oracle" else "-",
                      zorder=3 if strategy == "website" else 2)
        for start in edges[1:-1]:
            ax.axvline(start, color=LIGHT, lw=0.5, zorder=0)
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, width if labelled else OPERATING_DAYS)
        ax.spines["bottom"].set_bounds(0, OPERATING_DAYS)
        ax.set_xticks(range(0, OPERATING_DAYS + 1, 20))
        ax.set_xlabel("Operating day")
        # 다시 계획한 시점(데이터가 DB 비율에 닿은 날)을 위 축에 적는다.
        top = ax.secondary_xaxis("top")
        top.set_xticks(edges[:-1], [str(ratio) for ratio in SERVICE_RATIOS])
        top.tick_params(length=2, labelsize=6.3, colors=GRAY, pad=1)
        top.spines["top"].set_bounds(0, OPERATING_DAYS)
        top.spines["top"].set_color(LIGHT)
        top.set_xlabel("Re-planned at data share (%)", fontsize=6.8, color=GRAY, labelpad=2)
        ax.text(0.015, 0.97, f"{tag} Budget {BUDGET_LABEL[budget]}", transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top")
        if labelled:
            ends = {strategy: means[budget, strategy]["vus"].iloc[-1] for strategy in strategies}
            end_labels(ax, OPERATING_DAYS + 2, {**ends, "Chance": chance["chance"].mean()}, ymax * 0.07)
    axes[0].set_ylabel("VUS-PR of deployed model\n(mean of files)")
    save(fig, "fig3_operating_timeline")


def figure_cost(main) -> None:
    """평균 추정 비용과 평균 실현 VUS-PR. 웹사이트는 예산이 넉넉할 때만 비용을 쓴다."""
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN * 1.25, 2.4))
    ends = {}
    for strategy in ("cheapest", "no_similarity", "dp", "oracle", "website"):
        rows = main[main["strategy"] == strategy].set_index("budget").loc[list(BUDGETS)]
        ax.plot(rows["mean_spent_seconds"], rows["mean_vus_pr"], color=COLOR[strategy],
                lw=1.5 if strategy == "website" else 0.9, ls=(0, (4, 2)) if strategy == "oracle" else "-",
                marker="o", ms=3, zorder=3 if strategy == "website" else 2)
        ends[strategy] = (rows["mean_spent_seconds"].iloc[-1], rows["mean_vus_pr"].iloc[-1])
        if strategy == "website":
            for budget in ("100", "unlimited"):
                point = (rows.loc[budget, "mean_spent_seconds"], rows.loc[budget, "mean_vus_pr"])
                ax.annotate(BUDGET_LABEL[budget], point,
                            xytext=(-4, 5), textcoords="offset points", fontsize=6.3, color=COLOR[strategy], ha="right")
    random = main[main["strategy"] == "random"].iloc[0]
    ends["random"] = (random["mean_spent_seconds"], random["mean_vus_pr"])
    ax.plot(*ends["random"], "s", color=COLOR["random"], mfc="white", mew=0.8, ms=3.5)
    offsets = {"oracle": (5, 0, "left"), "website": (5, 4, "left"), "dp": (5, -5, "left"),
               "no_similarity": (0, -9, "center"), "cheapest": (4, -8, "left"), "random": (0, 7, "center")}
    for strategy, (dx, dy, align) in offsets.items():
        ax.annotate(LABEL[strategy], ends[strategy], xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    color=COLOR[strategy], ha=align, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(main["mean_vus_pr"].min() * 0.8, main["mean_vus_pr"].max() * 1.2)
    log_ticks(ax, (0.01, 0.02, 0.05, 0.1))
    ax.set_xlim(1, main["mean_spent_seconds"].max() * 8)
    ax.set_xlabel("Mean estimated cost per file (s)")
    ax.set_ylabel("Realized VUS-PR (mean of files)")
    save(fig, "fig4_cost_vs_vus_pr")


def figure_candidates(candidates, groups, chance) -> None:
    """후보마다 예측(가로)과 GHL 실측(세로). 동그라미는 (파일, 비율) 묶음에서 예측 1위였던 후보다."""
    panels = (("dev18_mean", "no_similarity", "(a) No similarity", "Predicted VUS-PR (Dev18 equal-weight mean)"),
              ("predicted", "similarity", "(b) Website", "Predicted VUS-PR (similarity-weighted)"))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (column, key, tag, xlabel) in zip(axes, panels):
        chance_band(ax, chance)
        others = candidates[~candidates["model"].isin(MODEL_COLOR)]
        ax.scatter(others[column], others["actual"], s=9, color="#BDBDBD", lw=0, zorder=2)
        for model, color in MODEL_COLOR.items():
            rows = candidates[candidates["model"] == model]
            ax.scatter(rows[column], rows["actual"], s=13, color=color, lw=0, zorder=3)
            # GDN은 실측이 가장 높은 점 왼쪽에, 바닥에 몰린 PaAno는 예측이 가장 높은 점 아래에 이름을 둔다.
            label = rows.loc[rows["actual"].idxmax()] if model == "GDN" else rows.loc[rows[column].idxmax()]
            ax.annotate(model, (label[column], label["actual"]), xytext=(-5, 0) if model == "GDN" else (0, -9),
                        textcoords="offset points", fontsize=7, color=color,
                        ha="right" if model == "GDN" else "center", va="center" if model == "GDN" else "top")
        for row in candidates[candidates[f"top1_{key}"] > 0].itertuples():
            point = (getattr(row, column), row.actual)
            ax.scatter(*point, s=55, facecolors="none", edgecolors=INK, lw=0.7, zorder=4)
            ax.annotate(f"top-1 in {int(getattr(row, f'top1_{key}'))}/{len(groups)}", point, xytext=(0, 9),
                        textcoords="offset points", fontsize=6.3, color=INK, ha="center", va="bottom")
        rho = spearmanr(candidates[column], candidates["actual"]).statistic
        ax.text(0.015, 0.97, tag, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")
        ax.text(0.015, 0.86, rf"Spearman $\rho = {rho:.2f}$ ({len(candidates)} candidates)", transform=ax.transAxes,
                fontsize=7, va="top")
        ax.set_xlabel(xlabel)
    axes[0].set_ylim(0, candidates["actual"].max() * 1.35)
    axes[0].set_ylabel("GHL VUS-PR\n(mean over files and data shares)")
    save(fig, "fig5_candidate_predictions")


def figure_learning(predictions, chance) -> None:
    """데이터 비율에 따른 GHL 실측. GDN은 중앙값이 아니라 위쪽 꼬리가 데이터와 함께 늘고, PaAno는 그대로다."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.2), sharey=True, gridspec_kw={"wspace": 0.08})
    x = numpy.arange(len(SERVICE_RATIOS))
    right = len(x) + 0.6
    for ax, model, tag in zip(axes, MODEL_COLOR, ("(a)", "(b)")):
        chance_band(ax, chance, right=(len(x) - 0.7 + 0.3) / (right + 0.3))
        rows = predictions[predictions["model"] == model].pivot_table(
            index=["series", "candidate_id"], columns="ratio", values="actual_vus_pr").reindex(columns=SERVICE_RATIOS)
        for values in rows.to_numpy():
            ax.plot(x, values, color=MODEL_COLOR[model], lw=0.4, alpha=0.35, zorder=1)
        median, mean = rows.median(), rows.mean()
        ax.plot(x, median, color=MODEL_COLOR[model], lw=1.6, marker="o", ms=3, zorder=3)
        ax.plot(x, mean, color=MODEL_COLOR[model], lw=1.0, ls=(0, (3, 2)), marker="o", ms=2.2, mfc="white", zorder=3)
        positions = spread([mean.iloc[-1], median.iloc[-1]], 0.12, log=True)
        for name, value, y in zip(("mean", "median"), (mean.iloc[-1], median.iloc[-1]), positions):
            ax.text(len(x) - 1 + 0.18, y, f"{name} {number(value, 3)}", color=MODEL_COLOR[model], va="center",
                    fontsize=6.6)
        configs = rows.index.get_level_values("candidate_id").nunique()
        files = rows.index.get_level_values("series").nunique()
        ax.text(0.01, 1.0, f"{tag} {model} ({configs} configs " + r"$\times$" + f" {files} files)",
                transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
        ax.set_xticks(x, [str(ratio) for ratio in SERVICE_RATIOS])
        ax.set_xlim(-0.3, right)
        ax.spines["bottom"].set_bounds(-0.3, len(x) - 0.7)
        ax.set_xlabel("Training data (% of the final prefix)")
    axes[0].set_yscale("log")
    values = predictions[predictions["model"].isin(MODEL_COLOR)]["actual_vus_pr"].dropna()
    axes[0].set_ylim(min(values.min(), chance["chance"].min()) * 0.8, values.max() * 1.4)
    log_ticks(axes[0], (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0))
    axes[0].set_ylabel("GHL VUS-PR")
    save(fig, "fig6_learning_curves")


# ---------------------------------------------------------------- 기록

FIGURES = (
    ("fig1_budget_vus_pr", "예산별 전략의 평균 실현 VUS-PR (막대: bootstrap 95% 구간, 회색 띠: 균등 난수 점수)",
     "Realized VUS-PR by budget and strategy; bars are file-level bootstrap 95% intervals, the gray band is the "
     "random-score level.", "plans.csv, chance_vus_pr.csv"),
    ("fig2_paired_differences", "파일별 웹사이트 − 기준선 차이 (점: 파일, 마름모: 평균, 막대: 95% t 구간)",
     "Per-file paired differences, website minus baseline; diamond and bar show the mean and its 95% t interval.",
     "plans.csv"),
    ("fig3_operating_timeline", "운영 기간 동안 배치된 모델의 실측 VUS-PR (100배, 제한 없음). 면적이 실현값이다",
     "VUS-PR of the deployed model over the operating period; the area under each line is the realized value.",
     "plans.csv path + predictions.csv actual_vus_pr"),
    ("fig4_cost_vs_vus_pr", "전략별 평균 추정 비용과 평균 실현 VUS-PR (점: 예산 1.5·10·100배·제한 없음)",
     "Mean estimated cost against realized VUS-PR; points along each line are the four budgets.", "plans.csv"),
    ("fig5_candidate_predictions", "후보별 예측 대 GHL 실측: (a) 유사도 없음 (b) 웹사이트. 동그라미는 예측 1위 후보",
     "Predicted against GHL VUS-PR per candidate without and with similarity; rings mark the candidates "
     "predicted first.", "predictions.csv"),
    ("fig6_learning_curves", "학습 데이터 비율별 GHL 실측 VUS-PR (가는 선: 설정·파일, 실선: 중앙값, 점선: 평균)",
     "GHL VUS-PR against training-data share for the two learned models; thin lines are config-file pairs.",
     "predictions.csv, chance_vus_pr.csv"),
)
TABLES = (
    ("table1_main", "예산 × 전략 주 결과"), ("table2_contrasts", "웹사이트 대 기준선 짝 비교"),
    ("table3_prediction", "ML 성능 예측 정확도 (유사도 유무)"), ("table4_top_candidates", "후보 단위 예측·실측"),
    ("table5_learning", "PaAno·GDN 학습 비율별 실측"), ("table6_paths", "전략이 고른 경로"),
    ("table7_stopping", "멈춤 규칙 점검"), ("table8_dev18_vs_ghl", "Dev18 백테스트와의 비교"),
    ("table9_cost_ranks", "비용 추정 대 실측 시간의 후보 순위 상관"),
)


def write_readme(chance) -> None:
    lines = [
        "# GHL 서비스 백테스트 보고서 자료", "",
        "`python -m tests.ghl_main.report_ghl_service`로 다시 만든다. 입력은 같은 폴더 위의 "
        "`plans.csv`, `predictions.csv`, `cost_ranks.csv`, `stopping.json`, `prediction_summary.json`, "
        "`chance_vus_pr.csv`다.", "",
        "## 그림 (figures/, PDF는 보고서·PNG는 슬라이드)", "",
        "| 파일 | 내용 | 영문 캡션 | 출처 |", "| --- | --- | --- | --- |",
        *[f"| {name} | {korean} | {english} | {source} |" for name, korean, english, source in FIGURES], "",
        "## 표 (tables/, md는 붙여넣기용·csv는 원값)", "",
        "| 파일 | 내용 |", "| --- | --- |", *[f"| {name} | {korean} |" for name, korean in TABLES], "",
        "## 해석할 때 주의", "",
        f"- GHL test 구간의 이상 비율은 {chance['prevalence'].min():.2%}~{chance['prevalence'].max():.2%}이고, "
        f"균등 난수 점수의 VUS-PR은 평균 {chance['chance'].mean():.4f}다. 0.01 안팎의 값은 우연 수준이다.",
        "- 10개 파일은 같은 설비(GHL)의 시나리오라 서로 독립인 현장이 아니다. 웹사이트 경로가 파일마다 거의 같다(표 6).",
        "- PaAno·GDN은 seed 3 하나로 돌렸다. 사후 최적은 이 단일 seed의 우연한 고점(예: 0.7)을 사후에 고를 수 있어 "
        "도달할 수 없는 상한이고, 그래서 사후 최적 격차 회수율은 보수적인 값이다.",
        "- 계획 비용은 웹사이트의 추정(DB 실측 기반)이다. GHL 실행 시간은 장치가 DB와 달라 후보 순위 상관만 본다(표 9).",
        "- 무작위는 실행에 실패해 실측이 없는 후보를 웹사이트 경로처럼 0으로 센다.",
        "- p값은 멈춘 뒤 한 번만 본다. 멈춤은 유의성이 아니라 신뢰구간 반폭으로 정했다(표 7).",
    ]
    (REPORT_DIRECTORY / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def key_numbers(tables, chance) -> dict:
    main = tables["main"].set_index(["budget", "strategy"])
    contrasts = tables["contrasts"].set_index(["budget", "baseline"])
    return {
        "files": int(tables["main"]["files"].max()),
        "prevalence_range": [float(chance["prevalence"].min()), float(chance["prevalence"].max())],
        "chance_vus_pr_mean": float(chance["chance"].mean()),
        "mean_vus_pr": {f"{budget}:{strategy}": float(value) for (budget, strategy), value
                        in main["mean_vus_pr"].items()},
        "website_minus": {f"{budget}:{baseline}": {
            "mean": float(row.mean_difference), "ci": [float(row.ci_low), float(row.ci_high)],
            "wilcoxon_p": float(row.wilcoxon_p), "holm_p": float(row.wilcoxon_p_holm),
            "sign_flip_p": float(row.sign_flip_p),
            "wins_ties_losses": [int(row.wins), int(row.ties), int(row.losses)],
            "leave_one_out": [float(row.leave_one_out_min), float(row.leave_one_out_max)]}
            for (budget, baseline), row in contrasts.iterrows()},
        "oracle_gap_closed_by_website": {budget: float(main.loc[(budget, "website"), "oracle_gap_closed"])
                                         for budget in BUDGETS},
        "prediction": tables["prediction"].set_index("metric")[["similarity", "no_similarity"]].to_dict("index"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chance-seeds", type=int, default=5, help="파일마다 우연 수준을 잴 난수 점수 수")
    parser.add_argument("--workers", type=int, default=4, help="우연 수준 계산 프로세스 수")
    arguments = parser.parse_args()

    backtest = load_backtest()
    plans, predictions = backtest["plans"], backtest["predictions"]
    chance = chance_levels(backtest["files"], arguments.chance_seeds, arguments.workers)
    segments = deployed_segments(plans, predictions)
    print(f"경로 {len(plans[plans['strategy'] != 'random'])}개를 실측으로 다시 더해 plans.csv와 맞는 것을 확인했다")

    main_rows = main_table(plans, chance)
    contrasts = contrast_table(plans)
    metrics, candidates, groups = prediction_tables(predictions, backtest["prediction_summary"])
    tables = {
        "main": main_rows, "contrasts": contrasts, "prediction": metrics, "candidates": candidates,
        "groups": groups, "learning": learning_table(predictions, chance), "paths": path_table(segments),
        "stopping": stopping_table(contrasts, backtest["stopping"]), "dev18": dev18_table(main_rows),
        "costs": cost_table(backtest["cost_ranks"]),
    }
    for folder in ("figures", "tables"):
        (REPORT_DIRECTORY / folder).mkdir(parents=True, exist_ok=True)
    write_tables(tables, chance)
    segments.to_csv(REPORT_DIRECTORY / "tables" / "deployed_segments.csv", index=False)
    chance.reset_index().to_csv(REPORT_DIRECTORY / "tables" / "chance_by_file.csv", index=False)

    apply_style()
    figure_budget(main_rows, chance)
    figure_differences(plans)
    figure_timeline(segments, chance)
    figure_cost(main_rows)
    figure_candidates(candidates, groups, chance)
    figure_learning(predictions, chance)
    write_readme(chance)
    numbers = key_numbers(tables, chance)
    (REPORT_DIRECTORY / "key_numbers.json").write_text(
        json.dumps(numbers, ensure_ascii=False, indent=2, default=int) + "\n", encoding="utf-8")

    for name in ("table1_main", "table2_contrasts", "table3_prediction"):
        print((REPORT_DIRECTORY / "tables" / f"{name}.md").read_text(encoding="utf-8"))
    print(f"표·그림을 {REPORT_DIRECTORY}에 저장했다")


if __name__ == "__main__":
    main()
