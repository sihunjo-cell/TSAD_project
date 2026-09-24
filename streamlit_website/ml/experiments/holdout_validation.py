"""Held-out Dev18 검증: leave-series-out으로 Recall@N 등을 측정한다.

## 검증 설계 (요구사항/결정 근거)

- **"좋은 후보"의 정의 (사용자 확정, 2026-09-24)**: held-out series의 실제 최종
  (q=100) VUS-PR 상위 K개 (config_id, head) 후보. 이미 있는 `vus_pr` 컬럼만 쓰는
  가장 단순한 V1 정의다 (예: cost 대비 성능 같은 다른 정의는 문서 근거가 없어
  채택하지 않았다). K는 이 harness에서 `top_n`과 같은 값을 쓴다 — Recall@N을
  "ML이 고른 N개짜리 Top-N"과 "실제로 좋은 N개"를 같은 크기로 비교하기 위해서다.
- **leave-series-out**: held-out series 자신의 prefix_features 행은 similarity
  비교 대상 pool(과거 사례)에서 제외한다. 그 series의 결과(성능)는 ground truth로만
  쓴다 — similarity가 "미래를 미리 본" 것이 되지 않도록 한다.
- **"현재 관측"은 held-out series의 한 q_percent 시점을 그대로 쓴다.** 이 q_percent
  선택 자체가 팀 피드백 문서가 요구한 "관측량(%) sensitivity" 축이므로, 이 함수는
  이를 파라미터(`observed_q_percent`)로 받는다 — 하나로 임의 고정하지 않는다.
  N(top_k_candidates)도 별도 축으로 파라미터화한다 (%와 N을 섞지 않는다는 원칙).

## 이 harness가 반영하는 것 (명시적 범위, 2026-09-25 갱신)

- **cost tie-break (2026-09-25 추가).** `cost.py::build_candidate_cost_model`/
  `estimate_stage_cost`를 그대로 재사용해 후보별 예상 총 비용(초)을 추정하고,
  `candidate_selection.py`와 동일하게 "predicted_performance desc, cost asc"
  순서로 predicted_top_n을 정렬한다. 비용 추정의 `inference_volume`/`training_rows`는
  held-out series 자신의 **실제 미래(q=100) 규모**를 그대로 쓴다 — 실제 운영
  조건(`operating_conditions`)을 held-out 검증이 임의로 지어낼 수 없으므로, "미래에
  실제로 그 series가 그 정도 규모가 될 것"이라는 이미 확정된 ground truth 값을
  대신 쓰는 것이며 이는 임의 가정이 아니다.
- **feasibility (2026-09-25 추가).** `feasibility.py::assess_future_feasibility`가
  요구하는 `channel_count`는 `channel_features` 테이블에서, `stage_training_rows`는
  held-out series 자신의 실제 q=100 규모에서 그대로 얻는다. 마지막으로 필요한
  `stage_inference_rows`("한 번에 모델이 보는 연속 스트림 길이")는 원래 실제
  운영 조건(`operating_conditions`) 입력인데, held-out 검증이 그 값을 임의로
  지어낼 수는 없다 — 대신 held-out series 자신의 **실제 q=100 시점 `test_observations`**
  (그 series를 실제로 평가할 때 쓰인 연속 테스트 구간 길이)를 쓴다. 실제 DB로
  확인한 결과 이 값은 같은 series 안에서 어느 후보(모델)로 평가하든 완전히
  동일한 series 속성이라(예: series '01'은 항상 1900), "그 series가 실제로 이
  길이의 연속 구간으로 평가됐다"는 사실을 그대로 쓰는 것이지 지어낸 시나리오가
  아니다. 이 값을 찾을 수 없는 series(이론상 가능, 실제 dev18 18개 series는 전부
  존재)에 대해서는 feasibility를 계산하지 않고 `feasibility_evaluated=False`로
  표시한다 (억지로 채우지 않는다).
- **top-N 순위 경쟁은 `candidate_selection.py`의 `_is_usable`과 동일한 기준을 쓴다**:
  feasible이고, predicted_performance와 estimated_total_cost가 모두 있는 후보만
  `predicted_top_n` 경쟁에 넣는다. feasibility를 계산하지 못한 경우(위 조건 미충족)에는
  이전 버전과 같이 feasibility를 걸지 않는다 — "모른다"를 "탈락"으로 처리하지 않는다.
- **cost 회귀의 데이터 누수 수정 (2026-09-25 팀 리뷰에서 발견).** similarity/performance는
  처음부터 `matches`(leave-series-out된 `historical_pool` 기반)만 써서 held-out
  series 자신의 데이터를 보지 않았지만, cost 회귀(`build_candidate_cost_model`)는
  `results_rows`(전체 series 포함, held-out 자신도 포함)를 그대로 넘기고 있었다 —
  "held-out의 미래 비용을 예측"한다면서 실제로는 그 series 자신의 실측 비용
  관측치가 회귀 학습 데이터에 섞여 있던 데이터 누수였다. 이제 cost 회귀에 넘기는
  데이터도 `results_rows`에서 held-out series 행을 제외해서 만든다.

## "성능만 검증하는 것 아니냐"는 팀 리뷰 지적에 대한 보완 (2026-09-25 추가)

기존 `ground_truth_top_n`(정답)은 실제 VUS-PR만으로 순위를 매긴다. `predicted_top_n`
(예측)은 "성능 desc → 비용 asc"로 순위를 매기는데, VUS-PR이 연속값이라 동률이
거의 없어 비용이 순위에 개입할 일이 드물다 — 즉 `recall_at_n`은 사실상 "성능
예측이 얼마나 맞는가"만 검증하고, 비용까지 고려한 정책이 맞는 선택을 하는지는
검증하지 못한다는 지적이 팀 리뷰에서 나왔다. 이를 보완하기 위해 두 가지를
추가했다 (코드 변경 없이 새 필드만 추가 — 기존 `recall_at_n`의 정의/의미는
그대로 유지):

- **`ground_truth_policy_top_n`/`recall_policy_at_n`**: 정답도 예측과 **똑같은
  정책**("성능 desc → 비용 asc")을 적용해서 다시 만든다. 다만 정답 쪽 비용은
  `cost.py`로 추정한 값이 아니라, held-out series에서 **실제로 관측된**
  `actual_runtime_seconds`(q=100, `status='complete'`인 실행들의 평균)를 쓴다 —
  정답은 추정이 아니라 실측이어야 하므로 예측 파이프라인이 쓰는 회귀 추정치를
  그대로 재사용하지 않는다. `actual_runtime_seconds` 관측이 없는 candidate는
  이 정답 집합에서 제외한다(억지로 채우지 않는다). 이 값이 기존 `recall_at_n`과
  크게 다르지 않다면 "비용을 고려해도 결론이 바뀌지 않는다"는 뜻이고, 크게
  다르다면 지금 검증이 성능에만 치우쳐 있었다는 뜻이다.
- **`actual_mean_cost_seconds_of_predicted_top_n`/
  `actual_mean_cost_seconds_of_performance_only_ground_truth_top_n`**: "성능만
  봤을 때 골랐을 top-N"과 "ML이 실제로 고른 top-N"이 held-out series에서 **실제로
  얼마나 비용이 들었는지**(`actual_runtime_seconds` 실측 평균)를 나란히 비교한다.
  후보 구성이 다르더라도, ML이 평균적으로 더 싼 후보를 고르는 경향이 있는지를
  직접 잰다 — 순위 일치 여부(Recall)와는 별개의, "선택이 비용 효율적이었는가"에
  대한 답이다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from streamlit_website.ml import db
from streamlit_website.ml.config import DEFAULT_CONFIG, SUPPORTED_SIMILARITY_METRICS, MLConfig
from streamlit_website.ml.cost import build_candidate_cost_model, estimate_stage_cost
from streamlit_website.ml.feasibility import assess_future_feasibility
from streamlit_website.ml.features import extract_historical_features
from streamlit_website.ml.performance import estimate_stage_performance
from streamlit_website.ml.pipeline import TRAINING_REQUIRED_TARGET_USE
from streamlit_website.ml.similarity import find_similar_historical_prefixes
from streamlit_website.ml.trajectory import get_series_trajectory, trajectory_point_at_q


@dataclass(frozen=True)
class HoldoutEvaluationResult:
    held_out_series: str
    observed_q_percent: int
    top_n: int
    total_candidates: int
    good_candidate_definition: str
    ground_truth_top_n: tuple[str, ...]   # 실제 q=100 vus_pr 상위 top_n개 candidate_id
    predicted_top_n: tuple[str, ...]      # (performance desc, cost asc)로 정렬한 상위 top_n개 candidate_id
    recall_at_n: float | None             # ground_truth_top_n이 비어 있으면 None (계산 불가)
    candidate_reduction_ratio: float      # 1 - top_n/total_candidates
    ml_runtime_seconds: float
    similarity_matches_found: int
    candidates_with_ground_truth: int     # q=100 결과가 실제로 있는 candidate 수
    candidates_with_prediction: int       # similarity로 예측 가능했던 candidate 수
    candidates_with_cost_estimate: int    # cost.py로 총비용을 추정할 수 있었던 candidate 수
    mean_estimated_total_cost_seconds_of_top_n: float | None  # predicted_top_n의 평균 예상 비용(초)
    feasibility_evaluated: bool           # held-out series에 대해 feasibility를 계산할 수 있었는지
    stage_inference_rows_used: int | None  # feasibility에 쓴 held-out series의 실제 test_observations
    candidates_feasible: int              # feasible=True인 candidate 수 (미평가 시 0)
    candidates_usable: int                # feasible+performance+cost가 모두 있어 top-N 경쟁에 실제로 들어간 candidate 수
    # "성능만 검증" 지적 보완 (Method 1/2, 2026-09-25 추가) — 모듈 docstring 참고.
    ground_truth_policy_top_n: tuple[str, ...]  # 정답도 예측과 동일 정책("성능desc,실측비용asc")으로 재정렬한 top_n
    recall_policy_at_n: float | None            # ground_truth_policy_top_n이 비어 있으면 None
    actual_mean_cost_seconds_of_predicted_top_n: float | None  # predicted_top_n의 실측 평균 비용(초)
    actual_mean_cost_seconds_of_performance_only_ground_truth_top_n: float | None  # ground_truth_top_n(성능만)의 실측 평균 비용(초)


GOOD_CANDIDATE_DEFINITION = (
    "held_out series의 실제 최종(q=100) VUS-PR 상위 top_n개 (config_id, head) 후보"
)


def _candidate_id(config_id: str, head: str) -> str:
    return f"{config_id}::{head}"


def evaluate_holdout(
    held_out_series: str,
    observed_q_percent: int,
    *,
    database=None,
    config: MLConfig = DEFAULT_CONFIG,
    top_n: int | None = None,
    _all_prefixes: list[dict] | None = None,
    _results_by_candidate: dict[tuple[str, str], list[dict]] | None = None,
    _candidate_definitions: dict[str, dict] | None = None,
    _channel_count_by_prefix: dict[str, int] | None = None,
) -> HoldoutEvaluationResult:
    """한 held-out series, 한 관측 시점(%), 한 N에 대해 Recall@N 등을 계산한다.

    `_all_prefixes`/`_results_by_candidate`/`_candidate_definitions`/
    `_channel_count_by_prefix`는 내부 최적화용이다 — `sweep_*` 함수들이 같은
    held-out series에 대해 N/k만 바꿔가며 여러 번 호출할 때, DB를 매번 다시 읽지
    않고 한 번 읽은 것을 재사용하기 위해 넘긴다. 직접 호출할 때는 생략하면 된다
    (이 함수가 알아서 DB에서 읽는다).
    """
    top_n = config.top_k_candidates if top_n is None else top_n
    if top_n < 1:
        raise ValueError("top_n은 1 이상이어야 한다")

    all_prefixes = _all_prefixes if _all_prefixes is not None else db.load_historical_prefixes(database)
    held_out_row = next(
        (row for row in all_prefixes if row["series"] == held_out_series and row["q_percent"] == observed_q_percent),
        None,
    )
    if held_out_row is None:
        raise ValueError(
            f"series={held_out_series!r}, q_percent={observed_q_percent!r}에 해당하는 "
            "prefix_features 행이 없다"
        )
    held_out_q100_row = next(
        (row for row in all_prefixes if row["series"] == held_out_series and row["q_percent"] == 100), None,
    )
    if held_out_q100_row is None:
        raise ValueError(f"series={held_out_series!r}에 q_percent=100 행이 없다 — ground truth를 만들 수 없다")

    historical_pool = [row for row in all_prefixes if row["series"] != held_out_series]  # leave-series-out
    current_features = extract_historical_features(held_out_row)

    start = time.perf_counter()
    matches, _standardizer = find_similar_historical_prefixes(current_features, historical_pool, config=config)

    # 후보 수와 무관하게 쿼리 1회로 끝나는 batch loader를 쓴다 (`db.py` docstring 참고) —
    # 후보별로 커넥션을 새로 여는 방식은 후보가 많아질수록(예: 3,000개) 선형으로 느려진다.
    results_by_candidate = (
        _results_by_candidate if _results_by_candidate is not None
        else db.load_all_results_grouped_by_candidate(database)
    )
    candidate_definitions = (
        _candidate_definitions if _candidate_definitions is not None else db.load_candidate_definitions(database)
    )
    channel_count_by_prefix = (
        _channel_count_by_prefix if _channel_count_by_prefix is not None
        else db.load_channel_count_by_prefix(database)
    )

    # feasibility에 쓸 held-out series 고유값 두 가지: 채널 수(q와 무관, series 속성)와
    # "실제 q=100 시점에 이 series를 평가할 때 쓰인 연속 테스트 구간 길이"(test_observations,
    # 실제 DB로 series 안에서 후보와 무관하게 항상 동일함을 확인함 — 모듈 docstring 참고).
    channel_count = channel_count_by_prefix.get(held_out_row["prefix_feature_id"])
    held_out_test_observations = next(
        (
            row.get("test_observations") for rows in results_by_candidate.values() for row in rows
            if row["series"] == held_out_series and row["q_percent"] == 100 and row.get("test_observations") is not None
        ),
        None,
    )
    feasibility_evaluated = channel_count is not None and held_out_test_observations is not None

    ground_truth: dict[str, float] = {}
    predicted: dict[str, float] = {}
    estimated_cost: dict[str, float] = {}
    actual_cost_of_held_out: dict[str, float] = {}  # Method 1/2: held-out series 자신의 실측 비용(초)
    feasible_candidates: set[str] = set()
    evaluated_for_feasibility: set[str] = set()
    for (config_id, head), results_rows in results_by_candidate.items():
        candidate_id = _candidate_id(config_id, head)
        definition = candidate_definitions.get(config_id)

        held_out_points = get_series_trajectory(results_rows, held_out_series)
        gt_point = trajectory_point_at_q(held_out_points, 100)
        if gt_point is not None and gt_point.vus_pr is not None:
            ground_truth[candidate_id] = gt_point.vus_pr

        # Method 1/2: held-out series 자신의 q=100 실측 실행시간(actual_runtime_seconds) 평균.
        # 추정치(cost.py)가 아니라 실측이므로 "정답" 쪽에만 쓴다 — 관측이 없으면 채우지 않는다.
        actual_runtimes = [
            row["actual_runtime_seconds"] for row in results_rows
            if row["series"] == held_out_series and row["q_percent"] == 100
            and row.get("status") == "complete" and row.get("actual_runtime_seconds") is not None
        ]
        if actual_runtimes:
            actual_cost_of_held_out[candidate_id] = sum(actual_runtimes) / len(actual_runtimes)

        estimate = estimate_stage_performance(
            results_rows, matches,
            future_rows=held_out_q100_row["observed_row"], current_rows=held_out_row["observed_row"],
        )
        if estimate.predicted_performance is not None:
            predicted[candidate_id] = estimate.predicted_performance

        # cost tie-break: held-out series의 실제 미래(q=100) 규모를 그대로 미래
        # 학습/추론 규모로 쓴다 (임의의 operating_conditions를 지어내지 않는다 —
        # 모듈 docstring의 "이 harness가 반영하는 것" 참고). 회귀 자체는 leave-series-out
        # 원칙에 따라 held-out series 자신의 실행 기록을 제외한 데이터로만 만든다 —
        # similarity/performance는 `matches`(historical_pool 기반)로 이미 이렇게 되어
        # 있었지만, cost는 `results_rows`(전체 series 포함)를 그대로 썼던 데이터
        # 누수였다 (2026-09-25 리뷰에서 발견, 수정).
        needs_training = (definition or {}).get("target_use", "") == TRAINING_REQUIRED_TARGET_USE
        cost_training_rows = [row for row in results_rows if row["series"] != held_out_series]
        cost_model = build_candidate_cost_model(cost_training_rows, config=config)
        _training_cost, _inference_cost, total_cost, _source, _exclusion = estimate_stage_cost(
            cost_model,
            training_rows=held_out_q100_row["observed_row"] if needs_training else None,
            inference_volume=held_out_q100_row["observed_row"],
            needs_training=needs_training,
        )
        if total_cost is not None:
            estimated_cost[candidate_id] = total_cost

        if feasibility_evaluated and definition is not None:
            evaluated_for_feasibility.add(candidate_id)
            feasibility = assess_future_feasibility(
                definition, stage_training_rows=held_out_q100_row["observed_row"],
                stage_inference_rows=held_out_test_observations, channel_count=channel_count,
            )
            if feasibility["status"] == "feasible":
                feasible_candidates.add(candidate_id)
    ml_runtime_seconds = time.perf_counter() - start

    # candidate_selection.py의 _is_usable과 동일한 기준: feasible(또는 미평가) + performance +
    # cost가 모두 있어야 top-N 경쟁에 들어간다. feasibility를 계산하지 못했으면(모듈 전체 또는
    # 이 candidate의 정의 부재) "모른다"를 "탈락"으로 처리하지 않고 그대로 통과시킨다.
    def _is_usable(candidate_id: str) -> bool:
        if candidate_id not in predicted or candidate_id not in estimated_cost:
            return False
        if candidate_id in evaluated_for_feasibility:
            return candidate_id in feasible_candidates
        return True

    ground_truth_ranked = sorted(ground_truth, key=lambda cid: ground_truth[cid], reverse=True)
    usable = [cid for cid in predicted if _is_usable(cid)]
    # candidate_selection.py와 동일한 정렬: performance desc, cost asc, candidate_id로 결정론적 tie-break.
    predicted_ranked = sorted(usable, key=lambda cid: (-predicted[cid], estimated_cost.get(cid, float("inf")), cid))
    ground_truth_top_n = tuple(ground_truth_ranked[:top_n])
    predicted_top_n = tuple(predicted_ranked[:top_n])

    recall_at_n = (
        len(set(predicted_top_n) & set(ground_truth_top_n)) / len(ground_truth_top_n)
        if ground_truth_top_n else None
    )

    # Method 1: 정답도 예측과 동일 정책("성능desc, 비용asc")으로 재정렬 — 단, 비용은
    # cost.py 추정치가 아니라 held-out series 자신의 실측(actual_cost_of_held_out)을 쓴다.
    # 실측 비용이 없는 candidate는 이 정답 집합에 넣지 않는다(억지로 채우지 않는다).
    policy_candidates = [cid for cid in ground_truth if cid in actual_cost_of_held_out]
    ground_truth_policy_ranked = sorted(
        policy_candidates, key=lambda cid: (-ground_truth[cid], actual_cost_of_held_out[cid], cid),
    )
    ground_truth_policy_top_n = tuple(ground_truth_policy_ranked[:top_n])
    recall_policy_at_n = (
        len(set(predicted_top_n) & set(ground_truth_policy_top_n)) / len(ground_truth_policy_top_n)
        if ground_truth_policy_top_n else None
    )

    # Method 2: "성능만 봤을 때 골랐을 top-N"(ground_truth_top_n) vs "ML이 실제로 고른
    # top-N"(predicted_top_n)이 held-out series에서 실제로 얼마나 비용이 들었는지 비교.
    def _mean_actual_cost(candidate_ids: tuple[str, ...]) -> float | None:
        costs = [actual_cost_of_held_out[cid] for cid in candidate_ids if cid in actual_cost_of_held_out]
        return sum(costs) / len(costs) if costs else None

    actual_mean_cost_of_predicted_top_n = _mean_actual_cost(predicted_top_n)
    actual_mean_cost_of_performance_only_ground_truth_top_n = _mean_actual_cost(ground_truth_top_n)

    total_candidates = len(results_by_candidate)
    top_n_costs = [estimated_cost[cid] for cid in predicted_top_n if cid in estimated_cost]
    return HoldoutEvaluationResult(
        held_out_series=held_out_series, observed_q_percent=observed_q_percent, top_n=top_n,
        total_candidates=total_candidates, good_candidate_definition=GOOD_CANDIDATE_DEFINITION,
        ground_truth_top_n=ground_truth_top_n, predicted_top_n=predicted_top_n, recall_at_n=recall_at_n,
        candidate_reduction_ratio=1 - (top_n / total_candidates) if total_candidates else 0.0,
        ml_runtime_seconds=ml_runtime_seconds, similarity_matches_found=len(matches),
        candidates_with_ground_truth=len(ground_truth), candidates_with_prediction=len(predicted),
        candidates_with_cost_estimate=len(estimated_cost),
        mean_estimated_total_cost_seconds_of_top_n=(
            sum(top_n_costs) / len(top_n_costs) if top_n_costs else None
        ),
        feasibility_evaluated=feasibility_evaluated,
        stage_inference_rows_used=held_out_test_observations,
        candidates_feasible=len(feasible_candidates),
        candidates_usable=len(usable),
        ground_truth_policy_top_n=ground_truth_policy_top_n,
        recall_policy_at_n=recall_policy_at_n,
        actual_mean_cost_seconds_of_predicted_top_n=actual_mean_cost_of_predicted_top_n,
        actual_mean_cost_seconds_of_performance_only_ground_truth_top_n=(
            actual_mean_cost_of_performance_only_ground_truth_top_n
        ),
    )


def sweep_n_sensitivity(
    held_out_series: str, observed_q_percent: int, n_values: list[int], *, database=None, config: MLConfig = DEFAULT_CONFIG,
) -> list[HoldoutEvaluationResult]:
    """같은 held-out series/관측 시점에서 N만 바꿔가며 Recall@N/reduction ratio를 비교한다.

    N은 실제 후보 풀 크기(`total_candidates`)를 넘을 수 있다 — 이 경우 top_n이 곧
    전체 후보와 같아지므로(reduction_ratio<=0) 결과에서 그대로 드러난다(임의로
    N을 자르지 않는다).

    DB는 N값 개수와 무관하게 한 번만 읽는다 (`_all_prefixes`/`_results_by_candidate`
    캐시를 모든 N에 재사용) — 후보 규모가 커져도(예: 3,000개) N을 여러 개 sweep할 때
    DB 재조회 비용이 N배로 늘지 않는다.
    """
    all_prefixes = db.load_historical_prefixes(database)
    results_by_candidate = db.load_all_results_grouped_by_candidate(database)
    candidate_definitions = db.load_candidate_definitions(database)
    channel_count_by_prefix = db.load_channel_count_by_prefix(database)
    return [
        evaluate_holdout(
            held_out_series, observed_q_percent, database=database, config=config, top_n=n,
            _all_prefixes=all_prefixes, _results_by_candidate=results_by_candidate,
            _candidate_definitions=candidate_definitions, _channel_count_by_prefix=channel_count_by_prefix,
        )
        for n in n_values
    ]


def sweep_k_sensitivity(
    held_out_series: str, observed_q_percent: int, k_values: list[int], *,
    database=None, top_n: int | None = None, base_config: MLConfig = DEFAULT_CONFIG,
) -> list[HoldoutEvaluationResult]:
    """`similarity_knn_k`만 바꿔가며 같은 held-out series/관측 시점을 재평가한다.

    N-sensitivity와 같은 원리(다른 축은 전부 고정)로, k만 바뀐 `MLConfig`를
    새로 만들어 `evaluate_holdout`을 그대로 재사용한다 — 예측/랭킹 로직을
    이 harness가 따로 구현하지 않는다.
    """
    all_prefixes = db.load_historical_prefixes(database)
    results_by_candidate = db.load_all_results_grouped_by_candidate(database)
    candidate_definitions = db.load_candidate_definitions(database)
    channel_count_by_prefix = db.load_channel_count_by_prefix(database)
    results = []
    for k in k_values:
        config = replace(base_config, similarity_knn_k=k)
        results.append(evaluate_holdout(
            held_out_series, observed_q_percent, database=database, config=config, top_n=top_n,
            _all_prefixes=all_prefixes, _results_by_candidate=results_by_candidate,
            _candidate_definitions=candidate_definitions, _channel_count_by_prefix=channel_count_by_prefix,
        ))
    return results


def compare_similarity_metrics(
    held_out_series: str, observed_q_percent: int, *,
    database=None, top_n: int | None = None, base_config: MLConfig = DEFAULT_CONFIG,
) -> dict[str, HoldoutEvaluationResult]:
    """Euclidean/Cosine 두 `similarity_metric`으로 같은 held-out 시나리오를 나란히 평가한다.

    V1 기본값(Euclidean)을 바꾸지 않는다는 원칙은 그대로 유지한다 — 이 함수는
    "어느 쪽이 더 낫다"를 이 harness가 판단해 기본값을 바꾸는 게 아니라, 비교
    결과를 나란히 반환할 뿐이다 (기본값 변경은 문서 근거 없이는 하지 않는다).
    """
    all_prefixes = db.load_historical_prefixes(database)
    results_by_candidate = db.load_all_results_grouped_by_candidate(database)
    candidate_definitions = db.load_candidate_definitions(database)
    channel_count_by_prefix = db.load_channel_count_by_prefix(database)
    return {
        metric: evaluate_holdout(
            held_out_series, observed_q_percent, database=database,
            config=replace(base_config, similarity_metric=metric), top_n=top_n,
            _all_prefixes=all_prefixes, _results_by_candidate=results_by_candidate,
            _candidate_definitions=candidate_definitions, _channel_count_by_prefix=channel_count_by_prefix,
        )
        for metric in SUPPORTED_SIMILARITY_METRICS
    }
