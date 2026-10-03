"""ML 단계의 단일 진입점: `run_ml_pipeline(ml_input, db_path=...)`.

```
MLInput (build_ml_input()의 산출물)
   |
   v
현재 prefix feature 정리 (features.py, compute_prefix_features 재사용)
   |
   v
similarity: 과거 prefix 검색 (similarity.py) ---- DB: prefix_features
   |
   v
미래 단계를 DB 등록 비율 경계로 나누기 (이 파일: build_future_stages)
   |
   v
후보별: historical trajectory 연결 (trajectory.py) ---- DB: results/cost_executions
        -> 단계별 성능 추정 (performance.py)
        -> 학습/추론 비용 추정 (cost.py)
        -> 단계별 feasibility 재판정 (feasibility.py, assess_candidate 재사용)
   |
   v
실행 가능·추정 가능 후보 분리 + checkpoint 유지 후보 (candidate_selection.py)
   |
   v
MLOutput (dp_input.build_dp_input이 DP 입력 계약으로 옮긴다)
```

이 모듈은 경로 최적화(DP)를 하지 않는다. 구간마다 실행 가능 여부, 예측 VUS-PR,
학습·추론 비용을 만들고 실행할 수 없거나 추정하지 못한 후보만 사유와 함께 뺀다.

## ML과 DP의 역할

- DP는 예산 안에서 구간 길이로 가중한 예측 VUS-PR을 최대화한다. 새 현장의 VUS-PR
  절대값은 0.3 정도 틀리지만 대부분 모든 후보에 같이 얹히는 오차라 계획 선택에는 거의
  영향이 없다(Dev18 held-out에서 예측 전체에 +0.3을 더해도 계획이 98% 같았다). 그래서
  ML은 절대 성능 하한으로 후보를 거르지 않는다.
- 예측 성능 순위로 후보 수를 자르지 않는다. 빠듯한 예산에서는 싼 후보가 답이 된다.
- feasibility의 `test_length`는 저장소 계약대로 "한 번에 평가하는 연속 구간 길이"다.
  `operating_conditions`에 `inference_batch_length`가 있으면 그 값을 쓰고, 없으면
  `inference_rows_per_day`로 근사하며 매 실행 `metadata["warnings"]`에 남긴다.
  구간 누적 추론량은 스트림 길이가 아니므로 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from streamlit_website.ml import db
from streamlit_website.ml.candidate_selection import select_stage_candidates
from streamlit_website.ml.config import DEFAULT_CONFIG, MLConfig
from streamlit_website.ml.cost import build_candidate_cost_model, estimate_stage_cost
from streamlit_website.ml.feasibility import assess_future_feasibility
from streamlit_website.ml.features import extract_similarity_features
from streamlit_website.ml.performance import estimate_checkpoint_maintenance_performance, estimate_stage_performance
from streamlit_website.ml.schemas import (
    CandidateEstimate,
    CheckpointMaintenanceOption,
    FutureStagePlan,
    MLInput,
    MLOutput,
)
from streamlit_website.ml.similarity import find_similar_historical_prefixes

# fit_full_prefix 후보만 "재학습"이 필요하다. training_free/strict_zero_shot은
# 데이터가 늘어나도 재학습 비용이 없다 (구조적으로 학습 자체가 없음).
TRAINING_REQUIRED_TARGET_USE = "fit_full_prefix"

def planned_final_rows(row_count: int, operating_conditions: dict) -> int:
    """운영 기간 끝까지 모을 정상 행 수. 이 값을 DB 등록 비율의 100%로 본다."""
    days = max(operating_conditions.get("operating_days") or 0, 0)
    return row_count + round((operating_conditions.get("collection_rows_per_second") or 0.0) * days * 86400)


def ratio_rows(q: int, final_rows: int) -> int:
    """DB와 같은 규칙: q% prefix는 floor(final_rows × q / 100)행이다."""
    return q * final_rows // 100


def ratio_percent(rows: int, final_rows: int) -> int:
    """final_rows를 100으로 볼 때 rows가 닿은 가장 큰 DB 등록 비율. 5% 미만도 5로 둔다."""
    return max((q for q in SUPPORTED_RATIO_PERCENTS if ratio_rows(q, final_rows) <= rows),
               default=SUPPORTED_RATIO_PERCENTS[0])


def build_future_stages(row_count: int, operating_conditions: dict) -> list[FutureStagePlan]:
    """남은 운영 기간을 데이터가 다음 DB 등록 비율(q)에 닿는 날로 나눈다.

    DB는 5·10·20·40·60·80·100% 비율에서만 성능을 쟀다. 운영 기간 끝까지 모을 행 수를
    100%로 보고 이 비율 경계에서 구간을 나누면, 구간 안에서는 예측이 바뀌지 않고 같은
    비율에서 두 번 학습하는 계획도 나오지 않는다. 구간이 시작할 때 확보한 행 수로
    학습하고 `inference_rows_per_day × 구간 일수`만큼 추론한다. `operating_days`가
    없거나 0 이하이면 추론량 0인 stage 0만 반환한다.
    """
    if row_count <= 0:
        raise ValueError("row_count는 1 이상이어야 한다")
    operating_days = operating_conditions.get("operating_days")
    if operating_days is None or operating_days <= 0:
        return [FutureStagePlan(stage=0, elapsed_days=0.0, future_rows=row_count, inference_volume=0.0)]
    rows_per_day = (operating_conditions.get("collection_rows_per_second") or 0.0) * 86400
    inference_per_day = operating_conditions.get("inference_rows_per_day") or 0.0
    final_rows = planned_final_rows(row_count, operating_conditions)
    now = ratio_percent(row_count, final_rows)
    later = [(q, ratio_rows(q, final_rows)) for q in SUPPORTED_RATIO_PERCENTS if now < q < 100]
    starts = [0.0] + [(rows - row_count) / rows_per_day for _, rows in later]
    return [
        FutureStagePlan(stage=index, elapsed_days=start, future_rows=rows,
                        inference_volume=inference_per_day * (end - start), ratio_percent=q)
        for index, (start, end, (q, rows)) in enumerate(zip(starts, starts[1:] + [operating_days],
                                                            [(now, row_count), *later]))
    ]


def _candidate_id(config_id: str, head: str) -> str:
    return f"{config_id}::{head}"


def _steady_state_inference_rows(operating_conditions: dict) -> tuple[int, bool]:
    """feasibility 판정에 쓸 "한 번에 흐르는 추론 스트림 길이"와, 그 값이

    `inference_batch_length`(명시적 입력, 있으면 최우선)에서 왔는지 아니면
    `inference_rows_per_day`(기존 proxy, 검증 안 됨)에서 왔는지를 함께 반환한다.

    `inference_batch_length`는 "한 번에 모델에 입력되는 연속 데이터 길이"를
    직접 받는 값이다 — 기존 `model_feasibility.assess_candidate()`의
    `test_length` 계약("연속 평가 구간 길이", `check_dev18_resources.py`에서
    `row_count - training_boundary`로 쓰이는 것과 같은 축)과 의미가 정확히
    일치한다. 이 값이 있으면 더 이상 proxy를 쓸 이유가 없으므로 그대로 쓴다.

    없으면 기존 `inference_rows_per_day`(일 단위 처리율) proxy로 fallback한다 —
    stage.inference_volume(그 구간 누적량)을 안 쓰는 이유는 여전히 같다: 구조적
    최소 길이는 "누적 추론량"이 아니라 "한 번에 들어오는 스트림 길이"로
    결정된다. 다만 이 proxy는 여전히 검증되지 않은
    근사치이므로, 두 번째 반환값(`used_unvalidated_proxy`)으로 어느 쪽이
    쓰였는지 호출자가 알 수 있게 한다.
    """
    batch_length = operating_conditions.get("inference_batch_length")
    if batch_length:
        return int(batch_length), False
    value = operating_conditions.get("inference_rows_per_day")
    return (int(value) if value else 0), True


def _build_candidate_estimate(
    candidate: dict, stage: FutureStagePlan, *, channel_count: int,
    operating_conditions: dict, matches, results_rows,
) -> CandidateEstimate:
    config_id, head, model = candidate["config_id"], candidate["head"], candidate["model"]
    target_use = candidate.get("target_use", "")
    candidate_id = _candidate_id(config_id, head)

    stage_inference_rows, _used_unvalidated_proxy = _steady_state_inference_rows(operating_conditions)
    feasibility = assess_future_feasibility(
        candidate, stage_training_rows=stage.future_rows,
        stage_inference_rows=stage_inference_rows,
        channel_count=channel_count,
    )
    feasible = feasibility["status"] == "feasible"

    performance = estimate_stage_performance(results_rows, matches, ratio_percent=stage.ratio_percent)
    cost_model = build_candidate_cost_model(results_rows, model=model)
    needs_training = target_use == TRAINING_REQUIRED_TARGET_USE
    training_cost, inference_cost, total_cost, cost_source, cost_exclusion = estimate_stage_cost(
        cost_model, training_rows=stage.future_rows if needs_training else None,
        inference_volume=stage.inference_volume, needs_training=needs_training, channel_count=channel_count,
    )

    exclusion_reason = None
    if not feasible:
        exclusion_reason = feasibility["reason"] or "structurally_infeasible"
    elif performance.predicted_performance is None:
        exclusion_reason = "no_historical_performance_estimate_for_candidate"
    elif cost_exclusion is not None:
        exclusion_reason = cost_exclusion

    return CandidateEstimate(
        stage=stage.stage, candidate_id=candidate_id, model=model, configuration=config_id, head=head,
        feasible=feasible,
        predicted_performance=performance.predicted_performance,
        performance_lower_bound=performance.performance_lower_bound,
        estimated_training_cost_seconds=training_cost,
        estimated_inference_cost_seconds=inference_cost,
        estimated_total_cost_seconds=total_cost,
        data_rows=stage.future_rows, inference_volume=stage.inference_volume,
        prediction_source=performance.prediction_source, cost_estimation_source=cost_source,
        exclusion_reason=exclusion_reason,
    )


def _find_current_model_candidates(current_model: dict, candidates: list[dict]) -> list[dict]:
    """화면에서 고른 현재 후보 ID(`config_id::head`)를 후보 풀에서 찾는다."""
    wanted = current_model.get("candidate_id")
    return [c for c in candidates if wanted and _candidate_id(c["config_id"], c["head"]) == wanted]


def _build_checkpoint_maintenance_options(
    ml_input: MLInput, future_stages: list[FutureStagePlan], matches, *, results_by_candidate,
) -> list[CheckpointMaintenanceOption]:
    matched_candidates = _find_current_model_candidates(ml_input.current_model, ml_input.candidates)
    if not matched_candidates:
        return []
    checkpoint_reference = ml_input.current_model.get("checkpoint") or None
    options = []
    for candidate in matched_candidates:
        config_id, head, model = candidate["config_id"], candidate["head"], candidate["model"]
        results_rows = results_by_candidate.get((config_id, head), [])
        trained_rows = ml_input.current_model.get("trained_rows") or ml_input.row_count
        performance = estimate_checkpoint_maintenance_performance(
            results_rows, matches, ratio_percent=ratio_percent(
                trained_rows, planned_final_rows(ml_input.row_count, ml_input.operating_conditions)),
        )
        cost_model = build_candidate_cost_model(results_rows, model=model)
        # 유지는 다시 학습하지 않으므로 학습 길이 구조 제약으로 막지 않는다.
        for stage in future_stages:
            _training_cost, inference_cost, total_cost, cost_source, cost_exclusion = estimate_stage_cost(
                cost_model, training_rows=None, inference_volume=stage.inference_volume, needs_training=False,
                channel_count=ml_input.channel_count,
            )
            exclusion_reason = None
            if performance.predicted_performance is None:
                exclusion_reason = "no_historical_performance_estimate_for_candidate"
            elif cost_exclusion is not None:
                exclusion_reason = cost_exclusion
            options.append(CheckpointMaintenanceOption(
                stage=stage.stage, candidate_id=_candidate_id(config_id, head), model=model,
                configuration=config_id, head=head, checkpoint_reference=checkpoint_reference,
                feasible=True,
                predicted_performance=performance.predicted_performance,
                performance_lower_bound=performance.performance_lower_bound,
                estimated_inference_cost_seconds=inference_cost,
                estimated_total_cost_seconds=total_cost,
                data_rows=stage.future_rows, inference_volume=stage.inference_volume,
                prediction_source="checkpoint_maintained", cost_estimation_source=cost_source,
                exclusion_reason=exclusion_reason,
            ))
    return options


def run_ml_pipeline(
    ml_input: dict | MLInput, *, db_path: str | Path | None = None, config: MLConfig = DEFAULT_CONFIG,
) -> MLOutput:
    """DB/Streamlit → ML → DP 사이의 단일 진입점.

    `ml_input`은 `build_ml_input()`의 반환값(dict)이거나 이미 `MLInput`으로 감싼
    것이어도 된다. `db_path`는 `recommendation.sqlite3` 경로 — 생략하면
    `db.resolve_database_path()`가 인자/환경변수/기본 경로 순으로 정한다.
    """
    if isinstance(ml_input, dict):
        ml_input = MLInput.from_ml_input_dict(ml_input)

    metadata: dict = {"config": config, "warnings": []}
    _stage_inference_rows, used_unvalidated_test_length_proxy = _steady_state_inference_rows(
        ml_input.operating_conditions,
    )
    if used_unvalidated_test_length_proxy:
        metadata["warnings"].append(
            "연속 추론 길이를 넣지 않아 실행 조건의 test_length를 하루 추론량으로 대신 판정했습니다. "
            "한 번에 모델에 넣는 데이터 길이를 넣으면 이 근사를 쓰지 않습니다."
        )

    summary = (ml_input.current_features or {}).get("summary")
    if summary is None:
        metadata["warnings"].append("현재 데이터의 feature가 없어 구간별 성능·비용을 추정하지 못했습니다.")
        return MLOutput(
            current_features=None, current_feature_reasons=None, similarity_matches=[], future_stages=[],
            stage_candidates={}, stage_candidates_excluded={},
            checkpoint_maintenance_options=[], metadata=metadata,
        )

    current_features, current_feature_reasons = extract_similarity_features(summary)
    historical_prefixes = db.load_historical_prefixes(db_path)
    matches, standardizer = find_similar_historical_prefixes(current_features, historical_prefixes, config=config)
    metadata["standardization"] = {
        "zero_variance_columns": sorted(standardizer.zero_variance_columns),
        "insufficient_data_columns": sorted(standardizer.insufficient_data_columns),
    }
    # "저유사도 warning"을 임의의 숫자 threshold로 확정하지 않는다 (요구사항에 근거
    # 없음) — 대신 실제 관측된 거리 분포를 그대로 노출해서, 사람이나 downstream
    # 소비자가 스스로 판단할 수 있게 한다. distance는 클수록(similarity_metric에
    # 따라 스케일이 다름) 덜 유사하다는 것만 공통이다.
    metadata["similarity_summary"] = {
        "similarity_metric": config.similarity_metric,
        "requested_k": config.similarity_knn_k,
        "matches_found": len(matches),
        "min_distance": min((m.distance for m in matches), default=None),
        "max_distance": max((m.distance for m in matches), default=None),
        "mean_distance": (sum(m.distance for m in matches) / len(matches)) if matches else None,
    }
    if len(matches) < config.similarity_knn_k:
        metadata["warnings"].append(
            f"비슷한 과거 prefix를 {config.similarity_knn_k}개 찾으려 했지만 {len(matches)}개만 찾았습니다"
            f"(similarity kNN, k={config.similarity_knn_k}). 과거 prefix가 적거나 공통 feature가 없을 때 생기며, "
            "예측을 더 조심해서 읽어 주세요."
        )

    future_stages = build_future_stages(ml_input.row_count, ml_input.operating_conditions)
    if not (ml_input.operating_conditions.get("operating_days") or 0) > 0:
        metadata["warnings"].append("남은 운영 기간이 0일이라 지금 구간 하나만 계획했습니다.")

    results_by_candidate = db.load_all_results_grouped_by_candidate(db_path)
    stage_candidates: dict[int, list] = {}
    stage_excluded: dict[int, list] = {}
    for stage in future_stages:
        estimates = [
            _build_candidate_estimate(
                candidate, stage, channel_count=ml_input.channel_count,
                operating_conditions=ml_input.operating_conditions, matches=matches,
                results_rows=results_by_candidate.get((candidate["config_id"], candidate["head"]), []),
            )
            for candidate in ml_input.candidates
        ]
        stage_candidates[stage.stage], stage_excluded[stage.stage] = select_stage_candidates(estimates)

    checkpoint_options = _build_checkpoint_maintenance_options(
        ml_input, future_stages, matches, results_by_candidate=results_by_candidate,
    )
    if not checkpoint_options and ml_input.current_model.get("candidate_id"):
        metadata["warnings"].append("현재 모델이 이번 후보 풀에 없어 기존 checkpoint를 유지하는 선택지를 만들지 못했습니다.")

    return MLOutput(
        current_features=current_features, current_feature_reasons=current_feature_reasons,
        similarity_matches=matches, future_stages=future_stages,
        stage_candidates=stage_candidates, stage_candidates_excluded=stage_excluded,
        checkpoint_maintenance_options=checkpoint_options, metadata=metadata,
    )
