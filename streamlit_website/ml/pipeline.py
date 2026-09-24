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
미래 단계 규모 계산 (이 파일: build_future_stages)
   |
   v
후보별: historical trajectory 연결 (trajectory.py) ---- DB: results/cost_executions
        -> 단계별 성능 추정 (performance.py)
        -> 학습/추론 비용 추정 (cost.py)
        -> 단계별 feasibility 재판정 (feasibility.py, assess_candidate 재사용)
   |
   v
단계별 top-k 후보 축소 + checkpoint 유지 후보 분리 (candidate_selection.py)
   |
   v
MLOutput (DP가 이 스키마만 소비한다)
```

이 모듈은 경로 최적화(DP)를 하지 않는다. similarity/성능/비용/feasibility/단계별
축소까지만 한다.
"""

from __future__ import annotations

from pathlib import Path

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


def build_future_stages(
    row_count: int, operating_conditions: dict, *, config: MLConfig = DEFAULT_CONFIG,
) -> list[FutureStagePlan]:
    """운영 기간·수집 속도·추론량으로 미래 단계 grid를 만든다.

    `future_rows = current_rows + collection_rows_per_second * elapsed_seconds`
    (README 6장 공식). stage 0은 "지금"(elapsed_days=0, future_rows=row_count).
    각 미래 stage의 `inference_volume`은 "그 stage 구간(직전 stage 이후 경과 기간)
    동안의 예상 추론 행 수"이며, `inference_rows_per_day × 그 구간의 일수`로 계산한다
    — 운영 조건에 있는 `inference_rows_per_day`를 그대로 쓴다(별도 파생 안 함).
    `operating_days`가 없거나 0 이하이면 미래 단계를 만들 수 없으므로 stage 0만
    반환한다 (미래 계획 자체를 세울 근거가 없다는 뜻).
    """
    if row_count <= 0:
        raise ValueError("row_count는 1 이상이어야 한다")
    stages = [FutureStagePlan(stage=0, elapsed_days=0.0, future_rows=row_count, inference_volume=0.0)]
    operating_days = operating_conditions.get("operating_days")
    if operating_days is None or operating_days <= 0:
        return stages
    collection_rate = operating_conditions.get("collection_rows_per_second") or 0.0
    inference_per_day = operating_conditions.get("inference_rows_per_day") or 0.0
    previous_elapsed = 0.0
    for stage_index, fraction in enumerate(config.future_stage_elapsed_day_fractions, start=1):
        elapsed_days = operating_days * fraction / 100
        future_rows = row_count + int(collection_rate * elapsed_days * 86400)
        interval_days = elapsed_days - previous_elapsed
        inference_volume = inference_per_day * interval_days
        stages.append(FutureStagePlan(
            stage=stage_index, elapsed_days=elapsed_days, future_rows=future_rows,
            inference_volume=inference_volume,
        ))
        previous_elapsed = elapsed_days
    return stages


def _candidate_id(config_id: str, head: str) -> str:
    return f"{config_id}::{head}"


def _steady_state_inference_rows(operating_conditions: dict) -> int:
    """feasibility 판정에 쓸 "한 번에 흐르는 추론 스트림 길이".

    stage.inference_volume(그 구간 누적량, stage 0은 0)과는 다른 개념이다 —
    구조적 최소 길이(window/patch/context 등)는 "그 stage까지 누적된 추론량"이
    아니라 "실제로 한 번에 들어오는 스트림 길이"에 의해 결정되므로,
    `inference_rows_per_day`(일 단위 예상 추론량)를 모든 stage에 동일하게 쓴다.
    stage 0(지금)도 이 값을 쓴다 — 그래야 "지금 도입 가능한가"를 0행 기준으로
    항상 infeasible 처리하는 오류를 피할 수 있다.
    """
    value = operating_conditions.get("inference_rows_per_day")
    return int(value) if value else 0


def _build_candidate_estimate(
    candidate: dict, stage: FutureStagePlan, *, current_rows: int, channel_count: int,
    operating_conditions: dict, matches, database,
) -> CandidateEstimate:
    config_id, head, model = candidate["config_id"], candidate["head"], candidate["model"]
    target_use = candidate.get("target_use", "")
    candidate_id = _candidate_id(config_id, head)

    feasibility = assess_future_feasibility(
        candidate, stage_training_rows=stage.future_rows,
        stage_inference_rows=_steady_state_inference_rows(operating_conditions),
        channel_count=channel_count,
    )
    feasible = feasibility["status"] == "feasible"

    results_rows = db.load_results_for_candidate(config_id, head, database=database)
    performance = estimate_stage_performance(
        results_rows, matches, future_rows=stage.future_rows, current_rows=current_rows,
    )
    cost_model = build_candidate_cost_model(results_rows)
    needs_training = target_use == TRAINING_REQUIRED_TARGET_USE
    training_cost, inference_cost, total_cost, cost_source, cost_exclusion = estimate_stage_cost(
        cost_model, training_rows=stage.future_rows if needs_training else None,
        inference_volume=stage.inference_volume, needs_training=needs_training,
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
    """`ml_input.current_model`(자유 텍스트)을 후보 풀의 (config_id, head)와 매칭한다.

    Streamlit 폼은 현재 모델의 `model` 이름과 `settings`(=config_id로 기대)만
    받고 head는 받지 않는다 (Phase 1/2 조사에서 확인) — 그래서 head가 여러 개인
    config(TSPulse)는 여기서 모호할 수 있고, 그 경우 매칭되는 모든 head를 반환한다
    (하나로 임의로 좁히지 않는다).
    """
    model_name = (current_model.get("model") or "").strip()
    config_id = (current_model.get("settings") or "").strip()
    if not model_name or not config_id:
        return []
    return [c for c in candidates if c["model"] == model_name and c["config_id"] == config_id]


def _build_checkpoint_maintenance_options(
    ml_input: MLInput, future_stages: list[FutureStagePlan], matches, *, database,
) -> list[CheckpointMaintenanceOption]:
    matched_candidates = _find_current_model_candidates(ml_input.current_model, ml_input.candidates)
    if not matched_candidates:
        return []
    checkpoint_reference = ml_input.current_model.get("checkpoint") or None
    options = []
    for candidate in matched_candidates:
        config_id, head, model = candidate["config_id"], candidate["head"], candidate["model"]
        results_rows = db.load_results_for_candidate(config_id, head, database=database)
        # 마지막 학습 시점의 성능은 "현재 단계" similarity 추정으로 근사한다
        # (last_trained_at이 구조화된 stage가 아니라 자유 텍스트라 정확한 과거
        # stage를 가리킬 수 없다 — 알려진 한계, 보고서에 기록).
        performance = estimate_checkpoint_maintenance_performance(
            results_rows, matches, current_rows=ml_input.row_count,
        )
        cost_model = build_candidate_cost_model(results_rows)
        for stage in future_stages:
            feasibility = assess_future_feasibility(
                candidate, stage_training_rows=stage.future_rows,
                stage_inference_rows=_steady_state_inference_rows(ml_input.operating_conditions),
                channel_count=ml_input.channel_count,
            )
            feasible = feasibility["status"] == "feasible"
            _training_cost, inference_cost, total_cost, cost_source, cost_exclusion = estimate_stage_cost(
                cost_model, training_rows=None, inference_volume=stage.inference_volume, needs_training=False,
            )
            exclusion_reason = None
            if not feasible:
                exclusion_reason = feasibility["reason"] or "structurally_infeasible"
            elif performance.predicted_performance is None:
                exclusion_reason = "no_historical_performance_estimate_for_candidate"
            elif cost_exclusion is not None:
                exclusion_reason = cost_exclusion
            options.append(CheckpointMaintenanceOption(
                stage=stage.stage, candidate_id=_candidate_id(config_id, head), model=model,
                configuration=config_id, head=head, checkpoint_reference=checkpoint_reference,
                feasible=feasible,
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

    summary = (ml_input.current_features or {}).get("summary")
    if summary is None:
        metadata["warnings"].append("current_features.summary가 없다 — 현재 prefix feature 계산이 실패했거나 아직 없음")
        return MLOutput(
            current_features=None, current_feature_reasons=None, similarity_matches=[], future_stages=[],
            stage_candidates={}, stage_candidates_excluded_from_top_k={},
            checkpoint_maintenance_options=[], metadata=metadata,
        )

    current_features, current_feature_reasons = extract_similarity_features(summary)
    historical_prefixes = db.load_historical_prefixes(db_path)
    matches, standardizer = find_similar_historical_prefixes(current_features, historical_prefixes, config=config)
    metadata["standardization"] = {
        "zero_variance_columns": sorted(standardizer.zero_variance_columns),
        "insufficient_data_columns": sorted(standardizer.insufficient_data_columns),
    }

    future_stages = build_future_stages(ml_input.row_count, ml_input.operating_conditions, config=config)
    if len(future_stages) == 1:
        metadata["warnings"].append("operating_days가 없거나 0 이하라 미래 단계를 계획하지 못했다 (stage 0만 있음)")

    stage_candidates: dict[int, list] = {}
    stage_excluded: dict[int, list] = {}
    for stage in future_stages:
        estimates = [
            _build_candidate_estimate(
                candidate, stage, current_rows=ml_input.row_count, channel_count=ml_input.channel_count,
                operating_conditions=ml_input.operating_conditions, matches=matches, database=db_path,
            )
            for candidate in ml_input.candidates
        ]
        top_k, excluded = select_stage_candidates(estimates, config=config)
        stage_candidates[stage.stage] = top_k
        stage_excluded[stage.stage] = excluded

    checkpoint_options = _build_checkpoint_maintenance_options(ml_input, future_stages, matches, database=db_path)
    if not checkpoint_options and (ml_input.current_model.get("model") or "").strip():
        metadata["warnings"].append("current_model이 candidate_scope 후보 풀과 매칭되지 않아 checkpoint_maintenance_options가 비어있다")

    return MLOutput(
        current_features=current_features, current_feature_reasons=current_feature_reasons,
        similarity_matches=matches, future_stages=future_stages,
        stage_candidates=stage_candidates, stage_candidates_excluded_from_top_k=stage_excluded,
        checkpoint_maintenance_options=checkpoint_options, metadata=metadata,
    )
