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

## ML/DP 책임 경계에 대한 명시적 결정 (2026-09-23 검토)

루트 `README.md` "경로 최적화" 절 원문: "DP의 상태는 `단계·모델·설정·head·마지막
학습 단계`로 구성한다. ... **예상 성능 하한을 만족하면서 전체 학습·추론비가
가장 작은 경로**를 고른다." — 성능 하한(performance_floor) 언급은 이 DP 절에만
있고, 이 파일이 구현하는 "유사도와 성능·비용 추정" 절에는 없다. 따라서:

- **`performance_floor`/`performance_metric`은 ML 단계에서 후보를 사전 제외하는
  데 쓰지 않는다.** ML은 `predicted_performance`/`estimated_total_cost_seconds`를
  계산해 후보로 전달만 하고, 하한 적용은 DP의 책임으로 남긴다 (README 근거).
  이 결정에 따라 `_build_candidate_estimate()`/`select_stage_candidates()`는
  `operating_conditions`의 `performance_floor`를 의도적으로 참조하지 않는다.
- **`performance_metric`(VUS-PR/Precision/Recall/F1)은 현재 ML 데이터 계층에서
  선택할 수 없다.** `recommendation.sqlite3`의 `results` 테이블에는 `vus_pr`
  컬럼만 있고 precision/recall/f1 컬럼이 없다 (실제 DB 스키마 확인). 즉 이건
  설계 판단이 아니라 **DB에 그 데이터가 아예 없다는 사실**이다. UI가
  `performance_metric` 선택지를 주더라도 ML은 항상 VUS-PR만 쓸 수 있으며, 이
  불일치는 `run_ml_pipeline()`이 `metadata["warnings"]`에 명시적으로 남긴다
  (침묵 처리하지 않는다).
- **stage별 top-k 정렬 공식(`candidate_selection.py`의 "성능 내림차순 → 비용
  오름차순" lexicographic 방식)은 요구사항 문서 어디에도 근거가 없는 임의
  정책이다.** README은 "단계별 후보 수는 k=10처럼 정하되, 후보 구성은
  단계마다 달라진다"까지만 말한다. Pareto frontier·비용 우선·floor 이상 중
  비용 최소 등 다른 정책도 동등하게 정당화 가능하며, 문서로 확정할 수 없다.
  다만 **top-k에서 밀려난 후보도 `stage_candidates_excluded_from_top_k`에
  전부 보존되므로, 이 정렬 순서가 DP의 후보 접근 자체를 막지는 않는다** —
  DP가 이 정렬을 신뢰하지 않는다면 두 리스트를 합쳐 직접 재정렬할 수 있다.
  `run_ml_pipeline()`은 이 정책을 `metadata["top_k_ranking_policy"]`에
  명시적으로 "provisional/arbitrary"로 표시한다.
- **feasibility의 `test_length`에 넣는 값(`_steady_state_inference_rows()`)의
  의미론은 아직 검증되지 않았다.** 저장소 기존 계약(`check_dev18_resources.py`,
  `select_conditional_policy.py`의 `assess_candidate()` 호출부)에서 `test_length`는
  "한 번에 평가되는 연속 구간의 길이"(예: `row_count - training_boundary`)를
  뜻하며, "누적 추론 총량"이 아니다. 이 파이프라인이 쓰는 `inference_rows_per_day`
  (일 단위 처리율)도, 후보였던 `stage.inference_volume`(그 stage 구간 누적량)도
  원래 계약과 정확히 일치하지 않는다 — 둘 다 "연속 구간 길이"라는 축의 값이
  아니다. 운영 조건 입력에 "한 번에 들어오는 배치/스트림 길이"에 해당하는 값이
  없으므로 **현재로서는 새 입력을 추가하거나 기존 입력을 재해석해야 하며,
  문서만으로 최종 확정할 수 없다.** `inference_rows_per_day`를 유지하는 이유는
  "0이 아닌 값이라는 점에서 그나마 `stage.inference_volume`(누적, 무한정
  증가)보다는 왜곡이 작다"는 상대적 판단일 뿐, 검증된 결론이 아니다.
  `run_ml_pipeline()`은 이 사실을 `metadata["warnings"]`에 남긴다.
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

# recommendation.sqlite3의 results.vus_pr만 historical performance로 쓸 수 있다
# (다른 컬럼 없음 — DB 스키마 확인 완료). performance_metric이 이 값이 아니면
# ML이 실제로 쓰는 지표와 UI가 요청한 지표가 다르다는 뜻이므로 경고를 남긴다.
SUPPORTED_PERFORMANCE_METRIC = "VUS-PR"


def _scope_boundary_warnings(operating_conditions: dict) -> list[str]:
    """ML/DP 책임 경계 관련, 모듈 docstring에 근거를 남긴 결정들을 매 실행마다
    metadata에 드러낸다 (침묵 처리하지 않는다). 이 함수는 필터링을 하지 않는다
    — 경고만 만든다."""
    warnings: list[str] = []
    metric = operating_conditions.get("performance_metric")
    if metric and metric != SUPPORTED_PERFORMANCE_METRIC:
        warnings.append(
            f"performance_metric='{metric}'이 선택됐지만 recommendation DB는 vus_pr만 "
            f"저장한다 — ML의 predicted_performance는 항상 VUS-PR 기준이다 (요청 지표 무시)."
        )
    if operating_conditions.get("performance_floor") is not None:
        warnings.append(
            f"performance_floor={operating_conditions['performance_floor']!r}을 받았지만 "
            "ML 단계는 이 값으로 후보를 제외하지 않는다 (README상 DP의 경로 선택 제약 — "
            "'예상 성능 하한을 만족하면서 비용이 가장 작은 경로'). DP가 적용해야 한다."
        )
    return warnings


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
    metadata["warnings"].extend(_scope_boundary_warnings(ml_input.operating_conditions))
    # 문서로 확정할 수 없는 두 가지 설계 결정을 매 실행마다 명시적으로 남긴다
    # (침묵 처리하지 않는다 — 모듈 docstring의 "ML/DP 책임 경계" 절 참고).
    metadata["top_k_ranking_policy"] = (
        "provisional_arbitrary: predicted_performance desc, then estimated_total_cost_seconds asc "
        "(candidate_selection.py). 요구사항에 근거 없음 — 탈락 후보는 "
        "stage_candidates_excluded_from_top_k에 전부 보존됨."
    )
    metadata["warnings"].append(
        "feasibility의 test_length(_steady_state_inference_rows)는 inference_rows_per_day를 쓰는데, "
        "기존 model_feasibility.assess_candidate() 계약의 test_length는 '연속 평가 구간 길이'를 뜻하고 "
        "'누적 추론량'이 아니다 — 의미론이 완전히 검증되지 않은 상태다."
    )

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
