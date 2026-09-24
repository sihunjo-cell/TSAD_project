# streamlit_website/ml — TSAD ML 단계

DB→Streamlit 연결 다음, DP(경로 최적화) 이전에 들어가는 ML 단계다. **DP는 이
모듈이 구현하지 않는다.** `run_ml_pipeline(ml_input, db_path=...) -> MLOutput`이
유일한 진입점이며, DP는 `MLOutput`만 소비하면 된다.

## 범위

1. 현재 prefix feature 계산/정리 (`features.py`, `compute_prefix_features.py` 재사용)
2. 과거 prefix similarity 검색 (`similarity.py`)
3. similarity 기반 historical trajectory 검색 (`trajectory.py`)
4. 단계별 성능 추정 (`performance.py`)
5. 학습/추론 비용 추정 (`cost.py`)
6. 단계별 feasibility 재판정 (`feasibility.py`, `model_feasibility.assess_candidate` 재사용)
7. 단계별 top-k 후보 축소 (`candidate_selection.py`)
8. checkpoint 유지 후보를 top-k와 별도로 보존
9. `MLOutput` 생성 (`schemas.py`, `pipeline.py`)

## 파일 구조

| 파일 | 책임 |
|---|---|
| `schemas.py` | `MLInput`/`MLOutput`과 중간 산출물 dataclass |
| `config.py` | k, feature 목록, 미래 단계 grid 등 조정 가능한 값 |
| `db.py` | `recommendation.sqlite3` 읽기 전용 접근 (계산 없음) |
| `features.py` | 현재/과거 feature 벡터 정리 (행 수 제외, NULL 보존) |
| `similarity.py` | population 표준화 + Euclidean + distance-weighted kNN |
| `trajectory.py` | series별 q trajectory, 성장 비율 → 등록 q 매핑, run_id 기반 물리 실행 재사용 추적 |
| `performance.py` | similarity + trajectory로 단계별 성능 추정 (provenance 포함) |
| `cost.py` | candidate별 독립 회귀로 학습/추론 비용(초) 추정 |
| `feasibility.py` | 미래 stage 행 수 기준 구조적 feasibility 재판정 (wrapper만) |
| `candidate_selection.py` | stage별 top-k 축소, 탈락 후보 보존 |
| `pipeline.py` | 위 전부를 묶는 `run_ml_pipeline()` |
| `test_ml_pipeline.py` | unit + 임시 DB + 실제 dev18 DB 통합 테스트 |

DB 경로는 인자 > 환경변수 `TSAD_RECOMMENDATION_DB` > 기본값
(`experiments/tuning/results/recommendation.sqlite3`, git에 커밋되지 않음) 순으로 정한다.

## Streamlit 연결

`streamlit_website/db_connection/view.py`의 `render_candidate_intake()`가
`build_ml_input()`으로 `st.session_state["ml_input"]`을 만든 다음, "ML 단계 실행"
버튼 하나로 `run_ml_pipeline(st.session_state["ml_input"])`을 호출해
`st.session_state["ml_output"]`에 저장한다. UI는 의도적으로 최소화했다 — 기존
1차 후보 축소 화면 구조는 바꾸지 않았다.

## postprocess_\*/l4_\* 테이블에 대한 노트 (existing DB evidence)

실제 DB에는 이 저장소 코드가 만들지 않는(= 다른 팀원이 별도로 구축한)
`postprocess_*`(예: `postprocess_model_best`, `postprocess_tier_best`,
`postprocess_candidate_scores` 등)와 `l4_*`(`l4_cost_inputs`,
`l4_candidate_costs`, `l4_tier_transition_costs`) 테이블/뷰가 있다. 이들은:

- **이 ML 파이프라인의 입력이나 정답으로 쓰지 않는다.** `postprocess_model_best`/
  `postprocess_tier_best`를 근거로 top-k를 직접 정하지 않는다 — `candidate_selection.py`의
  top-k는 이 모듈이 독립적으로 계산한 `predicted_performance`/`estimated_total_cost_seconds`만
  기준으로 삼는다.
- **참고용 대조 자료로만 취급한다.** 이 파이프라인이 계산한 성능/비용 추정치가
  `postprocess_*`의 사전 계산 결과와 방향이 일치하면 그 사실을 기록해 둘 가치가
  있지만(신뢰도의 방증), 이 저장소 코드에서 재생산할 수 없는 로직이므로 의존하지
  않는다.
- `l4_cost_inputs.l4_hourly_rate`는 이 DB에서 전부 NULL이라 금액(비용) 환산이
  불가능하다 — 그래서 `cost.py`는 초(seconds) 단위 비용만 만든다. 시간당 단가가
  채워지면 `CostCurve.predict()`가 반환하는 초 값에 단가를 곱하기만 하면 된다.

## ML/DP 책임 경계 — 검토 완료 (2026-09-23)

`pipeline.py` 모듈 docstring에 전문이 있다. 요약:

- **`performance_floor`는 DP 제약이다, ML이 적용하지 않는다.** 루트 `README.md`의
  "경로 최적화" 절에서만 "예상 성능 하한을 만족하면서 비용이 가장 작은 경로"라는
  표현이 나오고, ML 절에는 없다 — 텍스트로 확정 가능. `run_ml_pipeline()`은
  `performance_floor`를 받아도 후보를 거르지 않고, 매 실행마다
  `metadata["warnings"]`에 "받았지만 적용 안 함"을 명시적으로 남긴다.
- **`performance_metric`(VUS-PR 외 Precision/Recall/F1)은 ML 데이터 계층에서
  아예 지원 불가능하다.** `recommendation.sqlite3.results`에 `vus_pr` 외
  precision/recall/f1 컬럼이 없다 — 이건 설계 판단이 아니라 DB에 그 데이터가
  없다는 사실. VUS-PR이 아닌 값을 요청하면 경고를 남기고 무시한다(항상 VUS-PR
  기준으로 계산). UI(`view.py`)가 4개 선택지를 주는 것과 실제로 동작하는 범위
  사이에 gap이 있다는 뜻이므로, UI 쪽 수정이나 DB에 다른 지표 추가가 필요하면
  별도로 결정해야 한다(이 세션에서는 ML 코드만 다뤘다).

## 아직 해결하지 못한 가정/제약

- **top-k 정렬 공식**: "predicted_performance 내림차순 → cost 오름차순"은 문서
  근거가 없는 **명시적으로 provisional/arbitrary** 정책이다 (`candidate_selection.py`
  상단 docstring, `run_ml_pipeline()`의 `metadata["top_k_ranking_policy"]`).
  Pareto frontier·비용 우선·성능 하한 이상 중 비용 최소 등 다른 정책도 동등하게
  정당화 가능하지만, 문서만으로는 어느 것도 확정할 수 없어 바꾸지 않았다.
  단, top-k에서 밀린 후보도 `stage_candidates_excluded_from_top_k`에 전부
  보존되므로, 이 정렬이 DP의 후보 접근 자체를 막지는 않는다 — DP가 다른 정책을
  원하면 두 리스트를 합쳐 직접 재정렬하면 된다.
- **feasibility의 "추론 스트림 길이"(`test_length`) 의미론 미검증**: 저장소
  기존 계약(`check_dev18_resources.py`, `select_conditional_policy.py`)에서
  `test_length`는 "한 번에 평가되는 연속 구간 길이"(예: `row_count -
  training_boundary`)를 뜻하며 "누적 추론 총량"이 아니다. 이 파이프라인이 쓰는
  `inference_rows_per_day`(상수 처리율)도, 후보였던 `stage.inference_volume`
  (그 stage 구간 누적량, stage 0은 0)도 원래 의미와 정확히 일치하지 않는다 —
  둘 다 "연속 구간 길이" 축의 값이 아니다. 운영 조건 입력에 "한 번에 들어오는
  배치/스트림 길이"에 해당하는 값이 없어서 새 입력을 추가하거나 기존 입력을
  재해석해야 하는데, 문서만으로 확정할 수 없다. `inference_rows_per_day`를
  유지하는 건 "stage 0이 추론량 0 때문에 무조건 infeasible 되는 것보다,
  `stage.inference_volume`(무한정 누적 증가)보다 왜곡이 상대적으로 작다"는
  판단일 뿐 검증된 결론이 아니다 — 매 실행마다 `metadata["warnings"]`에 남긴다.
- **checkpoint maintenance 성능**: `last_trained_at`이 구조화된 stage가 아니라
  자유 텍스트라, "마지막 학습 시점의 성능"을 "현재 단계(stage 0) similarity 추정"으로
  근사한다 (`performance.py::estimate_checkpoint_maintenance_performance`). 최근에
  학습했을수록 정확하고 오래 전에 학습했을수록 낙관적일 수 있다.
- **checkpoint maintenance feasibility**: 재학습이 없는데도 매 stage마다 fresh
  training과 같은 fit/validation 구조 제약을 재사용해서 판정한다 (`pipeline.py::
  _build_checkpoint_maintenance_options`) — 실제로는 재학습을 하지 않으므로 이
  제약이 지나치게 엄격할 수 있다.
- **`PREDICTION_SOURCE_OBSERVED`** 상수(`config.py`)는 정의만 되어 있고 어떤
  코드 경로에서도 아직 쓰이지 않는다. distance==0인 완전 일치 historical
  case를 "관측값 그대로"로 구분해서 쓸지, 아니면 제거할지는 DP/다음 단계
  요구사항에 따라 결정할 문제로 남겨둔다.
