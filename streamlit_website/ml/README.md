# streamlit_website/ml — TSAD ML 단계

DB→Streamlit 연결 다음, DP(경로 최적화) 이전에 들어가는 ML 단계다. **DP는 이
모듈이 구현하지 않는다.** `run_ml_pipeline(ml_input, db_path=...) -> MLOutput`이
유일한 진입점이다. `build_dp_input(ml_output, ml_input)`이 결과를 DP 입력 계약
(`streamlit_website/DP/README.md`)으로 옮긴다.

## 범위

1. 현재 prefix feature 계산/정리 (`features.py`, `compute_prefix_features.py` 재사용)
2. 과거 prefix similarity 검색 (`similarity.py`)
3. similarity 기반 historical trajectory 검색 (`trajectory.py`)
4. 단계별 성능 추정 (`performance.py`)
5. 학습/추론 비용 추정 (`cost.py`)
6. 단계별 feasibility 재판정 (`feasibility.py`, `model_feasibility.assess_candidate` 재사용)
7. 단계별로 DP에 넘길 후보와 제외 후보 분리 (`candidate_selection.py`)
8. checkpoint 유지 후보를 별도로 보존
9. `MLOutput` 생성 (`schemas.py`, `pipeline.py`)
10. DP 입력 계약으로 변환 (`dp_input.py`)

## 파일 구조

| 파일 | 책임 |
|---|---|
| `schemas.py` | `MLInput`/`MLOutput`과 중간 산출물 dataclass |
| `config.py` | 이웃 수 k, feature 목록 등 조정 가능한 값 |
| `db.py` | `recommendation.sqlite3` 읽기 전용 접근 (계산 없음) |
| `features.py` | 현재/과거 feature 벡터 정리 (행 수 제외, NULL 보존) |
| `similarity.py` | population 표준화 + Euclidean + distance-weighted kNN |
| `trajectory.py` | series별 q trajectory, 목표 비율 이하의 마지막 관측값 조회, run_id 기반 물리 실행 재사용 추적 |
| `performance.py` | similarity + trajectory로 단계별 성능 추정 (provenance 포함) |
| `cost.py` | 후보별 실측 실행 시간으로 학습/추론 비용(초) 추정. PaAno는 행 수 회귀, 나머지는 `행 × 센서 수`당 중앙값 속도 |
| `feasibility.py` | 미래 stage 행 수 기준 구조적 feasibility 재판정 (wrapper만) |
| `candidate_selection.py` | stage별로 DP에 넘길 후보와 제외 후보(사유 포함) 분리 |
| `pipeline.py` | 위 전부를 묶는 `run_ml_pipeline()` |
| `dp_input.py` | `MLOutput`을 DP의 학습·추론 선택지로 변환하는 `build_dp_input()` |
| `test_ml_pipeline.py` | unit + 임시 DB + 실제 dev18 DB 통합 테스트, DP planner 입력 확인 |

`similarity_metric`(euclidean/cosine)과 `similarity_knn_k`는 `MLConfig`에서 바꾼다.
held-out 검증 코드와 결과(`experiments/`)는 저장소 배치 규칙에 따라 이 폴더로 가져오지
않았고 원 브랜치 `feature/ml-pipeline`에 남아 있다.

DB 경로는 인자 > 환경변수 `TSAD_RECOMMENDATION_DB` > 기본값
(`experiments/tuning/results/recommendation.sqlite3`, git에 커밋되지 않음) 순으로 정한다.

## 유사도 검증 (2026-09-25)

실제 Dev18 DB에서 파일 하나씩, 또 family 전체를 빼고 남은 파일로 뺀 파일의 후보별
VUS-PR을 예측했다. 새 현장은 Dev18 어느 family에도 속하지 않으므로 family를 뺀 결과를
기준으로 본다. 관측 q 6개와 그 이후 q 전부, 후보 45개를 썼고 p값은 파일 18개 단위
Wilcoxon이다. 하한 충족률은 예측 top-10 안에서 예측 성능이 하한을 넘는 후보 중 가장 싼
후보가 실제로도 하한을 넘은 비율이며, 하한은 파일마다 후보 실제 성능의 25·50·75% 분위수로
두었다. 비용은 그 파일의 실측 실행 시간이다. 일회성 검증이라 코드는 남기지 않았다.

| family 제외 검증 | 절대 오차 | 예측 1위 손실 | 하한 충족률 | 성공 시 비용 (최저 가능 대비) |
| --- | ---: | ---: | ---: | ---: |
| 유사도 k=10 (이전 기본값) | 0.335 | 0.259 | 0.36 | 18배 |
| 유사도 k=30 (현재) | 0.310 | 0.221 | 0.37 | 13배 |
| 유사도 없음, 과거 prefix 동일 가중 | 0.276 | 0.367 | 0.45 | 324배 |
| 유사도 없음, 절대 행 수 기준 평균 | 0.273 | 0.406 | 0.45 | 301배 |

유사도는 후보 사이의 상대 비교에 필요하다. 유사도 없이 예측 1위를 고르면 실제 최고
후보보다 0.37~0.41 낮은 후보를 고르며 k=10보다도 나쁘다(p≤0.012). 하한 충족률은
유사도 유무로 달라지지 않았고(k=30 대 유사도 없음 p≈0.22), 유사도를 쓰면 같은 수준의
충족률을 훨씬 싼 모델로 얻는다. k=30은 k=10과 같거나 나았다(family 제외 절대 오차
p=0.024, 파일 제외 예측 1위 손실 p=0.020). 파일마다 가장 가까운 q 하나만 쓰는 방식은
이득이 없었다.

병목은 절대 수준이다. 같은 후보·q에서도 파일 간 VUS-PR 표준편차가 0.31이고, 어떤
방법도 새 family 파일의 절대값을 평균 0.27보다 가깝게 맞히지 못했다. 그래서 절대 하한을
두는 목적함수를 버리고 예산 안에서 예측 성능을 최대화하는 쪽으로 바꿨다. 이 오차는
대부분 모든 후보에 같이 얹히므로 예측 전체에 +0.3을 더해도 고르는 계획은 98% 같았다.

한때 top-10 축소가 하한 충족률을 올리는 안전 여유로 보였지만(0.21 → 0.37), 계획을
만들지 못한 경우를 실패로 센 결과였다. 지금은 순위로 후보를 자르지 않는다.

## 예산 계획 백테스트 (2026-09-25)

Dev18에서 family를 뺀 채 실제 운영처럼 돌렸다. 데이터가 파일의 다음 DB 비율(5→10→…→80%)에
닿을 때마다 그 prefix로 ML과 DP를 다시 실행하고, 다음 재계산까지는 계획의 구간 일정을 그대로
따랐다. 실현값은 배치한 모델의 실제 VUS-PR을 시간으로 가중한 값이다. 운영 기간은 100일, 추론량은
test 구간 20개 분량, 예산은 파일마다 실제 가장 싼 계획 비용의 배수다. 평균은 family 10개 단위이고
p는 같은 단위 Wilcoxon이다. 일회성 검증이라 코드는 남기지 않았다.

| 예산 (q5→q100) | 이전 시간 격자 | DB 비율 구간 (현재) | 가장 싼 계획 | 사후 최적 |
| --- | ---: | ---: | ---: | ---: |
| 최저 비용의 1.5배 | 0.249 | 0.249 | 0.263 | 0.429 |
| 10배 | 0.242 | 0.243 | 0.263 | 0.474 |
| 100배 | 0.247 | 0.266 | 0.263 | 0.561 |
| 제한 없음 | 0.411 | 0.424 | 0.263 | 0.673 |

구간을 DB 비율 경계로 나누면 같은 비율에서 PaAno·GDN을 두 번 학습하는 계획이 사라진다. 예산이
없을 때 학습 횟수가 8.2회에서 5.4회로, 파일별 비용 중앙값이 151초에서 105초로 줄었고 성능은
같거나 조금 높았다. 기간 끝이 파일의 학습 구간 끝과 다른 설정(q5→q40)과 q20→q100에서도 12개
조합 모두 유의하게 나빠지지 않았다.

싼 모델만 들어가는 예산(100배 이하)에서는 어떤 방식도 가장 싼 계획을 유의하게 넘지 못했다.
이 범위의 선택은 MWVAR·SQDIFF 설정 사이의 선택인데, 같은 모델의 설정끼리는 유사도 예측이 실제
순위를 맞히지 못한다. 예산이 PaAno·GDN·TSPulse까지 허용해야 평균 0.16을 더 얻는다(p=0.16).

같이 시험하고 채택하지 않은 방안:

- **설정 예측을 모델·tier 평균으로 묶기**: 빠듯한 예산에서 +0.01 안팎(유의하지 않음), 예산이
  없을 때 −0.10(p=0.039)·−0.09(p=0.02). 비싼 모델의 설정 선택에는 유사도 정보가 쓸모 있었다.
- **유사도 없이 전체 평균**, **하한 신뢰구간(평균 − 표준편차)**: 모든 예산에서 같거나 나빴다.
- **beam search(폭 1·5)**: DP와 같거나 차이가 유의하지 않았다. DP는 예측값 기준 최적해를 이미
  정확히 찾으므로(전수 열거 테스트) 탐색 방식을 바꿔도 예측 오차는 줄지 않는다.
- **비용을 과거 실측 비율의 75·90% 분위로 부풀리기**: 예산 초과 파일이 6~11%에서 0~6%로 조금
  줄었지만 성능 이득이 없고 90%는 빠듯한 예산에서 −0.02였다.
- **이웃마다 성장 비율 경계를 따로 두기**: 구간이 최대 16개로 늘어 메모리가 모자라 끝까지 평가하지
  못했다. 성능으로 뺀 방안은 아니다.

허용 폭 ε는 0이 가장 좋았다(0.01~0.05에서 −0.07~0.10, p≤0.02). 데이터가 쌓일 때의 성능
향상을 아예 넣지 않으면 예산이 없을 때 0.04 낮았다(p=0.008).

## Streamlit 연결

`streamlit_website/db_connection/view.py`의 `render_candidate_intake()`는 후보 풀과
현재 feature를 만든 뒤 `run_ml_pipeline()`을 바로 실행한다. form 제출 뒤에 따로 누르는
버튼은 Streamlit 재실행에서 제출 상태가 사라져 동작하지 않으므로 제출과 함께 돌린다.
결과는 `build_dp_input()`으로 바꿔 `st.session_state["dp_input"]`에 두며, 상단
`운영 경로`가 이 값을 읽는다. 새 파일을 올리거나 다시 제출하면
`ml_input`과 함께 지운다.

## DP 연결

stage는 DP 계약과 같이 비용을 내는 구간이다. 운영 기간 끝까지 모을 정상 행 수를 100%로 보고,
데이터가 DB 등록 비율(5·10·20·40·60·80%)에 닿는 날마다 새 구간을 시작한다. 지금 비율은 넘지
않는 가장 큰 등록 비율로 내리며 5% 미만도 5%로 본다. 구간이 시작할 때 확보한 행 수로
feasibility·학습비를, 그 비율에서 잰 과거 결과로 성능을, 그 구간의 추론량으로 추론비를 추정한다.
DB는 이 비율에서만 성능을 쟀으므로 계획의 답도 "데이터가 몇 %일 때 무엇을 하라"로 나온다.
같은 비율은 같은 행 수가 아니라 모을 데이터 중 같은 몫이라는 뜻이다. 수집이 없으면 지금 데이터가
전부이므로 100% 구간 하나가 된다.

`build_dp_input()`은 구간마다 실행할 수 있고 성능·비용을 추정한 후보 전부를
`training_options`로 넘기고 학습 없는 후보의 학습비는 0으로 둔다. `inference_options`는
학습 원점마다 이후 구간의 추론비를 붙인다. 추론비는 학습 시점과 관계없이 그 구간의 후보
추정값이다. 구간 `weight`는 구간 길이(일)이며 DP가 합으로 나눠 비중으로 쓴다.

현재 후보(`config_id::head`)가 후보 풀에 있고 checkpoint ID와 성능 추정이 있으면
`current_checkpoint`와 `trained_stage=-1` 추론 행을 만든다. 유지 성능은 그 checkpoint를
학습한 행 수로 추정한다. 유지는 다시 학습하지 않으므로 학습 길이 구조 제약으로 막지 않는다.

CPU·GPU 시간당 비용이 있으면 후보의 측정 장치에 맞춰 초를 원으로 바꾸고 `cost_unit`을
`KRW`로 둔다. 측정 장치를 모르면 비싼 쪽 단가를 쓴다. 단가가 없으면 `seconds`다. 남은 예산은
같은 단위로 `budget`에 넣고, 0이나 빈 값은 제한 없음이다. 지표는 `VUS-PR`뿐이다.

## postprocess_\*/l4_\* 테이블에 대한 노트 (existing DB evidence)

실제 DB에는 이 저장소 코드가 만들지 않는(= 다른 팀원이 별도로 구축한)
`postprocess_*`(예: `postprocess_model_best`, `postprocess_tier_best`,
`postprocess_candidate_scores` 등)와 `l4_*`(`l4_cost_inputs`,
`l4_candidate_costs`, `l4_tier_transition_costs`) 테이블/뷰가 있다. 이들은:

- **이 ML 파이프라인의 입력이나 정답으로 쓰지 않는다.** `postprocess_model_best`/
  `postprocess_tier_best`로 후보를 정하지 않고, 이 모듈이 독립적으로 계산한
  `predicted_performance`와 비용 추정만 DP에 넘긴다.
- **참고용 대조 자료로만 취급한다.** 이 파이프라인이 계산한 성능/비용 추정치가
  `postprocess_*`의 사전 계산 결과와 방향이 일치하면 그 사실을 기록해 둘 가치가
  있지만(신뢰도의 방증), 이 저장소 코드에서 재생산할 수 없는 로직이므로 의존하지
  않는다.
- `l4_cost_inputs.l4_hourly_rate`는 이 DB에서 전부 NULL이다. 그래서 `cost.py`는 초 단위
  비용만 만들고, 금액 환산은 화면에서 받은 시간당 비용으로 `build_dp_input()`이 한다.

## ML/DP 책임 경계 (2026-09-25 개정)

`pipeline.py` 모듈 docstring에 전문이 있다. 요약:

- ML은 성능 하한으로 후보를 거르지 않고 예측 순위로 자르지도 않는다. 실행할 수 없거나
  성능·비용을 추정하지 못한 후보만 `stage_candidates_excluded`에 사유와 함께 남긴다.
- 예산과 성능의 균형은 DP가 정한다. 하한(`minimum_performance`)은 DP 계약에 선택 필드로만
  남아 있고 ML은 넣지 않는다.
- 지표는 VUS-PR 하나다. `recommendation.sqlite3.results`에 precision/recall/f1 열이 없어서
  화면의 지표 선택도 뺐다.

## 아직 해결하지 못한 가정/제약

- **데이터가 쌓일 때의 성능 향상**: 운영 기간 끝까지 모을 데이터를 DB의 100%로 보는 것은
  가정이다. Dev18에서 비율에 따라 성능이 달라지는 모델은 PaAno(+0.10)와 GDN(+0.07)뿐이고,
  나머지 33개 후보는 모든 비율에서 같은 실행을 재사용해 값이 같다. 그 폭이 현장마다 얼마인지는
  예측력이 낮으므로 재학습 시점 제안은 이 불확실성을 안고 있다.
- **예산 초과**: 비용은 과거 실측의 중앙값이라 예산을 꽉 채운 계획은 실제로 넘을 수 있다.
  롤링 백테스트에서 예산을 넘은 파일은 6~11%였고 대부분 1.2배 이내였다. 재계산 때 실제로 쓴
  비용을 빼고 남은 예산을 넣으면 이후 계획이 이를 반영한다.
- **feasibility의 `test_length`**: 저장소 계약에서 `test_length`는 한 번에 평가하는
  연속 구간 길이이며 누적 추론량이 아니다. 화면의 선택 입력 `inference_batch_length`가
  있으면 그 값을 쓰고, 없으면 `inference_rows_per_day`로 근사한다. 이 근사는 검증하지
  않았으며 매 실행 `metadata["warnings"]`에 남긴다.
- **checkpoint maintenance 성능**: `last_trained_at`은 표시용 자유 텍스트다. 유지 성능은
  화면에서 받은 학습 행 수(`trained_rows`)로 추정하고, 없으면 현재 행 수로 학습했다고
  본다(`performance.py::estimate_checkpoint_maintenance_performance`). 행 수를 비워 두면
  오래전에 학습한 checkpoint를 낙관적으로 볼 수 있다.
- **`PREDICTION_SOURCE_OBSERVED`** 상수(`config.py`)는 정의만 되어 있고 어떤
  코드 경로에서도 아직 쓰이지 않는다. distance==0인 완전 일치 historical
  case를 "관측값 그대로"로 구분해서 쓸지, 아니면 제거할지는 DP/다음 단계
  요구사항에 따라 결정할 문제로 남겨둔다.
