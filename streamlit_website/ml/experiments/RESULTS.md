# Held-out Dev18 검증 결과 (2026-09-24)

## ⚠️ 이 문서를 읽기 전에: "45개"와 "3,000개"를 혼동하지 말 것

아래 결과는 **두 가지 서로 다른 성격**으로 나뉜다. 하나를 다른 하나의 근거로 쓰지 않는다.

1. **실측 (45개 후보, 아래 "N/관측%/k/metric-sensitivity" 절)** — 실제 dev18 DB
   (`recommendation.sqlite3`)에 지금 등록된 후보 45개, 실제 결과 7,811건 기반. **진짜
   Recall/Precision 신호**지만, 최종 프로덕션 규모(팀 목표 ~3,000개 후보)보다 훨씬
   작은 prototype 규모의 결과다.
2. **구조 검증 (3,000개 후보, "3,000-규모 구조 검증" 절)** — 실제 데이터가 아니라
   **합성(무작위) 데이터**로 만든 가짜 DB. Recall 숫자 자체는 의미가 없다(무작위
   데이터이므로) — 이 절이 답하는 질문은 "숫자가 얼마냐"가 아니라 **"코드가 후보
   수를 하드코딩하지 않고, 3,000개 규모에서도 합리적인 시간에 동작하느냐"**다.

**따라서 N=100 같은 값을 "3,000개 규모에서 검증됐다"고 말할 수 없다** — 3,000개
규모에서는 아직 구조/성능만 확인했고, 실제 Recall@N 신호는 그 규모의 진짜 실행
결과가 쌓여야 얻을 수 있다.

`python3 -m streamlit_website.ml.experiments.run_holdout_experiments`로 재현한다.
콘솔에 마크다운 표를 출력함과 동시에, 같은 4개 sweep 결과를
`streamlit_website/ml/experiments/output/*.csv`(n_sensitivity.csv,
observed_q_percent_sensitivity.csv, k_sensitivity.csv,
similarity_metric_comparison.csv)로도 저장한다 — 이 폴더는 재생성 가능한
산출물이라 `.gitignore`에 등록돼 있다(추적 대상은 이 문서와 코드다).

## 방법

- **좋은 후보 정의(확정)**: held-out series의 실제 최종(q=100) VUS-PR 상위 N개 (config_id, head).
- **leave-series-out**: held-out series 자신의 prefix_features/결과는 similarity
  pool과 예측에서 전부 제외하고, ground truth로만 쓴다 (`holdout_validation.py` 참고).
- **범위**: 이 결과는 similarity+trajectory 기반 성능 예측에 feasibility와 cost
  tie-break까지 반영한 held-out 성능이다 (`candidate_selection.py`와 동일한
  usability 기준: feasible + performance + cost 모두 있어야 top-N 경쟁에 들어감).
  다만 feasibility의 `stage_inference_rows`는 held-out series 자신의 실제 q=100
  `test_observations`를 대신 쓴다 — 실제 운영 조건(`operating_conditions`)이 아니므로
  `run_ml_pipeline()`의 실제 판정과 완전히 같지는 않다 (`holdout_validation.py`
  모듈 docstring 참고).
- **실제 DB 규모**: 18개 series, 45개 (config_id, head) 후보. 피드백 문서가 예시로
  든 N=50/100/200/500은 이 규모를 넘어서므로 쓰지 않았다 — 3,000-후보 규모는
  미래/프로덕션 목표치로 합의됨 (구조만 확장 가능하게 만들고, 실측은 지금 규모로).

## N-sensitivity (관측 시점 고정: observed_q_percent=20%)

| N | Recall@N (18개 series 평균) | 후보 축소율 | ML 평균 실행 시간(s) |
|---|---|---|---|
| 5  | 0.222 | 0.889 | 0.117 |
| 10 | 0.367 | 0.778 | 0.110 |
| 20 | 0.553 | 0.556 | 0.109 |
| 45 (전체) | 0.990* | 0.000 | 0.108 |

\* 이론상 N=전체 후보 수면 Recall@N=1.0이어야 하지만, 일부 series에서 ground
truth/예측 후보 집합의 분모(실제 q=100 결과가 있는 후보 수)가 정확히 45가 아닌
경우가 있어 0.99로 나온다 — 임의로 1.0으로 보정하지 않고 실측값을 그대로 남긴다.

**해석**: 현재 45개 후보 규모에서는 N을 줄일수록(=축소율이 높을수록) Recall@N이
뚜렷하게 떨어진다. 이 표는 feasibility(구조적 실행 가능 여부)와 cost tie-break을
모두 반영한 `candidate_selection.py`와 동일한 usability 기준으로 나온 값이다 —
feasibility 반영 전(성능-전용 예측)보다 N=10 기준 Recall이 0.32→0.367로 올랐는데,
이는 일부 구조적으로 infeasible한 후보가 top-N 경쟁에서 제외되면서 남은 usable
후보들의 순위가 ground truth와 더 잘 맞았기 때문이다. **그럼에도 이 결과만으로
"적절한 N"을 하나로 확정하지 않는다** — 후보 수 자체가 45개뿐이라 N=10~20 같은
값이 이미 "전체의 상당 부분"을 차지해 향후 3,000-후보 규모에서의 거동과 다를 수
있기 때문이다. N을 최종 확정하려면 후보 수가 늘어난 future/production 규모에서의
재측정이 필요하다 — 다음 단계로 남긴다.

## 관측%(observed_q_percent)-sensitivity (N 고정: 10)

| 관측% | Recall@10 (18개 series 평균) | ML 평균 실행 시간(s) |
|---|---|---|
| 5  | 0.394 | 0.25 |
| 10 | 0.378 | 0.25 |
| 20 | 0.367 | 0.24 |
| 40 | 0.372 | 0.26 |
| 60 | 0.400 | 0.26 |
| 80 | 0.383 | 0.27 |

**해석**: 이 축(현재 데이터를 몇 %까지 관측했을 때 예측이 얼마나 맞는가)은 N축과
분리해서 측정했다. 관측%가 늘어난다고 Recall@10이 단조 증가하지는 않는다(5%에서
오히려 0.394로 60/80%와 비슷한 수준) — series 수(18개)가 적어 노이즈가 클 수
있으므로, 이 결과에서 "몇 % 이상 관측해야 한다"는 임의의 기준을 세우지 않는다.

## k-sensitivity (관측%=20, N=10 고정)

| k | Recall@10 (18개 series 평균) |
|---|---|
| 1  | 0.256 |
| 3  | 0.317 |
| 5  | 0.361 |
| 10 | 0.367 |

**해석**: k가 커질수록 Recall@10이 단조 증가한다 (feasibility/cost 반영 전에는
k=5에서 정점을 찍고 k=10에서 소폭 낮아지는 패턴이었으나, feasibility 반영 후에는
k=10이 가장 높다). 다만 series가 18개뿐이라 표본이 작으므로, 이 결과만으로
"k=10이 최적"이라고 확정하지 않는다 — N-sensitivity와 마찬가지로 향후 더 큰
규모에서 재검증이 필요한 참고 신호로 남긴다.

## Euclidean vs Cosine (관측%=20, N=10 고정)

| similarity_metric | Recall@10 (18개 series 평균) |
|---|---|
| euclidean (V1 기본값) | 0.367 |
| cosine | 0.400 |

**해석**: 이 18-series 표본에서는 cosine의 평균 Recall@10이 근소하게 더
높다(0.400 vs 0.367). 그러나 표본이 작고 차이가 크지 않아 **이 결과만으로
기본값을 Euclidean에서 Cosine으로 바꾸지 않는다** — README 근거(V1은 Euclidean을
첫 비교안으로 명시)와 "문서 근거 없이 기본값을 바꾸지 않는다"는 원칙을 그대로
유지한다. 이 비교는 향후 더 큰 데이터에서 재검증할 근거 자료로 남긴다.

## 3,000-규모 구조 검증 (합성 데이터, 2026-09-25 추가)

`python3 -m streamlit_website.ml.experiments.stress_test_scale`로 재현한다. 실제
DB가 아니라 합성 DB(series 18개는 실제와 동일하게, 후보 3,000개는 team 목표치를
그대로 반영해 무작위로 생성)를 만들어 harness를 그 규모에서 직접 실행한다.

**확인된 것**:

| 항목 | 결과 |
|---|---|
| 단일 `evaluate_holdout(top_n=100)`, 후보 3,000개 | 0.87s |
| N=(50,100,200,500) sweep 전체 (같은 DB 재사용) | 1.78s |
| 코드 변경 없이 3,000개 후보로 동작 | 성공 (후보 수 하드코딩 없음 확인) |

(feasibility 연산이 후보별로 추가되면서 이전 측정치(0.51s/1.13s)보다 다소 늘었지만,
3,000개 규모에서 여전히 2초 이내로 끝난다.)

**성능 개선 사항**: 원래 구현은 후보마다 DB 커넥션을 새로 열어(`load_results_for_candidate`
반복 호출) 후보 수에 선형으로 느려졌다(45개 기준 0.65초 → 3,000개면 약 43초 예상).
`db.load_all_results_grouped_by_candidate()`(한 번의 쿼리로 전체 후보를 묶어 읽음)로
바꾸고, `sweep_n_sensitivity`/`sweep_k_sensitivity`/`compare_similarity_metrics`가
N/k/metric을 몇 개를 sweep하든 DB를 한 번만 읽도록 캐시를 공유하게 했다. 그 결과
위 표처럼 3,000개 규모에서도 N 4개 sweep이 1초 남짓에 끝난다.

**이 검증이 말하지 않는 것**: 위 표의 Recall/Precision 값은 합성(무작위) 데이터에서
나온 것이라 의미가 없으므로 표에 싣지 않았다(코드에는 출력되지만 참고용일 뿐이다).
"N=50이 3,000개 규모에서 Recall 몇%다"라는 결론은 이 스트레스 테스트로 내릴 수
없다 — 실제 3,000개 규모의 결과가 DB에 쌓여야 `evaluate_holdout`/`sweep_n_sensitivity`를
그대로 그 DB에 돌려 진짜 신호를 얻을 수 있다 (코드 변경 불필요, `database=` 인자만
그 시점의 실제 DB 경로로 바꾸면 된다).

## Cost tie-break 연결 완료 (2026-09-25)

`holdout_validation.py`가 이제 `cost.py`(후보별 회귀 비용 추정)를 그대로 재사용해,
`candidate_selection.py`와 동일한 "predicted_performance desc, cost asc" 정책으로
Top-N을 정렬한다 (held-out series 자신의 실제 미래 규모를 학습/추론 규모로 써서
비용을 추정 — 임의의 운영 조건을 지어내지 않는다). 합성 데이터로 "성능이 동률일 때
저비용 후보가 먼저 온다"는 것도 테스트로 확인했다.

**실제 DB에서 확인한 것**: cost tie-break을 켜기 전/후로 45개 후보 기준 N-sensitivity
결과(Recall@N 등, 위 표)가 **달라지지 않았다** — 실제 vus_pr은 연속값이라 정확히
동률인 경우가 거의 없어 tie-break이 실제로 개입할 일이 드물기 때문이다. 즉 이번
변경은 "production 정책과 일치시킨 것" 자체가 목적이었고, 지금 규모에서 결과를
바꾸지는 않는다(예상된 결과이며, 회귀 아님).

## 남은 항목: "최적화 단계까지 포함한 N 비교"

팀 피드백이 요구한 최종 비교(N별로 **이후 최적화 결과** — 최종 성능/최종 cost/최적화
실행 시간 — 까지 비교)는 이 저장소에 DP/최적화(ALNS/Bee/Ant 등)가 전혀 구현돼 있지
않아 **아직 만들 수 없다** (여러 차례 grep으로 재확인, 임의로 만들지 않는다는 원칙).
이 ML 단계가 지금 줄 수 있는 것은:

- N별 Top-N 후보 목록 + `predicted_performance`(similarity 기반 추정) — 이미 구현됨.
- N별 estimated cost(`candidates_with_cost_estimate`, `mean_estimated_total_cost_seconds_of_top_n`) — 이번에 연결 완료.
- N별 ML 계산 시간(`ml_runtime_seconds`) — 이미 구현됨 (위 표 참고).
- N별 후보 축소율(`candidate_reduction_ratio`) — 이미 구현됨.
- **연결 완료(2026-09-25)**: feasibility(구조적 실행 가능 여부)도 이제 이 harness에
  반영됐다. `assess_future_feasibility`가 요구하는 `stage_inference_rows`(운영
  조건, 예: 하루 추론량)를 held-out 검증이 임의로 지어내는 대신, **held-out
  series 자신의 실제 q=100 `test_observations`**를 대신 썼다 — DB에서 이 값이
  series마다 (config_id와 무관하게) 하나로 고정된 상수임을 직접 쿼리로 확인한 뒤
  내린 결정이다 (`holdout_validation.py` 모듈 docstring 참고). `channel_count`도
  같은 방식으로 `channel_features` 테이블에서 series-level 상수로 가져온다.
  `candidate_selection.py`와 동일하게 "feasible + performance + cost 모두 있어야
  top-N 경쟁에 들어간다"는 `_is_usable` 기준을 harness에도 그대로 적용했다
  (`candidates_feasible`, `candidates_usable` 필드로 결과에 노출). 실제 DB 기준
  `evaluate_holdout('01', 20, top_n=10)` → 45개 후보 중 45개 feasible, 44개 usable.
- **최적화팀에서 채워야 함**: 최종 성능/최종 cost/최적화 실행 시간 — 이 저장소 범위
  밖이다. 최적화팀이 이 ML 출력(Top-N + predicted_performance + estimated cost)을
  입력으로 받아 실제로 최적화를 돌린 뒤에만 채울 수 있는 열이다.

## 다음 단계 (아직 하지 않은 것)

- "ML Top-N vs 전체 후보" 비교에서 **최적화 단계 이후의 결과**(최종 성능/최종
  cost/최적화 실행 시간)까지 채우는 것 — 위 "남은 항목"에 적었듯 이 저장소에
  DP/최적화가 구현돼 있지 않아 최적화팀의 몫으로 남는다. (feasibility/cost는
  이번에 연결 완료됐으므로 이 항목에서 제외.)
- prefix 관측%(observed_q_percent) 축을 N/k와 함께 3중으로 교차하는 실험(현재는
  각 축을 다른 축 고정 하에 개별적으로만 sweep했다).
