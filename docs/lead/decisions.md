# 현재 결정

갱신일: 2026-09-01

이 문서는 [계획서 v5](plan_v5.md)의 확정 결정을 짧게 기록한다. 설계 이유와 전체 절차는 계획서를
따르고, 값이 아직 정해지지 않은 항목을 완료된 결정처럼 쓰지 않는다.

## 연구 범위

최종 목적은 제조 현장의 cold-start TSAD 도입안을 비용까지 포함한 목적함수로 고르는 것이다.
본실험 데이터는 GHL과 HAI다. TSB-AD-M 비-GHL 18개는 모델과 recipe를 고르는 사전 튜닝 패널이며
세 번째 본실험 데이터셋이 아니다.

운영 주분석은 모델별 recipe를 고정한 채 정상 prefix 비율마다 Tier 대표 모델을 다시 고르는
`tier_adaptive`다. `tier_fixed`는 대표 모델까지 고정해 데이터 양의 효과를 분리하는 통제 비교로
보존한다. adaptive 곡선에는 모델 전환 효과가 섞이므로 데이터 양의 단독 효과로 해석하지 않는다.

## 데이터 역할

1. 공식 `TSB-AD-M-Tuning.csv` 20개는 provenance와 SHA-256 확인용 원본 목록이다.
2. GHL 09·18을 뺀 18개는 TSB 튜닝 패널이다. 나머지 파일과 순서는 고정한다.
3. GHL 09·18을 포함한 GHL25 전체는 주실험이다. 튜닝 패널과 GHL25의 교집합은 0건이다.
4. HAI 23.05는 `train1 → test1`, `train1+train2 → test2` 두 실행으로 외부 확인한다.
5. HAI train3·train4는 본 결과 뒤의 선택적 시간순 민감도에만 남긴다.

TSB 튜닝의 TAO 11·12번은 공식 `tr_500` 안에 이상 라벨이 각각 48개와 40개 있다. 공식
TSB-AD의 고정 commit도 `tr_` prefix를 라벨로 정제하지 않고 그대로 학습 입력에 쓴다. 이 두
파일을 빼거나 라벨로 걸러내면 18개 패널과 label-blind 계약이 함께 깨지므로 공식 prefix를
보존한다. 모델은 라벨을 받지 않으며 TAO family leave-one-out 결과로 오염 영향을 따로 확인한다.

기존 파일명과 저장 경로의 `dev18`은 TSB 튜닝 패널을 가리키는 내부 식별자다. 문서 산문에서는
혼선을 막으려고 “TSB 비-GHL 튜닝 패널 18개”라고 쓴다.

## 역할과 인수 순서

강혁은 데이터 EDA와 manifest를 승인한다. 모델 담당자는 이 근거로 정적 feasibility를 계산하고
모델·본실험 함수, 점수와 실행 증거를 만든다. 지우는 VUS-PR, `ℓ_max`, 튜닝 채점과 선택표를 맡는다.
주혜는 난이도, hit, Wilcoxon, TOST, bootstrap, 교차점과 결과표를 맡는다.

인수 순서는 `강혁 → 모델 담당자 → 지우 → 모델 담당자 final 실행 → 지우 final 채점 → 주혜 →
비용 최적화`다. 지우의 선택표는 모델 점수가 생긴 뒤 만들어지므로 정적 feasibility의 선행 조건이
아니다. 모델 runner는 지우의 선택식을 다시 계산하지 않고 `final_policy_membership.csv`만 실행
요청으로 소비한다.

## 활성 모델

- Tier 1: `MWVAR`, `SQDIFF_LAST3`, `PCA_LEGACY`
- Tier 2: `PaAno`, `ALoRa`, `GDN`
- Tier 3: `TimeRCD`, `TSPulse`

GDN은 공식 `d-ailin/GDN` commit
`9853899da860682669a134e4af315d036aab4eca`를 기준으로 한
`src/models/tier2/gdn_official/` 구현 하나만 활성화한다. 과거 HAI 전용 GDN 구현과 edge/Jaccard
보조 분석은 연구 범위에서 제외한다. 두 구현의 성능 비교나 동등성 검증도 하지 않는다.

`CATCH`는 license와 검증된 공식 실행 경로가 없어 활성 후보에서 제외한다. MOMENT, CrossAD,
DADA, CAROTS, ScatterAD도 이번 roster에 넣지 않는다. 결과를 본 뒤 후보를 추가하지 않는다.

## 비율과 분할

`Q={005,010,020,040,060,080,100}`이다. 각 파일의 manifest training prefix `[0,N)`에서
`floor(N×q/100)`개 앞쪽 시점만 제공한다. GHL·HAI의 prefix는 정상이며 TSB 튜닝의 TAO 두
파일에는 위에서 밝힌 원본 오염이 있다. 학습형 모델은 이 prefix 안에서 다시 80:20으로
fit·validation을 나눈다. 전체 정상 구간의 뒤쪽을 낮은 비율에 미리 주지 않는다.

Tier 2 입력 scaler는 fit에만 맞추고 validation과 test에는 transform만 적용한다. 학습형 score
교정 통계는 같은 prefix의 validation raw score만 쓴다. training-free Tier 1과 strict zero-shot은
target calibration을 쓰지 않는다. HAI window와 validation score는 파일 경계를 넘지 않는다.

정적 `q_floor={t1:5,t2:10,t3:5}`를 유지한다. 강혁 manifest로 길이·채널 제약을 다시 계산했을 때
모순이 생기면 점수 생성 전에 결정을 다시 봉인한다.

## 모델과 recipe 선택

primary HPO는 `equal_trial`이다. 정적 feasibility 뒤 Tier별 후보 수가 확정되면 각 모델에 같은
수의 full-fidelity trial을 배정한다. exact panel은 `budget_id=b5367ad431093`으로 봉인했으며
시계열당 물리 실행 65건, primary logical score 89행이다.
`runtime_matched`는 timing 근거가 모인 뒤의 선택적 민감도다.

`budget_id=b5367ad431093`과 기존 `c...` config 행은 새 project commit에서 TimeRCD Dev18 checkpoint
smoke, TSPulse batch 1 대 batch 32 동등성, non-interruptible L4 80% 자원 gate가 같은 실행 계약을
확인할 때만 유지한다. adaptive 정책과 수동 모델 실행은 이 gate의 fallback이 아니다.

Tier 대표 후보는 `q_floor` 이상 모든 주분석 비율과 GHL25·HAI 두 실행을 정적으로 지원해야 한다.
후보가 없으면 Tier를 `unavailable`로 남기며 시작 비율을 올려 되살리지 않는다. 모델별 고정 곡선은
각 모델의 실제 지원 범위에서 보존한다.

stochastic 모델의 TSB 튜닝 seed는 `{0,1,2}`, GHL final seed는 `{3,4,5,6,7}`이다.
deterministic 모델은 한 번 실행한다. 일부 데이터나 축소 epoch로 예선하지 않는다.

주 선택 지표는 `raw__trainnorm` VUS-PR이다. seed 평균 뒤 10개 family를 동일 가중한다. 모델은
family leave-one-out 바깥 검증으로 고르고, 선택 모델의 recipe는 같은 봉인 예산으로 18개 전체에서
한 번 고정한다. 점수 차이가 `1e-6` 이내면 `(model, config_id, score_variant)` 사전순으로 고른다.
비용은 동률 처리에 쓰지 않는다.

`model_fixed_policy.csv`는 모든 활성 모델의 고정 recipe와 물리 실행 합집합을 보존한다.
`tier_fixed_policy.csv`는 통제 비교이며 `ratio_adaptive_selection.csv`가 운영 주분석이다.
adaptive 선택도 모델마다 전체 지원 비율에서 정한 family-LOFO recipe와 최종 `model_fixed` config를
쓴다. 비율별 config 재선택이나 HPO 재실행은 하지 않는다. PCA_LEGACY는 adaptive 후보와 성능 gate에서
빼고 q100 VUS-PR 참고선으로만 그린다.

`final_policy_membership.csv`는 `model_fixed·tier_fixed·tier_adaptive` 294행이다. adaptive 행을
더해도 runnable 물리 key의 합집합은 `model_fixed`만 있을 때와 같다. 주실험 runner의 필수 입력은
이 membership 하나다.

## 점수와 재현성

adapter는 threshold 전 연속 score만 만든다. point adjustment, anomaly ratio, test-optimal
threshold, test-derived normalization과 test loss 기반 조기 종료를 쓰지 않는다. 주결과는
`raw__trainnorm`, smoothing과 `testnorm`은 민감도다.

GDN score는 1-step forecast 오차다. test 길이가 `L`, window가 `W`면 score 길이는 `L-W`,
`source_start=W`, label은 `labels[W:]`다. 활성 GDN 하나의 이 계약만 확인한다.

각 실행은 config, 입력 SHA-256, package 버전, source commit, checkpoint SHA-256, seed와 split을
snapshot에 남긴다. Dev18의 `dev18_registered_runner.v2`는 in-memory 등록 executor 안에서 잰
`split_preprocess_seconds`, `model_setup_seconds`, 학습·validation 추론·test 추론 시간을 남긴다.
다섯 값의 합은 `runtime_seconds`와 맞아야 한다. peak memory, artifact 크기, 관측 수와 검증 가능한
지속시간 근거도 저장한다. 실패·timeout·unavailable은 성공 score로 만들지 않고 manifest에 이유와
재시도 수를 적는다.

## 최종 평가와 비용

GHL25는 모든 활성 모델의 고정-recipe 곡선을 보존한다. Tier 대표만 남기지 않는다. HAI는 TSB
튜닝에서 정한 모델과 recipe를 두 실행에 그대로 적용한다. `tier_adaptive`가 운영 주분석이고
`tier_fixed`는 통제 비교다. GHL이나 HAI 결과로 정책을 다시 고르지 않는다.

주혜는 GHL `1/25` macro, paired bootstrap, Wilcoxon, TOST와 지속 교차점을 만든다. 같은 시계열의
일곱 `q`를 독립 표본으로 세지 않는다. HAI 두 실행은 따로 보고하고 GHL과 합쳐 검정하지 않는다.

최종 행동 단위는 `(tier, model, config_id, q, operating_threshold)`다. GHL·HAI split은 행동이
아니라 평가 근거를 구분하는 문맥이다. 목적함수는 정상
데이터 관측비, 현장 재학습비, 반복 추론비, memory·artifact 비용, 오탐비와 미탐비를 합친다. HPO는
개발비로 따로 보고한다. 비율 사이 모델 교체·검증·배포·중단 비용도 현장 입력이 있을 때 더한다.
값이 없으면 0으로 채우지 않는다. GHL에서 시간 근거가 없으면 관측 수를 임의의 초 단위로 바꾸지
않는다. 비용 가중치와 현장 제약은 성능·비용 원자료가 봉인된 뒤 정한다.

## 아직 열려 있는 결정

- GHL25·HAI Role-A manifest의 최종 승인 상태
- 비용 항목별 단가, 반복 횟수, latency·memory·최소 성능 제약
- 모델 교체·검증·배포·중단 비용을 포함한 현장 transition 단가와 제약

Dev18 입력, feasibility, VUS-PR, 시계열별 `ℓ_max`, exact panel과 공통 Python 환경은 승인됐다.
primary 물리 실행 1,170건과 1,602행 ledger가 완료됐다. 과거 TimeRCD·TSPulse checkpoint와 L4 80%
자원 보고서는 이 실행의 역사 증거이며 selection-only에서 다시 요구하지 않는다. 현재 게이트는
완료 ledger로 운영 선택표와 294행 membership을 봉인하는 일이다. GHL25·HAI 항목은 이 작업을
막지 않는다.
