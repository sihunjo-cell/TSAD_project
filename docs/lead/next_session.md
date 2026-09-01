# 다음 세션 시작점

## 현재 게이트

[계획서 v5](plan_v5.md)의 0·1단계와 2단계 exact panel 실행·채점을 마쳤다. primary 물리 실행
1,170건과 `dev18_trial_score_ledger.csv` 1,602행이 모두 완료됐다. 기존 score, VUS checkpoint와
ledger는 지우거나 다시 계산하지 않는다.

운영 주분석은 비율마다 Tier 대표 모델을 고르는 `tier_adaptive`다. 모델별 config와 score variant는
기존 `model_fixed` 값을 유지한다. `tier_fixed`는 통제 비교이고 PCA_LEGACY q100 점수는 그림의
참고선일 뿐 후보나 gate가 아니다.

## 다음에 할 일 하나

Lightning에 이미 있는
`experiments/01_ghl_main/results/dev18_tuning/dev18_trial_score_ledger.csv`를 입력으로
[완료 ledger에서 선택표만 다시 생성](lightning_studio.md#완료-ledger에서-선택표만-다시-생성) 명령
하나만 실행한다. 이 경로는 score manifest, 원본 CSV, score 배열과 checkpoint lock을 읽지 않고
VUS evaluator나 모델 runner를 실행하지 않는다. 봉인 evaluator·`ell_max` 신원만 현재 코드와 대조한다.

완료 조건은 `selection_complete.json`의 ledger 1,602행, membership 294행, 필수 결과 SHA-256이
실제 파일과 모두 일치하는 것이다. `selection.png`에는 Tier 선 세 개, PCA_LEGACY 참고 점선과 작은
모델명만 둔다. 후보·배제·전환 상세는 세 CSV에서 확인한다.

## 봉인 상태

- budget: `b5367ad431093`
- VUS-PR: 공식 TSB-AD `opt` 대조 통과, 최대 절대 오차 0
- 완료 ledger: 1,602행, 모두 `complete`
- 운영 선택 규칙: `tier_adaptive_family_lofo_v1`
- membership: `model_fixed·tier_fixed·tier_adaptive` 294행
- 물리 실행: adaptive 추가 전후 `model_fixed` runnable key 합집합이 같음
- PCA_LEGACY: q100 plot-only reference

TimeRCD·TSPulse checkpoint, TSPulse batch 동등성과 L4 80% 자원 gate는 완료된 모델 실행의 역사
증거다. selection-only를 실행하려고 다시 측정하지 않는다.

## 다음 게이트

선택 영수증이 맞으면 2단계를 닫는다. 3단계 GHL25를 열기 전에 Role-A manifest를 따로 승인한다.
ALoRa는 구현 실패가 아니라 Dev18 18개 전체를 한 config로 덮지 못해 `unavailable`이다. 저채널
시계열의 `pair_count < heads 8` 근거는 후보 감사 CSV에 남긴다.

## 완료 보고 형식

현재 게이트, 입력 ledger SHA-256과 1,602행 확인, 294행 membership, 비율별 Tier 대표 경로,
`selection_complete.json`과 결과 폴더만 보고한다.
