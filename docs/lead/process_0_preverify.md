# 사전 점검 결과

갱신일: 2026-08-27

## 현재 판정

현행 기준은 [계획서 v5](plan_v5.md)다. 본실험은 GHL과 HAI이며, TSB-AD-M 비-GHL 18개는
모델·recipe 선택용 튜닝 패널이다. 활성 roster는 Tier 1 `MWVAR·SQDIFF_LAST3·PCA_LEGACY`,
Tier 2 `PaAno·ALoRa·GDN`, Tier 3 `TimeRCD·TSPulse`다.

GDN은 공식 `d-ailin/GDN` 적응 구현 하나만 활성 모델로 인정한다. 과거 HAI 전용 GDN 실행과
그래프 보조 분석은 폐기했으며 활성 registry와 runner에서도 제거했다. 두 구현을 비교하거나
동등성을 증명하는 작업은 필요하지 않다.

Dev18 Role-A 감사와 manifest 인수, label-blind 정적 feasibility, exact `equal_trial` panel과
`budget_id=b5367ad431093`, VUS-PR evaluator와 시계열별 `ℓ_max`까지 닫았다. 현재 정지선은
전체 exact panel을 실행하기 직전이다.

Dev18 18개 원본은 감사와 160×2 gate smoke에만 썼다. 전체 모델 점수와 HPO는 실행하지 않았다.
라벨은 Role-A 오염·구간 감사와 evaluator 대조에만 썼고 feasibility, 예산, checkpoint forward와
gate smoke에서는 읽지 않았다. TimeRCD와 TSPulse의 준비 상태는 `ready`다.

## 데이터 EDA 인수 조건

강혁의 예외 역할로 Dev18은 아래 근거를 확정해 인수했다. GHL25와 HAI는 같은 기준의 최종
인수가 아직 남아 있다.

| 범주 | 필요한 근거 | 인수 판정 |
| --- | --- | --- |
| 파일 신원 | 역할, 상대 경로, 파일명, 바이트 크기, SHA-256 | 원본 목록과 모두 일치 |
| shape | 행 수, feature 수, feature 이름·순서 | 모델 입력 열을 재현 가능 |
| 경계 | 정상 학습·테스트 인덱스, prefix 기준 `N` | 범위가 겹치거나 비지 않음 |
| 값 품질 | numeric 여부, NaN·Inf·결측·중복 timestamp | 수치와 처리 방침이 명시됨 |
| 라벨 | 학습 구간 오염, 테스트 라벨 길이, 이상 구간 개수·길이 | 모델 입력과 채점 입력을 분리 가능 |
| 채널 | constant·IQR 0·고상관 현황 | 삭제 없이 원형 보존 |
| 주기성 | 학습 구간 ACF 후보와 lag 상한 | 지우의 `ℓ_max` 검토에 전달 가능 |
| HAI 세션 | 각 CSV의 독립 세션 여부, 86개 센서 순서 | window·통계가 파일 경계를 넘지 않음 |

공식 `TSB-AD-M-Tuning.csv` 20개는 provenance 목록이다. GHL 09·18을 뺀 18개 튜닝 패널과
GHL25의 교집합은 0건이어야 한다. 공식20과 GHL25의 교집합이 GHL 09·18 두 건인 것은 정상이다.

EDA가 모델 window, HPO winner, VUS-PR, threshold나 채널 삭제를 정하지 않는다. 값이 부족하면
모델 담당자가 추정하지 않고 강혁에게 보완을 요청한다.

## 공식 source 사전 기록

아래 source는 모델 구현의 출발점이다. 표의 기록만으로 실제 실행 준비가 끝난 것은 아니다.

| 모델·참고 코드 | 봉인 후보 commit | license | 현행 용도와 남은 확인 |
| --- | --- | --- | --- |
| One-Liners | `dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9` | MIT | MWVAR·SQDIFF 기준점. 다변량 채널 처리와 길이 `L` 계약 확인 |
| TSB-AD | `e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48` | Apache-2.0 | PCA 참고. benchmark의 threshold·후처리는 사용하지 않음 |
| PaAno | `d4c67116190efa4592dc6a8a157ced0def68b6af` | MIT | train/test 세션 분리와 point score 정렬 확인 |
| ALoRa | `97dcc4a337710e6dc72c1a67893717c9538bae1a` | EUPL-1.2 | GHL 19채널·HAI 86채널 shape와 라벨 threshold 분리 확인 |
| GDN | `9853899da860682669a134e4af315d036aab4eca` | MIT | 유일한 활성 GDN. 1-step score 정렬과 fit-only scaler 확인 |
| Time-RCD | `372bb980426b2f67007311c6f3165ab789c79bef` | Apache-2.0 | multi checkpoint SHA와 target-derived normalization 비사용 확인 |
| TSPulse | `9739fa59b61bd9f15cbfb06e5dc3dab28c72ee8d` | Apache-2.0 | AD revision·checkpoint SHA, raw score head와 길이 `L` 확인 |

`CATCH`는 공식 source의 license 파일과 검증된 실행 경로가 없어 활성 roster에서 제외했다.
과거 모델과 candidate-only 조사 항목은 현재 feasibility와 HPO 예산에 넣지 않는다.

## 분할과 누수 계약

비율은 `005`, `010`, `020`, `040`, `060`, `080`, `100`이다. 각 파일의 정상 학습 구간 앞쪽
`floor(N×q/100)`개만 현재 관측량으로 쓴다. 학습형 모델은 그 안에서
`fit=floor(0.8×available)`, `validation=available-fit`으로 시간순 분할한다.

Tier 2 입력 scaler는 fit에만 맞춘다. validation과 test에는 transform만 적용한다. score
calibration은 같은 `q` validation의 raw score만 쓴다. 전체 정상 구간의 마지막 10%를 낮은
비율에 미리 주지 않는다. training-free Tier 1과 strict zero-shot에는 `q`별 target calibration을
적용하지 않는다.

HAI 첫 실행은 train1의 현재 prefix, 둘째 실행은 train1·train2의 현재 prefix만 쓴다. 훈련 파일은
각자 80:20으로 나누고 window와 calibration score를 파일 사이에 만들지 않는다. 첫 scaler는
train1 fit, 둘째 scaler는 train1·train2 fit의 합에만 맞춘다.

## 정적 feasibility의 입력과 출력

강혁 manifest를 인수한 뒤 모든 `(model, config_id, q, tuning_series)`에 대해 아래 항목을
label-blind로 계산한다.

- `available_count`, `fit_count`, `validation_count`, `test_count`
- window·patch 생성 개수와 최소 연속 score 개수
- feature 수, GDN top-k와 attention pair 제약
- HAI 단일·다중 훈련 세션 지원 여부
- `valid`, `status_reason`, registry·manifest SHA-256

정적 불가능 조합은 실행하지 않고 `unavailable`로 남긴다. 같은 `q`의 튜닝 패널 18개 전부를
처리할 수 있는 설정만 해당 `q`의 후보가 된다. 늦게 시작하는 모델 때문에 Tier `q_floor`를
올리지 않는다.

정적 원표가 끝난 뒤에만 각 Tier의 feasible 모델 수와 공통 config 수를 계산하고 `equal_trial`
exact panel을 만든다. panel과 순서, seed, 실패·동률 규칙을 승인한 뒤 `budget_id`를 봉인한다.

Dev18 원표는 3,276행이며 feasible 2,592행, structurally infeasible 684행이다. 18개 모두 가능한
logical key는 79개이고 이에 해당하는 원표 행은 1,422개다. 중복 logical ratio를 물리 실행으로
접으면 43개 key가 남는다. ALoRa는 저채널 pair 제약으로 unavailable이다. PCA_LEGACY는 `q100`,
PaAno는 `q40` 이상, GDN은 `q10` 이상에서만 후보가 남는다.

equal-trial config 수는 Tier 1·2·3 순서로 `1·2·1`이다. target-free 실행을 `r100` 한 벌로 접은
exact panel은 물리 실행 65건, primary logical score 89행이다. TSPulse의 primary score variant는
`raw_max` 하나로 고정했고 `time·fft·pred`는 진단용으로만 남겼다. 예산은 `sealed`, 실행 준비는
`ready`다. 18개 전체 물리 실행은 1,170건이며 완성할 primary logical score 원표는 1,602행이다.

## 활성 모델 실행 전 확인

각 모델은 합성 입력에서만 아래 계약을 먼저 확인한다.

| 모델 | 최소 확인 |
| --- | --- |
| MWVAR | 공식 window 96, `ddof=1`, 중앙 정렬과 길이 `L` |
| SQDIFF_LAST3 | lag 3, trailing 4점, 계수 `4/3`, 길이 `L` |
| PCA_LEGACY | fit-only scaler·PCA, component grid와 길이 `L` |
| PaAno | patch score의 test-only stitching, 세션 경계와 길이 `L` |
| ALoRa | window stitching, fit-only pair 선택, 19·86채널 shape |
| GDN | 공식 활성 경로의 `[B,C,W]` 입력, `L-W` score, `labels[W:]` |
| TimeRCD | multi checkpoint forward, target normalization 비사용, 길이 `L` |
| TSPulse | 공식 time·FFT·prediction raw score, native scaling·smoothing 우회, 길이 `L` |

GDN 검사는 한 구현의 충실도 검사다. 과거 구현과 수치 결과를 맞추는 회귀나 HAI edge 재현은
수행하지 않는다.

## Dev18 실행 전에 닫은 항목

- 본실험은 GHL·HAI, TSB 비-GHL 18개는 사전 튜닝이라는 역할 구분
- 활성 모델 roster와 GDN 단일화 결정
- `Q`, 80:20 prefix 내부 분할, fit-only scaler와 라벨 비개입 원칙
- GHL25 25개와 HAI 두 실행의 평가 범위
- `raw__trainnorm` 주점수와 점수 정렬·metadata 기본 계약
- 모델 담당자, 강혁, 지우, 주혜의 생성 책임과 인수 순서
- 최종 비용 목적함수에 필요한 시간·memory·artifact·관측량 원자료 범위
- 공식 TSB-AD `opt` VUS-PR과 현재 evaluator의 고정 fixture 대조, 최대 절대 오차 0
- Dev18 18개 시계열별 training-only `ℓ_max`와 evaluator·generator SHA-256
- TimeRCD·TSPulse의 Dev18 1,536×2 checkpoint forward, finite·비상수·CPU 결정성
- CPython 3.11.14 `tsad_models_311` 공통 환경과 `requirements.txt`·`pip freeze`
- score manifest의 exact budget key, metadata·score·실행 증거 SHA와 최대 3회 시도
- 실제 Dev18 160×2 MWVAR gate smoke, label 비사용, 4.3초 실행

공식 구현 대조가 현재 evaluator SHA에 묶였으므로 독립 oracle을 별도 하드 게이트로 두지 않는다.
공식 TSB-AD가 비교 기준이며 point adjustment와 test-derived normalization은 이 실행에 들어가지
않는다.

## 현재 남은 항목

- Dev18: clean worktree에서 봉인된 단일 진입 파일로 exact panel 실행
- GHL25·HAI: 각 본실험을 열기 전 Role-A EDA·manifest 최종 승인
- 최종 평가: GHL·HAI score가 생긴 뒤 threshold와 보조 F1 원표 연결

GHL25·HAI 항목은 Dev18 튜닝을 막지 않는다. 다음 시작점은
`tests/ghl_main/run_dev18_tuning.py`를 옵션 없이 실행하는 2단계 exact panel이다.
