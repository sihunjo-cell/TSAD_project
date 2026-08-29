# 계획서 v5 — 제조 현장 cold-start TSAD 도입 의사결정

정상 데이터가 거의 없는 제조 현장에서 어떤 TSAD 계층과 모델을 언제 도입해야 하는지 판단하고,
마지막에는 성능·오탐·미탐·데이터 대기·학습·추론·자원 비용을 함께 넣은 목적함수로 운영안을
고르는 연구 계획이다.

## 0. 목적과 현재 범위

본실험 데이터는 GHL과 HAI 두 종류뿐이다. `TSB-AD-M-Tuning.csv`에서 고른 비-GHL 18개
시계열은 본실험 결과를 주장하는 데이터가 아니라, GHL을 보기 전에 모델과 recipe를 정하는
튜닝 패널이다. 저장 경로와 기존 인터페이스에 남은 `dev18`은 이 튜닝 패널을 가리키는 내부
식별자일 뿐 세 번째 실험 데이터셋을 뜻하지 않는다.

연구 흐름은 아래 한 줄로 고정한다.

```text
강혁 EDA·manifest → 모델 담당자 정적 feasibility·모델 함수·점수
→ 지우 VUS-PR·튜닝 선택 → GHL 주실험 → HAI 외부 확인
→ 주혜 통계·교차점 → 비용 목적함수와 최종 도입안
```

0·1단계는 끝났다. 현재는 2단계 TSB 튜닝의 exact panel을 실행하기 직전이다. 후보, 예산,
VUS-PR, 시계열별 `ℓ_max`, checkpoint와 실행 환경은 봉인했으며 전체 HPO는 아직 돌리지 않았다.

## 1. 최종 연구 질문

정상 데이터 비율 집합은 `Q={005,010,020,040,060,080,100}`이다. `q`는 target 설비의 정상
학습 구간에서 시간순으로 관측한 앞쪽 prefix 비율이다. foundation model의 사전학습 데이터량이나
TSB 파일의 표본 비율이 아니다.

주질문은 세 가지다.

1. 모델과 전처리·학습·정규화 recipe를 고정했을 때 target 정상 데이터가 늘수록 성능은 어떻게
   달라지는가.
2. 경량 통계, target 학습형 신경망, strict zero-shot 가운데 어느 계층이 어느 `q`부터 지속적으로
   우세한가.
3. 같은 성능 차이를 얻기 위해 필요한 데이터 대기, HPO, 재학습, 추론, 메모리와 artifact 비용은
   얼마이며 현장 제약 아래 어떤 행동이 가장 싼가.

GHL이 주된 성능·교차점 근거다. HAI는 제조 현장의 다변량 세션 구조에서도 결론이 유지되는지
확인한다. TSB 튜닝 점수는 모델·recipe 선택에만 쓰며 최종 성능이나 교차점 근거로 사용하지 않는다.

## 2. 데이터의 역할과 격리

### 공식 TSB-AD-M 목록

`Datasets/File_List/TSB-AD-M-Tuning.csv`의 20개 파일은 출처와 SHA-256을 확인하는 원본
목록으로 보존한다. 공식 split을 그대로 재현했다고 주장하지 않는다.

### TSB 비-GHL 튜닝 패널 18개

공식 20개 중 아래 GHL 두 파일을 제외한 18개만 모델·하이퍼파라미터 선택에 쓴다.

- `040_GHL_id_9_Sensor_tr_50000_1st_92001.csv`
- `049_GHL_id_18_Sensor_tr_50000_1st_109001.csv`

나머지 18개 파일과 순서는 바꾸거나 대체하지 않는다. family는 `MSL 2, MITDB 2, SMD 2,
LTDB 1, SVDB 3, TAO 2, OPPORTUNITY 1, CATSv2 1, SMAP 2, Exathlon 2`다. 파일 수가 많은
family가 선택을 좌우하지 않도록 family를 동일 가중한다.

Role-A 감사에서 TAO 두 파일의 공식 `tr_500` prefix에 이상 라벨 48개와 40개가 확인됐다.
공식 TSB-AD 실행도 이 prefix를 그대로 쓰며 라벨로 정제하지 않는다. 패널 구성과 label-blind
입력을 지키려고 두 파일을 보존하고 TAO family leave-one-out 결과를 함께 공개한다. 이 예외를
숨기고 TSB 튜닝의 모든 prefix가 정상이라고 부르지 않는다.

### GHL25

GHL 원본 25개 전체가 주실험이다. TSB 튜닝에서 제외한 GHL 09·18도 다시 포함한다. 튜닝 패널과
GHL25의 파일 교집합은 0건이어야 한다. GHL 라벨·점수·비용을 보기 전에 모델, 후보 recipe,
seed, 평가식, 실패와 동률 규칙을 봉인한다. GHL 결과를 보고 이 선택을 바꾸지 않는다.

### HAI 23.05

HAI는 `train1 → test1`과 `train1+train2 → test2`를 서로 다른 실행으로 유지한다. HAI 라벨로
모델이나 recipe를 고르지 않는다. train3·train4를 추가하는 시간순 민감도는 본 결과가 끝난 뒤
필요할 때만 연다. 한 숫자로 요약해야 할 때는 두 실행의 지표를 같은 가중치로 산술평균하며 행
수로 가중하지 않는다.

## 3. 실험 전에 받아야 할 EDA 근거

강혁의 EDA는 모델 선택이나 threshold를 정하지 않는다. 데이터가 실험 계약을 만족하는지 확인하고
모델 담당자가 정적 feasibility를 계산할 근거를 제공한다.

필수 근거는 다음과 같다.

- 파일별 경로, 행 수, feature 수, feature 이름과 순서, 입력 SHA-256
- 정상 학습 경계, 테스트 경계, 학습 구간 라벨 오염 여부
- 숫자가 아닌 값, NaN, Inf, 결측 비율과 중복 timestamp
- constant·IQR 0 채널과 고상관 채널 현황
- 이상 구간의 개수와 길이 요약, 라벨과 timestamp 행 수 일치
- 학습 구간만으로 계산한 주기성 후보와 lag 상한
- HAI 파일별 독립 세션 여부와 86개 센서 순서
- GHL25, TSB 튜닝 패널, HAI 본실험 파일의 역할 구분

constant·저분산·고상관은 감사 결과로 남길 뿐 채널 삭제 근거로 쓰지 않는다. 주기성은 지우의
`ℓ_max` 사전 결정을 돕는 입력이며, 모델의 window나 최종 성능을 대신 결정하지 않는다. 테스트
라벨 통계는 채점과 난이도 분석에만 넘긴다.

`docs/role_A/manifest.md`가 강혁 인수물의 기준이다. 과거 GHL·HAI Tier 2 EDA 보고서는 당시
형상·경계 점검 근거로만 보존한다. 폐기한 고정 validation과 과거 모델 roster를 현재 실행 근거로
재사용하지 않는다.

## 4. 역할과 인수 순서

| 담당 | 책임 | 넘기는 산출물 | 하지 않는 일 |
| --- | --- | --- | --- |
| 강혁 | GHL·HAI·TSB 튜닝 파일 EDA와 manifest 승인 | 데이터 manifest, 경계·품질·주기성·라벨 통계 | 모델 구현, checkpoint 검증, HPO, VUS-PR |
| 모델 담당자 | Tier 1·2·3 모델과 본실험 함수 설계, 정적 feasibility, 점수·metadata·snapshot·시간 로그 | 튜닝·GHL·HAI score manifest와 실행 증거 | VUS-PR 재구현, 정책 재선택, 통계 검정 |
| 지우 | 공통 VUS-PR, `ℓ_max`, threshold 격자, 튜닝 원표와 선택표 | 튜닝 score ledger, 고정 recipe 표, 최종 실행 요청 | 모델 코드와 runner 수정, 최종 통계 해석 |
| 주혜 | 난이도·hit·통계·교차점·결과표 | GHL·HAI 통계와 교차점, 비용 최적화용 성능 원표 | 모델·recipe·threshold 재선택 |

모델 담당자는 강혁의 manifest가 승인된 뒤 정적 feasibility를 계산한다. 지우는 모델 담당자의
연속 점수와 metadata를 채점하고 선택표를 만든다. 모델 담당자는 지우가 넘긴
`final_policy_membership.csv`만 최종 실행 요청으로 소비하며 지우의 선택식을 runner 안에서 다시
계산하지 않는다. 주혜는 봉인된 선택표와 최종 채점 원표만 분석한다.

지우나 주혜의 코드가 부족해도 모델 담당자가 대신 구현하지 않는다. 필요한 열, 파일과 상태를
문서에 적어 해당 담당자에게 넘긴다.

## 5. 모델 roster와 단일 GDN

| Tier | 의미 | 활성 후보 |
| --- | --- | --- |
| Tier 1 | 경량 통계·저비용 기준선 | `MWVAR`, `SQDIFF_LAST3`, `PCA_LEGACY` |
| Tier 2 | target 정상 prefix로 학습 | `PaAno`, `ALoRa`, `GDN` |
| Tier 3 | target 학습 없는 strict zero-shot | `TimeRCD`, `TSPulse` |

`GDN`은 하나만 쓴다. 공식 `d-ailin/GDN` commit
`9853899da860682669a134e4af315d036aab4eca`를 기준으로 만든
`src/models/tier2/gdn_official/` 경로가 유일한 활성 구현이다. 과거 GraGOD 기반 HAI 전용 구현,
별도 GDN preset, edge/Jaccard 보조 분석은 연구 roster와 산출물 계약에서 제외한다. 두 GDN을
비교하거나 동등성을 검증하지 않는다.

GDN의 필요한 검사는 활성 구현 하나의 공식 동작 충실도에 한정한다. 1-step forecast, `L-W`
점수 길이, `source_start=W`, `labels[W:]` 정렬, fit-only scaler, 라벨·threshold 비개입과 실행
증거가 맞는지만 본다.

`CATCH`는 공식 source의 license와 검증된 실행 경로가 없어 활성 후보에서 제외한다.
`MOMENT_ZS_LEGACY`는 과거 기록일 뿐 현행 모델이 아니다. CrossAD, DADA, CAROTS, ScatterAD도
이번 연구에 추가하지 않는다. 결과를 본 뒤 후보를 늘리지 않는다.

## 6. 비율 안의 분할과 누수 방지

학습이나 정상 validation이 필요한 모델은 각 파일의 현재 `q%` prefix 안에서만 시간순으로 나눈다.

```text
available_count(q)  = floor(N × q / 100)
fit_count(q)        = floor(0.8 × available_count(q))
validation_count(q) = available_count(q) - fit_count(q)
fit(q)              = [0, fit_count(q))
validation(q)       = [fit_count(q), available_count(q))
```

전체 정상 구간의 마지막 10%나 뒤쪽 20%를 모든 비율에 공통으로 주지 않는다. Tier 2의 입력
`MinMaxScaler`는 `fit(q)`에만 맞춘다. score의 median·IQR 교정값은 같은 `q`의 validation raw
score로만 추정한다. fit과 테스트 구간은 교정 통계에 넣지 않는다.

HAI의 각 훈련 파일은 현재 prefix 안에서 따로 80:20으로 나눈다. 첫 실행은 train1 fit에만,
둘째 실행은 train1·train2 fit을 합친 값에만 scaler를 맞춘다. 파일 경계를 넘는 window와
validation score를 만들지 않는다.

training-free Tier 1과 strict zero-shot은 target prefix로 학습·정규화·calibration하지 않는다.
동일한 물리 점수 한 벌을 여러 `q`에 표시할 수 있지만, 100% 데이터에서 설정을 골랐다는 뜻은
아니다. stride와 downsampling은 1이다.

정적 시작점은 `q_floor={t1:5,t2:10,t3:5}`다. 이 값은 성능이 아니라 현재 분할식과 최소 window
조건에서 정했다. 강혁 manifest의 실제 길이·채널 정보로 다시 계산했을 때 모순이 나면 점수를
보기 전에 중단하고 문서 결정을 고친다.

## 7. 정적 feasibility와 모델 함수 완료 조건

정적 feasibility는 점수와 라벨을 읽기 전에 계산한다. 단위는
`(model, config_id, q, tuning_series)`다. prefix·fit·validation·test 길이, window·patch 수,
채널 수, GDN top-k, HAI 다중 세션 지원 여부를 판정한다. 불가능한 조합은 0점으로 채우지 않고
`unavailable`과 이유를 남긴다.

모델 함수는 아래 조건을 모두 만족해야 완료로 센다.

1. 공식 source·license·checkpoint 신원이 snapshot에 고정돼 있다.
2. 합성 입력에서 score shape와 원시 시점 정렬이 맞는다.
3. label, test threshold, point adjustment와 test-derived normalization이 score 생성에 개입하지 않는다.
4. 같은 seed와 입력으로 재현되며 실패·timeout을 성공 점수로 저장하지 않는다.
5. 학습, validation 추론, 테스트 추론, 전처리 시간과 peak memory, artifact 크기를 남긴다.
6. GHL 19채널과 HAI 86채널, HAI 세션 경계를 지원하는지 모델별로 판정한다.

이 검사는 합성 자료와 정적 정보로 먼저 닫는다. 실제 TSB·GHL·HAI 점수 생성은 다음 실행 게이트가
열린 뒤 시작한다. 공식값은 후보 grid의 출발점이지 GHL·HAI 최적값이라고 부르지 않는다.

## 8. TSB 기반 튜닝

첫 튜닝 점수 전에 후보 roster, model별 config 순서, seed, `Q`, 분할, VUS-PR 구현,
`ℓ_max`, 실패와 동률 규칙을 봉인한다. stochastic 모델의 튜닝 seed는 `{0,1,2}`다.

primary HPO는 `equal_trial`이다. 정적 feasibility를 통과한 뒤 Tier 안 각 모델에 같은 수의
full-fidelity trial을 배정하고 exact config 목록과 순서를 budget manifest에 고정한다. 일부
시계열·일부 epoch로 예선하지 않는다. `runtime_matched`는 실제 timing 근거가 모인 뒤 필요한 경우에만
민감도로 추가하며 primary 선택을 바꾸는 사후 장치로 쓰지 않는다.

비율별 adaptive 정책은 이 panel의 실패나 지연을 대신할 fallback이 아니다. 새 config, 축소 배치,
수동 모델 실행으로 exact panel을 우회하지 않는다.

Tier 대표 후보는 해당 Tier의 `q_floor` 이상 모든 주분석 비율에서 공통 recipe가 있고 GHL25와
HAI 두 실행을 정적으로 지원해야 한다. 조건을 만족하는 후보가 없으면 해당 Tier를
`unavailable`로 보고하며 `q_floor`를 사후에 올리지 않는다. 모델별 고정 곡선은 각 모델이 실제로
지원하는 비율과 split에서 계속 보존한다.

지우는 먼저 seed 평균을 내고 10개 family를 같은 가중치로 평균한다.

```text
z(i,m,h,q) = mean_seed VUSPR(i,m,h,q,seed)
J(m,h,q)   = (1/10) × sum_family mean_{i in family} z(i,m,h,q)
```

모델 선택은 family leave-one-out 바깥 검증으로 한다. 각 holdout family마다 나머지 9개 family에서
recipe를 고르고 holdout 점수만 모아 모델 점수 `S(m)`을 만든다. 모델을 고른 뒤 같은 봉인 예산으로
18개 전체에서 고정 recipe 한 벌을 정한다. 점수 차이가 `1e-6` 이내면
`(model, config_id, score_variant)` 사전순으로 고른다. 계산비는 동률 처리에 쓰지 않는다.

주분석에 필요한 표는 두 개다.

- `model_fixed_policy.csv`: 활성 모델마다 지원 `q` 전체에 쓸 recipe 한 벌
- `tier_fixed_policy.csv`: Tier별 대표 모델과 그 고정 recipe

비율별 재튜닝과 Tier 내부 모델 교체는 같은 튜닝 원표에서 만드는 선택적 민감도다. 이 표들이
없어도 GHL·HAI 주실험과 최종 비용 최적화를 진행할 수 있어야 한다. 별도 HPO를 다시 돌리지 않는다.

## 9. GHL과 HAI 본실험

지우가 선택표와 `final_policy_membership.csv`를 봉인한 뒤에만 본실험을 연다. 모델 runner는
membership의 runnable 행만 실행하며 임의로 모델·recipe·비율을 추가하지 않는다.

GHL25에서는 모든 활성 모델의 `model_fixed` 곡선을 보존한다. Tier 대표 세 개만 남기면 비용
목적함수의 후보가 사라지기 때문이다. `tier_fixed` 곡선은 계층별 학습곡선과 교차점의 주분석이다.
stochastic 모델의 GHL seed는 `{3,4,5,6,7}`, deterministic 모델은 한 번 실행한다.

HAI는 두 실행을 따로 저장한다. 두 실행에서 모델과 recipe는 GHL 결과를 보지 않고 TSB 튜닝에서
고정한 값을 쓴다. 다중 세션을 안전하게 처리하지 못하면 해당 행을 `unavailable`로 남기며 세션을
이어 붙이지 않는다.

adaptive 곡선은 운영자가 매 비율마다 재튜닝하거나 모델을 교체할 수 있다는 별도 가정을 둔
민감도다. 데이터 양만의 효과로 해석하지 않는다.

## 10. 점수와 실행 증거

주지표 입력은 threshold 전 연속 `raw__trainnorm` 점수다. `smoothed`와 `testnorm`은 민감도다.
채널 점수가 있으면 정규화 후 채널 `max`로 집계하며, 공식 scalar score에는 가짜 채널 축을 만들지
않는다. smoothing은 후행 4칸 평균, 처음 3점 0으로 고정한다.

```text
{dataset}__{series}__{model}__{tier}__r{ratio}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}.npy
```

각 점수에는 `config_id`, 입력·source·checkpoint SHA-256, split, `q`, seed, source 범위,
정렬 방식, normalization 범위, label slice, 실행 상태와 재시도 수를 연결한다. `config_id`는 모델,
source commit, source checkpoint, hyperparameters와 고정 전처리 recipe로 만들며 dataset, series,
`q`, seed를 넣지 않는다.

비용 원자료는 아래 항목을 분리한다.

- `split_preprocess_seconds`
- `model_setup_seconds`
- `training_seconds`
- `validation_inference_seconds`
- `test_inference_seconds`
- `peak_memory_mb`
- `model_artifact_bytes`
- 세션별 `observation_count`, 검증 가능한 `observed_duration_seconds`, `duration_basis`
- `status`, `retry_count`, 실패 이유

Dev18 실행 증거의 `measurement_protocol_id`는 `dev18_registered_runner.v2`다. 이 값은 한
in-memory 등록 executor 안에서 split·전처리, 모델 setup, 학습과 추론을 잰 범위를 뜻하며, 다섯 timing 값의
합이 `runtime_seconds`와 같아야 한다.

TSB 튜닝 HPO 비용과 현장 재학습 비용은 같은 숫자로 합치지 않는다. GHL은 timestamp 근거가
없으면 관측 개수를 초 단위로 바꾸지 않는다. HAI도 timestamp 간격을 검증한 경우에만 관측 지속시간을
쓴다.

## 11. 채점·통계·교차점

지우는 VUS-PR과 보조 지표를 같은 정렬 계약으로 계산한다. threshold 250개는 VUS 적분 격자이며
현장 경보 threshold가 아니다. 운영 threshold는 비용 목적함수에서 오탐·미탐 비용과 함께 정한다.
`ℓ_max`는 강혁의 학습 구간 주기성 근거를 받고 테스트 성능을 보기 전에 고정한다.

GHL 주분석은 시계열 `1/25` macro다. 같은 GHL 시계열의 일곱 `q`를 독립 표본처럼 세지 않는다.
주혜는 paired bootstrap, Wilcoxon, TOST와 지속 교차점을 봉인된 GHL 원표에 적용한다. bootstrap은
같은 시계열을 모델·Tier·`q` 전반에서 함께 재표집한다. provenance 의존성이 확인되면 cluster
단위 결과를 우선한다.

```text
D_a,b(q) = (1/25) × sum_GHL [VUSPR(a,q) - VUSPR(b,q)]
c*_a,b   = min {p in Q : 모든 관측 q >= p에서 D_a,b(q) > 0}
```

관측 비율 사이를 연속 임계값으로 보간하지 않는다. 한 지점만 앞선 경우는 일시적 역전으로,
뒤의 모든 관측점에서 유지될 때만 지속 교차로 기록한다. HAI 두 실행은 먼저 따로 보고하며 GHL과
합쳐 하나의 표본처럼 검정하지 않는다. 별도 GDN 그래프 안정성 분석은 하지 않는다.

## 12. 최종 비용 목적함수

본실험은 비용함수를 미리 최적화하지 않는다. 먼저 모든 활성 모델의 고정-recipe 성능과 비용
원자료를 누수 없이 만든다. 마지막 단계의 행동 단위는 아래처럼 정의한다.

```text
a = (tier, model, config_id, q, operating_threshold)
```

GHL과 HAI의 `split`은 행동이 아니라 성능·비용 근거가 나온 평가 문맥이다. 목적함수는 현장
시나리오마다 두 데이터셋의 근거와 불확실성을 연결해 계산한다.

목적함수는 현장 가중치가 정해진 뒤 구성한다.

```text
TotalCost(a)
  = ObservationCost(q)
  + DeploymentTrainingCost(a)
  + ExpectedInferenceCost(a)
  + MemoryAndArtifactCost(a)
  + FalseAlarmCost(a)
  + MissCost(a)
```

HPO 비용은 모델을 채택하기 전의 개발비로 따로 보고한다. 반복되는 현장 재학습·추론비와 섞지
않는다. `ObservationCost(q)`에는 정상 데이터를 모으는 시간이나 생산 지연이 들어가지만, 실제
시간 근거가 없는 GHL에서는 관측량 대리값으로만 남긴다. `FalseAlarmCost`와 `MissCost`는
threshold별 confusion 결과와 현장 단가가 모두 있어야 계산한다.

최종 선택은 두 단계로 한다. 먼저 성능, 정상 데이터 요구량, 재학습·추론 시간, 메모리와 artifact의
Pareto 열위를 제거한다. 그다음 현장 비용 가중치와 latency·memory·최소 성능 제약을 넣어
`argmin_a TotalCost(a)`를 고른다. 가중치가 달라질 때 선택이 바뀌는 구간도 함께 보고한다.

## 13. 최소 산출물 계약

필수 산출물만 주실험의 차단 조건으로 둔다.

| 생성자 | 필수 산출물 | 용도 |
| --- | --- | --- |
| 강혁 | `docs/role_A/manifest.md`와 EDA 인수표 | 정적 feasibility의 입력 |
| 모델 담당자 | `dev18_feasibility_ledger.csv` | TSB 튜닝 가능 조합 봉인 |
| 모델 담당자 | `dev18_score_manifest.csv` | TSB 튜닝 점수와 실행 증거 |
| 지우 | `dev18_trial_score_ledger.csv` | VUS-PR 튜닝 원표 |
| 지우 | `model_fixed_policy.csv`, `tier_fixed_policy.csv` | 고정 recipe와 Tier 대표 |
| 지우 | `final_policy_membership.csv` | GHL·HAI 물리 실행 요청 |
| 모델 담당자 | `ghl25_score_manifest.csv`, `hai_score_manifest.csv` | 최종 점수와 비용 원자료 |
| 지우 | `ghl25_score_ledger.csv`, `hai_score_ledger.csv` | 최종 채점 원표 |
| 주혜 | 통계·교차점·난이도 결과 | 비용 최적화의 성능 입력 |

물리 runner는 `final_policy_membership.csv`와 registry·config SHA만 확인한다. 다섯 정책표와 두
ledger를 한꺼번에 읽는 8-file bundle은 필수 실행 계약에서 제거한다. 선택식 검증은 지우의
평가 단계가 맡고 runner는 모델 실행만 맡는다. 같은 물리 점수를 여러 분석이 쓰면 복사하지 않고
ledger가 SHA-256으로 참조한다.

## 14. 프로젝트 단계와 현재 게이트

- 0단계: 문서·역할을 정리하고 강혁의 EDA·manifest 인수 조건을 닫는다.
- 1단계: 모델 담당자가 정적 feasibility, 공식 구현 충실도, 합성 smoke와 실행 증거 연결을 닫고
  `equal_trial` budget을 봉인한다.
- 2단계: TSB 비-GHL 18개로 HPO·채점을 수행하고 고정 recipe와 Tier 대표를 봉인한다.
- 3단계: GHL25 주실험과 통계·교차점 원표를 만든다.
- 4단계: HAI 두 실행으로 외부 확인을 마친다.
- 5단계: 성능·비용 원자료에 현장 가중치와 제약을 결합해 최종 도입안을 고른다.

0단계의 강혁 Dev18 인수와 1단계의 정적 증거를 닫았다. 현재 게이트는 2단계 exact panel 직전이다.
다만 새 project commit에서 TimeRCD Dev18 checkpoint smoke, TSPulse batch 1 대 등록 batch 32의
실측 동등성, non-interruptible L4의 80% 자원 보고서를 모두 다시 통과하기 전에는 panel을 시작하지
않는다. 이 동등성이 확인될 때에만 기존 `c...` config 행과 `budget_id=b5367ad431093`을 그대로
유지한다. GHL25·HAI의 최종 Role-A 인수는 3·4단계 시작 전에 따로 닫으며 Dev18 튜닝의 선행 조건으로
되돌리지 않는다.

## 15. 중단과 완료 규칙

- manifest의 길이·feature·경계·SHA가 실제 입력과 맞지 않으면 실험을 시작하지 않는다.
- 선택에 쓸 후보나 HPO 예산이 결과를 본 뒤 바뀌면 TSB 튜닝부터 새 버전으로 다시 봉인한다.
- GHL이나 HAI 결과를 보고 모델·recipe·`ℓ_max`·전처리를 고치지 않는다.
- 점수 정렬, 라벨 비개입, fit-only 통계, 실행 증거 중 하나라도 깨지면 영향받은 실행을 폐기한다.
- 실패 조합을 0점으로 바꾸거나 조용히 제외하지 않는다.
- 비용 목적함수는 성능 원표, threshold별 오탐·미탐, 비용 단가와 운영 제약이 모두 승인된 뒤 연다.

모델 담당자의 역할은 모든 활성 모델과 본실험 함수가 이 계약대로 점수를 만들고, 비용 최적화에
필요한 실행 원자료까지 빠짐없이 넘기는 데서 닫힌다. 최종 정책 선택식과 통계 결과는 각 담당자의
산출물을 입력으로 받는다.

## 16. 참고 자료

- TSB-AD-M tuning 목록: https://github.com/TheDatumOrg/TSB-AD/blob/main/Datasets/File_List/TSB-AD-M-Tuning.csv
- TSB-AD 논문: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html
- Learning curves: https://doi.org/10.1109/TPAMI.2022.3220744
- Dataset-size effects: https://proceedings.mlr.press/v139/hoiem21a.html
- HPO 평가 원칙: https://doi.org/10.1002/widm.1484
- Selection bias: https://www.jmlr.org/papers/v11/cawley10a.html
- TSB-AutoAD: https://www.vldb.org/pvldb/vol18/p4364-liu.pdf
- One-Liners: https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/publications/iclr2026_timeseriesfoundationmodelsad
- PaAno: https://github.com/jinnnju/PaAno
- ALoRa: https://github.com/CharisShimillas/ALoRa
- GDN: https://github.com/d-ailin/GDN
- Time-RCD: https://github.com/thu-sail-lab/Time-RCD
- TSPulse: https://github.com/ibm-granite/granite-tsfm
