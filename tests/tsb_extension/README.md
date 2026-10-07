# TSB-AD-M 외부 확인: 평가 기준 사전 등록

2026-10-08에 sbatch 제출 전에 적고 commit했다. 아래 기준·지표·판정 방식은 결과를 본 뒤 바꾸지 않는다.
계산은 `evaluate_tsb_service.py`가 한다.

## 전제: 서비스와 평가를 나눈다

- 서비스는 새 현장에서 모델을 실측하지 않는다. 정상 데이터 prefix와 Dev18 DB만으로 모델·경로를 추천하고, 성능과 비용을
  모두 추정한다.
- 백테스트에서 대상 파일의 실측(VUS-PR, 실행 시간)은 추천을 채점하는 답안지로만 쓴다.

웹사이트 입력에 대상 파일 실측이 들어가는지 점검했다(2026-10-08, `tests/ghl_main/evaluate_ghl_service.py`의
`evaluate_file`). 들어가는 곳은 없다.

- `plan_input`은 학습 구간 prefix(`frame.iloc[:ratio * boundary // 100]`)의 센서 열만 `build_ml_input`에 넘긴다.
  Label 열은 빠지고, `build_ml_input`도 센서 열만 복사한다.
- `run_ml_pipeline`이 읽는 `recommendation.sqlite3`에는 Dev18 10개 family·18개 series만 있다. GHL·TSB 행은 없고,
  DB SHA는 점검 결과의 identity에 남는다.
- 후보 풀은 센서 수와 DB로만 정한다. 유사도 없음 기준선의 예측도 DB의 같은 비율 평균이다.
- 실측(`actual`)은 웹사이트가 고른 뒤의 실현값 계산(`simulate`), 무작위 기대값(`random_choice`), 사후 최적에만 쓴다.
- 참고로, 예산 배수의 기준인 "가장 싼 계획 비용"은 실측이 있는 후보로 푼 계획에서 잰다. 비용 값은 웹사이트
  추정이고 VUS-PR은 들어가지 않지만, 대상 파일에서 실행에 실패한 후보는 이 기준 계산에서 빠진다.
- 운영 조건의 추론량은 test 구간 길이로 정한다. 운영 조건이고 라벨이나 성능이 아니다.

## 파일 선택 규칙

TSB-AD-M 원본 목록 200개(TheDatumOrg/TSB-AD `Datasets/File_List/TSB-AD-M.csv`) 중 Dev18 밖 family이면서
파일명 도메인 태그가 Sensor·Facility인 파일 전부를 쓴다. GHL은 주실험이라 뺀다. 지문은
`configs/tsb_extension_manifest.yaml`에 있다.

| series | family | 파일 | 센서 | 학습 구간 행 | 전체 행 |
| --- | --- | --- | ---: | ---: | ---: |
| 01 | Genesis | 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv | 18 | 4,055 | 16,220 |
| 02 | SWaT | 171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv | 66 | 3,749 | 14,996 |
| 03 | GECCO | 173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv | 9 | 16,165 | 138,521 |
| 04 | PSM | 115_PSM_id_1_Facility_tr_50000_1st_129872.csv | 25 | 50,000 | 217,624 |
| 05 | SWaT | 172_SWaT_id_2_Sensor_tr_23700_1st_23800.csv | 51 | 23,700 | 399,919 |

규칙에 걸리지 않은 Dev18 밖 family는 CreditCard(Finance)·Daphnet(HumanActivity)다.

## 실행

- 후보는 웹사이트가 그 파일의 센서 수에서 계획 후보로 두는 설정 전부다. 학습형 후보는 계획이 학습하는 비율(5~80%)만 돈다.
- PaAno·GDN은 Genesis·SWaT id_1·GECCO(series 01~03)만 seed 3·4·5로 돌려 답안지를 안정시키고, PSM·SWaT id_2는
  seed 3 하나다. 결정론 모델은 seed 3 하나다. 제출은 `sbatch tests/tsb_extension/run_tsb_candidates.sbatch --seeds 3 4 5`다.
  GPU 시간이 없으면 `--seeds 3`으로 제출하고 그 사실을 보고한다. seed는 제출할 때 정하고, 요약을 본 뒤 더 돌리지 않는다.
- 주 판정은 제출한 seed의 요약(`experiments/03_tsb_extension/results/service_backtest/seeds_3_4_5/`)으로 한다.
  GHL은 seed 3 하나로 돌렸으므로, TSB와 GHL을 견주는 수치는 모든 파일을 seed 3으로 본 요약(`seeds_3/`)에서 읽는다.

## 1. 기준선과 주 지표

- 비교 대상은 실측 없이 고를 수 있는 기준선인 cheapest(가장 싼 계획), random(비율마다 무작위 후보),
  no_similarity(유사도 없이 Dev18 평균으로 고른 계획)다. dp는 참고로만 본다.
- oracle(사후 최적)은 "모든 후보를 실측했을 때"의 상한이다. 기준선이 아니다.
- **주 지표는 사후 최적 격차 회수율** (전략 − cheapest) / (oracle − cheapest)다. 실측 없이 실측의 가치를 몇 % 얻었는지를
  뜻한다. website·no_similarity·random 모두 계산한다(random은 예산을 보지 않아 제한 없음에서만).
  - 계산 단위는 family다. family 값은 그 family 파일들의 실현 VUS-PR 평균이고, 회수율은 family 안의 비라 데이터
    사이 VUS-PR 수준 차이에 영향받지 않는다. oracle과 cheapest가 같은 family는 회수율이 없다.
  - family 사이 요약은 중앙값이다.
- oracle은 PaAno·GDN 단일 seed(01~03은 3 seed 평균)의 우연한 고점을 사후에 고를 수 있어 높게 나올 수 있다.
  그래서 회수율은 보수적인 값이다.
- 보조 지표는 승/무/패와 우연 대비 배수(family 평균 VUS-PR / family 평균 우연 수준)의 중앙값이다. 우연 대비 배수의
  평균은 우연 수준이 낮은 파일·family에 끌려가므로 쓰지 않는다.

## 2. 판별력 기준 (평가자만 쓰는 채점 규칙, 서비스 기능 아님)

새 현장에서는 실측이 없어 판별력을 알 수 없다. 이 기준은 답안지가 웹사이트 추천을 가릴 수 있는 파일인지를
평가자가 정하는 데만 쓴다.

- 우연 수준: 균등 난수 점수의 VUS-PR을 실측 채점과 같은 test 구간·ℓ_max·threshold 250개로 잰 값(seed 5개 평균).
- 판별력: 파일의 (후보, 비율) 실측 중 우연 수준의 2배를 넘는 몫. (후보, 비율)은 그 비율에서 웹사이트가 제안한
  학습 선택지 중 실측이 있는 것이다(`predictions.csv` 행). `files.csv`에 남긴다.
- 판별력 10% 이상이면 "판별 가능", 아니면 "판별 불가"다.
- 주 판정은 판별 가능 파일만으로 한다. family는 판별 가능 파일이 하나라도 있으면 들어가고, 그 값은 판별 가능 파일만
  평균한다. 모든 파일의 결과는 `family_summary.csv`의 `scope=all`과 파일 단위 산출물에 그대로 보고한다.
- GHL도 같은 기준으로 같은 표에 넣는다. 기존 GHL 기록으로 미리 계산하면 파일별 0.0~2.4%(전체 0.95%)라 10개 모두
  판별 불가다. 이 값은 이 문서를 쓰기 전에 이미 본 GHL 결과다.

## 3. 주 판정

- 단위는 family다. Genesis, SWaT(두 파일 평균), GECCO, PSM, GHL(10개 파일을 family 1개로)이다.
- 주 비교는 셋이다.
  1. 예산 제한 없음: website 대 no_similarity
  2. 예산 100배: website 대 no_similarity
  3. 예산 제한 없음: website 대 random
- 판정: 비교마다 판별 가능 family 중 website의 family 값이 더 높은 수로 단측 부호 검정(website가 이길 확률 > 1/2)을
  한다. 값이 같은 family(무승부)는 검정에서 빼고 승/무/패에 적는다.
- 미리 적어 두는 한계: family가 3·4·5개이면 전부 이겨도 p는 0.125·0.0625·0.031이다. GHL은 판별 불가라 TSB에서
  판별 가능 family는 많아야 4개이고, TSB만으로는 p < 0.05에 닿을 수 없다.
- 보조: (파일, 비율)별 후보 순위 Spearman ρ의 유사도 − 유사도 없음 차이(평균)와, 예측 1위 후보의 실제 순위(1이 최고,
  중앙값). `ranking.csv`와 `verdict.json`에 있다.
- 파일 단위 Wilcoxon p(`summary.csv`)는 참고로만 본다.

## 4. 비용 예측 평가

서비스는 비용도 실측 없이 Dev18 DB의 실행 기록으로 추정한다. 이 추정이 후보의 비용 순서를 맞히는지 본다.

- 실측: 실행마다 `.meta.json`의 `execution_evidence`에 있는 `timing.training_seconds`·`timing.test_inference_seconds`·
  `runtime_seconds`. seed가 여럿이면 평균한다.
- 추정: 웹사이트 비용 곡선을 실측 실행과 같은 작업량(학습 = 그 비율의 prefix 행, 추론 = test 구간 행, 센서 수 반영)에
  넣은 값.
- (파일, 비율)마다 Spearman ρ를 잰다. 학습은 학습형 후보만, 추론은 모든 후보, 총 실행은 학습 + 추론 추정을
  `runtime_seconds`와 견준다. CPU 후보와 GPU 후보(DB의 측정 장치)는 나눈다. 실행 장치가 DB와 달라 절대값은 견주지
  않는다. head는 한 실행을 나눠 쓰므로 설정마다 한 번 센다.
- GHL 기존 기록(새 실행 없음, 10개 파일 × 6개 비율 = 60묶음)의 결과다. 모든 묶음에서 ρ > 0이다. GHL에는 학습하는
  CPU 후보가 없다.

  | 장치 | 비용 | 후보 수 | ρ 중앙값 | ρ 평균 |
  | --- | --- | ---: | ---: | ---: |
  | CPU | 추론 | 19 | 0.676 | 0.671 |
  | CPU | 총 실행 | 19 | 0.701 | 0.699 |
  | GPU | 학습 | 8~11 | 0.779 | 0.722 |
  | GPU | 추론 | 12~15 | 0.971 | 0.969 |
  | GPU | 총 실행 | 12~15 | 0.937 | 0.920 |

- TSB는 같은 계산을 `cost_ranks.csv`와 `status.json`의 `cost_ranks`에 남긴다. GHL은
  `experiments/01_ghl_main/results/service_backtest/cost_ranks.csv`와 보고서 표 9에 있다.

## 산출물

`experiments/03_tsb_extension/results/service_backtest/seeds_*/`

| 파일 | 내용 |
| --- | --- |
| `verdict.json` | 주 비교 셋의 family 승/무/패·단측 부호 검정 p, family 중앙값(우연 대비 배수·회수율), 순위 보조 지표 |
| `files.csv` | 파일별 우연 수준·판별력·판별 가능 여부·순위 보조 지표(GHL 포함) |
| `family_summary.csv` | family·예산·전략별 VUS-PR·우연 대비 배수·회수율(`scope`: judgeable/all) |
| `ranking.csv` | (파일, 비율)별 Spearman ρ와 예측 1위 후보의 실제 순위 |
| `cost_ranks.csv` | (파일, 비율, 장치, 비용)별 비용 추정 대 실측 시간 Spearman ρ |
| `plans.csv`·`predictions.csv`·`summary.csv`·`status.json` | 파일 단위 원값과 실행 상태 |

## 다음 순서

HAI는 빼지 않는다. 대상 도메인(제조)에 가장 가깝고, family 수가 늘어야 판정이 가능하다. TSB 다음에 같은 기준으로 돈다.
