# GHL·HAI 데이터 Manifest 초안

행 수는 헤더를 뺀 데이터 행만 센 값이다. 열 수는 CSV에 적힌 모든 열을 포함한다. 이 문서에는 데이터 검수 결과와 최종 판단만 남긴다.

## GHL

- 공식 목록: TheDatumOrg/TSB-AD의 GHL 파일 목록
- 경계 정의: 파일명의 `tr_` 값이 테스트 시작 0-based 인덱스다. 학습은 `[0, tr_)`, 테스트는 `[tr_, 행 수)`로 자른다. `1st_` 값은 전체 파일의 첫 이상 인덱스이며 학습·테스트 경계가 아니다.
- 열 구성: 센서 19열과 `Label` 1열, 합계 20열이다.

| 시계열 | 파일명 | 행 수 | 열 수 | 학습 인덱스 | 테스트 인덱스 | 39,938~50,000 확인 |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 01 | `032_GHL_id_1_Sensor_tr_50000_1st_65001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 02 | `033_GHL_id_2_Sensor_tr_50000_1st_51001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 03 | `034_GHL_id_3_Sensor_tr_50000_1st_122001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 04 | `035_GHL_id_4_Sensor_tr_50000_1st_90001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 05 | `036_GHL_id_5_Sensor_tr_50000_1st_67147.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 06 | `037_GHL_id_6_Sensor_tr_50000_1st_80001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 07 | `038_GHL_id_7_Sensor_tr_50000_1st_100001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 08 | `039_GHL_id_8_Sensor_tr_50000_1st_63030.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 09 | `040_GHL_id_9_Sensor_tr_50000_1st_92001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 10 | `041_GHL_id_10_Sensor_tr_50000_1st_57001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 11 | `042_GHL_id_11_Sensor_tr_50000_1st_150001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 12 | `043_GHL_id_12_Sensor_tr_39938_1st_40038.csv` | 200,001 | 20 | `[0, 39,938)` | `[39,938, 200,001)` | 범위 안(하한) |
| 13 | `044_GHL_id_13_Sensor_tr_50000_1st_145001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 14 | `045_GHL_id_14_Sensor_tr_50000_1st_85076.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 15 | `046_GHL_id_15_Sensor_tr_50000_1st_156462.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 16 | `047_GHL_id_16_Sensor_tr_50000_1st_77001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 17 | `048_GHL_id_17_Sensor_tr_50000_1st_154001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 18 | `049_GHL_id_18_Sensor_tr_50000_1st_109001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 19 | `050_GHL_id_19_Sensor_tr_43750_1st_55001.csv` | 175,001 | 20 | `[0, 43,750)` | `[43,750, 175,001)` | 범위 안 |
| 20 | `051_GHL_id_20_Sensor_tr_50000_1st_75110.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 21 | `052_GHL_id_21_Sensor_tr_50000_1st_98001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 22 | `053_GHL_id_22_Sensor_tr_50000_1st_126448.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 23 | `054_GHL_id_23_Sensor_tr_50000_1st_135001.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 24 | `055_GHL_id_24_Sensor_tr_50000_1st_118124.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |
| 25 | `056_GHL_id_25_Sensor_tr_50000_1st_105568.csv` | 200,001 | 20 | `[0, 50,000)` | `[50,000, 200,001)` | 범위 안 |

범위 밖 시계열은 없다. 학습 길이는 39,938, 43,750, 50,000 세 값이며 최솟값과 최댓값이 계획서 범위의 양 끝과 정확히 맞는다. 원본 25개를 다시 읽어 행·열 수와 파일명 경계를 `experiments/checks/datasets/ghl/logs/inventory.csv`와 대조했다.
원본 inventory 25개의 바이트 크기와 SHA-256은 `configs/input_manifest.yaml`에 고정돼 있다.
최종 GHL 평가 manifest는 09번과 18번을 포함한 25개 전체를 가리킨다. 두 파일은 공식
TSB-AD-M 목록에도 있지만 비-GHL 튜닝 패널에서는 제외하므로 선택·평가 교집합은 0건이다.

## 공식 TSB-AD-M 목록과 비-GHL 튜닝 패널

공식 목록 파일은 `Datasets/File_List/TSB-AD-M-Tuning.csv`다. header를 제외한 20개 파일을
provenance와 SHA 확인용으로 그대로 보존한다. 모델·설정 선택에는 이 목록에서 GHL 두 파일만 뺀
`TSB-AD-M non-GHL development v1` 18개를 쓴다. 이 이름은 기존 인터페이스의 내부 식별자다.
`005`부터 `100`까지 튜닝 파일 목록은 같고,
파일명의 `tr_<N>`이 가리키는 정상 학습 구간에서 모델에 제공하는 앞쪽 prefix 길이만 달라진다.

| 순서 | 파일명 | 상대 경로 | 최종 GHL 역할 |
| ---: | --- | --- | --- |
| 01 | `004_MSL_id_3_Sensor_tr_530_1st_630.csv` | `tuning/` | 해당 없음 |
| 02 | `011_MSL_id_10_Sensor_tr_1525_1st_4590.csv` | `tuning/` | 해당 없음 |
| 03 | `023_MITDB_id_5_Medical_tr_25000_1st_36913.csv` | `tuning/` | 해당 없음 |
| 04 | `028_MITDB_id_10_Medical_tr_37500_1st_39948.csv` | `tuning/` | 해당 없음 |
| 05 | `040_GHL_id_9_Sensor_tr_50000_1st_92001.csv` | `TSB-AD-M/` | GHL25 포함, 튜닝 제외 |
| 06 | `049_GHL_id_18_Sensor_tr_50000_1st_109001.csv` | `TSB-AD-M/` | GHL25 포함, 튜닝 제외 |
| 07 | `062_SMD_id_6_Facility_tr_7180_1st_15131.csv` | `tuning/` | 해당 없음 |
| 08 | `072_SMD_id_16_Facility_tr_7119_1st_15849.csv` | `tuning/` | 해당 없음 |
| 09 | `082_LTDB_id_4_Medical_tr_4456_1st_4556.csv` | `tuning/` | 해당 없음 |
| 10 | `090_SVDB_id_7_Medical_tr_12157_1st_12257.csv` | `tuning/` | 해당 없음 |
| 11 | `107_SVDB_id_24_Medical_tr_32805_1st_32905.csv` | `tuning/` | 해당 없음 |
| 12 | `113_SVDB_id_30_Medical_tr_4552_1st_4652.csv` | `tuning/` | 해당 없음 |
| 13 | `120_TAO_id_5_Environment_tr_500_1st_3.csv` | `tuning/` | 해당 없음 |
| 14 | `126_TAO_id_11_Environment_tr_500_1st_7.csv` | `tuning/` | 해당 없음 |
| 15 | `131_OPPORTUNITY_id_3_HumanActivity_tr_7016_1st_26691.csv` | `tuning/` | 해당 없음 |
| 16 | `140_CATSv2_id_3_Sensor_tr_28307_1st_28407.csv` | `tuning/` | 해당 없음 |
| 17 | `149_SMAP_id_6_Sensor_tr_2128_1st_5000.csv` | `tuning/` | 해당 없음 |
| 18 | `164_SMAP_id_21_Sensor_tr_1976_1st_4200.csv` | `tuning/` | 해당 없음 |
| 19 | `178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv` | `tuning/` | 해당 없음 |
| 20 | `195_Exathlon_id_22_Facility_tr_10766_1st_12590.csv` | `tuning/` | 해당 없음 |

TSB 튜닝 패널은 위 표의 비-GHL 18개, 곧 01~04번과 07~20번으로
고정한다. 비-GHL 파일을 빼거나 다른 파일로 대체하지 않는다. family 구성은 `MSL 2, MITDB 2,
SMD 2, LTDB 1, SVDB 3, TAO 2, OPPORTUNITY 1, CATSv2 1, SMAP 2, Exathlon 2`다.

manifest 계약은 세 가지다. 공식20은 provenance, 비-GHL 18개는 모델·파라미터 선택, GHL25는
최종 평가다. 튜닝 패널과 GHL25의 교집합은 0건이고 공식20과 GHL25의 교집합은 위 GHL 두
파일뿐이다.

### TSB 튜닝 패널 감사 결과

`tuning/`의 18개 실물을 다시 읽어 파일 SHA-256, 행 수, feature 이름과 순서, 수치·유한값,
경계와 라벨 길이를 대조했다. feature 이름과 순서는 파일별 SHA-256으로
`configs/input_manifest.yaml`에 봉인했다. 전체 feature 값에 결측·NaN·Inf·비수치는 없었고
18개 모두 파일명의 첫 이상 위치와 실제 라벨이 맞았다.

| series | family | 행 수 | feature 수 | training | test | training 이상 | ACF 후보 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | MSL | 2,430 | 55 | 530 | 1,900 | 0 | 125 |
| 02 | MSL | 6,100 | 55 | 1,525 | 4,575 | 0 | 194 |
| 03 | MITDB | 100,000 | 2 | 25,000 | 75,000 | 0 | 271 |
| 04 | MITDB | 150,000 | 2 | 37,500 | 112,500 | 0 | 255 |
| 05 | SMD | 28,722 | 38 | 7,180 | 21,542 | 0 | 125 |
| 06 | SMD | 28,479 | 38 | 7,119 | 21,360 | 0 | 125 |
| 07 | LTDB | 100,000 | 2 | 4,456 | 95,544 | 0 | 115 |
| 08 | SVDB | 50,000 | 2 | 12,157 | 37,843 | 0 | 112 |
| 09 | SVDB | 230,400 | 2 | 32,805 | 197,595 | 0 | 105 |
| 10 | SVDB | 230,400 | 2 | 4,552 | 225,848 | 0 | 116 |
| 11 | TAO | 10,000 | 3 | 500 | 9,500 | 48 | 221 |
| 12 | TAO | 10,000 | 3 | 500 | 9,500 | 40 | 125 |
| 13 | OPPORTUNITY | 28,066 | 248 | 7,016 | 21,050 | 0 | 36 |
| 14 | CATSv2 | 400,000 | 17 | 28,307 | 371,693 | 0 | 125 |
| 15 | SMAP | 8,512 | 25 | 2,128 | 6,384 | 0 | 98 |
| 16 | SMAP | 7,907 | 25 | 1,976 | 5,931 | 0 | 125 |
| 17 | Exathlon | 129,197 | 19 | 12,538 | 116,659 | 0 | 125 |
| 18 | Exathlon | 43,066 | 31 | 10,766 | 32,300 | 0 | 6 |

ACF 후보는 최대 20,000개 training 시점의 첫 feature만 쓴다. lag 3~400의 strict local
maximum 가운데 ACF가 가장 큰 lag를 고르며 후보가 없거나 300을 넘으면 125다. 이는 공식
TSB-AD period 규칙에서 test 구간만 뺀 방식이다. 상관 감사는 training prefix의 Pearson 절댓값
0.995 이상을 기록한다. constant·IQR 0·고상관 채널은 삭제하지 않는다.

TAO 11·12번은 원본 `tr_500` 안에 이상 라벨이 48개와 40개 있다. 공식 TSB-AD도 이 prefix를
그대로 쓰므로 라벨로 제거하지 않았다. 모델 입력은 계속 label-free이며 이 오염은 TAO
family leave-one-out 결과와 함께 공개한다. Dev18 감사 snapshot 상태는
`approved_with_disclosed_source_training_contamination`이다.

## HAI

- 기준 버전: `icsdataset/hai`의 HAI 23.05
- 기술문서: HAI 데이터셋 기술문서 PDF 2쪽(문서 1쪽)과 PDF 31쪽(문서 30쪽)
- 획득 검수: 공식 Kaggle 미러의 파일은 LFS 등록 SHA-256과 바이트 크기로 provenance를 대조한다.

| 용도 | 파일명 | 행 수 | 열 수 | 열 구성 | 첫 줄 확인 |
| --- | --- | ---: | ---: | --- | --- |
| 테스트 세션 1 | `hai-test1.csv` | 54,000 | 87 | timestamp 1 + 채널 86 | CSV 실물 |
| 테스트 세션 2 | `hai-test2.csv` | 230,400 | 87 | timestamp 1 + 채널 86 | CSV 실물 |
| 훈련 세션 1 | `hai-train1.csv` | 280,800 | 87 | timestamp 1 + 채널 86 | CSV 실물 |
| 훈련 세션 2 | `hai-train2.csv` | 291,600 | 87 | timestamp 1 + 채널 86 | CSV 실물 |
| 훈련 세션 3 | `hai-train3.csv` | 126,000 | 87 | timestamp 1 + 채널 86 | CSV 실물 |
| 훈련 세션 4 | `hai-train4.csv` | 198,000 | 87 | timestamp 1 + 채널 86 | CSV 실물 |
| 테스트 1 라벨 | `label-test1.csv` | 54,000 | 2 | timestamp 1 + label 1 | CSV 실물 |
| 테스트 2 라벨 | `label-test2.csv` | 230,400 | 2 | timestamp 1 + label 1 | CSV 실물 |

8개 CSV의 첫 줄은 모두 `timestamp,...` 헤더였고 `version https://git-lfs.github.com/spec/v1` 포인터 문구는 없었다. 테스트 파일과 짝이 되는 라벨 파일의 행 수도 각각 54,000과 230,400으로 같다. 훈련은 896,400행, 테스트는 284,400행이다.
inventory로 봉인한 train·test·label 8개 파일의 바이트 크기와 SHA-256은
`configs/input_manifest.yaml`에 고정했다. 본실험은 이 중 train1·train2·test1·test2와 두
라벨 파일만 읽고 train3·train4는 후속 민감도를 위해 보존한다.

### 버전과 시계열 단위 결정

HAI 23.05를 쓴다. HAIEnd 23.05는 같은 실험에서 수집한 225채널 DCS 내부 논리 데이터라
이번 범위와 다르다. HAI 23.05는 86채널이며 각 CSV가 독립된 연속 세션을 나타낸다.

“시계열 하나”는 `hai-train*.csv` 또는 `hai-test*.csv` 한 파일이 나타내는 연속 세션 하나로 센다. 훈련 4개와 테스트 2개, 합계 6개 시계열이다. 데이터 inventory는 여섯 세션을 모두 보존하지만 본실험은 `train1 → test1`과 `train1+train2 → test2` 두 실행만 쓴다. train3·train4를 추가하는 시간순 민감도는 본 결과 뒤로 미룬다. 파일 사이의 시간 간격에는 윈도를 만들지 않는다. 공식 문서가 보장하는 연속성 단위가 각 CSV이기 때문이다.

모델 입력에서 timestamp와 label을 빼면 `N=86`이다. Tier 2 활성 모델은 PaAno, ALoRa,
GDN이다. ALoRa 공식 HAI 설정의 78채널과 현재 86채널 차이는 모델 담당자가 합성 smoke test로
검사한다. GDN은 공식 `d-ailin/GDN` 적응 구현 하나만 쓰며 설정은 첫 튜닝 전에 봉인한다.

### 전처리 결정과 남은 확인

GHL·HAI 모두 다운샘플 배율 1을 쓴다. 정상 학습 구간의 앞쪽 5·10·20·40·60·80·100%를 먼저
자르고, 학습과 validation이 필요한 모델은 현재 prefix 안에서만 시간순으로 다시 나눈다. 전체
정상 구간의 마지막 10%를 낮은 비율에 제공하지 않는다. timestamp와 label은 입력에서 뺀다.
Tier 2의 `MinMaxScaler`는 현재 prefix의 fit 부분에만 맞추고 validation과 평가 구간에는 transform만
적용한다. PCA_LEGACY는 각 window를 행 단위 z-score로 바꾼 뒤 fit window feature에만
StandardScaler와 PCA를 맞춘다. 내부 validation 비율은 80:20으로 고정했다. Dev18 최소 길이
feasibility는 봉인한 shape와 registry로 확정했으며 GHL·HAI 지원 판정은 각 모델 smoke 뒤에 닫는다.

HAI 첫 실행은 train1의 현재 prefix, 두 번째 실행은 train1·train2의 같은 비율 prefix만 쓴다. 각
훈련 세션은 현재 prefix 안에서 독립적으로 80:20으로 나누며, 첫 실행의 scaler는 train1 fit에만,
둘째 실행의 scaler는 train1·train2 fit을 합친 값에만 맞춘다. 세션 경계에는 window와 validation
score를 만들지 않는다. constant·저분산·고상관을 이유로 채널을 삭제하지 않고 86개를 모두 쓴다.
training-free MWVAR·SQDIFF 계열과 strict zero-shot은 비율별 target calibration을 쓰지 않는다.
상세 계약은 `docs/lead/plan_v5.md`와 `docs/lead/process_0_preverify.md`가 관리한다. 과거 HAI
전용 GDN 설정은 폐기됐으며 주실험이나 HPO의 실행 근거가 아니다.

## 인수 상태

Dev18 18개는 파일 신원, shape, feature 순서, 값 품질, 학습 라벨 오염, 채널 품질과 training-only
ACF 후보까지 감사했다. 세 원표와 snapshot의 SHA-256도 현재 manifest·감사 코드와 맞는다. 따라서
Dev18 Role-A 인수는 `approved_with_disclosed_source_training_contamination`으로 닫았고, 정적
feasibility와 공정 예산의 입력으로 쓸 수 있다.

아래 `pending`은 GHL25·HAI 본실험 인수에만 해당한다.

| 항목 | 대상 | 상태 |
| --- | --- | --- |
| feature 이름·순서와 feature 목록 SHA-256 | GHL25, HAI 본실험 세션 | pending |
| numeric·NaN·Inf·결측·중복 timestamp 원표 | GHL25, HAI 본실험 세션 | pending |
| 정상 학습 구간 라벨 오염 확인 | GHL25 | pending |
| constant·IQR 0·고상관 채널 원표 | GHL25, HAI 본실험 세션 | pending |
| 이상 구간 개수와 길이의 최소·중앙·최대 | GHL25, HAI test1·test2 | pending |
| 학습 구간 ACF 후보와 lag 상한 | GHL25, HAI train1·train2 | pending |
| HAI 86개 센서 순서와 세션별 일치 | train1·train2·test1·test2 | pending |

이 표의 `pending`은 GHL·HAI 최종 지원 판정과 본실험 실행을 막는다. 이미 승인한 Dev18 원표와
`equal_trial` 예산을 되돌리지는 않는다.
