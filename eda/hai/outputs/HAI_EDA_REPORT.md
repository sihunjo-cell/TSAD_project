# HAI-23.05 파일 기반 EDA 보고서

생성 시각: 2026-08-21T00:28:42+09:00

## 분석 범위

이 보고서는 현재 제공된 HAI-23.05 파일에서 직접 확인할 수 있는 구조, 품질, 시간축, 라벨, 통계적 패턴만 다룬다. 논문의 공정 설명이나 공격 시나리오 의미는 사용하지 않았다. 라벨이 없는 네 개의 학습 파일은 정상이라고 단정하지 않고 `train_unlabeled`로 표기했다.

## 데이터 구조

- 데이터 파일: 6개 (학습 4개, 테스트 2개)
- 특성: 86개, timestamp: 1개
- 전체 관측치: 1,180,800개 (학습 896,400, 테스트 284,400)
- subsystem prefix별 특성 수: {"P1": 37, "P2": 24, "P3": 7, "P4": 11, "x1001": 2, "x1002": 2, "x1003": 3}

| source_file | split | n_rows | start_time | end_time | duration |
| --- | --- | --- | --- | --- | --- |
| hai-train1.csv | train_unlabeled | 280800 | 2022-08-04 18:00:01 | 2022-08-08 00:00:00 | 3d 05:59:59 |
| hai-test1.csv | test | 54000 | 2022-08-12 16:00:01 | 2022-08-13 07:00:00 | 14:59:59 |
| hai-train2.csv | train_unlabeled | 291600 | 2022-08-13 07:00:01 | 2022-08-16 16:00:00 | 3d 08:59:59 |
| hai-test2.csv | test | 230400 | 2022-08-17 00:00:01 | 2022-08-19 16:00:00 | 2d 15:59:59 |
| hai-train3.csv | train_unlabeled | 126000 | 2022-08-19 16:00:01 | 2022-08-21 03:00:00 | 1d 10:59:59 |
| hai-train4.csv | train_unlabeled | 198000 | 2022-08-22 17:00:01 | 2022-08-25 00:00:00 | 2d 06:59:59 |

- 모든 파일 내부는 정확한 1초 간격이며 시간 범위가 서로 겹치지 않는다.
- 파일 사이 미관측 구간: hai-train1.csv → hai-test1.csv: 112시간, hai-train2.csv → hai-test2.csv: 8시간, hai-train3.csv → hai-train4.csv: 38시간
- `hai-test1` → `hai-train2`, `hai-test2` → `hai-train3` 경계는 1초 차이로 연속된다.

## 데이터 품질

- 특성 결측값: 0개
- 특성 무한값: 0개
- 중복 timestamp: 0개
- 파일 내부 완전 중복 행: 0개
- 최빈 샘플링 간격과 다른 인접 구간: 0개
- cardinality 분류: {"high_cardinality": 55, "constant": 18, "binary": 8, "medium_cardinality": 4, "low_cardinality": 1}
- 전체 기간 동안 값이 변하지 않는 18개 특성: P1_PIT01_HH, P1_PP01AD, P1_PP01AR, P1_PP01BD, P1_PP01BR, P1_PP02D, P1_PP02R, P1_SOL01D, P1_SOL03D, P1_STSP, P2_RTR, P2_TripEx, P2_VTR01, P2_VTR02, P2_VTR03, P2_VTR04, P3_LH01, P3_LL01
- 학습 파일에서 값이 변하지 않는 특성은 20개다. 그중 학습에서만 고정된 특성은 P2_Emerg, P2_OnOff이며, 테스트 이상 라벨 구간에서는 P2_Emerg: 이상 라벨 구간 94행에서 학습 고정값과 다름, P2_OnOff: 이상 라벨 구간 115행에서 학습 고정값과 다름.

## 테스트 라벨 정합성

| test_file | n_rows | positive_points | positive_rate | actual_event_count | label_timestamp_alignment_mode | all_summary_table_events_match | summary_header_aggregate_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hai-test1.csv | 54000 | 2981 | 0.0552037 | 14 | exact_timestamp_positional | True | True |
| hai-test2.csv | 230400 | 8403 | 0.0364714 | 38 | minute_floor_timestamp_positional | True | False |

- 전체 이상 포인트: 11,384개 / 284,400개 (4.003%)
- 연속 이상 이벤트: 52개
- 이벤트 span: 최소 55초, 중앙값 132초, 최대 2051초
- `label-test2.csv`의 timestamp는 초 정보가 제거되어 230,400행 중 고유 timestamp가 3,841개뿐이다. 다만 `hai-test2.csv` timestamp를 분 단위로 내린 값과 전 행이 순서대로 일치하므로 원본 test timestamp를 유지한 positional alignment를 사용했다.
- `summary_label2.txt`의 이벤트 표 38개는 CSV와 모두 일치하지만, 상단 집계의 공격 수 40회와 총 span 02:25:35는 실제 연속 이벤트 38개와 총 span 02:19:25에 일치하지 않는다.
- 각 이벤트의 `끝-시작` span 합은 양 끝을 포함한 양성 포인트 수보다 이벤트당 1초씩 짧다. 모델 평가용 클래스 비율은 CSV의 양성 포인트 수를 기준으로 해야 한다.
- 연속 이벤트 사이 정상 포인트 수는 최소 1790, 중앙값 3918, 최대 50867개다.

## 정상 라벨 구간과 이상 라벨 구간의 차이가 큰 특성

각 test 파일 안에서 정상/이상 라벨의 전체 행을 사용해 KS 통계를 계산한 뒤, 두 test 파일을 동일 가중 평균했다. 따라서 test 파일 자체의 분포 차이가 이상 효과로 섞이는 문제를 줄였다. 인과관계나 공격 대상 변수를 뜻하지 않는다.

| feature | mean_within_test_ks | min_within_test_ks | max_within_test_ks | mean_within_test_abs_mean_shift_sd |
| --- | --- | --- | --- | --- |
| P1_FCV01Z | 0.293917 | 0.174515 | 0.413318 | 0.520672 |
| P1_PCV01Z | 0.293327 | 0.167424 | 0.41923 | 1.61803 |
| P1_FT02 | 0.292388 | 0.174439 | 0.410337 | 0.438442 |
| P1_FT02Z | 0.290405 | 0.169369 | 0.411442 | 0.476801 |
| x1003_10_SETPOINT_OUT | 0.290027 | 0.169382 | 0.410673 | 0.47692 |
| P1_FCV01D | 0.289884 | 0.171401 | 0.408367 | 0.521013 |
| P1_FCV02Z | 0.289772 | 0.149186 | 0.430357 | 0.473797 |
| P1_PCV01D | 0.286512 | 0.170168 | 0.402856 | 1.63362 |
| P1_TIT02 | 0.284382 | 0.154787 | 0.413977 | 0.487204 |
| P1_TIT01 | 0.266991 | 0.209669 | 0.324313 | 0.600294 |

파일별 상위 특성은 다음과 같아 두 test 파일의 이상 패턴이 완전히 같지는 않음을 보여준다.

| source_file | feature | ks_statistic | absolute_mean_difference_in_test_normal_sd |
| --- | --- | --- | --- |
| hai-test1.csv | P1_FCV02Z | 0.430357 | 0.880655 |
| hai-test1.csv | P1_TIT03 | 0.420289 | 0.781845 |
| hai-test1.csv | P1_PCV01Z | 0.41923 | 2.1764 |
| hai-test1.csv | P1_TIT02 | 0.413977 | 0.894538 |
| hai-test1.csv | P1_FCV01Z | 0.413318 | 0.634693 |
| hai-test2.csv | x1003_24_SUM_OUT | 0.264521 | 0.566577 |
| hai-test2.csv | x1003_18_SETPOINT_OUT | 0.257165 | 0.614686 |
| hai-test2.csv | P2_ManualSD | 0.228125 | 0.650561 |
| hai-test2.csv | P4_ST_PS | 0.219027 | 0.354589 |
| hai-test2.csv | x1001_05_SETPOINT_OUT | 0.213993 | 0.235218 |

파일 내부의 표준화 평균 이동량만 보면 다음 특성이 크다. 정상 라벨 구간에서 분산이 0인데 이상 구간에서 값이 변한 상태 변수(P2_OnOff, P2_Emerg)는 무한한 표준화 값이 되므로 이 순위에서 제외했다.

| feature | mean_within_test_abs_mean_shift_sd | mean_within_test_ks |
| --- | --- | --- |
| P1_PCV02D | 8.7995 | 0.0630877 |
| P1_PCV02Z | 4.09661 | 0.119373 |
| P2_VT01 | 1.77675 | 0.0298478 |
| P1_PCV01D | 1.63362 | 0.286512 |
| P1_PCV01Z | 1.61803 | 0.293327 |

## 학습 파일과 테스트 정상 라벨 구간의 분포 차이

아래 순위는 파일 행 수의 2%를 기본으로 하되 파일당 1,000–5,000행으로 제한한 결정론적 표본의 KS 통계 기준이며, 전체 행 평균의 차이를 train-unlabelled 표준편차 단위로 함께 표시한다. 이후 모델 실험의 데이터 분할을 결정한 결과가 아니다.

| feature | ks_statistic | absolute_mean_difference_in_train_unlabeled_sd |
| --- | --- | --- |
| P1_PCV02Z | 0.371712 | 0.236399 |
| P2_VIBTR04 | 0.324299 | 0.78184 |
| P1_FCV03Z | 0.317533 | 0.588355 |
| x1003_18_SETPOINT_OUT | 0.307966 | 0.395965 |
| P1_FCV03D | 0.307325 | 0.590906 |
| x1002_07_SETPOINT_OUT | 0.290417 | 0.467049 |
| x1002_08_SETPOINT_OUT | 0.28483 | 0.402751 |
| P2_ManualSD | 0.283274 | 0.0161457 |
| P2_AutoSD | 0.249193 | 0.30912 |
| P1_PP04SP | 0.248276 | 0.153411 |

파일별 평균이 크게 달라지는 특성은 다음과 같다. 값은 여섯 파일 평균의 최대–최소 범위를 pooled train-unlabelled 표준편차로 환산한 탐색 지표다.

| feature | file_mean_range_in_train_sd |
| --- | --- |
| P2_VIBTR04 | 1.86585 |
| P2_VIBTR03 | 1.73298 |
| x1003_18_SETPOINT_OUT | 1.73297 |
| P1_PP04SP | 1.63806 |
| x1003_24_SUM_OUT | 1.6327 |
| P4_HT_PS | 1.62612 |
| P1_PCV02Z | 1.53773 |
| P2_ManualSD | 1.38319 |
| x1002_08_SETPOINT_OUT | 1.3814 |
| P1_FT03Z | 1.34336 |

## 이벤트 반응 및 복귀의 탐색적 지표

반응 지연은 이벤트 직전 300초 median에서 file-normal SD의 3배를 벗어난 상태가 3초 지속되는 최초 시점이다. 이벤트 직전부터 이미 임계값을 벗어난 경우는 `preexisting_excursion`으로 분리했다. file-normal SD가 0이면 직전 300초 mode에서 3초 연속 달라지는 규칙을 사용했다. 복귀는 반응한 변수에 한해 이벤트 종료 후 local baseline의 1 SD 이내가 10초 지속되는 최초 시점이며, 다음 이벤트 전 또는 최대 600초까지만 검색했다.

| feature | eligible_events | detected_responses | response_rate | median_reaction_delay_seconds | recovery_rate_among_responses | median_recovery_delay_seconds | preexisting_excursion_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P3_FIT01 | 49 | 20 | 0.408163 | 72 | 1 | 5.5 | 3 |
| P1_FT01Z | 52 | 19 | 0.365385 | 70 | 1 | 8 | 0 |
| P1_PCV01D | 52 | 17 | 0.326923 | 71 | 1 | 29 | 0 |
| P1_FT01 | 52 | 17 | 0.326923 | 73 | 1 | 6 | 0 |
| P1_PCV01Z | 52 | 15 | 0.288462 | 54 | 1 | 39 | 0 |
| P1_PIT01 | 52 | 15 | 0.288462 | 85 | 1 | 80 | 0 |
| P4_HT_FD | 52 | 14 | 0.269231 | 82.5 | 1 | 1 | 0 |
| P4_ST_PT01 | 52 | 14 | 0.269231 | 87.5 | 1 | 81 | 0 |
| P1_LCV01D | 51 | 12 | 0.235294 | 74 | 1 | 93.5 | 1 |
| P1_LCV01Z | 51 | 11 | 0.215686 | 79 | 1 | 102 | 1 |

이 지표는 큰 수준 변화를 찾는 운영적 정의다. 라벨 경계가 물리적 반응 경계라는 보장이 없고, slope나 분산만 변하는 반응은 놓칠 수 있으므로 공격 메커니즘으로 해석하지 않는다.

## prefix별 파일 기반 패턴

| prefix | feature_count | global_constant_count | train_constant_count | median_within_test_anomaly_ks | max_within_test_anomaly_ks | median_train_test_normal_sample_ks |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | 37 | 10 | 10 | 0.188838 | 0.293917 | 0.0592107 |
| P2 | 24 | 6 | 8 | 0.0351786 | 0.134381 | 0.0326681 |
| P3 | 7 | 2 | 2 | 0.0533641 | 0.18404 | 0.0473449 |
| P4 | 11 | 0 | 0 | 0.0617196 | 0.265081 | 0.0219054 |
| x1001 | 2 | 0 | 0 | 0.0973448 | 0.15203 | 0.164848 |
| x1002 | 2 | 0 | 0 | 0.118717 | 0.138866 | 0.287624 |
| x1003 | 3 | 0 | 0 | 0.151147 | 0.290027 | 0.217871 |

## 다변량 및 시간 특성

- 상관행렬은 학습 파일에서 고정된 20개 특성을 제외한 66개 변동 특성으로 계산했다.
- 학습 표본에서 |Pearson r| ≥ 0.95인 특성 쌍: 40개
- PCA 사용 가능 특성: 66개
- PCA 설명분산: PC1 13.52%, PC2 12.62%
- 1, 10, 60, 300 sample lag 자기상관은 `tables/autocorrelation_train.csv`에 기록했다.

각 test 파일 내부에서 정상 표본 상관과 이상 전체행 상관의 차이를 계산했다. 변화가 큰 특성 쌍은 다음과 같으며, 이는 인과관계가 아니라 라벨 구간별 선형 관계 변화다.

| feature_1 | feature_2 | mean_absolute_correlation_change | max_absolute_correlation_change | contributing_test_files |
| --- | --- | --- | --- | --- |
| P2_MASW | P2_SIT01 | 0.776487 | 0.776487 | 1 |
| P2_MASW_Lamp | P2_SIT01 | 0.776487 | 0.776487 | 1 |
| P2_ManualGO | P2_SIT01 | 0.776487 | 0.776487 | 1 |
| P2_AutoGO | P2_SIT01 | 0.776487 | 0.776487 | 1 |
| P2_ATSW_Lamp | P2_SIT01 | 0.776487 | 0.776487 | 1 |
| P2_MASW_Lamp | P2_SCO | 0.679963 | 0.679963 | 1 |
| P2_ManualGO | P2_SCO | 0.679963 | 0.679963 | 1 |
| P2_MASW | P2_SCO | 0.679963 | 0.679963 | 1 |
| P2_ATSW_Lamp | P2_SCO | 0.679963 | 0.679963 | 1 |
| P2_AutoGO | P2_SCO | 0.679963 | 0.679963 | 1 |

`contributing_test_files=1`인 쌍은 다른 test 파일에서 한 변수의 분산이 0이라 상관을 정의할 수 없었던 경우이므로 두 파일에 공통된 변화로 일반화하지 않는다.

## 해석 시 주의사항

- EDA의 그룹 차이는 연관성이다. 공격 원인이나 센서의 물리적 의미로 해석하지 않는다.
- 학습 파일에는 제공된 라벨이 없으므로 파일 기반 EDA만으로 정상성을 검증할 수 없다.
- test 파일 내부 정상/이상 KS는 전체 행으로 계산했다. train/test drift의 KS, 분위수, 상관관계 및 PCA는 행 수의 2%를 기본으로 파일당 1,000–5,000행으로 제한한 결정론적 표본을 사용했고 평균·표준편차·결측 통계는 전체 행으로 계산했다.
- KS p-value는 강한 자기상관과 이산형 변수 때문에 독립·동일분포 및 연속분포 가정을 충족하지 않을 수 있으므로 유의성 추론에 사용하지 않고 서술적 참고값으로만 기록했다.
- 시간 순서를 보존했으며 서로 다른 파일 경계에 걸쳐 lag 또는 이벤트를 연결하지 않았다.
- 학습 데이터 5–100% 부분집합 구성과 모델링은 이 EDA에 포함하지 않았다.

## 산출물

- `tables/file_summary.csv`: 파일별 크기와 시간축 품질
- `tables/feature_catalog.csv`: 컬럼 prefix와 cardinality 분류
- `tables/feature_file_statistics.csv`: 파일·특성별 기술통계
- `tables/feature_group_statistics.csv`: 학습/테스트 정상/테스트 이상 전체행 통계
- `tables/label_validation.csv`: 테스트와 라벨 timestamp 및 summary 검증
- `tables/anomaly_events.csv`: 연속 이상 이벤트 목록
- `tables/anomaly_feature_effect.csv`: test 파일 내부 전수 KS의 macro 요약
- `tables/anomaly_feature_effect_by_test.csv`: test 파일별 전수 정상/이상 비교
- `tables/anomaly_feature_effect_pooled.csv`: 보조적인 pooled 표본 비교
- `tables/train_vs_test_normal_drift.csv`: 학습/테스트 정상 분포 차이
- `tables/event_feature_effects.csv`: 이벤트별 전·중·후 특성 평균
- `tables/event_response_metrics.csv`: 명시적 임계값 기준 이벤트·특성별 반응/복귀
- `tables/event_response_summary.csv`: 특성별 반응/복귀 요약
- `tables/autocorrelation_train.csv`: 파일 경계를 보존한 자기상관
- `tables/state_transitions.csv`: 저 cardinality 변수 상태 전환
- `tables/train_sample_correlation.csv`: 학습 표본 상관행렬
- `tables/high_correlation_pairs.csv`: 고상관 특성 쌍
- `tables/correlation_change_by_test.csv`: test 파일별 정상/이상 상관 변화
- `tables/correlation_change_macro.csv`: 상관 변화의 test 파일 macro 요약
- `tables/file_feature_mean_shift.csv`: 파일별 평균의 학습 기준 표준화 차이
- `tables/feature_prefix_summary.csv`: prefix·cardinality별 특성 수
- `tables/prefix_statistics.csv`: prefix별 품질·이상 차이·분포 변화 요약
- `tables/pca_projection_sample.csv`: PCA 시각화용 표본 좌표
- `run_metadata.json`: 실행 옵션과 입력 파일 기록
- `figures/`: 핵심 시각화
