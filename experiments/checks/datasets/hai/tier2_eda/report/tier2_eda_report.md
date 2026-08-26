# HAI 계층 2 적용 가능성 점검

> 보관 표지: 이 보고서는 당시 입력과 구현을 점검한 감사 근거다. 현재 모델 선택이나 실행 계약이
> 아니며, 현행 기준은 [연구 계획](../../../../../../docs/lead/plan_v5.md)과
> [사전 점검](../../../../../../docs/lead/process_0_preverify.md)을 따른다.

이 EDA는 폐기된 고정 설정안의 입력·구현 적합성을 확인한 기록이다. 아래 수치와 표는 당시 계산을
재현하는 감사 근거일 뿐, 현행 후보·recipe·분할·실행 규칙이 아니다. 테스트 이상 성능은 읽지
않았고 모델도 학습하지 않았다.

## 당시 감사 가정 — 현행 실험에 사용하지 않음

당시 계산은 전체 정상 학습 구간의 마지막 10%를 고정 validation으로 한 번만 떼었다. 이 분할은
폐기했다. 현행 계약은 각 `q` prefix 안에서 `floor(qN) → 80:20`으로 fit과 validation을 나눈다.

당시에는 학습 비율 외의 설정을 고정하고 유효 window 수와 batch·update 수만 계산했다. 이 가정은
현행 Tier 대표나 recipe를 정하지 않는다. 현행 주분석은 TSB 비-GHL 튜닝 패널 18개에서
모델·recipe·score variant를
함께 고른 뒤 봉인하고, HAI에서는 다시 선택하지 않는다.

당시 설정 충돌은 논문의 데이터셋별 실험값, 공식 config·notebook, 공식 CLI 기본값, TSB-AD
기본값, 프로젝트 임의값 순으로 판단했다.

GDN topk=22는 당시 HAI 성능을 보기 전에 둔 전이값이며 공식 HAI 설정값이 아니다. 현행
선택값으로 사용하지 않는다.

## 당시 입력 window 기록 — 현행 선택값 아님

| 모델 | 입력 window |
| --- | --- |
| GDN | 5 |

이 값은 GHL test 결과로 고른 최적값이 아니다. 당시 HAI GDN은 GHL과 모델 정의를 맞추려고
SWaT·WADI 논문 실험값 5를 사용했다. 출처별 차이는 `model_window_sources.csv`, 전체
파라미터·값·선정 이유는 `model_parameter_sources.csv`에 남겼다.

## 당시 가져온 코드의 정적 의미

| 모델 | forward 입력 | 학습 단위 | 전처리 상태 | core 점수 대응 | drop_last |
| --- | --- | --- | --- | --- | --- |
| GDN | [B,86,5] | 1-step graph forecast | 외부 fit-only MinMax 사용 | labels[5:] | False |

입력 형상은 모두 적용 가능했다. 다만 원본 wrapper를 그대로 실행해도 된다는 뜻은 아니다. GDN은 내부 scaling을 하지 않아 프로젝트의 외부 fit-only MinMax 규칙을 그대로 적용한다. 이 항목은 모델을 고치는 대신 실행 전 adapter·구현 검증 게이트로 남겼다.

학습 손실과 이상 점수는 구분한다. 각 모델의 학습 손실은 출처 구현을 따르되, 저장 점수는 프로젝트 공통 규칙인 채널별 절대오차를 쓴다. GDN은 1-step 대상의 채널별 절대오차를 쓴다. 그다음 학습·validation 오차로 robust normalization을 맞춘 뒤 채널 max를 취한다. 이 규칙은 모든 비율과 seed에서 같다.

| 모델 | 비율 고정 항목 수 | 출처 선택 대기 | adapter·불일치 행 |
| --- | --- | --- | --- |
| GDN | 20 | 없음 | 2 |

## 데이터 무결성

- 파일·세션: 6개, 센서: 86개
- NaN: 0, Inf: 0
- timestamp 중복: 0, 역전: 0, 1초 간격 이탈: 0
- 센서 이름 집합: 모두 일치
- 원본 열 순서가 달라 공통 순서로 재배열한 파일: 0개

HAI는 실제 timestamp로 중복·역전·1초 간격 이탈을 확인했고 여섯 세션의 센서 순서가 같았다. 파일별 수치는 `data_integrity.csv`에 있다.

## 범위별 채널 상태

| 비율 | 범위 | constant | zero range | IQR=0 | variance=0 | scaler 실행 준비 미충족 | 변화 있음 | NaN | Inf | 전체 조합 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5% | fit_pool | 82 | 82 | 106 | 82 | 0 | 262 | 0 | 0 | 344 |
| 5% | fit_subset | 130 | 130 | 139 | 130 | 0 | 214 | 0 | 0 | 344 |
| 5% | fixed_validation | 119 | 119 | 129 | 119 | 0 | 225 | 0 | 0 | 344 |
| 10% | fit_pool | 82 | 82 | 106 | 82 | 0 | 262 | 0 | 0 | 344 |
| 10% | fit_subset | 124 | 124 | 132 | 124 | 0 | 220 | 0 | 0 | 344 |
| 10% | fixed_validation | 119 | 119 | 129 | 119 | 0 | 225 | 0 | 0 | 344 |
| 20% | fit_pool | 82 | 82 | 106 | 82 | 0 | 262 | 0 | 0 | 344 |
| 20% | fit_subset | 105 | 105 | 124 | 105 | 0 | 239 | 0 | 0 | 344 |
| 20% | fixed_validation | 119 | 119 | 129 | 119 | 0 | 225 | 0 | 0 | 344 |
| 40% | fit_pool | 82 | 82 | 106 | 82 | 0 | 262 | 0 | 0 | 344 |
| 40% | fit_subset | 100 | 100 | 117 | 100 | 0 | 244 | 0 | 0 | 344 |
| 40% | fixed_validation | 119 | 119 | 129 | 119 | 0 | 225 | 0 | 0 | 344 |
| 60% | fit_pool | 82 | 82 | 106 | 82 | 0 | 262 | 0 | 0 | 344 |
| 60% | fit_subset | 86 | 86 | 110 | 86 | 0 | 258 | 0 | 0 | 344 |
| 60% | fixed_validation | 119 | 119 | 129 | 119 | 0 | 225 | 0 | 0 | 344 |
| 80% | fit_pool | 82 | 82 | 106 | 82 | 0 | 262 | 0 | 0 | 344 |
| 80% | fit_subset | 84 | 84 | 108 | 84 | 0 | 260 | 0 | 0 | 344 |
| 80% | fixed_validation | 119 | 119 | 129 | 119 | 0 | 225 | 0 | 0 | 344 |
| 100% | fit_pool | 82 | 82 | 106 | 82 | 0 | 262 | 0 | 0 | 344 |
| 100% | fit_subset | 82 | 82 | 106 | 82 | 0 | 262 | 0 | 0 | 344 |
| 100% | fixed_validation | 119 | 119 | 129 | 119 | 0 | 225 | 0 | 0 | 344 |

constant는 한 파일·세션의 한 채널이 해당 범위에서 한 값으로만 유지됐다는 뜻이다. zero range는 최솟값과 최댓값이 같다는 뜻이고, IQR=0은 중앙 50%가 같은 값이라는 뜻이라 각각 따로 센다. constant와 zero range는 정보 변화가 없다는 경고일 뿐이다. scikit-learn MinMaxScaler는 이런 열도 오류 없이 처리한다. 값이 없거나 NaN·Inf가 있을 때만 scaler 실행 준비 미충족으로 판정한다. fit pool, 비율별 fit subset, 고정 validation은 섞지 않았다. 채널별 min·max·variance·IQR은 `channel_scaling_readiness.csv`에 있다.


### HAI constant 집계 단위

| 비율 | 범위 | 훈련 세션·채널 조합 | 세션별 constant 채널 | pooled constant 채널 |
| --- | --- | --- | --- | --- |
| 5% | fit_pool | 82 | train_1=20;train_2=20;train_3=22;train_4=20 | 20 |
| 5% | fit_subset | 130 | train_1=33;train_2=29;train_3=33;train_4=35 | 26 |
| 5% | fixed_validation | 119 | train_1=29;train_2=28;train_3=31;train_4=31 | 25 |
| 10% | fit_pool | 82 | train_1=20;train_2=20;train_3=22;train_4=20 | 20 |
| 10% | fit_subset | 124 | train_1=30;train_2=28;train_3=31;train_4=35 | 26 |
| 10% | fixed_validation | 119 | train_1=29;train_2=28;train_3=31;train_4=31 | 25 |
| 20% | fit_pool | 82 | train_1=20;train_2=20;train_3=22;train_4=20 | 20 |
| 20% | fit_subset | 105 | train_1=21;train_2=22;train_3=31;train_4=31 | 20 |
| 20% | fixed_validation | 119 | train_1=29;train_2=28;train_3=31;train_4=31 | 25 |
| 40% | fit_pool | 82 | train_1=20;train_2=20;train_3=22;train_4=20 | 20 |
| 40% | fit_subset | 100 | train_1=21;train_2=22;train_3=30;train_4=27 | 20 |
| 40% | fixed_validation | 119 | train_1=29;train_2=28;train_3=31;train_4=31 | 25 |
| 60% | fit_pool | 82 | train_1=20;train_2=20;train_3=22;train_4=20 | 20 |
| 60% | fit_subset | 86 | train_1=21;train_2=22;train_3=22;train_4=21 | 20 |
| 60% | fixed_validation | 119 | train_1=29;train_2=28;train_3=31;train_4=31 | 25 |
| 80% | fit_pool | 82 | train_1=20;train_2=20;train_3=22;train_4=20 | 20 |
| 80% | fit_subset | 84 | train_1=21;train_2=20;train_3=22;train_4=21 | 20 |
| 80% | fixed_validation | 119 | train_1=29;train_2=28;train_3=31;train_4=31 | 25 |
| 100% | fit_pool | 82 | train_1=20;train_2=20;train_3=22;train_4=20 | 20 |
| 100% | fit_subset | 82 | train_1=20;train_2=20;train_3=22;train_4=20 | 20 |
| 100% | fixed_validation | 119 | train_1=29;train_2=28;train_3=31;train_4=31 | 25 |

각 행의 전체 조합 수 344는 `훈련 세션 4개 × 채널 86개`다. pooled 값은 같은 비율과 범위에서 네 세션의 값을 채널별로 합쳐 다시 판정한 채널 수다.


## 유효 window와 계산량

| 비율 | 모델 | 학습 window | validation window | epoch당 update | 최대 update |
| --- | --- | --- | --- | --- | --- |
| 5% | GDN | 40,318 | 89,620 | 1,260 | 63,000 |
| 10% | GDN | 80,656 | 89,620 | 2,521 | 126,050 |
| 20% | GDN | 161,332 | 89,620 | 5,042 | 252,100 |
| 40% | GDN | 322,684 | 89,620 | 10,084 | 504,200 |
| 60% | GDN | 484,036 | 89,620 | 15,127 | 756,350 |
| 80% | GDN | 645,388 | 89,620 | 20,169 | 1,008,450 |
| 100% | GDN | 806,740 | 89,620 | 25,211 | 1,260,550 |

이 표는 폐기된 훈련 4세션 계산의 기록이다. 당시에는 네 세션의 window를 합쳐 batch를 셌지만
세션 사이에는 window를 만들지 않았다. 현행 HAI 본실험은 `train1 → test1`과
`train1+train2 → test2` 두 실행만 사용하므로 이 workload를 가져오지 않는다. 최대 update는 당시
조기 종료가 없다고 본 상한이다.

## 점수와 원 시간축 정렬

| 모델 | W | 첫 core 대응 index | 저장 길이 | 원 wrapper 좌/우 padding | 판정 |
| --- | --- | --- | --- | --- | --- |
| GDN | 5 | 5 | L-W | 0/0 | 통과 |

본 실험 산출물은 원 wrapper의 복제 padding을 저장하지 않는다. GDN의 1-step 예측 core는 각 테스트 세션의 `labels[5:]`와 실제 timestamp에 대응한다.

## 과거 합성 형상 확인 기록

| 모델 | 입력 | 실제 출력 | 현재 저장소 core | 선정 출처 core |
| --- | --- | --- | --- | --- |
| GDN | [2, 86, 5] | [[2, 86]] | 통과 | 통과 |

합성 batch 2개로 모델 core의 forward 형상만 확인한 과거 기록이다. optimizer step, checkpoint,
실데이터 학습은 포함하지 않았다. HAI GDN은 현재 저장소 core와 선정 출처의 형상이 같았다.

## 모델별 적용 판정

| 모델 | 판정 | 남은 조건 |
| --- | --- | --- |
| GDN | 조건부 적용 가능 | adapter_required, source_implementation_mismatch |

## 완료 기준 판정

| 항목 | 상태 | 비고 |
| --- | --- | --- |
| 파라미터·출처 표 | 작성 | 출처 선택 대기 없음 |
| 비율별 동일 설정 | 과거 계산 | 폐기된 고정 validation 조건의 fit subset 길이만 바꿈 |
| 데이터 무결성 | 통과 | NaN 0, Inf 0 |
| window·batch·update 수 | 과거 계산 | 폐기된 GDN batch=32·topk=22 조건으로 산출 |
| 경계 침범 | 통과 | 실제 침범 0개 |
| 점수·timestamp 정렬 | 통과 | padding하지 않은 core 기준 |
| 합성 형상 확인 | 통과 | adapter·불일치 행 2개 |

당시 파라미터·데이터 적용 가능성 감사는 여기서 닫았다. 이 보고서는 현행 모델·recipe 선택이나
실제 학습을 승인하지 않는다. 남은 항목은 현행 사전 점검에서 별도로 판정한다.

## 이번 EDA에서 하지 않은 일

ACF peak, ADF, Ljung–Box, 고정 lag 자기상관으로 모델 window를 다시 고르지 않았다. test anomaly 성능도 쓰지 않았다. `vus_l_max`와 threshold는 지우 담당 평가 규칙이므로 계산하지 않았다.
