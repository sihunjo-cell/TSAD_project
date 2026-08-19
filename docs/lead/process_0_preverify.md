# 사전 점검 결과

이 문서는 실험 전에 확인한 문제와 현재 조치만 모은다. 조사 과정과 일회성 진단 코드는
보존하지 않는다.

## 현재 상태

저는 계층 1·2·3 모델과 점수 산출을 맡는다. 계층 2는 파라미터 EDA와 네 모델 adapter,
GHL·HAI runner, 합성 실행 검증까지 마쳤다. 실데이터 학습은 아직 하지 않았다. 강혁의 최종
검토가 현재 판단을 바꾸면 영향받는 부분만 다시 계산한다.

지우의 채점기와 ℓ_max, 주혜의 난이도·통계 코드는 대신 만들지 않았다.

## 확인한 문제와 조치

| 점검 대상 | 확인한 내용 | 현재 조치 |
| --- | --- | --- |
| GDN 입력 | GraGOD의 `reshape`가 batch와 node 축의 값 순서를 섞었다. | 해당 두 줄을 `permute`로 고치고 수정본을 GDN 코드 옆에 뒀다. |
| 점수 후처리 | GraGOD smoothing이 feature 축에 작용하고 마지막 점수를 버렸다. | 후행 4칸 smoothing과 `L-W` 정렬을 이 프로젝트에서 처리한다. |
| 분할과 정규화 | 비율별 누적 구간마다 validation을 다시 잘라 검증 구간까지 달라졌다. | 전체 정상 구간의 마지막 10%를 고정 validation으로 먼저 분리한다. 남은 fit pool의 누적 비율만 학습과 scaler fit에 쓰고 validation·test에는 transform만 적용한다. |
| 비율별 데이터 감사 | 과거 누적 구간을 전체 학습 구간의 통계와 비교해 이후 관측치가 기준에 들어갔다. | 전체 구간과의 대표성 계산과 해당 CSV 생성을 없앴다. 각 누적 구간 안의 표본 수·채널 변화·분산만 검사한다. |
| GHL 입력 | 25개 시계열의 학습 길이는 39,938·43,750·50,000이고 센서는 19개다. | 5·10·20·40·60·80·100%의 공통 누적 격자를 쓴다. 관계 구조 회복이나 최적 비율로 해석하지 않는다. |
| HAI 입력 | HAI 23.05는 훈련 4세션, 테스트 2세션, 센서 86개다. | 파일 경계를 보존하고 7개 공통 비율을 쓴다. `topk=22`는 본실험 전 프로젝트 전이값으로 고정했다. |
| 계층 2 입력 길이 | 고정 lag 6개의 자기상관 중앙값으로 모델 입력 길이 60·120을 정할 근거가 없었다. | 논문과 채택 구현을 우선해 CI-AE 100, LSTM-AD 100, USAD 10, GDN 5로 고정했다. EDA는 비율별 계산 가능성만 검사한다. |
| 센서 열 순서 | GHL 25개는 센서 이름 집합이 같지만 10개 파일에서 두 온도 센서의 열 위치가 바뀐다. | 원본 차이는 감사표에 남기고 센서명 기준 공통 순서로 재배열한다. HAI 6세션의 순서는 같았다. |
| 계층 2 원본 코드 | CI-AE scaler 축, LSTM-AD·USAD의 내부 분할과 split별 정규화가 공통 전처리 규칙과 충돌했다. CI-AE의 최저 손실 가중치도 복사본이 아니라 마지막 epoch 상태를 가리켰다. 공식 USAD와 TSB-AD 포팅은 구조와 점수식이 다르다. | 원 wrapper는 쓰지 않는다. CI-AE·LSTM-AD는 TSB-AD core만, USAD는 지정한 공식 core만 부르는 얇은 adapter를 만들었다. 분할·MinMax·점수 저장은 프로젝트 규칙이 맡는다. |
| 점수 정렬 | CI-AE·USAD·LSTM-AD 원 wrapper는 앞뒤 점수를 복제해 길이를 `L`로 만든다. | 복제 padding을 버리고 실제 core만 원시 시점 범위와 함께 저장하는 규칙으로 고정했다. |
| 적용 가능성 | 두 데이터셋의 NaN·Inf는 0개였고 세션·fit 경계를 넘는 window도 0개였다. | EDA 결과는 그대로 보존했다. 공식 USAD 구조를 포함한 네 adapter를 합성 데이터로 학습·복원·추론해 점수 길이와 source 범위를 다시 확인했다. |
| GDN 설정 | 논문, 공식 실행 코드와 GraGOD의 값이 일부 달랐다. | 로컬 adapter가 attention dropout 0, output dropout 0.2를 분리한다. scheduler와 gradient clipping은 쓰지 않으며 topk는 GHL 5·HAI 22로 고정했다. |
| GDN 실행 import | 배치 entrypoint가 삭제된 소문자 GDN 경로를 import했다. | 모든 실행 경로를 `src/models/tier2/GDN/`으로 통일하고 외부 GraGOD fork 의존성을 없앴다. |
| 실행 흐름 | snapshot부터 점수 저장까지 실제 호출 형태를 확인할 필요가 있었다. | 네 모델의 checkpoint 복원, 점수 8벌, 전체 metadata와 완료 판정을 합성 입력으로 확인했다. HAI형 GDN은 train 4세션·test 2세션과 인접행렬 2벌까지 검사했다. |

GHL의 5% fit subset은 475개 파일·채널 조합 중 246개가 constant했고 IQR이 0인 조합은
279개였다. 전체 fit pool에서는 constant가 0개, IQR이 0인 조합이 150개였다. 고정
validation은 전 비율에서 같은 36개 constant와 221개 IQR=0 조합을 유지했다. 5%는 충분한
학습량이 아니라 저데이터 경계를 보는 조건이다.

HAI에서 훈련 4세션을 채널별로 합친 pooled 기준 변화 채널은 5%와 10%에서 60개,
20% 이상에서 66개였다. 5% fit subset의 훈련 세션·채널 344개 조합 중 constant는 130개, IQR=0은
139개였다. 이 수치는 학습 충분성을 뜻하지 않는다.

고정한 window로 유효 학습·validation window가 0개인 조합은 없었다. HAI GDN의 합산
epoch당 batch는 10%에서 2,521개, 100%에서 25,211개다. GHL CI-AE는 센서별 모델 19개의
작업량을 합산했다. 표의 최대 update는 조기 종료가 없을 때의 상한이며 실제값은 실행 로그로
확정한다. USAD workload는 본실험 전 고정한 batch 128로 계산한다.

USAD batch와 GDN topk의 출처 선택 대기는 0개다. 두 EDA manifest는 같은 run ID를 쓰고
생성 파일 SHA-256 대조를 통과했다. 당시 manifest에 남긴 adapter 항목은 이번 구현과 합성
검증으로 닫았으며, 기존 EDA 산출물은 다시 만들지 않았다. `tsad_fixed`의 환경 버전 계약도 맞았다.

## 남긴 것

반복할 가치가 있는 데이터 감사와 Tier 2 합성 dry-run만 `tests/checks/`에 남겼다. dry-run의
원시 점수와 checkpoint는 시스템 임시 폴더에서 검증 후 지우고, 결과 요약만
`experiments/checks/tier2_implementation/implementation_manifest.json`에 남긴다. GraGOD 입력 축 수정은
`src/models/tier2/GDN/model.py`에 보존한다. GHL 실행 결과는
`experiments/01_ghl_main/`, HAI 실행 결과는 `experiments/02_hai_extension/`에 저장된다.

GHL·HAI 데이터 감사는 수정한 코드로 다시 실행해 로그와 snapshot을 갱신했다. 전체 구간을
기준으로 삼던 대표성 CSV는 제거했다. 계층 2 EDA는 사전 고정한 모델 입력 길이만 검사하며
`vus_l_max`는 계산하지 않는다.

문제를 찾는 데만 쓴 진단 코드와 중복 검증 문서는 삭제했다.

## 실데이터 실행 전 조건

계층 2 실데이터 학습 전에는 이번 변경을 commit해 작업 트리를 clean 상태로 만들고, 연구실
서버의 accelerator와 `tsad_fixed` 환경 계약을 확인한다. 서버에서도 합성 dry-run을 한 번
통과한 뒤 본실험 runner를 시작한다.

지우의 채점기와 주혜의 난이도·통계 절차는 계층 2 모델 구현의 종료 조건이 아니다. 모델 점수를
최종 지표와 통계 결과로 합칠 때 필요하므로, 해당 산출물이 오기 전에는 최종 통합 실험만 멈춘다.
