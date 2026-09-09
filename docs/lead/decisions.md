# 현재 결정

갱신일: 2026-09-10

## 자원 사용률 경고와 기존 실행 재개

RAM 총량이 8,192바이트 늘어난 것만으로 74% 실행의 재개를 막은 조건을 제거한다. 저장된
측정값·PCA 추정 근거는 당시 용량을 기준으로 검증하며 원본 보고서는 고쳐 쓰지 않는다.
사용자 지시로 RAM 80% 초과도 GPU와 같이 경고로 둔다. 실제 완료한 계산을 비율 때문에
실패로 바꾸지 않으며 OOM·계산 오류·잘못된 측정 근거는 계속 거절한다.

57e91eb 실행과 이번 검사·재개 연결 수정의 검토된 파일 내용만 소스 호환으로 인정한다.
기존 DB·점수·snapshot·자원 이력은 원래 commit을 보존하고 새 실행에는 실제 commit을 기록한다.
입력·예산·환경은 기존 검증을 유지한다. 모델·후보 풀·배치·학습·점수·선택식은 변경하지 않는다.

## One-Liners 앙상블의 상수 성분 처리

2026-09-09 공식 One-Liners의 최신 main commit은 기존 봉인
`dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9`와 같다. MIT license의
[앙상블 실행 코드](https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/publications/iclr2026_timeseriesfoundationmodelsad/-/blob/dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9/scripts/evaluate_baselines_ensemble.py)는
단변량 component min-max 뒤 두 점수의 최댓값을 사용하며 0 범위를 처리하지 않는다.

다변량 채널별 확장에서 한 상수 성분 때문에 전체 실행을 중단하지 않도록, 범위가 정확히 0인
성분·채널만 정규화 결과 0으로 둔다. 같은 채널의 다른 성분과 모든 채널을 유지한다. 양의 범위는
기존 공식 식을 그대로 쓰고 epsilon·저분산 threshold는 추가하지 않는다. NaN·Inf는 계속 거절한다.
이는 공식 원본에 없는 경계 조건 보완이다. `component_score_ranges`에 원래 최소·최대와 정책,
0부터 시작하는 대상 채널 번호를 기록한다. 공개 후보·창·차분·결합 순서와 q별 선택은 유지한다.

## PCA의 0 가중치에 한정한 full SVD 재계산

2026-09-09 공식 TSB-AD 기본 브랜치의 최신 commit은 기존 봉인
`6beac72e11d1155ade40870492c00d0d1cfdcaaf`와 같다. Apache-2.0의
[PCA 구현](https://github.com/TheDatumOrg/TSB-AD/blob/6beac72e11d1155ade40870492c00d0d1cfdcaaf/TSB_AD/models/PCA.py)은
기본 auto와 full SVD를 지원하며, [공식 호출부](https://github.com/TheDatumOrg/TSB-AD/blob/6beac72e11d1155ade40870492c00d0d1cfdcaaf/TSB_AD/model_wrapper.py)는 auto를 사용한다.
[scikit-learn 1.7.1](https://github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/decomposition/_pca.py)의
auto가 선택한 covariance_eigh에서 작은 음의 고유값을 0으로 보정하면 공식 가중 거리의 나눗셈이
정의되지 않는다. Lightning series 11에서는 이 방식의 0 가중치 하나가 full SVD에서 사라졌다.

사용자의 원 절차 보존 수정 요청에 따라 이 경우에만 full SVD로 재계산한다. 공식 자동 복구나
새 HPO 축으로 주장하지 않고 프로젝트 수치 보완으로 기록한다. 공개 후보·성분 수·평가 입력 fit·
zero pruning·가중 거리식은 유지하며 작은 분산에 임의 하한을 넣지 않는다. 실제 solver와 복구 전
0 가중치 수를 저장하고 재계산 후에도 정의되지 않는 점수는 계속 거절한다.

## GPU 메모리 여유율과 TSPulse 수치 비교 기준

2026-09-09 배치 32의 Lightning 검사에서 TSPulse 세 창 모두 추론을 마쳤다. 가장 큰 head 점수
차이는 1.4692894585444094e-6이며 기존 `atol=1e-8` 검사가 실패했다. 배치 비교는
[PyTorch float32 기본 비교 허용치](https://docs.pytorch.org/docs/2.10/testing.html#torch.testing.assert_close)인
`rtol=1.3e-6`, `atol=1e-5`를 적용한다. shape·head·유한값 검사와 큰 점수 차이 차단은 유지한다.
후처리 배열의 저장 dtype이 float64여도 모델 연산에서 생긴 float32 오차는 사라지지 않는다.

사용자의 GPU 100% 사용 요청을 반영해 자원 검사의 GPU 메모리 80% 기준은 경고로 바꾼다.
예약 메모리와 실행 전 점유량의 합을 기록하며 100% 이하에서 측정 실행을 마치면 이 비율만으로
차단하지 않는다. 실제 OOM·실행 오류·잘못된 관측값과 RAM 80% 이상은 실패로 남긴다. GPU 연산
사용률은 이 검사 대상이 아니다. 추론 배치 32와 공식 후보·점수식은 유지하며 전체 튜닝 완주는 확인 전이다.

## TSPulse 추론 배치는 L4에서 32로 제한

2026-09-09 Lightning의 248채널 입력에서 TSPulse 세 설정이 CUDA OOM으로 실패했다.
공식 경로가 기존 배치 제한을 덮어쓰지 않게 하고 추론 배치 32를 registry와 실행 정책에 기록한다.
이는 승인된 자원 조정 범위의 실행 분할이며 후보·창·head·학습 또는 점수식을 바꾸지 않는다.
[봉인한 공식 compute_score](https://github.com/ibm-granite/granite-tsfm/blob/fe7a35697723e2a2f5246ae979474bfc554e26c0/tsfm_public/models/tspulse/utils/ad_helpers.py)의
창별 집계를 유지하고 기존 원격 검사에서 배치 1과 수치를 대조한다. 32의 L4 통과는 확인 전이며
실제 OOM은 실패로 남기며 이후 GPU 경고선·RAM 합격선과 수치 허용치는 위 최신 결정을 따른다.
실패 보고서는 모델의 원래 오류와 관측치를 먼저 알린다.

## 시작 환경은 Python 3.11·3.12를 허용

2026-09-09 Lightning 준비 실패 수정 요청으로 Python 3.11.14 단일 고정을 3.11.x·3.12.x 허용으로
바꾼다. 고정 PyPI 패키지의 지원 선언과 [Time-RCD](https://github.com/thu-sail-lab/Time-RCD/blob/372bb980426b2f67007311c6f3165ab789c79bef/pyproject.toml),
[granite-tsfm](https://github.com/ibm-granite/granite-tsfm/blob/fe7a35697723e2a2f5246ae979474bfc554e26c0/pyproject.toml)의
공식 Python 범위에 근거한다. 이 선언만으로 실제 GPU 실행을 검증한 것으로 보지는 않는다.
설치 대상은 현재 터미널의 Python이며 torch는 공식 CUDA 12.6 빌드를 쓴다. 패키지 버전·공식
source commit 고정은 유지한다. 실제 Python patch·pip freeze·CUDA·소스는 첫 실행에 봉인하고
재개 때 같은 환경인지 검사한다. 환경 오류는 한 번에 모아 보고한다.

## 디스크 용량은 관측 정보로 기록

2026-09-09 사용자 요청으로 고정 25GiB 디스크 기준을 제거한다. 현재 파일시스템 용량을
기록하되 추정 필요량·임의 안전 배수·여유율로 실행을 제한하지 않는다. 검사 재사용 때도
새 관측을 이력에 남기고, 조회 실패와 실제 쓰기 실패를 구분한다. 실제 저장 실패는 기존대로
중단·보존하며 학습 증거 저장 오류가 종료 이력 오류에 가려져도 재학습하지 않는다.
모델·후보·선택식·DB schema·GPU/RAM 검사는 유지하고 원격 검증 뒤 새로 봉인한다.

## 최종 감사의 저장 누락과 PCA 추정 판정 보완

2026-09-09 사용자 승인으로 TSPulse의 실제 입력 scaler·head별 교정 범위·출력 scaler를
기존 metadata에 보존한다. 완료 검사와 추천 DB의 파일 검증이 이 수치를 확인하며 DB schema는
바꾸지 않는다. 학습 파일은 하나씩 저장한 뒤 참조를 이력에 남겨 다음 파일 저장 실패 때도 인수한다.

PCA의 작업 배열 6벌 추정은 보수적 사전 값으로 유지한다. 추정이 기준을 넘으면 최대 작업 배열
입력과 n_components=None의 등록 설정을 별도 원격 CPU 프로세스에서 실행하고 실제 RSS로
판정한다. RAM 80% 미만 기준을 유지하며 추정·실측·대표 실행 신원을 함께 보존한다. 이 측정은
전체 후보의 자원 상한 인증이 아니며 후보나 공식 계산식을 줄이지 않는다. 원격 검증과 새 봉인 전에는
본 튜닝을 시작하지 않는다.

## 추천 입력의 단위와 단일 명령 적재

2026-09-09 후속 요청으로 원시 진폭 통계와 추천 입력을 구분한다. 추천 DB schema 3의
`recommendation_inputs` VIEW·CSV에는 n·d, 무차원 특징, 채널별 IQR/std 중앙값과
상수·유효값·유효 채널/쌍 비율을 담는다. NULL과 사유를 보존하며 q·N·식별자는 문맥으로 둔다.
원시 통계·수집 산식 `prefix_features.v2`와 기존 모델 절차는 유지한다. 표본 간격과 float32
정밀도 한계, 학습 family 안에서의 스케일링·결측 처리 조건은 수집 계약에 기록한다.

기존 `run_ratio_tuning` 단일 진입점의 DB 적재·백업·CSV·인수에 새 조회를 연결한다.
저장 실패를 완료로 처리하지 않으며 같은 새 실험에서 저장된 특징·점수·채점 원표로 재개한다.
입력 열 계약은 준비와 최종 영수증에 함께 봉인한다. 이전 DB는 자동 변경하지 않는다.
구현·정적 검토 후 원격 회귀를 확인하며 본 튜닝 실행 게이트는 유지한다.

## 최신 공식 저장소 우선과 튜닝 절차 복원

2026-09-09 추가 사용자 지시로 연결된 공식 저장소의 최신 기본 브랜치 commit을 기준으로
삼는다. 아래의 논문 우선·PaAno 최소 500 제거 결정은 이 지시가 대체한다. 확인 시점의
commit을 registry·실행 신원·notice에 고정하며 매 실행마다 임의의 HEAD를 가져오지 않는다.

PaAno는 `official_minimum`을 복원한다. patch 수 P에 대해 memory 개수는
`max(min(500, max(1, P−1)), min(round(0.1×P), P−1))`이다. patch 1,000개면 memory는
500개다. 학습·feasibility·checkpoint 기록을 맞추고 `paper_fraction`은 명시적 역사 호환에만 남긴다.
GDN 세 후보 모두 최신 train.py의 patience15·Adam betas 0.9·0.999를 쓴다.
GDN self 포함 top-k·채널별 SMA 후 max와 PaAno negative 식은 공식 코드와 같아 유지한다.

PCA·TSPulse는 최신 source pin으로 갱신하되 관련 코드와 license가 같아 계산식을 유지한다.
나머지 네 저장소 HEAD와 Tier 3 checkpoint는 그대로다. 전체 후보는 36개 설정·45개 head이며
공식 HPO, 비교 실행값, 논문 근거에 코드 값을 보완한 조합을 구분한다. 출처 대조는
[사전 점검 기록](process_0_preverify.md), 현재 commit·license는 [notice](../../THIRD_PARTY_NOTICES.txt)를 따른다.

공식에 없는 공통 selector는 새로 만들지 않는다. 기존 q·모델·조건집단별 family 동일 가중
선택과 TSPulse의 승인한 공통 창 집계·고정 동률 순서를 프로젝트 보완으로 유지한다.
PCA 공식 HPO는 파일별 후보 점수만 저장하며 PaAno의 파일·Category 평균은 보고용이다.
Dev18·q·seed 반복·고정 평가 tail도 연구 설계로 구분한다. VUS 평가기 source의 과거 봉인은
검증 보고의 출처이므로 유지하며 PCA 모델 source 갱신을 새 평가기 검증으로 기록하지 않는다.
원격 검증과 새 코드·환경·예산 봉인 전에는 본 튜닝을 시작하지 않는다.

## 추천 DB 특징 확장과 SQL 무결성

2026-09-09 사용자 요청으로 IQR, 차분 절댓값 Q90/IQR, 전후반 중앙값 이동/IQR,
spectral entropy를 추가한다. 현재 q-prefix에서 채널별로 계산하고 유효 채널의 중앙값과
NULL 사유를 함께 저장한다. 정확한 산식은 [수집 계약](tuning_feature_capture_prompt.md)을 따른다.
원본·모델 입력·RNG·후보·선택식은 수집 때문에 바꾸지 않는다.

extractor는 `prefix_features.v2`, 추천 DB schema는 2로 올린다. 기존 표시 열 뒤에 추가하고
성능표 11열은 유지한다. 이전 DB의 신원과 새 산식을 섞거나 자동 이관하지 않는다.
SQL 원본의 결과 기본키·복합 외래키는 유지하며, 관측량 상한과 완료 증거의 빈 값을 막는다.
이번 결정은 추천 수집·저장 범위에만 적용하며 본 튜닝 실행 게이트는 열지 않는다.

## TSPulse 공통 창 집계 보완 승인

사용자가 Dev18 파일별 time·fft VUS-PR 평균을 파일 간 동일 가중 평균하는 창 선택식을
승인했다. 각 q에서 공통 창 64·96·128 중 하나를 고른 뒤 공식 코드의 데이터셋별 파일 평균으로
time·fft·pred·ensemble 중 head를 고른다. 창 집계식은 프로젝트 보완이며 논문의 미공개 원식으로
부르지 않는다. 창 점수의 동률은 기존 허용 오차 안에서 작은 창을 우선하고, head는 공식 CSV처럼
파일별 5자리 점수에서 time·fft·pred·ensemble 순서로 동률을 해소한다.

주 선택은 각 q의 전체 TSPulse 개발 패널에서 고른 공통 창과 데이터셋별 head를 사용한다.
LOFO와 같은 파일끼리의 모델 비교는 해당 fold의 선택용 파일에서 창과 head를 다시 고른다.
보류한 데이터셋은 선택에 넣지 않고 공식 미관측 fallback인 time으로 채점한다. 세 창·네 head의
원표를 모두 보존하며, 정책의 데이터셋별 head를 실제 점수·실행 증거에 연결한다.

TSPulse 후보·추론·L4 자원 설정은 유지한다. 선택 절차의 버전을 바꾸고 원격 검증 뒤 새 예산을 봉인한다.
아래 2026-09-07 기록의 평균식 승인 대기는 해제됐다.

## PaAno memory 비율과 나머지 풀 최종 검토

PaAno [논문 B.1](https://arxiv.org/html/2602.01359v3#A2.SS1)의 memory 비율은 patch의 10%다.
[공식 코드](https://github.com/jinnnju/PaAno/blob/d4c67116190efa4592dc6a8a157ced0def68b6af/utils/utils.py#L33)의
최소 500개 규칙과 충돌하므로 현행 경로는 논문의 10%를 우선한다. 개수의 반올림과 patch 수보다
작아야 하는 상한은 공식 코드를 따른다. 예를 들어 patch 1,000개면 memory는 500개가 아니라
100개다. 학습·feasibility·checkpoint를 같은 규칙으로 맞추고 실제 개수를 기록한다.
과거 official_minimum checkpoint는 명시된 호환 경로에서만 읽는다.

후보 풀은 One-Liners 16·PCA 4·PaAno 9·GDN 3·TimeRCD 1·TSPulse 3으로 유지한다.
One-Liners는 공개 비교 구현의 풀이다. GDN은 run.sh와 논문 주요 값에 공개 코드 값을 보완한
세 조합이며, out_layer_num=1에서 사용되지 않는 hidden 값을 별도 탐색 축으로 해석하지 않는다.
PaAno recipe와 선택 절차 신원이 바뀌었으므로 원격 검증 후 새 예산을 봉인한다.

## 논문 우선과 L4 실행 보완

논문에 명시된 방법을 공식 코드보다 우선한다. 논문이 생략한 세부는 공식 코드로 보완하고,
둘 다 공개하지 않은 절차는 다른 공개 구현을 확인한다. 출처가 확인되지 않은 자동 선택식을
논문 원식으로 부르지 않는다.

L4 24GB 한 장과 CPU 8개에서 튜닝한다. PaAno는 공식 batch 512로 GPU에서 추론하며,
memory 대표 탐색은 거리 행렬을 나눠 계산한다. 전체 fit embedding과 논문의 memory 10%를
유지한다. PCA도 공식 전체 평가 fit을 유지하고 거리 계산만 나눈다. 후보·학습 batch·
반복 수·정밀도·window·topk는 자원 문제를 이유로 줄이지 않는다.

채점 worker 상한은 8개이며 각 계산 thread는 1개다. 자원 검사는 호스트 RAM과 컨테이너
할당 한도 중 작은 값을 사용한다. GDN 사전 probe는 내부 검증 분할 뒤에도 최대 배치를
점검하도록 prefix 길이를 보충하고, 실제 학습 배치 수를 기록한다. 이 작은 probe는 전체
튜닝의 시간이나 KMeans·PCA 전체 fit 메모리를 보장하지 않는다.

TSPulse의 공통 창 집계식은 논문·공식 코드와 다른 공개 구현에서 확인하지 못했다.
논문의 공통 창 선택 → 데이터셋별 head 선택 순서를 따르고, 집계 세부는 위에서 승인한
프로젝트 보완을 적용한다.

## ALoRa 제외와 남은 공식 후보 풀 유지

최신 사용자 지시로 ALoRa와 전용 코드·폴더·실행 경로를 제외한다. 적은 데이터에서의
실행 제약과 튜닝 부담을 고려한 범위 변경이며 GHL·HAI 최종 성능을 보고 모델을 제외한
결정으로 해석하지 않는다. 대체 모델은 추가하지 않고 Tier 2는 PaAno·GDN으로 진행한다.

나머지 공식 논문·구현 기반 풀은 유지한다. One-Liners 16개, PCA 4개, PaAno 9개,
GDN 3개, TimeRCD 1개, TSPulse 3개로 총 36개 설정이다. TSPulse 네 head를 구분하면
45개 선택 항목이다. 논문 최종값만 남기는 추가 축소는 하지 않는다.

GDN은 공식 run.sh 한 조합과 논문 SWaT·WADI의 주요 값에 공개 코드 값을 보완한
두 조합을 유지한다. TSPulse aggregation 64·96·128은 논문 HPO 범위다. 세부 후보와
출처는 계획서 v5의 튜닝 표를 따른다. 모든 후보 수나 실행 비용이 같다는 공정성은 주장하지 않는다.

paper_tuning_v4는 과거 validation 없음·평가 통계 금지·공통 smoothing보다 우선한다.
PaAno native RevIN·memory, GDN 내부 검증·조기 종료, PCA 평가 입력 fit·zero pruning,
One-Liners 앙상블과 Tier 3의 공식 전체 평가 통계·native 점수를 유지한다. 가중치 학습에
평가 라벨을 넣지 않는다. 실제 통계 출처와 비인과적 평가 범위를 metadata에 기록한다.

모델·q·실행 조건 집단마다 설정을 독립 선택한다. 모든 q 또는 모든 CSV의 가능 후보를
교집합으로 줄이지 않는다. 짧은 prefix나 적은 채널로 불가능한 조합은 unavailable로 남기며
공식 batch·window·topk를 임의로 줄이지 않는다. Dev18·q·고정 평가 tail과 계층 비교는
사용자 연구 설계다. TSPulse의 공통 aggregation 선택 세부식은 공개 근거가 없어 복제하지 않는다.

과거 점수·checkpoint·선택표·예산은 폐기한 상태를 유지한다. 원본 데이터·감사·ell_max와
남은 모델의 공식 source·사전학습 가중치는 보존한다. 같은 새 실행에서만 완료 점수와
SQLite 연결·내보내기를 복구하며, 폐기한 Downloads 결과를 읽거나 복구하지 않는다.

코드·설정·문서와 회귀를 수정하며 로컬 실행은 하지 않는다. 원격 회귀와 작은 합성 점검을
마친 뒤 새 예산을 봉인한다. 실제 튜닝과 GHL·HAI 최종 실행은 별도 게이트를 따른다.

## 이전 결정 — 위 현행 계약과 충돌하는 부분은 폐기
## 추천 자료 저장 구현

수집 프롬프트에 따라 CSV/q별 최소 특징과 전체 후보 결과를 SQLite로 연결했다. 추가 descriptor는
필수 목록에서 제외하고 첨부의 요약 네 항목과 채널별 mean·std·median·ACF lag1, 유효 수와
미정의 사유만 계산한다. 입력은 scaling 전 float32 prefix이고 통계 계산은 float64다.

새 저장 schema는 `full_prefix_storage.v1`, 실행 증거는 `dev18_registered_runner.full_prefix_v3`,
예산 schema는 3으로 구분한다. 중복 fit/validation 행 수를 제거하고 `observed_row`로 맞췄다.
기존 reader는 역사 schema도 읽지만 새 완료 증거는 새 계약을 요구한다. 후보·선택식은 바꾸지 않았다.
실행 신원에 저장 계약이 들어가므로 코드를 확정한 뒤 config ID·예산을 다시 봉인한다.

새 prepare의 선택적 baseline SHA 수집과 종료 재검사를 제거했다. DB 저장 실패는 재학습으로
재시도하지 않고 같은 새 실험의 저장 결과로 연결을 복구한다. 전달물은 SQLite backup으로 만든
일관된 사본과 CSV이며, 큰 배열·checkpoint는 위치·SHA로 인수한다. 전달 전 상태는
`ready_for_handoff`로 남겨 묶음 생성 실패를 전체 완료로 표시하지 않는다.
이번에는 코드·문서·회귀를 작성하고 정적으로만 검토했다. 로컬 실행과 본 튜닝은 하지 않았다.

## 전면 재튜닝과 추천용 정보 수집

사용자는 과거 80/20 분할·과거 후보 풀·스케일러로 얻은 튜닝 결과를 폐기하고 전면 재실행한다.
Downloads의 과거 결과 묶음은 복구·보완하거나 새 성능 근거로 쓰지 않는다. 원본 데이터와
검증된 공식 사전학습 가중치는 그대로 사용하며, 새 실행은 폐기 결과 없이 독립적으로 시작한다.
이 결정은 뒤의 과거 결과 보존·보완 지시보다 우선한다. 이번 작업은 프롬프트 수정이며 물리
파일 삭제나 튜닝 실행을 수행한 것은 아니다.

재튜닝에 앞서 CSV/q별 입력 feature와 전체 후보의 설정·점수·학습 상태·비용·실패 연결을
수집하도록 코드를 수정한다. feature 검증 전에는 본 모델 배치를 시작하지 않고, 필수 추천
자료가 빠졌으면 최종 전체 완료로 처리하지 않는다. 재개는 같은 새 실험 안에서만 허용한다.
[구현 프롬프트](tuning_feature_capture_prompt.md)를 기준으로 수집 코드를 완성하고 원격에서
검증한 뒤 새 budget·계약을 봉인한다. 추천기 자체와 지원 상한은 후속 작업이다.

사용자의 [저장 양식](tuning_storage_format.xlsx)에 맞춰 성능 결과·입력 특징·채널별 특징·
모델 설정을 SQLite로 연결한다. 새 full-prefix 저장에는 중복 fit/validation 행 수를 두지
않고 observed_row와 전체 prefix 정책·target_use를 사용한다. 기존 저장 schema도 새 버전의
writer/reader를 함께 맞추며 과거 봉인 파일은 변조하지 않는다. 특징은 같은 CSV/q에서
한 번 계산해 여러 모델·설정·seed·head가 재사용하고 q가 바뀌면 별도로 계산한다.
필수 입력 통계는 첨부 항목과 유효 수·미정의 사유로 줄인다. 추가 descriptor는 원본 범위를
보존해 나중에 계산한다. DB 내부에는 head 구분과 실행 증거 연결을 유지하고 11열 성능
내보내기는 CSV/head별로 나눈다. 구조적 제외에는 dummy 실행·점수 파일을 만들지 않는다.
이번에는 수정 프롬프트와 첨부·계획 문서만 반영했으며 SQLite·수집 코드 구현은 아직 전이다.

## 공식 변형 재점검과 현행 recipe

사용자 승인으로 `source_faithful_v3`를 반영한다. 이 이름은 근거를 재점검한 recipe 버전이며
원본 실험이나 점수의 완전 재현을 뜻하지 않는다. 아래 결정이 뒤의 과거 전처리·후보 기록보다
우선한다. 학습형 q는 서로 새로 학습·선택하고, 같은 q의 모델별 recipe와 Tier 공동 우승을 각각
남긴다. 겹치는 prefix·평가 구간의 통계적 독립성을 주장하지 않는다.

| 대상 | 현행 결정과 근거 |
| --- | --- |
| Dev18·전체 prefix | 최종 GHL 누수를 막으려고 공식 20개에서 GHL 09·18만 제외한다. 각 q에 확보한 prefix 전부를 쓰고 평가 tail은 fit에 넣지 않는다. 공식 split 재현과는 다른 연구 질문이며 validation 성능 보장은 없다. |
| MWVAR·SQDIFF_LAST3·PCA | 공개 window 64·96, last-3 식, PCA 네 component 값을 유지한다. raw-unit 기준선에 임의 z-score를 추가하지 않는다. 특히 MWVAR를 자기 window 분산으로 표준화하면 검출량 자체를 없애므로 피한다. |
| PaAno | 공개 patch·학습률 9개와 100 update·batch 512·학습 loss checkpoint·최소 memory 규칙을 유지한다. 공식 multivariate 경로의 native RevIN 위에 추가했던 MinMax는 필수가 아니므로 제거한다. |
| ALoRa 입력·손실 | 공식 loader의 train-fit StandardScaler를 복원한다. `[:, 1:]`는 논문 식 (7)의 표본별 선두 특잇값 제외를 따른 수정이다. 원본 batch축 slicing과 수치가 다름을 명시하고 작은 수식 검사를 남긴다. regularizer의 batch 합은 유지한다. |
| ALoRa 선택 | 평가 loss를 보는 원본 checkpoint 선택은 누수 방지를 위해 제거한다. epoch 3·4·10과 공개 h1 네 값의 120개 합집합을 개발 패널에서 비교한다. 논문의 자동 h1 규칙을 완전 재현할 구현 근거는 부족하므로 자동 추정식을 새로 만들지 않는다. |
| GDN | 기존 네 그래프 묶음을 보존하고 논문의 64/topk15·128/topk30을 더한다. 여섯 묶음 × cap30·50의 12개를 비교한다. 공식 no-validation 분기처럼 epoch별 batch 평균 MSE 합이 최소인 state를 복원한다. cap 전체를 실행하며 선택 epoch로 비용을 줄여 적지 않는다. |
| TimeRCD | 공식 사전학습·추론의 채널 표준화를 되살리되 현재 유효 추론 블록에서만 추정하고 padding을 제외한다. std=0은 공식의 1e-8이다. 전체 입력 표준화와 범위가 다른 적응이며 블록 간 level·scale 이상을 약화할 수 있다. 문맥 5,000과 multi checkpoint는 유지한다. |
| TSPulse | 공식 외부 표준화의 필요성을 반영해 과거 512점에서만 평균·표준편차를 구한다. 미래 1점에는 같은 통계를 적용하며 native RevIN과 MSE는 유지한다. 공식 전체 파일 fit을 쓰지 않기 위한 범위 변경이다. 상수 채널 scale은 StandardScaler처럼 1이다. |
| TSPulse head | time·fft 각각 64·96·128과 aggregation과 무관한 pred 하나를 주후보로 둔다. pred는 96에 귀속해 중복 선택을 없앤다. 64·128은 공식 기본 96 주변의 프로젝트 확장이다. 척도 교정 없는 raw_max는 충분한 논리가 없어 보조로 내리며 원시 출력은 보존한다. |

근거는 봉인한 [One-Liners](https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/publications/iclr2026_timeseriesfoundationmodelsad/-/tree/dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9),
[TSB-AD 후보 목록](https://github.com/TheDatumOrg/TSB-AD/blob/e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48/TSB_AD/HP_list.py),
[PaAno 학습 코드](https://github.com/jinnnju/PaAno/blob/d4c67116190efa4592dc6a8a157ced0def68b6af/train.py)와
[논문](https://arxiv.org/html/2602.01359v2), [ALoRa loader](https://github.com/CharisShimillas/ALoRa/blob/97dcc4a337710e6dc72c1a67893717c9538bae1a/data_factory/data_loader.py),
[solver](https://github.com/CharisShimillas/ALoRa/blob/97dcc4a337710e6dc72c1a67893717c9538bae1a/solver.py)와
[논문](https://arxiv.org/html/2602.08467v1), [GDN train](https://github.com/d-ailin/GDN/blob/9853899da860682669a134e4af315d036aab4eca/train.py)과
[논문](https://arxiv.org/html/2106.06947), [TimeRCD 추론 코드](https://github.com/thu-sail-lab/Time-RCD/blob/372bb980426b2f67007311c6f3165ab789c79bef/time_rcd/_inference.py),
[TSPulse helper](https://github.com/ibm-granite/granite-tsfm/blob/9739fa59b61bd9f15cbfb06e5dc3dab28c72ee8d/tsfm_public/models/tspulse/utils/ad_helpers.py)다.

전체는 152개 설정·156개 주 선택 항목이다. CSV/q/seed와 feasibility를 반영한 실행 수는 새로
봉인한다. GDN cap 간 학습 공유는 구현하지 않았으므로 각각 실제 비용을 기록한다. ALoRa의
공유 학습과 TSPulse의 여러 head 추론 비용은 기존 이력대로 묶으며, 묶음 비용을 단일 head
최소 배포비라고 부르지 않는다. 새 common/preprocess recipe가 config ID를 바꾸므로 과거
raw·MinMax·마지막 epoch 결과를 새 완료 결과에 합치지 않는다.

GDN의 rho 0.3·0.25는 센서 수가 다른 파일에서 연결 수를 비율로 늘리는 프로젝트 비교 후보다.
두 수치가 이론적 최적값이라는 근거는 없으므로 공식 고정 topk와 함께 검증한다. topk 2는
저채널에서 센서 간 연결을 비교하려고 유지한다. PaAno patch·ALoRa window도 최단 파일에
맞춰 일괄 줄이지 않고 CSV/q별 가능 여부를 남긴다. 학습량·센서 수와 함께 기존 EDA의 주기·
상수 채널·오염 정보를 해석하되, 최종 평가 통계로 후보 범위를 다시 정하지 않는다. 실행 가능한
후보 집합이 같아도 센서 의미나 이상 유형이 같다는 보장은 없으며 작은 family 집단은 한계로 남긴다.

후속 선형·조합 최적화에는 전체 후보의 점수·설정·실행 근거를 남긴다. VUS 우승이나 개발
Pareto만으로 후보를 영구 삭제하지 않는다. 기업 조회에도 model_ratio와 tier_adaptive 양쪽의
별도 선택 설정을 넘긴다. 현장 비용·threshold·운영 기간·전환비가 준비돼야 최적화를 계산하며,
최종 데이터의 실행 메뉴가 일부라면 최적성도 그 메뉴 안으로 한정한다. 실패·미측정 비용은
0으로 채우지 않는다. family-LOFO는 선택 절차의 개발 근거이며 신규 현장 성능을 인증하지 않는다.

## 튜닝 중단과 저장 실패

중단 요청 뒤 실행 중인 VUS worker가 끝날 때까지 명령 이력과 잠금을 유지한다. 보고서 CSV는
완전히 기록한 임시 파일로 교체한다. 모델 실행이 끝난 뒤 저장에 실패하면 같은 명령에서
재학습하지 않는다. 완료 영수증은 다음 명령에서 검증·복구하며, 영수증이 없는 일반 모델은
새 시도로 실행한다. 실패한 부분 checkpoint를 모두 보관하지는 않지만 당시 snapshot 내용,
반환된 timing, 실패·중단 상태와 시도 횟수는 UUID 이력에 남긴다. ALoRa에도 같은 기록을 남긴다.

자원 검사는 개별 probe의 시작과 종료를 저장하고 같은 코드·입력·예산·패키지 환경·GPU·RAM의
완료 결과만 재사용한다. 이 시간과 모델 내부 timing은 명령 시간의 내역이며 합계에 다시 더하지
않는다. 강제 종료로 측정하지 못한 시간은 미확정이다. 이번 변경은 실행 검증 전이다.

## 원본 후보 보존과 단일 진입 튜닝

닫힌 원본 grid가 있으면 그 값을 포함하고, 실행 예시만 있으면 그 예시와 프로젝트 추가 후보를
구분한다. MWVAR 64·96, ALoRa batch 128·256과 GDN 공식 코드의 topk 5·batch 32를 포함한다.
GDN의 기존 rho 후보는 보존하고 저채널용 topk 2를 추가한다. topk를 채널 수에 맞춰 몰래
줄이지 않는다. 전체 prefix·고정 epoch·q별 family-LOFO와 strict 점수 경로는 명시한 프로젝트
프로토콜이며 원본 재현이나 동일 성능으로 설명하지 않는다.

`run_ratio_tuning.py`가 준비부터 선택 근거 저장까지 담당한다. 승인된 입력·VUS·ell_max는
선행 인수물로 유지한다. 완료 기록은 검증 후 재사용하고, 중단 시도의 비용·횟수는 남긴다.
강제 종료로 측정하지 못한 시간은 null로 두며 명령·모델 시도 시간을 중복 합산하지 않는다.
model_ratio와 tier_adaptive가 선택한 설정·head의 근거는 함께 보존한다. 튜닝 관측의 최대값은
서비스 지원 상한의 인증값이 아니다. 이번 변경은 정적 검토까지이며 원격 실행 검증은 남아 있다.

## 튜닝에서 추출하는 모델별 규모 근거

추가 요청에 따라 상한 근거의 보류를 해제했다. 각 q의 전체 후보·조건별 설정 선택은 유지하고
완료 튜닝에서 모델별 n·d·N·평가 길이·환경과 선택 설정의 모든 seed 증거를 묶는다. 별도 학습이나
VUS 재계산 없이 완료 보고에 관측 지점과 모델별 규모표를 추가한다. 행·열의 개별 최댓값을 합친
직사각형이나 최근접 q로 미관측 조합을 지원하지 않는다. 모델별 근거를 합쳐 후보를 찾되 모델
비교는 공통 파일을 유지한다. N까지 빠진 q가 있으면 계획 경로의 미관측 지점으로 남긴다.

이 표는 검증 가능한 상한 후보를 좁히는 개발 근거다. family-LOFO와 완료 실행만으로 신규 기업의
운영 성능·서비스 자원 한도를 인증하지 않는다. 반환값은 항상 현장 검증 후보이며 수치 상한의
확정에는 독립 운영 평가가 남는다. 선택표와 규모 JSON도 완료 영수증에 묶는다.

## Tier 3 공식 튜닝 대조

TimeRCD는 공식 논문의 고정 문맥 5,000과 multi checkpoint를 유지한다. 문맥 길이 민감도
실험을 CSV별 최적 설정 탐색으로 해석하지 않는다. TSPulse는 기존 aggregation 64·96·128에서
time·fft·pred·raw_max 네 점수를 모두 선택 후보로 쓴다. 개별 head를 진단용으로만 제한한
과거 equal-trial 규칙은 새 전수 탐색에 적용하지 않는다. 같은 추론의 네 출력을 쓰므로 물리
실행은 늘지 않는다. 기존 선택표와 예산은 보존하고 새 budget으로 채점·선택을 봉인한다.

예산과 실행 목록은 registry의 development seed를 함께 쓴다. deterministic은 첫 seed인
0으로 한 번 실행한다. 예산에만 1을 고정해 실행 후보가 빠지던 오류를 수정했다. TSPulse
보고서도 같은 config의 네 head를 범례에서 구분한다.

공식 TSPulse의 family별 head 선택은 같은 family로 이전하는 벤치마크 설계다. GHL을 튜닝에서
제외하고 HAI도 외부 평가로 남기는 이번 연구에는 조건 집단의 family 동일 가중·LOFO를
유지한다. 공식 코드의 전체 CSV 평가, 전체 입력·점수 정규화와 native smoothing은 도입하지
않는다. raw_max는 공통 native 구간에서 최대값을 구한 뒤 경계를 반복하는 프로젝트 앙상블이다.
공식의 head별 경계 반복·정규화·smoothing 후 최대값과 구분한다. 고정된 평가 tail과 학습 구간에서
봉인한 ℓ_max도 유지하므로 공식 벤치마크 수치의 직접 재현으로 부르지 않는다.

## full_prefix_v2

사용자의 전면 수정 요청에 따라 현재 q-prefix 전부를 학습에 쓴다. 80/20 holdout은 새 경로에서
제거하고 기존 실험 결과는 보존한다. 학습 loss로 checkpoint를 고르는 PaAno와 고정 epoch의
ALoRa·GDN을 구분한다. GDN의 fit 점수 교정은 독립 validation 성능으로 취급하지 않는다.

모델/q마다 같은 실행 가능 후보 집합을 가진 CSV를 묶고 그 안에서 설정을 선택한다. Tier는
모델별 후보 집합 조합까지 같은 집단에서 모델+설정을 공동 선택한다. seed·파일·family의 순서로
평균하고 family LOFO를 따로 보고한다. 짧은 CSV 하나로 긴 CSV의 patch64·96을 삭제하지 않는다.

ALoRa는 window 5개, epoch 3·4·10, h1 4개를 탐색한다. 같은 학습의 checkpoint와 추론을 공유하고
완료 결과를 덮어쓰지 않는다. 저채널 CSV 8개의 제외 사유는 원본 목록과 함께 보존한다.
이는 공식 저자의 동일 CSV subset이나 자동 h1 추정을 재현했다고 주장하는 실험이 아니다.

새 budget은 full_prefix_per_ratio이며 equal-trial 비교가 아니다. 점수→VUS 원표→LOFO→조건별
정책→CSV별 final membership→기존 GHL·HAI 결과 절차를 유지한다. 기존 462행 고정 규격은
폐기하고 split·series·group별 실제 요청 수를 사용한다.

기업 입력은 현재 정상 학습 행수 n, 센서 수 d, 예상 최종 학습 행수 N이다. 별도 운영 한도는
선택 입력으로 검사한다. 실제 n·d·N·평가 길이·환경이 관측 근거와 일치하는 등록 q만 조회한다.
상한만 맞거나 q만 같다는 이유로 추천하지 않는다.
수익 최적 경로는 현장 손실·운영 비용과 검증된 오류율이 준비된 뒤 구성한다.

기업의 N이 100%이며 서비스 상한 R_max와 구분한다. 가장 긴 CSV나 전체 파일 길이로 상한을
정하지 않는다. 기존 manifest에서 가장 긴 전체 파일은 400,000행이지만 학습 경계는 28,307행이다.
최대 학습 경계 37,500행은 2채널이고 최대 248채널 파일의 학습 경계는 7,016행이다. 이 최댓값을
조합한 규모 전체에 성능 근거가 있다고 보지 않는다. 실행 환경·동시 실행 조건·시간과 메모리 한도를
고정한 뒤 모델·센서 조건별 자원 경계와 독립 성능 근거를 함께 확인해 지원 범위를 봉인한다.
현재는 관측 범위 추출까지 구현하며 숫자의 서비스 인증은 미확정이다. GHL·HAI 최종 평가로 recipe를 다시 고르지 않는다.

학습 상태와 실행 이력은 재개 때 보존한다. 완료 조합과 ALoRa 저장 epoch를 재사용하고 명령·모델
시도별 시간을 별도 이력으로 남긴다. 강제 종료의 미확정 시간은 추정하지 않는다. 2026-09-06 수정은
당시 사용자 요청에 따라 실행 검증 없이 마쳤으며, 다음 시작점에 이 제한과 남은 검증을 기록했다.

같은 날 추가 요청에서는 튜닝에 필요한 미완료 부분만 다뤘다. 이미 연결한 학습 상태 저장과 이력
보존은 유지하고 전체 계획량·이번 미완료 수·완료 확인 시간·계산 시도 시간을 구분했다. ALoRa
재개 checkpoint 원본은 CPU로 읽어 실행 장치의 임시 중복 상태를 줄인다. 작은 검증만 수행했다.
당시 보류했던 서비스 규모 근거와 기업 조회는 이번 추가 요청으로 연결한다. 최종 threshold 분리는
계속 보류하며 현행 조건집단, family-LOFO, Q, 전체 prefix 설계를 바꾸지 않는다.

현재 결정의 세부 계약은 [계획서 v5](plan_v5.md)다. 아래는 이전 실행과 검토의 보존 기록이며
위 계약과 충돌하는 80/20·전역 recipe·별도 ALoRa 450건·절대 행 기준 조회는 현행 규칙이 아니다.

## 이전 결정 기록


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

## 2026-09-05 비율별 튜닝 확장

사용자 요청에 따라 공통 config 강제를 폐기한다. 각 q의 정적 feasibility가 허용한 등록
후보를 모두 실행하고 해당 q의 family-LOFO로 선택한다. 기존 완료 score·채점 ledger·checkpoint는
보존하며 새 예산과 결과를 분리한다. 동일 물리 key는 재사용하고 추가 후보만 실행·채점한다.
ALoRa는 heads 8을 바꾸지 않고 구조상 가능한 subset에서 따로 평가한다. 이 subset 점수를
18개 전체의 Tier 대표 선택에 섞지 않는다. 이 결정이 위의 고정 config 운영 계약보다 우선한다.

## 2026-09-05 기업 추천 범위와 비용 평가

사용자 보완에 따라 서비스의 고정 행 상한 R_max와 센서 열 상한 C_max를 둔다. 업로드가 어느
한쪽이라도 초과하면 추천하지 않는다. 기업의 확보율로 총량을 추정하던 설계는 폐기한다.
상한 수치는 적용 범위의 검증 근거와 함께 정하며 미설정 상태에서 추천을 열지 않는다.
파일별 실제 정상 학습 행 수와 센서 조건으로 기존 원표를 다시 묶어 가까운 지점의 설정을
후보로 삼는다. q별 family 평균 우승 설정을 행 수 하나에 대응시키지 않는다. 참조 거리와
모델의 길이·채널 조건을 만족하지 못하면 추천을 보류한다.

현재부터 R_max까지 모델 유지·재학습·전환의 누적 비용을 비교한다. 마지막 학습 때의
데이터량을 상태에 포함하며, 매 q 성능 1등을 채택하는 규칙으로 대신하지 않는다. 현재 독립 q
점수로 유지 경로를 계산할 때는 오류율 유지 가정을 명시한다. 실제 온라인 효과를 주장할 때만
별도 시간순 평가를 추가한다.

오탐·미탐과 비용 후보표는 저장 점수·라벨·validation reference·metadata를 우선 재사용한다.
원격 reference 보존 확인과 threshold 후처리 연결이 남았다. VUS-PR 1등만으로 비용 최적을
주장하지 않으며 추가 비용 후보·threshold 선택 규칙을 GHL·HAI 평가 전에 봉인한다. 현재
membership과 완료 산출물은 보존한다. 논문 수치는 비용 시나리오의 가정으로만 쓰고 기업
오탐률로 대체하지 않는다. 현장 손실 단가·발생 빈도·기존 운영안은 기업 입력으로 받는다.
이 항목은 계획이며 추천 제한·비용 최적화 코드나 새 운영 성능표가 구현됐다는 뜻은 아니다.

## 아직 열려 있는 결정

- GHL25·HAI Role-A manifest의 최종 승인 상태
- 비용 항목별 단가, 반복 횟수, latency·memory·최소 성능 제약
- 모델 교체·검증·배포·중단 비용을 포함한 현장 transition 단가와 제약
- 검증할 기업 적용 범위와 `R_max`·`C_max`, 절대 행 수 참조 거리, 비용 후보·운영 threshold와 사건별 경보 평가 규칙

Dev18 입력, feasibility, VUS-PR, 시계열별 `ℓ_max`, exact panel과 공통 Python 환경은 승인됐다.
primary 물리 실행 1,170건과 1,602행 ledger가 완료됐다. 과거 TimeRCD·TSPulse checkpoint와 L4 80%
자원 보고서는 이 실행의 역사 증거이며 selection-only에서 다시 요구하지 않는다. 현재 게이트는
완료 산출물을 재사용하는 비율별 추가 튜닝이다. main 2,556행 ledger·462행 membership과
ALoRa 별도 450행을 만들며 기존 294행 membership은 보존한다. GHL25·HAI의 후속 결정은
이미 허용된 Dev18 추가 실행을 막지 않는다.
