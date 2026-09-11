# 계획서 v5 — 제조 현장 cold-start TSAD 도입 의사결정

정상 데이터가 거의 없는 제조 현장에서 어떤 TSAD 계층과 모델을 언제 도입해야 하는지 판단하고,
마지막에는 성능·오탐·미탐·데이터 대기·학습·추론·자원 비용을 함께 넣은 목적함수로 운영안을
고르는 연구 계획이다.

## 0. 목적과 현재 범위

2026-09-12 현재 서버의 실제 가용 CPU 수를 PCA fit과 VUS-PR 병렬 채점에 적용한다.
종전 CPU 8개는 고정 상한으로 쓰지 않는다. CPU affinity·컨테이너 할당과 채점의 RAM·남은 작업
수를 반영하며 채점 worker 내부는 1스레드를 유지한다. PCA의 기존 1스레드 비교 기준,
타 모델의 실행 설정·후보·점수식은 유지하고 새 CPU 관측은 실행별 자원 기록에만 추가한다.

2026-09-12 사용자가 기존 기록을 활용한 PCA 기준 시간 복원을 요청했다. 기존 모델 시간과
비교할 값은 실제 8스레드 실행시간과 분리한다. 같은 조건의 1스레드 실측을 우선 연결하고,
없으면 기존 PCA의 스레드별 실측 비율 또는 입력별 설정 간 시간비로 추정해 출처를 남긴다.
추정값을 실측으로 표시하지 않으며 실제 HPO 개발비·점수·선택식은 유지한다.

2026-09-11 사용자 지시로 동일 L4·CPU 8개 환경에서 PCA fit에만 BLAS 스레드 8개를 허용한다.
다른 모델과 채점 worker의 스레드 설정, 후보·solver·점수식은 유지한다. 기존 완료 결과를
보존하며 새 실행의 실제 경과시간과 전체 스레드의 CPU 시간 합계를 구분해 기록한다. 후자는
1코어 환산 계산량으로만 사용하며 1스레드 실행시간이나 실제 장비 청구액으로 간주하지 않는다.
이 변경은 로컬 정적 검토 후 관련 원격 회귀·점수 대조를 거쳐 같은 예산으로 재개한다.

2026-09-10 사용자 지시로 RAM 사용률 80% 초과도 경고로 바꿨다. 메모리 총량의 작은 변동으로
완료한 자원 검사를 무효화하지 않으며 실제 실행 오류와 측정 근거 검증은 유지한다. 이번 검사
수정 전후의 결과를 이어 쓰되 모델·후보·배치·학습·선택식과 기존 실행 기록은 보존한다.

2026-09-09 다변량 One-Liners 앙상블의 상수 성분 점수로 실행이 중단됐다. 범위가 정확히 0인
성분·채널의 정규화 결과만 0으로 두는 보완을 기록한다. 정상 범위의 공식 min-max, 후보·채널·
결합식은 유지한다. 현재 게이트는 수정·정적 검토 후 원격 회귀 전이다.

2026-09-09 Lightning에서 TSPulse 자원 검사를 통과한 뒤 PCA series 11의 0 가중치로 실행이
멈췄다. 원격 단일 파일에서 full SVD로 유한한 점수 9,500개를 확인했다. auto의 covariance_eigh가
0 가중치를 만들 때만 공식 지원 full SVD로 다시 계산하며 실제 solver를 보존한다. 후보·성분 수·
전처리·점수식은 유지한다. 현재 게이트는 이 수정의 정적 검토 후 원격 회귀 전이다.

2026-09-09 배치 32의 실제 추론 완료와 최대 1.4692894585444094e-6 점수 차이를 확인했다.
사용자 요청에 따라 GPU 메모리 80% 초과는 경고로 기록하고 실제 OOM은 실패로 남긴다. RAM 기준은
유지하며 TSPulse 배치 비교는 PyTorch float32 기본 허용치를 적용한다. 현재는 수정·정적 검토 후
원격 검증 전이며 배치 32·공식 후보·점수식은 유지한다.

2026-09-09 사용자 요청으로 고정 디스크 여유 공간 기준을 제거했다. 현재 용량은 관측 정보로만
남기며 추정 용량이나 임의 여유율로 튜닝을 차단하지 않는다. 학습 증거 저장과 종료 이력 저장이
함께 실패해도 자동 재학습하지 않도록 즉시 재시도와 재기동의 판정을 맞췄다. 원격 검증 전이다.

2026-09-09 마지막 전면 감사에서 확인한 세 연결 오류를 보완했다. GDN 사전 검사를 scalar·
채널별 네 점수 파일에 맞췄고, 학습 증거 저장 실패 뒤 재기동에서도 자동 재학습을 막는다.
실제 채점에 쓴 ℓ_max snapshot과 공식 VUS 대조 보고서를 인수 묶음에 연결한다. 모델·후보·
선택식·DB schema는 유지하며 관련 원격 검증 뒤 새 코드·환경·예산을 봉인한다.

2026-09-09 최종 감사 요청으로 TSPulse의 실제 정규화 수치와 학습 파일의 부분 저장 참조를
보존한다. PCA는 보수적 RAM 추정 초과만으로 차단하지 않고 기존 자원 검사에서 대표 설정의
원격 CPU 실측을 확인한다. RAM 80% 미만 기준과 공식 후보·학습식·점수·선택식은 유지하며
실측은 해당 대표 실행의 근거로만 쓴다. 구현·정적 검토 후 관련 원격 검증과 새 봉인을 거친다.

2026-09-09 최종 검토 보완으로 PaAno·GDN의 학습 완료 증거를 추론 전에 시도별로 보존한다.
추론 실패 뒤에도 checkpoint·loss·학습 시간이 남고 정상 완료는 같은 파일을 재사용한다.
실제 채점 worker 수와 CPU 할당 정보도 명령 이력에 남긴다. 기존 trial 단위 재개와 학습식·
후보·선택식은 유지하며, 수정된 저장·인수 경로를 원격에서 검증한 뒤 새로 봉인한다.

2026-09-09 사용자 지시로 연결된 공식 저장소의 최신 기본 브랜치 commit을 기준으로
튜닝 절차와 풀을 대조한다. 일치하는 구현은 유지하고 다른 부분만 공식 코드에 맞춘다.
이 지시는 이전 논문 우선·PaAno memory 최소 500 제거보다 우선한다. 공식 코드가 공개하지
않은 풀이나 선택식은 추정하지 않으며 논문 근거와 기존 승인 보완을 따로 기록한다.
q·모델별 독립 선택, Dev18·holdout 격리와 로컬 실행 금지는 유지한다.
이번 튜닝 환경은 L4 24GB 한 장과 CPU 8개다. 학습 후보와 수식을 줄이지 않고 PaAno 배치
추론·대표 memory 탐색, PCA 거리 계산의 메모리 사용을 줄인다. 채점 worker는 최대 8개,
worker 내부 계산 thread는 1개로 제한한다. 실제 RAM 할당량은 원격에서 확인한다.

2026-09-07 지시로 공식 논문의 튜닝 후보와 절차를 포함하도록 전면 수정했다.
사용자는 모델별 검증 구간·조기 종료와 TSPulse 전체 평가 구간 통계 교정도 원문 방식으로
변경하도록 명시했다. 이 결정은 과거 validation 없음·strict 전처리 고정보다 우선한다.
최신 사용자 지시로 ALoRa와 전용 코드·실행 경로를 제외한다. 대체 모델은 추가하지 않으며
나머지 모델의 공식 논문·구현 기반 후보 풀과 q별 독립 선택을 유지한다.
q별 독립 선택과 Dev18/GHL·HAI 분리는 유지하며 공식 절차의 평가 구간 사용 범위를 결과에 명시한다.
실행 분기를 식별하는 `paper_tuning_v4` 이름은 유지한다. 최신 source commit·recipe·
hyperparameters는 config_id에 반영하며 기존 점수·checkpoint·선택표를 승계하지 않는다.

이번 구현은 공식 비교 구성을 포함한 Tier 1 후보, 모델별 학습·checkpoint 절차, Tier 3 공식
후처리와 ensemble, q별 선택·저장·재개 검증을 함께 바꾼다. 확인되지 않은 공식 자동 선택식은
추정하지 않고 필요한 근거가 없다는 상태를 남긴다. 회귀는 작성하되 로컬 테스트·Python·YAML
파싱·데이터 스캔·모델 실행은 하지 않는다. 구현과 정적 검토 후 원격 검증을 거쳐 새 예산을 봉인한다.

`full_prefix_v2`는 각 CSV의 현재 q-prefix 전부를 모델에 전달하며, `paper_tuning_v4`에서
모델 내부 검증 분할을 허용한다. 모델·q·실행 조건 집단마다 설정을 따로 선택한다.
q60과 q80의 PaAno 설정은 달라도 된다.
전체 q 교집합이나 18개 중 가장 짧은 파일을 기준으로 모든 후보를 배제하지 않는다.

2026-09-06 최신 지시로 과거 80/20·과거 후보 풀·스케일러의 튜닝 결과는 폐기하고 전부 다시
실행한다. Downloads의 과거 결과 묶음도 새 실행의 의존성에서 제외한다. 과거 추가 실행
계획과 고정 membership 행수는 사용하지 않는다. 새 예산은 수정된 registry와
CSV/q별 구조 조건으로 봉인한다. 과거 결과를 보존·보완하던 이전 지시보다 이 결정이 우선한다.

본실험은 GHL과 HAI다. 비-GHL Dev18은 사전 튜닝 패널이며 최종 성능의 근거가 아니다.
`source_faithful_v3`의 모델별 전처리·checkpoint·후보 수정은 정적 검토를 마쳤고 원격 검증 전이다.
2026-09-07 [전면 재튜닝 전 수집 프롬프트](tuning_feature_capture_prompt.md)에 따라 추천용
feature 수집과 SQLite 저장 코드를 반영했다. 입력 feature 검증을 본 튜닝의 시작 조건으로,
전체 후보·실행 증거와 추천 자료 인수를 최종 완료 조건으로 연결했다. 정적 검토만 마쳤으며
원격 검증 전에는 본 튜닝을 시작하지 않는다. 중단 재개·비용 이력은 같은 새 실험 안에서 사용한다.
저장 계약은 [사용자 엑셀 첨부본](tuning_storage_format.xlsx)의 네 묶음을 기준으로 한다.
새 full-prefix 저장의 중복 fit/validation 행 수를 없애고, CSV/q별 특징을 한 번 계산해
여러 모델·설정·seed·head가 재사용한다. 2026-09-09 사용자 요청으로 기존 입력 통계에
IQR·차분 절댓값 Q90/IQR·전후반 중앙값 이동/IQR·spectral entropy를 더한다. 채널별 값과
유효 채널 중앙값·NULL 사유를 SQLite 원표에 저장하고 기존 CSV 열 뒤에 추가한다.
extractor는 `prefix_features.v2`, 추천 DB schema는 3으로 구분하며 이전 DB를 덮어쓰지 않는다.
단위 검토에 따라 추천용 VIEW·CSV에는 원시 진폭 대신 무차원 특징과 채널별 IQR/std
중앙값·상수 비율·유효 비율·NULL 사유를 연결한다. 기존 단일 튜닝 명령에서 자동으로
적재·백업·내보내며, 열의 역할과 단위·표본 간격·fold 검증 계약을 영수증에 보존한다.
q가 다른 통계는 복사하지 않는다. 모델의 학습·추론·선택식은 이 수집 변경과 별개다.
성능표 기본 11열은 유지하며 score head는 DB 내부 키와 CSV/head별 출력으로 구분한다.
튜닝의 관측 범위를 서비스 지원 인증으로 해석하지 않는다. 최신 요청에 따라
로컬 실행 검증은 보류한다. 새 모델 학습, VUS 채점과 그림 생성은 원격에서
수행하며, GHL·HAI의 최종 인수 게이트는 별도로 유지한다.

2026-09-06 추가 요청으로 서비스 상한 근거의 보류를 해제한다. q별 후보·전체 prefix·조건집단·
family-LOFO는 유지하고, 완료된 튜닝 점수와 실행 증거에서 모델별 실제 관측 조합을 추출한다.
상한은 센서 수와 실행 환경에 따른 관측 범위로 남기며 행·열의 개별 최댓값으로 직사각형을 만들지
않는다. 같은 q라도 n·d·N이 다르면 기존 결과를 옮겨 추천하지 않는다. 기업의 N은 계획량이며
학습에 충분한 양이라는 뜻이 아니다. 운영 성능 하한·최종 threshold·비용 최적화는 후속 담당 범위다.

## 1. 최종 연구 질문

`Q={005,010,020,040,060,080,100}`를 유지한다. 파일 i의 100%는 봉인된 학습 구간 길이 N_i이며,
q%는 그 앞쪽 `floor(N_i*q/100)`행이다. 파일 수의 비율이나 전체 CSV의 테스트 포함 길이가 아니다.

각 q에서 실행 가능한 모델·파라미터를 따로 골랐을 때의 성능과 실행 비용을 비교한다. 이 곡선에는
데이터 증가와 설정 변경 효과가 함께 들어간다. 데이터 양만의 효과로 해석하지 않는다.
과거 고정 설정 곡선은 새 분석의 통제 결과로 쓰지 않는다. 새 분석에 필요한 비교값도 새 점수에서 계산한다.

기업에는 q 하나만으로 답하지 않는다. 현재 학습 가능 행수 n, 센서 수 d, 예상 최종 행수 N과
지원 상한을 함께 검사하고, 근거가 맞는 설정을 후보로 제시한다. 비용 최적 경로는 오탐·미탐과
현장 단가가 준비된 뒤 계산한다.

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
| 지우 | 공통 VUS-PR, `ℓ_max`, threshold 격자, 튜닝 원표와 선택표 | 튜닝 score ledger, 조건·q별 recipe 표, 최종 실행 요청 | 모델 코드와 runner 수정, 최종 통계 해석 |
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
| Tier 1 | 경량 통계·저비용 기준선 | `MWVAR`, `SQDIFF_LAST1`, `SQDIFF_LAST3`, `SQDIFF_CENTERED5`, 두 Var96 앙상블, `PCA_LEGACY` |
| Tier 2 | target 정상 prefix로 학습 | `PaAno`, `GDN` |
| Tier 3 | 가중치 학습 없는 공식 zero-shot | `TimeRCD`, `TSPulse` |

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

## 6. q-prefix와 모델별 공식 절차

`observed_row(i,q)=floor(N_i*q/100)`은 이용 가능한 정상 prefix 길이다. 공통 실행기는
`[0, observed_row)`를 모델에 전달하고 모델 내부에서 원문 절차를 적용한다. 고정 평가 tail은
`[N_i, 원본 CSV 끝)`이다. 실제 학습·검증 창과 평가 통계 출처를 관측 행 수와 구분해 기록한다.

PaAno는 native RevIN과 학습 loss checkpoint를 쓴다. memory는 최신 공식 코드의 10%와
최소 500개 규칙을 함께 적용하되 patch 수보다 작게 제한한다. 반올림·KMeans 구축 순서,
공식 batch와 patch 점수의 학습 문맥 연결을 유지한다. 현행 memory 정책은 `official_minimum`이다.

GDN은 fit-only MinMaxScaler 뒤 q-prefix의 window 중 연속된 검증 블록을 분리한다.
등록된 validation 비율·optimizer betas·patience를 쓰며 최저 검증 MSE checkpoint를 복원한다.
평가 채널 오차의 전체 median/IQR, 원본 trailing 4, max 집계가 끝난 점수를 반환한다.

PCA는 TSB-AD의 전체 평가 입력 fit과 window zero pruning을 복원한다. q-prefix를
학습하지 않는 실행으로 등록하되 metadata와 checkpoint에 평가 입력 fit 범위를 남긴다.
One-Liners의 앙상블도 원본의 전체 평가 component min-max를 사용한다.

TimeRCD는 전체 평가 입력 z-score, TSPulse는 전체 평가 StandardScaler와 공식 head별
경계 복원·정규화·smoothing·ensemble을 사용한다. 두 모델의 가중치는 갱신하지 않는다.
이 절차는 평가 파일 전체를 미리 보는 비인과적 처리다. 과거 문맥 제한 계약은 폐기한다.

각 adapter가 끝낸 native 점수에 공통 trailing 4나 median/IQR을 다시 적용하지 않는다.
저장 형식의 raw·smoothed 두 파일에는 같은 native 점수를 넣고 그 사실을 metadata에 적는다.
HAI는 기존 두 실행과 세션 경계를 유지한다. 입력·출력이 q와 무관한 모델의 한 점수는
일곱 q에서 참조하지만 q별 설정 선택은 따로 수행한다.
## 7. 정적 feasibility와 모델 함수 완료 조건

정적 feasibility는 점수와 라벨을 읽기 전에 계산한다. 단위는
`(model, config_id, q, tuning_series)`다. prefix·fit·validation·test 길이, window·patch 수,
채널 수, GDN top-k, HAI 다중 세션 지원 여부를 판정한다. 불가능한 조합은 0점으로 채우지 않고
`unavailable`과 이유를 남긴다.

모델 함수는 아래 조건을 모두 만족해야 완료로 센다.

1. 공식 source·license·checkpoint 신원이 snapshot에 고정돼 있다.
2. 합성 입력에서 score shape와 원시 시점 정렬이 맞는다.
3. 라벨·test threshold·point adjustment는 연속 점수 생성에 쓰지 않고, 공식 평가 통계의 출처를 기록한다.
4. 같은 seed와 입력으로 재현되며 실패·timeout을 성공 점수로 저장하지 않는다.
5. 학습, validation 추론, 테스트 추론, 전처리 시간과 peak memory, artifact 크기를 남긴다.
6. GHL 19채널과 HAI 86채널, HAI 세션 경계를 지원하는지 모델별로 판정한다.

이 검사는 합성 자료와 정적 정보로 먼저 닫는다. 실제 TSB·GHL·HAI 점수 생성은 다음 실행 게이트가
열린 뒤 시작한다. 공식값은 후보 grid의 출발점이지 GHL·HAI 최적값이라고 부르지 않는다.

## 8. TSB 기반 튜닝

실행 전 registry·입력·전처리·feasibility 코드와 후보 전체를 새 budget에 봉인한다. 새 프로토콜은
`full_prefix_per_ratio`이며 후보 수가 다른 전수 탐색이다. equal-trial 공정성을 주장하지 않는다.
stochastic seed는 0·1·2, deterministic은 development 첫 seed인 0으로 한 번 실행한다. 실패를 0점으로 바꾸지 않고 모든
필수 조합의 완료를 확인한 뒤 선택한다.

| 모델 | 공개 후보와 실행 절차 |
| --- | --- |
| One-Liners | MWVAR window 5·10·32·50·60·64·96·100·256·512·1024, Last1·Last3·Centered5, Var96+Last3·Var96+Centered5를 포함한다. |
| PCA_LEGACY | n_components 0.25·0.5·0.75·None과 window 100, 전체 평가 입력 fit·zero pruning을 쓴다. |
| PaAno | 공식 HPO loop가 없어 연결 논문 B.1의 patch 32·64·96 × LR 0.001·0.0001·0.00001을 유지한다. 100 iterations·batch 512와 공식 memory 10%·최소 500개 규칙을 쓴다. |
| GDN | 공식 run.sh의 k5·30 epoch·validation0.2 한 조합과 논문 주요 값에 코드를 보완한 k15/k30·50 epoch·validation0.1 두 조합을 유지한다. 모두 batch32, 최신 train.py의 patience15·Adam beta2 0.999를 쓴다. 완결된 공식 HPO grid는 공개되지 않았다. |
| TimeRCD | multi checkpoint와 context 5000의 공식 주실험 경로다. |
| TSPulse | aggregation 64·96·128 × time·fft·pred·ensemble의 점수 풀을 보존한다. 공통 창을 고른 뒤 데이터셋별 head를 고른다. pred 대표값 제한과 진단 raw_max는 폐기한다. |

현행 registry는 36개 설정이며 head를 구분하면 45개 선택 항목이다. CSV/q별 구조 조건을
적용한 실제 실행량은 새 예산에서 정한다. 과거 점수·checkpoint·선택표·예산을 승계하지 않는다.
공식 저장소의 기본 실행 한 조합을 전체 HPO 풀로 오인하지 않는다. 공개된 후보·기존 보완
tuple은 유지하되 코드가 고정한 학습값은 최신 공식 값을 따른다. 별도 민감도 실험의 값을
근거 없이 HPO grid로 만들거나 짧은 파일에 맞춰 전체 풀을 축소하지 않는다.
GDN의 hidden 값은 원본 tuple에 남기지만 공식 out_layer_num=1에서는 사용되지 않는다.
이를 실제 hidden 폭 탐색으로 세지 않으며, 세 조합은 embedding·topk·학습 설정으로 구분된다.

TSPulse는 각 q의 완전한 Dev18 TSPulse 패널에서 파일별 time·fft VUS-PR 평균을 구하고,
파일 간 동일 가중 평균이 가장 큰 공통 창을 고른다. 창 평균은 반올림하지 않으며 최고값과
1e-6 이내이면 작은 창을 택한다. 이 식은 2026-09-08 사용자가 승인한 프로젝트 보완이며,
공개되지 않은 논문 원식으로 부르지 않는다. 공식 실행은 aggregation 기본값 96을 쓰되 다른
값도 받는다. 선택한 창에서는 공식 CSV처럼 파일별 점수를 소수점 5자리로 반올림한 뒤
데이터셋별 파일 평균으로 head를 고른다. 동률은 프로젝트가 고정한 time·fft·pred·ensemble
순서다. 공식 코드의 비정렬 파일 순서는 재현 가능한 규칙을 보장하지 않는다.
관측하지 않은 데이터셋은 공식 fallback인 time을 쓴다.
주 정책은 q별 공통 창과 데이터셋별 head 표를 공유한다. LOFO의 모델·계층·공통 파일 비교는
각 fold의 학습 파일에서 두 단계를 모두 다시 계산해 holdout 점수가 섞이지 않게 한다.
정책은 데이터셋별 head 표와 fallback을 보존하고 실제 점수 파일을 참조할 때 native head로
해석한다. 이 참조는 새 head나 물리 실행을 추가하지 않는다.
Dev18·q·고정 평가 tail과 아래의 계층 비교는 사용자 연구 설계다. 원본 논문의 벤치마크 수치
자체를 재현한다고 부르지 않는다.
모델/q마다 실행 가능한 후보 집합이 같은 CSV를 하나의 조건 집단으로 묶는다. 후보 비교는 그
집단의 같은 파일에서 한다. seed 평균 → family 안 파일 평균 → 포함된 family 동일 가중 평균
순서다. 이는 공식 저장소에 공통 selector가 없어 유지하는 프로젝트 선택식이다. PCA 공식
HPO는 파일별 후보 점수를 저장할 뿐 최종값을 고르는 집계식을 공개하지 않는다. PaAno의
파일·Category 평균도 결과 보고용이며 선택식이 아니다. 앞서 명시한 TSPulse 내부 두 단계는
파일 동일 가중 평균을 쓰며, 그 결과의 모델·계층
비교에는 이 family 평균을 적용한다. 파일별 우승 점수만 모아 평균하지 않는다. 채널 scalar/max
집계와 CSV 간 평균은 별개다.
각 CSV에서 계산을 마친 VUS-PR만 합치며 서로 다른 CSV의 원시 점수를 이어 붙여 채점하지 않는다.
후보 원표와 모델별 튜닝 CSV에는 주선택값 `family_macro_vus_pr`, 같은 파일을 동일 가중한
보조값 `series_macro_vus_pr`, `file_count`와 `family_count`를 함께 저장한다. 두 평균은 같은
모델·설정·q·head·조건 집단의 파일별 seed 평균을 사용하며 보조값으로 우승 설정을 바꾸지 않는다.

family-LOFO는 각 fold에서 holdout family를 제외하고 설정을 선택한 뒤 holdout 점수를 평가한다.
한 family만 남은 집단은 선택 점수와 `insufficient_families`를 표시하고 일반화 검증으로 부르지
않는다. 실제 전달 설정은 Dev18 해당 집단 전체로 다시 고른다. 동률 허용은 1e-6이며
(model, config_id, score_variant) 사전순으로 정한다.

`tier_adaptive`도 tier/q마다 모델별 후보 집합의 조합이 같은 파일을 묶고 모델+설정을 함께
선택한다. 서로 다른 크기의 모델 패널 평균을 바로 비교하지 않는다. 모델 간 보조 비교는 겹치는
같은 파일에서 LOFO를 다시 계산한다. 선택표에는 q, 집단, 파일 목록, 후보 목록, 실제 prefix와
N·센서 수 범위, 선택 설정과 검증 상태를 함께 남긴다.

튜닝 진입점은 `tests/ghl_main/run_ratio_tuning.py` 하나다. 기본 실행은 후보 봉인, 누락된
합성·checkpoint·자원 검사, 환경 봉인, 모델 실행, VUS 채점, q별 선택과 근거 저장을 잇는다.
승인한 입력 inventory·VUS 검증·ell_max 및 설치 환경은 선행 인수물이다. 진입점이 대신
만들거나 기준을 완화하지 않는다. 기존 실패 검사와 완료 영수증은 보존하고, 통과 기록의
신원이 바뀌면 중단한다. 재실행 명령과 원격 검증 조건은 `lightning_studio.md`를 따른다.

자원 probe별 완료 이력을 보존하며 재사용 조건에 패키지 환경·GPU·RAM을 포함한다. 채점 중단은
이미 시작한 worker가 끝난 뒤 확정하고, 보고서 CSV는 원자적으로 교체한다. 모델 실행 후 저장
실패를 학습 실패로 재시도하지 않는다. 당시 snapshot과 반환된 timing은 UUID 이력에 남기며
명령 시간과 중복 합산하지 않는다. 강제 종료로 측정하지 못한 비용은 미확정으로 둔다.

완료 모델·VUS checkpoint는 검증 후 재사용한다. 학습형 모델의 미완료 trial은 처음부터
다시 실행하며 이전 시도의 시간과 횟수를
남긴다. 강제 종료로 종료 시각을 모르면 총비용을 확정하지 않는다. 명령 시간과 그 안의 모델
시도 시간은 겹치므로 합산하지 않는다. 이는 Dev18 개발 비용이며 GHL·HAI의 비용을 대신하지 않는다.

규모 근거에는 model_ratio와 tier_adaptive의 선택 합집합을 출처별로 남긴다. 같은 실행과
같은 추론의 여러 head를 독립 물리 실행으로 합산하지 않는다. 비선택 후보의 성능은 전체
ledger·candidate_audit에 보존한다. 규모표는 관측 기록이며 기업 응답 범위의 인증이 아니다.

## 9. GHL과 HAI 본실험

조건별 선택표를 최종 파일의 metadata와 대조해 `final_policy_membership.csv`를 만든다.
`model_ratio`와 `tier_adaptive` 행마다 split·series·q·group_id를 고정한다. 최종 파일의 실행
가능 후보 집합과 맞는 개발 집단이 없으면 unavailable이다. 최종 라벨로 집단이나 설정을 고르지 않는다.

GHL·HAI처럼 개발 패널보다 큰 입력은 `out_of_dev_support`로 표시해 외부 검증 대상으로 실행한다.
이 표시는 기업 추천 범위를 이미 검증했다는 뜻이 아니다. 최종 EDA 인수와 평가 게이트는 실행 전에
닫아야 한다. GHL stochastic seed는 3·4·5·6·7이다. HAI의 두 실행은 독립적으로 저장한다.

runner는 membership에 지정한 series만 실행한다. 같은 model/config/q/seed 점수를 여러 정책이
쓰면 한 번만 생성하고 참조한다. Tier 공동 선택이 모델별 선택과 다른 설정을 고르면 필요한 실행을
합집합에 추가하며, 물리 실행이 전혀 늘지 않는다고 가정하지 않는다.

## 10. 점수와 실행 증거

주지표는 threshold 전 `raw__trainnorm` 연속 점수다. 파일명·source_start/end·label_slice를
유지하며 모델별 native 후처리 뒤 공통 smoothing을 덧붙이지 않는다. `trainnorm` 파일명은 기존 인터페이스를
위해 유지하며 실제 교정 여부는 `calibration_mode`와 `normalization_scope`로 구분한다.
PaAno의 공식 scalar 점수를 임의로 채널 max로 바꾸지 않는다.

config_id에는 source/checkpoint·파라미터·전처리와 새 common_recipe가 들어간다. dataset/series/q/
seed는 넣지 않으므로 같은 설정의 식별자는 같고 물리 실행 키는 별도로 관리한다. 새 recipe로 인해
기존 80/20 config와 충돌하지 않는다. 점수·metadata·학습 snapshot·checkpoint SHA가 모두 맞아야
완료로 인정한다. 완료한 새 조합은 재실행하지 않고 VUS checkpoint도 이어서 사용한다.

새 측정 ID는 `dev18_registered_runner.full_prefix_v3`다. 전처리·setup·학습·calibration 추론·
test 추론을 기록한다. GDN의 내부 검증은 학습 절차와 training_protocol에 기록한다. GPU peak,
artifact 크기, 세션 관측량, 실패와 재시도도 보존한다. 원본 시간 근거가 없으면 행수를 초로 바꾸지 않는다.

재개 보고는 전체 계획량과 이번 명령 시작 전 미완료 수를 구분한다. panel 전체 시간과 별개로
완료 결과 확인 시간, 모델 계산·저장 시도 시간을 기록한다. 시도 시간에는 실패·재시도를 포함하며
같은 물리 실행을 head나 q별로 중복 합산하지 않는다. 실패한 시도에서 완료한 결과도 보존하므로 그
시도 시간을 전부 낭비 비용으로 해석하지 않는다. 명령·시도 이력은 HPO 개발비이며 현장 비용은 아니다.

과거 점수·manifest·ledger는 폐기하고 새 namespace `full_prefix_v2`에서 전부 다시 만든다.
학습형과 target-free 모두 새 실행 결과를 사용한다. 새로 만든 target-free 점수의 q 간 공유와
같은 새 실험의 중단 재개만 허용하며 과거 완료 파일은 승계하지 않는다.

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

본실험은 비용함수를 미리 최적화하지 않는다. 각 q의 후보 성능과 비용 원자료를 만들고 기존
고정 recipe는 통제 비교로 보존한다. 마지막 단계의 행동 단위는 아래처럼 정의한다.

```text
a = (tier, model, config_id, score_variant, q, operating_threshold)
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
  + TransitionCost(previous_action, a)
```

HPO 비용은 모델을 채택하기 전의 개발비로 따로 보고한다. 반복되는 현장 재학습·추론비와 섞지
않는다. `ObservationCost(q)`에는 정상 데이터를 모으는 시간이나 생산 지연이 들어가지만, 실제
시간 근거가 없는 GHL에서는 관측량 대리값으로만 남긴다. `FalseAlarmCost`와 `MissCost`는
threshold별 confusion 결과와 현장 단가가 모두 있어야 계산한다.

전체 후보의 연속 점수·설정·실행 비용을 보존한다. VUS 우승표나 개발 Pareto 표로 후속 비용
최적화의 후보를 미리 삭제하지 않는다. threshold별 오탐·미탐, 현장 비용 가중치와 latency·memory·
최소 성능 제약이 확정된 뒤 `argmin_a TotalCost(a)`를 고른다. Pareto 제거는 그 목적과 모든
제약에서 열위가 증명된 후보에만 적용하며 원자료는 남긴다. 최종 데이터에서 일부 메뉴만
실행했다면 최적성도 그 메뉴 안으로 제한한다. 가중치에 따른 선택 변화도 보고한다.
`TransitionCost`의 모델 교체·검증·배포·중단 비용은 웹사이트가 현장 값을 받은 뒤에만 계산한다.
값이 없으면 0으로 채우지 않는다.

### 기업 입력과 추천 범위

현재 정상 학습에 쓸 수 있는 행수 n과 센서 수 d를 확인하고, 기업이 예상 최종 학습 행수 N을
입력한다. `0 < n <= N`, `0 < d`를 확인한다. 별도 운영 한도 R_max·C_max는 선택 입력이며
지정하면 함께 검사한다. 이 한도를 통과해도 튜닝 근거에 없는 조합은 후보로 반환하지 않는다.

기업의 N이 100%이며 R_max는 이를 허용하는 서비스 한도다. 가장 긴 CSV나 서로 다른 파일의
행·열 최댓값을 합쳐 상한을 정하지 않는다. 실행 환경과 시간·메모리 한도를 고정하고, 모델·센서
조건별 자원 경계와 독립 성능 근거를 함께 확인해 지원 범위를 봉인한다. 파일 길이의 존재나 각 축의
최소·최대 안에 있다는 사실만으로 그 조합의 성능을 검증했다고 보지 않는다.

후보 조회는 `n=floor(N*q/100)`인 등록 q만 사용한다. 최근접 q나 보간으로 미관측 prefix를
채우지 않는다. 기업의 n·d·N, 평가할 관측 수, 실행 환경이 완료된 튜닝의 같은 조합과 일치해야
한다. 설정·head·예산·선택표도 확인한다. q만 같거나 R_max 아래라는 이유로 추천하지 않는다.
같은 5%라도 현재 1,000,000행인 기업을 몇백 행의 5%와 같게 취급하지 않는다.

`--finish-only`와 `--selection-only`는 기존 완료 원표에서 `tuning_support.json`,
`conditional_selection.json`, `model_support_limits.csv`를 만든다. 저장 위치는 기존
`experiments/01_ghl_main/results/dev18_tuning/full_prefix_v2/`다. 완료 영수증은 JSON 지문도
포함한다. 규모 근거에는 q별 모델·설정·head, 실제 n·d·N, 평가 길이, 모든 예정 seed의 점수와
실행 증거를 연결한다. scalar 출력의 channel_count를 입력 센서 수로 쓰지 않는다.

모델별 표는 센서 수·N·평가 길이·환경별 관측 q와 빈 q를 보존한다. `complete_from_ratio`는
그 q부터 100%까지 등록 지점이 모두 있다는 뜻이며 연속 구간이나 순차 운영 성능의 보증은 아니다.
같은 조건의 최대 계획량은 요약값이다. 그 이하 모든 길이나 다른 센서 수를 지원한다고 해석하지
않는다. 기업 조회는 모델별 근거의 합집합을 쓰고 모델 비교는 기존 공통 파일 비교를 유지한다.

범위 안 조회도 현장 검증 후보다. `service_status=unvalidated`를 유지하고 현재 지점의 후보와
N까지 빠진 q를 나눠 반환한다. 전체 q가 있어도 도메인·센서 의미가 다른 기업의 성능을 보장하지
않는다. family-LOFO는 선택 절차의 평가이며 최종 선택 설정의 독립 검증값이 아니다. 기록된
단일 실행의 시간·CUDA 할당량 또는 CPU tracemalloc 값을 서비스 지연·전체 RAM 한도로 쓰지 않는다.
공유 실행과 target-free의 q 재사용은 실행 참조로 남기며 논리 점수 행마다 비용을 합산하지 않는다.

수치 상한을 서비스 계약으로 확정하려면 이 표에서 후보 조합을 정한 뒤 운영 성능 기준과 자원
한도를 사전에 명시하고 독립 자료·동시 실행 조건에서 확인한다. 기준이나 근거가 빠지면
미검증으로 남긴다. 그 결과가 없는 현재 코드에는 임의 성능 하한이나 인증 전환 기능을 넣지 않는다.
현장 단가와 오탐·미탐 근거가 없으면 수익성 최적이라고 표시하지 않는다.

### 고정 상한까지의 비용 경로

현재 데이터량에서 기업의 N(서비스 상한 R_max 이하)까지 지원되는 실측 지점을 순서대로 놓고, 상태에는 현재 모델·설정·
경보 기준과 마지막 학습 때의 데이터량을 포함한다. 각 지점에서 유지, 재학습, 모델·설정 전환을
비교한다. 경로 비용은 각 구간의 운영·오탐·미탐 비용과 실제로 발생한 학습·전환 비용의 합이다.
매 q의 VUS-PR 1등을 순서대로 채택하는 규칙으로 경로 최적화를 대신하지 않는다.

수집속도로 각 구간의 운영 기간을 정하고 현장 단가·경보 빈도를 적용한다. 같은 전체 test의
오탐·미탐 건수를 모든 구간에 중복해서 더하지 않는다. N 도달까지 평가할지, 도달 후의
운영 기간까지 포함할지도 비용 비교 전에 정한다. 미래 열 수가 달라지면 적용 범위를 다시 검사한다.

현재 독립 q 실험은 각 데이터량에서 새로 학습한 설정의 근거다. 기존 모델을 유지하는 경로는
오류율이 유지된다는 가정 아래 기존 점수로 비용 시나리오를 계산할 수 있다. 실제 시간 경과에
따른 성능 변화, 이어 학습의 절감 효과와 전환 지연을 실측했다고 주장하지 않는다. 그런 효과를
최종 결론에 포함할 때만 별도 시간순 평가를 추가한다.

### 기존 실험으로 보완할 평가와 추가 입력

| 구분 | 남은 작업 | 모델 추가 실행 |
| --- | --- | --- |
| 저장 점수와 정답 라벨 | 공통 평가 구간에서 threshold별 TP·FP·TN·FN, FPR·FNR·precision·recall과 경보 건수를 계산한다. 시점과 사건 단위를 구분하고 경보 묶기 규칙을 먼저 고정한다. | 불필요 |
| 운영 threshold용 별도 자료 | GDN의 내부 검증은 checkpoint 선택용이다. 별도 정상 교정 자료나 사전 고정 기준으로 threshold를 정하고 독립 구간에서 오류율을 확인한다. 모델 내부 검증과 평가 통계를 독립 운영 평가로 취급하지 않는다. | 기존 적합한 자료가 없으면 추가 필요 |
| 기존 ledger와 metadata | 절대 관측량·채널 수별 근거, seed별 변동, 학습·추론 시간과 같은 파일 조건의 모델 비교를 만든다. | 완료 조합에는 불필요 |
| 비용 후보 선정 | VUS-PR 1등 외에도 threshold별 오탐·미탐과 실행 비용을 비교해 남길 후보를 정한다. | 완료된 Dev18 조합에는 불필요, GHL·HAI에서 미실행 후보를 검증할 때만 필요 |
| 기업 입력 | 오경보 조사비, 이상 사건별 손실, 발생 빈도·운영 기간, 수집속도, 자원 단가, 전환비와 기존 운영안의 비용·성능을 받는다. | 벤치마크 실험으로 대체 불가 |
| 새 환경의 성능 | 기업 데이터의 오탐·미탐, 실시간 지연, 새 장비의 처리량·메모리와 온라인 전환 효과를 검증한다. | 주장할 범위에 따라 새 추론·학습 또는 현장 시험 필요 |

새 튜닝으로 VUS 후보를 고르는 절차와 운영 threshold를 정하는 절차는 다르다. full-prefix
튜닝이 끝나도 최종 라벨로 threshold를 골라 독립 성능처럼 보고하지 않는다. 0.99 분위수는
미래 오탐률 1% 보장이 아니다. strict zero-shot에 target 교정을 추가하면 별도 운영 정책이다.
비용 후보와 threshold 선택 규칙은 GHL·HAI 결과를 보기 전에 봉인하고, 추가로 필요한 설정은
별도 실행 요청에 명시한다. 새 조건부 membership의 행수를 과거 462행으로 제한하지 않는다.

논문의 오탐률은 가정값을 바꿔 보는 시나리오에만 쓴다. 기업의 미라벨 데이터로는 경보량을
알 수 있지만 실제 오탐률·미탐률은 알 수 없다. 정상으로 확인한 별도 평가 구간은 오탐률을,
정답 이상 이력이 있는 구간은 미탐과 사건 탐지를 검증하는 근거가 된다. 논문 수치나 VUS-PR을
기업의 확정 손실률로 환산하지 않는다.

기록된 배치 시간은 같은 실행 환경의 비교 근거다. `model_artifact_bytes`가 0이어도 사전학습
모델의 배포 용량이 0이라는 뜻은 아니다. 필수 checkpoint·전처리 상태를 따로 집계한다. 저장
메모리 피크도 전체 시스템 메모리나 서빙 피크와 구분한다. `offline_noncausal` 점수의 탐지
위치를 실시간 지연으로 부르지 않으며 미래 문맥 대기와 처리 시간을 함께 검증한다.

최종 답은 검증 범위와 현장 입력을 전제로 한 후보 간 비용 비교다. 최소 성능을 만족하는
후보가 없으면 추천을 보류한다. 순이익 개선을 주장하려면 현재 운영안과 비교하고, 비용이나
발생 빈도가 미확정이면 선택이 바뀌는 구간을 제시한다. 독립 q 실험만으로 온라인 재학습·
전환 과정의 수익성을 입증했다고 주장하지 않는다.

## 13. 최소 산출물 계약

| 산출물 | 용도 |
| --- | --- |
| full_prefix_v2/budget.json·feasibility 원표 | 후보·집단·실제 CSV별 실행량과 코드 신원 봉인 |
| full_prefix_v2/recommendation_contract.json | 최소 특징 산식·저장 schema·예상 기록과 실험 신원 |
| recommendation_evidence/의 SQLite 백업·네 묶음 CSV·인수 영수증 | CSV/q별 특징과 전체 후보 결과 연결, head 구분·완전성 검증 |
| handoff/의 전달 묶음·인수 목록 | 작은 원표·DB·실행 이력과 큰 점수·checkpoint의 위치·SHA 연결 |
| dev18_full_prefix_v2_manifest.csv | 점수·metadata·완료와 실패 기록 |
| dev18_trial_score_ledger.csv | 기존 채점기와 정렬 절차로 계산한 전체 VUS 원표 |
| model_ratio_policy.csv·tier_adaptive.csv | 조건·q별 설정과 계층 대표 |
| tuning_support.json·conditional_selection.json·model_support_limits.csv | 완료 실행에 묶인 모델별 관측 조합·q 누락·규모 요약과 정확한 선택표 |
| ratio_family_lofo.csv·candidate_audit.csv·matched_model_comparison.csv | 선택 편향과 동일 파일 비교 근거 |
| structural_exclusions.csv | 불가능한 CSV·설정·q와 원인 |
| final_policy_membership.csv | 조건부 model_ratio·tier_adaptive의 split/series별 실행 합집합 |
| 모델별 CSV·PNG·완료 영수증 | 조건 집단을 유지한 비교와 실행/채점 경과시간 |

main ledger 행수와 physical_run_count는 실제 execution.series_ids 합으로 계산한다. head·q의
선택 항목과 물리 실행 수를 구분하고, 전체 결과에 무조건 18을 곱하지 않는다. GHL·HAI의
최종 원표와 통계·교차점 결과는 기존 실험 루트에 저장한다.

## 14. 프로젝트 단계와 현재 게이트

0·1단계의 입력·공식 source 근거는 유지하고 과거 2단계 튜닝 결과는 폐기한다. 현재 게이트는
2단계 전면 재튜닝 전 최신 공식 저장소·저장·인수 보완 반영과 정적 검토 완료, 원격 검증 전이다.
2026-09-09 여섯 저장소 최신 기본 브랜치와 36개 설정·45개 head 점수, q·모델별 독립 선택을
대조했다. PaAno memory와 GDN patience·Adam beta를 공식 값으로 복원하고 PCA·TSPulse의
source pin을 갱신했다. 최신 코드 우선 지시로 GDN graph·후처리의 논문 차이는 차단 사유가 아니다.
`paper_tuning_v4`와 추천용 feature·SQLite 저장의 원격 검증은 아직 남아 있다.
과거 검사를 현재 코드의 검증 결과로 쓰지 않는다.

적재 감사의 임시파일 재개·미배정 실패 이력 전달·인수 시간 기록·상관값 경계를 보완했다.
다음은 최신 source를 설치한 원격 환경의 관련 회귀와 소규모 저장·중단 재개·인수·L4 검사다.
Tier 1 평가 입력의 0범위 점수·퇴화 PCA 확인도 거친다. 이 검증을 마치고
새 코드·feature 계약·예산·환경을 봉인한다. 입력 feature 검증을 통과한 뒤 전체 튜닝과 채점을
처음부터 수행하며, 필수 추천 자료 인수까지 마쳐야 전체 완료로 판정한다.
새 실측 결과가 나오기 전에 GHL25(3단계), HAI(4단계), 기업 비용 최적화(5단계)로 넘어가지 않는다.
최종 EDA 인수·threshold·비용 후보 규칙과 현장 단가는 해당 단계의 미완료 과제로 남는다.

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
- GDN: https://github.com/d-ailin/GDN
- Time-RCD: https://github.com/thu-sail-lab/Time-RCD
- TSPulse: https://github.com/ibm-granite/granite-tsfm
