# 다음 세션 시작점

## 현재 게이트

2026-09-09 배치 32의 Lightning 보고서에서 TSPulse 세 창의 추론 완료를 확인했다. 최대 점수 차이는
1.4692894585444094e-6이며 너무 작은 절대 허용 오차로 세 설정 모두 실패했다. 창 128은 GPU
점유·예약 메모리 98.56%로 기존 80% 기준에도 걸렸다. 이번에는 실제 OOM이 없다.
배치 비교를 PyTorch float32 기본 허용치 `rtol=1.3e-6`, `atol=1e-5`로 바꾸고 GPU 80% 초과는
경고로 기록한다. 실제 OOM·실행 오류·큰 점수 차이·잘못된 관측값과 RAM 80% 이상은 계속 실패다.
배치 32와 후보·창·head·공식 점수식은 유지한다. 현재 게이트는 수정·정적 검토, 원격 확인 전이다.
로컬 Python·테스트·모델은 실행하지 않았다. 다음은 관련 회귀와 기존 실행 명령이다.
새 commit을 받을 때 아직 비어 있는 `recommendation_evidence/`와 `recommendation_contract.json`만
다시 만든다. 예산·feasibility·합성 및 checkpoint 보고서·원본 입력과 cache는 유지한다.
실패 자원 보고서와 probe 이력은 보존하며 기존 진입점이 새 코드로 자원 검사를 수행한다.

2026-09-09 Lightning의 Python 3.12.11을 고정 3.11.14 검사로 거부한 환경 준비 오류를 수정했다.
공식 의존성이 지원하는 Python 3.11.x·3.12.x를 허용하고 실제 patch 버전은 실행 환경에 봉인한다.
설치는 현재 터미널의 Python을 쓰며 CUDA 12.6 torch 빌드를 지정한다. 버전·설치 원본 오류는 한 번에
보고한다. 현재 게이트는 수정·정적 검토, 원격 확인 전이다. 로컬 Python·테스트는 실행하지 않았다.
다음은 수정본 설치와 `test_common_run_context`·`test_model_environment` 원격 회귀 후 사용자 요청의
기존 명령 재실행이다. 이번 실패는 예산·빈 추천 DB 준비 후, Tier 3 검사·환경 봉인·튜닝 전이다.
새 commit을 받으면 빈 추천 DB와 추천 계약, 이전 합성 검사 보고서만 지워 다시 만들고 예산·명령 이력·
원본 데이터·checkpoint cache는 보존한다. 전체 결과 초기화를 반복하지 않는다.

2026-09-09 Lightning 준비에서 감사 CSV의 Windows·Linux 줄바꿈 차이로 SHA 검사가 실패했다.
기존 줄바꿈 정규화 함수를 감사 CSV 세 개에 재사용하고, 검증된 기존 inventory SHA를 예산에
넘기도록 수정했다. 원본 데이터·코드·checkpoint의 바이트 검사는 유지한다. 세 원표의 줄바꿈별
SHA를 대조하고 회귀를 작성했으며 로컬 테스트는 실행하지 않았다. 다음은 이 수정의 원격 회귀와
기존 튜닝 명령 재실행이다. 이번 오류는 예산 생성 전이므로 결과 초기화는 다시 하지 않는다.

2026-09-09 사용자 요청으로 고정 25GiB 디스크 차단을 제거하고 현재 용량을 관측 정보로 남긴다.
사전 검사 보고서를 재사용해도 새 관측을 명령 이력에 저장하며 기존 보고서는 바꾸지 않는다.
디스크 조회 실패도 사유를 남기고 진행하되 실제 저장 오류는 기존대로 중단·보존한다.
학습 증거 저장과 종료 이력 저장이 함께 실패한 경우도 즉시 재학습하지 않도록 보완했다.
현재 게이트는 구현·정적 검토, 원격 검증 전이다. 다음은 `test_lightning_dev18_resource_tools`·
`test_run_history`·`test_full_prefix_execution_contract`와 기존 저장·재개·인수 검증이다.
통과한 코드·환경·새 예산을 봉인하며 로컬 실행과 본 튜닝은 하지 않는다.

2026-09-09 마지막 전면 감사의 수정 요청을 반영했다. GDN 사전 검사는 scalar·채널별
raw·smoothed 네 파일과 채널 max를 확인한다. 학습 증거 저장 실패·중단은 재기동에서도
자동 재학습하지 않으며 일반 학습·추론 실패의 trial 재시도는 유지한다. 실제 ℓ_max snapshot과
공식 VUS 대조 보고서를 채점 원표의 신원과 대조해 인수 묶음에 넣는다.
현재 게이트는 구현·정적 검토 완료, 원격 검증 전이다. 다음은 `test_model_smoke`·
`test_full_prefix_execution_contract`·`test_recommendation_handoff`·
`test_recommendation_storage_integration`을 포함한 기존 원격 회귀와 작은 저장·재개·인수 검사다.
모델·후보·선택식·DB schema와 로컬 실행 금지는 유지한다. 통과한 코드·환경·새 예산을
봉인하며 본 튜닝은 시작하지 않는다.

2026-09-09 최종 감사의 세 항목을 수정했다. TSPulse 입력·head·출력의 정규화 수치를
metadata에 보존하고 완료 검사·DB 검증에 연결했다. 학습 파일은 하나씩 저장 직후 이력에
기록해 부분 저장도 인수한다. PCA의 보수적 RAM 추정이 기준을 넘으면 대표 등록 설정의
원격 CPU 실측으로 판정하고 추정·실측 원본을 같은 자원 이력에 남긴다. RAM 80% 미만 기준,
공식 점수·학습식·후보·선택식·DB schema는 유지한다.
현재 게이트는 구현·정적 검토 완료, 원격 검증 전이다. `test_tspulse`·
`test_paper_tuning_output_contract`·`test_recommendation_evidence`·
`test_full_prefix_execution_contract`·`test_lightning_dev18_resource_tools`·
`test_recommendation_handoff`를 기존 원격 검증에 포함한다. 작은 저장·실패·재개·인수와
L4 자원 검사를 통과한 코드·환경·새 예산을 봉인하며 로컬 실행과 본 튜닝은 하지 않는다.

2026-09-09 최종 검토에서 확인한 학습 완료 증거와 실제 채점 자원 기록을 보완했다.
PaAno·GDN은 추론 전에 시도별 checkpoint·scaler·loss·학습 시간을 저장해 추론 실패 뒤에도
보존한다. 정상 완료는 기존 파일을 재사용하고 실패 시도의 학습 증거도 인수 목록에 연결한다.
실제 채점 worker 수와 CPU 할당 정보는 명령 이력에 남긴다. 학습식·후보·선택식·DB schema와
trial 단위 재개 정책은 유지한다. 현재 게이트는 구현·정적 검토이며 원격 검증 전이다.
`test_registered_executor`·`test_gdn_official`·`test_full_prefix_execution_contract`·
`test_recommendation_storage_integration`·`test_recommendation_handoff`·`test_scoring_environment`를
기존 원격 검증에 포함한다. 로컬 실행과 본 튜닝은 하지 않으며 검증 뒤 새 코드·환경·예산을 봉인한다.

2026-09-09 추가 수정 요청으로 GDN 집계 전 채널 점수와 실제 교정 통계를 보존하고 완료
검사·DB 검증·인수에 연결했다. 실행 구간의 RSS를 50ms 간격과 시작·종료 시점에 관측하며
최고 관측값·표본 수·오류·현재 프로세스 범위를 성공·실패 이력에 남긴다. 순간 peak나
자식 프로세스를 포함하는 정확한 RAM 상한으로 해석하지 않는다.
계산 완료 후 모든 출력은 남았으나 완료 행 기록이 끊긴 경우에도 metadata의 점수 지문과
시도 신원을 대조해 재학습 없이 복구한다. 실제 부분 저장과 지문 불일치는 보존하고 중단한다.
실제 writer→DB verifier→백업·인수의 합성 회귀도 추가했다. 현재 게이트는 구현·정적 검토,
원격 검증 전이다. `test_process_memory`·`test_recommendation_storage_integration`과 기존
`test_gdn_official`·`test_paper_tuning_output_contract`·`test_paper_tuning_pipeline`·
`test_full_prefix_execution_contract`·`test_recommendation_handoff`를 원격에서 확인한다.
후보·학습식·선택식·DB schema 3은 유지하고 통과한 코드·환경·새 예산을 봉인한다.
로컬 테스트·Python·파싱·데이터·모델 실행과 본 튜닝은 하지 않는다.

2026-09-09 최종 검토의 결함 수정 요청을 반영했다. 계산 완료 이력으로 저장 실패 뒤 중복
재학습을 막고 검증된 출력에서 영수증·manifest·DB 연결을 복구한다. 출력 저장이 덜 끝났으면
보존하고 중단한다. 실패 시 자원 계측과 PaAno 선택 iteration·두 학습 모델의 loss 이력을
남기며 Tier 3 공식 원본 참조와 같은 예산의 자원 probe 이력을 인수 묶음에 연결했다.
관련 회귀와 호출부를 정적으로 검토했으며 로컬에서는 실행하지 않았다. 현재 게이트는 결함
수정·정적 검토 완료, 원격 검증 전이다. 다음은 기존 저장 검사와 함께
`test_full_prefix_execution_contract`·`test_recommendation_handoff`·`test_paano`·
`test_gdn_official`·`test_dev18_tuning`을 원격에서 확인하는 작업이다. 후보·학습식·q별 선택·
DB schema 3은 유지하며 검증 뒤 새 코드·환경·예산을 봉인한다. 본 튜닝은 시작하지 않는다.

2026-09-09 사용자 요청으로 감사에서 발견한 보완을 반영했다. 산출물의 숨김 임시 경로만
Git 검사에서 제외하고 완료 영수증·metadata의 실패 정리를 보완했다. 예산 미배정 실패 이력은
별도로 전달하며 인수 검증·압축·해시는 `handoff/run_history/`에 시간·상태를 기록한다.
현재 인수 이력은 압축에서 제외한 뒤 닫아 함께 전달한다. 절대 상관값의 수치 경계도 보정했다.
현재 게이트는 발견사항 구현·정적 검토 완료, 원격 검증 전이다. 관련 회귀는 작성만 했다.
extractor v2·DB schema 3은 유지하지만 계약 내용·소스 신원이 바뀌므로 새로 봉인한다.
SQLite·모델·후보·q별 선택은 유지했다. 상세 내용은 [사전 점검 결과](process_0_preverify.md)에 남겼다.

2026-09-09 추천 DB 보완 요청은 SQL·SQLite·특징 수집 코드에 반영했다. 기존 키를
확인하고 완료 증거의 빈 값을 거부하며, IQR·차분 Q90/IQR·중앙값 이동/IQR·spectral
entropy를 채널별·prefix별로 저장한다. 산식은 `prefix_features.v2`, 추천 DB schema는 3이다.
단위 검토 후 `recommendation_inputs` VIEW·CSV에 채널별 IQR/std 중앙값, 무차원 특징,
상수·유효값 비율과 NULL 사유를 연결했다. 원시 진폭 통계는 기본 추천 입력에서 제외한다.
기존 단일 진입점의 튜닝·채점·DB 적재·백업 경로는 유지하고 실제 SQLite를 쓰는 회귀로
내보내기 실패 후 재개와 모델 중복 실행 방지를 검사하도록 했다.
회귀는 작성했으며 로컬 실행은 하지 않았다. 이 변경의 다음 검증은 원격의
`test_prefix_features`·`test_recommendation_evidence`·`test_recommendation_command`와 기존 저장·재개·인수 검사다.
이전 DB를 덮어쓰지 않고 새 신원으로 봉인한다. 아래 공식 소스 대조 작업과 본 튜닝
게이트는 별도로 유지한다.

2026-09-09 추가 지시로 공식 저장소 최신 commit을 우선해 필요한 수정을 마쳤다.
여섯 저장소의 최신 기본 브랜치·후보·license를 대조했고 36개 설정·45개 head 점수와
q·모델·조건집단별 독립 선택을 유지한다. GDN graph·후처리와 PaAno negative는 최신
공식 코드와 같아 그대로 두었다. 이전 논문·코드 충돌의 승인 대기는 해제됐다.
공식 소스 반영은 구현·정적 검토를 마쳤고 위 저장·인수 보완과 함께 원격 검증 전이다.
본 튜닝은 시작하지 않는다.

TSPulse는 각 q의 전체 개발 패널에서 파일별 time·fft VUS-PR 평균을 파일 간 동일 가중
평균해 창을 고른다. 선택한 창에서 공식 데이터셋별 head를 고르며, 미관측 데이터셋은 time을
쓴다. 이 창 집계식은 사용자 승인 보완이다. LOFO의 모든 비교는 학습 파일만으로 두 단계를
다시 선택한다. 저장·조회·실행 요청은 데이터셋별 head 표를 실제 native 점수에 연결한다.

PaAno는 공식 memory 최소값·10%·patch 상한을 복원하고 `paano_official_minimum_v5`로
식별한다. `paper_fraction`은 명시한 역사 정책과 checkpoint에서만 보존한다. GDN의 세
후보 모두 patience15·Adam betas 0.9·0.999를 쓴다. hidden은 out_layer_num=1에서 쓰이지 않는다.
PCA source는 `6beac72e11d1155ade40870492c00d0d1cfdcaaf`, TSPulse source는
`fe7a35697723e2a2f5246ae979474bfc554e26c0`으로 맞췄다. 두 모델의 관련 계산 코드와 license는
이전 봉인과 같고 checkpoint도 바뀌지 않았다. 자세한 대조는 [사전 점검 결과](process_0_preverify.md)에 남겼다.

L4 24GB 한 장·CPU 8개 기준의 배치 추론·거리 분할·worker 제한과 RAM 검사는 유지한다.
호출부·원식·회귀를 정적으로 교차 검토했다. 관련 회귀는 작성만 했으며 로컬 테스트·Python·
YAML 파싱·데이터·모델은 실행하지 않았다. 실제 L4 실행 가능성은 아직 확인 전이다.

다음 시작점은 최신 source를 설치한 Lightning 환경의 관련 원격 회귀와 작은 합성·저장·
중단 재개·인수·L4 자원 검사다. 위 임시파일·인수 실패·상관 경계 회귀도 함께 확인한다.
Tier 1 평가 입력의 0범위 component 점수·퇴화 PCA도 확인한다.
통과한 코드·환경과 full_prefix_tspulse_two_stage_family_lofo_v3의 새 예산을 봉인한다.
이전 TSPulse 평균식 승인 대기는 해제됐으며, 실제 튜닝은 별도 실행 요청 후 시작한다.

ER diagram 담당자에게 전달할 [입력 특징·모델 파라미터 명세](model_parameter_storage_spec.md)를
추가했다. 현행 파라미터의 자료형·후보값·고정값과 실행·선택 결과의 키를 구분했다.
조회 열 분리는 ERD 설계까지 정리했으며 DB schema나 튜닝 코드를 변경하지 않았다. 저장 구현을
진행할 때 현재 JSON·writer·reader와 대응시킨 뒤 원격 검증한다. 현재 실행 게이트는 유지한다.

2026-09-09 사용자 요청으로 [ERDCloud 설계 SQL](erdcloud_schema.sql)과
[엑셀 대응 안내](erdcloud_schema_guide.md)를 간소화했다. 기존 35개 표를 엑셀 중심 핵심 6개와
모델별 파라미터 8개, 총 14개로 교체했다. 실험당 DB 하나를 기준으로 하며, 선택·LOFO·최종
요청은 기존 원표와 JSON을 참조한다. 입력·설정·seed·head의 결과 키와 실제 실행 비용 공유는
유지했다. DB·튜닝 코드는 바꾸지 않았으며 ERDCloud 가져오기와 DB 실행 검증은 아직 하지 않았다.

## 직전 게이트

2026-09-07 최신 사용자 지시로 ALoRa를 활성 후보와 전용 코드·폴더·실행·재개·자원 점검에서
제외했다. 대체 모델은 추가하지 않는다. Tier 2는 PaAno·GDN이며 남은 공식 논문·구현 기반
풀은 36개 설정·45개 head별 선택 항목이다. 논문 최종값 하나로 추가 축소하지 않았다.
기존 inventory와 구조 조건에서 예상한 물리 실행은 4,092회다. 재시도·사전 점검은 제외한
값이며 원격에서 새 예산을 봉인하기 전에는 확정 실행 수나 시간으로 쓰지 않는다.

후보 확장 → 실제 호출 인자 → 모델 내부 학습·추론 → 저장·완료 검사 → q별 선택을 대조했다.
GDN과 두 One-Liner 앙상블의 calibration 표기를 실제 출력과 맞췄으며 native 후처리는 유지한다.
실제 registry를 읽는 완료 검사 회귀와 남은 후보·q별 설정 변경 회귀를 작성했다.
ALoRa 제거와 함께 새 Tier 1 변형의 source 신원·smoke 누락도 보완했다.

paper_tuning_v4의 내부 검증·checkpoint·전체 평가 통계·native 후처리, Dev18 분리와
q별 독립 선택을 유지한다. GDN의 세 조합은 공개 run.sh 하나와 논문 주요 값에 공개 코드
기본값을 보완한 두 조합이다. TSPulse의 공통 aggregation 선택 세부식은 공개되지 않았으므로
임의로 복제하지 않는다. Dev18의 q별 조건 집단 선택은 프로젝트 설계다.

직전 SQLite transaction·중단 재개 수정과 CSV→family 집계 보강은 유지했다.
로컬에서는 텍스트·호출부·diff만 확인하며 테스트·Python·YAML 파싱·데이터·모델을 실행하지 않는다.
현재 게이트는 ALoRa 제거·남은 풀 점검의 정적 검토이며 원격 검증 전이다.

다음 시작점은 lightning_studio.md에 적은 관련 원격 회귀와 작은 합성·저장·재개 점검이다.
이 검증을 통과한 뒤 코드·저장 계약·환경·새 예산을 봉인한다. 본 튜닝은 별도 요청 후 실행한다.
아래 기록의 ALoRa·옛 후보 수·예산·완료 판정은 당시 이력이며 현행 계약으로 쓰지 않는다.

## 직전 수집·저장 게이트 기록 — 현행 후보·절차 아님

2026-09-06 최신 지시로 과거 튜닝 결과를 폐기하고 수정된 후보 풀·스케일러·전체 prefix로
모두 다시 실행한다. 먼저 추천용 feature 수집 코드를 완성한다. 2026-09-07 사용자가
[수집 코드 수정 프롬프트](tuning_feature_capture_prompt.md)의 구현을 요청했다. 현재 게이트는
수집·SQLite 저장 코드 반영·정적 검토 완료이며 원격 실행 검증 전이다.
Downloads 결과 복구나 과거 점수 승계는 하지 않는다. 원본 데이터·공식 사전학습 가중치와
기존 미커밋 코드 수정은 유지한다. 다운로드 삭제·실험은 수행하지 않는다.

CSV/q별 최소 특징을 모델 실행 전에 계산하고 SQLite에 한 번 저장한다. 재개하면 검증한
저장값을 읽으며, 전부 제외된 CSV/q도 특징을 남긴다. 성능·입력·채널·전체 설정의 네 묶음은
UTF-8 BOM CSV로 내보내고 성능표 11열은 CSV/head별로 구분한다. 원본 엑셀은 바꾸지 않았다.

새 저장은 `full_prefix_storage.v1`, 실행 증거는 `dev18_registered_runner.full_prefix_v3`,
예산 schema는 3이다. metadata·snapshot·예산·새 feasibility 원표·선택 근거의 관측 행 수는 `observed_row`로
맞췄고 과거 reader는 유지했다. 실제 backend·메모리 측정 범위와 공유 실행 이력도 연결한다.
실행만 끝나면 `execution_complete`, 채점·추천 자료까지 끝나면 `ready_for_handoff`다.
일관된 DB 백업과 전달 묶음의 인수 파일까지 만들어야 전체 `complete`로 표시한다.

관련 회귀는 작성만 했다. 로컬 테스트·Python 구문 검사·YAML 파싱·원본 CSV 스캔·모델·
checkpoint·VUS·그림은 실행하지 않았다. 다음 작업은 원격의 관련 회귀와 작은 저장 경로
검증이다. 통과한 코드·저장 계약·새 예산·환경을 봉인한 뒤 별도 명령으로 전면 튜닝한다.
후보 값과 선택식은 유지했으며 152개 설정·156개 주 선택 항목의 실제 실행량은 새 예산에서
확인한다. GHL·HAI 최종 실행 게이트도 그대로다.

## 직전 게이트 기록

2026-09-06 추가 요청으로 튜닝 중단·재개의 남은 저장 경로를 수정했다. 채점 worker 종료 대기,
보고서 CSV의 원자 저장, 일반 모델의 저장 실패 후 자동 재학습 방지, 완료 자원 probe 재사용을
반영했다. 일반 모델과 ALoRa의 당시 snapshot 내용·반환된 timing도 UUID 이력에 보존한다.
후보·q별 선택식은 유지하며 GHL·HAI 내부는 다루지 않았다. 현재 게이트는 정적 검토 완료·
원격 검증 전이다. 실행 금지는 유지한다.

2026-09-06 원본 후보 보존과 단일 진입 튜닝의 수정·정적 검토를 마쳤다. 원격 실행 검증 전이다.
MWVAR window 64, ALoRa batch 128, GDN의 공식 topk 5·batch 32와 저채널용 topk 2를 추가했다.
현재 144개 설정이며 TSPulse head를 구분하면 153개 선택 항목이다. 실제 실행 수는 새로 봉인한다.

`run_ratio_tuning.py` 기본 실행은 준비·사전 검사·환경 봉인·실행·채점·q별 선택·근거 저장을 잇는다.
model_ratio·tier_adaptive의 근거를 함께 보존한다. 시작 이력으로 중단 시도의 횟수를 복구하고,
종료 시간을 모르는 시도는 비용 미측정으로 남긴다. 완료 trial·VUS checkpoint를 재사용하며
ALoRa만 저장 epoch에서 학습을 이어간다. 다른 미완료 trial은 비용 이력을 남기고 다시 실행한다.

GHL·HAI 내부 코드와 기업 최적화는 이번 작업에서 제외했다. 로컬에서는 파일·호출부·diff만
확인했다. 회귀 검사는 작성했으나 테스트·Python 구문 검사·실데이터·모델·그림은 실행하지 않았다.

## 이전 수정 기록 — 현행 실행 예산 아님

2026-09-06 추가 요청으로 튜닝 완료 결과의 모델별 규모 근거와 기업 조회 제한을 수정했다.
q별 설정 선택과 기존 예산을 유지하며 n·d·N·평가 길이·실행 환경이 함께 확인된 지점만 연결한다.
현재 게이트는 이 수정의 정적 검토 완료·원격 검증 전이다. 테스트·구문 검사·모델 실행은 계속 보류한다.

2026-09-06 Tier 3 공식 튜닝 대조에서 TimeRCD의 고정 문맥 5,000을 유지하고 TSPulse는
aggregation 64·96·128마다 time·fft·pred·raw_max를 모두 선택 후보로 확대한다. 같은 추론의
네 출력을 쓰며 기존 family 동일 가중·LOFO와 strict 전처리를 유지한다. 이번 게이트는
Tier 3 수정·정적 대조 완료이며 원격 실행 검증 전이다. 예산의 deterministic seed도 실행
목록과 같은 0으로 맞추고 보고서 범례에서 네 head를 구분한다.

최신 실행 금지 요청 전에 후보·예산의 순수 검증 5개가 통과했다. 모델은 실행하지 않았다.
그 뒤 seed와 범례를 추가 수정했으므로 이 5개를 최종 변경의 검증 결과로 쓰지 않는다.
최신 요청 이후에는 공식 코드·호출부·diff만 검토했다. 새 회귀 검사는 작성만 했고 로컬
모델·테스트·Python 구문 검사·원본 CSV 스캔·그림 생성은 모두 보류한다.

2026-09-06 추가 요청에 따라 현재 튜닝의 실행·채점·재개에 영향을 주는 미완료 부분만 수정했다.
전체 계획량과 이번 미완료 수, 완료 확인 시간과 모델 계산·저장 시도 시간을 구분했다. ALoRa는
재개 checkpoint 원본을 CPU로 읽고 필요한 상태만 실행 장치에 복원한다. 관련 순수·모킹 검증
10개가 통과했다. 모델 학습·forward·전체 테스트·원본 CSV 스캔은 실행하지 않았다.

직전 요청에서 추가한 PCA 학습 상태와 Tier 2 scaler 저장, 실행 이력·이전 영수증 보존은 유지한다.
그때는 실행 검증 없이 수정했고 이번에는 이력·scaler 저장과 모킹 재개만 확인했다. 실제 모델의
저장/복원 점수 일치는 아직 원격 검증 전이다. 현재 게이트는 튜닝 준비 보완 완료·원격 검증 전이다.

2026-09-05의 `full_prefix_v2` 통합 때 통과한 순수·모킹 검증 102개와 Python 40개 구문 검사는
이번 추가 수정의 검증 결과가 아니다. 새 모델 학습·VUS 채점·그림도 실행하지 않았다.
기존 산출물은 보존하며 과거 추가 738건·ALoRa 450건 계획을 더 이상 실행 예산으로 쓰지 않는다.

각 q-prefix 전부를 학습하고 CSV별 가능 후보를 유지한다. 같은 조건의 파일에서 모델/q별 설정과
Tier 모델+설정을 따로 고른다. PaAno는 q60·q80에서 다른 설정을 선택할 수 있다. ALoRa는
window/q/CSV/seed별 한 trajectory의 3·4·10 epoch와 h1별 점수를 공유하며 누락 결과만 재개한다.

후보 복원 전 registry의 metadata 계산은 config/CSV/q/seed 결과 16,691건, 공유 학습 5,363건,
논리 VUS 원표는 Tier 3 head 확대 전 17,339행이다. head 확대 후 설계 예산은 18,473행이며
물리 추론은 늘지 않는다. 이 중 ALoRa 12,240개 조합은 1,020개 학습 trajectory로 얻는다.
이는 실행 전 구조 계산이며 GPU 성공·소요시간 측정 결과가 아니다. 실제 budget ID는 코드가
고정된 뒤 `--prepare`가 봉인한다.

## 다음 작업 하나

최신 시작점은 수집·저장 코드의 원격 회귀와 소규모 저장 검증이다. `test_prefix_features`,
`test_recommendation_evidence`, `test_recommendation_command`, `test_feature_capture_execution`,
`test_full_prefix_storage_schema`, `test_recommendation_handoff`를 먼저 확인한다. 과거 결과
보완 경로를 만들거나 수집 검증 없이 본 튜닝을 시작하지 않는다.

원격 검증이 허용되면 변경된 registry·MWVAR·GDN·실행 인자·재개 이력·checkpoint 부분 검사와
선택 근거의 회귀를 먼저 실행한다. 핵심 대상은 `test_model_registry`, `test_model_feasibility`,
`test_tier1_models`, `test_gdn_official`, `test_registered_executor`, `test_checkpoint_smoke`,
`test_run_history`, `test_ratio_tuning_full_prefix`, `test_tuning_support`다. 이번 중단·저장 보완의
`test_dev18_tuning`, `test_full_prefix_execution_contract`, `test_lightning_dev18_resource_tools`도 포함한다.
이 검증을 통과한 코드로 [단일 진입 실행](lightning_studio.md#전체-prefix-튜닝)을 시작한다.
폐기한 이전 예산·점수·snapshot·ledger·학습 checkpoint를 새 실행에 승계하지 않는다.
현재 로컬의 테스트·구문 검사·모델 실행 금지는 그대로 유지한다.

## 재개 기록

같은 봉인 코드·환경·budget으로 같은 명령을 다시 실행한다. 완료 manifest와 completion 영수증,
연결된 점수·snapshot·checkpoint·scaler의 SHA를 확인하고 완료 조합은 건너뛴다. ALoRa는 저장된
epoch에서 누락 결과를 이어간다. 그 밖의 미완료 학습은 저장 지점이 없으면 해당 조합부터 다시
시작한다. 모든 모델을 임의의 batch 중간부터 이어 학습하는 기능은 아니다.

`experiments/01_ghl_main/logs/run_history/`에 명령과 모델 시도별 고유 ID·시작/종료·상태·경과시간을
보존한다. ALoRa 공유 batch는 한 시도로 기록하고 완료 config를 갱신한다. 명령 시간과 그 안의
모델 시도 시간은 중복 합산하지 않는다. 강제 종료로 `running`만 남으면 종료시간과 비용은 미확정이다.
다시 실행해도 그 기록을 덮어쓰거나 0으로 채우지 않는다. 이전에 기록하지 않은 실패 비용도 소급해
추정하지 않는다. 완료 영수증의 과거 내용은 `commands/receipts/`에 보존한다.

자원 probe는 `experiments/checks/reference_code/active_models/full_prefix_v2/resource_probe_history/`에
시작·결과·시간을 개별 저장한다. 코드·입력·예산·패키지 환경·GPU·RAM이 같은 완료 결과만
재사용한다. 채점 중 정상 중단을 요청하면 이미 시작한 worker의 종료까지 기다린 뒤 이력과
잠금을 닫는다. 강제 종료나 전원 손실은 종료시간을 확정하는 경로가 아니며 미측정으로 남는다.

`planned_runs`는 전체 계획량, `pending_runs_before`는 이번 명령 시작 전 manifest의 미완료 수다.
실제 실행 직전에는 완료 영수증도 확인하므로 미완료 수가 더 줄 수 있다. `execution_seconds`는
준비·확인·시도를 포함한 panel 전체 시간이다. `completion_check_seconds`는 완료 확인 구간만,
`model_attempt_seconds`는 새 계산·저장 시도의 시간만 누적한다. 실패·재시도도 후자에 포함되며
ALoRa 공유 batch는 한 번만 센다. 세 시간을 서로 더하지 않는다. 실패한 공유 시도에도 완료 결과가
있을 수 있으므로 실패 시도 전체를 낭비 비용으로 해석하지 않는다. 이 기록은 HPO 개발비 근거다.

## 이번에 보류한 항목

서비스 상한 근거·실제 규모 대조·기업 조회 근거는 추가 요청으로 이번 수정에 포함한다.
거리와 허용 오차로 참조 집단을 다시 고르지 않고 기존 조건집단과 family-LOFO를 유지한다.
최종 채점기의 validation threshold 분리는 GHL·HAI 평가를 연결할 때 다룬다. 현재 튜닝은 그 채점기를
호출하지 않고 연속 점수로 VUS-PR을 계산한다. 이 보류 항목들을 튜닝 시작 조건으로 추가하지 않는다.
기존 원표 재집계나 상한·매칭 검증용 실험은 이번에 수행하거나 새로 요구하지 않았다.

## 남은 결과와 설계

새 선택 결과와 조건부 membership, GHL25·HAI 최종 인수 및 실행, 오탐·미탐과 운영 threshold
검증이 남아 있다. 기업 후보 조회 함수는 n·d·예상 N·평가 길이·환경과 완료 영수증의 규모 근거를
받는다. 등록 q와 실제 조합이 일치해야 후보를 반환한다. 상한 수치는 아직 검증해 확정하지 않았으며, 조회도 수익 최적이나
도메인 일반화를 보장하지 않는다. 비용 경로 최적화에는 현장 단가·수집속도·발생 빈도가 필요하다.
기업의 N을 100%로 유지한다. 서비스 상한은 가장 긴 CSV가 아니라 고정 실행 환경의 자원 한도와
모델·센서 조건별 성능 근거를 함께 확인한 범위로 정한다. 이번에는 이 원칙만 정했고 수치는 봉인하지 않았다.

현재 코드의 설명과 재개 기준은 [계획서 v5](plan_v5.md), 검증 기록은
[사전 점검 결과](process_0_preverify.md)를 따른다.
