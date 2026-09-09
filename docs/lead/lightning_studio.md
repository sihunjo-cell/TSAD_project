# Lightning AI Dev18 실행 안내

## 전체 prefix 튜닝

현재 경로는 `full_prefix_v2`, recipe 버전은 `paper_tuning_v4`다. 로컬 변경을 검토해 커밋·push하고 원격에서는 작업 트리가
깨끗한지 확인한 뒤 같은 커밋을 checkout한다. 폐기한 과거 결과를 복구하지 않는다. 저장소에 남아 있는
사용자 보고서·노트북을 무조건 `git add .`로 함께 넣지 말고 이번 코드·설정·문서만 선택한다.

대상 환경은 L4 24GB 한 장과 CPU 8개다. RAM은 화면의 GPU 메모리와 별개이며 자원 검사가
컨테이너의 실제 할당 한도를 확인한다. 채점 worker는 RAM 여유와 CPU 수를 반영해 최대 8개,
각 worker의 계산 thread는 1개로 제한한다. 기존 Studio 환경변수가 더 큰 값이어도 이 제한을 적용한다.

Python은 3.11.x·3.12.x를 허용한다. `bash tests/checks/setup_lightning_studio.sh`는 현재 터미널의
Python에 고정 패키지와 CUDA 12.6 torch를 설치하고 환경 차이를 한 번에 보고한다. Python을
3.11.14로 내릴 필요는 없다. 실제 patch 버전과 설치 결과는 첫 실행에 봉인하며 재개 중에는 유지한다.

디스크는 고정 용량·여유율로 실행을 제한하지 않는다. 현재 전체·사용·여유 용량은 자원 보고서의
`disk_observation`에 기록하고, 사전 검사 재사용 때도 명령 이력에 새 관측을 남긴다. 조회 실패는
NULL과 사유로 남기며 전체 튜닝의 저장 가능 판정으로 해석하지 않는다. 실제 저장 오류는 중단·보존한다.

현재 게이트는 TSPulse 두 단계 선택·PaAno 공식 memory와 최신 공식 소스·후보 풀 반영,
저장 실패 뒤 중복 재학습 방지·실패 자원 계측·학습 loss 기록·공식 원본과 자원 이력 인수,
인수 시간 기록·상관값 경계·GDN 보조 점수·실행별 RSS 관측·완료 행 복구,
TSPulse 정규화 수치·학습 파일 부분 저장 참조·PCA 추정 초과 시 실측의 보완과 정적 검토다.
마지막 감사에서 확인한 GDN 사전 검사·학습 증거 저장 실패 후 재기동·채점 근거 인수도 함께 검증한다.
디스크 조회의 비차단·재개 시 관측 갱신과 학습 증거·종료 이력의 연속 저장 실패도 기존 회귀에서 확인한다.
아래 회귀를 원격에서 확인하고 작은 저장·중단 재개 경로를 검증한다. 로컬에서는 실행하지 않는다.
이 문서의 본 튜닝
명령은 검증 인수와 별도 실행 요청을 받은 뒤 사용한다.

```bash
python -m unittest tests.unit.test_prefix_features tests.unit.test_recommendation_evidence tests.unit.test_recommendation_command tests.unit.test_feature_capture_execution tests.unit.test_full_prefix_storage_schema tests.unit.test_recommendation_handoff
python -m unittest tests.unit.test_ratio_tuning_full_prefix tests.unit.test_tspulse_official_selection tests.unit.test_tuning_support tests.unit.test_conditional_policy tests.unit.test_service_support tests.unit.test_run_history
python -m unittest tests.unit.test_model_registry tests.unit.test_registered_runner tests.unit.test_registered_executor tests.unit.test_model_feasibility tests.unit.test_paper_tuning_output_contract tests.unit.test_paper_tuning_pipeline tests.unit.test_ratio_tuning_budget tests.unit.test_full_prefix_policy_integration
python -m unittest tests.unit.test_paano tests.unit.test_gdn_official tests.unit.test_tspulse tests.unit.test_tier1_models tests.unit.test_model_smoke tests.unit.test_lightning_dev18_resource_tools tests.unit.test_dev18_tuning tests.unit.test_full_prefix_execution_contract
python -m unittest tests.unit.test_process_memory tests.unit.test_recommendation_storage_integration
python -m unittest tests.unit.test_scoring_environment
```

새 준비 명령은 입력 CSV·모델을 실행하지 않고 봉인한 metadata로 실행 예산과 추천 저장 계약을
만든다. SQLite schema와 예상 기록도 준비하지만 특징 계산은 모델 배치를 열기 직전에 수행한다.

```bash
python -m tests.ghl_main.run_ratio_tuning --prepare
```

예산은 `experiments/01_ghl_main/snapshots/dev18_selection/full_prefix_v2/budget.json`에 저장된다.
ALoRa를 제외한 현재 36개 설정·45개 head별 선택 항목을 CSV/q별로 대조한다.
기존 inventory와 구조 조건에서 계산한 실행량은 4,092회이며 재시도·사전 점검을 제외한
예상값이다. 실제 실행 수는 새 준비 결과에서 봉인한다. 과거 예산과 고정 membership 행수는 쓰지 않는다.

TSPulse는 q별 공통 창을 고른 뒤 데이터셋별 head를 고른다. 창 선택은 승인한 파일별 time·fft
평균식이며 LOFO에서는 holdout을 제외하고 두 단계 모두 다시 고른다. 세 창·네 head 점수는
모두 보존한다. 선택 절차 신원은 `full_prefix_tspulse_two_stage_family_lofo_v3`다.

PaAno의 native RevIN·공식 memory 규칙, GDN 내부 검증의 최저 loss checkpoint와
Tier 3 전체 평가 통계·native 후처리를 유지한다. GDN 세 조합은 모두 공식 train.py의
patience=15와 Adam betas=(0.9,0.999)를 쓴다. 새 예산·합성·checkpoint·자원 검증을 봉인하고
과거 raw·공통 MinMax·마지막 epoch 결과를 새 완료 실행에 합치지 않는다. 테스트는 작성만
했으며 원격에서 관련 회귀를 먼저 확인한다. 로컬에서 실행하지 않는다.

PaAno의 배치 GPU 추론·분할 memory 거리 계산, PCA의 분할 거리 계산은 작은 입력에서
원식과 점수를 대조한다. PaAno는 전체 fit embedding과 official_minimum 정책을 쓴다.
학습 patch 수를 P라 하면 memory 개수는 공식 코드의
max(min(500,max(1,P−1)),min(round(0.1×P),P−1))이며 P=1,000이면 500개다.
전체 fit의 KMeans는 그대로 수행하므로
최대 배치 probe만으로 전체 학습 RAM이 충분하다고 판단하지 않는다. L4의 실제 최대 메모리와
첫 완료 실행의 시간을 확인한 뒤 나머지 튜닝을 진행한다.

PCA는 기존 RAM 추정이 80% 미만이면 그대로 통과한다. 그렇지 않으면 자원 검사 안에서 최대 작업 배열
입력과 n_components=None의 등록 설정을 별도 CPU 프로세스로 실행한다. 원격 Linux·CUDA
환경에서만 열리며 입력 로딩을 포함한 프로세스 생애 RSS가 RAM 80% 미만이어야 통과한다.
추정값·실측값·설정·입력 신원은 resource_probe_history에 함께 보존하고 인수에도 연결한다.
대표 실행의 측정이며 나머지 후보나 실제 서비스의 자원 상한을 보장하지 않는다.

TSPulse metadata에는 실제 입력 StandardScaler·내부 head 범위·출력 최대값과 MinMaxScaler를
남긴다. 원격에서 공식 점수 동일성과 metadata→완료 검사→DB 연결을 확인한다. 학습 파일 저장의
중간 실패도 검사해 먼저 기록한 checkpoint·scaler 참조가 인수에 남는지 확인한다.

설치 환경, 승인된 inventory·VUS 검증·ell_max가 준비돼 있으면 아래 한 명령으로 시작한다.
DATA_ROOT는 기존 tuning 폴더의 부모다. 파일 직접 실행과 `python -m` 실행은 같은 진입점이다.

```bash
python tests/ghl_main/run_ratio_tuning.py --data-root "$DATA_ROOT" --workers 0
```

기본 실행은 준비·합성 검사·봉인 checkpoint 확보와 검사·자원 검사·환경 봉인·126개 prefix 특징
검증·모델 실행·VUS 채점·q별 선택·추천 자료 인수를 이어간다. 현재 신원과 맞는 완료 검사는 재사용한다. 실패 검사 기록은
보존 후 다시 실행하며, 통과 기록이 다른 코드·입력·예산을 가리키면 중단한다. 사전 검사는
대표 설정의 실행 검증이며 모든 HPO 후보의 성공·성능이나 서비스 상한을 보장하지 않는다.

GPU 실행과 원격 CPU 채점을 나눌 때만 같은 파일의 부분 실행 옵션을 사용한다.

```bash
python -m tests.ghl_main.run_ratio_tuning --execute-only --data-root "$DATA_ROOT"
python -m tests.ghl_main.run_ratio_tuning --finish-only --remote-cpu --workers 0 --data-root "$DATA_ROOT"
```

같은 명령을 다시 실행하면 현재 계약의 완료 조합과 VUS checkpoint를 건너뛴다.
완료 snapshot/score/checkpoint의 신원이
다르면 덮어쓰기 대신 중단한다. 실행 도중 코드·환경·registry·입력을 바꾸지 않는다.

PaAno·GDN은 완료 trial 단위로 재개하며 학습·추론 자체가 실패한 미완료 trial은 처음부터 다시 실행한다.
이전 시도는 삭제하지 않고 재시도 한도와 비용 이력에 포함한다. 잠금 파일은 실행 중 동시 명령을
차단하며 프로세스가 종료되면 운영체제가 잠금을 해제한다. 잠금 파일을 수동으로 삭제하지 않는다.

학습을 마치면 추론 전에 checkpoint·scaler·loss 이력·학습 시간을 시도별로 저장한다. 추론이
실패해도 이 증거는 남으며 다음 시도에서 덮어쓰지 않는다. 정상 완료는 저장된 학습 파일을
재사용한다. 실패 시도의 학습 기록도 인수하며 큰 checkpoint는 외부 참조로 남긴다.
학습 증거 저장 시간은 `save_training` 단계에 따로 기록하고 모델 학습·추론 timing에서 제외한다.
전체 실행 시간·자원 관측 범위에는 이 저장 구간이 포함된다. 저장한 학습 증거를 자동 학습 재개점으로
쓰지는 않는다. 실제 채점 병렬 수와 CPU 정보는 명령 이력의 `scoring_environment`에서 확인한다.
요청 worker 수와 실제 사용 수를 구분하며, CPU 정보가 없으면 NULL과 사유를 남긴다.

학습 증거를 저장하다 실패하거나 중단한 시도는 재기동해도 자동 재학습하지 않는다.
종료 이력 저장까지 실패해도 같은 실행의 재시도 분기가 학습 증거 저장 실패를 확인하고 중단한다.
기록된 시도와 부분 파일을 보존하고 저장 오류를 알린다. 학습 증거 저장을 마친 뒤 발생한
일반 추론 실패는 기존 trial 재시도 대상이다.

모델 실행 후 저장 오류가 나면 자동 재학습을 멈춘다. 당시 snapshot 내용과 반환된 timing은
UUID 이력에 남는다. 계산 완료 사실과 검증된 출력 행을 영수증보다 먼저 기록한다. 저장 원인을
해결한 뒤 같은 명령을 실행하면 파일·신원·SHA를 대조하고 영수증을 복구한다. 계산 완료는
기록됐고 모든 출력이 남았으면 완료 행이 없어도 metadata의 파일 지문과 시도 신원을 검증해
행을 복구한다. 저장 파일이 불완전하거나 지문이 다르면 재학습하지 않고 보존한 채 중단한다.
점수 파일 저장 뒤 SQLite 기록이 실패한 경우에도 배치를 멈춘다. 재개 시 같은 실험의 완료
파일을 확인해 빠진 DB 연결만 복구하며, 특징은 CSV/q별 저장값을 재사용한다.
계산 완료 전에 중단된 trial의 부분 checkpoint는 학습 재개점으로 쓰지 않는다.
자원 probe도 개별 이력을 저장하므로 전체 자원 검사 도중 멈춰도 완료 근거는 남는다. 코드·입력·
예산·환경·GPU·RAM이 같을 때만 재사용한다. VUS 채점의 정상 중단은 실행 중 worker가 끝날 때까지
기다린다. 강제 종료 후에는 이전 작업 프로세스도 끝났는지 확인한 뒤 재개한다.

`tuning_cost_history.json`은 같은 예산의 명령 이력과 모델 시도 이력을 따로 연결한다. 각 이력의
시간은 중첩되므로 두 합계를 더하지 않는다. 강제 종료로 끝을 모르는 구간이 있으면 총시간은
null이며 확인된 하한과 미측정 이력을 함께 남긴다. 공급자 청구 시간이나 기업 배포비가 아니다.
예산 확정 전 실패 이력도 전달하지만 현재 예산 비용에 더하지 않는다. 인수 검증·압축·파일
해시는 이 명령 시간에서 제외하며, 별도의 `handoff/run_history/`에 성공·실패·중단과 시간을 남긴다.
인수 단계와 명령 단계는 겹치지 않지만 모델 시도 시간은 명령 시간 안에 포함된다.
모델 실행 실패 때도 가능한 CPU RSS·Python peak·요청 장치 CUDA peak를 남긴다. 실제
backend를 확인하지 못했으면 미확정으로 표시하고 계측 실패에는 NULL과 사유를 기록한다.
정상 종료한 학습의 loss 이력과 PaAno 선택 iteration·GDN 선택 epoch는 기존 학습 로그에 보존한다.
현재 프로세스의 실행 구간 RSS는 50ms 간격과 시작·종료에 관측하며 최고 관측값·표본 수·
오류를 resource_usage에 남긴다. 표본 사이 순간 peak와 자식 프로세스는 포함하지 않는다.
GDN의 공식 최종 scalar와 별도로 native 채널 점수 두 파일·median·IQR·epsilon·교정 범위를
보존한다. 보조 점수도 파일 지문 검사와 인수 외부 참조 목록에 포함한다.

기존 80/20 학습형 결과는 새 전체 prefix 결과로 재사용하지 않는다. target-free도 현재 config
신원이 달라 기존 파일을 자동 승계하지 않는다. 폐기한 결과는 복구하지 않는다.

새 결과는 `experiments/01_ghl_main/results/dev18_tuning/full_prefix_v2/`에 저장된다.
모델별 CSV/PNG, VUS 원표, 조건별 model_ratio·tier_adaptive, LOFO, 후보·제외 감사와
CSV별 final membership을 만든다. 과거 고정 설정 결과를 새 통제 자료로 쓰지 않는다.
`--no-plots`를 붙이면 원표와 선택까지만 만들고 그림은 생략한다.

`recommendation_evidence/`에는 SQLite 원표와 내용 지문을 붙인 일관된 백업,
`exports/`에는 입력 특징·채널별 특징·전체 모델 설정과 CSV/head별 성능표를 저장한다.
성능표는 첨부 양식의 11열을 유지하고 모든 CSV는 UTF-8 BOM으로 내보낸다. 최종
`recommendation_receipt.json`이 이번 백업과 내보내기의 위치·SHA·행 수를 지정한다.
DB는 `recommendation_evidence/recommendation.sqlite3`이며 추천 DB schema는 3이다.
같은 절차에서 `recommendation_inputs` VIEW와 `recommendation_inputs.csv`를 만든다.
추천 입력에는 원시 mean·std·IQR 대신 무차원 통계·크기·품질 정보를 쓰며, 채널별 IQR/std를
저장값으로 추가 계산한다. 입력 열의 역할과 NULL 처리 계약도 영수증에 보존한다.
단위·표본 간격과 검증 fold의 주의점은 [추천 입력 계약](tuning_feature_capture_prompt.md#추천-입력의-단위중복결측-처리)을 따른다.

`--execute-only`의 상태는 `execution_complete`이고 채점·추천 인수는 미완료다. 채점·선택과
DB 인수를 마친 `selection_complete.json`은 `ready_for_handoff`를 기록한다. 명령·비용 이력을
닫은 뒤 `handoff/recommendation_handoff.tar.gz`와 `handoff_manifest.json`을 만들면 최종 출력이
`complete`로 바뀐다. 전달 실패는 같은 새 실험의 finish-only로 복구한다. 묶음에는 DB 백업과
작은 원표·metadata·snapshot·시도 이력을 넣고 큰 점수·checkpoint·공식 가중치는 실제 위치와
SHA·바이트 수를 인수 목록에 남긴다. 사용 중인 DB 파일만 따로 복사하지 않는다.
실제 채점에 쓴 `dev18_ell_max.json`과 `official_tsb_ad_comparison.json`도 원표의 채점 신원과
대조해 묶음에 넣는다. 파일별 ℓ_max와 threshold 개수를 전달본에서 확인할 수 있다.
Tier 3 공식 가중치·config 참조는 snapshot.spec와 checkpoint smoke의 신원을 대조한다.
통과한 resource_gate와 같은 예산의 성공·실패·중단·종료 미확인 probe 이력도 묶음에 넣는다.
현재 인수 이력은 압축이 끝나야 닫히므로 묶음 밖에 남는다. 최종 출력의 `handoff.history`가
지정한 JSON도 압축파일과 함께 전달한다. 이력의 `complete`와 archive·manifest 지문으로
완료한 인수본을 확인한다. 같은 예산의 이전 인수 시도 이력은 다음 묶음에 포함된다.

기존 실행 시간만으로 새 전체 시간을 확정하지 않는다. PaAno memory 개수와 전체 prefix,
GDN 내부 검증·조기 종료를 반영해 원격 첫 완료 실행의 학습·추론 시간을 길이·채널·window별로
모아 잔여 시간을 추정한다. 같은 물리 실행을 head·q별로 중복 합산하지 않는다.

아래 내용은 이전 equal-trial 실행의 역사 기록이며 새 프로토콜의 실행 명령이 아니다.

## 모델 실행 당시 준비 파일 — 역사 기록

현재 게이트에는 `shared_data/TSAD_project/tuning/`의 CSV 18개만 필요하다. GHL25와 HAI는
올리지 않는다. Lightning에서 저장소와 데이터가 아래처럼 놓이면 기본 경로를 그대로 쓸 수 있다.

```text
/teamspace/studios/this_studio/
├── TSAD_project/
└── shared_data/
    └── TSAD_project/
        └── tuning/
            └── CSV 18개
```

## 모델 실행 당시 Studio 준비 — 역사 기록

대학 이메일로 만든 Lightning Studio를 무료 CPU 상태로 연다. 무료 계정의 credit과 GPU 시간은
가입 시점과 계정 인증 상태에 따라 화면에 표시된 잔액을 기준으로 한다. 저장소를 clone한 뒤 프로젝트
폴더에서 설치 파일을 실행한다. 설치 중에는 GPU를 켜지 않는다.

```bash
git clone -b codex/lightning-dev18 https://github.com/sihunjo-cell/TSAD_project.git
cd TSAD_project
bash tests/checks/setup_lightning_studio.sh
```

프로젝트 폴더에서 `mkdir -p .runtime`을 먼저 실행한다. 로컬에서 만든
`lightning_dev18_input.zip`을 `TSAD_project/.runtime/`에 올렸다면 아래처럼 푼다.

```bash
mkdir -p .runtime
mkdir -p ../shared_data/TSAD_project
unzip .runtime/lightning_dev18_input.zip -d ../shared_data/TSAD_project
```

압축을 쓰지 않으면 CSV 18개가 든 `tuning` 폴더를 위 경로에 그대로 올린다.

## 모델 실행 당시 초기화와 시작 — 역사 기록

Studio 장치를 `1×L4`, `Interruptible off`로 바꾼 뒤 프로젝트 폴더에서 아래 순서를 한 번만 따른다.
각 명령이 성공해야 다음 명령을 실행한다. 하나라도 실패하면 그 자리에서 멈춘다. reset은 새
project commit으로 실행을 옮길 때 한 번만 사용한다. 이전 Dev18 실행 결과와 runtime 봉인만
지우고 입력 ZIP, 설치한 package, checkpoint cache와 봉인된 Dev18 예산은 남긴다. 아래
series 13 GDN OOM 복구는 완료 산출물을 보존해야 하므로 이 reset 규칙의 유일한 예외다.

```bash
test -z "$(git status --porcelain)"
git pull --ff-only
test -z "$(git status --porcelain)"
python tests/checks/reset_lightning_dev18.py --confirm DELETE_DEV18_RUN
bash tests/checks/setup_lightning_studio.sh
python tests/checks/run_checkpoint_smoke.py --dev18-data-root ../shared_data/TSAD_project
python tests/checks/check_dev18_resources.py --data-root ../shared_data/TSAD_project --maximum-memory-percent 80
python -m json.tool .runtime/dev18_resource_gate.json
python tests/checks/run_lightning_dev18.py --data-root ../shared_data/TSAD_project
```

당시 checkpoint smoke는 해당 commit의 TimeRCD·TSPulse Dev18 1,536×2 증거를 새로 썼다. 자원 gate는
18개 입력의 크기·SHA-256와 저장 공간 25GB를 확인했다. 현행 디스크 고정 기준은 제거했다. `PaAno·GDN·TimeRCD·TSPulse`의 exact-panel
최대 배치를 별도 프로세스에서 실행한다. TimeRCD는 attention query chunk `64`를 쓰며 TSPulse는
등록 batch `32`와 batch `1`의 time·FFT·prediction·`raw_max` score를 대조한다. GPU나 RAM 사용률이
80%에 닿거나 동등성이 깨지면 실패다. GDN은 연속 여덟 개의 최대 training batch를 실행해 배치
사이 autograd graph가 GPU에 남지 않는지도 함께 확인한다. 자원 보고서에는
`pytorch_alloc_conf: "expandable_segments:True"`도 기록한다.

JSON에는 `status: "passed"`, 현재 `project_commit`, `budget_id=b5367ad431093`과 TSPulse
equivalence가 있어야 한다. 그때만 마지막 명령을 실행한다. 진입 파일은 Linux·CUDA와 자원 보고서를
먼저 확인하고 `.runtime/runtime.json`을 봉인한 뒤 1,170건 panel을 재개한다. config 변경, adaptive
정책, 모델별 수동 실행은 허용하지 않는다.

## series 13 GDN OOM 복구 — 역사 기록

이 절차는 project commit `501cb23cf0c02b9ebc9d94ca396d09e7049093d7`에서 primary 1,136건을
완료하고 첫 복구 commit `6d5bcafda7a10fa6247f9cc32fe35d5061a286fa`에서 네 번째 CUDA OOM이
난 실행에만 쓴다. series 13 GDN `c1168c94d4dfc`, q10, seed 0의 실패 기록을 모두 보존하고 같은
trial에 마지막 복구 시도 한 번을 추가한다. 새 commit은 기존 산출물의 SHA-256, 실행 환경, spec과
execution policy를 모두 다시 검증한다. 다른 commit이나 실패에는 이 예외를 적용하지 않는다.

Studio를 같은 `1×L4`, `Interruptible off`로 연다. `reset_lightning_dev18.py`, setup과 checkpoint
smoke는 실행하지 않는다. 아래 순서에서 기존 primary 완료 수가 1,136보다 작거나 새 자원 gate가
실패하면 멈춘다.

```bash
cd /teamspace/studios/this_studio/TSAD_project
test -z "$(git status --porcelain)"
git pull --ff-only origin codex/lightning-dev18
test -z "$(git status --porcelain)"
/home/zeus/miniconda3/bin/python -c 'import csv; rows=list(csv.DictReader(open("experiments/01_ghl_main/logs/dev18_score_manifest.csv"))); count=sum(row["status"]=="complete" and row["primary_score"]=="true" for row in rows); assert count >= 1136, count; print("preserved primary:", count)'
/home/zeus/miniconda3/bin/python tests/checks/check_dev18_resources.py --data-root ../shared_data/TSAD_project --maximum-memory-percent 80
/home/zeus/miniconda3/bin/python -c 'import json, subprocess; report=json.load(open(".runtime/dev18_resource_gate.json")); head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(); assert report["status"]=="passed" and report["project_commit"]==head and report["pytorch_alloc_conf"]=="expandable_segments:True", report; print("RESOURCE GATE PASSED")'
/home/zeus/miniconda3/bin/python tests/checks/run_lightning_dev18.py --data-root ../shared_data/TSAD_project
```

복구 진입점은 검증을 통과한 기존 1,136건을 건너뛰고 남은 primary 34건만 실행한다. 실행 중 Studio가
끊기면 같은 commit 재개 절차를 따른다. OOM 복구 trial도 외부 중단이면 처음부터 다시 시작하지만,
다섯 번째 모델 실패가 기록되면 추가 시도 없이 멈춘다. 최종 성공값은 manifest 1,224행, primary
1,170행, logical score 원표 1,602행이다.

## 완료 ledger에서 선택표만 다시 생성

물리 score 1,170건과 VUS-PR ledger 1,602행은 이미 완료됐다. 앞의 checkpoint smoke, TSPulse
동등성, L4 80% 자원 gate와 GDN OOM 복구는 당시 실행을 설명하는 역사 증거다. 아래 선택표 재생성
때 다시 실행하지 않는다. score manifest, 원본 CSV, score 배열, VUS checkpoint와
`finish_complete.json`도 읽거나 지우지 않는다.

현재 재개 명령은 아래 블록 하나다. Lightning에 이미 있는 ledger를 검증한 뒤 model-fixed,
tier-fixed, ratio-adaptive 정책과 상세 CSV, 294행 membership, 깨끗한 `selection.png`만 다시
만든다. 기존 ledger가 불완전하거나 registry·budget 봉인과 다르면 결과 파일을 덮기 전에 멈춘다.

```bash
set -euo pipefail
cd /teamspace/studios/this_studio/TSAD_project

PY=/home/zeus/miniconda3/bin/python
BRANCH=codex/lightning-dev18
RESULT_ROOT=experiments/01_ghl_main/results/dev18_tuning
LEDGER="$RESULT_ROOT/dev18_trial_score_ledger.csv"

test "$(git branch --show-current)" = "$BRANCH"
test -z "$(git status --porcelain --untracked-files=all)"
git pull --ff-only origin "$BRANCH"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "$LEDGER"

"$PY" tests/checks/finish_lightning_dev18.py \
    --selection-only \
    --ledger "$LEDGER" \
    --result-directory "$RESULT_ROOT"

"$PY" - "$RESULT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

from src.common.execution_identity import file_sha256

root = Path(sys.argv[1])
receipt = json.loads(
    (root / "selection_complete.json").read_text(encoding="utf-8")
)
required = (
    "model_fixed_policy.csv", "tier_fixed_policy.csv",
    "ratio_adaptive_selection.csv", "tier_ratio_candidate_audit.csv",
    "tier_policy_transitions.csv", "final_policy_membership.csv",
    "selection.png",
)
assert receipt["status"] == "complete", receipt
assert receipt["budget_id"] == "b5367ad431093", receipt["budget_id"]
assert receipt["selection_rule_id"] == "tier_adaptive_family_lofo_v1"
assert receipt["ledger_rows"] == 1602, receipt["ledger_rows"]
assert receipt["membership_rows"] == 294, receipt["membership_rows"]
assert receipt["expected_result_files"] == list(required)
assert receipt["ledger_sha256"] == file_sha256(
    root / "dev18_trial_score_ledger.csv"
)
assert receipt["final_policy_membership_sha256"] == file_sha256(
    root / "final_policy_membership.csv"
)
for name in required:
    path = root / name
    assert path.is_file(), path
    assert receipt["result_files_sha256"][name] == file_sha256(path), path

def rows(name):
    with (root / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))

assert len(rows("model_fixed_policy.csv")) == 8
assert len(rows("tier_fixed_policy.csv")) == 3
assert len(rows("ratio_adaptive_selection.csv")) == 21
assert len(rows("tier_ratio_candidate_audit.csv")) == 56
assert len(rows("tier_policy_transitions.csv")) == 21
assert len(rows("final_policy_membership.csv")) == 294
print(
    "선택표 재생성 완료: ledger 1602행 재사용, adaptive 21행, "
    "membership 294행, 필수 SHA-256 일치"
)
PY
```

`selection_complete.json`은 입력 ledger SHA-256, 선택 규칙, 비율별 대표 경로, 각 CSV 행 수,
membership SHA-256과 위 필수 파일 SHA-256을 기록한다. adaptive membership은
`model_fixed`의 runnable 물리 key만 재사용하므로 모델 실행을 늘리지 않는다.

## 모델 실행 당시 같은 commit 재개 — 역사 기록

Studio가 멈추거나 credit을 다 쓰면 환경을 고치지 않는다. 같은 Studio에서 이전과 같은 `1×L4`,
`Interruptible off`를 다시 선택한다. 이 재개 절차에서는 `reset_lightning_dev18.py`를 실행하지
않는다. setup, checkpoint smoke와 자원 측정도 되풀이하지 않는다. 아래 명령은 작업 트리가 깨끗하고
현재 HEAD가 기존 자원 보고서의 project commit과 같은지 확인한 뒤, 보고서와 runtime 봉인을
검증하는 단일 runner를 다시 시작한다. 앞 명령이 실패하면 즉시 멈추고 다음 명령으로 넘어가지 않는다.

```bash
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(python -c 'import json; print(json.load(open(".runtime/dev18_resource_gate.json"))["project_commit"])')"
python -m json.tool .runtime/dev18_resource_gate.json
python tests/checks/run_lightning_dev18.py --data-root ../shared_data/TSAD_project
```

runner는 현재 commit·GPU·입력·예산과 자원 보고서를 다시 대조하고 기존 runtime seal도 확인한다.
완료된 score와 metadata의 SHA-256이 맞으면 건너뛴다. 점수 저장 직후 진행 원표를 쓰기 전에 끊겨도
완료 영수증으로 복구하므로 끝난 trial을 다시 계산하지 않는다. 실행 중이던 trial에서 끊기면 그
trial 하나만 처음부터 다시 시작한다. CUDA 실행에 `--remote-execution`을 붙이지 않는다. 이 panel은
실측 timing이 아직 없어서 보유 credit 안에 전부 끝난다고 단정하지 않는다. 남은 실행 수와 누적
시간을 확인한 뒤에만 추가 credit을 산다.

진행 원표는 `experiments/01_ghl_main/logs/dev18_score_manifest.csv`, 최종 선택 결과는
`experiments/01_ghl_main/results/dev18_tuning/`에 생긴다. 전체 score까지 포함하면 저장 공간이
20GB를 넘는다는 당시 안내와 25GB 여유 기준은 현행 실행 기준으로 쓰지 않는다.
현재 용량을 관측하며 실제 저장 오류가 나면 기존 결과를 보존하고 중단한다.

## 모델 실행 당시 중단 조건 — 역사 기록

- `git status --short`가 비어 있지 않으면 실행하지 않는다.
- GPU 종류, Python, package 또는 source commit이 바뀌면 기존 panel과 섞지 않는다. GPU를 바꿀
  때는 새 commit 시작 절차의 초기화 명령으로 기존 Dev18 실행과 runtime 봉인을 함께 지운다.
- `.runtime/runtime.json` 불일치는 환경을 자동으로 덮어쓰지 않는다.
- 데이터 SHA-256이나 길이가 manifest와 다르면 원본을 다시 올리고 임의 수정본은 쓰지 않는다.
