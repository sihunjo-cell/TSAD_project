# Lightning AI에서 Dev18 튜닝 실행

이 절차는 [계획서 v5](plan_v5.md)의 2단계 Dev18 exact panel만 실행한다. 모델, config, seed,
예산과 채점 규칙은 바꾸지 않는다. 새 project commit에서 TimeRCD checkpoint smoke, TSPulse batch
1 대 batch 32 동등성, L4 80% 자원 보고서를 모두 통과하기 전에는 panel을 시작하지 않는다.

## 준비할 파일

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

## 무료 CPU Studio에서 한 번만 준비

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

## 새 commit에서 한 번만 초기화하고 시작

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

checkpoint smoke는 현재 commit의 TimeRCD·TSPulse Dev18 1,536×2 증거를 새로 쓴다. 자원 gate는
18개 입력의 크기·SHA-256와 저장 공간 25GB를 확인하고, `PaAno·GDN·TimeRCD·TSPulse`의 exact-panel
최대 배치를 별도 프로세스에서 실행한다. TimeRCD는 attention query chunk `64`를 쓰며 TSPulse는
등록 batch `32`와 batch `1`의 time·FFT·prediction·`raw_max` score를 대조한다. GPU나 RAM 사용률이
80%에 닿거나 동등성이 깨지면 실패다. GDN은 연속 두 개의 최대 training batch를 실행해 배치 사이
autograd graph가 GPU에 남지 않는지도 함께 확인한다.

JSON에는 `status: "passed"`, 현재 `project_commit`, `budget_id=b5367ad431093`과 TSPulse
equivalence가 있어야 한다. 그때만 마지막 명령을 실행한다. 진입 파일은 Linux·CUDA와 자원 보고서를
먼저 확인하고 `.runtime/runtime.json`을 봉인한 뒤 1,170건 panel을 재개한다. config 변경, adaptive
정책, 모델별 수동 실행은 허용하지 않는다.

## series 13 GDN OOM 복구

이 절차는 project commit `501cb23cf0c02b9ebc9d94ca396d09e7049093d7`에서 primary 1,136건을
완료한 실행에만 쓴다. series 13 GDN `c1168c94d4dfc`, q10, seed 0의 세 번째 CUDA OOM 기록을
보존하고 같은 trial에 복구 시도 한 번을 추가한다. 새 commit은 기존 산출물의 SHA-256, 실행 환경,
spec과 execution policy를 모두 다시 검증한다. 다른 commit이나 실패에는 이 예외를 적용하지 않는다.

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
/home/zeus/miniconda3/bin/python -c 'import json, subprocess; report=json.load(open(".runtime/dev18_resource_gate.json")); head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(); assert report["status"]=="passed" and report["project_commit"]==head, report; print("RESOURCE GATE PASSED")'
/home/zeus/miniconda3/bin/python tests/checks/run_lightning_dev18.py --data-root ../shared_data/TSAD_project
```

복구 진입점은 검증을 통과한 기존 1,136건을 건너뛰고 남은 primary 34건만 실행한다. 실행 중 Studio가
끊기면 같은 commit 재개 절차를 따른다. OOM 복구 trial도 외부 중단이면 처음부터 다시 시작하지만,
네 번째 모델 실패가 기록되면 추가 시도 없이 멈춘다. 최종 성공값은 manifest 1,224행, primary
1,170행, logical score 원표 1,602행이다.

## 같은 commit에서 중단 후 재개

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
20GB를 넘을 수 있으므로 Studio 여유 공간을 25GB 이상 유지한다. 현재 무료 계정의 persistent storage
한도 안에는 들어가지만 다른 Studio 파일과 합산한 여유 공간을 확인한다.

## 중단 조건

- `git status --short`가 비어 있지 않으면 실행하지 않는다.
- GPU 종류, Python, package 또는 source commit이 바뀌면 기존 panel과 섞지 않는다. GPU를 바꿀
  때는 새 commit 시작 절차의 초기화 명령으로 기존 Dev18 실행과 runtime 봉인을 함께 지운다.
- `.runtime/runtime.json` 불일치는 환경을 자동으로 덮어쓰지 않는다.
- 데이터 SHA-256이나 길이가 manifest와 다르면 원본을 다시 올리고 임의 수정본은 쓰지 않는다.
