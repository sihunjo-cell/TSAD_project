# Lightning AI에서 Dev18 튜닝 실행

이 절차는 [계획서 v5](plan_v5.md)의 2단계 Dev18 exact panel만 실행한다. 모델, config, seed,
예산과 채점 규칙은 바꾸지 않는다. 같은 Studio와 GPU 종류를 끝까지 유지하며, 중단되면 같은 명령을
다시 실행한다.

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

## GPU에서 실행

설치와 업로드를 마친 뒤 Studio 장치를 `1×L4`, `Interruptible off`로 바꾼다. 이전 T4 실행이
있다면 프로젝트 폴더에서 삭제 대상을 먼저 확인하고 Dev18 실행 결과와 T4 환경 봉인만 지운다.
입력 ZIP, 설치한 package, checkpoint cache와 봉인된 Dev18 예산은 남는다.

```bash
python tests/checks/reset_lightning_dev18.py
python tests/checks/reset_lightning_dev18.py --confirm DELETE_DEV18_RUN
```

본 튜닝보다 먼저 자원 gate를 실행한다. 18개 입력의 크기와 SHA-256, 저장 공간 25GB를 먼저
확인한다. 경량 모델은 입력 크기로 RAM 상한을 계산하고 `PaAno·GDN·TimeRCD·TSPulse`는 exact
panel의 config별 최대 배치만 별도 프로세스에서 실행한다. 학습형 모델은 한 update만 수행한다.
GPU나 RAM 사용률이 80%에 닿으면 실패로 판정한다.

```bash
python tests/checks/check_dev18_resources.py
```

마지막 JSON의 `status`가 `passed`일 때만 아래 명령으로 전체 panel을 시작한다. 점검 결과는
`.runtime/dev18_resource_gate.json`에 저장된다. 본실험 진입 파일은 이 파일이 없거나 현재
GPU·driver·VRAM·코드 commit·입력 manifest·예산과 다르면 실행을 거부한다.

```bash
python tests/checks/run_lightning_dev18.py
```

진입 파일은 Linux와 CUDA를 먼저 확인하고 현재 Python·package, GPU 종류, CUDA·driver를
`.runtime/runtime.json`에 봉인한다. 이어서 기존 Dev18 준비 검사를 통과한 뒤 1,170건 panel을
시작한다. 첫 봉인 뒤 package나 GPU 환경이 달라지면 재개하지 않는다.

Studio가 멈추거나 credit을 다 쓰면 환경을 고치지 말고 같은 Studio에서 같은 non-interruptible L4를
선택한 뒤 위 명령을 다시 실행한다. 완료된 score와 metadata의 SHA-256을
확인해 건너뛴다. 점수 저장 직후 진행 원표를 쓰기 전에 끊겨도 완료 영수증으로 복구하므로 끝난
trial을 다시 계산하지 않는다. 실행 중이던 trial에서 끊기면 그 trial 하나만 처음부터 다시 시작한다.
CUDA 실행에 `--remote-execution`을 붙이지 않는다. 이 panel은 실측
timing이 아직 없어서 보유 credit 안에 전부 끝난다고 단정하지 않는다. 남은 실행 수와 누적 시간을
확인한 뒤에만 추가 credit을 산다.

진행 원표는 `experiments/01_ghl_main/logs/dev18_score_manifest.csv`, 최종 선택 결과는
`experiments/01_ghl_main/results/dev18_tuning/`에 생긴다. 전체 score까지 포함하면 저장 공간이
20GB를 넘을 수 있으므로 Studio 여유 공간을 25GB 이상 유지한다. 현재 무료 계정의 persistent storage
한도 안에는 들어가지만 다른 Studio 파일과 합산한 여유 공간을 확인한다.

## 중단 조건

- `git status --short`가 비어 있지 않으면 실행하지 않는다.
- GPU 종류, Python, package 또는 source commit이 바뀌면 기존 panel과 섞지 않는다. GPU를 바꿀
  때는 위 초기화 명령으로 기존 Dev18 실행과 runtime 봉인을 함께 지운다.
- `.runtime/runtime.json` 불일치는 환경을 자동으로 덮어쓰지 않는다.
- 데이터 SHA-256이나 길이가 manifest와 다르면 원본을 다시 올리고 임의 수정본은 쓰지 않는다.
