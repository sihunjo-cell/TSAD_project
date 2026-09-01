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
80%에 닿거나 동등성이 깨지면 실패다. GDN은 연속 여덟 개의 최대 training batch를 실행해 배치
사이 autograd graph가 GPU에 남지 않는지도 함께 확인한다. 자원 보고서에는
`pytorch_alloc_conf: "expandable_segments:True"`도 기록한다.

JSON에는 `status: "passed"`, 현재 `project_commit`, `budget_id=b5367ad431093`과 TSPulse
equivalence가 있어야 한다. 그때만 마지막 명령을 실행한다. 진입 파일은 Linux·CUDA와 자원 보고서를
먼저 확인하고 `.runtime/runtime.json`을 봉인한 뒤 1,170건 panel을 재개한다. config 변경, adaptive
정책, 모델별 수동 실행은 허용하지 않는다.

## series 13 GDN OOM 복구

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

## 모델 완료 후 고코어 CPU에서 채점

manifest 1,224행과 primary 1,170행이 모두 생긴 뒤에만 이 절차를 쓴다. Studio 장치를 CPU로
바꿔도 된다. 이 경로는 모델을 실행하지 않으며, 기존 score·metadata와 원본 CSV의 SHA-256을
다시 확인한 뒤 물리 점수 1,170개를 VUS-PR로 채점한다. 원래 evaluator는 바꾸지 않는다. 같은
물리 점수를 논리 비율별로 다시 계산하지 않고 한 번만 계산해 원표 1,602행으로 펼친다.

먼저 현재 GPU commit에서 완료 수를 확인한다. 이 검사가 실패하면 `git pull`도 하지 않고 L4 복구
절차로 돌아간다. 1,170건 완료를 확인한 뒤 private 저장소 인증을 점검하고 CPU 채점 commit을
받는다. 기존 인증이 유효하면 `gh auth login` 단계는 자동으로 건너뛴다. 인증이 없으면 브라우저에
표시되는 GitHub device 절차를 마친다. token을 clone URL이나 명령 기록에 넣지 않는다.

```bash
set -euo pipefail
cd /teamspace/studios/this_studio/TSAD_project

PY=/home/zeus/miniconda3/bin/python
DATA_ROOT=../shared_data/TSAD_project
BRANCH=codex/lightning-dev18

test "$(git branch --show-current)" = "$BRANCH"
test -z "$(git status --porcelain --untracked-files=all)"

"$PY" - <<'PY'
from src.common.execution_identity import load_input_manifest_role
from tests.ghl_main.run_dev18_tuning import (
    DEFAULT_BUDGET_PATH,
    _load_score_manifest,
    _read_json,
    _validate_primary_manifest_rows,
)

manifest, _ = load_input_manifest_role(
    "configs/input_manifest.yaml", "development", "dev18_selection",
)
entries = manifest["datasets"]["DEV18"]["files"]
rows = _load_score_manifest()
primary = _validate_primary_manifest_rows(
    rows, _read_json(DEFAULT_BUDGET_PATH),
    tuple(sorted(entry["series"] for entry in entries)),
)
assert len(rows) == 1224, len(rows)
assert len(primary) == 1170, len(primary)
print("pull 전 완료 확인: manifest 1224행, primary 1170행")
PY

if ! GIT_TERMINAL_PROMPT=0 git ls-remote origin HEAD >/dev/null 2>&1; then
    command -v gh >/dev/null || {
        echo "private GitHub 인증에 gh CLI가 필요합니다. 여기서 중단합니다."
        exit 1
    }
    gh auth login --hostname github.com --git-protocol https --web
    gh auth setup-git
fi
GIT_TERMINAL_PROMPT=0 git ls-remote origin HEAD >/dev/null
git pull --ff-only origin "$BRANCH"
test -z "$(git status --porcelain --untracked-files=all)"
test -f tests/checks/finish_lightning_dev18.py
"$PY" tests/checks/finish_lightning_dev18.py --help >/dev/null

"$PY" - "$DATA_ROOT" <<'PY'
import sys
from pathlib import Path

from src.common.execution_identity import load_input_manifest_role
from tests.ghl_main.run_dev18_tuning import (
    DEFAULT_BUDGET_PATH,
    _load_score_manifest,
    _read_json,
    _validate_primary_manifest_rows,
)
from tests.ghl_main.run_registered_models import _verify_manifest_file

data_root = Path(sys.argv[1]).resolve()
input_manifest, _ = load_input_manifest_role(
    "configs/input_manifest.yaml", "development", "dev18_selection",
)
entries = input_manifest["datasets"]["DEV18"]["files"]
assert len(entries) == 18, len(entries)
for entry in entries:
    _verify_manifest_file(
        data_root / entry["source_directory"] / entry["name"], entry,
    )

rows = _load_score_manifest()
budget = _read_json(DEFAULT_BUDGET_PATH)
primary = _validate_primary_manifest_rows(
    rows, budget, tuple(sorted(entry["series"] for entry in entries)),
)
assert len(rows) == 1224, len(rows)
assert len(primary) == 1170, len(primary)
print("사전검사 통과: 원본 18개, manifest 1224행, primary 1170행")
PY
```

마지막 문구가 정확히 나온 뒤 아래 블록으로 백그라운드 채점을 시작한다. `--workers 0`은 할당된
CPU와 가용 메모리를 확인해 최대 16개 worker를 고른다. 각 worker의 BLAS thread는 하나로 고정해
CPU 과다 할당을 막는다. lock은 같은 채점이 동시에 두 번 시작되는 것을 막는다.

```bash
cd /teamspace/studios/this_studio/TSAD_project

PY=/home/zeus/miniconda3/bin/python
DATA_ROOT=../shared_data/TSAD_project
CHECKPOINT_ROOT=.runtime/dev18_vus_pr
PID_FILE="$CHECKPOINT_ROOT/finish.pid"
LOG_FILE="$CHECKPOINT_ROOT/finish.log"

mkdir -p "$CHECKPOINT_ROOT"
CURRENT_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ "$CURRENT_PID" =~ ^[0-9]+$ ]] \
   && kill -0 "$CURRENT_PID" 2>/dev/null \
   && ps -p "$CURRENT_PID" -o args= | grep -Fq 'finish_lightning_dev18.py'; then
    echo "채점이 이미 실행 중입니다. PID=$CURRENT_PID"
else
    nohup env PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "$PY" tests/checks/finish_lightning_dev18.py \
        --data-root "$DATA_ROOT" \
        --workers 0 \
        --checkpoint-directory "$CHECKPOINT_ROOT" \
        >> "$LOG_FILE" 2>&1 &
    echo "채점 진입점을 요청했습니다. launch PID=$!"
    echo "실제 scorer PID는 lock 획득 뒤 $PID_FILE 에 기록됩니다."
fi

tail -n 80 "$LOG_FILE"
```

진행 상황은 아래처럼 확인한다. `tail -f`에서 `Ctrl+C`를 눌러도 로그 보기만 끝나며 채점은 계속된다.

```bash
cd /teamspace/studios/this_studio/TSAD_project
CHECKPOINT_ROOT=.runtime/dev18_vus_pr
PID_FILE="$CHECKPOINT_ROOT/finish.pid"
LOG_FILE="$CHECKPOINT_ROOT/finish.log"
CURRENT_PID="$(cat "$PID_FILE" 2>/dev/null || true)"

if [[ "$CURRENT_PID" =~ ^[0-9]+$ ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    echo "채점 실행 중: PID=$CURRENT_PID"
else
    echo "실행 중인 PID가 없습니다. 완료 또는 중단 여부를 로그에서 확인하세요."
fi
find "$CHECKPOINT_ROOT" -type f -name '*.json' | wc -l
tail -f "$LOG_FILE"
```

Studio가 중지돼 프로세스가 사라졌다면 시작 블록을 그대로 다시 실행한다. 재개할 때도 원본
score·metadata SHA-256을 검증하며, 신원이 정확히 맞는 JSON checkpoint만 재사용한다. 깨진
checkpoint는 조용히 넘기지 않고 중단한다. 최종 CSV는 1,170개 채점이 모두 끝난 뒤 원자적으로
교체하므로 중간 파일을 결과로 오인하지 않는다. checkpoint에는 Python/package 봉인과 채점
commit도 묶인다. 채점 중 commit이나 작업 트리가 바뀌면 최종 정책표 완료로 인정하지 않는다.

모든 정책표와 그림을 쓴 뒤에만 `finish_complete.json`이 원자적으로 생긴다. 완료 후 이 영수증의
commit·manifest·결과 파일 SHA와 필수 행 수를 확인하고 한 파일로 묶는다.

```bash
set -euo pipefail
cd /teamspace/studios/this_studio/TSAD_project

PY=/home/zeus/miniconda3/bin/python
RESULT_ROOT=experiments/01_ghl_main/results/dev18_tuning
RECEIPT=.runtime/dev18_vus_pr/finish_complete.json

tail -n 100 .runtime/dev18_vus_pr/finish.log
"$PY" - "$RESULT_ROOT" "$RECEIPT" <<'PY'
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
required = (
    "dev18_trial_score_ledger.csv", "model_fixed_policy.csv",
    "tier_fixed_policy.csv", "family_lofo.csv",
    "final_policy_membership.csv", "selection.csv", "models.csv",
    "selection.png", "models.png",
)
missing = [name for name in required if not (root / name).is_file()]
assert not missing, missing

from src.common.execution_identity import file_sha256

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert receipt["status"] == "complete", receipt
assert receipt["project_commit"] == head, (receipt["project_commit"], head)
assert receipt["budget_id"] == "b5367ad431093", receipt["budget_id"]
assert receipt["ledger_rows"] == 1602, receipt["ledger_rows"]
assert receipt["score_manifest_sha256"] == file_sha256(
    "experiments/01_ghl_main/logs/dev18_score_manifest.csv"
)
for relative_path, expected_sha256 in receipt["result_files_sha256"].items():
    path = root / relative_path
    assert path.is_file(), path
    assert file_sha256(path) == expected_sha256, path
assert set(required) <= set(receipt["result_files_sha256"])
assert receipt["final_policy_membership_sha256"] == file_sha256(
    root / "final_policy_membership.csv"
)

def rows(name):
    with (root / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))

ledger = rows("dev18_trial_score_ledger.csv")
assert len(ledger) == 1602, len(ledger)
assert all(row["status"] == "complete" for row in ledger)
assert all(
    math.isfinite(float(row["vus_pr"])) and 0 <= float(row["vus_pr"]) <= 1
    for row in ledger
)
assert len(rows("model_fixed_policy.csv")) == 7
assert len(rows("tier_fixed_policy.csv")) == 3
assert len(rows("final_policy_membership.csv")) == 210
print(
    "최종 검증 통과: 완료 영수증과 모든 결과 SHA 일치, "
    "ledger 1602행, 모델 정책 7행, Tier 정책 3행, membership 210행"
)
PY

tar -czf .runtime/dev18_tuning_results.tar.gz "$RESULT_ROOT" "$RECEIPT"
sha256sum .runtime/dev18_tuning_results.tar.gz
ls -lh .runtime/dev18_tuning_results.tar.gz "$RESULT_ROOT"
```

이 절차에서는 `run_lightning_dev18.py`, `reset_lightning_dev18.py`, 일반
`run_dev18_tuning.py`를 실행하지 않는다. 셋 모두 완료된 score를 채점만 하는 진입점이 아니다.

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
