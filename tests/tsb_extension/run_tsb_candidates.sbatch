#!/bin/bash
# GHL·Dev18 밖 TSB-AD-M 파일을 GPU 한 장으로 차례대로 실행·채점·점검한다. 저장소 루트에서 한 번만 제출한다.
# 판정 기준은 tests/tsb_extension/README.md에 있고, 제출 전에 commit한다. --seeds는 제출할 때 정하고 결과를 본 뒤 바꾸지 않는다.
#   sbatch tests/tsb_extension/run_tsb_candidates.sbatch --seeds 3 4 5          # Genesis·SWaT id_1·GECCO는 PaAno·GDN seed 3개
#   sbatch tests/tsb_extension/run_tsb_candidates.sbatch --seeds 3              # GPU 시간이 없으면 모든 파일 seed 3 하나
#   sbatch --mem=120G tests/tsb_extension/run_tsb_candidates.sbatch --seeds 3 4 5   # 메모리를 바꾸면 재제출에도 이어진다
# 파일 순서는 configs/tsb_extension_manifest.yaml을 따른다. 파일마다 모델은 GPU에서 하나씩 돌고, CPU 8개 중 7개가
# 동시에 채점한다. 파일이 끝나면 그 파일의 웹사이트 점검을 하고, 모두 끝나면 같은 commit으로 GHL 점검을 갱신한 뒤
# 준 seed의 요약(주 판정)과 seed 3 요약(GHL과 견줄 때)을 만든다. 파일이 5개라 GHL의 멈춤 규칙은 없다.
# 11시간 30분이 되면 멈추고 같은 메모리로 자기 자신을 다시 제출한다. singleton이라 같은 이름의
# 작업은 앞 작업이 끝난 뒤에만 시작해 GPU를 한 장만 쓴다. 끝난 실행·채점·점검은 건너뛴다.
# 마지막 파일(SWaT id_2)의 PCA_LEGACY는 약 60GB를 쓸 것으로 추정한다. 메모리가 모자라 프로세스가 죽으면 실패 기록 없이
# 작업이 멈추고(종료 코드 137) 재제출하지 않는다.
#SBATCH --job-name=tsb_runs
#SBATCH --partition=laal_3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --dependency=singleton
#SBATCH --output=experiments/03_tsb_extension/logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# laal1에는 계정 이름이 등록돼 있지 않다. 이름 조회 대신 환경변수를 읽도록 채워 둔다.
export USER="${USER:-$SLURM_JOB_USER}" LOGNAME="${LOGNAME:-$SLURM_JOB_USER}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/tsad/bin/python}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
count=$("$PYTHON" -c "from tests.tsb_extension.run_tsb_candidates import tsb_files; print(len(tsb_files()))") || exit 1

deadline=$((SECONDS + 690 * 60))
status=0
for index in $(seq 0 $((count - 1))); do
    left=$((deadline - SECONDS))
    if [ "$left" -le 0 ]; then status=124; break; fi
    timeout "$left" "$PYTHON" -m tests.tsb_extension.run_tsb_candidates --file-index "$index" "$@"
    status=$?
    if [ "$status" -ne 0 ]; then break; fi
    "$PYTHON" -m tests.tsb_extension.evaluate_tsb_service --file-index "$index" "$@"
    status=$?
    if [ "$status" -ne 0 ]; then break; fi
done

if [ "$status" -eq 124 ]; then
    next=tests/tsb_extension/run_tsb_candidates.sbatch
    sbatch --mem="${SLURM_MEM_PER_NODE}M" "$next" "$@" || echo "재제출 실패: sbatch --mem=${SLURM_MEM_PER_NODE}M $next $*"
elif [ "$status" -eq 0 ]; then
    # TSB 요약은 같은 commit의 GHL 점검을 family 하나로 함께 읽는다. 이미 같은 commit이면 GHL 점검은 건너뛴다.
    "$PYTHON" -m tests.ghl_main.evaluate_ghl_service --summarize --seeds 3 \
        && "$PYTHON" -m tests.tsb_extension.evaluate_tsb_service --summarize "$@" \
        && "$PYTHON" -m tests.tsb_extension.evaluate_tsb_service --summarize --seeds 3
    status=$?
fi
exit "$status"
