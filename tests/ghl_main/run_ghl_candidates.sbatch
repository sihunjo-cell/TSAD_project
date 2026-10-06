#!/bin/bash
# GHL 파일 25개를 배열 작업으로 나눠 웹사이트 후보 전체를 실행·채점한다. 저장소 루트에서 한 번만 제출한다.
#   sbatch tests/ghl_main/run_ghl_candidates.sbatch
#   sbatch tests/ghl_main/run_ghl_candidates.sbatch --seeds 3    # PaAno·GDN을 첫 final seed만
# 모델은 GPU에서 하나씩 돌고, 할당된 CPU에서 하나를 뺀 수만큼 프로세스가 동시에 채점한다.
# 12시간 한도 전에 멈추면 같은 파일을 다시 제출해 이어서 돈다. 25개 파일의 실행·채점이 모두 끝나면
# 서비스 점검(evaluate_ghl_service.sbatch)을 한 번 제출한다. 다시 자동 제출하려면
# experiments/01_ghl_main/results/.evaluation_submitted 폴더를 지운다. 동시에 쓰는 GPU는 4장이다.
#SBATCH --job-name=ghl_runs
#SBATCH --partition=laal_3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-24%4
#SBATCH --output=experiments/01_ghl_main/logs/%x_%A_%a.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# laal1에는 계정 이름이 등록돼 있지 않다. 이름 조회 대신 환경변수를 읽도록 채워 둔다.
export USER="${USER:-$SLURM_JOB_USER}" LOGNAME="${LOGNAME:-$SLURM_JOB_USER}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/tsad/bin/python}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

timeout 690m "$PYTHON" -m tests.ghl_main.run_ghl_candidates --file-index "$SLURM_ARRAY_TASK_ID" "$@"
status=$?
if [ "$status" -eq 124 ]; then
    next=tests/ghl_main/run_ghl_candidates.sbatch
    sbatch --array="$SLURM_ARRAY_TASK_ID" "$next" "$@" || echo "재제출 실패: sbatch --array=$SLURM_ARRAY_TASK_ID $next $*"
elif [ "$status" -eq 0 ] && "$PYTHON" -m tests.ghl_main.run_ghl_candidates --check-complete "$@" \
        && mkdir experiments/01_ghl_main/results/.evaluation_submitted 2>/dev/null; then
    next=tests/ghl_main/evaluate_ghl_service.sbatch
    sbatch "$next" || echo "제출 실패: sbatch $next"
fi
exit "$status"
