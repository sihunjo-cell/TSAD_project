#!/bin/bash
# 끝난 GHL 파일을 모두 점검하고 요약을 다시 만든다. GPU는 쓰지 않는다. 보통은 run_ghl_candidates.sbatch가
# 마지막에 직접 요약하므로, 실행을 중간에 멈췄을 때만 저장소 루트에서 쓴다.
#   sbatch tests/ghl_main/evaluate_ghl_service.sbatch --seeds 3
#SBATCH --job-name=ghl_evaluate
#SBATCH --partition=laal_3090
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=experiments/01_ghl_main/logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export USER="${USER:-$SLURM_JOB_USER}" LOGNAME="${LOGNAME:-$SLURM_JOB_USER}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/tsad/bin/python}"
"$PYTHON" -m tests.ghl_main.evaluate_ghl_service --summarize "$@"
