#!/bin/bash
# 25개 파일의 실행·채점이 끝난 뒤 웹사이트 추천을 GHL에서 점검한다. GPU는 쓰지 않는다.
# run_ghl_candidates.sbatch가 마지막에 자동으로 제출한다. 직접 돌릴 때는 저장소 루트에서
#   sbatch tests/ghl_main/evaluate_ghl_service.sbatch
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
"$PYTHON" -m tests.ghl_main.evaluate_ghl_service "$@"
