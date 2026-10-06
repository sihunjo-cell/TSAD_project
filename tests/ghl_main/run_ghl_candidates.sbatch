#!/bin/bash
# GHL 파일을 GPU 한 장으로 차례대로 실행·채점·점검한다. 저장소 루트에서 한 번만 제출한다.
#   sbatch tests/ghl_main/run_ghl_candidates.sbatch --seeds 3    # PaAno·GDN을 첫 final seed만
# 파일마다 모델은 GPU에서 하나씩 돌고, CPU 8개 중 7개가 동시에 채점한다. 파일이 끝나면 그 파일의 웹사이트
# 점검을 하고, 정해 둔 멈춤 규칙(evaluate_ghl_service.py)을 만족하거나 25개를 마치면 요약을 만든다.
# 11시간 30분이 되면 멈추고 자기 자신을 다시 제출한다. singleton이라 같은 이름의 작업은 앞 작업이 끝난
# 뒤에만 시작해 GPU를 한 장만 쓴다. 끝난 실행·채점·점검은 건너뛴다.
#SBATCH --job-name=ghl_runs
#SBATCH --partition=laal_3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --dependency=singleton
#SBATCH --output=experiments/01_ghl_main/logs/%x_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
# laal1에는 계정 이름이 등록돼 있지 않다. 이름 조회 대신 환경변수를 읽도록 채워 둔다.
export USER="${USER:-$SLURM_JOB_USER}" LOGNAME="${LOGNAME:-$SLURM_JOB_USER}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/tsad/bin/python}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

deadline=$((SECONDS + 690 * 60))
status=0
for index in $(seq 0 24); do
    left=$((deadline - SECONDS))
    if [ "$left" -le 0 ]; then status=124; break; fi
    timeout "$left" "$PYTHON" -m tests.ghl_main.run_ghl_candidates --file-index "$index" "$@"
    status=$?
    if [ "$status" -ne 0 ]; then break; fi
    "$PYTHON" -m tests.ghl_main.evaluate_ghl_service --file-index "$index" "$@"
    status=$?
    if [ "$status" -ne 0 ]; then break; fi
    "$PYTHON" -m tests.ghl_main.evaluate_ghl_service --stop-check
    check=$?
    if [ "$check" -eq 0 ]; then break; fi
    if [ "$check" -ne 10 ]; then status=$check; break; fi
done

if [ "$status" -eq 124 ]; then
    next=tests/ghl_main/run_ghl_candidates.sbatch
    sbatch "$next" "$@" || echo "재제출 실패: sbatch $next $*"
elif [ "$status" -eq 0 ]; then
    "$PYTHON" -m tests.ghl_main.evaluate_ghl_service --summarize "$@"
    status=$?
fi
exit "$status"
