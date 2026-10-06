#!/bin/bash
# GHL 파일 25개를 GPU 한 장으로 차례대로 실행·채점한다. 저장소 루트에서 한 번만 제출한다.
#   sbatch tests/ghl_main/run_ghl_candidates.sbatch
#   sbatch tests/ghl_main/run_ghl_candidates.sbatch --seeds 3    # PaAno·GDN을 첫 final seed만
# 모델은 GPU에서 하나씩 돌고, 할당된 CPU에서 하나를 뺀 수만큼 프로세스가 동시에 채점한다. 끝난 실행·채점은
# 건너뛴다. 11시간 30분이 되면 멈추고 자기 자신을 다시 제출한다. singleton이라 같은 이름의 작업은 앞 작업이
# 끝난 뒤에만 시작해 GPU를 한 장만 쓴다. 25개 파일이 모두 끝나면 서비스 점검(evaluate_ghl_service.sbatch)을
# 한 번 제출한다. 다시 자동 제출하려면 experiments/01_ghl_main/results/.evaluation_submitted 폴더를 지운다.
#SBATCH --job-name=ghl_runs
#SBATCH --partition=laal_3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
done

if [ "$status" -eq 124 ]; then
    next=tests/ghl_main/run_ghl_candidates.sbatch
    sbatch "$next" "$@" || echo "재제출 실패: sbatch $next $*"
elif [ "$status" -eq 0 ] && "$PYTHON" -m tests.ghl_main.run_ghl_candidates --check-complete "$@" \
        && mkdir experiments/01_ghl_main/results/.evaluation_submitted 2>/dev/null; then
    next=tests/ghl_main/evaluate_ghl_service.sbatch
    sbatch "$next" || echo "제출 실패: sbatch $next"
fi
exit "$status"
