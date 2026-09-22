#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Lightning Linux Studio에서만 실행하세요." >&2
  exit 1
fi
python - <<'PY'
import sys

if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit("Python 3.11 또는 3.12 환경에서 실행하세요.")
print(f"설치 대상: {sys.executable} (Python {sys.version.split()[0]})", flush=True)
PY
python -m pip install --upgrade pip
python -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r src/models/requirements.txt
python - <<'PY'
from pathlib import Path

import torch
import yaml

from src.common.verify_run_context import verify_runtime_versions

versions = verify_runtime_versions(yaml.safe_load(Path("configs/environment.yaml").read_text(encoding="utf-8")))
assert torch.version.cuda == "12.6", f"CUDA 12.6 빌드가 필요하다: {torch.__version__}"
print(f"설치 완료: Python {versions['python']}, torch {versions['torch']}, CUDA {torch.version.cuda}")
PY

echo "설치가 끝났습니다. non-interruptible L4 Studio에서 기존 run_ratio_tuning 명령을 실행하세요."
