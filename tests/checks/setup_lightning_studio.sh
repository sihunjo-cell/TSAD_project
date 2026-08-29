#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Lightning Linux Studio에서만 실행하세요." >&2
  exit 1
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "Lightning 기본 conda 환경을 찾지 못했습니다." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda install --yes --channel conda-forge python=3.11.14
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r src/models/requirements.txt
python -c "import platform, torch; assert platform.python_version() == '3.11.14'; assert torch.__version__.split('+', 1)[0] == '2.10.0'; assert torch.version.cuda == '12.6'; print(platform.python_version(), torch.__version__, torch.version.cuda)"

echo "설치가 끝났습니다. Studio를 non-interruptible L4로 바꾼 뒤 아래 명령을 실행하세요."
echo "python tests/checks/check_dev18_resources.py"
echo "점검 결과가 passed일 때만 python tests/checks/run_lightning_dev18.py 를 실행하세요."
