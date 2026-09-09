"""Lightning Linux GPU에서 Dev18 환경을 봉인하고 exact panel을 재개한다."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
PYTORCH_ALLOC_CONF_VALUE = "expandable_segments:True"


def configure_cuda_environment() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTORCH_ALLOC_CONF"] = PYTORCH_ALLOC_CONF_VALUE


configure_cuda_environment()


def require_lightning_cuda(*, system_name=None, cuda_available=None) -> None:
    """Windows 봉인 혼입과 무료 CPU Studio의 실험 실행을 차단한다."""
    configure_cuda_environment()
    system_name = system_name or platform.system()
    if system_name != "Linux":
        raise RuntimeError("Lightning 진입점은 Linux Studio에서만 실행한다")
    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("Lightning Studio를 GPU로 전환한 뒤 실행한다")


def seal_lightning_runtime(
    *, set_reproducible_seed, collect_environment_identity,
    ensure_runtime_snapshot,
) -> dict:
    """trial과 같은 결정론 상태를 먼저 만든 뒤 실행 환경을 봉인한다."""
    set_reproducible_seed(0)
    environment = collect_environment_identity()
    return ensure_runtime_snapshot(environment=environment)


def validate_then_seal_runtime(*, validate_resource_gate, seal_runtime) -> dict:
    """현재 GPU의 자원 gate가 통과된 뒤에만 실행 환경을 봉인한다."""
    resource_gate = validate_resource_gate()
    return {"resource_gate": resource_gate, "runtime": seal_runtime()}


def require_legacy_registry(registry=None) -> None:
    """새 실험을 이전 예산·manifest 경로로 실행하지 않는다."""
    if registry is None:
        from src.common.model_registry import load_model_registry

        registry = load_model_registry()
    if any(model.get("target_use") == "fit_full_prefix" for model in registry["models"].values()):
        raise RuntimeError("새 full-prefix 실험은 python -m tests.ghl_main.run_ratio_tuning 경로를 사용한다")


def main() -> None:
    from src.common.set_reproducible_seed import set_reproducible_seed
    from tests.checks.check_dev18_resources import validate_resource_report
    from tests.checks.seal_runtime_environment import (
        collect_runtime_environment_identity,
        ensure_runtime_snapshot,
    )
    from tests.ghl_main.run_dev18_tuning import DEFAULT_DATA_ROOT, run_tuning

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    arguments = parser.parse_args()
    require_legacy_registry()
    require_lightning_cuda()
    set_reproducible_seed(0)
    validate_then_seal_runtime(
        validate_resource_gate=validate_resource_report,
        seal_runtime=lambda: seal_lightning_runtime(
            set_reproducible_seed=set_reproducible_seed,
            collect_environment_identity=collect_runtime_environment_identity,
            ensure_runtime_snapshot=ensure_runtime_snapshot,
        ),
    )
    result = run_tuning(data_root=arguments.data_root, device="cuda")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
