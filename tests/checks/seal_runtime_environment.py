"""현재 Python 환경을 재개 가능한 로컬 runtime snapshot으로 봉인한다."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

from src.common.execution_identity import file_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNTIME_PATH = REPOSITORY_ROOT / ".runtime" / "runtime.json"
TRACKED_RUNTIME_PATH = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
    / "environment" / "runtime.json"
)


def _installed_packages() -> list[str]:
    return subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()


def collect_cuda_device_identity(*, torch_module=None, driver_version=None) -> dict:
    """재개 중 GPU 종류와 driver가 바뀌지 않도록 장치 신원을 읽는다."""
    if torch_module is None:
        import torch as torch_module
    if driver_version is None:
        output = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        versions = {row.strip() for row in output if row.strip()}
        if len(versions) != 1:
            raise RuntimeError("NVIDIA driver version을 하나로 확정하지 못했다")
        driver_version = versions.pop()
    device = torch_module.cuda.current_device()
    properties = torch_module.cuda.get_device_properties(device)
    return {
        "name": torch_module.cuda.get_device_name(device),
        "compute_capability": list(
            torch_module.cuda.get_device_capability(device)
        ),
        "total_memory_bytes": properties.total_memory,
        "driver_version": driver_version,
    }


def collect_runtime_environment_identity() -> dict:
    """기존 package 신원에 활성 CUDA 장치 신원을 덧붙인다."""
    from tests.checks.run_checkpoint_smoke import collect_environment_identity

    environment = collect_environment_identity()
    if environment["torch"]["cuda_available"]:
        environment = {
            **environment,
            "cuda_device": collect_cuda_device_identity(),
        }
    return environment


def _runtime_files(repository_root: Path) -> tuple[Path, Path]:
    return (
        repository_root / "src" / "models" / "requirements.txt",
        repository_root / "configs" / "environment.yaml",
    )


def _sealed_packages(runtime: dict, runtime_path: Path) -> list[str]:
    value = runtime.get("pip_freeze")
    if isinstance(value, list) and all(isinstance(row, str) for row in value):
        return value
    if isinstance(value, str):
        path = runtime_path.parent / value
        if path.is_file():
            return path.read_text(encoding="utf-8").splitlines()
    raise ValueError("runtime snapshot의 pip freeze 봉인이 없다")


def validate_runtime_snapshot(
    environment: dict, *, repository_root=REPOSITORY_ROOT,
    installed_packages=None, python_executable=None,
) -> dict:
    """로컬 봉인을 우선 검증하고 없으면 기존 추적 snapshot을 쓴다."""
    repository_root = Path(repository_root)
    local_path = repository_root / ".runtime" / "runtime.json"
    tracked_path = (
        repository_root / "experiments" / "checks" / "reference_code"
        / "environment" / "runtime.json"
    )
    runtime_path = local_path if local_path.is_file() else tracked_path
    if not runtime_path.is_file():
        raise FileNotFoundError(f"runtime snapshot이 없다: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    requirements_path, environment_path = _runtime_files(repository_root)
    current_packages = (
        _installed_packages() if installed_packages is None else list(installed_packages)
    )
    sealed_packages = _sealed_packages(runtime, runtime_path)
    current_python = Path(python_executable or sys.executable).resolve()
    valid = (
        runtime.get("environment") == "tsad_models_311"
        and Path(runtime.get("python_executable", "")).resolve() == current_python
        and runtime.get("requirements") == {
            "file": "src/models/requirements.txt",
            "sha256": file_sha256(requirements_path),
        }
        and runtime.get("environment_config") == {
            "file": "configs/environment.yaml",
            "sha256": file_sha256(environment_path),
        }
        and runtime.get("identity") == environment
        and sealed_packages == current_packages
    )
    if not valid:
        raise ValueError("현재 Python 환경이 runtime 봉인과 다르다")
    return {
        "sha256": file_sha256(runtime_path),
        "python_executable": str(current_python),
        "requirements": runtime["requirements"],
        "environment_config": runtime["environment_config"],
        "identity": environment,
        "pip_freeze": sealed_packages,
    }


def ensure_runtime_snapshot(
    *, repository_root=REPOSITORY_ROOT, environment=None,
    installed_packages=None, python_executable=None,
) -> dict:
    """첫 실행 환경을 봉인하고 이후에는 같은 환경만 재사용한다."""
    repository_root = Path(repository_root)
    runtime_path = repository_root / ".runtime" / "runtime.json"
    if environment is None:
        environment = collect_runtime_environment_identity()
    current_packages = (
        _installed_packages() if installed_packages is None else list(installed_packages)
    )
    current_python = Path(python_executable or sys.executable).resolve()
    if not runtime_path.is_file():
        requirements_path, environment_path = _runtime_files(repository_root)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = runtime_path.with_name(f".{runtime_path.name}.tmp")
        temporary_path.write_text(json.dumps({
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "environment": "tsad_models_311",
            "python_executable": str(current_python),
            "requirements": {
                "file": "src/models/requirements.txt",
                "sha256": file_sha256(requirements_path),
            },
            "environment_config": {
                "file": "configs/environment.yaml",
                "sha256": file_sha256(environment_path),
            },
            "identity": environment,
            "pip_freeze": current_packages,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(runtime_path)
    return validate_runtime_snapshot(
        environment,
        repository_root=repository_root,
        installed_packages=current_packages,
        python_executable=current_python,
    )
