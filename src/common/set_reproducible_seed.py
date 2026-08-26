"""Registry 실행의 Python·NumPy·PyTorch 결정론을 한곳에서 고정한다."""

import os
import random

import numpy
import torch


def set_reproducible_seed(seed: int) -> dict:
    """난수열과 지원되는 torch 결정론 상태를 고정하고 snapshot을 반환한다."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed는 0 이상의 정수여야 한다")

    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized():
        if configured_workspace not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "CUDA가 CUBLAS_WORKSPACE_CONFIG보다 먼저 초기화됐다"
            )
    else:
        configured_workspace = ":4096:8"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = configured_workspace
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    cudnn_available = torch.backends.cudnn.is_available()
    if cudnn_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "cublas_workspace_config": configured_workspace,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_seeded": cuda_available,
        "cudnn_deterministic": (
            torch.backends.cudnn.deterministic if cudnn_available else None
        ),
        "cudnn_benchmark": (
            torch.backends.cudnn.benchmark if cudnn_available else None
        ),
    }
