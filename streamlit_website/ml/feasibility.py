"""미래 stage의 데이터 규모를 기준으로 구조적 feasibility를 다시 판정한다.

모델별 최소 길이 판정 로직은 새로 만들지 않는다 — `src/common/model_feasibility.py
::assess_candidate()`가 이미 MWVAR/SQDIFF류/PCA_LEGACY/PaAno/GDN/TimeRCD/TSPulse
전부를 구현하고 있으므로 그대로 재사용한다. 이 모듈은 "미래 stage의 행 수 →
fit/validation/test length" 변환만 새로 만든다.

원칙: 현재 데이터가 부족해서 지금 infeasible이어도, 미래 stage에서 행 수가 충분해
지면 그 stage에서는 다시 feasible일 수 있다 — 이 모듈은 stage마다 독립적으로
판정하고 이전 stage의 판정을 이어받지 않는다.
"""

from __future__ import annotations

from src.common.model_feasibility import assess_candidate
from src.data_split.split_ratio_prefix import compute_prefix_counts

FULL_PREFIX_TARGET_USE = "fit_full_prefix"


def compute_stage_training_lengths(target_use: str, stage_training_rows: int) -> tuple[int, int]:
    """그 stage까지 누적될 학습 행 수에서 (fit_length, validation_length)를 만든다.

    `compute_prefix_counts(total_length, ratio_percent=100, ...)`를 재사용한다 —
    "그 stage에서 확보할 전체 학습 데이터(100%)"라는 뜻으로 ratio_percent=100을
    쓰는 것이며, Dev18의 q_percent(과거 실험 비율)와는 별개의 용법이다.
    """
    if stage_training_rows < 0:
        raise ValueError("stage_training_rows는 0 이상이어야 한다")
    full_prefix = target_use == FULL_PREFIX_TARGET_USE
    _available, fit_count, validation_count = compute_prefix_counts(
        stage_training_rows, 100, full_prefix=full_prefix,
    )
    return fit_count, validation_count


def assess_future_feasibility(
    candidate: dict, *, stage_training_rows: int, stage_inference_rows: int, channel_count: int,
) -> dict:
    """한 candidate가 미래 한 stage에서 구조적으로 실행 가능한지 판정한다.

    `candidate`는 `db_connection/filter_candidates.py`가 만드는 1차 축소 통과
    후보 dict다 (`model`, `parameters`, `target_use`, `recipe` 등을 가진다).
    반환값은 `assess_candidate()`와 동일한 형태:
    `{"status": "feasible"|"structurally_infeasible", "reason": str, "derived": dict}`.
    """
    model = candidate["model"]
    hyperparameters = candidate["parameters"]
    target_use = candidate.get("target_use", "")
    recipe = candidate.get("recipe") or {}
    official_protocol = recipe.get("methodology_revision") == "paper_tuning_v4"
    fit_length, validation_length = compute_stage_training_lengths(target_use, stage_training_rows)
    return assess_candidate(
        model, hyperparameters, fit_length, validation_length, stage_inference_rows, channel_count,
        full_prefix=target_use == FULL_PREFIX_TARGET_USE, official_protocol=official_protocol,
    )
