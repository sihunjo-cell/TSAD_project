"""새 registry 모델의 scalar·채널 점수를 같은 계약으로 저장한다."""

import hashlib
import json
import re
from pathlib import Path

import numpy

from src.common.build_config_id import build_common_recipe_id
from src.common.execution_identity import (
    expected_dataset_for_identity,
    validate_execution_identity,
)
from src.common.execution_evidence import (
    TARGET_FREE_USES,
    validate_execution_evidence_for_run,
    validate_training_evidence_bindings,
)
from src.common.naming import build_score_filename
from src.common.model_registry import model_registry_sha256
from src.common.normalization import apply_median_iqr, estimate_median_iqr
from src.common.save_scores import save_score_arrays, validate_scores


CONFIG_ID_PATTERN = re.compile(r"^c[0-9a-f]{12}$")
COMMON_RECIPE_ID_PATTERN = re.compile(r"^r[0-9a-f]{12}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SCORE_VARIANT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
CALIBRATION_MODES = ("none", "validation_median_iqr")
NATIVE_METADATA_FIELDS = (
    "native_source_start", "native_source_end_exclusive",
    "boundary_repeat", "boundary_policy", "lookahead",
    "maximum_effective_lookahead",
)


def _read_recipe_value(recipe: dict, path: str):
    value = recipe
    try:
        for key in path.split("."):
            value = value[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"common_recipe 필드가 빠졌다: {path}") from error
    return value


def _validate_common_recipe(common_recipe: dict, common_recipe_id: str) -> tuple[float, int]:
    if not COMMON_RECIPE_ID_PATTERN.fullmatch(common_recipe_id):
        raise ValueError(f"잘못된 common_recipe_id이다: {common_recipe_id}")
    if build_common_recipe_id(common_recipe) != common_recipe_id:
        raise ValueError("common_recipe와 common_recipe_id가 다르다")
    expected = {
        "score_calibration.fit_validation": "validation_median_iqr",
        "score_calibration.target_free": "none",
        "smoothing.kind": "trailing_mean",
        "smoothing.window": 4,
        "smoothing.boundary": "first_three_timesteps_zero",
        "order.raw": "normalize_then_aggregate",
        "order.smoothed": "normalize_then_smooth_per_channel_then_aggregate",
        "aggregation.channel_scores": "max",
        "aggregation.scalar_scores": "model_native",
    }
    for path, supported in expected.items():
        actual = _read_recipe_value(common_recipe, path)
        if actual != supported:
            raise ValueError(
                f"지원하지 않는 common_recipe {path}: {actual!r} != {supported!r}"
            )
    epsilon = _read_recipe_value(common_recipe, "score_calibration.epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon < 0:
        raise ValueError("common_recipe score_calibration.epsilon은 0 이상이어야 한다")
    return float(epsilon), expected["smoothing.window"]


def _calibrate(output: dict, validation_scores, epsilon: float):
    scores = validate_scores(output["scores"], "scores")
    mode = output["calibration_mode"]
    if mode not in CALIBRATION_MODES:
        raise ValueError(f"지원하지 않는 calibration_mode이다: {mode}")
    if validation_scores is None:
        supplied = ()
    elif isinstance(validation_scores, numpy.ndarray):
        supplied = (validation_scores,) if len(validation_scores) else ()
    else:
        supplied = tuple(validation_scores)
    if mode == "none":
        if len(supplied):
            raise ValueError("calibration_mode=none에는 validation 점수를 넘기지 않는다")
        return scores, (), None, None
    if epsilon < 0:
        raise ValueError("epsilon은 0 이상이어야 한다")
    references = tuple(
        validate_scores(values, "validation_scores") for values in supplied
    )
    if len(references) == 0:
        raise ValueError("validation_median_iqr에는 validation 점수가 필요하다")
    if any(reference.ndim != scores.ndim for reference in references):
        raise ValueError("validation과 test 점수 차원이 다르다")
    if scores.ndim == 2 and any(
        reference.shape[1] != scores.shape[1] for reference in references
    ):
        raise ValueError("validation과 test 채널 수가 다르다")
    reference = numpy.concatenate(references, axis=0)
    median, iqr = estimate_median_iqr(reference)
    return apply_median_iqr(scores, median, iqr, epsilon), references, median, iqr


def _save_calibration_reference(
    references, source_starts, median, iqr, output_dir: Path, raw_name: str,
):
    if not references:
        return None, None
    reference_name = (
        raw_name.removesuffix("__raw__trainnorm.npy")
        + "__calibration_reference.npz"
    )
    reference_path = output_dir / reference_name
    numpy.savez_compressed(reference_path, **{
        f"session_{index:03d}": scores
        for index, scores in enumerate(references)
    })
    with reference_path.open("rb") as reference_file:
        reference_sha256 = hashlib.file_digest(reference_file, "sha256").hexdigest()
    metadata = {
        "file": reference_name,
        "sha256": reference_sha256,
        "session_shapes": [list(scores.shape) for scores in references],
        "source_starts": list(source_starts),
        "sample_count": sum(len(scores) for scores in references),
        "median": numpy.asarray(median).tolist(),
        "iqr": numpy.asarray(iqr).tolist(),
        "score_space": "adapter_native_pre_calibration",
    }
    return str(reference_path), metadata


def save_model_score(
    output: dict,
    output_dir,
    *,
    dataset: str,
    series: int,
    model: str,
    target_use: str,
    tier: str,
    ratio: int,
    seed: int,
    config_id: str,
    common_recipe: dict,
    common_recipe_id: str,
    normalization_scope: str,
    validation_scores=(),
    validation_source_starts=(),
    score_variant: str | None = None,
    config_registry_sha256: str | None = None,
    execution_identity: dict,
    execution_evidence: dict,
) -> dict:
    """adapter가 선언한 정렬과 calibration만 적용해 점수·metadata를 저장한다."""
    identity = validate_execution_identity(execution_identity)
    evidence = validate_execution_evidence_for_run(
        execution_evidence,
        dataset_role=identity["dataset_role"],
        target_use=target_use,
    )
    expected_dataset = expected_dataset_for_identity(identity)
    if dataset != expected_dataset:
        raise ValueError(
            f"execution identity의 dataset과 저장 대상이 다르다: "
            f"{expected_dataset} != {dataset}"
        )
    required = {
        "scores", "source_start", "source_end_exclusive", "alignment",
        "primitive", "calibration_mode", "evaluation_mode", "lookahead",
        "maximum_effective_lookahead", "normalization_scope",
    }
    missing = required - output.keys()
    if missing:
        raise ValueError(f"adapter score 필드가 빠졌다: {sorted(missing)}")
    expected_calibration_mode = (
        "none" if target_use in TARGET_FREE_USES else "validation_median_iqr"
    )
    if output["calibration_mode"] != expected_calibration_mode:
        raise ValueError(
            "target_use와 calibration_mode가 다르다: "
            f"{target_use}에는 {expected_calibration_mode}이 필요하다"
        )
    if not CONFIG_ID_PATTERN.fullmatch(config_id):
        raise ValueError(f"잘못된 config_id이다: {config_id}")
    epsilon, smoothing_window = _validate_common_recipe(
        common_recipe, common_recipe_id,
    )
    if score_variant is not None and not SCORE_VARIANT_PATTERN.fullmatch(score_variant):
        raise ValueError(f"잘못된 score_variant이다: {score_variant}")
    registry_sha = config_registry_sha256 or model_registry_sha256()
    if not SHA256_PATTERN.fullmatch(registry_sha):
        raise ValueError("config_registry_sha256은 64자리 소문자 hex여야 한다")
    if output["evaluation_mode"] not in ("offline_noncausal", "causal"):
        raise ValueError("evaluation_mode가 공통 계약과 다르다")
    for field in ("lookahead", "maximum_effective_lookahead"):
        if type(output[field]) is not int or output[field] < 0:
            raise ValueError(f"{field}는 0 이상의 정수여야 한다")
    if output["normalization_scope"] != normalization_scope:
        raise ValueError("adapter와 saver의 normalization_scope가 다르다")

    scores = validate_scores(output["scores"], "scores")
    source_start = int(output["source_start"])
    source_end = int(output["source_end_exclusive"])
    if source_start < 0 or source_end - source_start != len(scores):
        raise ValueError("score 길이와 source 범위가 다르다")
    if len(evidence["test_sessions"]) != 1:
        raise ValueError("score 하나에는 execution_evidence test session 하나가 필요하다")
    if evidence["test_sessions"][0]["observation_count"] != source_end:
        raise ValueError(
            "execution_evidence test observation_count가 score source 범위와 다르다"
        )
    calibrated, references, median, iqr = _calibrate(
        output, validation_scores, epsilon,
    )
    if not references:
        source_starts = ()
    else:
        source_starts = tuple(validation_source_starts) or (0,) * len(references)
        if len(source_starts) != len(references):
            raise ValueError("validation score와 source start 수가 다르다")
    validate_training_evidence_bindings(
        evidence,
        ratio=ratio,
        validation_session_lengths=tuple(len(reference) for reference in references),
        validation_source_starts=source_starts,
    )

    output_dir = Path(output_dir)
    score_paths = save_score_arrays(
        calibrated, str(output_dir), dataset, series, model, tier, ratio, seed,
        "trainnorm", smoothing_window,
    )
    raw_name = build_score_filename(
        dataset, series, model, tier, ratio, seed, "raw", "trainnorm",
    )
    calibration_reference_path, calibration_reference = _save_calibration_reference(
        references, source_starts, median, iqr, output_dir, raw_name,
    )
    metadata_path = output_dir / raw_name.replace(".npy", ".meta.json")
    metadata = {
        **identity,
        "execution_evidence": evidence,
        "dataset": dataset,
        "series": series,
        "model": model,
        "target_use": target_use,
        "tier": tier,
        "ratio": ratio,
        "seed": seed,
        "config_id": config_id,
        "common_recipe_id": common_recipe_id,
        "config_registry_sha256": registry_sha,
        "score_variant": score_variant,
        "window_size": source_start,
        "test_length": evidence["test_sessions"][0]["observation_count"],
        "score_length": len(scores),
        "score_shape": list(scores.shape),
        "channel_count": scores.shape[1] if scores.ndim == 2 else 0,
        "aggregation_mode": "max" if scores.ndim == 2 else "model_native_scalar",
        "source_start": source_start,
        "source_end_exclusive": source_end,
        "label_slice": [source_start, source_end],
        "alignment": output["alignment"],
        "primitive": output["primitive"],
        "calibration_mode": output["calibration_mode"],
        "calibration_reference": calibration_reference,
        "epsilon": epsilon,
        "normalization_scope": normalization_scope,
        "evaluation_mode": output["evaluation_mode"],
        "smoothing": {
            "kind": common_recipe["smoothing"]["kind"],
            "window": smoothing_window,
            "boundary": common_recipe["smoothing"]["boundary"],
        },
        "pipeline_order": common_recipe["order"],
    }
    metadata.update({
        field: output[field]
        for field in NATIVE_METADATA_FIELDS
        if field in output
    })
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {
        "score_paths": score_paths,
        "metadata_path": str(metadata_path),
        "calibration_reference_path": calibration_reference_path,
    }
