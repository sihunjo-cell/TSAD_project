"""새 registry 모델의 scalar·채널 점수를 같은 계약으로 저장한다."""

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import numpy

from src.common.build_config_id import build_common_recipe_id
from src.common.execution_identity import (
    expected_dataset_for_identity,
    validate_execution_identity,
    EXECUTION_IDENTITY_FIELDS,
)
from src.common.execution_evidence import (
    FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
    FULL_PREFIX_STORAGE_SCHEMA_VERSION,
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
CALIBRATION_MODES = ("none", "validation_median_iqr", "fit_median_iqr", "official_full_evaluation")
NATIVE_METADATA_FIELDS = (
    "native_source_start", "native_source_end_exclusive",
    "boundary_repeat", "boundary_policy", "lookahead",
    "maximum_effective_lookahead",
    "input_normalization", "input_normalization_scope", "persistent_target_fit",
    "native_postprocessing", "official_protocol", "fit_source", "actual_fit_row_count",
    "checkpoint_selection", "training_data_scope", "rank_threshold_policy",
    "score_postprocessing", "fit_source_range", "window_normalization", "zero_pruning",
    "zero_pruned_window_feature_count", "retained_window_feature_count", "pca_solver",
    "native_postprocessing_recipe", "training_context_rows",
    "component_score_ranges", "score_normalization_source_start", "score_normalization_source_end_exclusive",
    "native_smoothing_window", "boundary_repeat_scope", "official_procedure",
    "native_calibration",
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
    official = common_recipe.get("methodology_revision") == "paper_tuning_v4"
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
    if official:
        expected = {
            "smoothing.kind": "model_native", "smoothing.window": 0,
            "smoothing.boundary": "model_native",
            "order.raw": "model_native_then_aggregate",
            "order.smoothed": "identical_to_raw_native_score",
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
    if mode in {"none", "official_full_evaluation"}:
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
    temporary_path = reference_path.with_name(f".{reference_name}.tmp")
    try:
        with temporary_path.open("wb") as destination:
            numpy.savez_compressed(destination, **{
                f"session_{index:03d}": scores
                for index, scores in enumerate(references)
            })
        temporary_path.replace(reference_path)
    finally:
        temporary_path.unlink(missing_ok=True)
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


def _write_metadata(path: Path, metadata: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_gdn_native_channels(channel_scores, scalar_scores, calibration, *,
                                 source_start: int, source_end_exclusive: int):
    """공식 GDN의 집계 전 점수와 실제 교정 통계를 같은 범위로 확인한다."""
    channels = validate_scores(channel_scores, "GDN native_channel_scores")
    scalar = validate_scores(scalar_scores, "GDN scores")
    if (channels.ndim != 2 or scalar.ndim != 1 or len(channels) != len(scalar)
            or source_end_exclusive - source_start != len(scalar)):
        raise ValueError("GDN native 채널 점수 shape가 scalar 범위와 다르다")
    if not numpy.array_equal(channels.max(axis=1), scalar):
        raise ValueError("GDN native 채널 max가 최종 scalar 점수와 다르다")
    expected = {
        "source": "full_evaluation", "source_start": source_start,
        "source_end_exclusive": source_end_exclusive, "score_space": "absolute_error",
        "epsilon": 0.01,
    }
    if (not isinstance(calibration, dict)
            or any(calibration.get(field) != value for field, value in expected.items())
            or any(type(calibration.get(field)) is not int
                   for field in ("source_start", "source_end_exclusive"))):
        raise ValueError("GDN native_calibration의 출처·범위·epsilon이 다르다")
    for field in ("median", "iqr"):
        values = calibration.get(field)
        if (not isinstance(values, list) or len(values) != channels.shape[1]
                or any(type(value) not in (int, float) for value in values)):
            raise ValueError(f"GDN native_calibration.{field}의 채널 shape가 다르다")
        values = numpy.asarray(values, dtype=float)
        if not numpy.isfinite(values).all() or numpy.any(values < 0):
            raise ValueError(f"GDN native_calibration.{field}는 유한한 0 이상이어야 한다")
    return channels


def validate_tspulse_native_calibration(calibration, *, source_start, source_end_exclusive,
                                       score_variant, input_column=None) -> None:
    """공식 TSPulse 교정 수치를 head·평가 범위·입력 채널에 연결한다."""
    expected = {"schema_version": 1, "source": "full_evaluation", "source_start": source_start,
                "source_end_exclusive": source_end_exclusive, "score_head": score_variant,
                "output_maximum_epsilon": 1e-5}
    if (not isinstance(calibration, dict) or score_variant not in {"time", "fft", "pred", "ensemble"}
            or any(calibration.get(name) != value for name, value in expected.items())
            or any(type(calibration.get(name)) is not int for name in
                   ("schema_version", "source_start", "source_end_exclusive"))):
        raise ValueError("TSPulse native_calibration의 head·출처·범위가 다르다")
    count = source_end_exclusive - source_start
    if source_start != 0 or count < 1:
        raise ValueError("TSPulse 교정 범위는 전체 평가 입력이어야 한다")

    def read_scaler_values(state, fields, width):
        if (not isinstance(state, dict) or type(state.get("sample_count")) is not int
                or state["sample_count"] != count):
            raise ValueError("TSPulse scaler의 sample_count가 평가 길이와 다르다")
        arrays = {}
        for field in fields:
            values = state.get(field)
            if (not isinstance(values, list) or len(values) != width
                    or any(type(value) not in (int, float) for value in values)
                    or not numpy.isfinite(values).all()):
                raise ValueError(f"TSPulse scaler.{field}의 shape 또는 수치가 잘못됐다")
            arrays[field] = numpy.asarray(values)
        return arrays

    input_state = calibration.get("input_standard_scaler")
    means = input_state.get("mean") if isinstance(input_state, dict) else None
    if not isinstance(means, list) or not means or input_column is not None and len(means) != input_column:
        raise ValueError("TSPulse input_standard_scaler의 채널 수가 다르다")
    inputs = read_scaler_values(input_state, ("mean", "variance", "scale"), len(means))
    if numpy.any(inputs["variance"] < 0) or numpy.any(inputs["scale"] <= 0):
        raise ValueError("TSPulse 입력 variance·scale이 잘못됐다")
    output = read_scaler_values(calibration.get("output_minmax_scaler"),
                                ("data_min", "data_max", "data_range", "scale", "offset"), 1)
    if (numpy.any(output["data_range"] < 0) or numpy.any(output["scale"] <= 0)
            or not numpy.array_equal(output["data_range"], output["data_max"] - output["data_min"])):
        raise ValueError("TSPulse 출력 minmax 범위·scale이 잘못됐다")
    for field in ("output_maximum", "output_maximum_divisor"):
        if type(calibration.get(field)) not in (int, float) or not numpy.isfinite(calibration[field]):
            raise ValueError(f"TSPulse {field}는 유한한 수여야 한다")
    if calibration["output_maximum_divisor"] != calibration["output_maximum"] + 1e-5:
        raise ValueError("TSPulse 출력 최대값과 정규화 divisor가 다르다")
    heads = calibration.get("head_minmax")
    expected_heads = {"time", "fft", "pred"} if score_variant == "ensemble" else {score_variant}
    if not isinstance(heads, dict) or heads.keys() != expected_heads:
        raise ValueError("TSPulse 내부 head 교정 수치가 빠졌거나 다른 head를 가리킨다")
    for state in heads.values():
        fields = ("data_min", "data_max", "data_range", "score_exponent",
                  "least_significant_scale", "least_significant_score")
        if (not isinstance(state, dict) or type(state.get("sample_count")) is not int
                or state["sample_count"] != count
                or state.get("transform") != "sklearn.preprocessing.MinMaxScaler()"
                or any(type(state.get(field)) not in (int, float) or not numpy.isfinite(state[field])
                       for field in fields)
                or state["data_range"] < 0 or state["data_range"] != state["data_max"] - state["data_min"]
                or (state["score_exponent"], state["least_significant_scale"], state["least_significant_score"])
                != (1.0, 0.0, 1.0)):
            raise ValueError("TSPulse 내부 head minmax 범위 또는 공식 교정값이 다르다")


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
    calibration_scores=(),
    calibration_source_starts=(),
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
    if (common_recipe.get("storage_schema_version") == FULL_PREFIX_STORAGE_SCHEMA_VERSION
            and evidence["measurement_protocol_id"] != FULL_PREFIX_MEASUREMENT_PROTOCOL_ID):
        raise ValueError("현재 recipe에는 새 full-prefix 저장 증거가 필요하다")
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
    full_prefix = target_use == "fit_full_prefix"
    official = common_recipe.get("methodology_revision") == "paper_tuning_v4"
    expected_calibration_mode = (
        ("fit_median_iqr" if model == "GDN" else "none") if full_prefix
        else "none" if target_use in TARGET_FREE_USES else "validation_median_iqr"
    )
    if official:
        expected_calibration_mode = ("official_full_evaluation" if model in {"GDN", "TSPulse"}
                                     else "none")
        if output.get("native_postprocessing") is not True or output.get("official_protocol") != "paper_tuning_v4":
            raise ValueError("공식 튜닝 출력에는 완료한 native 후처리와 절차 표시가 필요하다")
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
    if official:
        if any(len(values) for values in (validation_scores, validation_source_starts,
                                          calibration_scores, calibration_source_starts)):
            raise ValueError("공식 후처리 점수에 외부 교정 참조를 다시 적용하지 않는다")
        reference_scores, reference_starts = (), ()
    elif full_prefix:
        if len(validation_scores) or len(validation_source_starts):
            raise ValueError("full-prefix 실행에는 validation 참조를 넘기지 않는다")
        recipe_key = "full_prefix_channels" if model == "GDN" else "full_prefix_scalar"
        if _read_recipe_value(common_recipe, f"score_calibration.{recipe_key}") != expected_calibration_mode:
            raise ValueError("full-prefix calibration recipe가 모델과 다르다")
        if normalization_scope != ("current_prefix_fit" if model == "GDN" else "none"):
            raise ValueError("full-prefix normalization_scope가 모델과 다르다")
        reference_scores = calibration_scores
        reference_starts = calibration_source_starts
    else:
        if len(calibration_scores) or len(calibration_source_starts):
            raise ValueError("legacy 실행에는 fit calibration 참조를 넘기지 않는다")
        reference_scores = validation_scores
        reference_starts = validation_source_starts
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
    if (common_recipe.get("methodology_revision") == "source_faithful_v3"
            and model in {"TimeRCD", "TSPulse"}):
        expected_scope = "valid_inference_block" if model == "TimeRCD" else "past_context"
        if (output.get("input_normalization") != "inference_context_zscore"
                or output.get("input_normalization_scope") != expected_scope
                or output.get("persistent_target_fit") is not False):
            raise ValueError("Tier 3 입력 정규화가 등록한 추론 문맥 범위와 다르다")

    scores = validate_scores(output["scores"], "scores")
    if full_prefix and scores.ndim != (2 if model == "GDN" and not official else 1):
        raise ValueError("full-prefix 모델의 scalar·채널 출력 계약이 다르다")
    source_start = int(output["source_start"])
    source_end = int(output["source_end_exclusive"])
    if source_start < 0 or source_end - source_start != len(scores):
        raise ValueError("score 길이와 source 범위가 다르다")
    native_channels = None
    if official and model == "GDN":
        native_channels = validate_gdn_native_channels(
            output.get("native_channel_scores"), scores, output.get("native_calibration"),
            source_start=source_start, source_end_exclusive=source_end,
        )
    if official and model == "TSPulse":
        validate_tspulse_native_calibration(output.get("native_calibration"),
            source_start=source_start, source_end_exclusive=source_end, score_variant=score_variant)
    if len(evidence["test_sessions"]) != 1:
        raise ValueError("score 하나에는 execution_evidence test session 하나가 필요하다")
    if evidence["test_sessions"][0]["observation_count"] != source_end:
        raise ValueError(
            "execution_evidence test observation_count가 score source 범위와 다르다"
        )
    calibrated, references, median, iqr = _calibrate(
        output, reference_scores, epsilon,
    )
    if not references:
        source_starts = ()
    else:
        source_starts = tuple(reference_starts) or (0,) * len(references)
        if len(source_starts) != len(references):
            raise ValueError("validation score와 source start 수가 다르다")
    validate_training_evidence_bindings(
        evidence,
        ratio=ratio,
        validation_session_lengths=tuple(len(reference) for reference in references),
        validation_source_starts=source_starts,
        target_use=target_use if full_prefix else "fit_validation",
        calibration_source=("fit" if references else "none") if full_prefix else "validation",
    )

    output_dir = Path(output_dir)
    score_paths = save_score_arrays(
        native_channels if native_channels is not None else calibrated,
        str(output_dir), dataset, series, model, tier, ratio, seed,
        "trainnorm", smoothing_window,
        native_postprocessing=official,
    )
    raw_name = build_score_filename(
        dataset, series, model, tier, ratio, seed, "raw", "trainnorm",
    )
    calibration_reference_path, calibration_reference = _save_calibration_reference(
        references, source_starts, median, iqr, output_dir, raw_name,
    )
    if full_prefix and calibration_reference is not None:
        calibration_reference["source"] = "fit"
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
    if evidence["measurement_protocol_id"] == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID:
        metadata["storage_schema_version"] = FULL_PREFIX_STORAGE_SCHEMA_VERSION
    if native_channels is not None:
        metadata["native_channel_score_shape"] = list(native_channels.shape)
    metadata.update({
        field: output[field]
        for field in NATIVE_METADATA_FIELDS
        if field in output
    })
    _write_metadata(metadata_path, metadata)
    return {
        "score_paths": score_paths,
        "metadata_path": str(metadata_path),
        "calibration_reference_path": calibration_reference_path,
    }


def save_execution_result(
    result: dict, output_dir, *, spec: dict, dataset: str, series: int,
    execution_evidence: dict, test_session_index: int = 0, score_variant=None,
) -> dict:
    """등록 실행 결과의 참조 출처를 유지해 한 test 점수를 저장한다."""
    full_prefix = spec["target_use"] == "fit_full_prefix"
    references = tuple(result.get("calibration_outputs" if full_prefix else "validation_outputs") or ())
    for output in references:
        if output["source_end_exclusive"] - output["source_start"] != len(output["scores"]):
            raise ValueError("calibration output의 source 범위와 score 길이가 다르다")
    output = result["test_outputs"][test_session_index]
    if score_variant is not None:
        output = output[score_variant]
    evidence = deepcopy(execution_evidence)
    evidence["test_sessions"] = [evidence["test_sessions"][test_session_index]]
    reference_arguments = {
        ("calibration_scores" if full_prefix else "validation_scores"): tuple(reference["scores"] for reference in references),
        ("calibration_source_starts" if full_prefix else "validation_source_starts"): tuple(reference["source_start"] for reference in references),
    }
    saved = save_model_score(
        output, output_dir, dataset=dataset, series=series,
        model=spec["model"], target_use=spec["target_use"], tier=spec["tier"],
        ratio=spec["ratio"], seed=spec["seed"], config_id=spec["config_id"],
        common_recipe=spec["common_recipe"], common_recipe_id=spec["common_recipe_id"],
        normalization_scope=output["normalization_scope"], score_variant=score_variant,
        config_registry_sha256=spec["config_registry_sha256"],
        execution_identity={field: spec[field] for field in EXECUTION_IDENTITY_FIELDS},
        execution_evidence=evidence, **reference_arguments,
    )
    extra = {field: result[field] for field in ("effective_execution", "training_protocol")
             if result.get(field) is not None}
    if extra:
        path = Path(saved["metadata_path"])
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(extra)
        _write_metadata(path, metadata)
    return saved
