"""registry 실행 하나의 점수·metadata 완결성을 검사한다."""

import hashlib
import json
from pathlib import Path

import numpy

from src.common.execution_identity import (
    EXECUTION_IDENTITY_FIELDS,
    expected_dataset_for_identity,
    validate_execution_identity,
    validate_input_manifest_role,
)
from src.common.execution_evidence import (
    DEV18_MEASUREMENT_PROTOCOL_ID,
    FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
    FULL_PREFIX_STORAGE_SCHEMA_VERSION,
    TARGET_FREE_USES,
    validate_execution_evidence_for_run,
    validate_training_evidence_bindings,
)
from src.common.load_final_membership import (
    build_final_execution_union,
    load_final_membership,
)
from src.common.naming import build_score_filename
from src.common.model_registry import load_model_registry_with_sha
from src.common.normalization import estimate_median_iqr
from src.common.save_model_artifacts import validate_gdn_native_channels, validate_tspulse_native_calibration


def _check_calibration_reference(
    output_directory: Path, metadata: dict, raw_name: str,
) -> tuple[int, ...]:
    if "calibration_reference" not in metadata:
        raise ValueError("교정 기준 metadata가 없다")
    reference = metadata["calibration_reference"]
    if metadata["calibration_mode"] == "fit_median_iqr" and (
        not isinstance(reference, dict) or reference.get("source") != "fit"
    ):
        raise ValueError("fit 교정 기준에는 source=fit 근거가 필요하다")
    if metadata["calibration_mode"] in {"none", "official_full_evaluation"}:
        if reference is not None:
            raise ValueError("외부 교정이 없는 점수에 교정 기준이 선언됐다")
        return (), ()
    if metadata["calibration_mode"] not in {"validation_median_iqr", "fit_median_iqr"}:
        raise ValueError("지원하지 않는 calibration_mode이다")
    required = {
        "file", "sha256", "session_shapes", "sample_count", "median", "iqr",
        "score_space",
    }
    if not isinstance(reference, dict) or required - reference.keys():
        raise ValueError("교정 기준 metadata가 불완전하다")
    reference_name = reference["file"]
    if not isinstance(reference_name, str) or Path(reference_name).name != reference_name:
        raise ValueError("교정 기준 파일명은 현재 산출물 폴더 안에 있어야 한다")
    expected_name = (
        raw_name.removesuffix("__raw__trainnorm.npy")
        + "__calibration_reference.npz"
    )
    if reference_name != expected_name:
        raise ValueError("교정 기준 파일명이 점수 파일 계약과 다르다")
    reference_path = output_directory / reference_name
    if not reference_path.is_file():
        raise ValueError(f"교정 기준 파일이 없다: {reference_name}")
    with reference_path.open("rb") as reference_file:
        actual_sha256 = hashlib.file_digest(reference_file, "sha256").hexdigest()
    if actual_sha256 != reference["sha256"]:
        raise ValueError("교정 기준 파일의 SHA-256이 metadata와 다르다")

    session_shapes = reference["session_shapes"]
    expected_keys = [
        f"session_{index:03d}" for index in range(len(session_shapes))
    ]
    with numpy.load(reference_path, allow_pickle=False) as saved_reference:
        if saved_reference.files != expected_keys or not expected_keys:
            raise ValueError("교정 기준 세션 구성이 metadata와 다르다")
        sessions = [saved_reference[key] for key in expected_keys]
        if [list(scores.shape) for scores in sessions] != session_shapes:
            raise ValueError("교정 기준 shape가 metadata와 다르다")
        if not all(numpy.isfinite(scores).all() for scores in sessions):
            raise ValueError("교정 기준 점수에 non-finite 값이 있다")
        reference_scores = numpy.concatenate(sessions, axis=0)
    if len(reference_scores) != reference["sample_count"]:
        raise ValueError("교정 기준 sample_count가 metadata와 다르다")
    median, iqr = estimate_median_iqr(reference_scores)
    saved_median = numpy.asarray(reference["median"])
    if saved_median.shape != median.shape or not numpy.array_equal(median, saved_median):
        raise ValueError("교정 기준 median이 metadata와 다르다")
    saved_iqr = numpy.asarray(reference["iqr"])
    if saved_iqr.shape != iqr.shape or not numpy.array_equal(iqr, saved_iqr):
        raise ValueError("교정 기준 IQR이 metadata와 다르다")
    if reference["score_space"] != "adapter_native_pre_calibration":
        raise ValueError("교정 기준 score_space가 공통 계약과 다르다")
    source_starts = reference.get("source_starts", [0] * len(sessions))
    if (
        type(source_starts) is not list
        or len(source_starts) != len(sessions)
        or any(type(start) is not int or start < 0 for start in source_starts)
    ):
        raise ValueError("교정 기준 source_starts가 잘못됐다")
    return tuple(len(scores) for scores in sessions), tuple(source_starts)


def check_registered_output(
    output_directory, spec: dict, *, dataset: str, series: int,
    input_manifest_path, final_policy_membership_path=None, score_variant=None,
) -> dict:
    """완결된 산출물만 resume 가능한 상태로 판정한다."""
    identity = validate_execution_identity(spec)
    expected_dataset = expected_dataset_for_identity(identity)
    if dataset != expected_dataset:
        raise ValueError(
            f"execution identity의 dataset과 검사 대상이 다르다: "
            f"{expected_dataset} != {dataset}"
        )
    current_manifest_sha256 = validate_input_manifest_role(
        input_manifest_path, identity["dataset_role"], identity["split_role"],
    )
    if current_manifest_sha256 != identity["input_manifest_sha256"]:
        raise ValueError("현재 input manifest SHA-256이 spec과 다르다")
    if identity["dataset_role"] == "development":
        if final_policy_membership_path is not None:
            raise ValueError("development 완료 검사에는 membership 경로를 줄 수 없다")
    elif final_policy_membership_path is None:
        raise ValueError("final 완료 검사에는 membership 경로가 필요하다")
    registry, current_registry_sha = load_model_registry_with_sha()
    if spec.get("config_registry_sha256") != current_registry_sha:
        raise ValueError("spec의 registry SHA가 현재 봉인본과 다르다")
    target_use = spec.get("target_use")
    if target_use not in {"training_free", "strict_zero_shot", "fit_validation", "fit_full_prefix"}:
        raise ValueError("spec의 target_use가 없거나 잘못됐다")
    model = registry.get("models", {}).get(spec.get("model"))
    if model is None or model.get("execution_status", "ready") != "ready":
        raise ValueError("spec의 model이 ready registry에 없다")
    if spec.get("tier") != model.get("tier"):
        raise ValueError("spec의 tier가 registry model과 다르다")
    if target_use != model.get("target_use"):
        raise ValueError("spec의 target_use가 registry model과 다르다")
    if spec.get("config_id") not in {
        candidate.get("config_id") for candidate in model.get("candidates", ())
    }:
        raise ValueError("spec의 config_id가 registry model과 다르다")
    if identity["dataset_role"] == "final":
        membership_sha256, membership = load_final_membership(
            final_policy_membership_path, registry,
        )
        if membership_sha256 != identity["final_policy_membership_sha256"]:
            raise ValueError("현재 final membership SHA-256이 spec과 다르다")
        execution_union = build_final_execution_union(
            membership, identity["split_role"], series=series,
        )
        physical_key = (spec.get("model"), spec.get("config_id"), spec.get("ratio"))
        if physical_key not in execution_union:
            raise ValueError("spec의 물리 실행이 final membership에 없다")
        if spec.get("score_variants") != execution_union[physical_key]:
            raise ValueError("spec의 score_variants가 final membership과 다르다")

    variants = spec.get("score_variants")
    if spec.get("model") == "TSPulse":
        if not isinstance(variants, tuple) or score_variant not in variants:
            raise ValueError("검사할 TSPulse score_variant가 물리 spec에 없다")
        expected_metadata_variant = score_variant
    else:
        if variants != ("",) or score_variant not in (None, ""):
            raise ValueError("non-TSPulse 물리 spec은 빈 score variant만 가져야 한다")
        expected_metadata_variant = None
    output_directory = Path(output_directory)
    raw_name = build_score_filename(
        dataset=dataset, series=series, model=spec["model"], tier=spec["tier"],
        ratio=spec["ratio"], seed=spec["seed"], smoothing_kind="raw",
        norm_kind="trainnorm",
    )
    metadata_path = output_directory / raw_name.replace(".npy", ".meta.json")
    if not metadata_path.is_file():
        raise ValueError(f"metadata가 없다: {metadata_path.name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_execution_identity(metadata)
    for field in EXECUTION_IDENTITY_FIELDS:
        if metadata[field] != identity[field]:
            raise ValueError(f"metadata와 spec의 {field}가 다르다")
    if metadata.get("config_registry_sha256") != spec["config_registry_sha256"]:
        raise ValueError("metadata의 registry SHA가 현재 봉인본과 다르다")
    if metadata.get("target_use") != target_use:
        raise ValueError("metadata와 spec의 target_use가 다르다")
    for field, value in (("dataset", dataset), ("series", series)):
        if metadata[field] != value:
            raise ValueError(
                f"metadata와 요청 대상이 다르다: {field}={metadata[field]!r} "
                f"!= {value!r}"
            )
    for field in ("model", "tier", "ratio", "seed", "config_id"):
        if metadata[field] != spec[field]:
            raise ValueError(
                f"metadata와 registry가 다르다: {field}={metadata[field]!r} "
                f"!= {spec[field]!r}"
            )
    if metadata.get("score_variant") != expected_metadata_variant:
        raise ValueError("metadata와 registry의 score_variant가 다르다")
    evidence = validate_execution_evidence_for_run(
        metadata.get("execution_evidence"),
        dataset_role=identity["dataset_role"], target_use=target_use,
    )
    full_prefix = spec["common_recipe"].get("training_split") == "full_prefix_v2"
    official = spec["common_recipe"].get("methodology_revision") == "paper_tuning_v4"
    expected_protocol = FULL_PREFIX_MEASUREMENT_PROTOCOL_ID if full_prefix else DEV18_MEASUREMENT_PROTOCOL_ID
    if identity["dataset_role"] == "development" and evidence["measurement_protocol_id"] != expected_protocol:
        raise ValueError("Dev18 measurement_protocol_id가 현재 분할 프로토콜과 다르다")
    if full_prefix and metadata.get("storage_schema_version") != FULL_PREFIX_STORAGE_SCHEMA_VERSION:
        raise ValueError("metadata의 full-prefix 저장 schema가 현재 계약과 다르다")
    expected_calibration_mode = (
        "none" if target_use in TARGET_FREE_USES else "validation_median_iqr"
    )
    if target_use == "fit_full_prefix" or official:
        expected_calibration_mode = model["preprocess_recipe"]["calibration"]
    if metadata.get("calibration_mode") != expected_calibration_mode:
        raise ValueError("metadata의 target_use와 calibration_mode가 다르다")
    if official:
        if (metadata.get("native_postprocessing") is not True
                or metadata.get("official_protocol") != "paper_tuning_v4"):
            raise ValueError("공식 점수의 native 후처리 또는 절차 표시가 빠졌다")
        if metadata.get("smoothing") != {"kind": "model_native", "window": 0, "boundary": "model_native"}:
            raise ValueError("공식 점수에 공통 smoothing이 다시 선언됐다")
    if (spec["common_recipe"].get("methodology_revision") == "source_faithful_v3"
            and spec["model"] in {"TimeRCD", "TSPulse"}):
        expected_scope = "valid_inference_block" if spec["model"] == "TimeRCD" else "past_context"
        if (metadata.get("input_normalization") != "inference_context_zscore"
                or metadata.get("input_normalization_scope") != expected_scope
                or metadata.get("persistent_target_fit") is not False):
            raise ValueError("Tier 3 metadata의 입력 정규화 범위가 registry와 다르다")
    if (
        len(evidence["test_sessions"]) != 1
        or evidence["test_sessions"][0]["observation_count"]
        != metadata.get("source_end_exclusive")
    ):
        raise ValueError(
            "execution_evidence test observation_count가 metadata source_end_exclusive와 다르다"
        )
    validation_session_lengths, validation_source_starts = _check_calibration_reference(
        output_directory, metadata, raw_name,
    )
    calibration_arguments = {}
    if target_use == "fit_full_prefix":
        calibration_arguments = {
            "target_use": target_use,
            "calibration_source": "fit" if expected_calibration_mode == "fit_median_iqr" else "none",
        }
    validate_training_evidence_bindings(
        evidence,
        ratio=spec["ratio"],
        validation_session_lengths=validation_session_lengths,
        validation_source_starts=validation_source_starts,
        **calibration_arguments,
    )

    file_prefix = "__".join(raw_name.split("__")[:6]) + "__"
    score_paths = tuple(output_directory.glob(file_prefix + "*.npy"))
    native_gdn = official and spec["model"] == "GDN"
    native_shape = metadata.get("native_channel_score_shape")
    if native_gdn:
        if (not isinstance(native_shape, list) or len(native_shape) != 2
                or any(type(size) is not int or size < 1 for size in native_shape)
                or native_shape[0] != metadata["score_length"]
                or metadata["score_shape"] != [metadata["score_length"]]):
            raise ValueError("GDN native_channel_score_shape가 scalar 범위와 다르다")
        input_column = metadata.get("effective_execution", {}).get("input_column")
        if input_column is not None and native_shape[1] != input_column:
            raise ValueError("GDN native 채널 수가 실제 입력 채널 수와 다르다")
    channel_scores = native_gdn or len(metadata["score_shape"]) == 2
    filename_arguments = {
        "dataset": metadata["dataset"], "series": metadata["series"],
        "model": metadata["model"], "tier": metadata["tier"],
        "ratio": metadata["ratio"], "seed": metadata["seed"],
        "norm_kind": "trainnorm",
    }
    expected_names = {
        build_score_filename(
            smoothing_kind=smoothing, channels=channels, **filename_arguments,
        )
        for smoothing in ("raw", "smoothed")
        for channels in ((False, True) if channel_scores else (False,))
    }
    actual_names = {path.name for path in score_paths}
    if actual_names != expected_names:
        raise ValueError(
            f"점수 파일명이 다르다: {sorted(actual_names)} != {sorted(expected_names)}"
        )
    saved_scores = {}
    for path in score_paths:
        scores = numpy.load(path, allow_pickle=False)
        expected_shape = (
            tuple(native_shape if native_gdn else metadata["score_shape"])
            if path.stem.endswith("__channels")
            else (metadata["score_length"],)
        )
        if scores.shape != expected_shape:
            raise ValueError(
                f"점수 배열 shape가 metadata와 다르다: {path.name} "
                f"{scores.shape} != {expected_shape}"
            )
        if not numpy.isfinite(scores).all():
            raise ValueError(f"점수 배열이 metadata 계약을 어겼다: {path.name}")
        if official:
            saved_scores[path.name] = scores
    if official:
        for name, scores in saved_scores.items():
            if "__raw__" in name and not numpy.array_equal(
                scores, saved_scores[name.replace("__raw__", "__smoothed__")],
            ):
                raise ValueError("공식 native 점수의 raw·smoothed 저장값이 다르다")
    if native_gdn:
        validate_gdn_native_channels(
            saved_scores[raw_name.replace(".npy", "__channels.npy")], saved_scores[raw_name],
            metadata.get("native_calibration"), source_start=metadata["source_start"],
            source_end_exclusive=metadata["source_end_exclusive"],
        )
    if official and spec["model"] == "TSPulse":
        validate_tspulse_native_calibration(metadata.get("native_calibration"),
            source_start=metadata["source_start"], source_end_exclusive=metadata["source_end_exclusive"],
            score_variant=score_variant, input_column=metadata.get("effective_execution", {}).get("input_column"))
    return {
        "status": "complete",
        "metadata_path": str(metadata_path),
        "score_file_count": len(score_paths),
    }
