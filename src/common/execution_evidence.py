"""후속 비용 최적화가 소비할 실행 원자료를 검증한다."""

import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy

from src.data_split.split_ratio_prefix import compute_prefix_counts


EXECUTION_PHASES = {
    "development_hpo", "final_retraining", "target_free_inference",
}
TARGET_USES = {"fit_validation", "fit_full_prefix", "training_free", "strict_zero_shot"}
TARGET_FREE_USES = {"training_free", "strict_zero_shot"}
DEV18_MEASUREMENT_PROTOCOL_ID = "dev18_registered_runner.v2"
LEGACY_FULL_PREFIX_MEASUREMENT_PROTOCOL_ID = "dev18_registered_runner.full_prefix_v2"
FULL_PREFIX_MEASUREMENT_PROTOCOL_ID = "dev18_registered_runner.full_prefix_v3"
FULL_PREFIX_STORAGE_SCHEMA_VERSION = "full_prefix_storage.v1"
TIMING_FIELDS = (
    "split_preprocess_seconds",
    "model_setup_seconds",
    "training_seconds",
    "validation_inference_seconds",
    "test_inference_seconds",
)
LEGACY_FULL_PREFIX_TIMING_FIELDS = (*TIMING_FIELDS, "calibration_inference_seconds")
FULL_PREFIX_TIMING_FIELDS = (
    *(field for field in TIMING_FIELDS if field != "validation_inference_seconds"),
    "calibration_inference_seconds",
)
EVIDENCE_FIELDS = {
    "measurement_protocol_id", "execution_phase", "status", "retry_count",
    "training_sessions", "test_sessions", "timing", "runtime_seconds",
    "peak_memory_mb", "model_artifact_bytes",
}
TRAINING_SESSION_FIELDS = {
    "available_count", "fit_count", "validation_count", "observation_count",
    "observed_duration_seconds", "duration_basis",
}
FULL_PREFIX_TRAINING_SESSION_FIELDS = {
    "training_boundary", "observed_row", "observed_duration_seconds", "duration_basis",
}
TEST_SESSION_FIELDS = {
    "observation_count", "observed_duration_seconds", "duration_basis",
}
DURATION_FIELDS = {"observed_duration_seconds", "duration_basis"}
STABLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _require_mapping(values, name: str) -> Mapping:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name}은 mapping이어야 한다")
    return values


def _require_exact_fields(values: Mapping, expected: set[str], name: str) -> None:
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected, key=repr)
        raise ValueError(f"{name} 필드가 다르다: missing={missing}, extra={extra}")


def _stable_id(value, name: str) -> str:
    if type(value) is not str or not STABLE_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{name}은 비어 있지 않은 안정 ID여야 한다")
    return value


def _nonnegative_int(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}은 0 이상의 정수여야 한다")
    return value


def _nonnegative_number(value, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name}은 유한한 0 이상의 수여야 한다")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{name}은 유한한 0 이상의 수여야 한다") from error
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name}은 유한한 0 이상의 수여야 한다")
    return normalized


def _normalize_duration(values: Mapping, name: str) -> dict:
    _require_mapping(values, name)
    basis = _stable_id(values["duration_basis"], f"{name}.duration_basis")
    seconds = values["observed_duration_seconds"]
    if seconds is None:
        if basis != "unavailable":
            raise ValueError(
                f"{name}의 알 수 없는 duration은 basis=unavailable이어야 한다"
            )
    else:
        seconds = _nonnegative_number(seconds, f"{name}.observed_duration_seconds")
        if basis == "unavailable":
            raise ValueError(
                f"{name}의 duration 값에는 unavailable basis를 쓸 수 없다"
            )
    return {
        "observed_duration_seconds": seconds,
        "duration_basis": basis,
    }


def _normalize_training_session(values: Mapping, index: int) -> dict:
    name = f"training_sessions[{index}]"
    values = _require_mapping(values, name)
    _require_exact_fields(values, TRAINING_SESSION_FIELDS, name)
    counts = {
        field: _nonnegative_int(values[field], f"{name}.{field}")
        for field in (
            "available_count", "fit_count", "validation_count",
            "observation_count",
        )
    }
    if counts["available_count"] != counts["fit_count"] + counts["validation_count"]:
        raise ValueError(f"{name}의 available_count는 fit_count+validation_count여야 한다")
    if counts["available_count"] > counts["observation_count"]:
        raise ValueError(f"{name}의 available_count가 observation_count보다 크다")
    return {**counts, **_normalize_duration(values, name)}


def _normalize_prefix_session(values: Mapping, index: int) -> dict:
    name = f"training_sessions[{index}]"
    values = _require_mapping(values, name)
    _require_exact_fields(values, FULL_PREFIX_TRAINING_SESSION_FIELDS, name)
    counts = {field: _nonnegative_int(values[field], f"{name}.{field}")
              for field in ("training_boundary", "observed_row")}
    if not 0 < counts["observed_row"] <= counts["training_boundary"]:
        raise ValueError(f"{name}의 observed_row는 정상 학습 경계 안의 양수여야 한다")
    return {**counts, **_normalize_duration(values, name)}


def _training_session_normalizer(protocol_id: str):
    return (_normalize_prefix_session if protocol_id == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID
            else _normalize_training_session)


def _timing_fields(protocol_id: str) -> tuple:
    if protocol_id == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID:
        return FULL_PREFIX_TIMING_FIELDS
    if protocol_id == LEGACY_FULL_PREFIX_MEASUREMENT_PROTOCOL_ID:
        return LEGACY_FULL_PREFIX_TIMING_FIELDS
    return TIMING_FIELDS


def _normalize_test_session(values: Mapping, index: int) -> dict:
    name = f"test_sessions[{index}]"
    values = _require_mapping(values, name)
    _require_exact_fields(values, TEST_SESSION_FIELDS, name)
    observation_count = _nonnegative_int(
        values["observation_count"], f"{name}.observation_count",
    )
    if observation_count == 0:
        raise ValueError(f"{name}.observation_count는 1 이상이어야 한다")
    return {
        "observation_count": observation_count,
        **_normalize_duration(values, name),
    }


def _normalize_session_list(values, name: str, normalizer) -> list[dict]:
    if type(values) is not list:
        raise ValueError(f"{name}은 ordered list여야 한다")
    return [normalizer(row, index) for index, row in enumerate(values)]


def _normalize_timing(values: Mapping, protocol_id: str = "") -> dict:
    values = _require_mapping(values, "timing")
    fields = _timing_fields(protocol_id)
    _require_exact_fields(values, set(fields), "timing")
    return {
        field: _nonnegative_number(values[field], f"timing.{field}")
        for field in fields
    }


def _sum_timing(timing: Mapping) -> float:
    try:
        total = math.fsum(timing.values())
    except (KeyError, OverflowError) as error:
        raise ValueError("다섯 timing 값의 합은 유한해야 한다") from error
    if not math.isfinite(total):
        raise ValueError("다섯 timing 값의 합은 유한해야 한다")
    return total


def derive_execution_phase(dataset_role: str, target_use: str) -> str:
    """봉인된 데이터 역할과 target 사용 규칙에서 실행 단계를 정한다."""
    if type(target_use) is not str or target_use not in TARGET_USES:
        raise ValueError(f"지원하지 않는 target_use이다: {target_use!r}")
    if dataset_role == "development":
        return "development_hpo"
    if dataset_role != "final":
        raise ValueError(f"지원하지 않는 dataset_role이다: {dataset_role!r}")
    return "target_free_inference" if target_use in TARGET_FREE_USES else "final_retraining"


def validate_execution_evidence(values: Mapping) -> dict:
    """완료 score에 붙일 증거만 plain JSON-safe 값으로 반환한다."""
    values = _require_mapping(values, "execution_evidence")
    protocol_id = _stable_id(values.get("measurement_protocol_id"), "measurement_protocol_id")
    current_storage = protocol_id == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID
    fields = EVIDENCE_FIELDS | {"storage_schema_version", "resource_usage"} if current_storage else EVIDENCE_FIELDS
    _require_exact_fields(values, fields, "execution_evidence")
    if current_storage and values["storage_schema_version"] != FULL_PREFIX_STORAGE_SCHEMA_VERSION:
        raise ValueError("full-prefix 저장 schema가 다르다")
    phase = values["execution_phase"]
    if type(phase) is not str or phase not in EXECUTION_PHASES:
        raise ValueError(f"지원하지 않는 execution_phase이다: {phase!r}")
    if values["status"] != "complete":
        raise ValueError(
            "score의 execution_evidence.status는 complete만 허용한다. "
            "실패·timeout·unavailable은 manifest status 행에 기록한다"
        )
    retry_count = _nonnegative_int(values["retry_count"], "retry_count")
    training_sessions = _normalize_session_list(
        values["training_sessions"], "training_sessions", _training_session_normalizer(protocol_id),
    )
    test_sessions = _normalize_session_list(
        values["test_sessions"], "test_sessions", _normalize_test_session,
    )
    if phase == "target_free_inference" and training_sessions:
        raise ValueError("target_free_inference에는 training session을 기록하지 않는다")
    if phase == "final_retraining" and not training_sessions:
        raise ValueError("final_retraining에는 training session이 필요하다")
    if not test_sessions:
        raise ValueError("complete score에는 test session이 필요하다")

    timing = _normalize_timing(values["timing"], protocol_id)
    if protocol_id == LEGACY_FULL_PREFIX_MEASUREMENT_PROTOCOL_ID:
        if any(session["fit_count"] != session["available_count"] or session["validation_count"] for session in training_sessions):
            raise ValueError("full-prefix 증거는 fit=available, validation=0이어야 한다")
        if timing["validation_inference_seconds"] != 0:
            raise ValueError("full-prefix 실행에는 validation 추론 시간이 없다")
    runtime_seconds = _nonnegative_number(values["runtime_seconds"], "runtime_seconds")
    timing_sum = _sum_timing(timing)
    if not math.isclose(runtime_seconds, timing_sum, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("runtime_seconds가 다섯 timing 값의 합과 다르다")
    result = {
        "measurement_protocol_id": protocol_id,
        "execution_phase": phase,
        "status": "complete",
        "retry_count": retry_count,
        "training_sessions": training_sessions,
        "test_sessions": test_sessions,
        "timing": timing,
        "runtime_seconds": timing_sum,
        "peak_memory_mb": None if current_storage and values["peak_memory_mb"] is None else _nonnegative_number(
            values["peak_memory_mb"], "peak_memory_mb",
        ),
        "model_artifact_bytes": _nonnegative_int(
            values["model_artifact_bytes"], "model_artifact_bytes",
        ),
    }
    if current_storage:
        usage = _require_mapping(values["resource_usage"], "resource_usage")
        if not usage:
            raise ValueError("resource_usage에는 측정 범위 또는 미측정 사유가 필요하다")
        try:
            json.dumps(usage, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("resource_usage는 유한한 JSON 값이나 null이어야 한다") from error
        result.update(storage_schema_version=FULL_PREFIX_STORAGE_SCHEMA_VERSION,
                      resource_usage=deepcopy(dict(usage)))
    return result


def validate_execution_evidence_for_run(
    values: Mapping, *, dataset_role: str, target_use: str,
) -> dict:
    """증거의 실행 단계와 학습 세션을 봉인된 실행 역할에 대조한다."""
    evidence = validate_execution_evidence(values)
    expected_phase = derive_execution_phase(dataset_role, target_use)
    if evidence["execution_phase"] != expected_phase:
        raise ValueError("execution_evidence.execution_phase가 실행 역할과 다르다")
    if target_use in TARGET_FREE_USES and evidence["training_sessions"]:
        raise ValueError("target-free 실행에는 training session을 기록하지 않는다")
    if target_use in {"fit_validation", "fit_full_prefix"} and not evidence["training_sessions"]:
        raise ValueError(f"{target_use} 실행에는 training session이 필요하다")
    if (evidence["measurement_protocol_id"] == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID
            and target_use == "fit_validation"):
        raise ValueError("새 full-prefix 저장에는 fit_validation을 사용할 수 없다")
    if target_use == "fit_full_prefix" and evidence["measurement_protocol_id"] not in {
        FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, LEGACY_FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
    }:
        raise ValueError("fit_full_prefix 실행에는 full-prefix 측정 프로토콜이 필요하다")
    return evidence


def validate_training_evidence_bindings(
    values: Mapping, *, ratio: int, validation_session_lengths,
    validation_source_starts=None,
    target_use: str = "fit_validation", calibration_source: str = "validation",
) -> None:
    """학습 증거 count를 봉인 ratio와 ordered validation 입력 길이에 묶는다."""
    values = _require_mapping(values, "execution_evidence")
    if "training_sessions" not in values:
        raise ValueError("execution_evidence.training_sessions가 빠졌다")
    sessions = _normalize_session_list(
        values["training_sessions"],
        "training_sessions",
        _training_session_normalizer(values.get("measurement_protocol_id", "")),
    )
    if type(validation_session_lengths) not in (list, tuple):
        raise ValueError("validation_session_lengths는 ordered sequence여야 한다")
    lengths = [
        _nonnegative_int(length, f"validation_session_lengths[{index}]")
        for index, length in enumerate(validation_session_lengths)
    ]
    if validation_source_starts is None:
        starts = [0] * len(lengths)
    elif type(validation_source_starts) in (list, tuple):
        starts = [
            _nonnegative_int(start, f"validation_source_starts[{index}]")
            for index, start in enumerate(validation_source_starts)
        ]
    else:
        raise ValueError("validation_source_starts는 ordered sequence여야 한다")
    if len(starts) != len(lengths):
        raise ValueError("validation score 길이와 source start 수가 다르다")
    full_prefix = target_use == "fit_full_prefix"
    if calibration_source not in ({"none", "fit"} if full_prefix else {"validation"}):
        raise ValueError("target_use와 calibration source가 다르다")
    if calibration_source == "none" and (lengths or starts):
        raise ValueError("calibration source=none에는 참조 점수가 없다")
    if calibration_source != "none" and len(sessions) != len(lengths):
        raise ValueError(
            "execution_evidence training session 수와 validation session 수가 다르다"
        )
    for index, session in enumerate(sessions):
        current_storage = values.get("measurement_protocol_id") == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID
        boundary = session["training_boundary" if current_storage else "observation_count"]
        expected = compute_prefix_counts(boundary, ratio, full_prefix=full_prefix)
        actual = ((session["observed_row"], session["observed_row"], 0) if current_storage
                  else tuple(session[field] for field in ("available_count", "fit_count", "validation_count")))
        if actual != expected:
            raise ValueError(
                f"execution_evidence training_sessions[{index}] count가 spec ratio와 다르다"
            )
        if calibration_source == "none":
            continue
        validation_length, source_start = lengths[index], starts[index]
        reference_count = session["observed_row" if current_storage else "fit_count" if full_prefix else "validation_count"]
        if source_start > reference_count:
            raise ValueError(
                f"execution_evidence training_sessions[{index}]의 validation source start가 "
                "validation 길이를 벗어났다"
            )
        if validation_length != reference_count - source_start:
            raise ValueError(
                f"execution_evidence training_sessions[{index}].validation_count가 "
                "validation score 정렬 범위와 다르다"
            )


def _range(values: Mapping, field: str, name: str) -> tuple[int, int]:
    if field not in values:
        raise ValueError(f"{name}.{field}가 빠졌다")
    bounds = values[field]
    if type(bounds) not in (list, tuple) or len(bounds) != 2:
        raise ValueError(f"{name}.{field}는 길이 2의 range여야 한다")
    start = _nonnegative_int(bounds[0], f"{name}.{field}[0]")
    end = _nonnegative_int(bounds[1], f"{name}.{field}[1]")
    if start > end:
        raise ValueError(f"{name}.{field}의 시작이 끝보다 크다")
    return start, end


def _training_counts(split: Mapping, index: int, expected_ratio: int, *, full_prefix: bool = False) -> dict:
    name = f"split[{index}]"
    split = _require_mapping(split, name)
    normal = _range(split, "normal_range", name)
    if normal[0] != 0:
        raise ValueError(f"{name}.normal_range는 0에서 시작해야 한다")
    if "ratio_percent" not in split:
        raise ValueError(f"{name}.ratio_percent가 빠졌다")
    if split["ratio_percent"] != expected_ratio:
        raise ValueError(f"{name}.ratio_percent가 spec.ratio와 다르다")
    available_count, fit_count, validation_count = compute_prefix_counts(
        normal[1], split["ratio_percent"], full_prefix=full_prefix,
    )
    expected_ranges = {
        "normal_range": (0, normal[1]),
        "available_range": (0, available_count),
        "fit_range": (0, fit_count),
        "validation_range": (fit_count, available_count),
        "unseen_range": (available_count, normal[1]),
    }
    for field, expected in expected_ranges.items():
        if _range(split, field, name) != expected:
            raise ValueError(f"{name}.{field}가 ratio prefix 계약과 다르다")
    return {
        "available_count": available_count,
        "fit_count": fit_count,
        "validation_count": validation_count,
        "observation_count": normal[1],
    }


def _test_observation_count(session, index: int) -> int:
    name = f"test_input_sessions[{index}]"
    shape = getattr(session, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise ValueError(f"{name}은 shape가 있는 2차원 session이어야 한다")
    observation_count = _nonnegative_int(shape[0], f"{name}.shape[0]")
    if observation_count == 0:
        raise ValueError(f"{name}은 비어 있지 않아야 한다")
    return observation_count


def build_execution_evidence(
    split,
    timing: Mapping,
    *,
    spec: Mapping,
    measurement_protocol_id: str,
    retry_count: int,
    training_session_durations: list,
    test_input_sessions,
    test_session_durations: list,
    peak_memory_mb,
    model_artifact_bytes: int,
    resource_usage: Mapping | None = None,
) -> dict:
    """등록 실행기의 split·timing을 완료 score 증거로 바꾼다."""
    spec = _require_mapping(spec, "spec")
    try:
        dataset_role = spec["dataset_role"]
        target_use = spec["target_use"]
        ratio = _nonnegative_int(spec["ratio"], "spec.ratio")
    except KeyError as error:
        raise ValueError(f"spec 필드가 빠졌다: {error.args[0]}") from error
    execution_phase = derive_execution_phase(dataset_role, target_use)
    if target_use in TARGET_FREE_USES and ratio != 100:
        raise ValueError("target-free 실행의 spec.ratio는 100이어야 한다")
    if split is None:
        splits = []
    elif isinstance(split, Mapping):
        splits = [split]
    elif type(split) in (list, tuple):
        splits = list(split)
    else:
        raise ValueError("split은 mapping, ordered sequence 또는 null이어야 한다")
    if type(training_session_durations) is not list:
        raise ValueError("training_session_durations는 ordered list여야 한다")
    if len(splits) != len(training_session_durations):
        raise ValueError("split과 training duration의 session 수가 다르다")

    training_sessions = []
    for index, (session_split, duration) in enumerate(
        zip(splits, training_session_durations)
    ):
        duration = _require_mapping(duration, f"training_session_durations[{index}]")
        _require_exact_fields(
            duration, DURATION_FIELDS, f"training_session_durations[{index}]",
        )
        counts = _training_counts(session_split, index, ratio, full_prefix=target_use == "fit_full_prefix")
        if measurement_protocol_id == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID:
            counts = {"training_boundary": counts["observation_count"], "observed_row": counts["available_count"]}
        training_sessions.append({
            **counts,
            **_normalize_duration(duration, f"training_session_durations[{index}]"),
        })

    if type(test_input_sessions) not in (list, tuple):
        raise ValueError("test_input_sessions는 ordered sequence여야 한다")
    if type(test_session_durations) is not list:
        raise ValueError("test_session_durations는 ordered list여야 한다")
    if len(test_input_sessions) != len(test_session_durations):
        raise ValueError("test input과 duration의 session 수가 다르다")
    test_sessions = []
    for index, (session, duration) in enumerate(
        zip(test_input_sessions, test_session_durations)
    ):
        duration = _require_mapping(duration, f"test_session_durations[{index}]")
        _require_exact_fields(
            duration, DURATION_FIELDS, f"test_session_durations[{index}]",
        )
        test_sessions.append({
            "observation_count": _test_observation_count(session, index),
            **_normalize_duration(duration, f"test_session_durations[{index}]"),
        })

    runner_timing = _require_mapping(timing, "runner timing")
    timing_fields = _timing_fields(measurement_protocol_id)
    if (measurement_protocol_id == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID
            and runner_timing.get("validation_inference_seconds", 0) != 0):
        raise ValueError("full-prefix 실행에는 validation 추론 시간이 없다")
    evidence_timing = {
        field: runner_timing[field]
        for field in timing_fields
        if field in runner_timing
    }
    normalized_timing = _normalize_timing(evidence_timing, measurement_protocol_id)
    evidence = {
        "measurement_protocol_id": measurement_protocol_id,
        "execution_phase": execution_phase,
        "status": "complete",
        "retry_count": retry_count,
        "training_sessions": training_sessions,
        "test_sessions": test_sessions,
        "timing": normalized_timing,
        "runtime_seconds": _sum_timing(normalized_timing),
        "peak_memory_mb": peak_memory_mb,
        "model_artifact_bytes": model_artifact_bytes,
    }
    if measurement_protocol_id == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID:
        evidence.update(
            storage_schema_version=FULL_PREFIX_STORAGE_SCHEMA_VERSION,
            resource_usage=dict(resource_usage) if resource_usage is not None
            else {"status": "unavailable", "reason": "not_recorded"},
        )
    return validate_execution_evidence_for_run(
        evidence, dataset_role=dataset_role, target_use=target_use,
    )
