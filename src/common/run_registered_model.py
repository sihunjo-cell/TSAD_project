"""봉인한 registry spec을 label-free 모델 호출로 연결한다."""

import importlib
import math
import time
from collections.abc import Mapping
from copy import deepcopy

import numpy
from sklearn.preprocessing import MinMaxScaler

from src.common.model_registry import (
    load_model_registry_with_sha,
    validate_execution_status,
)
from src.common.execution_identity import (
    EXECUTION_IDENTITY_FIELDS,
    validate_execution_identity,
)
from src.data_split.split_ratio_prefix import split_ratio_prefix


MODEL_ENTRYPOINTS = {
    "MWVAR": ("src.models.tier1.mwvar", "score_mwvar"),
    "SQDIFF_LAST3": ("src.models.tier1.sqdiff_last3", "score_sqdiff_last3"),
    "PCA_LEGACY": ("src.models.tier1.pca_legacy", "PcaLegacy"),
    "PaAno": ("src.models.tier2.paano.adapter", "PaAnoAdapter"),
    "ALoRa": ("src.models.tier2.alora.adapter", "run_alora_sessions"),
    "GDN": ("src.models.tier2.gdn_official.adapter", "run_gdn_sessions"),
    "TimeRCD": ("src.models.tier3.time_rcd", "score_time_rcd_official"),
    "TSPulse": ("src.models.tier3.tspulse", "score_tspulse_official"),
}
MODEL_PARAMETER_KEYS = {
    "MWVAR": {"window", "centered", "ddof"},
    "SQDIFF_LAST3": {"lag", "window", "correction"},
    "PCA_LEGACY": {"window", "zero_pruning", "n_components"},
    "PaAno": {
        "iterations", "batch_size", "weight_decay", "memory_fraction",
        "memory_seed", "neighbors", "use_revin", "patch_size", "learning_rate",
    },
    "ALoRa": {
        "layers", "heads", "learning_rate", "lambda_regularization",
        "rank_threshold", "max_pairs", "epochs", "batch_size", "patience",
        "dropout", "token_kernel_size", "optimizer", "weight_decay", "window",
    },
    "GDN": {
        "window", "epochs", "patience", "stride", "batch_size",
        "learning_rate", "weight_decay", "out_layer_num", "graph_heads",
        "graph_dropout", "output_dropout", "optimizer", "embedding", "hidden",
        "rho",
    },
    "TimeRCD": {"context_length", "checkpoint_variant", "score_head"},
    "TSPulse": {"context_length", "patch_size", "heads", "aggregation_window"},
}
TARGET_FREE = {"training_free", "strict_zero_shot"}
SESSION_RUNNERS = {"ALoRa", "GDN"}
SEALED_SPEC_FIELDS = (
    "tier", "target_use", "source_commit", "source_checkpoint_sha256",
    "checkpoint_config_sha256", "checkpoint_revision", "preprocess_recipe",
    "common_recipe", "common_recipe_id",
)


def set_reproducible_seed(seed: int) -> dict:
    """모델 실행 시점까지 PyTorch를 불러오지 않는다."""
    from src.common.set_reproducible_seed import set_reproducible_seed as set_seed

    return set_seed(seed)


def validate_registered_spec(spec: dict) -> None:
    """현재 ready registry와 정확히 일치하는 실행 spec만 허용한다."""
    if not isinstance(spec, Mapping):
        raise ValueError("registry spec은 mapping이어야 한다")
    validate_execution_identity(spec)
    registry, registry_sha = load_model_registry_with_sha()
    if spec.get("config_registry_sha256") != registry_sha:
        raise ValueError("registry SHA와 spec이 일치하지 않는다")
    model_name = spec.get("model")
    model = registry["models"].get(model_name)
    if model is None:
        raise ValueError(f"registry에 없는 model이다: {model_name!r}")
    if validate_execution_status(model_name, model) != "ready":
        raise ValueError(f"ready registry model이 아니다: {model_name}")
    candidate = next(
        (
            candidate
            for candidate in model["candidates"]
            if candidate["config_id"] == spec.get("config_id")
        ),
        None,
    )
    if candidate is None:
        raise ValueError(f"registry에 없는 config_id이다: {spec.get('config_id')!r}")
    if spec.get("hyperparameters") != candidate["hyperparameters"]:
        raise ValueError("registry candidate와 spec hyperparameters가 일치하지 않는다")

    expected = {
        "tier": model["tier"],
        "target_use": model["target_use"],
        "source_commit": model["source_commit"],
        "source_checkpoint_sha256": model["source_checkpoint_sha256"],
        "checkpoint_config_sha256": model.get("checkpoint_config_sha256"),
        "checkpoint_revision": model.get("checkpoint_revision"),
        "preprocess_recipe": model["preprocess_recipe"],
        "common_recipe": registry["common_recipe"],
        "common_recipe_id": registry["common_recipe_id"],
    }
    for field in SEALED_SPEC_FIELDS:
        if field not in spec or spec[field] != expected[field]:
            raise ValueError(f"registry와 spec identity가 다르다: {field}")


def load_model_entrypoint(model: str):
    """무거운 모델 package를 실제 호출 직전까지 import하지 않는다."""
    try:
        module_name, attribute = MODEL_ENTRYPOINTS[model]
    except KeyError as error:
        raise ValueError(f"활성 entrypoint가 없는 모델이다: {model}") from error
    return getattr(importlib.import_module(module_name), attribute)


def _require_fixed(parameters: dict, expected: dict) -> None:
    for name, value in expected.items():
        if parameters.get(name) != value:
            raise ValueError(f"지원하지 않는 고정값이다: {name}={parameters.get(name)!r}")


def _require_parameter_keys(model: str, parameters: dict) -> None:
    expected = MODEL_PARAMETER_KEYS[model]
    if parameters.keys() != expected:
        raise ValueError(
            f"지원하지 않는 파라미터 키다: "
            f"extra={sorted(parameters.keys() - expected)}, "
            f"missing={sorted(expected - parameters.keys())}"
        )


def build_entrypoint_arguments(spec: dict, *, device: str, channel_count: int) -> dict:
    """registry 이름을 각 공식 adapter의 인자 이름으로만 번역한다."""
    model = spec["model"]
    parameters = dict(spec["hyperparameters"])
    _require_parameter_keys(model, parameters)
    if model == "MWVAR":
        _require_fixed(parameters, {"window": 96, "centered": True, "ddof": 1})
        return {}
    if model == "SQDIFF_LAST3":
        _require_fixed(parameters, {
            "lag": 3, "window": 4, "correction": 1.3333333333333333,
        })
        return {}
    if model == "PCA_LEGACY":
        _require_fixed(parameters, {"window": 100, "zero_pruning": False})
        return {"n_components": parameters["n_components"]}
    if model == "PaAno":
        return {**parameters, "device": device}
    if model == "ALoRa":
        _require_fixed(parameters, {
            "rank_threshold": 0.01, "dropout": 0.0, "token_kernel_size": 3,
            "optimizer": "Adam", "weight_decay": 0.0,
        })
        return {
            "window_size": parameters["window"], "device": device,
            "epochs": parameters["epochs"], "batch_size": parameters["batch_size"],
            "learning_rate": parameters["learning_rate"],
            "low_rank_weight": parameters["lambda_regularization"],
            "pair_embedding_dimension": parameters["max_pairs"],
            "attention_heads": parameters["heads"],
            "encoder_layers": parameters["layers"],
            "patience": parameters["patience"],
        }
    if model == "GDN":
        _require_fixed(parameters, {
            "stride": 1, "weight_decay": 0.0, "out_layer_num": 1,
            "graph_heads": 1, "graph_dropout": 0.0, "output_dropout": 0.2,
            "optimizer": "Adam",
        })
        return {
            "embedding_dimension": parameters["embedding"],
            "hidden_dimension": parameters["hidden"], "rho": parameters["rho"],
            "device": device, "window_size": parameters["window"],
            "epochs": parameters["epochs"], "patience": parameters["patience"],
            "batch_size": parameters["batch_size"],
            "learning_rate": parameters["learning_rate"],
        }
    if model == "TimeRCD":
        _require_fixed(parameters, {
            "checkpoint_variant": "multi", "score_head": "probability",
        })
        return {"context_length": parameters["context_length"], "device": device}
    if model == "TSPulse":
        _require_fixed(parameters, {
            "patch_size": 8, "heads": ["time", "fft", "pred", "raw_max"],
        })
        return {
            "aggregation_window": parameters["aggregation_window"],
            "context_length": parameters["context_length"], "device": device,
        }
    raise ValueError(f"활성 registry에 없는 모델이다: {model}")


def _as_sessions(name: str, input_sessions) -> tuple[numpy.ndarray, ...]:
    sessions = tuple(numpy.asarray(values) for values in input_sessions)
    if not sessions:
        raise ValueError(f"{name}는 비어 있지 않아야 한다")
    for values in sessions:
        if values.ndim != 2 or not len(values) or values.shape[1] == 0:
            raise ValueError(f"{name}는 비어 있지 않은 2차원 배열이어야 한다")
        if not numpy.issubdtype(values.dtype, numpy.number) or not numpy.isfinite(values).all():
            raise ValueError(f"{name}에는 유한한 수치만 있어야 한다")
    channel_count = sessions[0].shape[1]
    if any(values.shape[1] != channel_count for values in sessions):
        raise ValueError(f"{name}의 채널 수가 다르다")
    return sessions


def _as_test_sessions(test_sessions) -> tuple[numpy.ndarray, ...]:
    return _as_sessions("test_sessions", test_sessions)


def _scaler_state(scaler: MinMaxScaler) -> dict:
    return {
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
        "scale": scaler.scale_.tolist(),
        "offset": scaler.min_.tolist(),
        "sample_count": int(scaler.n_samples_seen_),
    }


def prepare_session_inputs(
    *,
    test_sessions,
    ratio_percent: int,
    scale: bool,
    normal_training=None,
    normal_training_sessions=None,
) -> dict:
    """세션마다 q-prefix를 나누고 필요할 때 fit 부분에만 scaler를 맞춘다."""
    if normal_training is not None and normal_training_sessions is not None:
        raise ValueError("normal_training과 normal_training_sessions를 함께 줄 수 없다")
    if normal_training is None and normal_training_sessions is None:
        raise ValueError("정상 training 입력이 없다")

    training = _as_sessions(
        "normal_training_sessions",
        (normal_training,) if normal_training_sessions is None else normal_training_sessions,
    )
    tests = _as_test_sessions(test_sessions)
    channel_count = training[0].shape[1]
    if any(values.shape[1] != channel_count for values in (*training, *tests)):
        raise ValueError("training과 test session의 채널 수가 다르다")

    split_results = tuple(
        split_ratio_prefix(values, ratio_percent) for values in training
    )
    fit_sessions = tuple(result[0] for result in split_results)
    validation_sessions = tuple(result[1] for result in split_results)
    scaler_state = None
    if scale:
        scaler = MinMaxScaler()
        for values in fit_sessions:
            scaler.partial_fit(values)
        fit_sessions = tuple(scaler.transform(values) for values in fit_sessions)
        validation_sessions = tuple(
            scaler.transform(values) for values in validation_sessions
        )
        tests = tuple(scaler.transform(values) for values in tests)
        scaler_state = _scaler_state(scaler)

    return {
        "fit_sessions": fit_sessions,
        "validation_sessions": validation_sessions,
        "test_sessions": tests,
        "session_splits": tuple(result[2] for result in split_results),
        "scaler_state": scaler_state,
    }


def _result_identity(spec: dict) -> dict:
    fields = (
        "model", "config_id", "config_registry_sha256", "source_commit",
        "source_checkpoint_sha256", "checkpoint_config_sha256",
        "checkpoint_revision", "preprocess_recipe", "common_recipe",
        "common_recipe_id", "target_use", *EXECUTION_IDENTITY_FIELDS,
    )
    return {field: deepcopy(spec.get(field)) for field in fields}


def execute_registered_model(
    spec: dict, *, normal_training=None, normal_training_sessions=None,
    test_sessions, device: str, entrypoint=None,
) -> dict:
    """한 registry spec을 현재 q-prefix와 label-free test session에 실행한다."""
    validate_registered_spec(spec)
    seed_state = set_reproducible_seed(spec["seed"])
    model = spec["model"]
    target_use = spec["target_use"]
    tests = _as_test_sessions(test_sessions)
    arguments = build_entrypoint_arguments(
        spec, device=device, channel_count=tests[0].shape[1],
    )
    timing = {
        "split_preprocess_seconds": 0.0,
        "training_seconds": 0.0,
        "validation_inference_seconds": 0.0,
        "test_inference_seconds": 0.0,
        "accelerator": device,
    }
    result = {
        **_result_identity(spec),
        "split": None,
        "scaler_state": None,
        "validation_outputs": (),
        "test_outputs": (),
        "checkpoint": None,
        "training_log": None,
        "timing": timing,
        "seed_state": seed_state,
        "calibration_scope": (
            "none" if target_use in TARGET_FREE else "current_prefix_validation"
        ),
    }

    if target_use in TARGET_FREE:
        if spec["ratio"] != 100:
            raise ValueError("target-free 모델의 실행 ratio는 100이어야 한다")
        if normal_training is not None and normal_training_sessions is not None:
            raise ValueError("normal_training과 normal_training_sessions를 함께 줄 수 없다")
        entrypoint = entrypoint or load_model_entrypoint(model)
        started = time.perf_counter()
        result["test_outputs"] = tuple(
            entrypoint(values, **arguments) for values in tests
        )
        timing["test_inference_seconds"] = time.perf_counter() - started
        return result

    started = time.perf_counter()
    prepared = prepare_session_inputs(
        normal_training=normal_training,
        normal_training_sessions=normal_training_sessions,
        test_sessions=tests,
        ratio_percent=spec["ratio"],
        scale=spec["tier"] == "t2",
    )
    fit_sessions = prepared["fit_sessions"]
    validation_sessions = prepared["validation_sessions"]
    tests = prepared["test_sessions"]
    if len(fit_sessions) > 1 and model not in SESSION_RUNNERS:
        raise ValueError(f"{model} entrypoint는 여러 training session을 받지 않는다")
    result["scaler_state"] = prepared["scaler_state"]
    splits = prepared["session_splits"]
    result["split"] = splits[0] if len(splits) == 1 else splits
    timing["split_preprocess_seconds"] = time.perf_counter() - started
    entrypoint = entrypoint or load_model_entrypoint(model)

    if model in SESSION_RUNNERS:
        run_result = entrypoint(
            fit_sessions, validation_sessions, tests, **arguments,
        )
        for field in (
            "checkpoint", "validation_outputs", "test_outputs", "training_log",
        ):
            result[field] = run_result.get(field)
        required_timing = (
            "training_seconds", "validation_inference_seconds", "test_inference_seconds",
        )
        runner_timing = run_result.get("timing")
        if not isinstance(runner_timing, Mapping):
            raise ValueError("session runner timing은 mapping이어야 한다")
        missing_timing = set(required_timing) - runner_timing.keys()
        if missing_timing:
            raise ValueError(f"session runner timing 필드가 빠졌다: {sorted(missing_timing)}")
        for field in required_timing:
            value = runner_timing[field]
            if type(value) not in (int, float):
                raise ValueError(f"session runner timing.{field}은 유한한 0 이상의 수여야 한다")
            try:
                normalized = float(value)
            except OverflowError as error:
                raise ValueError(
                    f"session runner timing.{field}은 유한한 0 이상의 수여야 한다"
                ) from error
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError(f"session runner timing.{field}은 유한한 0 이상의 수여야 한다")
            timing[field] = normalized
        return result

    fit_values = fit_sessions[0]
    validation_values = validation_sessions[0]
    adapter = entrypoint(**arguments)
    training_started = time.perf_counter()
    if model == "PCA_LEGACY":
        adapter.fit(fit_values)
    else:
        result["training_log"] = adapter.fit(fit_values)
    timing["training_seconds"] = time.perf_counter() - training_started
    checkpoint = getattr(adapter, "checkpoint", None)
    result["checkpoint"] = checkpoint() if callable(checkpoint) else None

    validation_started = time.perf_counter()
    result["validation_outputs"] = (adapter.score(validation_values),)
    timing["validation_inference_seconds"] = (
        time.perf_counter() - validation_started
    )
    test_started = time.perf_counter()
    result["test_outputs"] = tuple(adapter.score(values) for values in tests)
    timing["test_inference_seconds"] = time.perf_counter() - test_started
    return result
