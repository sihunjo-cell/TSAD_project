"""봉인한 registry spec을 label-free 모델 호출로 연결한다."""

import importlib
import math
import time
from collections.abc import Callable, Mapping
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
    "SQDIFF_LAST1": ("src.models.tier1.sqdiff", "score_sqdiff_last1"),
    "SQDIFF_CENTERED5": ("src.models.tier1.sqdiff", "score_sqdiff_centered5"),
    "MWVAR96_SQDIFF_LAST3": ("src.models.tier1.one_liner_ensemble", "score_mwvar96_sqdiff_last3"),
    "MWVAR96_SQDIFF_CENTERED5": ("src.models.tier1.one_liner_ensemble", "score_mwvar96_sqdiff_centered5"),
    "PCA_LEGACY": ("src.models.tier1.pca_legacy", "PcaLegacy"),
    "PaAno": ("src.models.tier2.paano.adapter", "PaAnoAdapter"),
    "GDN": ("src.models.tier2.gdn_official.adapter", "run_gdn_sessions"),
    "TimeRCD": ("src.models.tier3.time_rcd", "score_time_rcd_official"),
    "TSPulse": ("src.models.tier3.tspulse", "score_tspulse_official"),
}
TARGET_FREE_SETUP_ENTRYPOINTS = {
    "TimeRCD": ("src.models.tier3.time_rcd", "build_time_rcd_official_scorer"),
    "TSPulse": ("src.models.tier3.tspulse", "build_tspulse_official_scorer"),
}
TSPULSE_INFERENCE_BATCH_SIZE = 32
MODEL_PARAMETER_KEYS = {
    "MWVAR": {"window", "centered", "ddof"},
    "SQDIFF_LAST3": {"lag", "window", "correction"},
    "SQDIFF_LAST1": {"lag", "window", "correction"},
    "SQDIFF_CENTERED5": {"window", "correction", "centered"},
    "MWVAR96_SQDIFF_LAST3": {"variance_window", "difference_window", "difference_centered", "difference_correction"},
    "MWVAR96_SQDIFF_CENTERED5": {"variance_window", "difference_window", "difference_centered", "difference_correction"},
    "PCA_LEGACY": {"window", "zero_pruning", "n_components"},
    "PaAno": {
        "iterations", "batch_size", "weight_decay", "memory_fraction",
        "memory_seed", "neighbors", "use_revin", "patch_size", "learning_rate",
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
SESSION_RUNNERS = {"GDN"}
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


def load_target_free_setup_entrypoint(model: str) -> Callable | None:
    """준비 단계를 제공하는 target-free scorer builder를 늦게 불러온다."""
    target = TARGET_FREE_SETUP_ENTRYPOINTS.get(model)
    if target is None:
        return None
    module_name, attribute = target
    return getattr(importlib.import_module(module_name), attribute)


def _synchronize_cuda(device: str) -> None:
    if str(device).split(":", 1)[0] == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _require_fixed(parameters: dict, expected: dict) -> None:
    for name, value in expected.items():
        if parameters.get(name) != value:
            raise ValueError(f"지원하지 않는 고정값이다: {name}={parameters.get(name)!r}")


def _require_parameter_keys(model: str, parameters: dict, *, official: bool = False) -> None:
    expected = MODEL_PARAMETER_KEYS[model]
    if official and model == "GDN":
        expected = expected | {"validation_ratio", "optimizer_betas"}
    if model == "GDN" and "topk" in parameters:
        expected = (expected - {"rho"}) | {"topk"}
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
    official = spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"
    _require_parameter_keys(model, parameters, official=official)
    if model == "MWVAR":
        _require_fixed(parameters, {"centered": True, "ddof": 1})
        return {"window": parameters["window"]}
    if model == "SQDIFF_LAST3":
        _require_fixed(parameters, {
            "lag": 3, "window": 4, "correction": 1.3333333333333333,
        })
        return {}
    if model == "SQDIFF_LAST1":
        _require_fixed(parameters, {"lag": 1, "window": 2, "correction": 2})
        return {}
    if model == "SQDIFF_CENTERED5":
        _require_fixed(parameters, {"window": 5, "correction": 1.25, "centered": True})
        return {}
    if model in {"MWVAR96_SQDIFF_LAST3", "MWVAR96_SQDIFF_CENTERED5"}:
        centered = model == "MWVAR96_SQDIFF_CENTERED5"
        _require_fixed(parameters, {"variance_window": 96,
                                   "difference_window": 5 if centered else 4,
                                   "difference_centered": centered,
                                   "difference_correction": 1.25 if centered else 1.3333333333333333})
        return {}
    if model == "PCA_LEGACY":
        _require_fixed(parameters, {"window": 100, "zero_pruning": official})
        return {"n_components": parameters["n_components"],
                **({"zero_pruning": True} if official else {})}
    if model == "PaAno":
        return {
            **parameters, "device": device,
            **({"official_procedure": True} if official else {}),
            **({"full_prefix": True, "memory_policy": "official_minimum"}
               if spec["target_use"] == "fit_full_prefix" else {}),
        }
    if model == "GDN":
        _require_fixed(parameters, {
            "stride": 1, "weight_decay": 0.0, "out_layer_num": 1,
            "graph_heads": 1, "graph_dropout": 0.0, "output_dropout": 0.2,
            "optimizer": "Adam",
        })
        return {
            "embedding_dimension": parameters["embedding"],
            "hidden_dimension": parameters["hidden"],
            **({"fixed_topk": parameters["topk"]} if "topk" in parameters
               else {"rho": parameters["rho"]}),
            "device": device, "window_size": parameters["window"],
            "epochs": parameters["epochs"], "patience": parameters["patience"],
            "batch_size": parameters["batch_size"],
            "learning_rate": parameters["learning_rate"],
            **({"official_procedure": True,
                "validation_ratio": parameters["validation_ratio"],
                "optimizer_betas": tuple(parameters["optimizer_betas"]),
                "split_seed": spec["seed"]} if official else {}),
            **({"full_prefix": True} if spec["target_use"] == "fit_full_prefix" else {}),
        }
    if model == "TimeRCD":
        _require_fixed(parameters, {
            "checkpoint_variant": "multi", "score_head": "probability",
        })
        return {
            "context_length": parameters["context_length"], "device": device,
            **({"official_protocol": True} if official else {}),
            **({"inference_context_normalization": True}
               if spec["common_recipe"].get("methodology_revision") == "source_faithful_v3" else {}),
        }
    if model == "TSPulse":
        _require_fixed(parameters, {
            "patch_size": 8, "heads": ["time", "fft", "pred", "ensemble" if official else "raw_max"],
        })
        return {
            "aggregation_window": parameters["aggregation_window"],
            "context_length": parameters["context_length"],
            "batch_size": 128 if official else TSPULSE_INFERENCE_BATCH_SIZE, "device": device,
            **({"official_protocol": True} if official else {}),
            **({"inference_context_normalization": True}
               if spec["common_recipe"].get("methodology_revision") == "source_faithful_v3" else {}),
        }
    raise ValueError(f"활성 registry에 없는 모델이다: {model}")


def build_registered_execution_policy(spec: dict) -> dict:
    """config ID 밖에서 고정한 등록 실행 일정을 산출물에 묶는다."""
    parameters = spec.get("hyperparameters", {})
    if spec.get("model") == "TimeRCD":
        from src.models.tier3.time_rcd import TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE

        return {
            "context_length": parameters["context_length"],
            "attention_query_chunk_size": TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE,
        }
    if spec.get("model") == "TSPulse":
        return {
            "context_length": parameters["context_length"],
            "aggregation_window": parameters["aggregation_window"],
            "batch_size": (128 if spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"
                           else TSPULSE_INFERENCE_BATCH_SIZE),
        }
    return {}


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


def select_input_scaler(spec: dict) -> str:
    if spec["tier"] != "t2":
        return "none"
    if spec.get("common_recipe", {}).get("methodology_revision") in {"source_faithful_v3", "paper_tuning_v4"}:
        return {"PaAno": "none", "GDN": "minmax"}[spec["model"]]
    return "minmax"


def _scaler_state(scaler) -> dict:
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
    full_prefix: bool = False,
    scaler_kind: str = "minmax",
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
        split_ratio_prefix(values, ratio_percent, full_prefix=full_prefix) for values in training
    )
    if full_prefix:
        for values, (fit_values, validation_values, split) in zip(training, split_results):
            observed_row = len(values) * ratio_percent // 100
            if (len(fit_values) != observed_row or len(validation_values)
                    or split["fit_range"] != (0, observed_row)
                    or split["validation_range"] != (observed_row, observed_row)
                    or split["normal_range"] != (0, len(values))
                    or fit_values.ctypes.data != values.ctypes.data
                    or fit_values.strides != values.strides):
                raise ValueError("full-prefix 학습은 현재 prefix 전부만 사용해야 한다")
    fit_sessions = tuple(result[0] for result in split_results)
    validation_sessions = () if full_prefix else tuple(result[1] for result in split_results)
    scaler_state = None
    if scale:
        if scaler_kind != "minmax":
            raise ValueError("지원하지 않는 입력 scaler이다")
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


def _finish_execution_result(result: dict, spec: dict, arguments: dict, device: str) -> dict:
    """실제 호출 인자와 backend를 남기고 없는 validation 비용은 저장하지 않는다."""
    current_storage = (spec["target_use"] == "fit_full_prefix"
                       or spec.get("common_recipe", {}).get("training_split") == "full_prefix_v2")
    if current_storage:
        if (result["timing"].get("validation_inference_seconds", 0) != 0
                or len(result.get("validation_outputs") or ())):
            raise ValueError("full-prefix 실행에서 validation 결과가 생성됐다")
        result["timing"].pop("validation_inference_seconds", None)
        result["effective_execution"] = {
            "backend": arguments.get("device", "cpu"), "requested_device": device,
            "entrypoint_arguments": deepcopy(arguments),
            "input_column": result["input_column"],
            "input_scaler": select_input_scaler(spec),
        }
        result["timing"]["accelerator"] = arguments.get("device", "cpu")
    result.pop("input_column", None)
    return result


def execute_registered_model(
    spec: dict, *, normal_training=None, normal_training_sessions=None,
    test_sessions, device: str, entrypoint=None,
    on_training_complete: Callable[[dict], None] | None = None,
) -> dict:
    """한 registry spec을 현재 q-prefix와 label-free test session에 실행한다."""
    validate_registered_spec(spec)
    seed_state = set_reproducible_seed(spec["seed"])
    model = spec["model"]
    target_use = spec["target_use"]
    official = spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"
    full_prefix = target_use == "fit_full_prefix"
    tests = _as_test_sessions(test_sessions)
    arguments = build_entrypoint_arguments(
        spec, device=device, channel_count=tests[0].shape[1],
    )
    timing = {
        "split_preprocess_seconds": 0.0,
        "model_setup_seconds": 0.0,
        "training_seconds": 0.0,
        "validation_inference_seconds": 0.0,
        "test_inference_seconds": 0.0,
        "accelerator": device,
    }
    if full_prefix or spec.get("common_recipe", {}).get("input_dispatch") == "registered_executor_full_prefix_v2":
        timing["calibration_inference_seconds"] = 0.0
    result = {
        **_result_identity(spec),
        "input_column": tests[0].shape[1],
        "split": None,
        "scaler_state": None,
        "validation_outputs": (),
        "calibration_outputs": (),
        "test_outputs": (),
        "checkpoint": None,
        "training_log": None,
        "timing": timing,
        "seed_state": seed_state,
        "calibration_scope": (
            "full_evaluation" if official and model in {"GDN", "TSPulse"} else
            ("current_prefix_fit" if model == "GDN" else "none") if full_prefix
            else "none" if target_use in TARGET_FREE else "current_prefix_validation"
        ),
    }

    if target_use in TARGET_FREE:
        if spec["ratio"] != 100:
            raise ValueError("target-free 모델의 실행 ratio는 100이어야 한다")
        if normal_training is not None and normal_training_sessions is not None:
            raise ValueError("normal_training과 normal_training_sessions를 함께 줄 수 없다")
        scorer = entrypoint
        score_arguments = arguments
        if scorer is None:
            setup_started = time.perf_counter()
            setup_entrypoint = load_target_free_setup_entrypoint(model)
            if setup_entrypoint is not None:
                scorer = setup_entrypoint(
                    channel_count=tests[0].shape[1], **arguments,
                )
                score_arguments = {}
            else:
                scorer = (getattr(importlib.import_module("src.models.tier1.pca_legacy"), "score_pca_official")
                          if official and model == "PCA_LEGACY" else load_model_entrypoint(model))
            _synchronize_cuda(device)
            timing["model_setup_seconds"] = time.perf_counter() - setup_started
        inference_started = time.perf_counter()
        result["test_outputs"] = tuple(
            scorer(values, **score_arguments)
            for values in tests
        )
        if official and model == "PCA_LEGACY":
            result["checkpoint"] = {
                "fit_source": "full_evaluation",
                "sessions": [output.pop("checkpoint") for output in result["test_outputs"]],
            }
        if official and spec["tier"] == "t1":
            for output in result["test_outputs"]:
                output["native_postprocessing"] = True
                output["official_protocol"] = "paper_tuning_v4"
        _synchronize_cuda(device)
        timing["test_inference_seconds"] = time.perf_counter() - inference_started
        return _finish_execution_result(result, spec, arguments, device)

    started = time.perf_counter()
    prepared = prepare_session_inputs(
        normal_training=normal_training,
        normal_training_sessions=normal_training_sessions,
        test_sessions=tests,
        ratio_percent=spec["ratio"],
        scale=select_input_scaler(spec) != "none",
        scaler_kind=select_input_scaler(spec),
        full_prefix=full_prefix,
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

    if model in SESSION_RUNNERS:
        if entrypoint is None:
            setup_started = time.perf_counter()
            entrypoint = load_model_entrypoint(model)
            _synchronize_cuda(device)
            timing["model_setup_seconds"] = time.perf_counter() - setup_started
        training_callback = {}
        if on_training_complete is not None:
            def report_training_complete(training_result):
                partial = {
                    **result,
                    "checkpoint": training_result["checkpoint"],
                    "training_log": training_result["training_log"],
                    "timing": {
                        **timing, **training_result["timing"],
                        "model_setup_seconds": (
                            timing["model_setup_seconds"]
                            + training_result["timing"]["model_setup_seconds"]
                        ),
                    },
                }
                if training_result.get("training_protocol") is not None:
                    partial["training_protocol"] = training_result["training_protocol"]
                on_training_complete(_finish_execution_result(partial, spec, arguments, device))

            training_callback["on_training_complete"] = report_training_complete
        run_result = entrypoint(
            fit_sessions, validation_sessions, tests, **arguments, **training_callback,
        )
        for field in (
            "checkpoint", "validation_outputs", "test_outputs", "training_log",
        ):
            result[field] = run_result.get(field)
        result["calibration_outputs"] = run_result.get("calibration_outputs", ())
        if run_result.get("training_protocol") is not None:
            result["training_protocol"] = run_result["training_protocol"]
        required_timing = (
            "model_setup_seconds", "training_seconds",
            "validation_inference_seconds", "test_inference_seconds",
        )
        if full_prefix:
            required_timing = (*required_timing, "calibration_inference_seconds")
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
            if field == "model_setup_seconds":
                timing[field] += normalized
            else:
                timing[field] = normalized
        return _finish_execution_result(result, spec, arguments, device)

    fit_values = fit_sessions[0]
    setup_started = time.perf_counter()
    entrypoint = entrypoint or load_model_entrypoint(model)
    adapter = entrypoint(**arguments)
    prepare_model = getattr(adapter, "prepare_model", None)
    if callable(prepare_model) and not (official and model == "PaAno"):
        prepare_model(fit_values.shape[1])
    _synchronize_cuda(device)
    timing["model_setup_seconds"] = time.perf_counter() - setup_started
    training_started = time.perf_counter()
    if model == "PCA_LEGACY":
        adapter.fit(fit_values)
    else:
        result["training_log"] = adapter.fit(fit_values)
    _synchronize_cuda(device)
    timing["training_seconds"] = time.perf_counter() - training_started
    checkpoint = getattr(adapter, "checkpoint", None)
    result["checkpoint"] = checkpoint() if callable(checkpoint) else None
    if on_training_complete is not None:
        on_training_complete(_finish_execution_result(
            {**result, "timing": dict(timing)}, spec, arguments, device,
        ))

    if not full_prefix:
        validation_started = time.perf_counter()
        result["validation_outputs"] = (adapter.score(validation_sessions[0]),)
        _synchronize_cuda(device)
        timing["validation_inference_seconds"] = time.perf_counter() - validation_started
    test_started = time.perf_counter()
    result["test_outputs"] = tuple(adapter.score(values) for values in tests)
    if full_prefix and not official:
        result["test_outputs"] = tuple(
            {**output, "calibration_mode": "none", "normalization_scope": "none"}
            for output in result["test_outputs"]
        )
    _synchronize_cuda(device)
    timing["test_inference_seconds"] = time.perf_counter() - test_started
    return _finish_execution_result(result, spec, arguments, device)
