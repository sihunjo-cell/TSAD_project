"""봉인한 TimeRCD·TSPulse checkpoint를 합성·Dev18 소입력으로 감사한다."""

import gc
import hashlib
import argparse
import json
import platform
import sys
import time
from contextlib import contextmanager, nullcontext
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.set_reproducible_seed import set_reproducible_seed
from src.common.equal_trial_budget import _canonical_bytes, registry_space_sha256
from src.common.execution_identity import file_sha256
from src.common.model_registry import load_model_registry_with_sha
from src.common.verify_run_context import verify_runtime_versions
from src.models.tier3 import time_rcd as time_rcd_adapter
from src.models.tier3 import tspulse as tspulse_adapter
from tests.ghl_main.run_registered_models import build_specs, load_registered_inputs


OUTPUT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
)
DEV18_OUTPUT_ROOT = REPOSITORY_ROOT / ".runtime" / "dev18_checkpoint_smoke"
ENVIRONMENT_PATH = REPOSITORY_ROOT / "configs" / "environment.yaml"
SYNTHETIC_SEED = 20260825
CHANNEL_COUNTS = (19, 86)
TIME_RCD_SMOKE_LENGTH = 160
TSPULSE_SMOKE_LENGTH = 1536
TSPULSE_SMOKE_AGGREGATION_WINDOW = 96
MODEL_DIRECTORIES = {"TimeRCD": "time_rcd", "TSPulse": "tspulse"}
DEV18_QUICK_SERIES = "03"
DEV18_QUICK_LENGTH = 1536
DEV18_BUDGET_PATH = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "snapshots"
    / "dev18_selection" / "dev18_budget_manifest.json"
)
FULL_PREFIX_BUDGET_PATH = DEV18_BUDGET_PATH.parent / "full_prefix_v2" / "budget.json"
FULL_PREFIX_OUTPUT_ROOT = OUTPUT_ROOT / "active_models" / "full_prefix_v2" / "checkpoints"


class CheckpointProbeError(RuntimeError):
    """검증한 파일 신원을 잃지 않고 model load·forward 오류를 전달한다."""

    def __init__(self, error, *, files):
        super().__init__(str(error))
        self.original_error = error
        self.files = files


def make_synthetic_wave(*, length, channel_count, seed=SYNTHETIC_SEED):
    """고정 seed의 다주기 파형을 만들며 외부 데이터는 받지 않는다."""
    if not isinstance(length, int) or length < 2:
        raise ValueError("length must be an integer greater than one")
    if not isinstance(channel_count, int) or channel_count < 1:
        raise ValueError("channel_count must be a positive integer")
    random = numpy.random.default_rng(seed + length + 1009 * channel_count)
    timestep = numpy.arange(length, dtype=numpy.float64)[:, None]
    channel = numpy.arange(channel_count, dtype=numpy.float64)[None, :]
    phase = random.uniform(-numpy.pi, numpy.pi, size=(1, channel_count))
    period = 37.0 + 3.0 * (channel % 17.0)
    values = (
        numpy.sin(2.0 * numpy.pi * timestep / period + phase)
        + 0.35 * numpy.cos(2.0 * numpy.pi * timestep / (period * 2.3) - phase)
        + 0.15 * numpy.sin(2.0 * numpy.pi * timestep / 211.0 + channel / 11.0)
        + 0.01 * random.standard_normal((length, channel_count))
    )
    return values.astype(numpy.float32)


def validate_score_pair(first, second, *, expected_length):
    """두 CPU score가 유한·비상수·동일 길이·결정적인지 검사한다."""
    first = numpy.asarray(first, dtype=numpy.float64)
    second = numpy.asarray(second, dtype=numpy.float64)
    expected_shape = (expected_length,)
    if first.shape != expected_shape or second.shape != expected_shape:
        raise ValueError(
            f"score length mismatch: {first.shape}, {second.shape} != {expected_shape}",
        )
    if not numpy.isfinite(first).all() or not numpy.isfinite(second).all():
        raise ValueError("score must contain only finite values")
    peak_to_peak = float(numpy.ptp(first))
    repeated_peak_to_peak = float(numpy.ptp(second))
    if min(peak_to_peak, repeated_peak_to_peak) <= 1e-12:
        raise ValueError("score is constant within 1e-12")
    if not numpy.allclose(first, second, rtol=1e-7, atol=1e-9):
        raise ValueError("CPU score is not deterministic within allclose tolerance")
    return {
        "shape": [expected_length],
        "finite": True,
        "peak_to_peak": peak_to_peak,
        "cpu_deterministic": True,
        "maximum_absolute_repeat_difference": float(
            numpy.max(numpy.abs(first - second)),
        ),
    }


@contextmanager
def forbid_project_scaler_fit(
    *, standard_scaler_class=None, minmax_scaler_class=None,
):
    """실행 중 project-wide StandardScaler·MinMaxScaler fit을 차단한다."""
    if standard_scaler_class is None or minmax_scaler_class is None:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        standard_scaler_class = standard_scaler_class or StandardScaler
        minmax_scaler_class = minmax_scaler_class or MinMaxScaler

    def reject_standard_scaler(*_arguments, **_keywords):
        raise RuntimeError("forbidden StandardScaler.fit call")

    def reject_minmax_scaler(*_arguments, **_keywords):
        raise RuntimeError("forbidden MinMaxScaler.fit call")

    with (
        patch.object(standard_scaler_class, "fit", reject_standard_scaler),
        patch.object(minmax_scaler_class, "fit", reject_minmax_scaler),
    ):
        yield


@contextmanager
def forbid_time_rcd_window_dataset(*, inference_module=None):
    """입력 전체 mean/std를 계산하는 공식 _WindowDataset 경로를 차단한다."""
    inference_module = inference_module or import_module("time_rcd._inference")

    class ForbiddenWindowDataset:
        def __init__(self, *_arguments, **_keywords):
            raise RuntimeError("forbidden TimeRCD _WindowDataset call")

    with patch.object(inference_module, "_WindowDataset", ForbiddenWindowDataset):
        yield


def collect_environment_identity():
    """고정 환경을 검증하고 실제 Python·torch build 정보를 반환한다."""
    environment = yaml.safe_load(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    versions = verify_runtime_versions(environment)
    torch_version = str(torch.__version__)
    build_suffix = torch_version.partition("+")[2] or None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: versions[name] for name in environment["packages"]
        },
        "sources": {
            name: versions[f"{name}_source"]
            for name in environment.get("sources", {})
        },
        "torch": {
            "version": torch_version,
            "build_suffix": build_suffix,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
    }


class _VerifiedFileRecorder:
    def __init__(self, downloader, verifier):
        self._downloader = downloader
        self._verifier = verifier
        self._requests = {}
        self._files = {}

    def download(self, *, repo_id, filename, revision):
        path = Path(self._downloader(
            repo_id=repo_id, filename=filename, revision=revision,
        )).absolute()
        self._requests[path] = {
            "repository": repo_id,
            "file": filename,
            "revision": revision,
        }
        return str(path)

    def verify(self, path, *, expected_sha256):
        path = Path(path).absolute()
        digest = self._verifier(path, expected_sha256=expected_sha256)
        if path not in self._requests:
            raise RuntimeError(f"verified file was not downloaded by this run: {path}")
        request = self._requests[path]
        self._files[request["file"]] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        return digest

    def identities(self, *, checkpoint_file, config_file):
        missing = {checkpoint_file, config_file} - self._files.keys()
        if missing:
            raise RuntimeError(f"verified file identity is missing: {sorted(missing)}")
        return {
            "checkpoint": dict(self._files[checkpoint_file]),
            "config": dict(self._files[config_file]),
        }

    def available_identities(self):
        identities = {}
        for file_name, identity in self._files.items():
            kind = "config" if Path(file_name).name == "config.json" else "checkpoint"
            identities[kind] = dict(identity)
        return identities


def _probe_time_rcd(
    *, channel_count, session,
    context_length=time_rcd_adapter.TIME_RCD_CONTEXT_LENGTH,
    downloader=None,
    inference_context_normalization=False,
    official_protocol=False,
):
    recorder = _VerifiedFileRecorder(
        downloader or _download_from_local_cache,
        time_rcd_adapter.verify_time_rcd_checkpoint,
    )
    try:
        with forbid_project_scaler_fit(), forbid_time_rcd_window_dataset():
            tester, model = time_rcd_adapter.load_time_rcd_components(
                channel_count=channel_count,
                device="cpu",
                hub_download=recorder.download,
                file_verifier=recorder.verify,
            )
            first = time_rcd_adapter.score_time_rcd(
                model, session, context_length=context_length,
                device="cpu",
                inference_context_normalization=inference_context_normalization,
                official_protocol=official_protocol,
            )["scores"].copy()
            second = time_rcd_adapter.score_time_rcd(
                model, session, context_length=context_length,
                device="cpu",
                inference_context_normalization=inference_context_normalization,
                official_protocol=official_protocol,
            )["scores"].copy()
    except Exception as error:
        raise CheckpointProbeError(
            error, files=recorder.available_identities(),
        ) from error
    del model, tester
    gc.collect()
    return {
        "score_pairs": {"score": (first, second)},
        "files": recorder.identities(
            checkpoint_file=time_rcd_adapter.TIME_RCD_CHECKPOINT_FILE,
            config_file=time_rcd_adapter.TIME_RCD_CONFIG_FILE,
        ),
        "forbidden_calls": {
            "standard_scaler_fit": False,
            "minmax_scaler_fit": False,
            "time_rcd_window_dataset": False,
        },
    }


def _probe_tspulse(
    *, channel_count, session,
    context_length=tspulse_adapter.TSPULSE_CONTEXT_LENGTH,
    aggregation_window=TSPULSE_SMOKE_AGGREGATION_WINDOW,
    downloader=None,
    inference_context_normalization=False,
    official_protocol=False,
):
    recorder = _VerifiedFileRecorder(
        downloader or _download_from_local_cache,
        tspulse_adapter.verify_tspulse_checkpoint,
    )
    try:
        with (nullcontext() if official_protocol else forbid_project_scaler_fit()):
            model, utility = tspulse_adapter.load_tspulse_components(
                aggregation_window=aggregation_window,
                channel_count=channel_count,
                device="cpu",
                hub_download=recorder.download,
                file_verifier=recorder.verify,
                **({"official_protocol": True} if official_protocol else {}),
            )
            raw_head_function = tspulse_adapter.build_tspulse_raw_head_function(
                utility,
                aggregation_window=aggregation_window,
                context_length=context_length,
                device="cpu",
                inference_context_normalization=inference_context_normalization,
                **({"batch_size": 128} if official_protocol else {}),
            )
            scorer = tspulse_adapter.score_tspulse_paper if official_protocol else tspulse_adapter.score_tspulse
            first = scorer(
                session,
                raw_head_function=raw_head_function,
                aggregation_window=aggregation_window,
                context_length=context_length,
                **({"utility": utility} if official_protocol else {}),
            )
            second = scorer(
                session,
                raw_head_function=raw_head_function,
                aggregation_window=aggregation_window,
                context_length=context_length,
                **({"utility": utility} if official_protocol else {}),
            )
    except Exception as error:
        raise CheckpointProbeError(
            error, files=recorder.available_identities(),
        ) from error
    files = recorder.identities(
        checkpoint_file=tspulse_adapter.TSPULSE_CHECKPOINT_FILE,
        config_file=tspulse_adapter.TSPULSE_CONFIG_FILE,
    )
    checkpoint_parent = Path(files["checkpoint"]["path"]).parent
    config_parent = Path(files["config"]["path"]).parent
    if checkpoint_parent != config_parent:
        raise RuntimeError("TSPulse verified files are not in one local snapshot")
    score_pairs = {
        head: (first[head]["scores"].copy(), second[head]["scores"].copy())
        for head in (("time", "fft", "pred", "ensemble") if official_protocol
                     else ("time", "fft", "pred", "raw_max"))
    }
    del model, utility, raw_head_function
    gc.collect()
    return {
        "score_pairs": score_pairs,
        "files": files,
        **({"native_preprocessing": "full_input_standardscaler_head_minmax_smoothing_ensemble"}
           if official_protocol else {"forbidden_calls": {
               "standard_scaler_fit": False, "minmax_scaler_fit": False,
           }}),
        "loader": {
            "from_pretrained_source": "verified_local_snapshot",
            "snapshot_directory": str(checkpoint_parent),
        },
    }


def _base_model_report(
    model, *, environment, seed_state, input_length, input_metadata=None,
):
    if model == "TimeRCD":
        source_commit = time_rcd_adapter.TIME_RCD_SOURCE_COMMIT
        repository = time_rcd_adapter.TIME_RCD_MODEL_NAME
        revision = time_rcd_adapter.TIME_RCD_CHECKPOINT_REVISION
        checkpoint_file = time_rcd_adapter.TIME_RCD_CHECKPOINT_FILE
        checkpoint_sha = time_rcd_adapter.TIME_RCD_CHECKPOINT_SHA256
        config_file = time_rcd_adapter.TIME_RCD_CONFIG_FILE
        config_sha = time_rcd_adapter.TIME_RCD_CONFIG_SHA256
    else:
        source_commit = tspulse_adapter.TSPULSE_SOURCE_COMMIT
        repository = tspulse_adapter.TSPULSE_MODEL_NAME
        revision = tspulse_adapter.TSPULSE_CHECKPOINT_REVISION
        checkpoint_file = tspulse_adapter.TSPULSE_CHECKPOINT_FILE
        checkpoint_sha = tspulse_adapter.TSPULSE_CHECKPOINT_SHA256
        config_file = tspulse_adapter.TSPULSE_CONFIG_FILE
        config_sha = tspulse_adapter.TSPULSE_CONFIG_SHA256
    return {
        "model": model,
        "status": "failed",
        "input": input_metadata or {
            "kind": "fixed_seed_synthetic_wave",
            "seed": SYNTHETIC_SEED,
            "length": input_length,
            "channel_counts": list(CHANNEL_COUNTS),
            "uses_labels": False,
            "uses_normal_training": False,
        },
        "source": {"commit": source_commit},
        "checkpoint": {
            "repository": repository,
            "revision": revision,
            "file": checkpoint_file,
            "expected_sha256": checkpoint_sha,
        },
        "config": {
            "repository": repository,
            "revision": revision,
            "file": config_file,
            "expected_sha256": config_sha,
        },
        "environment": environment,
        "seed_state": seed_state,
        "channels": {},
    }


def _format_error(error):
    return {"type": type(error).__name__, "message": str(error)}


def _failed_base_report(model, *, environment, seed_state, stage, error):
    input_length = (
        TIME_RCD_SMOKE_LENGTH if model == "TimeRCD" else TSPULSE_SMOKE_LENGTH
    )
    report = _base_model_report(
        model,
        environment=environment,
        seed_state=seed_state,
        input_length=input_length,
    )
    report.update({"stage": stage, "error": _format_error(error)})
    return report


def _merge_probe_identity(report, payload):
    files = payload.get("files", {})
    if set(files) != {"checkpoint", "config"}:
        raise ValueError("checkpoint or config verified identity is missing")
    _merge_verified_files(report, files)

    if report.get("official_protocol") and report["model"] == "TSPulse":
        if payload.get("native_preprocessing") != "full_input_standardscaler_head_minmax_smoothing_ensemble":
            raise ValueError("official TSPulse preprocessing evidence is incomplete")
        report["native_preprocessing"] = payload["native_preprocessing"]
    else:
        evidence = payload.get("forbidden_calls")
        required = {"standard_scaler_fit", "minmax_scaler_fit"}
        if report["model"] == "TimeRCD":
            required.add("time_rcd_window_dataset")
        if not isinstance(evidence, dict) or any(evidence.get(name) is not False for name in required):
            raise ValueError("forbidden preprocessing evidence is incomplete")
        if "forbidden_calls" in report and report["forbidden_calls"] != evidence:
            raise ValueError("forbidden preprocessing evidence changed across channels")
        report["forbidden_calls"] = dict(evidence)

    if report["model"] == "TSPulse":
        loader = payload.get("loader")
        if not isinstance(loader, dict) or loader.get("from_pretrained_source") != (
            "verified_local_snapshot"
        ):
            raise ValueError("TSPulse local snapshot loader evidence is missing")
        if "loader" in report and report["loader"] != loader:
            raise ValueError("TSPulse loader identity changed across channel counts")
        report["loader"] = dict(loader)


def _merge_verified_files(report, files):
    for name, details in files.items():
        if name not in {"checkpoint", "config"} or not isinstance(details, dict):
            raise ValueError(f"unexpected verified file identity: {name}")
        previous = {
            key: report[name][key]
            for key in ("path", "sha256", "bytes")
            if key in report[name]
        }
        if previous and previous != details:
            raise ValueError(f"{name} identity changed across channel counts")
        report[name].update(details)


def validate_model_report_identity(report):
    """통과 판정에 필요한 source·파일·환경 신원이 완전한지 검사한다."""
    for name in ("source", "checkpoint", "config", "environment"):
        if not isinstance(report.get(name), dict) or not report[name]:
            raise ValueError(f"{name} identity is missing")
    if not report["source"].get("commit"):
        raise ValueError("source commit identity is missing")
    for name in ("checkpoint", "config"):
        identity = report[name]
        for field in ("repository", "revision", "file", "expected_sha256", "sha256", "bytes"):
            if identity.get(field) in (None, ""):
                raise ValueError(f"{name} {field} identity is missing")
        if identity["sha256"] != identity["expected_sha256"]:
            raise ValueError(f"{name} SHA-256 does not match the sealed value")
        if not isinstance(identity["bytes"], int) or identity["bytes"] < 1:
            raise ValueError(f"{name} byte size identity is invalid")
    environment = report["environment"]
    if not environment.get("python") or not environment.get("packages"):
        raise ValueError("environment Python or package identity is missing")
    torch_identity = environment.get("torch")
    required_torch_fields = {
        "version", "build_suffix", "cuda_available", "cuda_version",
        "cudnn_version", "deterministic_algorithms", "cudnn_deterministic",
        "cudnn_benchmark",
    }
    if not isinstance(torch_identity, dict) or not required_torch_fields.issubset(torch_identity):
        raise ValueError("environment torch identity is missing")
    if torch_identity["deterministic_algorithms"] is not True:
        raise ValueError("environment deterministic algorithms are not enabled")
    if report.get("model") == "TSPulse" and report.get("loader", {}).get(
        "from_pretrained_source"
    ) != "verified_local_snapshot":
        raise ValueError("TSPulse local snapshot identity is missing")


def _run_model_checkpoint_smoke(
    model, *, environment, seed_state, input_length, expected_heads, probe,
    cases=None, input_metadata=None, model_identity=None,
):
    report = _base_model_report(
        model, environment=environment, seed_state=seed_state,
        input_length=input_length, input_metadata=input_metadata,
    )
    if model_identity:
        report.update(model_identity)
    started = time.perf_counter()
    cases = cases or tuple(
        (channel_count, make_synthetic_wave(
            length=input_length, channel_count=channel_count,
        ))
        for channel_count in CHANNEL_COUNTS
    )
    for channel_count, session in cases:
        channel_started = time.perf_counter()
        try:
            payload = probe(channel_count=channel_count, session=session)
            _merge_probe_identity(report, payload)
            score_pairs = payload.get("score_pairs")
            if not isinstance(score_pairs, dict) or set(score_pairs) != set(expected_heads):
                raise ValueError(
                    f"score heads mismatch: {sorted(score_pairs or {})} != "
                    f"{sorted(expected_heads)}",
                )
            summaries = {
                head: validate_score_pair(
                    *score_pairs[head], expected_length=input_length,
                )
                for head in expected_heads
            }
        except Exception as error:  # 보고서에 각 channel 오류를 보존한다.
            original_error = error
            identity_preservation_error = None
            if isinstance(error, CheckpointProbeError):
                original_error = error.original_error
                try:
                    _merge_verified_files(report, error.files)
                except Exception as preservation_error:
                    identity_preservation_error = _format_error(preservation_error)
            report["channels"][str(channel_count)] = {
                "status": "failed",
                "error": _format_error(original_error),
                "cpu_wall_time_seconds": time.perf_counter() - channel_started,
            }
            if identity_preservation_error:
                report["channels"][str(channel_count)][
                    "identity_preservation_error"
                ] = identity_preservation_error
        else:
            report["channels"][str(channel_count)] = {
                "status": "passed",
                "scores": summaries,
                "cpu_wall_time_seconds": time.perf_counter() - channel_started,
            }
    report["cpu_wall_time_seconds"] = time.perf_counter() - started
    report["status"] = (
        "passed" if all(
            result["status"] == "passed" for result in report["channels"].values()
        ) else "failed"
    )
    try:
        validate_model_report_identity(report)
    except ValueError as error:
        report["identity_error"] = _format_error(error)
        report["status"] = "failed"
    return report


def run_time_rcd_checkpoint_smoke(
    *, environment, seed_state, probe=None,
):
    return _run_model_checkpoint_smoke(
        "TimeRCD",
        environment=environment,
        seed_state=seed_state,
        input_length=TIME_RCD_SMOKE_LENGTH,
        expected_heads=("score",),
        probe=probe or _probe_time_rcd,
    )


def run_tspulse_checkpoint_smoke(
    *, environment, seed_state, probe=None,
):
    return _run_model_checkpoint_smoke(
        "TSPulse",
        environment=environment,
        seed_state=seed_state,
        input_length=TSPULSE_SMOKE_LENGTH,
        expected_heads=("time", "fft", "pred", "raw_max"),
        probe=probe or _probe_tspulse,
    )


def _dev18_budget_specs(budget_path=None):
    registry, registry_sha = load_model_registry_with_sha()
    if budget_path is None:
        budget_path = FULL_PREFIX_BUDGET_PATH if registry["common_recipe"].get("input_dispatch") == "registered_executor_full_prefix_v2" else DEV18_BUDGET_PATH
    artifact = json.loads(Path(budget_path).read_text(encoding="utf-8"))
    budget = artifact.get("budget", artifact)
    full_prefix = budget.get("experiment_mode") == "full_prefix_v2"
    if full_prefix:
        core = {key: value for key, value in budget.items() if key not in {"budget_id", "budget_sha256"}}
        digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
        if budget.get("budget_sha256") != digest or budget.get("budget_id") != "b" + digest[:12]:
            raise ValueError("full-prefix budget digest가 다르다")
        if artifact.get("attestation", {}).get("config_registry_sha256") != registry_sha:
            raise ValueError("full-prefix budget registry SHA가 다르다")
    elif budget.get("budget_id") != registry["selection"].get("budget_id"):
        raise ValueError("Dev18 budget ID가 registry와 다르다")
    if budget.get("registry_space_sha256") != registry_space_sha256(registry):
        raise ValueError("Dev18 budget 후보 공간이 registry와 다르다")

    specs = build_specs("development", include_pending=True)
    selected = {}
    for model in MODEL_DIRECTORIES:
        rows = [row for row in budget["execution_panel"] if row["model"] == model]
        eligible = [row for row in rows if not full_prefix or DEV18_QUICK_SERIES in row.get("series_ids", [])]
        if not eligible or (not full_prefix and len(rows) != 1):
            raise ValueError(f"Dev18 quick smoke의 {model} budget 행이 유일하지 않다")
        for row in rows:
            matches = [
                spec for spec in specs
                if spec["model"] == model
                and spec["config_id"] == row["config_id"]
                and spec["ratio"] == row["physical_ratio"]
                and spec["seed"] == row["seed"]
            ]
            if len(matches) != 1:
                raise ValueError(f"Dev18 quick smoke의 {model} spec이 유일하지 않다")
            if row in eligible:
                selected.setdefault(model, matches[0])
    return budget, registry_sha, selected


def _download_from_local_cache(*, repo_id, filename, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision,
        local_files_only=True,
    )


def run_dev18_checkpoint_smoke(
    *, data_root, output_root=None, budget_path=None,
    models=("TimeRCD", "TSPulse"), downloader=None,
    input_loader=load_registered_inputs, probes=None,
    environment_collector=collect_environment_identity,
    seed_setter=set_reproducible_seed,
):
    """Dev18 series 03의 정상 prefix 일부로 exact Tier 3 checkpoint를 확인한다."""
    if not models or not set(models) <= {"TimeRCD", "TSPulse"} or len(set(models)) != len(models):
        raise ValueError("checkpoint 검사 모델은 중복 없는 TimeRCD·TSPulse여야 한다")
    budget, registry_sha, specs = _dev18_budget_specs(budget_path)
    if output_root is None:
        output_root = FULL_PREFIX_OUTPUT_ROOT if budget.get("experiment_mode") == "full_prefix_v2" else DEV18_OUTPUT_ROOT
    inputs = input_loader(
        spec=specs["TimeRCD"], data_root=data_root, series=DEV18_QUICK_SERIES,
    )
    normal_training = numpy.asarray(inputs["normal_training"], dtype=numpy.float32)
    if normal_training.ndim != 2 or normal_training.shape[0] < DEV18_QUICK_LENGTH:
        raise ValueError("Dev18 quick smoke 정상 prefix 길이가 부족하다")
    session = normal_training[:DEV18_QUICK_LENGTH].copy()
    if session.shape != (DEV18_QUICK_LENGTH, 2) or not numpy.isfinite(session).all():
        raise ValueError("Dev18 quick smoke는 series 03의 유한한 1536×2 입력이어야 한다")

    input_identity = inputs.get("input_identity", {})
    input_metadata = {
        "kind": "dev18_normal_prefix",
        "series": DEV18_QUICK_SERIES,
        "source": input_identity.get("name"),
        "source_sha256": input_identity.get("sha256"),
        "input_manifest_sha256": specs["TimeRCD"]["input_manifest_sha256"],
        "slice": [0, DEV18_QUICK_LENGTH],
        "shape": list(session.shape),
        "uses_labels": False,
        "uses_test": False,
        "uses_normal_training": True,
    }
    seed_state = seed_setter(SYNTHETIC_SEED)
    environment = environment_collector()
    downloader = downloader or _download_from_local_cache
    probes = probes or {
        "TimeRCD": lambda **arguments: _probe_time_rcd(
            **arguments, downloader=downloader,
        ),
        "TSPulse": lambda **arguments: _probe_tspulse(
            **arguments, downloader=downloader,
        ),
    }
    reports = {}
    for model, expected_heads in (
        ("TimeRCD", ("score",)),
        ("TSPulse", ("time", "fft", "pred", "raw_max")),
    ):
        if model not in models:
            continue
        spec = specs[model]
        parameters = spec["hyperparameters"]
        normalize_context = spec["common_recipe"].get("methodology_revision") == "source_faithful_v3"
        official_protocol = spec["common_recipe"].get("methodology_revision") == "paper_tuning_v4"
        if model == "TSPulse" and official_protocol:
            expected_heads = ("time", "fft", "pred", "ensemble")
        if model == "TimeRCD":
            probe = lambda **arguments: probes[model](
                **arguments, context_length=parameters["context_length"],
                inference_context_normalization=normalize_context,
                **({"official_protocol": True} if official_protocol else {}),
            )
        else:
            probe = lambda **arguments: probes[model](
                **arguments,
                context_length=parameters["context_length"],
                aggregation_window=parameters["aggregation_window"],
                inference_context_normalization=normalize_context,
                **({"official_protocol": True} if official_protocol else {}),
            )
        report = _run_model_checkpoint_smoke(
            model,
            environment=environment,
            seed_state=seed_state,
            input_length=DEV18_QUICK_LENGTH,
            expected_heads=expected_heads,
            probe=probe,
            cases=((2, session),),
            input_metadata=input_metadata,
            model_identity={
                "config_id": spec["config_id"],
                "probe_config_id": spec["config_id"],
                "source_compatible_config_ids": list(dict.fromkeys(
                    row["config_id"] for row in budget["execution_panel"] if row["model"] == model
                )),
                "hyperparameters": parameters,
                "inference_context_normalization": normalize_context,
                "official_protocol": official_protocol,
                "budget_id": budget["budget_id"],
                "budget_sha256": budget["budget_sha256"],
                "registry_space_sha256": budget["registry_space_sha256"],
                "config_registry_sha256_at_probe": registry_sha,
                "smoke_code_sha256": file_sha256(Path(__file__)),
                "adapter_sha256": file_sha256(
                    REPOSITORY_ROOT / "src" / "models" / "tier3"
                    / ("time_rcd.py" if model == "TimeRCD" else "tspulse.py")
                ),
            },
        )
        _write_model_report(output_root, model, report, "dev18")
        reports[model] = report
    return {
        "status": (
            "passed" if all(report["status"] == "passed" for report in reports.values())
            else "failed"
        ),
        "models": reports,
    }


def _write_model_report(output_root, model, report, scope):
    path = (
        Path(output_root) / MODEL_DIRECTORIES[model]
        / f"{scope}_checkpoint_smoke.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_checkpoint_smoke(
    *, output_root=OUTPUT_ROOT, model_runners=None,
    environment_collector=collect_environment_identity,
    seed_setter=set_reproducible_seed,
):
    """두 모델을 끝까지 실행하고 모델별 JSON을 각각 남긴다."""
    runners = model_runners or {
        "TimeRCD": run_time_rcd_checkpoint_smoke,
        "TSPulse": run_tspulse_checkpoint_smoke,
    }
    missing = set(MODEL_DIRECTORIES) - runners.keys()
    if missing:
        raise ValueError(f"checkpoint smoke runner is missing: {sorted(missing)}")
    seed_state = {"status": "unavailable"}
    environment = {"status": "unavailable"}
    try:
        seed_state = seed_setter(SYNTHETIC_SEED)
        environment = environment_collector()
    except Exception as error:
        reports = {
            model: _failed_base_report(
                model,
                environment=environment,
                seed_state=seed_state,
                stage="environment",
                error=error,
            )
            for model in MODEL_DIRECTORIES
        }
    else:
        reports = {}
        for model in MODEL_DIRECTORIES:
            try:
                report = runners[model](
                    environment=environment, seed_state=seed_state,
                )
                if (
                    not isinstance(report, dict)
                    or report.get("model") != model
                    or report.get("status") not in {"passed", "failed"}
                ):
                    raise ValueError(f"{model} runner returned an invalid report")
            except Exception as error:  # 다른 모델 실행과 JSON 저장은 계속한다.
                report = _failed_base_report(
                    model,
                    environment=environment,
                    seed_state=seed_state,
                    stage="model_runner",
                    error=error,
                )
            reports[model] = report
    for model, report in reports.items():
        _write_model_report(output_root, model, report, "synthetic")
    return {
        "status": (
            "passed" if all(report["status"] == "passed" for report in reports.values())
            else "failed"
        ),
        "models": reports,
    }


def require_checkpoint_smoke_success(report):
    failed = [
        model for model, result in report["models"].items()
        if result["status"] != "passed"
    ]
    if failed:
        raise RuntimeError(f"checkpoint smoke failed: {', '.join(failed)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev18-data-root", type=Path)
    parser.add_argument("--budget-path", type=Path)
    arguments = parser.parse_args()
    report = (
        run_dev18_checkpoint_smoke(data_root=arguments.dev18_data_root, budget_path=arguments.budget_path)
        if arguments.dev18_data_root else run_checkpoint_smoke()
    )
    require_checkpoint_smoke_success(report)


if __name__ == "__main__":
    main()
