"""Dev18 등록 배치와 PCA 대표 실행의 자원 사용량을 원격에서 점검한다."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
from tests.checks.run_lightning_dev18 import (
    PYTORCH_ALLOC_CONF_VALUE,
    configure_cuda_environment,
)

configure_cuda_environment()

import numpy

GPU_PROBE_MODELS = ("PaAno", "GDN", "TimeRCD", "TSPulse")
RESULT_PREFIX = "DEV18_RESOURCE_RESULT="
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / ".runtime" / "dev18_resource_gate.json"
FULL_PREFIX_REPORT_PATH = REPOSITORY_ROOT / "experiments/checks/reference_code/active_models/full_prefix_v2/resource_gate.json"


def capacity_status(peak_bytes: int, total_bytes: int, maximum_percent: float) -> str:
    if peak_bytes < 0 or total_bytes <= 0 or not 0 < maximum_percent < 100:
        raise ValueError("자원 사용량과 합격선이 잘못됐다")
    return "passed" if peak_bytes * 100 < total_bytes * maximum_percent else "failed"


def observe_disk_space(path: Path) -> dict:
    """현재 파일시스템 용량을 기록한다. 전체 산출물의 저장 가능 판정은 아니다."""
    observation = {
        "path": str(path), "observed_at_unix_seconds": time.time(),
        "policy": "informational_only", "total_bytes": None, "used_bytes": None, "free_bytes": None,
    }
    try:
        directory = Path(path).resolve()
        while not directory.exists():
            directory = directory.parent
        usage = shutil.disk_usage(directory)
    except OSError as error:
        return {**observation, "status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    return {
        **observation, "status": "observed", "filesystem_path": str(directory),
        "total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free, "reason": None,
    }


def verify_input_files(entries: list[dict], data_root: Path) -> dict:
    """18개 입력의 경로·크기·SHA-256을 모델 실행 전에 확인한다."""
    root = Path(data_root).resolve()
    total_bytes = 0
    for entry in entries:
        path = (root / entry["source_directory"] / entry["name"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Dev18 입력 경로가 data root 밖을 가리킨다") from error
        if not path.is_file():
            raise FileNotFoundError(f"Dev18 입력 파일이 없다: {path}")
        size = path.stat().st_size
        if size != entry["size_bytes"]:
            raise ValueError(f"Dev18 입력 크기가 다르다: {entry['series']}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"Dev18 입력 SHA-256이 다르다: {entry['series']}")
        total_bytes += size
    return {
        "resource": "input_files",
        "status": "passed",
        "file_count": len(entries),
        "total_bytes": total_bytes,
    }


def _risk_value(spec: dict, entry: dict) -> int:
    parameters = spec["hyperparameters"]
    channels = entry["feature_count"]
    test_count = entry["row_count"] - entry["training_boundary"]
    available = math.floor(entry["training_boundary"] * spec["ratio"] / 100)
    fit_count = available if spec.get("target_use") == "fit_full_prefix" else math.floor(available * 0.8)
    model = spec["model"]
    if model == "PaAno":
        patches = max(0, fit_count - parameters["patch_size"] + 1)
        return (
            min(parameters["batch_size"], patches)
            * parameters["patch_size"] * channels
        )
    if model == "GDN":
        batch = min(
            parameters["batch_size"],
            max(0, fit_count - parameters["window"]),
        )
        return (
            channels * channels * parameters["embedding"]
            + batch * channels * parameters["window"]
        )
    if model == "TimeRCD":
        context = min(test_count, parameters["context_length"])
        return context * context * channels
    if model == "TSPulse":
        from src.common.run_registered_model import build_entrypoint_arguments

        context = parameters["context_length"]
        registered = build_entrypoint_arguments(
            spec, device="cuda", channel_count=channels,
        )
        batch = min(registered["batch_size"], max(0, test_count - context))
        return batch * context * channels
    raise ValueError(f"GPU probe 대상이 아니다: {model}")


def select_probe_cases(specs: list[dict], entries: list[dict]) -> list[dict]:
    """실행 config마다 최고 ratio·최대 자원 입력 한 건을 고른다."""
    cases = []
    for model in GPU_PROBE_MODELS:
        model_specs = [spec for spec in specs if spec["model"] == model]
        for config_id in sorted({spec["config_id"] for spec in model_specs}):
            candidates = [spec for spec in model_specs if spec["config_id"] == config_id]
            spec = sorted(
                candidates, key=lambda value: (-value["ratio"], value["seed"]),
            )[0]
            eligible_entries = entries
            if spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4":
                from src.common.model_feasibility import assess_candidate
                from src.data_split.split_ratio_prefix import compute_prefix_counts

                eligible_entries = []
                for value in entries:
                    _, fit_count, validation_count = compute_prefix_counts(
                        value["training_boundary"], spec["ratio"],
                        full_prefix=spec["target_use"] == "fit_full_prefix",
                    )
                    if assess_candidate(model, spec["hyperparameters"], fit_count, validation_count,
                        value["row_count"] - value["training_boundary"], value["feature_count"],
                        full_prefix=spec["target_use"] == "fit_full_prefix", official_protocol=True,
                    )["status"] == "feasible":
                        eligible_entries.append(value)
                if not eligible_entries:
                    raise ValueError(f"실행 가능한 자원 probe 입력이 없다: {model}/{config_id}")
            entry = max(
                eligible_entries,
                key=lambda value: (_risk_value(spec, value), value["series"]),
            )
            cases.append({
                "model": model,
                "config_id": config_id,
                "ratio": spec["ratio"],
                "seed": spec["seed"],
                "series": entry["series"],
            })
    return cases


def _system_memory_bytes() -> int:
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            limit = int(Path(path).read_text(encoding="utf-8").strip())
            if limit > 0:
                total = min(total, limit)
        except (OSError, ValueError):
            continue
    return total


def _maximum_rss_bytes() -> int:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True,
    ).strip()


def _load_resource_budget():
    from src.common.model_registry import load_model_registry, validate_primary_hpo_seal
    from tests.ghl_main.run_dev18_tuning import DEFAULT_BUDGET_PATH, SNAPSHOT_DIRECTORY, _read_json

    registry = load_model_registry()
    full_prefix = registry["common_recipe"].get("training_split") == "full_prefix_v2"
    path = SNAPSHOT_DIRECTORY / "full_prefix_v2/budget.json" if full_prefix else DEFAULT_BUDGET_PATH
    artifact = _read_json(path)
    budget = artifact["budget"] if full_prefix else artifact
    validate_primary_hpo_seal(registry, budget=budget if full_prefix else None)
    return budget


def _load_plan():
    from src.common.execution_identity import load_input_manifest_role
    from tests.ghl_main.run_dev18_tuning import (
        _specs_for_budget,
    )

    budget = _load_resource_budget()
    specs, _ = _specs_for_budget(budget)
    manifest, _ = load_input_manifest_role(
        REPOSITORY_ROOT / "configs" / "input_manifest.yaml",
        "development", "dev18_selection",
    )
    return specs, manifest["datasets"]["DEV18"]["files"]


def _find_case(model: str, config_id: str, series: str):
    specs, entries = _load_plan()
    if model == "PCA_LEGACY":
        estimate = _pca_static_check(specs, entries, 80)
        case = {name: estimate[name] for name in ("model", "config_id", "series", "ratio", "seed")}
        if case["config_id"] != config_id or case["series"] != series:
            raise ValueError("PCA resource probe가 현재 대표 입력·설정과 다르다")
        return next(spec for spec in specs if all(spec[name] == case[name]
                    for name in ("model", "config_id", "ratio", "seed"))), case
    case = next(
        value for value in select_probe_cases(specs, entries)
        if value["model"] == model and value["config_id"] == config_id
    )
    if case["series"] != series:
        raise ValueError("resource probe 입력 선택이 현재 manifest와 다르다")
    spec = next(
        value for value in specs
        if value["model"] == model
        and value["config_id"] == config_id
        and value["ratio"] == case["ratio"]
        and value["seed"] == case["seed"]
    )
    return spec, case


def summarize_tspulse_equivalence(
    reference_outputs: dict, registered_outputs: dict, *, registered_batch_size: int,
    official_protocol: bool = False,
) -> dict:
    """배치 1과 등록 배치의 실제 checkpoint 점수를 head별로 대조한다."""
    heads = ("time", "fft", "pred", "ensemble") if official_protocol else ("time", "fft", "pred", "raw_max")
    if set(reference_outputs) != set(heads) or set(registered_outputs) != set(heads):
        raise ValueError("TSPulse equivalence score heads가 다르다")
    maximum_differences = {}
    equivalent = True
    for head in heads:
        reference = numpy.asarray(reference_outputs[head]["scores"])
        registered = numpy.asarray(registered_outputs[head]["scores"])
        if reference.shape != registered.shape or not (
            numpy.isfinite(reference).all() and numpy.isfinite(registered).all()
        ):
            raise ValueError(f"TSPulse {head} equivalence score가 유효하지 않다")
        maximum_differences[head] = float(numpy.max(numpy.abs(reference - registered)))
        equivalent &= numpy.allclose(reference, registered, rtol=1e-6, atol=1e-8)
    return {
        "status": "passed" if equivalent else "failed",
        "reference_batch_size": 1,
        "registered_batch_size": registered_batch_size,
        "rtol": 1e-6,
        "atol": 1e-8,
        "head_maximum_absolute_differences": maximum_differences,
    }


def _run_model_probe(spec: dict, inputs: dict, *, device: str):
    from src.common.run_registered_model import (
        build_entrypoint_arguments,
        load_model_entrypoint,
        prepare_session_inputs,
        select_input_scaler,
    )

    model = spec["model"]
    full_prefix = spec.get("target_use") == "fit_full_prefix"
    official_protocol = spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"
    test = inputs["test_sessions"][0]
    arguments = build_entrypoint_arguments(
        spec, device=device, channel_count=test.shape[1],
    )
    if model != "TSPulse":
        entrypoint = load_model_entrypoint(model)
    if model in {"PaAno", "GDN"}:
        prepared = prepare_session_inputs(
            normal_training=inputs["normal_training"],
            test_sessions=(test,), ratio_percent=spec["ratio"],
            scale=select_input_scaler(spec) != "none",
            scaler_kind=select_input_scaler(spec),
            full_prefix=full_prefix,
        )
        fit = prepared["fit_sessions"][0]
        validation = None if full_prefix else prepared["validation_sessions"][0]
        test = prepared["test_sessions"][0]
    if model == "PaAno":
        arguments["iterations"] = 1
        length = arguments["patch_size"] + arguments["batch_size"] - 1
        adapter = entrypoint(**arguments)
        adapter.fit(fit[:length])
        adapter.score(test[:length] if full_prefix else validation[:length])
    elif model == "GDN":
        arguments.update({"epochs": 1, "patience": 1})
        window_count = 8 * arguments["batch_size"]
        if official_protocol:
            window_count = math.ceil(window_count / (1 - arguments["validation_ratio"]))
        length = arguments["window_size"] + window_count
        entrypoint(
            (fit[:length],), () if full_prefix else (validation[:length],), (test[:length],),
            **arguments,
        )
        training_windows = max(0, min(len(fit), length) - arguments["window_size"])
        if official_protocol:
            training_windows -= int(training_windows * arguments["validation_ratio"])
        return {
            "probe_scope": ("one epoch over prefix windows with official internal validation"
                            if official_protocol else "exact maximum batch; eight consecutive training updates"),
            "training_full_batch_count": training_windows // arguments["batch_size"],
        }
    elif model == "TimeRCD":
        import torch

        from src.models.tier3.time_rcd import TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE

        arguments["query_chunk_size"] = TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE
        started = time.perf_counter()
        entrypoint(test[:arguments["context_length"]], **arguments)
        torch.cuda.synchronize(device)
        return {
            "wall_time_seconds": time.perf_counter() - started,
            "execution_policy": {
                "status": "passed",
                "context_length": arguments["context_length"],
                "attention_query_chunk_size": arguments["query_chunk_size"],
                **({"official_protocol": True} if official_protocol else {}),
            },
        }
    elif model == "TSPulse":
        import torch

        from src.models.tier3.tspulse import (
            build_tspulse_raw_head_function,
            load_tspulse_components,
            score_tspulse,
            score_tspulse_paper,
        )

        context = arguments["context_length"]
        aggregation_window = arguments["aggregation_window"]
        batch_size = arguments["batch_size"]
        length = context + max(batch_size + 1, aggregation_window // 2 + 1)
        session = test[:length]
        model_instance, utility = load_tspulse_components(
            aggregation_window=aggregation_window,
            channel_count=test.shape[1], device=device,
            **({"official_protocol": True} if official_protocol else {}),
        )

        def score_with_batch(current_batch_size):
            raw_head_function = build_tspulse_raw_head_function(
                utility, aggregation_window=aggregation_window,
                context_length=context, batch_size=current_batch_size, device=device,
                inference_context_normalization=arguments.get("inference_context_normalization", False),
            )
            scorer = score_tspulse_paper if official_protocol else score_tspulse
            return scorer(
                session, raw_head_function=raw_head_function,
                aggregation_window=aggregation_window, context_length=context,
                **({"utility": utility} if official_protocol else {}),
            )

        reference_outputs = score_with_batch(1)
        registered_outputs = score_with_batch(batch_size)
        equivalence = summarize_tspulse_equivalence(
            reference_outputs, registered_outputs,
            registered_batch_size=batch_size,
            official_protocol=official_protocol,
        )
        torch.cuda.synchronize(device)
        del reference_outputs, registered_outputs
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        measured_outputs = score_with_batch(batch_size)
        torch.cuda.synchronize(device)
        wall_time_seconds = time.perf_counter() - started
        del measured_outputs, model_instance, utility
        return {
            "wall_time_seconds": wall_time_seconds,
            "execution_policy": {
                "status": "passed",
                "batch_size": batch_size,
                "context_length": context,
                "aggregation_window": aggregation_window,
                **({"official_protocol": True} if official_protocol else {}),
            },
            "equivalence": equivalence,
        }
    else:
        raise ValueError(f"GPU probe 대상이 아니다: {model}")
    return {}


def run_child_probe(
    *, model: str, config_id: str, series: str, data_root: Path,
    maximum_memory_percent: float,
) -> dict:
    import torch

    from src.common.set_reproducible_seed import set_reproducible_seed
    from tests.ghl_main.run_registered_models import load_registered_inputs

    if model == "PCA_LEGACY":
        from tests.checks.run_lightning_dev18 import require_lightning_cuda

        require_lightning_cuda()
    spec, case = _find_case(model, config_id, series)
    set_reproducible_seed(case["seed"])
    inputs = load_registered_inputs(
        spec=spec,
        input_manifest_path=REPOSITORY_ROOT / "configs" / "input_manifest.yaml",
        series=series, data_root=data_root,
    )
    if model == "PCA_LEGACY":
        from src.common.run_registered_model import execute_registered_model

        started = time.perf_counter()
        error = ""
        try:
            execute_registered_model(
                spec, normal_training=inputs.get("normal_training"),
                test_sessions=inputs["test_sessions"], device="cpu",
            )
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        elapsed = time.perf_counter() - started
        peak, total = _maximum_rss_bytes(), _system_memory_bytes()
        status = "failed" if error else capacity_status(peak, total, maximum_memory_percent)
        return {
            **case, "status": status,
            "error": error or ("" if status == "passed" else "PCA 실제 RAM 사용량이 합격선을 넘었다"),
            "measurement_kind": "process_rss", "actual_backend": "cpu",
            "ram_peak_bytes": peak, "ram_total_bytes": total,
            "ram_peak_percent": round(peak * 100 / total, 2),
            "maximum_memory_percent": maximum_memory_percent, "wall_time_seconds": elapsed,
            "probe_scope": "isolated process lifetime RSS including inputs and registered PCA fit/score; representative case only",
        }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats("cuda")
    free_before, total_gpu = torch.cuda.mem_get_info("cuda")
    try:
        started = time.perf_counter()
        evidence = _run_model_probe(spec, inputs, device="cuda")
        torch.cuda.synchronize()
        evidence.setdefault("wall_time_seconds", time.perf_counter() - started)
        peak_reserved = torch.cuda.max_memory_reserved("cuda")
        peak_gpu = total_gpu - free_before + peak_reserved
        peak_ram = _maximum_rss_bytes()
        gpu_status = capacity_status(
            peak_gpu, total_gpu, maximum_memory_percent,
        )
        ram_status = capacity_status(
            peak_ram, _system_memory_bytes(), maximum_memory_percent,
        )
        evidence_passed = all(
            value.get("status") == "passed"
            for key, value in evidence.items()
            if key in {"execution_policy", "equivalence"}
        )
        status = (
            "passed"
            if gpu_status == ram_status == "passed" and evidence_passed
            else "failed"
        )
        error = "" if status == "passed" else "실행 정책 또는 자원 합격선을 통과하지 못했다"
    except (MemoryError, RuntimeError) as exception:
        evidence = {}
        peak_reserved = torch.cuda.max_memory_reserved("cuda")
        peak_gpu = min(total_gpu, total_gpu - free_before + peak_reserved)
        peak_ram = _maximum_rss_bytes()
        status = "failed"
        error = f"{type(exception).__name__}: {exception}"
    return {
        **case,
        **evidence,
        "status": status,
        "error": error,
        "gpu_peak_bytes": peak_gpu,
        "gpu_total_bytes": total_gpu,
        "gpu_peak_percent": round(peak_gpu * 100 / total_gpu, 2),
        "ram_peak_bytes": peak_ram,
        "ram_total_bytes": _system_memory_bytes(),
        "ram_peak_percent": round(peak_ram * 100 / _system_memory_bytes(), 2),
        "maximum_memory_percent": maximum_memory_percent,
        "probe_scope": evidence.get(
            "probe_scope",
            "exact maximum batch; one training update for learned models",
        ),
    }


def _pca_static_check(specs, entries, maximum_memory_percent: float) -> dict:
    pca_specs = [spec for spec in specs if spec["model"] == "PCA_LEGACY"]
    if not pca_specs:
        return {"model": "PCA_LEGACY", "status": "not_in_panel"}
    spec = min(pca_specs, key=lambda value: (
        value["hyperparameters"]["n_components"] is not None, -value["ratio"], value["seed"], value["config_id"],
    ))
    window = spec["hyperparameters"]["window"]

    def estimate(entry):
        return max(
            0,
            (entry["row_count"] - entry["training_boundary"] if spec.get("target_use") == "training_free"
             else math.floor(entry["training_boundary"] * spec["ratio"] / 100 * (1.0 if spec.get("target_use") == "fit_full_prefix" else 0.8)))
            - window + 1,
        ) * window * entry["feature_count"] * 8 * 6

    entry = max(entries, key=lambda value: (estimate(value), value["series"]))
    peak_bytes = estimate(entry)
    total = _system_memory_bytes()
    return {
        "model": "PCA_LEGACY",
        "config_id": spec["config_id"], "ratio": spec["ratio"], "seed": spec["seed"], "series": entry["series"],
        "status": ("passed" if capacity_status(peak_bytes, total, maximum_memory_percent) == "passed"
                   else "requires_measurement"),
        "measurement_kind": "static_estimate",
        "estimated_ram_bytes": peak_bytes,
        "ram_total_bytes": total,
        "maximum_memory_percent": maximum_memory_percent,
        "probe_scope": "보수적 PCA 작업 배열 6벌 RAM 상한",
    }


def _check_pca_resources(specs, entries, maximum_memory_percent, *, data_root, history_directory, identity):
    estimate = _pca_static_check(specs, entries, maximum_memory_percent)
    if estimate["status"] != "requires_measurement":
        return estimate
    case = {name: estimate[name] for name in ("model", "config_id", "series", "ratio", "seed")}
    command = [sys.executable, str(Path(__file__).resolve()), "--model", case["model"],
               "--config-id", case["config_id"], "--series", case["series"],
               "--data-root", str(data_root), "--maximum-memory-percent", str(maximum_memory_percent)]
    print(f"PCA RAM 추정 초과: series {case['series']}의 등록 설정을 별도 CPU 프로세스에서 측정합니다", flush=True)
    return _run_resource_probe(case, command, history_directory, {**identity, "pca_estimate": estimate})


def _write_resource_report(report: dict, path=DEFAULT_REPORT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_finite_number(value, *, positive=False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _has_consistent_capacity_evidence(
    result_rows: list[dict], expected_models: set, maximum_memory_percent: float,
) -> bool:
    if "PCA_LEGACY" in expected_models:
        pca_rows = [row for row in result_rows if row.get("model") == "PCA_LEGACY"]
        if len(pca_rows) != 1:
            return False
        row = pca_rows[0]
        total = row.get("ram_total_bytes")
        if row.get("maximum_memory_percent") != maximum_memory_percent or not _is_finite_number(total, positive=True):
            return False
        if row.get("measurement_kind") == "static_estimate":
            peak = row.get("estimated_ram_bytes")
        elif row.get("measurement_kind") == "process_rss":
            peak = row.get("ram_peak_bytes")
            if (row.get("actual_backend") != "cpu" or not _is_finite_number(peak)
                    or row.get("ram_peak_percent") != round(peak * 100 / total, 2)
                    or not _is_finite_number(row.get("wall_time_seconds"), positive=True)):
                return False
        else:
            return False
        if not _is_finite_number(peak) or capacity_status(peak, total, maximum_memory_percent) != "passed":
            return False
    measured_models = expected_models & set(GPU_PROBE_MODELS)
    measured_rows = [
        row for row in result_rows if row.get("model") in measured_models
    ]
    if {row.get("model") for row in measured_rows} != measured_models:
        return False
    for row in measured_rows:
        if row.get("maximum_memory_percent") != maximum_memory_percent:
            return False
        for resource_name in ("gpu", "ram"):
            peak = row.get(f"{resource_name}_peak_bytes")
            total = row.get(f"{resource_name}_total_bytes")
            percent = row.get(f"{resource_name}_peak_percent")
            if not (
                _is_finite_number(peak)
                and _is_finite_number(total, positive=True)
                and _is_finite_number(percent)
                and percent == round(peak * 100 / total, 2)
                and capacity_status(peak, total, maximum_memory_percent) == "passed"
            ):
                return False
    return True


def _has_required_pca_evidence(result_rows: list[dict], expected_models: set, maximum_memory_percent: float) -> bool:
    if "PCA_LEGACY" not in expected_models:
        return True
    specs, entries = _load_plan()
    estimate = _pca_static_check(specs, entries, maximum_memory_percent)
    rows = [row for row in result_rows if row.get("model") == "PCA_LEGACY"]
    if len(rows) != 1 or any(rows[0].get(name) != estimate.get(name)
                             for name in ("config_id", "series", "ratio", "seed")):
        return False
    row = rows[0]
    if row.get("measurement_kind") == "static_estimate":
        return row == estimate and estimate["status"] == "passed"
    return (row.get("measurement_kind") == "process_rss" and estimate["status"] == "requires_measurement"
            and row.get("pca_estimate") == estimate and isinstance(row.get("probe_history"), dict))


def _has_required_tier3_evidence(result_rows: list[dict], expected_models: set) -> bool:
    from src.common.run_registered_model import (
        build_entrypoint_arguments,
    )
    from src.models.tier3.time_rcd import TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE

    tier3_models = expected_models & {"TimeRCD", "TSPulse"}
    if not tier3_models:
        return True
    specs, entries = _load_plan()
    cases = [
        case for case in select_probe_cases(specs, entries)
        if case["model"] in tier3_models
    ]
    expected = {}
    for case in cases:
        spec = next(
            value for value in specs
            if all(
                value[name] == case[name]
                for name in ("model", "config_id", "ratio", "seed")
            )
        )
        entry = next(value for value in entries if value["series"] == case["series"])
        arguments = build_entrypoint_arguments(
            spec, device="cuda", channel_count=entry["feature_count"],
        )
        official_protocol = spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"
        if case["model"] == "TimeRCD":
            policy = {
                "status": "passed",
                "context_length": arguments["context_length"],
                "attention_query_chunk_size": TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE,
                **({"official_protocol": True} if official_protocol else {}),
            }
        else:
            policy = {
                "status": "passed",
                "batch_size": arguments["batch_size"],
                "context_length": arguments["context_length"],
                "aggregation_window": arguments["aggregation_window"],
                **({"official_protocol": True} if official_protocol else {}),
            }
        expected[(case["model"], case["config_id"])] = (case, policy)

    tier3_rows = [row for row in result_rows if row.get("model") in tier3_models]
    rows = {
        (row.get("model"), row.get("config_id")): row
        for row in tier3_rows
    }
    if len(rows) != len(tier3_rows) or rows.keys() != expected.keys():
        return False
    for key, (case, policy) in expected.items():
        row = rows[key]
        if (
            any(row.get(name) != case[name] for name in ("ratio", "seed", "series"))
            or row.get("execution_policy") != policy
            or not _is_finite_number(row.get("wall_time_seconds"), positive=True)
            or any(
                not _is_finite_number(row.get(name))
                for name in (
                    "gpu_peak_bytes", "gpu_peak_percent",
                    "ram_peak_bytes", "ram_peak_percent",
                )
            )
        ):
            return False
        if key[0] == "TSPulse":
            equivalence = row.get("equivalence", {})
            differences = equivalence.get("head_maximum_absolute_differences", {})
            if not (
                equivalence.get("status") == "passed"
                and equivalence.get("reference_batch_size") == 1
                and equivalence.get("registered_batch_size") == policy["batch_size"]
                and equivalence.get("rtol") == 1e-6
                and equivalence.get("atol") == 1e-8
                and set(differences) == ({"time", "fft", "pred", "ensemble"}
                                         if policy.get("official_protocol") else {"time", "fft", "pred", "raw_max"})
                and all(_is_finite_number(value) for value in differences.values())
            ):
                return False
    return True


def validate_resource_report(
    path=None, *, project_commit=None, gate_code_sha256=None,
    input_manifest_sha256=None, budget_id=None, cuda_device=None,
    expected_models=None, environment=None, ram_total_bytes=None,
) -> dict:
    from src.common.execution_identity import file_sha256
    from tests.checks.seal_runtime_environment import (
        collect_cuda_device_identity, collect_runtime_environment_identity,
    )
    if path is None:
        path = FULL_PREFIX_REPORT_PATH if _load_resource_budget().get("experiment_mode") == "full_prefix_v2" else DEFAULT_REPORT_PATH
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "Dev18 자원 gate 결과가 없다. check_dev18_resources.py를 먼저 실행한다"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") == "failed":
        failures = [
            f"{row.get('model', row.get('resource', 'unknown'))} "
            f"{row.get('config_id', '')} / series {row.get('series', '-')}: "
            f"{row.get('error') or row.get('status')} "
            f"(GPU {row.get('gpu_peak_percent', '미측정')}%, "
            f"RAM {row.get('ram_peak_percent', '미측정')}%, "
            f"기준 {row.get('maximum_memory_percent', '미기록')}%)"
            for row in report.get("results", []) if row.get("status") != "passed"
        ]
        raise ValueError(f"Dev18 자원 점검 실패: {path}\n" + "\n".join(failures))
    if project_commit is None:
        project_commit = _git_head()
    if gate_code_sha256 is None:
        gate_code_sha256 = file_sha256(Path(__file__))
    if input_manifest_sha256 is None:
        input_manifest_sha256 = file_sha256(
            REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
        )
    if budget_id is None:
        budget_id = _load_resource_budget()["budget_id"]
    if cuda_device is None:
        cuda_device = collect_cuda_device_identity()
    if expected_models is None:
        expected_models = {spec["model"] for spec in _load_plan()[0]}
    if report.get("cuda_device") != cuda_device:
        raise ValueError("자원 gate를 통과한 GPU와 현재 GPU가 다르다")
    if environment is None:
        environment = collect_runtime_environment_identity()
    if ram_total_bytes is None:
        ram_total_bytes = _system_memory_bytes()
    result_rows = report.get("results")
    maximum_memory_percent = report.get("maximum_memory_percent")
    valid = (
        report.get("status") == "passed"
        and report.get("project_commit") == project_commit
        and report.get("gate_code_sha256") == gate_code_sha256
        and report.get("input_manifest_sha256") == input_manifest_sha256
        and report.get("budget_id") == budget_id
        and report.get("environment") == environment
        and report.get("ram_total_bytes") == ram_total_bytes
        and report.get("pytorch_alloc_conf") == PYTORCH_ALLOC_CONF_VALUE
        and _is_finite_number(maximum_memory_percent, positive=True)
        and maximum_memory_percent <= 80
        and set(report.get("checked_models", ())) == set(expected_models)
        and isinstance(result_rows, list)
        and bool(result_rows)
        and all(row.get("status") == "passed" for row in result_rows)
        and all(row.get("ram_total_bytes") == ram_total_bytes
                for row in result_rows if row.get("model") in (*GPU_PROBE_MODELS, "PCA_LEGACY"))
        and _has_consistent_capacity_evidence(
            result_rows, set(expected_models), maximum_memory_percent,
        )
        and _has_required_pca_evidence(result_rows, set(expected_models), maximum_memory_percent)
        and _has_required_tier3_evidence(result_rows, set(expected_models))
    )
    if not valid:
        raise ValueError("Dev18 자원 gate 결과가 현재 코드·입력·예산과 다르다")
    return report


def _run_resource_probe(case, command, history_directory, identity):
    from src.common.execution_identity import file_sha256
    from tests.ghl_main.record_run_history import (
        _load_histories, record_run_history, save_run_history,
    )

    identity = {**identity, "kind": "resource_probe", "case": dict(case)}
    for path, _, previous in _load_histories(history_directory, identity["budget_id"]):
        result = previous.get("result", {})
        if (previous["identity"] == identity and previous["status"] == "complete"
                and result.get("status") == "passed"):
            if (any(result.get(name) != value for name, value in case.items())
                    or (case["model"] == "PCA_LEGACY" and result.get("pca_estimate") != identity.get("pca_estimate"))
                    or not _has_consistent_capacity_evidence(
                        [result], {case["model"]}, identity["maximum_memory_percent"],
                    ) or result.get("ram_total_bytes") != identity["ram_total_bytes"]):
                raise ValueError("저장된 자원 probe 근거가 실행 신원과 다르다")
            return {**result, "probe_history": {"file": str(path), "sha256": file_sha256(path)}}

    with record_run_history(history_directory, identity=identity) as history:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            try:
                history["child_pid"] = process.pid
                save_run_history(history)
                stdout, stderr = process.communicate()
            except BaseException:
                process.kill()
                process.wait()
                raise
        line = next((value for value in reversed(stdout.splitlines())
                     if value.startswith(RESULT_PREFIX)), None)
        if process.returncode or line is None:
            result = {**case, "status": "failed", "error": stderr.strip() or stdout.strip()
                      or f"probe process exit {process.returncode}"}
        else:
            result = json.loads(line.removeprefix(RESULT_PREFIX))
            if any(result.get(name) != value for name, value in case.items()):
                raise ValueError("자원 probe가 다른 실행의 결과를 반환했다")
        if case["model"] == "PCA_LEGACY":
            result["pca_estimate"] = identity["pca_estimate"]
        history["result"] = result
    path = Path(history["history_file"])
    return {**result, "probe_history": {"file": str(path), "sha256": file_sha256(path)}}


def run_resource_check(
    *, data_root: Path, maximum_memory_percent: float = 80,
    output_path=None,
) -> dict:
    from src.common.execution_identity import file_sha256
    from tests.checks.run_lightning_dev18 import require_lightning_cuda
    from tests.checks.seal_runtime_environment import collect_runtime_environment_identity
    from src.common.set_reproducible_seed import set_reproducible_seed
    from tests.ghl_main.run_dev18_tuning import _require_clean_worktree

    require_lightning_cuda()
    _require_clean_worktree()
    set_reproducible_seed(0)
    specs, entries = _load_plan()
    cases = select_probe_cases(specs, entries)
    budget = _load_resource_budget()
    full_prefix = budget.get("experiment_mode") == "full_prefix_v2"
    if output_path is None:
        output_path = FULL_PREFIX_REPORT_PATH if full_prefix else DEFAULT_REPORT_PATH
    environment = collect_runtime_environment_identity()
    identity = {
        "project_commit": _git_head(),
        "gate_code_sha256": file_sha256(Path(__file__)),
        "input_manifest_sha256": file_sha256(REPOSITORY_ROOT / "configs/input_manifest.yaml"),
        "budget_id": budget["budget_id"],
        "pytorch_alloc_conf": os.environ["PYTORCH_ALLOC_CONF"],
        "cuda_device": environment["cuda_device"],
        "environment": environment,
        "ram_total_bytes": _system_memory_bytes(),
        "maximum_memory_percent": maximum_memory_percent,
    }
    history_directory = Path(output_path).parent / "resource_probe_history"
    disk_observation = observe_disk_space(REPOSITORY_ROOT / "experiments/01_ghl_main")
    results = [verify_input_files(entries, data_root)] + [
        {
            "model": model,
            "status": "passed",
            "probe_scope": "배열 연산이며 GPU OOM 위험 없음",
        }
        for model in ("MWVAR", "SQDIFF_LAST1", "SQDIFF_LAST3", "SQDIFF_CENTERED5",
                      "MWVAR96_SQDIFF_LAST3", "MWVAR96_SQDIFF_CENTERED5")
        if any(spec["model"] == model for spec in specs)
    ]
    results.append(_check_pca_resources(
        specs, entries, maximum_memory_percent, data_root=data_root,
        history_directory=history_directory, identity=identity,
    ))
    for case in cases:
        print(
            f"점검 중: {case['model']} {case['config_id']} / series {case['series']}",
            flush=True,
        )
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--model", case["model"],
            "--config-id", case["config_id"],
            "--series", case["series"],
            "--data-root", str(data_root),
            "--maximum-memory-percent", str(maximum_memory_percent),
        ]
        results.append(_run_resource_probe(case, command, history_directory, identity))
    failed = [result for result in results if result["status"] != "passed"]
    failed_policies = ", ".join(
        result.get("model", result.get("resource", "unknown"))
        for result in failed
    )
    report = {
        **identity,
        "status": "passed" if not failed else "failed",
        "disk_observation": disk_observation,
        "probe_history_directory": str(history_directory),
        "probe_cost_scope": "사전 검사 내역이며 단일 진입 명령의 시간에 포함된다. 두 시간을 더하지 않는다.",
        "checked_models": sorted(
            {result["model"] for result in results if "model" in result}
        ),
        "results": results,
        "next_command": (
            ("python -m tests.ghl_main.run_ratio_tuning --execute-only" if full_prefix else "python tests/checks/run_lightning_dev18.py")
            if not failed
            else (
                f"본 튜닝을 시작하지 말고 실패한 정책({failed_policies})의 "
                "실행 근거와 자원 사용량을 확인하세요."
            )
        ),
    }
    _write_resource_report(report, output_path)
    return report


def main() -> None:
    from tests.ghl_main.run_dev18_tuning import DEFAULT_DATA_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--maximum-memory-percent", type=float, default=80)
    parser.add_argument("--model", choices=(*GPU_PROBE_MODELS, "PCA_LEGACY"))
    parser.add_argument("--config-id")
    parser.add_argument("--series")
    arguments = parser.parse_args()
    child = any((arguments.model, arguments.config_id, arguments.series))
    if child and not all((arguments.model, arguments.config_id, arguments.series)):
        parser.error("내부 model probe 인자를 모두 지정해야 한다")
    if child:
        result = run_child_probe(
            model=arguments.model, config_id=arguments.config_id,
            series=arguments.series, data_root=arguments.data_root,
            maximum_memory_percent=arguments.maximum_memory_percent,
        )
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(
            run_resource_check(
                data_root=arguments.data_root,
                maximum_memory_percent=arguments.maximum_memory_percent,
            ),
            ensure_ascii=False, indent=2,
        ))


if __name__ == "__main__":
    main()
