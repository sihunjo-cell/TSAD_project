"""Dev18 exact panel의 최대 배치 자원 위험을 GPU별로 짧게 점검한다."""

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

import numpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

GPU_PROBE_MODELS = ("PaAno", "GDN", "TimeRCD", "TSPulse")
RESULT_PREFIX = "DEV18_RESOURCE_RESULT="
DEFAULT_REPORT_PATH = REPOSITORY_ROOT / ".runtime" / "dev18_resource_gate.json"
MINIMUM_FREE_DISK_BYTES = 25 * 1024 ** 3


def capacity_status(peak_bytes: int, total_bytes: int, maximum_percent: float) -> str:
    if peak_bytes < 0 or total_bytes <= 0 or not 0 < maximum_percent < 100:
        raise ValueError("자원 사용량과 합격선이 잘못됐다")
    return "passed" if peak_bytes * 100 < total_bytes * maximum_percent else "failed"


def disk_status(free_bytes: int, minimum_bytes: int = MINIMUM_FREE_DISK_BYTES) -> str:
    if free_bytes < 0 or minimum_bytes <= 0:
        raise ValueError("디스크 용량이 잘못됐다")
    return "passed" if free_bytes >= minimum_bytes else "failed"


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
    fit_count = math.floor(available * 0.8)
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
            entry = max(
                entries,
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
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _maximum_rss_bytes() -> int:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True,
    ).strip()


def _load_plan():
    from src.common.execution_identity import load_input_manifest_role
    from tests.ghl_main.run_dev18_tuning import (
        DEFAULT_BUDGET_PATH,
        _read_json,
        _specs_for_budget,
    )

    budget = _read_json(DEFAULT_BUDGET_PATH)
    specs, _ = _specs_for_budget(budget)
    manifest, _ = load_input_manifest_role(
        REPOSITORY_ROOT / "configs" / "input_manifest.yaml",
        "development", "dev18_selection",
    )
    return specs, manifest["datasets"]["DEV18"]["files"]


def _find_case(model: str, config_id: str, series: str):
    specs, entries = _load_plan()
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
) -> dict:
    """배치 1과 등록 배치의 실제 checkpoint 점수를 head별로 대조한다."""
    heads = ("time", "fft", "pred", "raw_max")
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
    )

    model = spec["model"]
    test = inputs["test_sessions"][0]
    arguments = build_entrypoint_arguments(
        spec, device=device, channel_count=test.shape[1],
    )
    if model != "TSPulse":
        entrypoint = load_model_entrypoint(model)
    if model in {"PaAno", "GDN"}:
        prepared = prepare_session_inputs(
            normal_training=inputs["normal_training"],
            test_sessions=(test,), ratio_percent=spec["ratio"], scale=True,
        )
        fit = prepared["fit_sessions"][0]
        validation = prepared["validation_sessions"][0]
    if model == "PaAno":
        arguments["iterations"] = 1
        length = arguments["patch_size"] + arguments["batch_size"] - 1
        adapter = entrypoint(**arguments)
        adapter.fit(fit[:length])
        adapter.score(validation[:length])
    elif model == "GDN":
        arguments.update({"epochs": 1, "patience": 1})
        length = arguments["window_size"] + arguments["batch_size"]
        entrypoint(
            (fit[:length],), (validation[:length],), (test[:length],),
            **arguments,
        )
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
            },
        }
    elif model == "TSPulse":
        import torch

        from src.models.tier3.tspulse import (
            build_tspulse_raw_head_function,
            load_tspulse_components,
            score_tspulse,
        )

        context = arguments["context_length"]
        aggregation_window = arguments["aggregation_window"]
        batch_size = arguments["batch_size"]
        length = context + max(batch_size + 1, aggregation_window // 2 + 1)
        session = test[:length]
        model_instance, utility = load_tspulse_components(
            aggregation_window=aggregation_window,
            channel_count=test.shape[1], device=device,
        )

        def score_with_batch(current_batch_size):
            raw_head_function = build_tspulse_raw_head_function(
                utility, aggregation_window=aggregation_window,
                context_length=context, batch_size=current_batch_size, device=device,
            )
            return score_tspulse(
                session, raw_head_function=raw_head_function,
                aggregation_window=aggregation_window, context_length=context,
            )

        reference_outputs = score_with_batch(1)
        registered_outputs = score_with_batch(batch_size)
        equivalence = summarize_tspulse_equivalence(
            reference_outputs, registered_outputs,
            registered_batch_size=batch_size,
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

    from tests.ghl_main.run_registered_models import load_registered_inputs

    spec, case = _find_case(model, config_id, series)
    inputs = load_registered_inputs(
        spec=spec,
        input_manifest_path=REPOSITORY_ROOT / "configs" / "input_manifest.yaml",
        series=series, data_root=data_root,
    )
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
        "probe_scope": "exact maximum batch; one training update for learned models",
    }


def _pca_static_check(specs, entries, maximum_memory_percent: float) -> dict:
    pca_specs = [spec for spec in specs if spec["model"] == "PCA_LEGACY"]
    if not pca_specs:
        return {"model": "PCA_LEGACY", "status": "not_in_panel"}
    spec = sorted(pca_specs, key=lambda value: (-value["ratio"], value["seed"]))[0]
    window = spec["hyperparameters"]["window"]
    peak_bytes = max(
        max(
            0,
            math.floor(entry["training_boundary"] * spec["ratio"] / 100 * 0.8)
            - window + 1,
        )
        * window * entry["feature_count"] * 8 * 6
        for entry in entries
    )
    total = _system_memory_bytes()
    return {
        "model": "PCA_LEGACY",
        "status": capacity_status(peak_bytes, total, maximum_memory_percent),
        "estimated_ram_bytes": peak_bytes,
        "ram_total_bytes": total,
        "maximum_memory_percent": maximum_memory_percent,
        "probe_scope": "보수적 PCA 작업 배열 6벌 RAM 상한",
    }


def _write_resource_report(report: dict, path=DEFAULT_REPORT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _has_required_tier3_evidence(result_rows: list[dict], expected_models: set) -> bool:
    from src.common.run_registered_model import TSPULSE_INFERENCE_BATCH_SIZE
    from src.models.tier3.time_rcd import TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE

    rows = {row.get("model"): row for row in result_rows if row.get("model")}
    for model in expected_models & {"TimeRCD", "TSPulse"}:
        row = rows.get(model, {})
        if not (
            isinstance(row.get("wall_time_seconds"), (int, float))
            and row["wall_time_seconds"] > 0
            and row.get("gpu_peak_bytes", 0) > 0
            and row.get("ram_peak_bytes", 0) > 0
            and row.get("execution_policy", {}).get("status") == "passed"
        ):
            return False
    if "TimeRCD" in expected_models:
        policy = rows["TimeRCD"]["execution_policy"]
        if (
            policy.get("attention_query_chunk_size")
            != TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE
            or policy.get("context_length", 0) < 1
        ):
            return False
    if "TSPulse" in expected_models:
        policy = rows["TSPulse"]["execution_policy"]
        equivalence = rows["TSPulse"].get("equivalence", {})
        differences = equivalence.get("head_maximum_absolute_differences", {})
        if not (
            policy.get("batch_size") == TSPULSE_INFERENCE_BATCH_SIZE
            and policy.get("context_length", 0) > 0
            and policy.get("aggregation_window", 0) > 0
            and equivalence.get("status") == "passed"
            and equivalence.get("reference_batch_size") == 1
            and equivalence.get("registered_batch_size") == TSPULSE_INFERENCE_BATCH_SIZE
            and equivalence.get("rtol") == 1e-6
            and equivalence.get("atol") == 1e-8
            and set(differences) == {"time", "fft", "pred", "raw_max"}
            and all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in differences.values()
            )
        ):
            return False
    return True


def validate_resource_report(
    path=DEFAULT_REPORT_PATH, *, project_commit=None, gate_code_sha256=None,
    input_manifest_sha256=None, budget_id=None, cuda_device=None,
    expected_models=None,
) -> dict:
    from src.common.execution_identity import file_sha256
    from tests.checks.seal_runtime_environment import collect_cuda_device_identity
    from tests.ghl_main.run_dev18_tuning import DEFAULT_BUDGET_PATH, _read_json

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "Dev18 자원 gate 결과가 없다. check_dev18_resources.py를 먼저 실행한다"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if project_commit is None:
        project_commit = _git_head()
    if gate_code_sha256 is None:
        gate_code_sha256 = file_sha256(Path(__file__))
    if input_manifest_sha256 is None:
        input_manifest_sha256 = file_sha256(
            REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
        )
    if budget_id is None:
        budget_id = _read_json(DEFAULT_BUDGET_PATH)["budget_id"]
    if cuda_device is None:
        cuda_device = collect_cuda_device_identity()
    if expected_models is None:
        expected_models = {spec["model"] for spec in _load_plan()[0]}
    if report.get("cuda_device") != cuda_device:
        raise ValueError("자원 gate를 통과한 GPU와 현재 GPU가 다르다")
    result_rows = report.get("results")
    valid = (
        report.get("status") == "passed"
        and report.get("project_commit") == project_commit
        and report.get("gate_code_sha256") == gate_code_sha256
        and report.get("input_manifest_sha256") == input_manifest_sha256
        and report.get("budget_id") == budget_id
        and 0 < report.get("maximum_memory_percent", 101) <= 80
        and set(report.get("checked_models", ())) == set(expected_models)
        and isinstance(result_rows, list)
        and bool(result_rows)
        and all(row.get("status") == "passed" for row in result_rows)
        and _has_required_tier3_evidence(result_rows, set(expected_models))
    )
    if not valid:
        raise ValueError("Dev18 자원 gate 결과가 현재 코드·입력·예산과 다르다")
    return report


def run_resource_check(
    *, data_root: Path, maximum_memory_percent: float = 80,
    output_path=DEFAULT_REPORT_PATH,
) -> dict:
    from src.common.execution_identity import file_sha256
    from tests.checks.run_lightning_dev18 import require_lightning_cuda
    from tests.checks.seal_runtime_environment import collect_cuda_device_identity
    from tests.ghl_main.run_dev18_tuning import _require_clean_worktree
    from tests.ghl_main.run_dev18_tuning import DEFAULT_BUDGET_PATH, _read_json

    require_lightning_cuda()
    _require_clean_worktree()
    specs, entries = _load_plan()
    cases = select_probe_cases(specs, entries)
    free_disk = shutil.disk_usage(REPOSITORY_ROOT).free
    results = [
        verify_input_files(entries, data_root),
        {
            "resource": "disk",
            "status": disk_status(free_disk),
            "free_bytes": free_disk,
            "minimum_bytes": MINIMUM_FREE_DISK_BYTES,
        },
    ] + [
        {
            "model": model,
            "status": "passed",
            "probe_scope": "배열 연산이며 GPU OOM 위험 없음",
        }
        for model in ("MWVAR", "SQDIFF_LAST3")
        if any(spec["model"] == model for spec in specs)
    ]
    results.append(_pca_static_check(specs, entries, maximum_memory_percent))
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
        completed = subprocess.run(command, capture_output=True, text=True)
        line = next(
            (value for value in reversed(completed.stdout.splitlines())
             if value.startswith(RESULT_PREFIX)),
            None,
        )
        if completed.returncode or line is None:
            results.append({
                **case,
                "status": "failed",
                "error": completed.stderr.strip() or completed.stdout.strip()
                or f"probe process exit {completed.returncode}",
            })
        else:
            results.append(json.loads(line.removeprefix(RESULT_PREFIX)))
    failed = [result for result in results if result["status"] != "passed"]
    failed_policies = ", ".join(
        result.get("model", result.get("resource", "unknown"))
        for result in failed
    )
    budget = _read_json(DEFAULT_BUDGET_PATH)
    report = {
        "status": "passed" if not failed else "failed",
        "project_commit": _git_head(),
        "gate_code_sha256": file_sha256(Path(__file__)),
        "input_manifest_sha256": file_sha256(
            REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
        ),
        "budget_id": budget["budget_id"],
        "cuda_device": collect_cuda_device_identity(),
        "maximum_memory_percent": maximum_memory_percent,
        "checked_models": sorted(
            {result["model"] for result in results if "model" in result}
        ),
        "results": results,
        "next_command": (
            "python tests/checks/run_lightning_dev18.py"
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
    parser.add_argument("--model", choices=GPU_PROBE_MODELS)
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
