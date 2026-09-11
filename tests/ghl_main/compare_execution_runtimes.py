"""기존 PCA 실측으로 비교 시간을 연결하며 실제 실행 시간은 보존한다."""

import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from functools import lru_cache
from statistics import median

from src.common.execution_evidence import FULL_PREFIX_TIMING_FIELDS
from tests.ghl_main.record_run_history import _load_histories


LEGACY_SINGLE_THREAD_BLOBS = (
    "061eeeaa78325718aab2362281d7b43ca0d65f12",
    "798bd4e8396c3bbcb31304ce145a6532580f79d0",
)


def _seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=32)
def _legacy_single_thread(commit, repository_root):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        return False
    try:
        blobs = subprocess.check_output([
            "git", "rev-parse", f"{commit}:src/models/tier1/pca_legacy.py",
            f"{commit}:tests/ghl_main/run_ratio_tuning.py",
        ], cwd=repository_root, text=True, stderr=subprocess.DEVNULL).splitlines()
        return tuple(blobs) == LEGACY_SINGLE_THREAD_BLOBS
    except (OSError, subprocess.CalledProcessError):
        return False


def _threads(snapshot, repository_root):
    resources = snapshot.get("execution_resources", {})
    requested = resources.get("pca_fit_blas_threads_requested")
    distance_workers = resources.get("pca_distance_workers_requested", 1)
    if (type(requested) is int and requested > 0
            and type(distance_workers) is int and distance_workers > 0):
        return (requested, distance_workers), "recorded_thread_policy"
    if _legacy_single_thread(snapshot.get("project_commit"), str(repository_root)):
        return (1, 1), "legacy_source_policy"
    return None, "unrecorded_thread_policy"


def _cohort(snapshot):
    spec = snapshot.get("spec", {})
    if (spec.get("model") != "PCA_LEGACY" or not spec.get("source_commit")
            or spec.get("common_recipe", {}).get("methodology_revision") != "paper_tuning_v4"
            or not snapshot.get("environment")):
        return None
    return _canonical({
        "source_commit": spec["source_commit"], "preprocess_recipe": spec.get("preprocess_recipe"),
        "common_recipe": spec["common_recipe"], "seed": spec.get("seed"),
        "environment": {key: value for key, value in snapshot["environment"].items()
                        if key != "runtime_snapshot"},
    })


def _input(snapshot):
    digest = snapshot.get("input_identity", {}).get("sha256")
    ranges = snapshot.get("source_ranges", {}).get("test_sessions")
    return _canonical([digest, ranges]) if digest and ranges else None


def load_runtime_references(directory, budget_id):
    """완료한 물리 실행의 작은 JSON만 읽는다. q·head 행은 늘리지 않는다."""
    references, seen = [], set()
    for path, content, record in _load_histories(directory, budget_id):
        if (record.get("status") != "complete" or record.get("model_execution_complete") is not True
                or record.get("run_id") in seen):
            continue
        timing = record.get("model_timing", {})
        if not all(_seconds(timing.get(field)) for field in FULL_PREFIX_TIMING_FIELDS):
            continue
        snapshot = record.get("run_snapshot")
        if not isinstance(snapshot, dict):
            continue
        seen.add(record["run_id"])
        references.append({
            "snapshot": snapshot, "runtime_seconds": sum(timing[field] for field in FULL_PREFIX_TIMING_FIELDS),
            "reference": {"file": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                          "run_id": record["run_id"]},
        })
    return references


def compare_execution_runtime(snapshot, runtime_seconds, references, *, repository_root):
    """실측 재사용, 같은 설정의 속도비, 기존 PCA 설정 간 시간비 순서로 연결한다."""
    result = {
        "actual_runtime_seconds": runtime_seconds,
        "comparison_runtime_seconds": runtime_seconds,
        "comparison_runtime_status": "measured",
        "comparison_runtime_basis": "recorded_execution",
        "comparison_runtime_sources": [],
        "comparison_runtime_details": {"scope": "sum_of_registered_model_timing_fields"},
    }
    spec = snapshot.get("spec", {})
    if spec.get("model") != "PCA_LEGACY":
        return result
    threads, thread_basis = _threads(snapshot, repository_root)
    details = result["comparison_runtime_details"]
    details.update(reference_pca_threads=1, actual_pca_threads=threads[0] if threads else None,
                   actual_pca_distance_workers_requested=threads[1] if threads else None, thread_basis=thread_basis,
                   hardware_basis="sealed_environment_and_user_stated_machine_continuity")
    if threads == (1, 1):
        result["comparison_runtime_basis"] = "single_thread_measurement"
        return result
    result.update(comparison_runtime_seconds=None, comparison_runtime_status="unavailable",
                  comparison_runtime_basis="no_matching_single_thread_evidence")
    cohort, target_input, target_config = _cohort(snapshot), _input(snapshot), spec.get("config_id")
    if cohort is None or target_input is None or not target_config:
        return result
    groups = defaultdict(list)
    for reference in references:
        source = reference["snapshot"]
        if _cohort(source) != cohort or not _seconds(reference["runtime_seconds"]):
            continue
        source_input = _input(source)
        source_threads, _ = _threads(source, repository_root)
        if source_input is not None and source_threads is not None:
            groups[(source_input, source["spec"].get("config_id"), source_threads)].append(reference)
    durations = {key: median(item["runtime_seconds"] for item in values) for key, values in groups.items()}

    def finish(value, basis, keys, predictions=()):
        used = {item["reference"]["run_id"]: item["reference"] for key in keys for item in groups[key]}
        result.update(comparison_runtime_seconds=value, comparison_runtime_basis=basis,
                      comparison_runtime_status="estimated" if predictions else "measured",
                      comparison_runtime_sources=list(used.values()))
        details["reference_run_count"] = len(used)
        if predictions:
            details.update(
                donor_input_count=len(predictions),
                reference_predictions=list(predictions),
                reference_prediction_min_seconds=min(item["predicted_seconds"] for item in predictions),
                reference_prediction_max_seconds=max(item["predicted_seconds"] for item in predictions),
                range_meaning="reference_predictions_not_a_confidence_interval",
                transfer_assumption="timing_ratios_transfer_across_inputs_not_guaranteed",
            )
        return result

    exact = (target_input, target_config, (1, 1))
    if exact in durations:
        return finish(durations[exact], "reused_single_thread_measurement", [exact])
    predictions, used_keys = [], []
    if threads is not None and threads != (1, 1) and _seconds(runtime_seconds):
        for key, serial in durations.items():
            parallel = (key[0], target_config, threads)
            if key[1:] == (target_config, (1, 1)) and durations.get(parallel, 0) > 0 and serial > 0:
                ratio = serial / durations[parallel]
                predictions.append({"input": key[0], "time_ratio": ratio,
                                    "predicted_seconds": runtime_seconds * ratio})
                used_keys.extend((key, parallel))
        if predictions:
            return finish(median(item["predicted_seconds"] for item in predictions),
                          "estimated_from_paired_pca_thread_timings", used_keys, predictions)

    anchors = [key for key in durations if key[0] == target_input and key[2] == (1, 1) and durations[key] > 0]
    for donor, target_seconds in durations.items():
        if donor[1:] != (target_config, (1, 1)) or donor[0] == target_input or target_seconds <= 0:
            continue
        transfers = []
        for anchor in anchors:
            partner = (donor[0], anchor[1], (1, 1))
            if durations.get(partner, 0) <= 0:
                continue
            ratio = target_seconds / durations[partner]
            transfers.append({"reference_config_id": anchor[1], "time_ratio": ratio,
                              "anchor_seconds": durations[anchor],
                              "predicted_seconds": durations[anchor] * ratio})
            used_keys.extend((donor, partner, anchor))
        if transfers:
            predictions.append({"input": donor[0], "configuration_transfers": transfers,
                                "predicted_seconds": median(item["predicted_seconds"] for item in transfers)})
    if predictions:
        # ponytail: 입력 간 비율 전이를 추정으로만 쓴다. 정밀도가 필요하면 같은 입력의 실측쌍을 추가한다.
        return finish(median(item["predicted_seconds"] for item in predictions),
                      "estimated_from_historical_pca_config_ratios", used_keys, predictions)
    return result


def summarize_runtime_comparisons(references, *, repository_root):
    """시도별 비교 근거를 내보내며 개발비 총액으로 합산하지 않는다."""
    return {
        "scope": "model_runtime_comparison_not_actual_hpo_billing",
        "runs": [{**item["reference"],
                  **{key: item["snapshot"].get("spec", {}).get(key)
                     for key in ("model", "series", "config_id", "ratio", "seed")},
                  **compare_execution_runtime(item["snapshot"], item["runtime_seconds"], references,
                                              repository_root=repository_root)} for item in references],
    }
