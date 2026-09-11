"""Dev18 exact panel을 실행하고 family-LOFO 선택표와 그림을 만든다."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
import numpy
from joblib import cpu_count as available_cpu_count

matplotlib.use("Agg")
from matplotlib import pyplot

from src.common.equal_trial_budget import build_equal_trial_budget, registry_space_sha256
from src.common.execution_identity import file_sha256, load_input_manifest_role
from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from src.common.measure_process_memory import start_process_memory_sampling
from src.common.model_registry import (
    load_model_registry_with_sha,
    validate_primary_hpo_seal,
)
from src.common.verify_run_context import verify_runtime_versions
from tests.checks.seal_runtime_environment import (
    collect_runtime_environment_identity,
    validate_runtime_snapshot,
)
from tests.ghl_main.build_dev18_budget import _load_current_feasibility
from tests.ghl_main.record_run_history import (
    has_run_persistence_failure, load_attempt_counts, record_run_history, record_run_stage, save_run_history,
)


def vus_pr(*args, **kwargs):
    from src.채점기.vus_pr import vus_pr as evaluate

    return evaluate(*args, **kwargs)


DEFAULT_DATA_ROOT = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project"
SNAPSHOT_DIRECTORY = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "snapshots"
    / "dev18_selection"
)
DEFAULT_BUDGET_PATH = SNAPSHOT_DIRECTORY / "dev18_budget_manifest.json"
DEFAULT_FEASIBILITY_LEDGER_PATH = SNAPSHOT_DIRECTORY / "dev18_feasibility_ledger.csv"
DEFAULT_ELL_MAX_PATH = SNAPSHOT_DIRECTORY / "dev18_ell_max.json"
DEFAULT_VUS_REPORT_PATH = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
    / "vus_pr" / "official_tsb_ad_comparison.json"
)
DEFAULT_RESULT_DIRECTORY = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "results"
    / "dev18_tuning"
)
DEFAULT_SCORE_MANIFEST_PATH = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "logs"
    / "dev18_score_manifest.csv"
)
DEFAULT_VUS_CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / ".runtime" / "dev18_vus_pr"
DEFAULT_ALLOCATOR_RECOVERY_PATH = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "logs"
    / "dev18_allocator_recovery.json"
)
DEFAULT_REAL_GATE_REPORT_PATH = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
    / "active_models" / "dev18_real_gate_smoke.json"
)
OFFICIAL_TSB_AD_COMMIT = "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48"
DEV18_RECOVERY_SOURCE_COMMIT = "501cb23cf0c02b9ebc9d94ca396d09e7049093d7"
DEV18_RECOVERY_PARENT_COMMIT = "6d5bcafda7a10fa6247f9cc32fe35d5061a286fa"
DEV18_RECOVERY_BUDGET_ID = "b5367ad431093"
DEV18_RECOVERY_RETRY_COUNT = 4
DEV18_RECOVERY_TRIAL_KEY = (
    "13", "GDN", "c1168c94d4dfc", "10", "0", "",
)
DEV18_RECOVERY_CHANGED_PATHS = frozenset({
    "docs/lead/lightning_studio.md",
    "docs/lead/next_session.md",
    "docs/lead/plan_v5.md",
    "docs/lead/process_0_preverify.md",
    "tests/checks/check_dev18_resources.py",
    "tests/checks/reset_lightning_dev18.py",
    "tests/checks/run_lightning_dev18.py",
    "tests/ghl_main/run_dev18_tuning.py",
    "tests/unit/test_dev18_tuning.py",
    "tests/unit/test_lightning_dev18.py",
    "tests/unit/test_lightning_dev18_resource_tools.py",
})
FINAL_SPLIT_ROLES = (
    "ghl25_final", "train1_to_test1", "train1_train2_to_test2",
)
RATIO_ADAPTIVE_SELECTION_RULE_ID = "tier_adaptive_family_lofo_v1"
EXPECTED_MODEL_RUNTIME_ORDER = {
    "SQDIFF_LAST3": 0,
    "SQDIFF_LAST1": 0,
    "SQDIFF_CENTERED5": 0,
    "MWVAR": 1,
    "MWVAR96_SQDIFF_LAST3": 1,
    "MWVAR96_SQDIFF_CENTERED5": 1,
    "PCA_LEGACY": 2,
    "TimeRCD": 3,
    "PaAno": 4,
    "GDN": 5,
    "TSPulse": 6,
}


def _execution_priority(spec: dict) -> tuple:
    return (
        EXPECTED_MODEL_RUNTIME_ORDER[spec["model"]],
        spec["ratio"],
        spec["hyperparameters"].get("embedding", 0),
        spec["config_id"],
        spec["seed"],
    )


def _series_execution_priority(entry: dict) -> tuple:
    return entry["row_count"] * entry["feature_count"], entry["order"]


def _read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_csv(path, rows, fieldnames=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"빈 표는 저장하지 않는다: {path.name}")
        fieldnames = tuple(rows[0])
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _require_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError("Dev18 tuning은 봉인된 clean worktree에서만 실행한다")


def validate_vus_evidence(report_path, evaluator_path) -> dict:
    """공식 TSB-AD 대조 보고서를 현재 evaluator에 묶는다."""
    report = _read_json(report_path)
    evaluator_sha256 = file_sha256(evaluator_path)
    valid = (
        report.get("status") == "passed"
        and report.get("official", {}).get("commit") == OFFICIAL_TSB_AD_COMMIT
        and report.get("official", {}).get("version") == "opt"
        and report.get("n_thresholds") == 250
        and report.get("evaluator_sha256") == evaluator_sha256
        and report.get("maximum_absolute_difference", float("inf"))
        <= report.get("absolute_tolerance", -1)
    )
    if not valid:
        raise ValueError("공식 VUS-PR 대조 보고서와 현재 evaluator가 다르다")
    return report


def _validate_ell_max(path, input_manifest_sha256: str) -> dict:
    snapshot = _read_json(path)
    rows = snapshot.get("series")
    payload = {key: value for key, value in snapshot.items() if key != "ell_max_id"}
    expected_id = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()
    if (
        snapshot.get("series_count") != 18
        or snapshot.get("input_manifest_sha256") != input_manifest_sha256
        or snapshot.get("ell_max_id") != expected_id
        or not isinstance(rows, list)
        or [row.get("series") for row in rows] != [f"{index:02d}" for index in range(1, 19)]
        or any(type(row.get("l_max_samples")) is not int or row["l_max_samples"] < 0 for row in rows)
    ):
        raise ValueError("Dev18 ell_max snapshot이 현재 입력과 다르다")
    generator_path = REPOSITORY_ROOT / "tests" / "ghl_main" / "build_dev18_ell_max.py"
    if snapshot.get("generator_sha256") != file_sha256(generator_path):
        raise ValueError("Dev18 ell_max generator SHA-256이 현재 코드와 다르다")
    return snapshot


def _validate_checkpoint_report(model_name: str, registry: dict, budget: dict) -> dict:
    directory = "time_rcd" if model_name == "TimeRCD" else "tspulse"
    path = (
        REPOSITORY_ROOT / ".runtime" / "dev18_checkpoint_smoke"
        / directory / "dev18_checkpoint_smoke.json"
    )
    if budget.get("experiment_mode") == "full_prefix_v2":
        path = (REPOSITORY_ROOT / "experiments/checks/reference_code/active_models"
                / "full_prefix_v2/checkpoints" / directory / "dev18_checkpoint_smoke.json")
    report = _read_json(path)
    model = registry["models"][model_name]
    input_manifest_sha256 = file_sha256(
        REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
    )
    adapter_path = (
        REPOSITORY_ROOT / "src" / "models" / "tier3"
        / ("time_rcd.py" if model_name == "TimeRCD" else "tspulse.py")
    )
    official_protocol = registry["common_recipe"].get("methodology_revision") == "paper_tuning_v4"
    expected_heads = (
        {"score"} if model_name == "TimeRCD"
        else {"time", "fft", "pred", "ensemble"} if official_protocol
        else {"time", "fft", "pred", "raw_max"}
    )
    channel_report = report.get("channels", {}).get("2", {})
    score_reports = channel_report.get("scores", {})
    expected_config_ids = {
        row["config_id"] for row in budget["execution_panel"]
        if row["model"] == model_name
    }
    valid = (
        report.get("status") == "passed"
        and report.get("model") == model_name
        and report.get("official_protocol", False) == official_protocol
        and report.get("inference_context_normalization", False)
        == (registry["common_recipe"].get("methodology_revision") == "source_faithful_v3")
        and report.get("config_id") in expected_config_ids
        and report.get("budget_id") == budget["budget_id"]
        and report.get("budget_sha256") == budget["budget_sha256"]
        and report.get("registry_space_sha256") == registry_space_sha256(registry)
        and report.get("source", {}).get("commit") == model["source_commit"]
        and report.get("checkpoint", {}).get("sha256")
        == model["source_checkpoint_sha256"]
        and report.get("config", {}).get("sha256")
        == model["checkpoint_config_sha256"]
        and report.get("input", {}).get("uses_labels") is False
        and report.get("input", {}).get("uses_test") is False
        and report.get("input", {}).get("uses_normal_training") is True
        and report.get("input", {}).get("input_manifest_sha256") == input_manifest_sha256
        and report.get("input", {}).get("slice") == [0, 1536]
        and report.get("input", {}).get("shape") == [1536, 2]
        and report.get("smoke_code_sha256")
        == file_sha256(REPOSITORY_ROOT / "tests" / "checks" / "run_checkpoint_smoke.py")
        and report.get("adapter_sha256") == file_sha256(adapter_path)
        and channel_report.get("status") == "passed"
        and set(score_reports) == expected_heads
        and all(
            score.get("shape") == [1536]
            and score.get("finite") is True
            and score.get("peak_to_peak", 0) > 0
            and score.get("cpu_deterministic") is True
            and score.get("maximum_absolute_repeat_difference") == 0
            for score in score_reports.values()
        )
    )
    if not valid:
        raise ValueError(f"{model_name} checkpoint smoke 증거가 현재 봉인과 다르다")
    return report


def _validate_environment_snapshot(environment: dict) -> dict:
    return validate_runtime_snapshot(
        environment, repository_root=REPOSITORY_ROOT,
    )


def prepare_tuning(
    *, data_root=DEFAULT_DATA_ROOT, require_clean=True,
    require_execution_environment=True, budget=None,
) -> dict:
    """기존 봉인을 다시 계산하지 않고 tuning 시작 조건만 대조한다."""
    if require_clean:
        _require_clean_worktree()
    registry, registry_sha256 = load_model_registry_with_sha()
    full_prefix = budget is not None and budget.get("experiment_mode") == "full_prefix_v2"
    if full_prefix:
        validate_primary_hpo_seal(registry, budget=budget)
    else:
        validate_primary_hpo_seal(registry)
        _, _, feasibility = _load_current_feasibility(
            REPOSITORY_ROOT, SNAPSHOT_DIRECTORY, registry_sha256,
            config_count=sum(len(model["candidates"]) for model in registry["models"].values()),
        )
        expected_budget = build_equal_trial_budget(registry, feasibility)
        budget = _read_json(DEFAULT_BUDGET_PATH)
        for field, value in expected_budget.items():
            if budget.get(field) != value:
                raise ValueError(f"Dev18 budget {field}가 현재 봉인과 다르다")
        if (
            budget.get("seal_status") != "sealed"
            or budget.get("execution_readiness_status") != "ready"
            or budget.get("pending_execution_models") != []
            or budget.get("attestation", {}).get("config_registry_sha256") != registry_sha256
        ):
            raise ValueError("Dev18 budget 실행 준비가 닫히지 않았다")

    input_manifest_path = REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
    input_manifest_sha256 = file_sha256(input_manifest_path)
    if full_prefix and (
        budget["input_manifest_sha256"] != input_manifest_sha256
        or budget["data_preprocessing_sha256"] != file_sha256(REPOSITORY_ROOT / "configs/data_preprocessing.yaml")
    ):
        raise ValueError("full-prefix budget의 입력 또는 전처리 신원이 바뀌었다")
    ell_max = _validate_ell_max(DEFAULT_ELL_MAX_PATH, input_manifest_sha256)
    evaluator_path = REPOSITORY_ROOT / "src" / "채점기" / "vus_pr.py"
    vus_report = validate_vus_evidence(DEFAULT_VUS_REPORT_PATH, evaluator_path)
    environment_config = __import__("yaml").safe_load((
        REPOSITORY_ROOT / "configs" / "environment.yaml"
    ).read_text(encoding="utf-8"))
    verify_runtime_versions(environment_config)
    checkpoints = {}
    environment = {}
    if require_execution_environment:
        if full_prefix:
            from tests.checks.run_model_smoke import validate_full_prefix_smoke

            validate_full_prefix_smoke()
        checkpoints = {
            model: _validate_checkpoint_report(model, registry, budget)
            for model in ("TimeRCD", "TSPulse")
        }
        environment = collect_runtime_environment_identity()
        runtime_snapshot = _validate_environment_snapshot(environment)
        environment = {**environment, "runtime_snapshot": runtime_snapshot}

    data_root = Path(data_root)
    if not (data_root / "tuning").is_dir():
        raise FileNotFoundError(f"Dev18 tuning 폴더가 없다: {data_root / 'tuning'}")
    return {
        "status": "ready",
        "budget_id": budget["budget_id"],
        "physical_execution_count_per_series": budget["physical_execution_count"],
        "series_count": 18,
        "expected_primary_ledger_rows": budget.get("expected_ledger_rows", 18 * budget["primary_logical_score_row_count"]),
        "evaluator_sha256": vus_report["evaluator_sha256"],
        "ell_max_id": ell_max["ell_max_id"],
        "checkpoint_models": sorted(checkpoints),
        "environment": environment,
        "execution_environment_verified": require_execution_environment,
        "package_versions_verified": True,
    }


def smoke_real_data_gate(
    *, data_root=DEFAULT_DATA_ROOT, series="03", sample_count=160,
    output_path=DEFAULT_REAL_GATE_REPORT_PATH, require_clean=True,
) -> dict:
    """실제 Dev18의 짧은 label-free 조각으로 열린 batch gate를 통과한다."""
    if require_clean:
        _require_clean_worktree()
    if type(sample_count) is not int or sample_count < 96:
        raise ValueError("MWVAR 실데이터 smoke에는 최소 96개 시점이 필요하다")
    budget = _read_json(DEFAULT_BUDGET_PATH)
    panel = next(
        row for row in budget["execution_panel"]
        if row["model"] == "MWVAR"
    )
    feasible_key = {
        (panel["model"], panel["config_id"], panel["physical_ratio"]),
    }
    from src.common.run_registered_model import execute_registered_model
    from tests.ghl_main.run_registered_models import load_registered_inputs, run_batch

    def execute(spec):
        inputs = load_registered_inputs(
            spec=spec,
            input_manifest_path=REPOSITORY_ROOT / "configs" / "input_manifest.yaml",
            series=series, data_root=data_root,
        )
        test = inputs["test_sessions"][0][:sample_count]
        result = execute_registered_model(
            spec,
            normal_training=inputs["normal_training"][:sample_count],
            test_sessions=(test,), device="cpu",
        )
        output = result["test_outputs"][0]
        scores = numpy.asarray(output["scores"])
        if (
            scores.ndim not in (1, 2) or len(scores) != sample_count
            or not numpy.isfinite(scores).all()
        ):
            raise ValueError("MWVAR 실데이터 smoke score가 길이·유한값 계약을 어겼다")
        aggregated = scores.max(axis=1) if scores.ndim == 2 else scores
        return {
            "model": spec["model"], "config_id": spec["config_id"],
            "input_sha256": inputs["input_identity"]["verified_files"][0]["sha256"],
            "input_shape": [sample_count, test.shape[1]],
            "native_score_shape": list(scores.shape),
            "aggregated_score_shape": list(aggregated.shape),
            "score_peak_to_peak": float(numpy.ptp(aggregated)),
            "uses_labels": False,
        }

    results = run_batch(
        dataset_role="development", allow_real_data=True,
        feasible_keys=feasible_key, executor=execute,
    )
    if len(results) != 1:
        raise ValueError("실데이터 smoke는 MWVAR 물리 실행 하나여야 한다")
    report = {
        "status": "passed", "gate": "real_data_execution",
        "series": series, "sample_count": sample_count,
        "budget_id": budget["budget_id"], "result": results[0],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report


def _seed_means(rows) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "complete":
            raise ValueError("선택 입력에는 complete 점수만 있어야 한다")
        key = tuple(row[field] for field in (
            "series", "family", "tier", "model", "config_id", "ratio", "score_variant",
        ))
        groups[key].append(float(row["vus_pr"]))
    return [
        {
            **dict(zip((
                "series", "family", "tier", "model", "config_id", "ratio", "score_variant",
            ), key)),
            "vus_pr": float(numpy.mean(values)),
        }
        for key, values in sorted(groups.items())
    ]


TRIAL_SCORE_LEDGER_FIELDS = (
    "series", "family", "tier", "model", "config_id", "ratio", "seed",
    "score_variant", "normalization", "vus_pr", "score_file", "score_sha256",
    "evaluator_sha256", "ell_max_id", "status", "status_reason",
)


def load_trial_score_ledger(
    path, budget: dict, *, expected_evaluator_sha256: str, expected_ell_max_id: str,
    allow_partial=False,
) -> list[dict]:
    """완료 원표의 점수·채점 신원과 budget 논리 키를 확인해 읽는다."""
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != TRIAL_SCORE_LEDGER_FIELDS:
            raise ValueError("Dev18 trial ledger header가 다르다")
        rows = []
        for row in reader:
            if row["status"] != "complete":
                raise ValueError("Dev18 trial ledger에는 complete 행만 있어야 한다")
            vus_pr_value = float(row["vus_pr"])
            if not math.isfinite(vus_pr_value) or not 0 <= vus_pr_value <= 1:
                raise ValueError("Dev18 trial ledger의 VUS-PR은 유한한 [0, 1] 값이어야 한다")
            rows.append({**row, "ratio": int(row["ratio"]), "seed": int(row["seed"]),
                         "vus_pr": vus_pr_value})

    keys = [
        (row["series"], row["model"], row["config_id"], row["ratio"],
         row["seed"], row["score_variant"])
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Dev18 trial ledger에 duplicate 논리 키가 있다")
    if any(row["evaluator_sha256"] != expected_evaluator_sha256 for row in rows):
        raise ValueError("Dev18 trial ledger의 evaluator SHA가 공식 봉인과 다르다")
    if any(row["ell_max_id"] != expected_ell_max_id for row in rows):
        raise ValueError("Dev18 trial ledger의 ell_max ID가 공식 봉인과 다르다")
    series_families = {}
    tiers_by_model = {panel["model"]: panel["tier"] for panel in budget["model_panels"]}
    for row in rows:
        family = series_families.setdefault(row["series"], row["family"])
        if family != row["family"]:
            raise ValueError("Dev18 trial ledger의 series와 family 연결이 하나가 아니다")
        if tiers_by_model.get(row["model"]) != row["tier"]:
            raise ValueError("Dev18 trial ledger model의 tier가 budget과 다르다")
    if budget.get("budget_id") == DEV18_RECOVERY_BUDGET_ID and len({row["series"] for row in rows}) != 18:
        raise ValueError("Dev18 production trial ledger는 18개 series여야 한다")
    if budget.get("budget_id") == DEV18_RECOVERY_BUDGET_ID and len(set(series_families.values())) != 10:
        raise ValueError("Dev18 production trial ledger는 10개 family여야 한다")

    expected_by_model = {
        panel["model"]: {
            (config_id, ratio, seed, variant)
            for ratio in panel["logical_ratios"]
            for config_id in panel.get("selected_config_ids_by_ratio", {}).get(
                str(ratio), panel["selected_config_ids"],
            )
            for seed in panel.get("seeds", budget["seeds"])
            for variant in panel["primary_score_variants"]
        }
        for panel in budget["model_panels"]
    }
    expected_keys = {
        (series, model, config_id, ratio, seed, variant)
        for series in budget.get("series_ids", {row["series"] for row in rows})
        for model, entries in expected_by_model.items()
        for config_id, ratio, seed, variant in entries
    }
    if budget.get("experiment_mode") == "full_prefix_v2":
        expected_keys = {
            (series, panel["model"], panel["config_id"], ratio, panel["seed"], variant)
            for panel in budget["execution_panel"]
            for series in panel["series_ids"]
            for ratio in panel["logical_ratios"]
            for variant in panel["primary_score_variants"]
        }
    if not set(keys) <= expected_keys or (not allow_partial and set(keys) != expected_keys):
        raise ValueError("Dev18 trial ledger의 논리 키가 budget과 다르다")
    return rows


def _candidate_score(
    rows, *, model: str, config_id: str, score_variant: str,
    ratios, included_families=None,
) -> float:
    ratios = set(ratios)
    family_values = defaultdict(list)
    for row in rows:
        if (
            row["model"] == model
            and row["config_id"] == config_id
            and row["score_variant"] == score_variant
            and row["ratio"] in ratios
            and (included_families is None or row["family"] in included_families)
        ):
            family_values[row["family"]].append(row["vus_pr"])
    expected_families = (
        set(included_families) if included_families is not None
        else {row["family"] for row in rows if row["model"] == model}
    )
    if not expected_families or set(family_values) != expected_families:
        raise ValueError(f"{model}/{config_id}의 family 점수가 불완전하다")
    return float(numpy.mean([
        numpy.mean(family_values[family]) for family in sorted(family_values)
    ]))


def _pick(candidates, tolerance: float):
    if not candidates:
        raise ValueError("선택 후보가 비었다")
    best_score = max(candidate[0] for candidate in candidates)
    tied = [candidate for candidate in candidates if best_score - candidate[0] <= tolerance]
    return min(tied, key=lambda candidate: candidate[1:])


def _hyperparameters(registry: dict, model_name: str, config_id: str) -> dict:
    candidates = [
        candidate["hyperparameters"]
        for candidate in registry["models"][model_name]["candidates"]
        if candidate["config_id"] == config_id
    ]
    if len(candidates) != 1:
        raise ValueError(f"registry config가 유일하지 않다: {model_name}/{config_id}")
    return candidates[0]


def build_ratio_adaptive_plot_data(rows, selection: dict) -> dict:
    """최종 그림에 필요한 adaptive 경로와 PCA 참고값만 만든다."""
    pca_policies = [
        row for row in selection["model_fixed"]
        if row["model"] == "PCA_LEGACY" and row["selection_status"] == "selected"
    ]
    if len(pca_policies) != 1:
        raise ValueError("PCA_LEGACY model-fixed 정책이 하나여야 한다")
    pca_policy = pca_policies[0]
    pca_reference = _candidate_score(
        _seed_means(rows), model="PCA_LEGACY",
        config_id=pca_policy["config_id"],
        score_variant=pca_policy["score_variant"], ratios=[100],
    )

    points = []
    unavailable = []
    for policy in selection["tier_adaptive"]:
        if policy["selection_status"] == "selected":
            points.append({
                "tier": policy["tier"], "ratio": policy["ratio"],
                "model": policy["selected_model"],
                "selection_score": float(policy["selection_score"]),
            })
        elif policy["selection_status"] == "unavailable":
            unavailable.append({"tier": policy["tier"], "ratio": policy["ratio"]})
        else:
            raise ValueError("tier_adaptive 상태는 selected 또는 unavailable이어야 한다")
    return {
        "points": sorted(points, key=lambda row: (row["tier"], row["ratio"])),
        "unavailable": sorted(
            unavailable, key=lambda row: (row["tier"], row["ratio"]),
        ),
        "pca_reference_vus_pr": pca_reference,
    }


def _draw_ratio_adaptive_selection(axis, data: dict) -> None:
    colors = {"t1": "#1f77b4", "t2": "#d95f02", "t3": "#2ca02c"}
    label_offsets = {"t1": 8, "t2": 8, "t3": -12}
    for tier in ("t1", "t2", "t3"):
        points = [row for row in data["points"] if row["tier"] == tier]
        axis.plot(
            [row["ratio"] for row in points],
            [row["selection_score"] for row in points],
            color=colors[tier], marker="o", markersize=4.5, linewidth=1.8,
            label=f"Tier {tier[1:]}",
        )
        changes = [
            point for index, point in enumerate(points)
            if index == 0 or point["model"] != points[index - 1]["model"]
        ]
        labels = [points[-1]] if len(changes) == 1 and points else changes
        for point in labels:
            axis.annotate(
                point["model"],
                (point["ratio"], point["selection_score"]),
                xytext=(0, label_offsets[tier]), textcoords="offset points",
                ha="center", va="bottom" if label_offsets[tier] > 0 else "top",
                fontsize=7, color=colors[tier],
            )
    axis.axhline(
        data["pca_reference_vus_pr"], color="#7f7f7f", linestyle="--",
        linewidth=1.1, label="PCA q100 reference",
    )
    for point in data["unavailable"]:
        axis.text(
            point["ratio"], 0.03, f"Tier {point['tier'][1:]} unavailable",
            transform=axis.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7, color="#7f7f7f",
        )
    axis.set(
        title="Adaptive Tier representatives",
        xlabel="Observed normal prefix (%)", ylabel="Family-LOFO VUS-PR",
        xticks=SUPPORTED_RATIO_PERCENTS,
    )
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.margins(x=0.02, y=0.14)
    axis.legend(frameon=False, fontsize=8, ncol=2)


def select_ratio_adaptive_policies(
    rows, registry: dict, budget: dict, *, model_fixed, evaluator_sha256: str,
) -> dict:
    """모델별 고정 recipe를 유지한 채 비율마다 Tier 대표를 고른다."""
    seed_rows = _seed_means(rows)
    tolerance = float(budget["tie_rule"]["tolerance"])
    fixed_by_model = {row["model"]: row for row in model_fixed}
    panels_by_tier = defaultdict(list)
    for panel in budget["model_panels"]:
        panels_by_tier[panel["tier"]].append(panel)

    adaptive_lofo = []
    model_ratio_scores = defaultdict(lambda: defaultdict(list))
    for panel in budget["model_panels"]:
        model_name = panel["model"]
        fixed = fixed_by_model[model_name]
        if (
            model_name == "PCA_LEGACY"
            or fixed["selection_status"] != "selected"
            or not panel["selected_config_ids"]
        ):
            continue
        families = sorted({
            row["family"] for row in seed_rows if row["model"] == model_name
        })
        for holdout in families:
            training_families = set(families) - {holdout}
            fold_candidates = [
                (
                    _candidate_score(
                        seed_rows, model=model_name, config_id=config_id,
                        score_variant=variant, ratios=panel["logical_ratios"],
                        included_families=training_families,
                    ), model_name, config_id, variant,
                )
                for config_id in panel["selected_config_ids"]
                for variant in panel["primary_score_variants"]
            ]
            _, _, config_id, variant = _pick(fold_candidates, tolerance)
            for ratio in panel["logical_ratios"]:
                holdout_score = _candidate_score(
                    seed_rows, model=model_name, config_id=config_id,
                    score_variant=variant, ratios=[ratio], included_families={holdout},
                )
                model_ratio_scores[model_name][ratio].append(holdout_score)
                adaptive_lofo.append({
                    "tier": panel["tier"], "ratio": ratio, "model": model_name,
                    "holdout_family": holdout, "selected_config_id": config_id,
                    "score_variant": variant, "holdout_vus_pr": holdout_score,
                })

    tier_adaptive = []
    candidate_audit = []
    for tier, panels in sorted(panels_by_tier.items()):
        for ratio in SUPPORTED_RATIO_PERCENTS:
            audit_by_model = {}
            candidates = []
            for panel in sorted(panels, key=lambda item: item["model"]):
                model_name = panel["model"]
                fixed = fixed_by_model[model_name]
                audit = {
                    "tier": tier, "ratio": ratio, "model": model_name,
                    "config_id": fixed["config_id"],
                    "score_variant": fixed["score_variant"],
                    "eligibility": "eligible", "family_lofo_vus_pr": None,
                    "selected": False, "reason_code": "not_selected",
                    "reason": "tolerance와 결정적 동률 규칙을 적용해 미선택",
                }
                if model_name == "PCA_LEGACY":
                    audit.update({
                        "eligibility": "reference_only", "reason_code": "reference_only",
                        "reason": "PCA_LEGACY는 adaptive 대표 후보가 아닌 참고선이다",
                    })
                elif fixed["selection_status"] != "selected":
                    audit.update({
                        "eligibility": "full_panel_config_unavailable",
                        "reason_code": "full_panel_config_unavailable",
                        "reason": "18개 panel을 덮는 model-fixed config가 없다",
                    })
                elif ratio not in panel["logical_ratios"]:
                    audit.update({
                        "eligibility": "ratio_unsupported", "reason_code": "ratio_unsupported",
                        "reason": "이 비율은 모델의 실행 지원 범위 밖이다",
                    })
                else:
                    score = float(numpy.mean(model_ratio_scores[model_name][ratio]))
                    audit["family_lofo_vus_pr"] = score
                    candidates.append((
                        score, model_name, fixed["config_id"], fixed["score_variant"],
                    ))
                audit_by_model[model_name] = audit
                candidate_audit.append(audit)

            if not candidates:
                tier_adaptive.append({
                    "tier": tier, "ratio": ratio, "selected_model": "",
                    "config_id": "", "score_variant": "", "hyperparameters": {},
                    "q_floor": registry["selection"]["tier_q_floor"].get(tier),
                    "selection_score": None, "selection_status": "unavailable",
                    "selection_reason": "이 비율에서 실행 가능한 Tier 후보가 없다",
                    "budget_id": budget["budget_id"], "evaluator_sha256": evaluator_sha256,
                })
                continue
            score, selected_model, _, _ = _pick(candidates, tolerance)
            fixed = fixed_by_model[selected_model]
            audit_by_model[selected_model].update({
                "selected": True, "reason_code": "selected",
                "reason": "family-LOFO 점수가 동률 규칙을 적용한 후보 중 최고다",
            })
            tier_adaptive.append({
                "tier": tier, "ratio": ratio, "selected_model": selected_model,
                "config_id": fixed["config_id"],
                "score_variant": fixed["score_variant"],
                "hyperparameters": fixed["hyperparameters"],
                "q_floor": registry["selection"]["tier_q_floor"].get(tier),
                "selection_score": score, "selection_status": "selected",
                "selection_reason": (
                    f"family-LOFO 점수 {score:.6f}가 {tier} 후보 중 가장 높아 선택"
                ),
                "budget_id": budget["budget_id"], "evaluator_sha256": evaluator_sha256,
                "source_commit": fixed["source_commit"],
                "source_checkpoint_sha256": fixed["source_checkpoint_sha256"],
            })

    policy_transitions = []
    for tier, policies in sorted({
        tier: [row for row in tier_adaptive if row["tier"] == tier]
        for tier in panels_by_tier
    }.items()):
        previous_selected = None
        for policy in sorted(policies, key=lambda row: row["ratio"]):
            if policy["selection_status"] == "unavailable":
                transition = "unavailable"
            elif previous_selected is None:
                transition = "initial"
            elif (
                policy["selected_model"] == previous_selected["selected_model"]
                and policy["config_id"] == previous_selected["config_id"]
            ):
                transition = "keep"
            else:
                transition = "switch"
            transition_row = {
                "tier": tier,
                "previous_ratio": previous_selected["ratio"] if previous_selected else None,
                "current_ratio": policy["ratio"],
                "previous_model": previous_selected["selected_model"] if previous_selected else "",
                "previous_config_id": previous_selected["config_id"] if previous_selected else "",
                "current_model": policy["selected_model"],
                "current_config_id": policy["config_id"],
                "transition": transition,
            }
            transition_row["transition_key"] = "tr" + hashlib.sha256(
                _json(transition_row).encode("utf-8")
            ).hexdigest()[:12]
            policy_transitions.append(transition_row)
            if policy["selection_status"] == "selected":
                previous_selected = policy

    return {
        "tier_adaptive": tier_adaptive, "adaptive_lofo": adaptive_lofo,
        "candidate_audit": candidate_audit, "policy_transitions": policy_transitions,
    }


def select_tuning_policies(
    rows, registry: dict, budget: dict, *, evaluator_sha256: str,
) -> dict:
    """seed 평균 → family 균형 → LOFO 모델 선택 → full-panel recipe 고정을 수행한다."""
    seed_rows = _seed_means(rows)
    tolerance = float(budget["tie_rule"]["tolerance"])
    model_fixed = []
    fixed_by_model = {}
    for panel in budget["model_panels"]:
        model_name = panel["model"]
        model = registry["models"][model_name]
        if not panel["selected_config_ids"]:
            policy = {
                "tier": panel["tier"], "model": model_name, "q_support": [],
                "config_id": "", "hyperparameters": {}, "j_fixed": None,
                "score_variant": "", "hpo_regime": budget["primary_hpo_regime"],
                "budget_id": budget["budget_id"], "source_commit": model["source_commit"],
                "source_checkpoint_sha256": model["source_checkpoint_sha256"],
                "selection_status": "unavailable",
                "selection_reason": "Dev18 exact panel에서 공통 실행 가능한 config가 없음",
            }
        else:
            candidates = []
            for config_id in panel["selected_config_ids"]:
                for variant in panel["primary_score_variants"]:
                    candidates.append((
                        _candidate_score(
                            seed_rows, model=model_name, config_id=config_id,
                            score_variant=variant, ratios=panel["logical_ratios"],
                        ), model_name, config_id, variant,
                    ))
            score, _, config_id, variant = _pick(candidates, tolerance)
            policy = {
                "tier": panel["tier"], "model": model_name,
                "q_support": list(panel["logical_ratios"]), "config_id": config_id,
                "hyperparameters": _hyperparameters(registry, model_name, config_id),
                "j_fixed": score, "score_variant": variant,
                "hpo_regime": budget["primary_hpo_regime"],
                "budget_id": budget["budget_id"], "source_commit": model["source_commit"],
                "source_checkpoint_sha256": model["source_checkpoint_sha256"],
                "selection_status": "selected",
                "selection_reason": (
                    f"18개 패널의 seed 평균·family 동일 가중 VUS-PR {score:.6f}가 "
                    "동률 규칙을 적용한 model-fixed 후보 중 최고"
                ),
            }
        model_fixed.append(policy)
        fixed_by_model[model_name] = policy

    tier_fixed = []
    lofo = []
    tiers = sorted({panel["tier"] for panel in budget["model_panels"]})
    for tier in tiers:
        panels = [
            panel for panel in budget["model_panels"]
            if panel["tier"] == tier and panel["dev18_tier_representative_eligible"]
        ]
        if not panels:
            raise ValueError(f"{tier}의 대표 가능 모델이 없다")
        common_ratios = sorted(set.intersection(*(
            set(panel["logical_ratios"]) for panel in panels
        )))
        families = sorted({
            row["family"] for row in seed_rows
            if row["model"] in {panel["model"] for panel in panels}
        })
        model_scores = []
        for panel in panels:
            holdout_scores = []
            for holdout in families:
                training_families = set(families) - {holdout}
                candidates = []
                for config_id in panel["selected_config_ids"]:
                    for variant in panel["primary_score_variants"]:
                        candidates.append((
                            _candidate_score(
                                seed_rows, model=panel["model"], config_id=config_id,
                                score_variant=variant, ratios=common_ratios,
                                included_families=training_families,
                            ), panel["model"], config_id, variant,
                        ))
                _, _, config_id, variant = _pick(candidates, tolerance)
                holdout_score = _candidate_score(
                    seed_rows, model=panel["model"], config_id=config_id,
                    score_variant=variant, ratios=common_ratios,
                    included_families={holdout},
                )
                holdout_scores.append(holdout_score)
                lofo.append({
                    "tier": tier, "model": panel["model"], "holdout_family": holdout,
                    "selected_config_id": config_id, "score_variant": variant,
                    "holdout_vus_pr": holdout_score,
                })
            selection_score = float(numpy.mean(holdout_scores))
            fixed = fixed_by_model[panel["model"]]
            model_scores.append((
                selection_score, panel["model"], fixed["config_id"], fixed["score_variant"],
            ))
        selection_score, selected_model, _, _ = _pick(model_scores, tolerance)
        fixed = fixed_by_model[selected_model]
        j_tier = _candidate_score(
            seed_rows, model=selected_model, config_id=fixed["config_id"],
            score_variant=fixed["score_variant"], ratios=common_ratios,
        )
        tier_fixed.append({
            "tier": tier, "selected_model": selected_model,
            "config_id": fixed["config_id"], "hyperparameters": fixed["hyperparameters"],
            "q_floor": registry["selection"]["tier_q_floor"][tier],
            "selection_q_common": common_ratios,
            "evaluation_q_support": {
                split_role: common_ratios for split_role in FINAL_SPLIT_ROLES
            },
            "score_variant": fixed["score_variant"],
            "selection_score": selection_score, "j_tier": j_tier,
            "hpo_regime": budget["primary_hpo_regime"],
            "budget_id": budget["budget_id"], "evaluator_sha256": evaluator_sha256,
            "source_commit": fixed["source_commit"],
            "source_checkpoint_sha256": fixed["source_checkpoint_sha256"],
            "selection_status": "selected",
            "selection_reason": (
                f"{len(families)}개 family-LOFO 점수가 {selection_score:.6f}로 "
                f"{tier} 후보 중 가장 높아 선택"
            ),
        })
    adaptive = select_ratio_adaptive_policies(
        rows, registry, budget, model_fixed=model_fixed,
        evaluator_sha256=evaluator_sha256,
    )
    return {"model_fixed": model_fixed, "tier_fixed": tier_fixed, "lofo": lofo, **adaptive}


def build_final_membership_rows(
    selection: dict, registry: dict, *, ratios=SUPPORTED_RATIO_PERCENTS,
    split_roles=FINAL_SPLIT_ROLES,
) -> list[dict]:
    """고정·비율별 정책을 final runner가 소비하는 한 CSV 행으로 펼친다."""
    rows = []
    target_free = {"training_free", "strict_zero_shot"}

    def append(
        kind, split_role, policy, model_name, evaluation_ratio, *,
        supported=None, unavailable_reason=None,
    ):
        model = registry["models"][model_name]
        if supported is None:
            supported = evaluation_ratio in policy.get(
                "q_support", policy.get("selection_q_common", []),
            )
        selected = policy.get("selection_status") == "selected" and supported
        rows.append({
            "analysis_kind": kind, "split_role": split_role, "tier": model["tier"],
            "model": model_name, "evaluation_ratio": evaluation_ratio,
            "physical_ratio": (
                100 if selected and model["target_use"] in target_free
                else evaluation_ratio if selected else ""
            ),
            "config_id": policy.get("config_id") or model["candidates"][0]["config_id"],
            "score_variant": policy.get("score_variant", ""),
            "status": "runnable" if selected else "unavailable",
            "status_reason": "" if selected else (
                unavailable_reason or "Dev18 고정 정책에서 지원하지 않는 비율"
            ),
        })

    fixed_by_tier = {policy["tier"]: policy for policy in selection["tier_fixed"]}
    adaptive_by_key = {
        (policy["tier"], policy["ratio"]): policy
        for policy in selection["tier_adaptive"]
    }
    for split_role in split_roles:
        for policy in selection.get("model_ratio", []):
            append("model_ratio", split_role, policy, policy["model"], policy["ratio"],
                   unavailable_reason=policy.get("selection_reason"))
        for policy in selection["model_fixed"]:
            for ratio in ratios:
                append("model_fixed", split_role, policy, policy["model"], ratio)
        for policy in selection["tier_fixed"]:
            for ratio in ratios:
                append("tier_fixed", split_role, policy, policy["selected_model"], ratio)
        for tier, fixed in fixed_by_tier.items():
            for ratio in ratios:
                policy = adaptive_by_key[(tier, ratio)]
                selected = policy["selection_status"] == "selected"
                model_name = policy["selected_model"] if selected else fixed["selected_model"]
                membership_policy = policy if selected else {
                    **policy,
                    "config_id": fixed["config_id"],
                    "score_variant": fixed.get("score_variant", ""),
                }
                append(
                    "tier_adaptive", split_role, membership_policy, model_name, ratio,
                    supported=selected,
                    unavailable_reason=policy["selection_reason"],
                )
    return rows


def _serializable_rows(rows):
    return [
        {
            key: _json(value) if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def write_selection_reports(
    rows, selection: dict, output_directory, *, write_plots=True,
) -> None:
    """모델별·Tier별 CSV와 PNG를 같은 폴더에 단순한 이름으로 저장한다."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    seed_rows = _seed_means(rows)
    fixed = {row["model"]: row for row in selection["model_fixed"]}
    ratio_policies = {
        (row["model"], row["ratio"]): row for row in selection.get("model_ratio", [])
    }
    models = sorted(row["model"] for row in selection["model_fixed"])
    model_tables = {}
    for model_name in models:
        model_rows = []
        combinations = sorted({
            (row["config_id"], row["ratio"], row["score_variant"])
            for row in seed_rows if row["model"] == model_name
        })
        for config_id, ratio, variant in combinations:
            policy = ratio_policies.get((model_name, ratio), fixed[model_name])
            filtered = [
                row for row in seed_rows
                if row["model"] == model_name and row["config_id"] == config_id
                and row["ratio"] == ratio and row["score_variant"] == variant
            ]
            model_rows.append({
                "model": model_name, "config_id": config_id, "ratio": ratio,
                "score_variant": variant,
                "family_macro_vus_pr": _candidate_score(
                    seed_rows, model=model_name, config_id=config_id,
                    score_variant=variant, ratios=[ratio],
                ),
                "series_macro_vus_pr": float(numpy.mean([row["vus_pr"] for row in filtered])),
                "selected_recipe": (
                    policy["selection_status"] == "selected"
                    and policy["config_id"] == config_id
                    and policy["score_variant"] == variant
                ),
                "selected_hyperparameters": policy["hyperparameters"],
                "selection_reason": policy["selection_reason"],
            })
        if not model_rows:
            model_rows = [{
                "model": model_name, "config_id": "", "ratio": "",
                "score_variant": "", "family_macro_vus_pr": "",
                "series_macro_vus_pr": "", "selected_recipe": False,
                "selected_hyperparameters": fixed[model_name]["hyperparameters"],
                "selection_reason": fixed[model_name]["selection_reason"],
                "selection_status": fixed[model_name]["selection_status"],
            }]
        model_tables[model_name] = model_rows
        _write_csv(output_directory / f"{model_name}.csv", _serializable_rows(model_rows))
        if not write_plots:
            continue
        figure, axis = pyplot.subplots(figsize=(7, 4))
        if fixed[model_name]["selection_status"] == "unavailable":
            axis.axis("off")
            axis.set_title(model_name)
            axis.text(
                0.5, 0.58, "Unavailable", ha="center", va="center",
                fontsize=16, transform=axis.transAxes,
            )
            axis.text(
                0.5, 0.42, "No eligible recipe covers the full tuning panel.",
                ha="center", va="center", transform=axis.transAxes,
            )
        else:
            recipes = sorted({
                (row["config_id"], row["score_variant"]) for row in model_rows
            })
            other_recipe_labeled = False
            selected_curve = None
            curves = []
            for config_id, variant in recipes:
                curve = [
                    row for row in model_rows
                    if row["config_id"] == config_id and row["score_variant"] == variant
                ]
                curves.append(curve)
                selected = curve[0]["selected_recipe"] and not ratio_policies
                if selected:
                    label = "Selected recipe"
                    selected_curve = curve
                elif not other_recipe_labeled:
                    label = "Other tested recipe"
                    other_recipe_labeled = True
                else:
                    label = "_nolegend_"
                axis.plot(
                    [row["ratio"] for row in curve],
                    [row["family_macro_vus_pr"] for row in curve], marker="o",
                    color="#1f77b4" if selected else "#a6a6a6",
                    linestyle="-" if selected else "--",
                    linewidth=2.5 if selected else 1.2, label=label,
                )
            if ratio_policies:
                selected_curve = sorted(
                    (row for row in model_rows if row["selected_recipe"]),
                    key=lambda row: row["ratio"],
                )
                axis.plot([row["ratio"] for row in selected_curve],
                          [row["family_macro_vus_pr"] for row in selected_curve],
                          marker="o", color="#1f77b4", linewidth=2.5,
                          label="Selected at each ratio")
            axis.set(
                title=model_name, xlabel="Observed normal prefix (%)",
                ylabel="Family-macro VUS-PR",
            )
            axis.set_xticks(SUPPORTED_RATIO_PERCENTS)
            axis.set_xlim(
                min(SUPPORTED_RATIO_PERCENTS) - 2,
                max(SUPPORTED_RATIO_PERCENTS) + 2,
            )
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if len(recipes) > 1:
                axis.legend(frameon=False, fontsize=8)
            if selected_curve:
                endpoint = selected_curve[-1]
                axis.annotate(
                    f"{endpoint['family_macro_vus_pr']:.3f}",
                    (endpoint["ratio"], endpoint["family_macro_vus_pr"]),
                    xytext=(-5, 7), textcoords="offset points", ha="right",
                    fontsize=8, color="#1f77b4",
                )
            if all(
                len(curve) > 1
                and numpy.allclose(
                    [row["family_macro_vus_pr"] for row in curve],
                    curve[0]["family_macro_vus_pr"],
                )
                for curve in curves
            ):
                axis.text(
                    0.01, 0.03, "Same score reused across ratios",
                    transform=axis.transAxes, fontsize=8, color="#666666",
                )
            elif model_name == "PCA_LEGACY":
                axis.text(
                    0.01, 0.03, "q100 reference only",
                    transform=axis.transAxes, fontsize=8, color="#666666",
                )
        figure.tight_layout()
        figure.savefig(output_directory / f"{model_name}.png", dpi=160)
        pyplot.close(figure)

    lofo_scores = defaultdict(list)
    for row in selection["lofo"]:
        lofo_scores[(row["tier"], row["model"])].append(row["holdout_vus_pr"])
    for tier_policy in selection["tier_fixed"]:
        tier = tier_policy["tier"]
        tier_number = int(tier[1:])
        tier_rows = []
        for (row_tier, model_name), values in sorted(lofo_scores.items()):
            if row_tier != tier:
                continue
            tier_rows.append({
                "tier": tier, "model": model_name,
                "fixed_config_id": fixed[model_name]["config_id"],
                "model_hyperparameters": fixed[model_name]["hyperparameters"],
                "family_lofo_score": float(numpy.mean(values)),
                "selected_model": model_name == tier_policy["selected_model"],
                "selected_hyperparameters": tier_policy["hyperparameters"],
                "selection_reason": tier_policy["selection_reason"],
            })
        _write_csv(
            output_directory / f"Tier{tier_number}.csv",
            _serializable_rows(tier_rows),
        )
        if not write_plots:
            continue
        figure, axis = pyplot.subplots(figsize=(7, 4))
        bars = axis.barh(
            [row["model"] for row in tier_rows],
            [row["family_lofo_score"] for row in tier_rows],
            color=[
                "#1f77b4" if row["selected_model"] else "#bdbdbd"
                for row in tier_rows
            ],
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axis.invert_yaxis()
        maximum = max(row["family_lofo_score"] for row in tier_rows)
        axis.set_xlim(0, maximum * 1.25)
        axis.set(
            title=f"Tier {tier_number} fixed representative",
            xlabel="Family-LOFO VUS-PR", ylabel="",
        )
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.98, 0.04,
            "Only eligible representative" if len(tier_rows) == 1
            else f"Selected: {tier_policy['selected_model']}",
            transform=axis.transAxes, ha="right", fontsize=8, color="#555555",
        )
        figure.tight_layout()
        figure.savefig(output_directory / f"Tier{tier_number}.png", dpi=160)
        pyplot.close(figure)

    selection_rows = _serializable_rows(selection["tier_fixed"])
    _write_csv(output_directory / "selection.csv", selection_rows)
    _write_csv(output_directory / "family_lofo.csv", selection["lofo"])
    model_summary = _serializable_rows(selection["model_fixed"])
    _write_csv(output_directory / "models.csv", model_summary)
    if ratio_policies:
        _write_csv(output_directory / "model_ratio_policy.csv",
                   _serializable_rows(selection["model_ratio"]))
        _write_csv(output_directory / "ratio_family_lofo.csv", selection["adaptive_lofo"])
    _write_csv(
        output_directory / "ratio_adaptive_selection.csv",
        _serializable_rows(selection["tier_adaptive"]),
    )
    _write_csv(
        output_directory / "tier_ratio_candidate_audit.csv",
        _serializable_rows(selection["candidate_audit"]),
    )
    _write_csv(
        output_directory / "tier_policy_transitions.csv",
        _serializable_rows(selection["policy_transitions"]),
    )
    if not write_plots:
        return
    figure, axis = pyplot.subplots(figsize=(11, max(4, len(model_summary) * 0.58)))
    axis.axis("off")
    table_rows = []
    for row in selection["model_fixed"]:
        support = row["q_support"]
        if not support:
            support_label = "-"
        elif len(support) == 1:
            support_label = f"{support[0]}%"
        else:
            support_label = f"{min(support)}-{max(support)}% ({len(support)} levels)"
        selected_scores = [
            value["family_macro_vus_pr"] for value in model_tables[row["model"]]
            if value["selected_recipe"]
        ]
        if row["selection_status"] == "unavailable":
            key_point = "No full-panel recipe"
        elif row["model"] == "PCA_LEGACY":
            key_point = "q100 reference only"
        elif len(selected_scores) > 1 and numpy.allclose(
            selected_scores, selected_scores[0],
        ):
            key_point = "Same score reused"
        else:
            key_point = ""
        table_rows.append([
            row["tier"][1:], row["model"],
            "Available" if row["selection_status"] == "selected" else "Unavailable",
            f"{row['j_fixed']:.3f}" if row["j_fixed"] is not None else "-",
            support_label, key_point,
        ])
    table = axis.table(
        cellText=table_rows,
        colLabels=["Tier", "Model", "Status", "VUS-PR", "Prefix support", "Key point"],
        colWidths=[0.07, 0.18, 0.13, 0.11, 0.21, 0.30], loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    axis.set_title("Model-fixed recipe summary", pad=16)
    figure.tight_layout()
    figure.savefig(output_directory / "models.png", dpi=160)
    pyplot.close(figure)
    figure, axis = pyplot.subplots(figsize=(8.5, 4.8))
    _draw_ratio_adaptive_selection(
        axis, build_ratio_adaptive_plot_data(rows, selection),
    )
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.88)
    figure.savefig(output_directory / "selection.png", dpi=160)
    pyplot.close(figure)


SCORE_MANIFEST_FIELDS = (
    "series", "family", "tier", "model", "config_id", "physical_ratio",
    "seed", "score_variant", "primary_score", "status", "status_reason",
    "score_file", "score_sha256", "metadata_file", "metadata_sha256", "budget_id",
    "retry_count",
)


def _load_final_target_shapes(input_manifest):
    """기존 감사의 shape만 읽고 최종 CSV·라벨은 읽지 않는다."""
    datasets = input_manifest["datasets"]
    audit_root = REPOSITORY_ROOT / "experiments/checks/datasets"
    with (audit_root / "ghl/logs/inventory.csv").open(encoding="utf-8", newline="") as source:
        inventory = {row["file_name"]: row for row in csv.DictReader(source)}
    targets = []
    for entry in datasets["GHL"]["files"]:
        row = inventory[entry["name"]]
        if row["sha256"] != entry["sha256"]:
            raise ValueError("GHL shape 감사와 입력 manifest의 신원이 다르다")
        targets.append({
            "split_role": "ghl25_final", "series": f"{int(row['series']):02d}",
            "training_lengths": [int(row["train_length"])],
            "test_length": int(row["test_length"]), "feature_count": int(row["feature_count"]),
        })
    audit = _read_json(audit_root / "hai/tier2_eda/json/eda_manifest.json")
    audit_inputs = {Path(row["path"]).name: row["sha256"] for row in audit["inputs"]}
    with (audit_root / "hai/tier2_eda/csv/data_integrity.csv").open(encoding="utf-8", newline="") as source:
        inventory = {row["file_name"]: row for row in csv.DictReader(source)}
    hai_entries = {row["name"]: row for row in datasets["HAI"]["files"]}
    for series, split_role in (("01", "train1_to_test1"), ("02", "train1_train2_to_test2")):
        role = input_manifest["roles"][split_role]
        names = role["normal_training_files"] + role["test_files"]
        if any(audit_inputs[name] != hai_entries[name]["sha256"] for name in names):
            raise ValueError("HAI shape 감사와 입력 manifest의 신원이 다르다")
        channels = {int(inventory[name]["feature_count"]) for name in names}
        if len(channels) != 1:
            raise ValueError("HAI 실행 세션의 센서 수가 다르다")
        targets.append({
            "split_role": split_role, "series": series,
            "training_lengths": [int(inventory[name]["row_count"]) for name in role["normal_training_files"]],
            "test_length": int(inventory[role["test_files"][0]]["row_count"]),
            "feature_count": channels.pop(),
        })
    return targets


def build_conditional_membership_rows(selection, registry, input_manifest=None, *, targets=None):
    """튜닝에서 고정한 조건별 정책을 최종 파일의 실행 요청으로 바꾼다."""
    from src.common.model_feasibility import assess_candidate
    from src.common.run_registered_model import SESSION_RUNNERS, TARGET_FREE
    from src.common.select_conditional_policy import match_conditional_policy, match_joint_support
    from src.data_split.split_ratio_prefix import compute_prefix_counts

    if targets is None:
        if input_manifest is None:
            input_manifest = __import__("yaml").safe_load((REPOSITORY_ROOT / "configs/input_manifest.yaml").read_text(encoding="utf-8"))
        targets = _load_final_target_shapes(input_manifest)
    rows = []
    for target in targets:
        training_boundary = sum(target["training_lengths"])
        for ratio in SUPPORTED_RATIO_PERCENTS:
            prefixes = [compute_prefix_counts(length, ratio, full_prefix=True)[0]
                        for length in target["training_lengths"]]
            available = sum(prefixes)
            signatures = {}
            for name, model in registry["models"].items():
                supports_sessions = len(prefixes) == 1 or name in SESSION_RUNNERS or model["target_use"] in TARGET_FREE
                official_protocol = registry.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"
                checks = ([(sum(prefixes), tuple(prefixes))] if official_protocol and name == "GDN"
                          else [(length, None) for length in prefixes])
                signatures[name] = [candidate["config_id"] for candidate in model["candidates"]
                                    if supports_sessions and all(assess_candidate(
                                        name, candidate["hyperparameters"], length, 0,
                                        target["test_length"], target["feature_count"], full_prefix=True,
                                        official_protocol=official_protocol, fit_session_lengths=sessions,
                                    )["status"] == "feasible" for length, sessions in checks)]
                match = match_conditional_policy(
                    selection, registry, model=name, ratio=ratio, available_count=available,
                    feature_count=target["feature_count"], test_length=target["test_length"],
                    training_boundary=training_boundary, feasible_config_ids=signatures[name],
                    allow_out_of_support=True,
                ) if available else {
                    "status": "unavailable", "policy": None, "support_status": "unavailable",
                    "reasons": ["현재 q-prefix가 0행임"],
                }
                rows.append(_conditional_membership_row(
                    "model_ratio", target, ratio, name, model["tier"], match.get("policy"),
                    match["support_status"], match["status"] == "matched", "; ".join(match["reasons"]), registry,
                ))
            for tier in sorted({model["tier"] for model in registry["models"].values()}):
                signature = {name: configs for name, configs in signatures.items()
                             if registry["models"][name]["tier"] == tier and configs}
                matches = [policy for policy in selection.get("tier_adaptive", [])
                           if policy["tier"] == tier and policy["ratio"] == ratio
                           and policy.get("candidate_ids_by_model") == signature
                           and policy["selection_status"] == "selected"]
                if len(matches) > 1:
                    raise ValueError("최종 조건에 대응하는 Tier 정책이 하나가 아니다")
                policy = matches[0] if matches else None
                name = policy["selected_model"] if policy else next(
                    name for name, model in registry["models"].items() if model["tier"] == tier)
                support_reasons = match_joint_support(
                    policy.get("support"), available_count=available,
                    feature_count=target["feature_count"], training_boundary=training_boundary,
                ) if policy else ["conditional_group_missing"]
                has_support = policy is not None and support_reasons != ["conditional_group_support_missing"]
                rows.append(_conditional_membership_row(
                    "tier_adaptive", target, ratio, name, tier, policy,
                    "within_dev_support" if not support_reasons else "out_of_dev_support" if has_support else "unavailable",
                    has_support, "; ".join(support_reasons), registry,
                ))
    return rows


def _conditional_membership_row(kind, target, ratio, model, tier, policy, support_status, runnable, reason, registry):
    from src.common.tuning_support import resolve_policy_score_variant

    family = target.get("family") or {
        "ghl25_final": "GHL", "train1_to_test1": "HAI", "train1_train2_to_test2": "HAI",
    }.get(target["split_role"], "")
    return {
        "analysis_kind": kind, "split_role": target["split_role"], "tier": tier, "model": model,
        "evaluation_ratio": ratio,
        "physical_ratio": (100 if registry["models"][model]["target_use"] in {"training_free", "strict_zero_shot"}
                           else ratio) if runnable else "",
        "config_id": policy["config_id"] if runnable else "",
        "score_variant": resolve_policy_score_variant(policy, family) if runnable else "",
        "status": "runnable" if runnable else "unavailable",
        "status_reason": "" if runnable else reason or "조건별 선택 근거가 없음",
        "series": target["series"], "group_id": policy["group_id"] if policy else "",
        "support_status": support_status if runnable else "unavailable",
    }


def _manifest_key(row) -> tuple:
    return tuple(str(row[field]) for field in (
        "series", "model", "config_id", "physical_ratio", "seed", "score_variant",
    ))


def _validate_primary_manifest_rows(manifest_rows, budget: dict, series_ids) -> list[dict]:
    """채점할 물리 score가 exact panel의 primary key와 정확히 같은지 확인한다."""
    primary_rows = [row for row in manifest_rows if row.get("primary_score") == "true"]
    actual_keys = [_manifest_key(row) for row in primary_rows]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("Dev18 primary score manifest에 중복 exact budget key가 있다")
    expected_keys = {
        tuple(map(str, (
            series, panel["model"], panel["config_id"], panel["physical_ratio"],
            panel["seed"], variant,
        )))
        for panel in budget["execution_panel"]
        for series in panel.get("series_ids", series_ids)
        for variant in panel["primary_score_variants"]
    }
    if set(actual_keys) != expected_keys:
        raise ValueError("Dev18 primary score manifest가 exact budget key와 다르다")
    legacy = {} if budget.get("experiment_mode") == "full_prefix_v2" else budget.get("legacy_budget", {})
    legacy_keys = {
        tuple(map(str, (series, panel["model"], panel["config_id"],
                       panel["physical_ratio"], panel["seed"], variant)))
        for series in series_ids for panel in legacy.get("execution_panel", [])
        for variant in panel["primary_score_variants"]
    }
    if any(row.get("status") != "complete" or (
        row.get("budget_id") != budget["budget_id"] and not (
            row.get("budget_id") == legacy.get("budget_id")
            and _manifest_key(row) in legacy_keys
        )
    ) for row in primary_rows):
        raise ValueError("Dev18 exact budget primary score가 모두 complete가 아니다")
    return primary_rows


def _load_score_manifest(path=DEFAULT_SCORE_MANIFEST_PATH) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != SCORE_MANIFEST_FIELDS:
            raise ValueError("Dev18 score manifest 열이 계약과 다르다")
        return list(reader)


def _merge_manifest_rows(existing, replacements) -> list[dict]:
    by_key = {_manifest_key(row): row for row in existing}
    for row in replacements:
        by_key[_manifest_key(row)] = {
            field: row.get(field, "") for field in SCORE_MANIFEST_FIELDS
        }
    return sorted(by_key.values(), key=_manifest_key)


def _replace_manifest_rows(existing, replacements, path=DEFAULT_SCORE_MANIFEST_PATH) -> list[dict]:
    path = Path(path)
    rows = _merge_manifest_rows(existing, replacements)
    temporary_path = path.with_name(f".{path.name}.tmp")
    _write_csv(temporary_path, rows, SCORE_MANIFEST_FIELDS)
    temporary_path.replace(path)
    return rows


def _relative(path) -> str:
    return Path(path).resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True,
    ).strip()


def _compatible_resume_source() -> str | None:
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"], cwd=REPOSITORY_ROOT, text=True,
    ).strip()
    if parent != DEV18_RECOVERY_PARENT_COMMIT:
        return None
    changed_paths = set(subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
        cwd=REPOSITORY_ROOT, text=True,
    ).splitlines())
    return (
        DEV18_RECOVERY_SOURCE_COMMIT
        if changed_paths == DEV18_RECOVERY_CHANGED_PATHS
        else None
    )


def _require_same_worktree(expected_head: str) -> None:
    _require_clean_worktree()
    if _git_head() != expected_head:
        raise RuntimeError("Dev18 tuning 도중 project commit이 바뀌었다")


def _json_default(value):
    if isinstance(value, numpy.generic):
        return value.item()
    if isinstance(value, numpy.ndarray):
        return value.tolist()
    raise TypeError(f"JSON으로 저장할 수 없는 값이다: {type(value).__name__}")


def _write_run_snapshot(output_directory: Path, spec: dict, inputs: dict, environment: dict) -> Path:
    from src.common.execution_evidence import FULL_PREFIX_STORAGE_SCHEMA_VERSION
    from src.common.run_registered_model import build_registered_execution_policy
    from src.data_split.split_ratio_prefix import compute_prefix_counts

    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "run_snapshot.json"
    current_storage = spec.get("common_recipe", {}).get("training_split") == "full_prefix_v2"
    prefix_observation = {}
    execution_resources = {}
    if (spec["model"] == "PCA_LEGACY"
            and spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"):
        from src.models.tier1.pca_legacy import PCA_FIT_BLAS_THREADS

        execution_resources["pca_fit_blas_threads_requested"] = PCA_FIT_BLAS_THREADS
        execution_resources["cpu"] = _collect_scoring_cpu_environment()
    if current_storage and spec["target_use"] == "fit_full_prefix":
        training_boundary = len(inputs["normal_training"])
        prefix_observation = {
            "training_boundary": training_boundary,
            "observed_row": compute_prefix_counts(training_boundary, spec["ratio"], full_prefix=True)[0],
        }
    path.write_text(json.dumps({
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **({"storage_schema_version": FULL_PREFIX_STORAGE_SCHEMA_VERSION}
           if current_storage else {}),
        **prefix_observation,
        **({"execution_resources": execution_resources} if execution_resources else {}),
        "project_commit": _git_head(),
        "spec": spec,
        "execution_policy": build_registered_execution_policy(spec),
        "input_identity": inputs["input_identity"],
        "source_ranges": inputs["source_ranges"],
        "environment": environment,
    }, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _evidence_directory(output_directory: Path, series: int) -> Path:
    return Path(output_directory) / f"series_{series:02d}"


def _completion_receipt_path(spec: dict, series: int) -> Path:
    from tests.ghl_main.run_registered_models import build_output_directory

    experiment_directory = REPOSITORY_ROOT / "experiments" / "01_ghl_main"
    return _evidence_directory(
        build_output_directory(experiment_directory, spec), series,
    ) / "completion.json"


def _write_completion_receipt(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_completion_rows(
    rows, *, spec: dict, panel_row: dict, series: int, budget_id: str,
) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("Dev18 완료 영수증 형식이 잘못됐다")
    expected_keys = {
        tuple(map(str, (
            f"{series:02d}", spec["model"], spec["config_id"], spec["ratio"],
            spec["seed"], variant,
        )))
        for variant in _expected_variants(panel_row)
    }
    actual_keys = [_manifest_key(row) for row in rows]
    if (
        len(actual_keys) != len(set(actual_keys))
        or set(actual_keys) != expected_keys
        or any(
            row.get("status") != "complete"
            or row.get("budget_id") != budget_id
            for row in rows
        )
    ):
        raise ValueError("Dev18 완료 영수증이 exact 실행과 다르다")
    return rows


def _load_completion_receipt(
    path: Path, *, spec: dict, panel_row: dict, series: int, budget_id: str,
) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    return _validate_completion_rows(
        _read_json(path), spec=spec, panel_row=panel_row, series=series, budget_id=budget_id,
    )


def _recover_saved_run_rows(*, spec, panel_row, series, family, budget_id, computed):
    """출력은 저장됐지만 완료 이력 쓰기가 끊긴 실행의 행을 복구한다."""
    from src.common.naming import build_score_filename
    from tests.ghl_main.run_registered_models import build_output_directory

    history_path = Path(computed["history_file"]).resolve()
    history_path.relative_to(REPOSITORY_ROOT)
    history = _read_json(history_path)
    recovered = []
    for variant in _expected_variants(panel_row):
        output_directory = build_output_directory(
            REPOSITORY_ROOT / "experiments/01_ghl_main", spec,
            score_variant=variant if spec["model"] == "TSPulse" else None,
        )
        raw = output_directory / build_score_filename(
            "DEV18", series, spec["model"], spec["tier"], spec["ratio"], spec["seed"], "raw", "trainnorm",
        )
        metadata_path = raw.with_suffix(".meta.json")
        if not metadata_path.is_file():
            return []
        metadata = _read_json(metadata_path)
        attempt = metadata.get("execution_attempt", {})
        if (attempt.get("run_id") != history["run_id"]
                or not attempt.get("history_file")
                or (REPOSITORY_ROOT / attempt["history_file"]).resolve() != history_path):
            raise ValueError("저장된 출력이 복구할 모델 시도와 다르다")
        references = metadata.get("score_files", [])
        if not references:
            return []
        by_path = {}
        for reference in references:
            path = (REPOSITORY_ROOT / reference["file"]).resolve()
            path.relative_to(output_directory.resolve())
            if not path.is_file():
                return []
            if path in by_path or file_sha256(path) != reference["sha256"] or path.stat().st_size != reference["bytes"]:
                raise ValueError("복구할 점수의 SHA-256 또는 크기가 저장 근거와 다르다")
            by_path[path] = reference
        if raw.resolve() not in by_path:
            return []
        recovered.append({
            "series": f"{series:02d}", "family": family, "tier": spec["tier"], "model": spec["model"],
            "config_id": spec["config_id"], "physical_ratio": spec["ratio"], "seed": spec["seed"],
            "score_variant": variant, "primary_score": str(variant in panel_row["primary_score_variants"]).lower(),
            "status": "complete", "status_reason": "", "score_file": _relative(raw),
            "score_sha256": by_path[raw.resolve()]["sha256"], "metadata_file": _relative(metadata_path),
            "metadata_sha256": file_sha256(metadata_path), "budget_id": budget_id,
            "retry_count": computed["attempt"],
        })
    return _validate_completion_rows(
        recovered, spec=spec, panel_row=panel_row, series=series, budget_id=budget_id,
    )


def _record_seed_state(snapshot_path: Path, result: dict) -> None:
    snapshot = _read_json(snapshot_path)
    snapshot["seed_state"] = result["seed_state"]
    if "effective_execution" in result:
        snapshot["effective_execution"] = result["effective_execution"]
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _validate_bound_run_files(
    metadata: dict, *, expected_project_commit: str | None = None,
    compatible_project_commit: str | None = None,
    expected_environment: dict | None = None, expected_spec: dict | None = None,
    allow_compatible_history: bool = False,
) -> None:
    references = [metadata.get("run_snapshot")]
    references.extend((metadata.get("training_files") or {}).values())
    references.extend(metadata.get("score_files", []))
    if not references or any(not isinstance(reference, dict) for reference in references):
        raise ValueError("Dev18 run snapshot·training file 연결이 없다")
    for reference in references:
        path = (REPOSITORY_ROOT / reference.get("file", "")).resolve()
        try:
            path.relative_to(REPOSITORY_ROOT)
        except ValueError as error:
            raise ValueError("Dev18 실행 증거가 저장소 밖을 가리킨다") from error
        if not path.is_file() or file_sha256(path) != reference.get("sha256"):
            raise ValueError("Dev18 실행 증거 SHA-256이 실제 파일과 다르다")
        if "bytes" in reference and path.stat().st_size != reference["bytes"]:
            raise ValueError("Dev18 checkpoint byte 수가 실제 파일과 다르다")
    snapshot_path = (
        REPOSITORY_ROOT / metadata["run_snapshot"]["file"]
    ).resolve()
    snapshot = _read_json(snapshot_path)
    current_storage = (expected_spec or snapshot.get("spec", {})).get("common_recipe", {}).get("training_split") == "full_prefix_v2"
    if current_storage:
        from src.common.execution_evidence import FULL_PREFIX_STORAGE_SCHEMA_VERSION

        if snapshot.get("storage_schema_version") != FULL_PREFIX_STORAGE_SCHEMA_VERSION:
            raise ValueError("Dev18 full-prefix snapshot의 저장 계약이 현재 실행과 다르다")
        compatible_project_commit = None
        allow_compatible_history = False
    if expected_project_commit is not None or expected_environment is not None:
        from tests.checks.validate_resource_resume import resource_resume_compatible

        accepted_commits = {expected_project_commit, compatible_project_commit} - {None}
        if accepted_commits and snapshot.get("project_commit") not in accepted_commits:
            if not (current_storage and resource_resume_compatible(
                snapshot.get("project_commit"), expected_project_commit, REPOSITORY_ROOT,
            )):
                if not allow_compatible_history:
                    raise ValueError("Dev18 재개 snapshot의 project commit이 현재 HEAD와 다르다")
                _validate_resume_source(snapshot.get("project_commit"), expected_project_commit)
        if (
            expected_environment is not None and not allow_compatible_history
            and snapshot.get("environment") != expected_environment
        ):
            raise ValueError("Dev18 재개 snapshot의 실행 환경이 현재 봉인과 다르다")
    from src.common.run_registered_model import build_registered_execution_policy, select_input_scaler

    snapshot_spec = snapshot.get("spec")
    if (
        not isinstance(snapshot_spec, dict)
        or (
            expected_spec is not None
            and _json(snapshot_spec) != _json(expected_spec)
        )
        or snapshot.get("execution_policy")
        != build_registered_execution_policy(expected_spec or snapshot_spec)
    ):
        raise ValueError("Dev18 run snapshot의 execution policy가 현재 등록 일정과 다르다")
    if current_storage and snapshot_spec["target_use"] == "fit_full_prefix":
        from src.data_split.split_ratio_prefix import compute_prefix_counts

        source_range = snapshot["source_ranges"]["normal_training"]
        boundary = source_range[1] - source_range[0]
        observed = compute_prefix_counts(boundary, snapshot_spec["ratio"], full_prefix=True)[0]
        if snapshot.get("training_boundary") != boundary or snapshot.get("observed_row") != observed:
            raise ValueError("Dev18 snapshot의 관측 행 수가 현재 prefix와 다르다")
    if metadata.get("training_state_version") == 1:
        training_files = metadata.get("training_files") or {}
        if (snapshot_spec["target_use"] in {"fit_validation", "fit_full_prefix"}
                or snapshot_spec["model"] == "PCA_LEGACY"
                and snapshot_spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"):
            if "checkpoint" not in training_files:
                raise ValueError("학습형 모델의 checkpoint가 없다")
            if select_input_scaler(snapshot_spec) != "none" and "scaler_state" not in training_files:
                raise ValueError("Tier 2 입력 scaler 상태가 없다")


@lru_cache(maxsize=None)
def _validate_resume_source(previous_commit: str, current_commit: str) -> None:
    """선택·예산 변경만 허용하고 점수 생성 코드가 바뀐 이력은 재사용하지 않는다."""
    changed = subprocess.check_output([
        "git", "diff", "--name-only", previous_commit, current_commit, "--",
        "src", "configs", "tests/ghl_main/run_registered_models.py",
        ":(exclude)src/common/load_final_membership.py",
        ":(exclude)configs/deployment_scenario.schema.json",
    ], cwd=REPOSITORY_ROOT, text=True).splitlines()
    graph_path = "src/models/tier2/gdn_official/official.py"
    if graph_path in changed:
        previous_graph = subprocess.check_output(
            ["git", "show", f"{previous_commit}:{graph_path}"],
            cwd=REPOSITORY_ROOT, encoding="utf-8",
        )
        # 완료 배치의 메모리 복구는 사용하지 않는 attention 캐시만 detach했다.
        previous_graph = previous_graph.replace(
            "self.attention_weights = attention\n", "self.attention_weights = attention.detach()\n",
        )
        current_graph = (REPOSITORY_ROOT / graph_path).read_text(encoding="utf-8")
        if ast.dump(ast.parse(previous_graph)) == ast.dump(ast.parse(current_graph)):
            changed.remove(graph_path)
    if changed:
        raise ValueError(f"기존 점수 생성 코드가 바뀌어 재사용할 수 없다: {changed}")
    relative_path = "tests/ghl_main/run_dev18_tuning.py"
    previous = subprocess.check_output(
        ["git", "show", f"{previous_commit}:{relative_path}"],
        cwd=REPOSITORY_ROOT, encoding="utf-8",
    )
    names = {"_run_one_spec", "_write_run_snapshot", "_record_seed_state",
             "_save_training_files", "_json_default", "_evidence_directory"}
    def bodies(source):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name in names}
    if bodies(previous) != bodies(Path(__file__).read_text(encoding="utf-8")):
        raise ValueError("기존 점수 저장·실행 함수가 바뀌어 재사용할 수 없다")


def _save_training_files(output_directory: Path, result: dict, *, on_file_saved=None) -> tuple[int, dict]:
    training_directory = output_directory / "training"
    training_directory.mkdir(parents=True, exist_ok=True)
    files = {}
    checkpoint = result.get("checkpoint")
    if checkpoint is not None:
        import torch

        checkpoint_path = training_directory / "checkpoint.ckpt"
        temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
        files["checkpoint"] = {
            "file": _relative(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        }
        if on_file_saved is not None:
            on_file_saved("checkpoint", files["checkpoint"])
    for name, value in (
        ("scaler_state", result.get("scaler_state")),
        ("training_log", result.get("training_log")),
        ("timing", result.get("timing")),
        ("effective_execution", result.get("effective_execution")),
    ):
        if value is None:
            continue
        path = training_directory / f"{name}.json"
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
        files[name] = {
            "file": _relative(path), "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        if on_file_saved is not None:
            on_file_saved(name, files[name])
    artifact_bytes = sum(files.get(name, {}).get("bytes", 0)
                         for name in ("checkpoint", "scaler_state"))
    return artifact_bytes, files


def _save_training_completion(output_directory: Path, result: dict, history: dict) -> None:
    """추론 전에 학습 증거를 시도별로 보존한다. 실패한 추론의 학습 재개점은 아니다."""
    _require_same_worktree(history["run_snapshot"]["project_commit"])
    attempt_directory = output_directory / "training_attempts" / history["run_id"]
    history["training_complete"] = {
        "status": "saving", "directory": _relative(attempt_directory),
        "timing": dict(result["timing"]), "seed_state": result["seed_state"], "files": {},
    }
    history["model_execution_seconds_scope"] = "execution_wall_including_training_evidence_storage"

    def record_training_file(name, reference):
        history["training_complete"]["files"][name] = reference
        save_run_history(history)

    try:
        with record_run_stage(history, "save_training"):
            _, files = _save_training_files(attempt_directory, result, on_file_saved=record_training_file)
            history["training_complete"].update(status="complete", files=files)
            save_run_history(history)
    except Exception as error:
        raise RunResultPersistenceError("학습 완료 증거 저장에 실패했다") from error


def _specs_for_budget(budget: dict) -> tuple[list[dict], dict[tuple, dict]]:
    from tests.ghl_main.run_registered_models import build_specs

    panel_by_key = {
        (row["model"], row["config_id"], row["physical_ratio"], row["seed"]): row
        for row in budget["execution_panel"]
    }
    feasible_keys = {
        (row["model"], row["config_id"], row["physical_ratio"])
        for row in budget["execution_panel"]
    }
    specs = [
        spec for spec in build_specs("development", feasible_keys=feasible_keys)
        if (spec["model"], spec["config_id"], spec["ratio"], spec["seed"])
        in panel_by_key
    ]
    if len(specs) != budget["physical_execution_count"] or len(panel_by_key) != len(specs):
        raise ValueError("Dev18 exact budget과 registry spec 수가 다르다")
    return specs, panel_by_key


def _expected_variants(panel_row: dict) -> tuple[str, ...]:
    return tuple(panel_row["primary_score_variants"] + panel_row["diagnostic_score_variants"])


def _completed_run(
    manifest_rows, *, spec: dict, panel_row: dict, series: int,
    input_manifest_path, expected_project_commit: str,
    compatible_project_commit: str | None,
    expected_environment: dict,
    allow_compatible_history: bool = False,
) -> bool:
    from tests.ghl_main.check_registered_outputs import check_registered_output
    from tests.ghl_main.run_registered_models import build_output_directory

    by_key = {_manifest_key(row): row for row in manifest_rows}
    experiment_directory = REPOSITORY_ROOT / "experiments" / "01_ghl_main"
    for variant in _expected_variants(panel_row):
        key = tuple(map(str, (
            f"{series:02d}", spec["model"], spec["config_id"], spec["ratio"],
            spec["seed"], variant,
        )))
        row = by_key.get(key)
        if row is None or row["status"] != "complete":
            return False
        output_directory = build_output_directory(
            experiment_directory, spec,
            score_variant=variant if spec["model"] == "TSPulse" else None,
        )
        check_registered_output(
            output_directory, spec, dataset="DEV18", series=series,
            input_manifest_path=input_manifest_path,
            score_variant=variant if variant else None,
        )
        score_path = REPOSITORY_ROOT / row["score_file"]
        metadata_path = REPOSITORY_ROOT / row["metadata_file"]
        if (
            file_sha256(score_path) != row["score_sha256"]
            or file_sha256(metadata_path) != row["metadata_sha256"]
        ):
            raise ValueError("Dev18 score manifest SHA-256이 실제 산출물과 다르다")
        _validate_bound_run_files(
            _read_json(metadata_path),
            expected_project_commit=expected_project_commit,
            compatible_project_commit=compatible_project_commit,
            expected_environment=expected_environment,
            expected_spec=spec,
            allow_compatible_history=allow_compatible_history,
        )
    return True


def _authorized_oom_recovery_row(
    prior_rows, *, budget_id: str, compatible_project_commit: str | None,
) -> dict | None:
    if (
        compatible_project_commit != DEV18_RECOVERY_SOURCE_COMMIT
        or budget_id != DEV18_RECOVERY_BUDGET_ID
        or len(prior_rows) != 1
    ):
        return None
    row = prior_rows[0]
    if (
        _manifest_key(row) == DEV18_RECOVERY_TRIAL_KEY
        and row.get("status") == "failed"
        and row.get("budget_id") == DEV18_RECOVERY_BUDGET_ID
        and str(row.get("retry_count")) == str(DEV18_RECOVERY_RETRY_COUNT - 1)
        and str(row.get("status_reason", "")).startswith(
            "OutOfMemoryError: CUDA out of memory."
        )
    ):
        return row
    return None


def _write_oom_recovery_receipt(
    path: Path, original_failure: dict, *, recovery_project_commit: str,
) -> None:
    path = Path(path)
    payload = {
        "schema_version": 1,
        "source_project_commit": DEV18_RECOVERY_SOURCE_COMMIT,
        "failed_project_commit": DEV18_RECOVERY_PARENT_COMMIT,
        "recovery_project_commit": recovery_project_commit,
        "budget_id": DEV18_RECOVERY_BUDGET_ID,
        "trial_key": list(DEV18_RECOVERY_TRIAL_KEY),
        "authorized_retry_count": DEV18_RECOVERY_RETRY_COUNT,
        "original_failure": dict(original_failure),
    }
    if path.is_file():
        if _read_json(path) != payload:
            raise ValueError("Dev18 OOM 복구 영수증이 현재 실패 기록과 다르다")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _authorized_attempt_limit(
    maximum_attempts: int, *, recovery_row: dict | None,
) -> int:
    if recovery_row is None:
        return maximum_attempts
    return max(maximum_attempts, DEV18_RECOVERY_RETRY_COUNT + 1)


def _pca_blas_recovery(*, budget_id, trial_key, attempts_used, maximum_attempts):
    if (budget_id != "b2f61f74691c6"
            or trial_key != ("13", "PCA_LEGACY", "cb3ca230f385a", "100", "0")
            or attempts_used != 3 or maximum_attempts != 3):
        return None
    return {
        "reason": "openblas_initialization_sigsegv",
        "affected_project_commit": "241d703e1afe04fe3f8bf0d709025d15da3382dc",
        "previous_attempts_preserved": attempts_used,
        "sealed_maximum_total_attempts": maximum_attempts,
        "maximum_total_attempts": 4,
    }


class RunResultPersistenceError(RuntimeError):
    """학습·추론 증거의 저장 실패를 모델 재시도와 구분한다."""


def _read_process_memory() -> dict:
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
        return {"rss_bytes": int(fields["VmRSS"].split()[0]) * 1024,
                "process_lifetime_peak_bytes": int(fields["VmHWM"].split()[0]) * 1024,
                "reason": None}
    except (OSError, KeyError, ValueError) as error:
        return {"rss_bytes": None, "process_lifetime_peak_bytes": None,
                "reason": f"proc_status_unavailable:{type(error).__name__}"}


def _start_run_resources(device: str) -> dict:
    import torch

    usage = {"requested_device": device, "cpu_start": _read_process_memory(),
             "gpu_allocated_start_bytes": None, "gpu_reserved_start_bytes": None}
    if device.startswith("cuda") and torch.cuda.is_available():
        usage.update(gpu_allocated_start_bytes=torch.cuda.memory_allocated(device),
                     gpu_reserved_start_bytes=torch.cuda.memory_reserved(device))
    return usage


def _finish_run_resources(start: dict, result: dict, *, python_peak_bytes) -> dict:
    import torch

    backend = result.get("effective_execution", {}).get("backend")
    end = _read_process_memory()
    cuda_backend = isinstance(backend, str) and backend.startswith("cuda")
    device = start["requested_device"]
    gpu_measured = cuda_backend and device.startswith("cuda") and torch.cuda.is_available()
    return {
        "requested_device": device, "actual_backend": backend,
        "actual_backend_reason": None if backend is not None else "execution_backend_not_recorded",
        "scope": "model_execution",
        "cpu_rss_start_bytes": start["cpu_start"]["rss_bytes"],
        "cpu_rss_end_bytes": end["rss_bytes"],
        "cpu_process_lifetime_peak_bytes": end["process_lifetime_peak_bytes"],
        "cpu_peak_scope": "process_lifetime",
        "cpu_rss_reason": start["cpu_start"]["reason"] or end["reason"],
        "python_tracemalloc_peak_bytes": python_peak_bytes,
        "python_measurement_scope": "python_traced_allocations_since_start",
        "gpu_allocated_start_bytes": start["gpu_allocated_start_bytes"] if gpu_measured else None,
        "gpu_reserved_start_bytes": start["gpu_reserved_start_bytes"] if gpu_measured else None,
        "gpu_allocated_peak_bytes": torch.cuda.max_memory_allocated(device) if gpu_measured else None,
        "gpu_reserved_peak_bytes": torch.cuda.max_memory_reserved(device) if gpu_measured else None,
        "gpu_measurement_scope": "allocator_absolute_peak_since_reset" if gpu_measured else "not_applicable",
        "gpu_measurement_reason": None if gpu_measured else "cuda_backend_not_used_or_unavailable",
    }


def _finish_failed_run_resources(start: dict) -> dict:
    """실패 직전 측정값을 남기되 실제 backend를 추정하지 않는다."""
    import torch

    device = start["requested_device"]
    end = _read_process_memory()
    gpu_started = start["gpu_allocated_start_bytes"] is not None
    usage = {
        "requested_device": device, "actual_backend": None,
        "actual_backend_reason": "execution_failed_before_result", "scope": "failed_model_execution",
        "cpu_rss_start_bytes": start["cpu_start"]["rss_bytes"],
        "cpu_rss_end_bytes": end["rss_bytes"],
        "cpu_process_lifetime_peak_bytes": end["process_lifetime_peak_bytes"],
        "cpu_peak_scope": "process_lifetime", "cpu_rss_reason": start["cpu_start"]["reason"] or end["reason"],
        "python_tracemalloc_peak_bytes": None,
        "python_measurement_scope": "python_traced_allocations_since_start",
        "gpu_allocated_start_bytes": start["gpu_allocated_start_bytes"],
        "gpu_reserved_start_bytes": start["gpu_reserved_start_bytes"],
        "gpu_allocated_peak_bytes": None, "gpu_reserved_peak_bytes": None,
        "gpu_measurement_scope": "requested_device_allocator_absolute_peak_since_reset" if gpu_started else "not_applicable",
        "gpu_measurement_reason": "actual_backend_unknown_after_failure" if gpu_started else "cuda_measurement_not_started",
        "measurement_errors": {},
    }
    measurements = [("python_tracemalloc_peak_bytes", lambda: tracemalloc.get_traced_memory()[1])]
    if gpu_started:
        measurements.extend((
            ("gpu_allocated_peak_bytes", lambda: torch.cuda.max_memory_allocated(device)),
            ("gpu_reserved_peak_bytes", lambda: torch.cuda.max_memory_reserved(device)),
        ))
    for field, measure in measurements:
        try:
            usage[field] = measure()
        except Exception as error:
            usage["measurement_errors"][field] = f"{type(error).__name__}: {error}"
    return usage


def _run_one_spec(
    spec: dict, panel_row: dict, inputs: dict, *, series: int,
    device: str, environment: dict, input_manifest_path, retry_count: int,
    budget_id: str, history: dict,
) -> list[dict]:
    from src.common.run_registered_model import execute_registered_model
    from tests.ghl_main.run_registered_models import build_output_directory

    experiment_directory = REPOSITORY_ROOT / "experiments" / "01_ghl_main"
    base_directory = build_output_directory(experiment_directory, spec)
    evidence_directory = _evidence_directory(base_directory, series)
    snapshot_path = _write_run_snapshot(
        evidence_directory, spec, inputs, environment,
    )
    history["run_snapshot_file"] = _relative(snapshot_path)
    history["run_snapshot"] = _read_json(snapshot_path)
    save_run_history(history)
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    resource_start = _start_run_resources(device)
    memory_sampler = None
    tracemalloc.start()
    execution_started = time.perf_counter()
    pca_cpu_started = time.process_time() if spec["model"] == "PCA_LEGACY" else None
    pca_resources = {}
    try:
        memory_sampler = start_process_memory_sampling(_read_process_memory)
        result = execute_registered_model(
            spec,
            normal_training=inputs.get("normal_training"),
            normal_training_sessions=inputs.get("normal_training_sessions"),
            test_sessions=inputs["test_sessions"], device=device,
            **({"on_training_complete": lambda trained: _save_training_completion(
                evidence_directory, trained, history,
            )} if spec["model"] in {"PaAno", "GDN"} else {}),
        )
    except BaseException:
        pca_cpu_seconds = time.process_time() - pca_cpu_started if pca_cpu_started is not None else None
        history.update(model_execution_complete=False,
                       model_execution_seconds=time.perf_counter() - execution_started)
        try:
            history["resource_usage"] = _finish_failed_run_resources(resource_start)
        except Exception as error:
            history["resource_usage"] = {
                "status": "unavailable", "scope": "failed_model_execution",
                "reason": f"resource_measurement_failed:{type(error).__name__}: {error}",
                "start": resource_start,
            }
        raise
    else:
        pca_cpu_seconds = time.process_time() - pca_cpu_started if pca_cpu_started is not None else None
        history.update(model_execution_complete=True,
                       model_execution_seconds=time.perf_counter() - execution_started)
        _, peak_bytes = tracemalloc.get_traced_memory()
        save_run_history(history)
    finally:
        if pca_cpu_started is not None:
            pca_resources["pca_compute"] = {
                "wall_seconds": history["model_execution_seconds"],
                "cpu_core_seconds": pca_cpu_seconds,
                "scope": "registered_model_call_all_process_threads_excluding_children",
                "single_core_equivalent_basis": "measured_cpu_core_seconds_including_parallel_overhead",
                "single_thread_wall_seconds": None,
                "single_thread_wall_seconds_reason": "not_measured_cannot_recover_from_parallel_run",
            }
            history.setdefault("resource_usage", {}).update(pca_resources)
        tracemalloc.stop()
        sampled_resources = memory_sampler.stop() if memory_sampler is not None else {
            "cpu_rss_sampled_peak_bytes": None, "cpu_rss_sampling_status": "unavailable",
            "cpu_rss_sampling_reason": "sampling_interrupted_before_start",
        }
        history.setdefault("resource_usage", {}).update(sampled_resources)
        history["resource_usage"]["includes_training_evidence_storage"] = "training_complete" in history
    try:
        if pca_resources:
            pca_resources["pca_compute"]["parallelism"] = [
                output.get("pca_parallelism") for output in result["test_outputs"]
            ]
        history["model_timing"] = dict(result["timing"])
        resource_usage = {**_finish_run_resources(resource_start, result, python_peak_bytes=peak_bytes),
                          **sampled_resources,
                          **pca_resources,
                          "includes_training_evidence_storage": "training_complete" in history}
        history["resource_usage"] = resource_usage
        with record_run_stage(history, "save_result"):
            peak_memory_mb = resource_usage["gpu_allocated_peak_bytes"]
            if spec.get("common_recipe", {}).get("training_split") != "full_prefix_v2":
                peak_memory_mb = (
                    torch.cuda.max_memory_allocated(device) / 1024 ** 2
                    if device.startswith("cuda") and torch.cuda.is_available()
                    else peak_bytes / 1024 ** 2
                )
            elif peak_memory_mb is not None:
                peak_memory_mb /= 1024 ** 2
            return _save_run_result(
                result, spec, panel_row, inputs, series=series,
                snapshot_path=snapshot_path, peak_memory_mb=peak_memory_mb,
                input_manifest_path=input_manifest_path, retry_count=retry_count, budget_id=budget_id,
                resource_usage=resource_usage,
                execution_attempt={"run_id": history["run_id"], "history_file": _relative(history["history_file"])},
                history=history,
            )
    except Exception as error:
        raise RunResultPersistenceError(
            f"모델 실행 후 결과 저장에 실패했다: {snapshot_path}: {type(error).__name__}: {error}"
        ) from error


def _save_run_result(
    result, spec, panel_row, inputs, *, series, snapshot_path, peak_memory_mb,
    input_manifest_path, retry_count, budget_id,
    resource_usage=None,
    execution_attempt=None,
    history=None,
):
    from src.common.execution_evidence import (
        DEV18_MEASUREMENT_PROTOCOL_ID, FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, build_execution_evidence,
    )
    from src.common.save_model_artifacts import _write_metadata, save_execution_result
    from tests.ghl_main.check_registered_outputs import check_registered_output
    from tests.ghl_main.run_registered_models import build_output_directory

    experiment_directory = REPOSITORY_ROOT / "experiments/01_ghl_main"
    evidence_directory = snapshot_path.parent
    _require_same_worktree(_read_json(snapshot_path)["project_commit"])
    _record_seed_state(snapshot_path, result)
    preserved = (history or {}).get("training_complete", {}).get("files", {})

    def record_result_file(name, reference):
        history.setdefault("result_files", {})[name] = reference
        save_run_history(history)

    artifact_bytes, training_files = _save_training_files(
        evidence_directory, {**result, **{name: None for name in preserved
                                         if name in {"checkpoint", "scaler_state", "training_log"}}},
        on_file_saved=record_result_file if history is not None else None,
    )
    if preserved:
        training_files = {**preserved, **training_files}
        artifact_bytes = sum(training_files.get(name, {}).get("bytes", 0)
                             for name in ("checkpoint", "scaler_state"))
    split = result["split"]
    split_count = 0 if split is None else 1 if isinstance(split, dict) else len(split)
    unavailable_duration = {
        "observed_duration_seconds": None, "duration_basis": "unavailable",
    }
    evidence = build_execution_evidence(
        split, result["timing"], spec=spec,
        measurement_protocol_id=(FULL_PREFIX_MEASUREMENT_PROTOCOL_ID
                                 if spec["common_recipe"].get("training_split") == "full_prefix_v2"
                                 else DEV18_MEASUREMENT_PROTOCOL_ID), retry_count=retry_count,
        training_session_durations=[dict(unavailable_duration) for _ in range(split_count)],
        test_input_sessions=inputs["test_sessions"],
        test_session_durations=[dict(unavailable_duration) for _ in inputs["test_sessions"]],
        peak_memory_mb=peak_memory_mb, model_artifact_bytes=artifact_bytes,
        resource_usage=resource_usage,
    )
    variants = _expected_variants(panel_row)
    saved_rows = []
    for variant in variants:
        output_directory = build_output_directory(
            experiment_directory, spec,
            score_variant=variant if spec["model"] == "TSPulse" else None,
        )
        saved = save_execution_result(
            result, output_directory, spec=spec, dataset="DEV18", series=series,
            score_variant=variant or None, execution_evidence=evidence,
        )
        metadata_path = Path(saved["metadata_path"])
        metadata = _read_json(metadata_path)
        metadata["run_snapshot"] = {
            "file": _relative(snapshot_path), "sha256": file_sha256(snapshot_path),
        }
        metadata["training_files"] = training_files
        metadata["training_state_version"] = 1
        if execution_attempt is not None:
            metadata["execution_attempt"] = execution_attempt
        metadata["score_files"] = [{
            "file": _relative(path), "sha256": file_sha256(path), "bytes": Path(path).stat().st_size,
        } for path in saved["score_paths"]]
        _write_metadata(metadata_path, metadata)
        _validate_bound_run_files(metadata, expected_spec=spec)
        check_registered_output(
            output_directory, spec, dataset="DEV18", series=series,
            input_manifest_path=input_manifest_path,
            score_variant=variant if variant else None,
        )
        raw_score = next(
            Path(path) for path in saved["score_paths"]
            if "__raw__trainnorm.npy" in Path(path).name
            and not Path(path).stem.endswith("__channels")
        )
        saved_rows.append({
            "series": f"{series:02d}", "family": inputs["family"],
            "tier": spec["tier"], "model": spec["model"],
            "config_id": spec["config_id"], "physical_ratio": spec["ratio"],
            "seed": spec["seed"], "score_variant": variant,
            "primary_score": str(variant in panel_row["primary_score_variants"]).lower(),
            "status": "complete", "status_reason": "",
            "score_file": _relative(raw_score), "score_sha256": file_sha256(raw_score),
            "metadata_file": _relative(metadata_path),
            "metadata_sha256": file_sha256(metadata_path),
            "budget_id": budget_id,
            "retry_count": retry_count,
        })
    if history is not None:
        history["result"] = saved_rows
        save_run_history(history)
    _write_completion_receipt(
        evidence_directory / "completion.json", saved_rows,
    )
    return saved_rows


def _require_cuda_or_remote(*, device: str, remote_execution: bool) -> None:
    if remote_execution:
        return
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "고비용 Dev18 실행은 활성 CUDA 또는 명시한 원격 환경에서만 허용한다"
        )


def execute_panel(
    *, data_root=DEFAULT_DATA_ROOT, device="cuda", remote_execution=False,
    budget=None, manifest_path=DEFAULT_SCORE_MANIFEST_PATH, preserved_rows=(),
    execution_timing=None, recommendation_directory=None,
) -> list[dict]:
    """봉인한 예산의 CSV별 실행 목록을 재개형으로 처리한다."""
    execution_timing = {} if execution_timing is None else execution_timing
    execution_timing.update(completion_check_seconds=0.0, model_attempt_seconds=0.0)
    _require_cuda_or_remote(device=device, remote_execution=remote_execution)
    if device.startswith("cuda"):
        from src.common.set_reproducible_seed import set_reproducible_seed

        set_reproducible_seed(0)
    budget = budget or _read_json(DEFAULT_BUDGET_PATH)
    full_prefix = budget.get("experiment_mode") == "full_prefix_v2"
    if full_prefix and (preserved_rows or "legacy_budget" in budget):
        raise ValueError("새 full-prefix 실행은 과거 결과를 승계하지 않는다")
    readiness = prepare_tuning(data_root=data_root, require_clean=True, budget=budget)
    specs, panel_by_key = _specs_for_budget(budget)
    specs.sort(key=_execution_priority)
    manifest_rows = _merge_manifest_rows(preserved_rows, _load_score_manifest(manifest_path))
    if full_prefix and any(row["budget_id"] != budget["budget_id"] for row in manifest_rows):
        raise ValueError("새 full-prefix manifest에 다른 실험 결과가 있다")
    preserved_keys = {_manifest_key(row)[:5] for row in preserved_rows}
    input_manifest_path = REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
    input_manifest, _ = load_input_manifest_role(
        input_manifest_path, "development", "dev18_selection",
    )
    series_entries = sorted(
        input_manifest["datasets"]["DEV18"]["files"],
        key=_series_execution_priority,
    )
    environment = readiness["environment"]
    project_commit = _git_head()
    compatible_project_commit = None if full_prefix else _compatible_resume_source()
    computed_attempts = {}
    prior_attempts = load_attempt_counts(
        REPOSITORY_ROOT / "experiments/01_ghl_main/logs/run_history/model_attempts", budget["budget_id"],
        computed_attempts=computed_attempts,
    )
    from tests.ghl_main.run_registered_models import load_registered_inputs

    with ExitStack() as recommendation_stack:
        recommendation_store = None
        if full_prefix:
            from src.data_split.load_dev18_series import load_dev18_registered_inputs
            from tests.ghl_main.store_recommendation_evidence import (
                DEFAULT_RECOMMENDATION_DIRECTORY, open_recommendation_evidence,
            )

            recommendation_store = recommendation_stack.enter_context(open_recommendation_evidence(
                load_model_registry_with_sha()[0], budget,
                directory=(recommendation_directory if recommendation_directory is not None
                           else DEFAULT_RECOMMENDATION_DIRECTORY),
                project_commit=project_commit,
            ))
            recommendation_store.bind_environment(environment)
            feature_started = time.perf_counter()
            feature_status = recommendation_store.prepare_features(
                lambda entry: load_dev18_registered_inputs(entry, data_root),
            )
            execution_timing["feature_preparation_seconds"] = time.perf_counter() - feature_started
            if not feature_status["feature_complete"]:
                raise ValueError("Dev18 입력 특징 수집이 끝나지 않아 모델 실행을 열지 않는다")
        for entry in series_entries:
            if budget.get("series_ids") and entry["series"] not in budget["series_ids"]:
                continue
            series = int(entry["series"])
            inputs = None
            for spec in specs:
                if full_prefix:
                    spec = {**spec, "series": f"{series:02d}", "series_input_sha256": entry["sha256"]}
                key = (spec["model"], spec["config_id"], spec["ratio"], spec["seed"])
                panel_row = panel_by_key[key]
                if f"{series:02d}" not in panel_row.get("series_ids", budget.get("series_ids", [f"{series:02d}"])):
                    continue
                check_started = time.perf_counter()
                try:
                    if _completed_run(
                        manifest_rows, spec=spec, panel_row=panel_row, series=series,
                        input_manifest_path=input_manifest_path,
                        expected_project_commit=project_commit,
                        compatible_project_commit=compatible_project_commit,
                        expected_environment=environment,
                        allow_compatible_history=not full_prefix and "legacy_budget" in budget,
                    ):
                        if recommendation_store is not None:
                            recommendation_store.sync_results([
                                row for row in manifest_rows
                                if _manifest_key(row)[:5] == tuple(map(str, (f"{series:02d}", *key)))
                            ], manifest_path=manifest_path)
                        continue
                    if tuple(map(str, (f"{series:02d}", *key))) in preserved_keys:
                        raise ValueError("보존 대상 실행이 불완전하다. 기존 경로를 덮어쓰지 않는다")
                    recovered = _load_completion_receipt(
                        _completion_receipt_path(spec, series),
                        spec=spec, panel_row=panel_row, series=series,
                        budget_id=budget["budget_id"],
                    )
                    computed = computed_attempts.get(tuple(map(str, (f"{series:02d}", *key))))
                    recover_receipt = not recovered and computed is not None
                    if recover_receipt:
                        recovered = computed.get("result") or _recover_saved_run_rows(
                            spec=spec, panel_row=panel_row, series=series, family=entry.get("family", ""),
                            budget_id=budget["budget_id"], computed=computed,
                        )
                        if not recovered:
                            # ponytail: 부분 저장은 보존하고 중단한다. 필요하면 해당 산출물의 복구만 추가한다.
                            raise RunResultPersistenceError(
                                "모델 실행의 저장이 미완료다. 기존 파일을 보존하며 재학습하지 않는다: "
                                + computed["history_file"]
                            )
                        recovered = _validate_completion_rows(
                            recovered, spec=spec, panel_row=panel_row, series=series,
                            budget_id=budget["budget_id"],
                        )
                    if recovered:
                        candidate_rows = _merge_manifest_rows(manifest_rows, recovered)
                        if not _completed_run(
                            candidate_rows, spec=spec, panel_row=panel_row, series=series,
                            input_manifest_path=input_manifest_path,
                            expected_project_commit=project_commit,
                            compatible_project_commit=compatible_project_commit,
                            expected_environment=environment,
                            allow_compatible_history=not full_prefix and "legacy_budget" in budget,
                        ):
                            raise ValueError("Dev18 완료 영수증 산출물을 복구하지 못했다")
                        if recover_receipt:
                            _write_completion_receipt(_completion_receipt_path(spec, series), recovered)
                        manifest_rows = _replace_manifest_rows(
                            manifest_rows, recovered, manifest_path,
                        )
                        if recommendation_store is not None:
                            recommendation_store.sync_results(recovered, manifest_path=manifest_path)
                        continue
                finally:
                    execution_timing["completion_check_seconds"] += time.perf_counter() - check_started
                prior_rows = [
                    row for row in manifest_rows
                    if _manifest_key(row)[:5] == tuple(map(str, (
                        f"{series:02d}", spec["model"], spec["config_id"],
                        spec["ratio"], spec["seed"],
                    )))
                ]
                attempts_used = max(
                    (int(row.get("retry_count") or 0) + 1 for row in prior_rows),
                    default=0,
                )
                attempts_used = max(attempts_used, prior_attempts.get(
                    tuple(map(str, (f"{series:02d}", *key))), 0,
                ))
                maximum_attempts = budget["failure_rules"]["maximum_total_attempts"]
                recovery_row = _authorized_oom_recovery_row(
                    prior_rows,
                    budget_id=budget["budget_id"],
                    compatible_project_commit=compatible_project_commit,
                )
                maximum_attempts = _authorized_attempt_limit(
                    maximum_attempts, recovery_row=recovery_row,
                )
                if recovery_row is not None:
                    _write_oom_recovery_receipt(
                        DEFAULT_ALLOCATOR_RECOVERY_PATH,
                        recovery_row,
                        recovery_project_commit=project_commit,
                    )
                pca_recovery = _pca_blas_recovery(
                    budget_id=budget["budget_id"], trial_key=tuple(map(str, (f"{series:02d}", *key))),
                    attempts_used=attempts_used, maximum_attempts=maximum_attempts,
                )
                if pca_recovery is not None:
                    maximum_attempts = pca_recovery["maximum_total_attempts"]
                if attempts_used >= maximum_attempts:
                    raise RuntimeError("Dev18 trial이 봉인된 최대 시도 횟수에 도달했다")
                if inputs is None:
                    inputs = load_registered_inputs(
                        spec=spec, input_manifest_path=input_manifest_path,
                        series=f"{series:02d}", data_root=data_root,
                    )
                for attempt in range(attempts_used, maximum_attempts):
                    completed = None
                    history = None
                    try:
                        with record_run_history(
                            REPOSITORY_ROOT / "experiments/01_ghl_main/logs/run_history/model_attempts",
                            identity={
                                "kind": "model_attempt", "budget_id": budget["budget_id"],
                                "project_commit": project_commit, "series": f"{series:02d}",
                                "model": spec["model"], "config_id": spec["config_id"],
                                "ratio": spec["ratio"], "seed": spec["seed"],
                                "attempt": attempt, "device": device,
                                **({"execution_recovery": pca_recovery} if pca_recovery is not None else {}),
                            },
                        ) as history:
                            completed = _run_one_spec(
                                spec, panel_row, inputs, series=series, device=device,
                                environment=environment, input_manifest_path=input_manifest_path,
                                retry_count=attempt, budget_id=budget["budget_id"], history=history,
                            )
                            history["result"] = completed
                    except Exception as error:
                        if (completed is not None or isinstance(error, RunResultPersistenceError)
                                or (history and (history.get("model_execution_complete")
                                                 or has_run_persistence_failure(history)))):
                            # 저장·이력 오류로 재학습하지 않는다. 완료 영수증은 다음 명령에서 복구한다.
                            raise
                        failed = [{
                            "series": f"{series:02d}", "family": inputs["family"],
                            "tier": spec["tier"], "model": spec["model"],
                            "config_id": spec["config_id"], "physical_ratio": spec["ratio"],
                            "seed": spec["seed"], "score_variant": variant,
                            "primary_score": str(
                                variant in panel_row["primary_score_variants"]
                            ).lower(),
                            "status": "failed",
                            "status_reason": f"{type(error).__name__}: {error}",
                            "score_file": "", "score_sha256": "", "metadata_file": "",
                            "metadata_sha256": "", "budget_id": budget["budget_id"],
                            "retry_count": attempt,
                        } for variant in _expected_variants(panel_row)]
                        manifest_rows = _replace_manifest_rows(manifest_rows, failed, manifest_path)
                        if recommendation_store is not None:
                            recommendation_store.sync_results(failed, manifest_path=manifest_path)
                        if attempt + 1 == maximum_attempts:
                            raise
                        continue
                    finally:
                        if history and history["elapsed_seconds"] is not None:
                            execution_timing["model_attempt_seconds"] += history["elapsed_seconds"]
                    manifest_rows = _replace_manifest_rows(manifest_rows, completed, manifest_path)
                    if recommendation_store is not None:
                        recommendation_store.sync_results(completed, manifest_path=manifest_path)
                    break
                _require_same_worktree(project_commit)
        return manifest_rows


def _load_dev18_labels(entry: dict, data_root) -> numpy.ndarray:
    import pandas

    path = Path(data_root) / entry["source_directory"] / entry["name"]
    labels = pandas.read_csv(path, usecols=["Label"])["Label"].to_numpy(dtype=int)
    labels = labels[entry["training_boundary"]:]
    if not numpy.all(numpy.isin(labels, [0, 1])) or not numpy.any(labels == 1):
        raise ValueError(f"Dev18 {entry['series']} test label이 VUS-PR 계약을 어겼다")
    return labels


_SCORE_WORKER_LABELS = {}


def _initialize_score_worker(labels_by_series) -> None:
    global _SCORE_WORKER_LABELS
    _SCORE_WORKER_LABELS = labels_by_series


def _detect_available_memory_bytes() -> int | None:
    candidates = []
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    for limit_path, used_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        try:
            limit_text = limit_path.read_text(encoding="utf-8").strip()
            if limit_text == "max":
                continue
            limit = int(limit_text)
            used = int(used_path.read_text(encoding="utf-8").strip())
            if 0 < limit < 2 ** 60:
                candidates.append(max(0, limit - used))
        except (OSError, ValueError):
            continue
    return min(candidates) if candidates else None


def _collect_scoring_cpu_environment() -> dict:
    import platform

    environment = {
        "cpu_model": None, "cpu_model_source": None, "os_logical_cpu_count": None,
        "affinity_cpu_ids": None, "affinity_cpu_count": None, "cgroup_cpu_quota": None,
    }
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            name, separator, value = line.partition(":")
            if separator and name.strip().lower() in {"model name", "hardware"} and value.strip():
                environment.update(cpu_model=value.strip(), cpu_model_source="/proc/cpuinfo")
                break
    except OSError:
        pass
    if environment["cpu_model"] is None:
        try:
            processor = platform.processor()
            if processor:
                environment.update(cpu_model=processor, cpu_model_source="platform.processor")
        except OSError:
            pass
    try:
        environment["os_logical_cpu_count"] = os.cpu_count()
    except OSError:
        pass
    try:
        affinity = sorted(os.sched_getaffinity(0))
        environment.update(affinity_cpu_ids=affinity, affinity_cpu_count=len(affinity))
    except (AttributeError, OSError):
        pass

    cgroup_roots = {
        2: [Path("/sys/fs/cgroup")],
        1: [Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu,cpuacct"),
            Path("/sys/fs/cgroup/cpuacct,cpu")],
    }
    directories = {version: list(roots) for version, roots in cgroup_roots.items()}
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8", errors="replace").splitlines():
            _, controllers, group = line.split(":", 2)
            version = 2 if controllers == "" else 1 if "cpu" in controllers.split(",") else None
            if version is None:
                continue
            for root in cgroup_roots[version]:
                directory = root / group.lstrip("/")
                while root in directory.parents:
                    directories[version].append(directory)
                    directory = directory.parent
    except (OSError, ValueError):
        pass
    for version, paths in directories.items():
        limits = []
        for directory in dict.fromkeys(paths):
            try:
                quota_path = directory / ("cpu.max" if version == 2 else "cpu.cfs_quota_us")
                if version == 2:
                    quota_text, period_text = quota_path.read_text(encoding="utf-8", errors="replace").split()
                else:
                    quota_text = quota_path.read_text(encoding="utf-8", errors="replace").strip()
                    period_text = (directory / "cpu.cfs_period_us").read_text(
                        encoding="utf-8", errors="replace",
                    ).strip()
                quota = None if quota_text in {"max", "-1"} else int(quota_text)
                period = int(period_text)
                if period <= 0 or (quota is not None and quota <= 0):
                    continue
                limits.append({
                    "version": version, "source": str(quota_path),
                    "quota_microseconds": quota, "period_microseconds": period,
                    "limit_cpu_count": quota / period if quota is not None else None,
                    "unlimited": quota is None,
                })
            except (OSError, ValueError):
                continue
        if limits:
            environment["cgroup_cpu_quota"] = {
                **min(limits, key=lambda item: item["limit_cpu_count"]
                      if item["limit_cpu_count"] is not None else math.inf),
                "scope": "minimum_readable_process_cgroup_and_ancestor_quota",
            }
            break
    environment["null_reasons"] = {
        name: "not_exposed_or_unreadable_by_runtime"
        for name, value in environment.items() if value is None
    }
    return environment


def _record_scoring_environment(history, *, workers, worker_count, pending_count) -> None:
    if history is None:
        return
    history["scoring_environment"] = {
        "requested_workers": workers, "actual_workers": worker_count,
        "worker_count_scope": "resolved_concurrency_before_dispatch",
        "pending_physical_scores": pending_count,
        "scoring_status": "workers_selected" if pending_count else "no_new_scoring",
        "cpu": _collect_scoring_cpu_environment(),
    }
    save_run_history(history)


def _resolve_score_workers(
    workers, *, pending_count, cpu_count=None, available_memory_bytes=None,
) -> int:
    if type(workers) is not int or workers < 0:
        raise ValueError("workers는 0 이상의 정수여야 한다")
    if type(pending_count) is not int or pending_count < 1:
        raise ValueError("pending_count는 1 이상의 정수여야 한다")
    if cpu_count is None:
        cpu_count = available_cpu_count()
    cpu_count = max(1, int(cpu_count))
    if available_memory_bytes is None:
        available_memory_bytes = _detect_available_memory_bytes()
    if available_memory_bytes is None:
        memory_limit = cpu_count
    else:
        gibibyte = 1024 ** 3
        memory_limit = max(1, (int(available_memory_bytes) - 2 * gibibyte) // gibibyte)
    safe_limit = min(cpu_count, memory_limit, pending_count)
    return safe_limit if workers == 0 else min(workers, safe_limit)


@lru_cache(maxsize=32)
def _compatible_score_commits(project_commit, repository_root) -> tuple:
    from tests.checks.validate_resource_resume import RESOURCE_RESUME_SOURCE, resource_resume_compatible

    if not resource_resume_compatible(RESOURCE_RESUME_SOURCE, project_commit, repository_root):
        return (project_commit,)
    if project_commit == RESOURCE_RESUME_SOURCE:
        return (project_commit,)
    try:
        previous_commits = subprocess.check_output([
            "git", "rev-list", f"{RESOURCE_RESUME_SOURCE}..{project_commit}",
        ], cwd=repository_root, text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        previous_commits = []
    return tuple(dict.fromkeys((RESOURCE_RESUME_SOURCE, project_commit, *(
        commit for commit in previous_commits
        if resource_resume_compatible(commit, project_commit, repository_root)
    ))))


def _score_checkpoint_path(directory, identity: dict, *, normalize_commit=True) -> Path:
    path_identity = identity
    if normalize_commit and "project_commit" in identity:
        path_identity = {**identity, "project_commit": _compatible_score_commits(
            identity["project_commit"], str(REPOSITORY_ROOT),
        )[0]}
    digest = hashlib.sha256(_json(path_identity).encode("utf-8")).hexdigest()
    return Path(directory) / f"{digest}.json"


def _checkpoint_payload(identity: dict, vus_pr_value: float) -> dict:
    payload = {"identity": identity, "vus_pr": float(vus_pr_value)}
    return {
        **payload,
        "payload_sha256": hashlib.sha256(
            _json(payload).encode("utf-8")
        ).hexdigest(),
    }


def _write_score_checkpoint(path, identity: dict, vus_pr_value: float) -> None:
    if not numpy.isfinite(vus_pr_value):
        raise ValueError("VUS-PR checkpoint 값은 유한해야 한다")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(
            _checkpoint_payload(identity, vus_pr_value),
            ensure_ascii=False, sort_keys=True, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_score_checkpoint(path, identity: dict) -> float | None:
    path = Path(path)
    if not path.is_file():
        commits = _compatible_score_commits(identity["project_commit"], str(REPOSITORY_ROOT)) \
            if "project_commit" in identity else ()
        for commit in commits:
            candidate = _score_checkpoint_path(
                path.parent, {**identity, "project_commit": commit}, normalize_commit=False,
            )
            if candidate.is_file():
                path = candidate
                break
        else:
            return None
    try:
        payload = _read_json(path)
        vus_pr_value = payload["vus_pr"]
        saved_identity = payload["identity"]
        if not isinstance(saved_identity, dict):
            raise ValueError("checkpoint identity must be an object")
        expected = _checkpoint_payload(saved_identity, vus_pr_value)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"VUS-PR checkpoint를 읽을 수 없다: {path.name}") from error
    matching_identity = saved_identity == identity
    if not matching_identity and "project_commit" in identity and saved_identity.keys() == identity.keys():
        from tests.checks.validate_resource_resume import resource_resume_compatible

        matching_identity = (
            {**saved_identity, "project_commit": identity["project_commit"]} == identity
            and resource_resume_compatible(saved_identity["project_commit"], identity["project_commit"], REPOSITORY_ROOT)
        )
    if (
        payload != expected
        or not matching_identity
        or type(vus_pr_value) not in (int, float)
        or not numpy.isfinite(vus_pr_value)
    ):
        raise ValueError(f"VUS-PR checkpoint가 손상됐다: {path.name}")
    return float(vus_pr_value)


def _score_primary_row(task: dict) -> dict:
    from src.채점기.parser import load_and_validate_score

    manifest_row = task["manifest_row"]
    score_path = (REPOSITORY_ROOT / manifest_row["score_file"]).resolve()
    metadata_path = (REPOSITORY_ROOT / manifest_row["metadata_file"]).resolve()
    for path in (score_path, metadata_path):
        try:
            path.relative_to(REPOSITORY_ROOT)
        except ValueError as error:
            raise ValueError("Dev18 score manifest가 저장소 밖 파일을 가리킨다") from error
    if file_sha256(score_path) != manifest_row["score_sha256"]:
        raise ValueError("Dev18 primary score SHA-256이 manifest와 다르다")
    if (
        metadata_path != score_path.with_suffix(".meta.json")
        or file_sha256(metadata_path) != manifest_row["metadata_sha256"]
    ):
        raise ValueError("Dev18 primary metadata SHA-256이 manifest와 다르다")

    scores, info, metadata = load_and_validate_score(score_path)
    series = manifest_row["series"]
    expected_file_info = {
        "dataset": "DEV18", "series": int(series),
        "model": manifest_row["model"], "tier": manifest_row["tier"],
        "ratio": int(manifest_row["physical_ratio"]),
        "seed": int(manifest_row["seed"]), "smoothing_kind": "raw",
        "norm_kind": "trainnorm", "channels": False,
    }
    if info != expected_file_info:
        raise ValueError("Dev18 primary score 파일명이 manifest와 다르다")
    metadata_fields = {
        "dataset": "DEV18", "series": int(series),
        "model": manifest_row["model"], "tier": manifest_row["tier"],
        "ratio": int(manifest_row["physical_ratio"]),
        "seed": int(manifest_row["seed"]),
        "config_id": manifest_row["config_id"],
        "score_variant": manifest_row["score_variant"] or None,
    }
    if any(metadata.get(field) != value for field, value in metadata_fields.items()):
        raise ValueError("Dev18 primary metadata 신원이 manifest와 다르다")
    if manifest_row["family"] != task["entry"]["family"]:
        raise ValueError("Dev18 family가 봉인 manifest와 다르다")
    start, end = metadata["label_slice"]
    labels = _SCORE_WORKER_LABELS[series][slice(start, end)]
    if len(labels) != len(scores):
        raise ValueError("Dev18 score-label 정렬 길이가 다르다")

    label_sha256 = hashlib.sha256(
        numpy.ascontiguousarray(labels, dtype=numpy.uint8).tobytes()
    ).hexdigest()
    identity = {
        "schema_version": 1,
        "manifest_key": list(_manifest_key(manifest_row)),
        "budget_id": task["budget_id"],
        "budget_sha256": task["budget_sha256"],
        "input_manifest_sha256": task["input_manifest_sha256"],
        "environment_sha256": task["environment_sha256"],
        "project_commit": task["project_commit"],
        "score_file": manifest_row["score_file"],
        "score_sha256": manifest_row["score_sha256"],
        "metadata_file": manifest_row["metadata_file"],
        "metadata_sha256": manifest_row["metadata_sha256"],
        "evaluator_sha256": task["evaluator_sha256"],
        "ell_max_id": task["ell_max_id"],
        "l_max_samples": task["l_max_samples"],
        "n_thresholds": task["n_thresholds"],
        "label_sha256": label_sha256,
    }
    checkpoint_path = _score_checkpoint_path(task["checkpoint_directory"], identity)
    vus_pr_value = _load_score_checkpoint(checkpoint_path, identity)
    reused = vus_pr_value is not None
    if vus_pr_value is None:
        vus_pr_value = vus_pr(
            scores, labels, task["l_max_samples"],
            n_thresholds=task["n_thresholds"],
        )
        _write_score_checkpoint(checkpoint_path, identity, vus_pr_value)
    return {
        "manifest_key": list(_manifest_key(manifest_row)),
        "normalization": info["norm_kind"],
        "vus_pr": vus_pr_value,
        "reused": reused,
    }


def _score_primary_rows(tasks, labels_by_series, workers, *, scoring_history=None) -> dict:
    tasks = sorted(
        tasks,
        key=lambda task: (-task["estimated_cost"], _manifest_key(task["manifest_row"])),
    )
    worker_count = _resolve_score_workers(workers, pending_count=len(tasks)) if tasks else 0
    _record_scoring_environment(
        scoring_history, workers=workers, worker_count=worker_count, pending_count=len(tasks),
    )
    if not tasks:
        return {}
    print(
        f"VUS-PR 채점 시작: physical={len(tasks)}, workers={worker_count}",
        flush=True,
    )
    results = {}
    reused_count = 0

    def record(result):
        nonlocal reused_count
        key = tuple(result["manifest_key"])
        if key in results:
            raise ValueError("VUS-PR worker가 중복 manifest key를 반환했다")
        results[key] = result
        reused_count += int(result["reused"])
        completed = len(results)
        if completed == 1 or completed % 10 == 0 or completed == len(tasks):
            print(
                f"VUS-PR 진행: {completed}/{len(tasks)} "
                f"(checkpoint 재사용 {reused_count})",
                flush=True,
            )

    if worker_count == 1:
        _initialize_score_worker(labels_by_series)
        for task in tasks:
            record(_score_primary_row(task))
    else:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_score_worker,
            initargs=(labels_by_series,),
        )
        futures = []
        try:
            futures = [executor.submit(_score_primary_row, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                record(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    return results


def _expand_primary_score(
    manifest_row, *, logical_ratios, normalization, vus_pr_value,
    evaluator_sha256, ell_max_id,
) -> list[dict]:
    return [{
        "series": manifest_row["series"], "family": manifest_row["family"],
        "tier": manifest_row["tier"], "model": manifest_row["model"],
        "config_id": manifest_row["config_id"], "ratio": ratio,
        "seed": int(manifest_row["seed"]),
        "score_variant": manifest_row["score_variant"],
        "normalization": normalization, "vus_pr": vus_pr_value,
        "score_file": manifest_row["score_file"],
        "score_sha256": manifest_row["score_sha256"],
        "evaluator_sha256": evaluator_sha256,
        "ell_max_id": ell_max_id,
        "status": "complete", "status_reason": "",
    } for ratio in logical_ratios]


def build_trial_score_ledger(
    manifest_rows, *, data_root=DEFAULT_DATA_ROOT, output_path=None, workers=1,
    checkpoint_directory=DEFAULT_VUS_CHECKPOINT_DIRECTORY, project_commit=None,
    budget=None, reused_ledger=(), scoring_history=None,
) -> list[dict]:
    """완료된 primary score만 라벨에 연결해 18×89 채점 원표를 만든다."""
    budget = budget or _read_json(DEFAULT_BUDGET_PATH)
    input_manifest_path = REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
    input_manifest_sha256 = file_sha256(input_manifest_path)
    environment_sha256 = file_sha256(REPOSITORY_ROOT / "configs" / "environment.yaml")
    project_commit = project_commit or _git_head()
    ell_max = _validate_ell_max(
        DEFAULT_ELL_MAX_PATH, input_manifest_sha256,
    )
    evaluator_path = REPOSITORY_ROOT / "src" / "채점기" / "vus_pr.py"
    vus_report = validate_vus_evidence(DEFAULT_VUS_REPORT_PATH, evaluator_path)
    import yaml

    manifest = yaml.safe_load(input_manifest_path.read_text(encoding="utf-8"))
    entries = {entry["series"]: entry for entry in manifest["datasets"]["DEV18"]["files"]}
    if budget.get("series_ids"):
        entries = {series: entry for series, entry in entries.items()
                   if series in budget["series_ids"]}
    l_max_by_series = {row["series"]: row["l_max_samples"] for row in ell_max["series"]}
    panel_by_key = {
        (row["model"], row["config_id"], str(row["physical_ratio"]), str(row["seed"])): row
        for row in budget["execution_panel"]
    }
    primary_rows = _validate_primary_manifest_rows(
        manifest_rows, budget, tuple(sorted(entries)),
    )
    from tests.ghl_main.run_registered_models import _verify_manifest_file

    reused = _reuse_trial_scores(
        primary_rows, reused_ledger, evaluator_sha256=vus_report["evaluator_sha256"],
        ell_max_id=ell_max["ell_max_id"],
    )
    tasks = []
    for manifest_row in primary_rows:
        if _manifest_key(manifest_row) in reused:
            continue
        series = manifest_row["series"]
        entry = entries[series]
        tasks.append({
            "manifest_row": manifest_row,
            "entry": entry,
            "budget_id": budget["budget_id"],
            "budget_sha256": budget["budget_sha256"],
            "input_manifest_sha256": input_manifest_sha256,
            "environment_sha256": environment_sha256,
            "project_commit": project_commit,
            "evaluator_sha256": vus_report["evaluator_sha256"],
            "ell_max_id": ell_max["ell_max_id"],
            "l_max_samples": l_max_by_series[series],
            "n_thresholds": vus_report["n_thresholds"],
            "checkpoint_directory": str(Path(checkpoint_directory)),
            "estimated_cost": (
                entry["row_count"] - entry["training_boundary"]
            ) * (l_max_by_series[series] + 1),
        })
    needed_series = {task["manifest_row"]["series"] for task in tasks}
    for series in needed_series:
        entry = entries[series]
        _verify_manifest_file(
            Path(data_root) / entry["source_directory"] / entry["name"], entry,
        )
    labels_by_series = {
        series: _load_dev18_labels(entries[series], data_root) for series in needed_series
    }
    scored = dict(reused)
    if tasks or scoring_history is not None:
        scored.update(_score_primary_rows(
            tasks, labels_by_series, workers,
            **({"scoring_history": scoring_history} if scoring_history is not None else {}),
        ))
    ledger = []
    for manifest_row in sorted(primary_rows, key=_manifest_key):
        panel = panel_by_key[(
            manifest_row["model"], manifest_row["config_id"],
            manifest_row["physical_ratio"], manifest_row["seed"],
        )]
        result = scored[_manifest_key(manifest_row)]
        ledger.extend(_expand_primary_score(
            manifest_row,
            logical_ratios=panel["logical_ratios"],
            normalization=result["normalization"],
            vus_pr_value=result["vus_pr"],
            evaluator_sha256=vus_report["evaluator_sha256"],
            ell_max_id=ell_max["ell_max_id"],
        ))
    expected_rows = budget.get("expected_ledger_rows", len(entries) * budget["primary_logical_score_row_count"])
    if len(ledger) != expected_rows:
        raise ValueError(f"Dev18 tuning ledger 행 수가 다르다: {len(ledger)} != {expected_rows}")
    if output_path is not None:
        output_path = Path(output_path)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        _write_csv(temporary_path, ledger)
        temporary_path.replace(output_path)
    return ledger


def _reuse_trial_scores(manifest_rows, ledger, *, evaluator_sha256, ell_max_id):
    """같은 점수 파일과 채점 신원을 가진 완료값만 물리 key에 다시 연결한다."""
    by_score = {}
    for row in ledger:
        if (row["status"] != "complete" or row["evaluator_sha256"] != evaluator_sha256
                or row["ell_max_id"] != ell_max_id or row["normalization"] != "trainnorm"):
            raise ValueError("재사용 ledger의 채점 신원이 다르다")
        value = float(row["vus_pr"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("재사용 VUS-PR 값이 잘못됐다")
        key = (row["score_file"], row["score_sha256"])
        result = {"normalization": row["normalization"], "vus_pr": value, "reused": True}
        if key in by_score and by_score[key] != result:
            raise ValueError("같은 점수 파일의 재사용 채점값이 서로 다르다")
        by_score[key] = result
    return {_manifest_key(row): by_score[(row["score_file"], row["score_sha256"])]
            for row in manifest_rows
            if (row["score_file"], row["score_sha256"]) in by_score}


def _policy_csv_rows(rows):
    return _serializable_rows(rows)


def finish_selection_from_ledger(
    ledger_path, *, result_directory=DEFAULT_RESULT_DIRECTORY,
) -> dict:
    """완료 ledger만 검증해 선택표·membership·그림을 다시 만든다."""
    project_commit = _git_head()
    _require_same_worktree(project_commit)
    registry, registry_sha256 = load_model_registry_with_sha()
    validate_primary_hpo_seal(registry)
    budget = _read_json(DEFAULT_BUDGET_PATH)
    selection_seal = registry["selection"]
    budget_core = {
        field: value for field, value in budget.items()
        if field not in {
            "budget_sha256", "budget_id", "seal_status",
            "execution_readiness_status", "pending_execution_models", "attestation",
        }
    }
    budget_sha256 = hashlib.sha256(_json(budget_core).encode("utf-8")).hexdigest()
    if (
        budget.get("budget_sha256") != budget_sha256
        or budget.get("budget_id") != "b" + budget_sha256[:12]
        or budget.get("budget_id") != selection_seal["budget_id"]
        or budget.get("primary_hpo_regime") != selection_seal["primary_hpo_regime"]
        or budget.get("selection_rule_id") != selection_seal["selection_rule_id"]
        or budget.get("registry_space_sha256") != registry_space_sha256(registry)
        or budget.get("seal_status") != "sealed"
        or budget.get("execution_readiness_status") != "ready"
        or budget.get("pending_execution_models") != []
        or budget.get("attestation", {}).get("config_registry_sha256")
        != registry_sha256
    ):
        raise ValueError("Dev18 budget과 현재 registry 봉인이 다르다")

    ledger_path = Path(ledger_path)
    ledger_sha256 = file_sha256(ledger_path)
    input_manifest_sha256 = file_sha256(
        REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
    )
    ell_max = _validate_ell_max(DEFAULT_ELL_MAX_PATH, input_manifest_sha256)
    vus_report = validate_vus_evidence(
        DEFAULT_VUS_REPORT_PATH, REPOSITORY_ROOT / "src" / "채점기" / "vus_pr.py",
    )
    ledger = load_trial_score_ledger(
        ledger_path, budget,
        expected_evaluator_sha256=vus_report["evaluator_sha256"],
        expected_ell_max_id=ell_max["ell_max_id"],
    )
    selection = select_tuning_policies(
        ledger, registry, budget, evaluator_sha256=ledger[0]["evaluator_sha256"],
    )
    membership = build_final_membership_rows(selection, registry)
    expected_membership_count = (len(registry["models"]) + 2 * len(selection["tier_fixed"])) * len(SUPPORTED_RATIO_PERCENTS) * len(FINAL_SPLIT_ROLES)
    if len(membership) != expected_membership_count:
        raise ValueError(f"final membership 행 수가 다르다: {len(membership)} != {expected_membership_count}")

    result_directory = Path(result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)
    from src.common.load_final_membership import FIELDS, load_final_membership

    membership_path = result_directory / "final_policy_membership.csv"
    temporary_membership_path = membership_path.with_name(
        f".{membership_path.name}.tmp"
    )
    _write_csv(temporary_membership_path, membership, FIELDS)
    membership_sha256, loaded = load_final_membership(
        temporary_membership_path, registry,
    )
    if len(loaded) != len(membership):
        raise ValueError("final membership 저장 행 수가 다르다")
    temporary_membership_path.replace(membership_path)
    _write_csv(
        result_directory / "model_fixed_policy.csv",
        _policy_csv_rows(selection["model_fixed"]),
    )
    _write_csv(
        result_directory / "tier_fixed_policy.csv",
        _policy_csv_rows(selection["tier_fixed"]),
    )
    write_selection_reports(ledger, selection, result_directory)
    if file_sha256(ledger_path) != ledger_sha256:
        raise RuntimeError("Dev18 selection 도중 입력 ledger가 바뀌었다")
    _require_same_worktree(project_commit)
    selected_path = [{
        "tier": row["tier"], "ratio": row["ratio"],
        "model": row["selected_model"], "config_id": row["config_id"],
        "status": row["selection_status"],
    } for row in selection["tier_adaptive"]]
    return {
        "status": "complete", "budget_id": budget["budget_id"],
        "selection_rule_id": RATIO_ADAPTIVE_SELECTION_RULE_ID,
        "project_commit": project_commit, "ledger_path": str(ledger_path.resolve()),
        "ledger_sha256": ledger_sha256, "ledger_rows": len(ledger),
        "membership_rows": len(membership),
        "final_policy_membership_sha256": membership_sha256,
        "selected_path": selected_path,
        "result_rows": {
            "model_fixed_policy.csv": len(selection["model_fixed"]),
            "tier_fixed_policy.csv": len(selection["tier_fixed"]),
            "ratio_adaptive_selection.csv": len(selection["tier_adaptive"]),
            "tier_ratio_candidate_audit.csv": len(selection["candidate_audit"]),
            "tier_policy_transitions.csv": len(selection["policy_transitions"]),
            "final_policy_membership.csv": len(membership),
        },
        "result_directory": str(result_directory),
    }


def finish_tuning(
    manifest_rows, *, data_root=DEFAULT_DATA_ROOT, workers=1,
    checkpoint_directory=DEFAULT_VUS_CHECKPOINT_DIRECTORY,
    require_execution_environment=True,
) -> dict:
    """완료 score를 채점하고 정책표·membership·검토용 CSV/PNG를 한 폴더에 쓴다."""
    project_commit = _git_head()
    prepare_tuning(
        data_root=data_root, require_clean=True,
        require_execution_environment=require_execution_environment,
    )
    _require_same_worktree(project_commit)
    registry, _ = load_model_registry_with_sha()
    budget = _read_json(DEFAULT_BUDGET_PATH)
    DEFAULT_RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    ledger_path = DEFAULT_RESULT_DIRECTORY / "dev18_trial_score_ledger.csv"
    ledger = build_trial_score_ledger(
        manifest_rows, data_root=data_root, output_path=ledger_path,
        workers=workers, checkpoint_directory=checkpoint_directory,
        project_commit=project_commit,
    )
    evaluator_sha256 = ledger[0]["evaluator_sha256"]
    selection = select_tuning_policies(
        ledger, registry, budget, evaluator_sha256=evaluator_sha256,
    )
    model_path = DEFAULT_RESULT_DIRECTORY / "model_fixed_policy.csv"
    tier_path = DEFAULT_RESULT_DIRECTORY / "tier_fixed_policy.csv"
    membership_path = DEFAULT_RESULT_DIRECTORY / "final_policy_membership.csv"
    _write_csv(model_path, _policy_csv_rows(selection["model_fixed"]))
    _write_csv(tier_path, _policy_csv_rows(selection["tier_fixed"]))
    membership = build_final_membership_rows(selection, registry)
    from src.common.load_final_membership import FIELDS, load_final_membership

    _write_csv(membership_path, membership, FIELDS)
    membership_sha256, loaded = load_final_membership(membership_path, registry)
    if len(loaded) != len(membership):
        raise ValueError("final membership 저장 행 수가 다르다")
    write_selection_reports(ledger, selection, DEFAULT_RESULT_DIRECTORY)
    _require_same_worktree(project_commit)
    return {
        "status": "complete", "budget_id": budget["budget_id"],
        "project_commit": project_commit,
        "ledger_rows": len(ledger), "selected_models": {
            row["tier"]: row["selected_model"] for row in selection["tier_fixed"]
        },
        "final_policy_membership_sha256": membership_sha256,
        "result_directory": str(DEFAULT_RESULT_DIRECTORY),
    }


def run_tuning(
    *, data_root=DEFAULT_DATA_ROOT, device="cuda", remote_execution=False,
) -> dict:
    manifest_rows = execute_panel(
        data_root=data_root, device=device, remote_execution=remote_execution,
    )
    return finish_tuning(manifest_rows, data_root=data_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--remote-execution", action="store_true")
    arguments = parser.parse_args()
    if arguments.prepare:
        result = prepare_tuning(data_root=arguments.data_root)
    elif arguments.smoke:
        _require_cuda_or_remote(
            device=arguments.device, remote_execution=arguments.remote_execution,
        )
        result = smoke_real_data_gate(data_root=arguments.data_root)
    else:
        result = run_tuning(
            data_root=arguments.data_root, device=arguments.device,
            remote_execution=arguments.remote_execution,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
