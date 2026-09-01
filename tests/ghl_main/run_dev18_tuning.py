"""Dev18 exact panel을 실행하고 family-LOFO 선택표와 그림을 만든다."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tracemalloc
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib
import numpy

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot

WINDOWS_MALGUN_FONT_PATHS = tuple(dict.fromkeys((
    Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf",
    Path("C:/Windows/Fonts/malgun.ttf"),
)))


def _find_installed_font(family: str):
    normalized = family.replace(" ", "").casefold()
    for entry in font_manager.fontManager.ttflist:
        if entry.name.replace(" ", "").casefold() == normalized:
            path = Path(entry.fname)
            if path.is_file():
                return path
    return None


def _apply_plot_font(path: Path, source: str) -> dict:
    try:
        font_manager.fontManager.addfont(path)
        family = font_manager.FontProperties(fname=path).get_name()
    except Exception as error:
        raise ValueError(f"글꼴 파일을 읽을 수 없다: {path}") from error
    matplotlib.rcParams["font.family"] = family
    matplotlib.rcParams["axes.unicode_minus"] = False
    return {"source": source, "family": family, "path": str(path)}


def configure_plot_font() -> dict:
    """명시 경로, Malgun Gothic, NanumGothic 순으로 그림 글꼴을 고른다."""
    explicit = os.environ.get("TSAD_KOREAN_FONT_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"TSAD_KOREAN_FONT_PATH 글꼴 파일이 없다: {path}"
            )
        return _apply_plot_font(path, "environment")

    for path in WINDOWS_MALGUN_FONT_PATHS:
        if path.is_file():
            return _apply_plot_font(path, "windows_malgun")
    malgun = _find_installed_font("Malgun Gothic")
    if malgun is not None:
        return _apply_plot_font(malgun, "windows_malgun")
    nanum = _find_installed_font("NanumGothic")
    if nanum is not None:
        return _apply_plot_font(nanum, "nanum_gothic")

    matplotlib.rcParams["axes.unicode_minus"] = False
    families = list(matplotlib.rcParams["font.family"])
    return {
        "source": "matplotlib_default", "family": families[0], "path": None,
    }

from src.common.equal_trial_budget import build_equal_trial_budget, registry_space_sha256
from src.common.execution_identity import file_sha256, load_input_manifest_role
from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
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
    "MWVAR": 1,
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
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    report = _read_json(path)
    model = registry["models"][model_name]
    input_manifest_sha256 = file_sha256(
        REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
    )
    adapter_path = (
        REPOSITORY_ROOT / "src" / "models" / "tier3"
        / ("time_rcd.py" if model_name == "TimeRCD" else "tspulse.py")
    )
    expected_heads = (
        {"score"} if model_name == "TimeRCD"
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
    require_execution_environment=True,
) -> dict:
    """기존 봉인을 다시 계산하지 않고 tuning 시작 조건만 대조한다."""
    if require_clean:
        _require_clean_worktree()
    registry, registry_sha256 = load_model_registry_with_sha()
    validate_primary_hpo_seal(registry)
    _, _, feasibility = _load_current_feasibility(
        REPOSITORY_ROOT, SNAPSHOT_DIRECTORY, registry_sha256,
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
        "expected_primary_ledger_rows": 18 * budget["primary_logical_score_row_count"],
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


def load_structural_block_evidence(path, registry: dict) -> dict:
    """봉인 feasibility ledger에서 low-pair series만 요약한다."""
    blocked = defaultdict(set)
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            model = registry["models"].get(row["model"], {})
            heads = model.get("fixed", {}).get("heads")
            if row["status"] != "structurally_infeasible" or heads is None:
                continue
            pair_count = json.loads(row["derived_json"]).get("pair_count")
            if pair_count is not None and int(pair_count) < int(heads):
                blocked[row["model"]].add(row["series"])
    return {
        model: {"heads": registry["models"][model]["fixed"]["heads"],
                "blocked_series": sorted(series)}
        for model, series in blocked.items()
    }


def load_trial_score_ledger(path, budget: dict) -> list[dict]:
    """완료된 원표의 schema와 budget 논리 키만 확인해 읽는다."""
    with Path(path).open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != TRIAL_SCORE_LEDGER_FIELDS:
            raise ValueError("Dev18 trial ledger header가 다르다")
        rows = []
        for row in reader:
            if row["status"] != "complete":
                raise ValueError("Dev18 trial ledger에는 complete 행만 있어야 한다")
            rows.append({**row, "ratio": int(row["ratio"]), "seed": int(row["seed"]),
                         "vus_pr": float(row["vus_pr"])})

    keys = [
        (row["series"], row["model"], row["config_id"], row["ratio"],
         row["seed"], row["score_variant"])
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Dev18 trial ledger에 duplicate 논리 키가 있다")
    if len({row["evaluator_sha256"] for row in rows}) != 1:
        raise ValueError("Dev18 trial ledger의 evaluator SHA가 하나가 아니다")
    if len({row["ell_max_id"] for row in rows}) != 1:
        raise ValueError("Dev18 trial ledger의 ell_max ID가 하나가 아니다")
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
            for config_id in panel["selected_config_ids"]
            for ratio in panel["logical_ratios"]
            for seed in panel.get("seeds", budget["seeds"])
            for variant in panel["primary_score_variants"]
        }
        for panel in budget["model_panels"]
    }
    expected_keys = {
        (series, model, config_id, ratio, seed, variant)
        for series in {row["series"] for row in rows}
        for model, entries in expected_by_model.items()
        for config_id, ratio, seed, variant in entries
    }
    if set(keys) != expected_keys:
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
        for point in points:
            axis.annotate(
                point["model"],
                (point["ratio"], point["selection_score"]),
                xytext=(0, label_offsets[tier]), textcoords="offset points",
                ha="center", va="bottom" if label_offsets[tier] > 0 else "top",
                fontsize=7, color=colors[tier],
            )
    axis.axhline(
        data["pca_reference_vus_pr"], color="#7f7f7f", linestyle="--",
        linewidth=1.1, label="PCA_LEGACY q100 reference only",
    )
    for point in data["unavailable"]:
        axis.text(
            point["ratio"], 0.03, "unavailable",
            transform=axis.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7, color="#7f7f7f",
        )
    axis.set(
        title="Ratio-adaptive Tier representatives",
        xlabel="Normal prefix (%)", ylabel="Family-LOFO VUS-PR",
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
    structural_block_evidence=None,
) -> dict:
    """모델별 고정 recipe를 유지한 채 비율마다 Tier 대표를 고른다."""
    seed_rows = _seed_means(rows)
    tolerance = float(budget["tie_rule"]["tolerance"])
    structural_block_evidence = structural_block_evidence or {}
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
                    evidence = structural_block_evidence.get(model_name)
                    if evidence:
                        audit["reason"] = (
                            f"pair_count < heads {evidence['heads']}; 차단 series: "
                            + ", ".join(evidence["blocked_series"])
                        )
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
    structural_block_evidence=None,
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
        structural_block_evidence=(
            load_structural_block_evidence(DEFAULT_FEASIBILITY_LEDGER_PATH, registry)
            if structural_block_evidence is None else structural_block_evidence
        ),
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
    if write_plots:
        configure_plot_font()
    seed_rows = _seed_means(rows)
    fixed = {row["model"]: row for row in selection["model_fixed"]}
    models = sorted(row["model"] for row in selection["model_fixed"])
    model_tables = {}
    for model_name in models:
        model_rows = []
        combinations = sorted({
            (row["config_id"], row["ratio"], row["score_variant"])
            for row in seed_rows if row["model"] == model_name
        })
        for config_id, ratio, variant in combinations:
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
                    fixed[model_name]["selection_status"] == "selected"
                    and fixed[model_name]["config_id"] == config_id
                    and fixed[model_name]["score_variant"] == variant
                ),
                "selected_hyperparameters": fixed[model_name]["hyperparameters"],
                "selection_reason": fixed[model_name]["selection_reason"],
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
            axis.text(
                0.5, 0.58, f"{model_name}: unavailable", ha="center", va="center",
                fontsize=16, transform=axis.transAxes,
            )
            axis.text(
                0.5, 0.42, fixed[model_name]["selection_reason"],
                ha="center", va="center", wrap=True, transform=axis.transAxes,
            )
        else:
            for config_id, variant in sorted({
                (row["config_id"], row["score_variant"]) for row in model_rows
            }):
                curve = [
                    row for row in model_rows
                    if row["config_id"] == config_id and row["score_variant"] == variant
                ]
                axis.plot(
                    [row["ratio"] for row in curve],
                    [row["family_macro_vus_pr"] for row in curve], marker="o",
                    linewidth=2.5 if curve[0]["selected_recipe"] else 1,
                    label=f"{config_id}{':' + variant if variant else ''}",
                )
            axis.set(
                title=model_name, xlabel="Normal prefix (%)",
                ylabel="Family-macro VUS-PR",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
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
        for tier_row in tier_rows:
            curve = [
                row for row in model_tables[tier_row["model"]]
                if row["config_id"] == tier_row["fixed_config_id"]
            ]
            axis.plot(
                [row["ratio"] for row in curve],
                [row["family_macro_vus_pr"] for row in curve], marker="o",
                linewidth=3 if tier_row["selected_model"] else 1,
                label=tier_row["model"],
            )
        axis.set(title=f"Tier {tier_number}", xlabel="Normal prefix (%)", ylabel="Family-macro VUS-PR")
        axis.grid(alpha=0.25)
        axis.legend()
        axis.text(
            0, -0.27,
            f"선택: {tier_policy['selected_model']} · 파라미터: {_json(tier_policy['hyperparameters'])}\n"
            f"이유: {tier_policy['selection_reason']}",
            transform=axis.transAxes, fontsize=7, va="top", wrap=True,
        )
        figure.subplots_adjust(bottom=0.31)
        figure.tight_layout()
        figure.savefig(output_directory / f"Tier{tier_number}.png", dpi=160)
        pyplot.close(figure)

    selection_rows = _serializable_rows(selection["tier_fixed"])
    _write_csv(output_directory / "selection.csv", selection_rows)
    _write_csv(output_directory / "family_lofo.csv", selection["lofo"])
    model_summary = _serializable_rows(selection["model_fixed"])
    _write_csv(output_directory / "models.csv", model_summary)
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
    figure, axis = pyplot.subplots(figsize=(12, max(4, len(model_summary) * 0.65)))
    axis.axis("off")
    table_rows = [[
        row["model"], row["tier"], row["selection_status"], row["config_id"] or "-",
        f"{row['j_fixed']:.6f}" if row["j_fixed"] is not None else "-",
        _json(row["hyperparameters"]), row["selection_reason"],
    ] for row in selection["model_fixed"]]
    table = axis.table(
        cellText=table_rows,
        colLabels=["Model", "Tier", "Status", "Config", "VUS-PR", "Parameters", "Reason"],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    axis.set_title("Model-fixed selection", pad=18)
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
        for series in series_ids
        for panel in budget["execution_panel"]
        for variant in panel["primary_score_variants"]
    }
    if set(actual_keys) != expected_keys:
        raise ValueError("Dev18 primary score manifest가 exact budget key와 다르다")
    if any(
        row.get("status") != "complete"
        or row.get("budget_id") != budget["budget_id"]
        for row in primary_rows
    ):
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
    from src.common.run_registered_model import build_registered_execution_policy

    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "run_snapshot.json"
    path.write_text(json.dumps({
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
    temporary_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_completion_receipt(
    path: Path, *, spec: dict, panel_row: dict, series: int, budget_id: str,
) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = _read_json(path)
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


def _record_seed_state(snapshot_path: Path, result: dict) -> None:
    snapshot = _read_json(snapshot_path)
    snapshot["seed_state"] = result["seed_state"]
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _validate_bound_run_files(
    metadata: dict, *, expected_project_commit: str | None = None,
    compatible_project_commit: str | None = None,
    expected_environment: dict | None = None, expected_spec: dict | None = None,
) -> None:
    references = [metadata.get("run_snapshot")]
    references.extend((metadata.get("training_files") or {}).values())
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
    if expected_project_commit is not None or expected_environment is not None:
        accepted_commits = {expected_project_commit, compatible_project_commit} - {None}
        if accepted_commits and snapshot.get("project_commit") not in accepted_commits:
            raise ValueError("Dev18 재개 snapshot의 project commit이 현재 HEAD와 다르다")
        if (
            expected_environment is not None
            and snapshot.get("environment") != expected_environment
        ):
            raise ValueError("Dev18 재개 snapshot의 실행 환경이 현재 봉인과 다르다")
    from src.common.run_registered_model import build_registered_execution_policy

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


def _save_training_files(output_directory: Path, result: dict) -> tuple[int, dict]:
    training_directory = output_directory / "training"
    training_directory.mkdir(parents=True, exist_ok=True)
    files = {}
    checkpoint = result.get("checkpoint")
    if checkpoint is not None:
        import torch

        checkpoint_path = training_directory / "checkpoint.ckpt"
        torch.save(checkpoint, checkpoint_path)
        files["checkpoint"] = {
            "file": _relative(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        }
    for name, value in (
        ("training_log", result.get("training_log")),
        ("timing", result.get("timing")),
    ):
        if value is None:
            continue
        path = training_directory / f"{name}.json"
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        files[name] = {"file": _relative(path), "sha256": file_sha256(path)}
    return files.get("checkpoint", {}).get("bytes", 0), files


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


def _run_one_spec(
    spec: dict, panel_row: dict, inputs: dict, *, series: int,
    device: str, environment: dict, input_manifest_path, retry_count: int,
    budget_id: str,
) -> list[dict]:
    from src.common.execution_evidence import (
        DEV18_MEASUREMENT_PROTOCOL_ID,
        build_execution_evidence,
    )
    from src.common.execution_identity import EXECUTION_IDENTITY_FIELDS
    from src.common.run_registered_model import execute_registered_model
    from src.common.save_model_artifacts import save_model_score
    from tests.ghl_main.check_registered_outputs import check_registered_output
    from tests.ghl_main.run_registered_models import build_output_directory

    experiment_directory = REPOSITORY_ROOT / "experiments" / "01_ghl_main"
    base_directory = build_output_directory(experiment_directory, spec)
    evidence_directory = _evidence_directory(base_directory, series)
    snapshot_path = _write_run_snapshot(
        evidence_directory, spec, inputs, environment,
    )
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    tracemalloc.start()
    try:
        result = execute_registered_model(
            spec,
            normal_training=inputs.get("normal_training"),
            normal_training_sessions=inputs.get("normal_training_sessions"),
            test_sessions=inputs["test_sessions"], device=device,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    _record_seed_state(snapshot_path, result)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 2
        if device.startswith("cuda") and torch.cuda.is_available()
        else peak_bytes / 1024 ** 2
    )
    artifact_bytes, training_files = _save_training_files(
        evidence_directory, result,
    )
    split = result["split"]
    split_count = 0 if split is None else 1 if isinstance(split, dict) else len(split)
    unavailable_duration = {
        "observed_duration_seconds": None, "duration_basis": "unavailable",
    }
    evidence = build_execution_evidence(
        split, result["timing"], spec=spec,
        measurement_protocol_id=DEV18_MEASUREMENT_PROTOCOL_ID, retry_count=retry_count,
        training_session_durations=[dict(unavailable_duration) for _ in range(split_count)],
        test_input_sessions=inputs["test_sessions"],
        test_session_durations=[dict(unavailable_duration) for _ in inputs["test_sessions"]],
        peak_memory_mb=peak_memory_mb, model_artifact_bytes=artifact_bytes,
    )
    validation_outputs = tuple(result.get("validation_outputs") or ())
    for output in validation_outputs:
        required_alignment = {
            "scores", "source_start", "source_end_exclusive", "alignment",
        }
        if not required_alignment <= output.keys():
            raise ValueError("validation output에 시점 정렬 필드가 없다")
        if (
            int(output["source_end_exclusive"]) - int(output["source_start"])
            != len(output["scores"])
        ):
            raise ValueError("validation output의 source 범위와 score 길이가 다르다")
    validation_scores = tuple(output["scores"] for output in validation_outputs)
    validation_source_starts = tuple(
        int(output.get("source_start", 0)) for output in validation_outputs
    )
    test_output = result["test_outputs"][0]
    variants = _expected_variants(panel_row)
    saved_rows = []
    for variant in variants:
        output = test_output[variant] if spec["model"] == "TSPulse" else test_output
        output_directory = build_output_directory(
            experiment_directory, spec,
            score_variant=variant if spec["model"] == "TSPulse" else None,
        )
        saved = save_model_score(
            output, output_directory, dataset="DEV18", series=series,
            model=spec["model"], target_use=spec["target_use"], tier=spec["tier"],
            ratio=spec["ratio"], seed=spec["seed"], config_id=spec["config_id"],
            common_recipe=spec["common_recipe"],
            common_recipe_id=spec["common_recipe_id"],
            normalization_scope=output["normalization_scope"],
            validation_scores=validation_scores,
            validation_source_starts=validation_source_starts,
            score_variant=variant or None,
            config_registry_sha256=spec["config_registry_sha256"],
            execution_identity={field: spec[field] for field in EXECUTION_IDENTITY_FIELDS},
            execution_evidence=evidence,
        )
        metadata_path = Path(saved["metadata_path"])
        metadata = _read_json(metadata_path)
        metadata["run_snapshot"] = {
            "file": _relative(snapshot_path), "sha256": file_sha256(snapshot_path),
        }
        metadata["training_files"] = training_files
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
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
) -> list[dict]:
    """18개 series에 exact 65-run panel을 재개형으로 실행한다."""
    _require_cuda_or_remote(device=device, remote_execution=remote_execution)
    if device.startswith("cuda"):
        from src.common.set_reproducible_seed import set_reproducible_seed

        set_reproducible_seed(0)
    readiness = prepare_tuning(data_root=data_root, require_clean=True)
    budget = _read_json(DEFAULT_BUDGET_PATH)
    specs, panel_by_key = _specs_for_budget(budget)
    specs.sort(key=_execution_priority)
    manifest_rows = _load_score_manifest()
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
    compatible_project_commit = _compatible_resume_source()
    from tests.ghl_main.run_registered_models import load_registered_inputs

    for entry in series_entries:
        series = int(entry["series"])
        inputs = load_registered_inputs(
            spec=specs[0], input_manifest_path=input_manifest_path,
            series=f"{series:02d}", data_root=data_root,
        )
        for spec in specs:
            key = (spec["model"], spec["config_id"], spec["ratio"], spec["seed"])
            panel_row = panel_by_key[key]
            if _completed_run(
                manifest_rows, spec=spec, panel_row=panel_row, series=series,
                input_manifest_path=input_manifest_path,
                expected_project_commit=project_commit,
                compatible_project_commit=compatible_project_commit,
                expected_environment=environment,
            ):
                continue
            recovered = _load_completion_receipt(
                _completion_receipt_path(spec, series),
                spec=spec, panel_row=panel_row, series=series,
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
                ):
                    raise ValueError("Dev18 완료 영수증 산출물을 복구하지 못했다")
                manifest_rows = _replace_manifest_rows(
                    manifest_rows, recovered,
                )
                continue
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
            if attempts_used >= maximum_attempts:
                raise RuntimeError("Dev18 trial이 봉인된 최대 시도 횟수에 도달했다")
            for attempt in range(attempts_used, maximum_attempts):
                try:
                    completed = _run_one_spec(
                        spec, panel_row, inputs, series=series, device=device,
                        environment=environment, input_manifest_path=input_manifest_path,
                        retry_count=attempt, budget_id=budget["budget_id"],
                    )
                except Exception as error:
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
                    manifest_rows = _replace_manifest_rows(manifest_rows, failed)
                    if attempt + 1 == maximum_attempts:
                        raise
                    continue
                manifest_rows = _replace_manifest_rows(manifest_rows, completed)
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


def _resolve_score_workers(
    workers, *, pending_count, cpu_count=None, available_memory_bytes=None,
) -> int:
    if type(workers) is not int or workers < 0:
        raise ValueError("workers는 0 이상의 정수여야 한다")
    if type(pending_count) is not int or pending_count < 1:
        raise ValueError("pending_count는 1 이상의 정수여야 한다")
    if cpu_count is None:
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count() or 1
    cpu_count = max(1, int(cpu_count))
    if available_memory_bytes is None:
        available_memory_bytes = _detect_available_memory_bytes()
    if available_memory_bytes is None:
        memory_limit = 4
    else:
        gibibyte = 1024 ** 3
        memory_limit = max(1, (int(available_memory_bytes) - 2 * gibibyte) // gibibyte)
    safe_limit = min(16, cpu_count, memory_limit, pending_count)
    return safe_limit if workers == 0 else min(workers, safe_limit)


def _score_checkpoint_path(directory, identity: dict) -> Path:
    digest = hashlib.sha256(_json(identity).encode("utf-8")).hexdigest()
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
        return None
    try:
        payload = _read_json(path)
        vus_pr_value = payload["vus_pr"]
        expected = _checkpoint_payload(identity, vus_pr_value)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"VUS-PR checkpoint를 읽을 수 없다: {path.name}") from error
    if (
        payload != expected
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


def _score_primary_rows(tasks, labels_by_series, workers) -> dict:
    tasks = sorted(
        tasks,
        key=lambda task: (-task["estimated_cost"], _manifest_key(task["manifest_row"])),
    )
    worker_count = _resolve_score_workers(workers, pending_count=len(tasks))
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
            executor.shutdown(wait=False, cancel_futures=True)
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
) -> list[dict]:
    """완료된 primary score만 라벨에 연결해 18×89 채점 원표를 만든다."""
    budget = _read_json(DEFAULT_BUDGET_PATH)
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
    l_max_by_series = {row["series"]: row["l_max_samples"] for row in ell_max["series"]}
    panel_by_key = {
        (row["model"], row["config_id"], str(row["physical_ratio"]), str(row["seed"])): row
        for row in budget["execution_panel"]
    }
    primary_rows = _validate_primary_manifest_rows(
        manifest_rows, budget, tuple(sorted(entries)),
    )
    from tests.ghl_main.run_registered_models import _verify_manifest_file

    for entry in entries.values():
        _verify_manifest_file(
            Path(data_root) / entry["source_directory"] / entry["name"], entry,
        )
    labels_by_series = {
        series: _load_dev18_labels(entry, data_root) for series, entry in entries.items()
    }
    tasks = []
    for manifest_row in primary_rows:
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
    scored = _score_primary_rows(tasks, labels_by_series, workers)
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
    expected_rows = 18 * budget["primary_logical_score_row_count"]
    if len(ledger) != expected_rows:
        raise ValueError(f"Dev18 tuning ledger 행 수가 다르다: {len(ledger)} != {expected_rows}")
    if output_path is not None:
        output_path = Path(output_path)
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        _write_csv(temporary_path, ledger)
        temporary_path.replace(output_path)
    return ledger


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
    ledger = load_trial_score_ledger(ledger_path, budget)
    selection = select_tuning_policies(
        ledger, registry, budget, evaluator_sha256=ledger[0]["evaluator_sha256"],
    )
    membership = build_final_membership_rows(selection, registry)
    if len(membership) != 294:
        raise ValueError(f"final membership 행 수가 다르다: {len(membership)} != 294")

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
