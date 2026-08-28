"""Dev18 exact panel을 실행하고 family-LOFO 선택표와 그림을 만든다."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
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

KOREAN_FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
if KOREAN_FONT_PATH.is_file():
    font_manager.fontManager.addfont(KOREAN_FONT_PATH)
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

from src.common.equal_trial_budget import build_equal_trial_budget, registry_space_sha256
from src.common.execution_identity import file_sha256, load_input_manifest_role
from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from src.common.model_registry import (
    load_model_registry_with_sha,
    validate_primary_hpo_seal,
)
from src.common.verify_run_context import verify_runtime_versions
from src.채점기.vus_pr import vus_pr
from tests.checks.seal_runtime_environment import (
    collect_runtime_environment_identity,
    validate_runtime_snapshot,
)
from tests.ghl_main.build_dev18_budget import _load_current_feasibility


DEFAULT_DATA_ROOT = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project"
SNAPSHOT_DIRECTORY = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "snapshots"
    / "dev18_selection"
)
DEFAULT_BUDGET_PATH = SNAPSHOT_DIRECTORY / "dev18_budget_manifest.json"
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
DEFAULT_REAL_GATE_REPORT_PATH = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
    / "active_models" / "dev18_real_gate_smoke.json"
)
OFFICIAL_TSB_AD_COMMIT = "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48"
FINAL_SPLIT_ROLES = (
    "ghl25_final", "train1_to_test1", "train1_train2_to_test2",
)
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
        REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
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


def prepare_tuning(*, data_root=DEFAULT_DATA_ROOT, require_clean=True) -> dict:
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
    checkpoints = {
        model: _validate_checkpoint_report(model, registry, budget)
        for model in ("TimeRCD", "TSPulse")
    }
    environment = collect_runtime_environment_identity()
    verify_runtime_versions(
        __import__("yaml").safe_load((
            REPOSITORY_ROOT / "configs" / "environment.yaml"
        ).read_text(encoding="utf-8"))
    )
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


def select_tuning_policies(rows, registry: dict, budget: dict, *, evaluator_sha256: str) -> dict:
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
    return {"model_fixed": model_fixed, "tier_fixed": tier_fixed, "lofo": lofo}


def build_final_membership_rows(
    selection: dict, registry: dict, *, ratios=SUPPORTED_RATIO_PERCENTS,
    split_roles=FINAL_SPLIT_ROLES,
) -> list[dict]:
    """두 고정 정책을 final runner가 소비하는 한 CSV 행으로 펼친다."""
    rows = []
    target_free = {"training_free", "strict_zero_shot"}

    def append(kind, split_role, policy, model_name, evaluation_ratio):
        model = registry["models"][model_name]
        supported = evaluation_ratio in policy.get("q_support", policy.get("selection_q_common", []))
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
            "status_reason": "" if selected else "Dev18 고정 정책에서 지원하지 않는 비율",
        })

    for split_role in split_roles:
        for policy in selection["model_fixed"]:
            for ratio in ratios:
                append("model_fixed", split_role, policy, policy["model"], ratio)
        for policy in selection["tier_fixed"]:
            for ratio in ratios:
                append("tier_fixed", split_role, policy, policy["selected_model"], ratio)
    return rows


def _serializable_rows(rows):
    return [
        {
            key: _json(value) if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        }
        for row in rows
    ]


def write_selection_reports(rows, selection: dict, output_directory) -> None:
    """모델별·Tier별 CSV와 PNG를 같은 폴더에 단순한 이름으로 저장한다."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
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
    figure, (axis, table_axis) = pyplot.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 2]},
    )
    labels = [f"{row['tier']} · {row['selected_model']}" for row in selection["tier_fixed"]]
    values = [row["selection_score"] for row in selection["tier_fixed"]]
    axis.bar(labels, values)
    axis.set(title="Selected tier representatives", ylabel="Family-LOFO VUS-PR")
    axis.tick_params(axis="x", rotation=20)
    table_axis.axis("off")
    selection_table = table_axis.table(
        cellText=[[
            row["selected_model"], _json(row["hyperparameters"]), row["selection_reason"],
        ] for row in selection["tier_fixed"]],
        colLabels=["Model", "Parameters", "Reason"],
        loc="center", cellLoc="left",
    )
    selection_table.auto_set_font_size(False)
    selection_table.set_fontsize(7)
    selection_table.scale(1, 1.6)
    figure.tight_layout()
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
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "run_snapshot.json"
    path.write_text(json.dumps({
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_commit": _git_head(),
        "spec": spec,
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
    expected_environment: dict | None = None,
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
    if expected_project_commit is not None or expected_environment is not None:
        snapshot_path = (
            REPOSITORY_ROOT / metadata["run_snapshot"]["file"]
        ).resolve()
        snapshot = _read_json(snapshot_path)
        if (
            expected_project_commit is not None
            and snapshot.get("project_commit") != expected_project_commit
        ):
            raise ValueError("Dev18 재개 snapshot의 project commit이 현재 HEAD와 다르다")
        if (
            expected_environment is not None
            and snapshot.get("environment") != expected_environment
        ):
            raise ValueError("Dev18 재개 snapshot의 실행 환경이 현재 봉인과 다르다")


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
            expected_environment=expected_environment,
        )
    return True


def _run_one_spec(
    spec: dict, panel_row: dict, inputs: dict, *, series: int,
    device: str, environment: dict, input_manifest_path, retry_count: int,
    budget_id: str,
) -> list[dict]:
    from src.common.execution_evidence import build_execution_evidence
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
        measurement_protocol_id="dev18_registered_runner.v1", retry_count=retry_count,
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
        _validate_bound_run_files(metadata)
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


def build_trial_score_ledger(
    manifest_rows, *, data_root=DEFAULT_DATA_ROOT, output_path=None,
) -> list[dict]:
    """완료된 primary score만 라벨에 연결해 18×89 채점 원표를 만든다."""
    from src.채점기.parser import load_and_validate_score

    budget = _read_json(DEFAULT_BUDGET_PATH)
    input_manifest_path = REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
    ell_max = _validate_ell_max(
        DEFAULT_ELL_MAX_PATH, file_sha256(input_manifest_path),
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
    labels_by_series = {
        series: _load_dev18_labels(entry, data_root) for series, entry in entries.items()
    }
    ledger = []
    for manifest_row in sorted(primary_rows, key=_manifest_key):
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
        expected_metadata_variant = manifest_row["score_variant"] or None
        metadata_fields = {
            "dataset": "DEV18", "series": int(series),
            "model": manifest_row["model"], "tier": manifest_row["tier"],
            "ratio": int(manifest_row["physical_ratio"]),
            "seed": int(manifest_row["seed"]),
            "config_id": manifest_row["config_id"],
            "score_variant": expected_metadata_variant,
        }
        if any(metadata.get(field) != value for field, value in metadata_fields.items()):
            raise ValueError("Dev18 primary metadata 신원이 manifest와 다르다")
        if manifest_row["family"] != entries[series]["family"]:
            raise ValueError("Dev18 family가 봉인 manifest와 다르다")
        start, end = metadata["label_slice"]
        labels = labels_by_series[series][slice(start, end)]
        if len(labels) != len(scores):
            raise ValueError("Dev18 score-label 정렬 길이가 다르다")
        panel = panel_by_key[(
            manifest_row["model"], manifest_row["config_id"],
            manifest_row["physical_ratio"], manifest_row["seed"],
        )]
        for ratio in panel["logical_ratios"]:
            ledger.append({
                "series": series, "family": manifest_row["family"],
                "tier": manifest_row["tier"], "model": manifest_row["model"],
                "config_id": manifest_row["config_id"], "ratio": ratio,
                "seed": int(manifest_row["seed"]),
                "score_variant": manifest_row["score_variant"],
                "normalization": info["norm_kind"],
                "vus_pr": vus_pr(
                    scores, labels, l_max_by_series[series],
                    n_thresholds=vus_report["n_thresholds"],
                ),
                "score_file": manifest_row["score_file"],
                "score_sha256": manifest_row["score_sha256"],
                "evaluator_sha256": vus_report["evaluator_sha256"],
                "ell_max_id": ell_max["ell_max_id"],
                "status": "complete", "status_reason": "",
            })
    expected_rows = 18 * budget["primary_logical_score_row_count"]
    if len(ledger) != expected_rows:
        raise ValueError(f"Dev18 tuning ledger 행 수가 다르다: {len(ledger)} != {expected_rows}")
    if output_path is not None:
        _write_csv(output_path, ledger)
    return ledger


def _policy_csv_rows(rows):
    return _serializable_rows(rows)


def finish_tuning(manifest_rows, *, data_root=DEFAULT_DATA_ROOT) -> dict:
    """완료 score를 채점하고 정책표·membership·검토용 CSV/PNG를 한 폴더에 쓴다."""
    prepare_tuning(data_root=data_root, require_clean=True)
    registry, _ = load_model_registry_with_sha()
    budget = _read_json(DEFAULT_BUDGET_PATH)
    DEFAULT_RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    ledger_path = DEFAULT_RESULT_DIRECTORY / "dev18_trial_score_ledger.csv"
    ledger = build_trial_score_ledger(
        manifest_rows, data_root=data_root, output_path=ledger_path,
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
    return {
        "status": "complete", "budget_id": budget["budget_id"],
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
