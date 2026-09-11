"""Dev18 후보 봉인부터 q별 튜닝·채점·선택까지 한 명령으로 재개한다."""

import argparse
import csv
import io
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

# OpenBLAS를 1스레드로 초기화한 뒤 늘리면 일부 빌드의 SVD가 충돌한다.
try:
    blas_capacity = len(os.sched_getaffinity(0))
except (AttributeError, OSError):
    blas_capacity = os.cpu_count() or 1
os.environ["OPENBLAS_NUM_THREADS"] = str(blas_capacity)
import numpy
import scipy.linalg
from threadpoolctl import threadpool_limits

_BLAS_SERIAL_LIMIT = threadpool_limits(limits=1, user_api="blas")
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.common.execution_identity import file_sha256
from src.common.execution_evidence import FULL_PREFIX_STORAGE_SCHEMA_VERSION
from src.common.model_registry import load_model_registry_with_sha
from tests.ghl_main import run_dev18_tuning as tuning
from tests.ghl_main.build_dev18_budget import _load_current_feasibility
from tests.ghl_main.build_ratio_tuning_budget import (
    build_full_prefix_budget,
    build_ratio_tuning_budget,
)
from tests.ghl_main.select_ratio_tuning import select_ratio_tuning_policies
from tests.ghl_main.store_recommendation_evidence import (
    DEFAULT_RECOMMENDATION_DIRECTORY, open_recommendation_evidence,
)
from tests.ghl_main.record_run_history import (
    preserve_run_receipt,
    record_run_history,
    record_run_stage,
    save_run_history,
    summarize_run_history,
    summarize_pca_compute,
    hold_tuning_lock,
)
from tests.ghl_main.compare_execution_runtimes import load_runtime_references, summarize_runtime_comparisons


def _paths(*, full_prefix=False):
    if full_prefix:
        return {
            "budget": tuning.SNAPSHOT_DIRECTORY / "full_prefix_v2" / "budget.json",
            "result": tuning.DEFAULT_RESULT_DIRECTORY / "full_prefix_v2",
            "manifest": tuning.DEFAULT_SCORE_MANIFEST_PATH.with_name("dev18_full_prefix_v2_manifest.csv"),
            "checkpoint": tuning.DEFAULT_RESULT_DIRECTORY / "full_prefix_v2" / "vus_checkpoints",
            "recommendation": DEFAULT_RECOMMENDATION_DIRECTORY,
        }
    name = "main"
    return {
        "budget": tuning.SNAPSHOT_DIRECTORY / "ratio_tuned_v1" / name / "budget.json",
        "result": tuning.DEFAULT_RESULT_DIRECTORY / "ratio_tuned_v1" / name,
        "manifest": tuning.DEFAULT_SCORE_MANIFEST_PATH.with_name(f"dev18_ratio_{name}_manifest.csv"),
        "checkpoint": tuning.DEFAULT_VUS_CHECKPOINT_DIRECTORY / "ratio_tuned_v1" / name,
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _seal_json(path, payload):
    if path.exists():
        if tuning._read_json(path) != payload:
            raise ValueError(f"이미 봉인한 실행 계획 또는 코드 신원이 바뀌었다: {path}")
    else:
        _write_json(path, payload)


def _pending_check(path, history):
    if not Path(path).exists():
        return True
    if tuning._read_json(path).get("status") == "failed":
        archived = preserve_run_receipt(path, Path(history["history_file"]).parent / "receipts")
        history.setdefault("prior_failed_checks", []).append(str(archived))
        save_run_history(history)
        return True
    return False


def prepare_tuning_environment(registry, budget, paths, *, data_root, history):
    """완료한 사전 검사를 대조하고 누락된 검사만 실행한 뒤 환경을 봉인한다."""
    from tests.checks.run_lightning_dev18 import configure_cuda_environment, seal_lightning_runtime
    from tests.checks.run_model_smoke import (
        FULL_PREFIX_OUTPUT_PATH, run_full_prefix_smoke, validate_full_prefix_smoke,
    )
    from tests.checks.run_checkpoint_smoke import (
        FULL_PREFIX_OUTPUT_ROOT, MODEL_DIRECTORIES, run_dev18_checkpoint_smoke,
        require_checkpoint_smoke_success,
    )
    from tests.checks.check_dev18_resources import (
        FULL_PREFIX_REPORT_PATH, observe_disk_space, run_resource_check, validate_resource_report,
    )
    from tests.checks.seal_runtime_environment import collect_runtime_environment_identity, ensure_runtime_snapshot
    from src.common.set_reproducible_seed import set_reproducible_seed
    from huggingface_hub import hf_hub_download

    configure_cuda_environment()
    tuning._require_cuda_or_remote(device="cuda", remote_execution=False)
    tuning._require_clean_worktree()
    set_reproducible_seed(0)
    if budget.get("experiment_mode") == "full_prefix_v2":
        with record_run_stage(history, "model_smoke"):
            if _pending_check(FULL_PREFIX_OUTPUT_PATH, history):
                _write_json(FULL_PREFIX_OUTPUT_PATH, run_full_prefix_smoke())
            validate_full_prefix_smoke()
        with record_run_stage(history, "checkpoint_smoke"):
            missing = []
            for model in ("TimeRCD", "TSPulse"):
                path = FULL_PREFIX_OUTPUT_ROOT / MODEL_DIRECTORIES[model] / "dev18_checkpoint_smoke.json"
                if _pending_check(path, history):
                    missing.append(model)
                else:
                    tuning._validate_checkpoint_report(model, registry, budget)
            if missing:
                require_checkpoint_smoke_success(run_dev18_checkpoint_smoke(
                    data_root=data_root, budget_path=paths["budget"], models=missing,
                    downloader=hf_hub_download,
                ))
            for model in ("TimeRCD", "TSPulse"):
                tuning._validate_checkpoint_report(model, registry, budget)
        with record_run_stage(history, "resource_gate") as stage:
            stage["disk_observation"] = observe_disk_space(paths["result"])
            save_run_history(history)
            if _pending_check(FULL_PREFIX_REPORT_PATH, history):
                run_resource_check(data_root=Path(data_root))
            validate_resource_report()
    with record_run_stage(history, "runtime_seal"):
        seal_lightning_runtime(
            set_reproducible_seed=set_reproducible_seed,
            collect_environment_identity=collect_runtime_environment_identity,
            ensure_runtime_snapshot=ensure_runtime_snapshot,
        )


def _prepare_full_prefix_tuning(registry, registry_sha):
    import hashlib
    import yaml
    from src.common.model_feasibility import (
        FEASIBILITY_LEDGER_FIELDS, build_dev18_feasibility_rows, summarize_dev18_feasibility,
    )
    from tests.ghl_main.build_dev18_feasibility import _load_approved_inventory

    manifest_path = tuning.REPOSITORY_ROOT / "configs/input_manifest.yaml"
    manifest_sha = file_sha256(manifest_path)
    entries = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["datasets"]["DEV18"]["files"]
    _, _, inventory_sha, _ = _load_approved_inventory(tuning.REPOSITORY_ROOT, entries, manifest_sha)
    feasibility_rows = build_dev18_feasibility_rows(
        registry, entries, config_registry_sha256=registry_sha,
        input_manifest_sha256=manifest_sha, inventory_sha256=inventory_sha,
    )
    feasibility_text = io.StringIO(newline="")
    fields = tuple("observed_row" if field == "available_count" else field
                   for field in FEASIBILITY_LEDGER_FIELDS
                   if field not in {"fit_count", "validation_count"})
    writer = csv.DictWriter(feasibility_text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({**{field: row.get(field, "") for field in fields if field != "observed_row"},
                     "observed_row": row["available_count"]} for row in feasibility_rows)
    feasibility_bytes = feasibility_text.getvalue().encode("utf-8")
    summary = {
        **summarize_dev18_feasibility(feasibility_rows, registry),
        "input_manifest_sha256": manifest_sha, "inventory_sha256": inventory_sha,
        "data_preprocessing_sha256": file_sha256(tuning.REPOSITORY_ROOT / "configs/data_preprocessing.yaml"),
        "ledger_sha256": hashlib.sha256(feasibility_bytes).hexdigest(),
        "config_registry_sha256": registry_sha,
        "storage_schema_version": FULL_PREFIX_STORAGE_SCHEMA_VERSION,
        "labels_or_scores_read": False,
    }
    budget = build_full_prefix_budget(registry, feasibility_rows, entries, summary)
    tuning.validate_vus_evidence(
        tuning.DEFAULT_VUS_REPORT_PATH, tuning.REPOSITORY_ROOT / "src/채점기/vus_pr.py",
    )
    tuning._validate_ell_max(tuning.DEFAULT_ELL_MAX_PATH, manifest_sha)
    paths = _paths(full_prefix=True)
    attestation = {"config_registry_sha256": registry_sha}
    for field, relative in (
        ("builder_sha256", "tests/ghl_main/build_ratio_tuning_budget.py"),
        ("feasibility_code_sha256", "src/common/model_feasibility.py"),
        ("split_code_sha256", "src/data_split/split_ratio_prefix.py"),
        ("registry_code_sha256", "src/common/model_registry.py"),
    ):
        attestation[field] = file_sha256(tuning.REPOSITORY_ROOT / relative)
    _seal_json(paths["budget"], {"budget": budget, "attestation": attestation})
    _seal_json(paths["budget"].parent / "dev18_feasibility_summary.json", summary)
    feasibility_path = paths["budget"].parent / "dev18_feasibility_ledger.csv"
    if feasibility_path.exists():
        if file_sha256(feasibility_path) != summary["ledger_sha256"]:
            raise ValueError("봉인한 full-prefix feasibility 원표가 바뀌었다")
    else:
        temporary = feasibility_path.with_name(f".{feasibility_path.name}.tmp")
        temporary.write_bytes(feasibility_bytes)
        temporary.replace(feasibility_path)
    with open_recommendation_evidence(registry, budget, directory=paths["recommendation"]) as evidence:
        _seal_json(paths["budget"].parent / "recommendation_contract.json", evidence.contract)
    return registry, budget, [], [], paths, {}


def _validate_preserved_manifest(rows, budget, series_ids):
    expected = {
        tuple(map(str, (series, panel["model"], panel["config_id"], panel["physical_ratio"],
                       panel["seed"], variant)))
        for series in series_ids for panel in budget["execution_panel"]
        for variant in tuning._expected_variants(panel)
    }
    keys = [tuning._manifest_key(row) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected or any(
        row["status"] != "complete" or row["budget_id"] != budget["budget_id"] for row in rows
    ):
        raise ValueError("보존 manifest의 primary·diagnostic 완료 기록이 불완전하다")


def _validate_partial_manifest(rows, budget):
    expected = {
        tuple(map(str, (series, panel["model"], panel["config_id"], panel["physical_ratio"], panel["seed"], variant)))
        for panel in budget["execution_panel"] for series in panel["series_ids"]
        for variant in tuning._expected_variants(panel)
    }
    seen = set()
    for row in rows:
        key = tuning._manifest_key(row)
        if (key in seen or key not in expected or row["budget_id"] != budget["budget_id"]
                or row["status"] not in {"complete", "failed"}):
            raise ValueError("재개 manifest에 중복·다른 실행 신원·잘못된 상태가 있다")
        seen.add(key)


def prepare_ratio_tuning(*,
                         baseline_manifest=tuning.DEFAULT_SCORE_MANIFEST_PATH,
                         baseline_ledger=tuning.DEFAULT_RESULT_DIRECTORY / "dev18_trial_score_ledger.csv"):
    """원본 배열·모델을 읽지 않고 새 실행 계획과 저장 계약을 봉인한다."""
    registry, registry_sha = load_model_registry_with_sha()
    if any(model["target_use"] == "fit_full_prefix" for model in registry["models"].values()):
        return _prepare_full_prefix_tuning(registry, registry_sha)
    _, _, feasibility = _load_current_feasibility(
        tuning.REPOSITORY_ROOT, tuning.SNAPSHOT_DIRECTORY, registry_sha,
        config_count=sum(len(model["candidates"]) for model in registry["models"].values()),
    )
    legacy = tuning._read_json(tuning.DEFAULT_BUDGET_PATH)
    expected = tuning.build_equal_trial_budget(registry, feasibility)
    if any(legacy.get(field) != value for field, value in expected.items()):
        raise ValueError("기존 예산이 현재 등록 후보와 다르다")
    budget = build_ratio_tuning_budget(registry, feasibility, legacy)
    evaluator = tuning.validate_vus_evidence(
        tuning.DEFAULT_VUS_REPORT_PATH, tuning.REPOSITORY_ROOT / "src/채점기/vus_pr.py",
    )
    ell_max = tuning._validate_ell_max(tuning.DEFAULT_ELL_MAX_PATH, budget["input_manifest_sha256"])
    baseline_rows = tuning._load_score_manifest(baseline_manifest)
    _validate_preserved_manifest(baseline_rows, legacy, budget["series_ids"])
    tuning._validate_primary_manifest_rows(baseline_rows, legacy, budget["series_ids"])
    reused_ledger = tuning.load_trial_score_ledger(
        baseline_ledger, legacy, expected_evaluator_sha256=evaluator["evaluator_sha256"],
        expected_ell_max_id=ell_max["ell_max_id"],
    )
    sources = {"manifest_sha256": file_sha256(baseline_manifest),
               "ledger_sha256": file_sha256(baseline_ledger)}
    matched = tuning._reuse_trial_scores(
        [row for row in baseline_rows if row["primary_score"] == "true"], reused_ledger,
        evaluator_sha256=evaluator["evaluator_sha256"], ell_max_id=ell_max["ell_max_id"],
    )
    if len(matched) != legacy["physical_execution_count"] * len(budget["series_ids"]):
        raise ValueError("기존 manifest와 ledger의 점수 신원이 다르다")
    paths = _paths()
    artifact = {"budget": budget, "preserved_sources": sources}
    if paths["budget"].exists():
        if tuning._read_json(paths["budget"]) != artifact:
            raise ValueError("이미 봉인한 비율별 예산 또는 원본 산출물이 바뀌었다")
    else:
        _write_json(paths["budget"], artifact)
    return registry, budget, baseline_rows, reused_ledger, paths, sources


def write_full_prefix_reports(selection, budget, result_directory, *, write_plots=True, support_report=None):
    """조건 집단을 유지한 원표와 곡선을 저장한다."""
    result_directory = Path(result_directory)
    if support_report is not None:
        from src.common.tuning_support import summarize_tuning_support

        limits = summarize_tuning_support(support_report["points"])
        _write_json(result_directory / "tuning_support.json", {**support_report, "model_limits": limits})
        _write_json(result_directory / "conditional_selection.json", selection)
        rows = tuning._serializable_rows(limits)
        tuning._write_csv(result_directory / "model_support_limits.csv", rows,
                          tuple(rows[0]) if rows else ("model", "feature_count", "training_boundary"))
    for name, rows, empty_fields in (
        ("model_ratio_policy", selection["model_ratio"], ("group_id", "model", "ratio")),
        ("tier_adaptive", selection["tier_adaptive"], ("group_id", "tier", "ratio")),
        ("ratio_family_lofo", selection["adaptive_lofo"], ("group_id", "model", "ratio", "holdout_family")),
        ("candidate_audit", selection["candidate_audit"], ("group_id", "model", "ratio", "config_id")),
        ("matched_model_comparison", selection["model_comparison"], ("ratio", "left_model", "right_model")),
        ("tspulse_official_heads", selection.get("tspulse_official_heads", []), ("ratio", "family", "config_id", "score_variant")),
        ("structural_exclusions", budget["structural_exclusions"], ("model", "config_id", "ratio", "series")),
    ):
        serialized = tuning._serializable_rows(rows)
        tuning._write_csv(result_directory / f"{name}.csv", serialized,
                          tuple(serialized[0]) if serialized else empty_fields)
    for panel in budget["model_panels"]:
        model_name = panel["model"]
        rows = [row for row in selection["candidate_audit"] if row["model"] == model_name
                and row["analysis_kind"] == "model_ratio"]
        tuning._write_csv(result_directory / f"{model_name}.csv", tuning._serializable_rows(rows),
                          tuple(rows[0]) if rows else ("group_id", "model", "ratio", "config_id"))
        if not write_plots:
            continue
        figure, axis = tuning.pyplot.subplots(figsize=(7, 4))
        curves = {}
        for row in rows:
            key = (tuple(row["series_ids"]), row["config_id"], row["score_variant"])
            curves.setdefault(key, []).append(row)
        for (members, config_id, score_variant), curve in curves.items():
            curve.sort(key=lambda row: row["ratio"])
            label = f"{config_id} / {score_variant}" if score_variant else config_id
            axis.plot([row["ratio"] for row in curve],
                      [row["family_macro_vus_pr"] for row in curve], marker="o",
                      label=f"{label} / {len(members)} files")
            chosen = [row for row in curve if row["selected"]]
            axis.scatter([row["ratio"] for row in chosen],
                         [row["family_macro_vus_pr"] for row in chosen], marker="*", s=100)
        axis.set(title=f"{model_name}: conditional development scores",
                 xlabel="Normal prefix (%)", ylabel="Family mean VUS-PR", ylim=(0, 1))
        if 0 < len(curves) <= 12:
            axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(result_directory / f"{model_name}.png", dpi=150)
        tuning.pyplot.close(figure)
    for tier in sorted({row["tier"] for row in selection["tier_adaptive"]}):
        rows = [row for row in selection["tier_adaptive"] if row["tier"] == tier]
        tuning._write_csv(result_directory / f"Tier{tier[1:]}.csv", tuning._serializable_rows(rows))
        if not write_plots:
            continue
        figure, axis = tuning.pyplot.subplots(figsize=(7, 4))
        curves = {}
        for row in rows:
            if row["family_lofo_vus_pr"] is None:
                continue
            key = (tuple(row["series_ids"]), json.dumps(row["candidate_ids_by_model"], sort_keys=True))
            curves.setdefault(key, []).append(row)
        for (members, _), curve in curves.items():
            curve.sort(key=lambda row: row["ratio"])
            axis.plot([row["ratio"] for row in curve], [row["family_lofo_vus_pr"] for row in curve],
                      marker="o", label=f"{len(members)} matched files")
        axis.set(title=f"Tier {tier[1:]}: conditional family LOFO",
                 xlabel="Normal prefix (%)", ylabel="Held-out family VUS-PR", ylim=(0, 1))
        if curves:
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(result_directory / f"Tier{tier[1:]}.png", dpi=150)
        tuning.pyplot.close(figure)


def finish_ratio_tuning(registry, budget, manifest_rows, reused_ledger, paths, *,
                        data_root, workers=1, selection_only=False, write_plots=True,
                        scoring_history=None):
    tuning._require_clean_worktree()
    project_commit = tuning._git_head()
    with ExitStack() as resources:
        evidence = None
        if budget.get("experiment_mode") == "full_prefix_v2":
            evidence = resources.enter_context(open_recommendation_evidence(
                registry, budget, directory=paths["recommendation"], project_commit=project_commit,
            ))
            evidence.require_features()
        return _finish_ratio_tuning(
            registry, budget, manifest_rows, reused_ledger, paths, data_root=data_root,
            workers=workers, selection_only=selection_only, write_plots=write_plots,
            project_commit=project_commit, recommendation_store=evidence,
            **({"scoring_history": scoring_history} if scoring_history is not None else {}),
        )


def _finish_ratio_tuning(registry, budget, manifest_rows, reused_ledger, paths, *,
                         data_root, workers, selection_only, write_plots,
                         project_commit, recommendation_store, scoring_history=None):
    result_directory = paths["result"]
    ledger_path = result_directory / "dev18_trial_score_ledger.csv"
    full_prefix = budget.get("experiment_mode") == "full_prefix_v2"
    if not selection_only:
        tuning.prepare_tuning(data_root=data_root, require_execution_environment=False,
                              **({"budget": budget} if full_prefix else {}))
    tuning._validate_primary_manifest_rows(manifest_rows, budget, budget["series_ids"])
    if selection_only or ledger_path.exists():
        evaluator = tuning.validate_vus_evidence(
            tuning.DEFAULT_VUS_REPORT_PATH, tuning.REPOSITORY_ROOT / "src/채점기/vus_pr.py",
        )
        ell_max = tuning._validate_ell_max(tuning.DEFAULT_ELL_MAX_PATH, budget["input_manifest_sha256"])
        ledger = tuning.load_trial_score_ledger(
            ledger_path, budget, expected_evaluator_sha256=evaluator["evaluator_sha256"],
            expected_ell_max_id=ell_max["ell_max_id"],
            **({"allow_partial": not selection_only} if full_prefix else {}),
        )
        if full_prefix and not selection_only and len(ledger) < budget["expected_ledger_rows"]:
            ledger = tuning.build_trial_score_ledger(
                manifest_rows, data_root=data_root, output_path=ledger_path,
                workers=workers, checkpoint_directory=paths["checkpoint"],
                project_commit=project_commit, budget=budget, reused_ledger=ledger,
                **({"scoring_history": scoring_history} if scoring_history is not None else {}),
            )
        else:
            tuning._record_scoring_environment(
                scoring_history, workers=workers, worker_count=0, pending_count=0,
            )
    else:
        ledger = tuning.build_trial_score_ledger(
            manifest_rows, data_root=data_root, output_path=ledger_path,
            workers=workers, checkpoint_directory=paths["checkpoint"],
            project_commit=project_commit, budget=budget, reused_ledger=reused_ledger,
            **({"scoring_history": scoring_history} if scoring_history is not None else {}),
        )
    primary_rows = [row for row in manifest_rows if row["primary_score"] == "true"]
    matched = tuning._reuse_trial_scores(
        primary_rows, ledger, evaluator_sha256=ledger[0]["evaluator_sha256"],
        ell_max_id=ledger[0]["ell_max_id"],
    )
    if len(matched) != len(primary_rows):
        raise ValueError("비율별 ledger와 현재 manifest의 점수 신원이 다르다")
    selection = select_ratio_tuning_policies(
        ledger, registry, budget, evaluator_sha256=ledger[0]["evaluator_sha256"],
    )
    if full_prefix:
        from src.common.load_final_membership import CONDITIONAL_FIELDS, load_final_membership
        from tests.ghl_main.build_tuning_support import build_tuning_support

        support_report = build_tuning_support(selection, registry, budget, ledger, primary_rows)
        membership = tuning.build_conditional_membership_rows(selection, registry)
        membership_path = result_directory / "final_policy_membership.csv"
        temporary = membership_path.with_name(".final_policy_membership.csv.tmp")
        tuning._write_csv(temporary, membership, CONDITIONAL_FIELDS)
        load_final_membership(temporary, registry)
        temporary.replace(membership_path)
        write_full_prefix_reports(selection, budget, result_directory, write_plots=write_plots,
                                  support_report=support_report)
    else:
        from src.common.load_final_membership import FIELDS, load_final_membership

        membership = tuning.build_final_membership_rows(selection, registry)
        membership_path = result_directory / "final_policy_membership.csv"
        temporary = membership_path.with_name(".final_policy_membership.csv.tmp")
        tuning._write_csv(temporary, membership, FIELDS)
        load_final_membership(temporary, registry)
        temporary.replace(membership_path)
        tuning._write_csv(result_directory / "model_fixed_policy.csv",
                          tuning._policy_csv_rows(selection["model_fixed"]))
        tuning._write_csv(result_directory / "tier_fixed_policy.csv",
                          tuning._policy_csv_rows(selection["tier_fixed"]))
        tuning.write_selection_reports(ledger, selection, result_directory, write_plots=write_plots)
    tuning._require_same_worktree(project_commit)
    recommendation = None
    if recommendation_store is not None:
        recommendation_store.sync_results(
            manifest_rows, ledger_rows=ledger, manifest_path=paths["manifest"], ledger_path=ledger_path,
        )
        recommendation = recommendation_store.finalize(require_complete=True)
        if recommendation["status"] != "complete":
            raise ValueError("추천 자료 인수를 마치기 전에는 전체 완료로 처리하지 않는다")
    if full_prefix:
        output_names = (
            "dev18_trial_score_ledger.csv", "final_policy_membership.csv", "model_support_limits.csv",
            "model_ratio_policy.csv", "tier_adaptive.csv", "ratio_family_lofo.csv", "candidate_audit.csv",
            "matched_model_comparison.csv", "structural_exclusions.csv", "tspulse_official_heads.csv",
            "tuning_support.json", "conditional_selection.json",
            *(panel["model"] + ".csv" for panel in budget["model_panels"]),
            *(f"Tier{tier[1:]}.csv" for tier in sorted({row["tier"] for row in selection["tier_adaptive"]})),
        )
        output_paths = [result_directory / name for name in output_names]
    else:
        output_paths = list(result_directory.glob("*.csv"))
    return {"status": "complete", "execution_status": "complete",
            "scoring_selection_status": "complete",
            "recommendation_status": "complete" if recommendation is not None else "not_required",
            "recommendation_evidence": recommendation, "budget_id": budget["budget_id"],
            "selection_rule_id": budget["selection_rule_id"], "ledger_rows": len(ledger),
            "model_ratio_rows": len(selection["model_ratio"]),
            "result_directory": str(result_directory), "project_commit": project_commit,
            "outputs": {path.name: file_sha256(path) for path in sorted(output_paths)}}


def _run_ratio_tuning_command(arguments, history):
    started = time.perf_counter()
    history.update(cost_scope="hpo_development", execution_timing={})
    save_run_history(history)
    with record_run_stage(history, "prepare"):
        registry, budget, preserved, ledger, paths, sources = prepare_ratio_tuning(
            baseline_manifest=arguments.baseline_manifest,
            baseline_ledger=arguments.baseline_ledger,
        )
    history["identity"].update(
        budget_id=budget["budget_id"], budget_sha256=budget["budget_sha256"],
        experiment_mode=budget.get("experiment_mode", "ratio_tuned_v1"),
    )
    history["result_directory"] = str(paths["result"])
    save_run_history(history)
    reused_count = len({tuning._manifest_key(row)[:5] for row in preserved if row["status"] == "complete"})
    current_rows = tuning._load_score_manifest(paths["manifest"])
    if budget.get("experiment_mode") == "full_prefix_v2":
        _validate_partial_manifest(current_rows, budget)
    completed_keys = {
        tuning._manifest_key(row) for row in tuning._merge_manifest_rows(
            preserved, current_rows,
        ) if row["status"] == "complete"
    }
    pending_runs = sum(not all(
        tuple(map(str, (series, panel["model"], panel["config_id"], panel["physical_ratio"],
                       panel["seed"], variant))) in completed_keys
        for variant in tuning._expected_variants(panel)
    ) for panel in budget["execution_panel"]
        for series in panel.get("series_ids", budget["series_ids"]))
    physical_runs = budget.get("physical_run_count",
                               len(budget["series_ids"]) * budget["physical_execution_count"])
    report = {"budget_id": budget["budget_id"], "series_count": len(budget["series_ids"]),
              "experiment_mode": budget.get("experiment_mode", "ratio_tuned_v1"),
              "physical_runs": physical_runs,
              "training_runs": budget.get("training_run_count", physical_runs),
              "preserved_runs": reused_count,
              "pending_runs": pending_runs,
              "expected_ledger_rows": budget.get("expected_ledger_rows",
                  len(budget["series_ids"]) * budget["primary_logical_score_row_count"]),
              "budget_path": str(paths["budget"]), "excluded_series": budget.get("excluded_series", [])}
    if budget.get("experiment_mode") == "full_prefix_v2":
        with open_recommendation_evidence(registry, budget, directory=paths["recommendation"]) as evidence:
            feature_status = evidence.status()
        report.update(planned_runs=physical_runs, pending_runs_before=pending_runs,
                      recommendation_evidence=feature_status)
    else:
        report["additional_runs"] = physical_runs - reused_count
    if arguments.prepare:
        history["result"] = report
        return report
    receipt = paths["result"] / ("execution_complete.json" if arguments.execute_only else "selection_complete.json")
    archived = preserve_run_receipt(receipt, Path(history["history_file"]).parent / "receipts")
    history["previous_receipt_file"] = str(archived) if archived else None
    save_run_history(history)
    execution_seconds = 0.0
    if not (arguments.finish_only or arguments.selection_only):
        prepare_tuning_environment(registry, budget, paths, data_root=arguments.data_root, history=history)
        execution_started = time.perf_counter()
        with record_run_stage(history, "execute"):
            manifest_rows = tuning.execute_panel(
                data_root=arguments.data_root, device="cuda", budget=budget,
                manifest_path=paths["manifest"], preserved_rows=preserved,
                execution_timing=history["execution_timing"],
                **({"recommendation_directory": paths["recommendation"]}
                   if budget.get("experiment_mode") == "full_prefix_v2" else {}),
            )
        execution_seconds = time.perf_counter() - execution_started
    else:
        manifest_rows = tuning._merge_manifest_rows(preserved, tuning._load_score_manifest(paths["manifest"]))
    if not arguments.execute_only:
        tuning._require_cuda_or_remote(device="cuda", remote_execution=arguments.remote_cpu)
        scoring_started = time.perf_counter()
        with record_run_stage(history, "score_select_export"):
            report.update(finish_ratio_tuning(
                registry, budget, manifest_rows, ledger, paths, data_root=arguments.data_root,
                workers=arguments.workers, selection_only=arguments.selection_only,
                write_plots=not arguments.no_plots, scoring_history=history,
            ))
        report["scoring_and_selection_seconds"] = time.perf_counter() - scoring_started
    full_prefix = budget.get("experiment_mode") == "full_prefix_v2"
    if full_prefix and arguments.execute_only:
        tuning._validate_primary_manifest_rows(manifest_rows, budget, budget["series_ids"])
        with open_recommendation_evidence(registry, budget, directory=paths["recommendation"]) as evidence:
            evidence.require_features()
            evidence.sync_results(manifest_rows, manifest_path=paths["manifest"])
            report["recommendation_evidence"] = evidence.finalize(require_complete=False)
            if not report["recommendation_evidence"]["execution_complete"]:
                raise ValueError("주·보조 head의 실행 증거가 빠져 실행 완료로 처리하지 않는다")
        report.update(execution_status="complete", scoring_selection_status="pending",
                      recommendation_status="pending")
    if not full_prefix:
        for key, path in (("manifest_sha256", arguments.baseline_manifest),
                          ("ledger_sha256", arguments.baseline_ledger)):
            if key in sources and file_sha256(path) != sources[key]:
                raise RuntimeError("실행 중 기존 manifest 또는 ledger가 바뀌었다")
        report["preserved_sources"] = sources
    report.update(started_at=history["started_at"],
                  status=("execution_complete" if arguments.execute_only else "ready_for_handoff")
                  if full_prefix else "complete",
                  completion_scope="execution" if arguments.execute_only else "selection",
                  execution_seconds=execution_seconds,
                  execution_seconds_scope="panel_wall_including_checks_and_attempts",
                  execution_timing=history["execution_timing"],
                  elapsed_seconds=time.perf_counter() - started,
                  run_id=history["run_id"], history_file=history["history_file"])
    report["artifacts"] = {
        name: {"file": str(paths[name]), "sha256": file_sha256(paths[name])}
        for name in ("budget", "manifest")
    }
    history["result"] = report
    _write_json(receipt, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    for flag in ("prepare", "execute-only", "finish-only", "selection-only"):
        mode.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--data-root", type=Path, default=tuning.DEFAULT_DATA_ROOT)
    parser.add_argument("--baseline-manifest", type=Path, default=tuning.DEFAULT_SCORE_MANIFEST_PATH)
    parser.add_argument("--baseline-ledger", type=Path,
                        default=tuning.DEFAULT_RESULT_DIRECTORY / "dev18_trial_score_ledger.csv")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--remote-cpu", action="store_true",
                        help="명시한 원격 CPU에서 finish-only 또는 selection-only만 실행")
    arguments = parser.parse_args()
    if arguments.remote_cpu and not (arguments.finish_only or arguments.selection_only):
        parser.error("--remote-cpu는 --finish-only 또는 --selection-only와 함께 쓴다")
    command_mode = next((name for name in ("prepare", "execute_only", "finish_only", "selection_only")
                         if getattr(arguments, name)), "run").replace("_", "-")
    identity = {
        "kind": "tuning_command", "mode": command_mode, "budget_id": None,
        "arguments": {name: str(value) if isinstance(value, Path) else value
                      for name, value in vars(arguments).items()},
    }
    directory = tuning.DEFAULT_SCORE_MANIFEST_PATH.parent / "run_history" / "commands"
    history = None
    with hold_tuning_lock(tuning.REPOSITORY_ROOT / ".runtime/ratio_tuning.lock"):
        try:
            with record_run_history(directory, identity=identity) as history:
                report = _run_ratio_tuning_command(arguments, history)
        finally:
            if history and history.get("result_directory"):
                budget_id = history["identity"]["budget_id"]
                runtime_references = load_runtime_references(directory.parent / "model_attempts", budget_id)
                _write_json(Path(history["result_directory"]) / "tuning_cost_history.json", {
                    "budget_id": budget_id, "cost_scope": "hpo_development",
                    "command_wall": summarize_run_history(directory, budget_id),
                    "unassigned_command_wall": summarize_run_history(directory, None),
                    "model_attempt_wall": summarize_run_history(directory.parent / "model_attempts", budget_id),
                    "pca_cpu_accounting": summarize_pca_compute(directory.parent / "model_attempts", budget_id),
                    "runtime_comparisons": summarize_runtime_comparisons(
                        runtime_references, repository_root=tuning.REPOSITORY_ROOT),
                    "accounting_rule": "command and model-attempt clocks overlap; never add them; unassigned commands are not charged to this budget",
                    "excluded_stages": ["handoff_archive_and_hash"],
                    "scope": "recorded command time, not provider billing or deployment cost",
                })
        if report.get("experiment_mode") == "full_prefix_v2" and report.get("status") == "ready_for_handoff":
            from tests.ghl_main.package_recommendation_handoff import package_recommendation_handoff

            report["handoff"] = package_recommendation_handoff(report)
            report.update(status="complete", completion_scope="handoff")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
