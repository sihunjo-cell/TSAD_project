"""GHL·HAI 계층 2 EDA의 단일 실행 진입점."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.checks.tier2_eda.audit_channel_activity import (
    build_scope_summary_rows,
    summarize_channel_activity,
    validate_nested_constant_sets,
)
from tests.checks.tier2_eda.audit_dataset_structure import (
    build_data_integrity_rows,
    inspect_dataset,
    read_training_features,
)
from tests.checks.tier2_eda.audit_label_alignment import build_score_alignment_rows
from tests.checks.tier2_eda.audit_model_feasibility import build_model_feasibility_rows
from tests.checks.tier2_eda.audit_model_forward import run_forward_smoke
from tests.checks.tier2_eda.audit_model_parameters import build_model_parameter_rows
from tests.checks.tier2_eda.audit_training_ratios import (
    MODEL_CANDIDATES,
    SOURCE_COMMITS,
    build_model_window_source_rows,
    build_ratio_rows,
    build_training_workload_summary_rows,
    compute_split_lengths,
)
from tests.checks.tier2_eda.build_tier2_eda_report import build_report
from src.common.experiment_config import load_dataset_ratios
from src.common.verify_run_context import verify_runtime_versions


ARTIFACT_FILES = (
    "csv/constant_scope_summary.csv",
    "csv/channel_scaling_readiness.csv",
    "csv/data_integrity.csv",
    "csv/training_ratio_feasibility.csv",
    "csv/training_workload_summary.csv",
    "csv/model_feasibility.csv",
    "csv/score_alignment.csv",
    "csv/model_window_sources.csv",
    "csv/model_parameter_sources.csv",
    "csv/input_forward_smoke.csv",
    "report/tier2_eda_report.md",
)
MANIFEST_PATH = "json/eda_manifest.json"
CONFIG_FILES = (
    "configs/data_preprocessing.yaml",
    "configs/environment.yaml",
    "configs/gdn_hyperparams.yaml",
)
GENERATION_CODE_FILES = tuple(
    f"tests/checks/tier2_eda/{name}"
    for name in (
        "audit_channel_activity.py",
        "audit_dataset_structure.py",
        "audit_label_alignment.py",
        "audit_model_feasibility.py",
        "audit_model_forward.py",
        "audit_model_parameters.py",
        "audit_training_ratios.py",
        "build_tier2_eda_report.py",
        "run_tier2_eda.py",
    )
)
GENERATION_CODE_FILES += (
    "tests/ghl_main/run_tier2.py",
    "tests/hai_extension/run_gdn.py",
    "src/common/experiment_config.py",
    "src/common/verify_run_context.py",
    "src/data_split/take_training_prefix.py",
    "src/data_split/validation_split.py",
    "src/data_split/load_ghl_series.py",
    "src/data_split/load_hai_sessions.py",
    "src/models/tier2/AE/AE_raw(TSB).py",
    "src/models/tier2/LSTMAD/LSTMAD_raw.py",
    "src/models/tier2/USAD/USAD_raw.py",
    "src/models/tier2/GDN/model.py",
    "src/models/tier2/GDN/modules.py",
    "src/models/tier2/GDN/trainer.py",
    "src/models/tier2/GDN/types.py",
    "src/models/tier2/utils/utility.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ghl", "hai", "both"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "checks" / "datasets",
    )
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    pandas.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_repository_files(paths: tuple[str, ...]) -> dict[str, str]:
    return {path: sha256_path(REPOSITORY_ROOT / path) for path in paths}


def build_manifest_statuses(
    *,
    artifacts_generated: bool,
    row_counts_match: bool,
    checksums_match: bool,
    environment_verified: bool,
    source_choice_count: int,
    ratios_fixed: bool,
    fixed_validation_invariant: bool,
    fit_subsets_nested: bool,
    infeasible_count: int,
    boundary_crossing_count: int,
    alignment_passed: bool,
    parameters_fixed: bool,
    full_imports_passed: bool,
    adapter_count: int,
    reference_forward_passed: bool,
) -> dict[str, bool]:
    return {
        "eda_generation_success": all((
            artifacts_generated, row_counts_match, checksums_match, environment_verified,
        )),
        "parameter_eda_closed": all((
            source_choice_count == 0,
            ratios_fixed,
            fixed_validation_invariant,
            fit_subsets_nested,
            infeasible_count == 0,
            boundary_crossing_count == 0,
            alignment_passed,
            parameters_fixed,
        )),
        "tier2_execution_ready": all((
            full_imports_passed,
            adapter_count == 0,
            reference_forward_passed,
        )),
    }


def read_runtime_environment() -> tuple[str, dict[str, str], str]:
    environment = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "environment.yaml").read_text(encoding="utf-8")
    )
    try:
        versions = verify_runtime_versions(environment)
    except RuntimeError as error:
        return "pending", {}, str(error)
    return "verified", versions, ""


def read_git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True, check=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT,
        capture_output=True, check=True, text=True,
    ).stdout.strip())
    return commit, dirty


def execute(dataset: str, output_dir: Path, command: str, run_id: str) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    environment_gate, dependency_versions, environment_error = read_runtime_environment()
    descriptors, structure_rows, feature_names = inspect_dataset(dataset)
    ratios = load_dataset_ratios(dataset)
    gdn_parameters = yaml.safe_load(
        (REPOSITORY_ROOT / "configs" / "gdn_hyperparams.yaml").read_text(encoding="utf-8")
    )
    for directory in (output_dir / "csv", output_dir / "json", output_dir / "report"):
        directory.mkdir(parents=True, exist_ok=True)

    ratio_rows = []
    activity_rows = []
    for descriptor_index, descriptor in enumerate(descriptors, start=1):
        print(f"[{dataset} {descriptor_index}/{len(descriptors)}] {descriptor['source']}", flush=True)
        values = read_training_features(descriptor, descriptor["feature_names"])
        for ratio in ratios:
            fit_pool_length, fit_subset_length, _ = compute_split_lengths(len(values), ratio)
            ratio_rows.extend(build_ratio_rows(
                dataset,
                descriptor["source"],
                len(values),
                ratio,
                feature_count=len(feature_names),
            ))
            scopes = {
                "fit_pool": values[:fit_pool_length],
                "fit_subset": values[:fit_subset_length],
                "fixed_validation": values[fit_pool_length:],
            }
            for scope, scoped_values in scopes.items():
                activity_rows.extend(summarize_channel_activity(
                    scoped_values,
                    descriptor["feature_names"],
                    ratio,
                    dataset,
                    descriptor["source"],
                    scope,
                ))

    validate_nested_constant_sets(activity_rows)
    scope_rows = build_scope_summary_rows(activity_rows)
    integrity_rows = build_data_integrity_rows(dataset, structure_rows, feature_names)
    workload_rows = build_training_workload_summary_rows(ratio_rows, dataset, len(feature_names))
    model_rows = build_model_feasibility_rows(ratio_rows, len(feature_names))
    test_sources = (
        [
            {
                "source": descriptor["source"],
                "length": descriptor["test_length"],
                "timestamp_start": "",
                "sampling_interval_seconds": None,
            }
            for descriptor in descriptors
        ]
        if dataset == "GHL"
        else [
            {
                "source": row["file_or_session"],
                "length": int(row["row_count"]),
                "timestamp_start": row["timestamp_start"],
                "sampling_interval_seconds": float(row["sampling_interval_seconds"]),
            }
            for row in structure_rows
            if row["split"] == "test"
        ]
    )
    score_rows = [
        row
        for source in test_sources
        for row in build_score_alignment_rows(
            dataset,
            source["source"],
            source["length"],
            MODEL_CANDIDATES[dataset],
            source["timestamp_start"],
            source["sampling_interval_seconds"],
        )
    ]
    source_rows = build_model_window_source_rows(dataset)
    parameter_rows = build_model_parameter_rows(dataset)
    topk = int(gdn_parameters["model_params"]["topk_by_dataset"][dataset])
    forward_rows = run_forward_smoke(dataset, len(feature_names), topk)
    outputs = {
        "constant_scope_summary.csv": scope_rows,
        "channel_scaling_readiness.csv": activity_rows,
        "data_integrity.csv": integrity_rows,
        "training_ratio_feasibility.csv": ratio_rows,
        "training_workload_summary.csv": workload_rows,
        "model_feasibility.csv": model_rows,
        "score_alignment.csv": score_rows,
        "model_window_sources.csv": source_rows,
        "model_parameter_sources.csv": parameter_rows,
        "input_forward_smoke.csv": forward_rows,
    }
    for name, rows in outputs.items():
        write_rows(output_dir / "csv" / name, rows)

    expected_row_counts = {
        "training_ratio_feasibility.csv": len(descriptors) * len(ratios) * len(MODEL_CANDIDATES[dataset]),
        "model_feasibility.csv": len(descriptors) * len(ratios) * len(MODEL_CANDIDATES[dataset]),
        "training_workload_summary.csv": (
            len(descriptors) * len(ratios) * len(MODEL_CANDIDATES[dataset])
            if dataset == "GHL" else len(ratios) * len(MODEL_CANDIDATES[dataset])
        ),
        "channel_scaling_readiness.csv": len(descriptors) * len(ratios) * 3 * len(feature_names),
        "constant_scope_summary.csv": len(ratios) * 3,
        "score_alignment.csv": len(test_sources) * len(MODEL_CANDIDATES[dataset]),
    }
    actual_row_counts = {name: len(rows) for name, rows in outputs.items()}
    mismatches = {
        name: {"expected": expected, "actual": actual_row_counts[name]}
        for name, expected in expected_row_counts.items()
        if actual_row_counts[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"EDA 산출물 행 수가 계산 계약과 다릅니다: {mismatches}")

    summary = build_report(
        dataset, output_dir, len(feature_names), len(structure_rows), environment_gate,
    )
    dataset_dir = descriptors[0]["path"].parent
    all_paths = sorted(dataset_dir / row["file_name"] for row in structure_rows)
    project_commit, worktree_dirty = read_git_state()
    artifact_hashes = {
        path: sha256_path(output_dir / path) for path in ARTIFACT_FILES
    }
    finished_at = datetime.now(timezone.utc).isoformat()
    grouped_ratio_rows = {
        (row["file_or_session"], row["model"]): [
            candidate for candidate in ratio_rows
            if (candidate["file_or_session"], candidate["model"])
            == (row["file_or_session"], row["model"])
        ]
        for row in ratio_rows
    }
    fixed_validation_invariant = all(
        len({row["validation_length"] for row in rows}) == 1
        for rows in grouped_ratio_rows.values()
    )
    fit_subsets_nested = all(
        [row["ratio"] for row in sorted(rows, key=lambda row: row["ratio"])] == list(ratios)
        and [row["fit_subset_length"] for row in sorted(rows, key=lambda row: row["ratio"])]
        == sorted(row["fit_subset_length"] for row in rows)
        for rows in grouped_ratio_rows.values()
    )
    boundary_crossing_count = sum(
        row["fit_validation_boundary_crossing_window_count"]
        + row["file_or_session_boundary_crossing_window_count"]
        for row in ratio_rows
    )
    source_choice_count = sum(
        row["status"] == "source_choice_required" for row in parameter_rows
    )
    adapter_count = sum(
        row["status"] in {"adapter_required", "local_source_mismatch"}
        for row in parameter_rows
    )
    full_imports_passed = all(row["full_module_import_ready"] for row in forward_rows)
    reference_forward_passed = all(row["reference_forward_passed"] for row in forward_rows)
    statuses = build_manifest_statuses(
        artifacts_generated=all((output_dir / path).is_file() for path in ARTIFACT_FILES),
        row_counts_match=not mismatches,
        checksums_match=all(
            sha256_path(output_dir / path) == digest for path, digest in artifact_hashes.items()
        ),
        environment_verified=environment_gate == "verified",
        source_choice_count=source_choice_count,
        ratios_fixed=tuple(ratios) == (5, 10, 20, 40, 60, 80, 100),
        fixed_validation_invariant=fixed_validation_invariant,
        fit_subsets_nested=fit_subsets_nested,
        infeasible_count=sum(not row["execution_feasible"] for row in ratio_rows),
        boundary_crossing_count=boundary_crossing_count,
        alignment_passed=all(row["alignment_valid"] for row in score_rows),
        parameters_fixed=all(row["fixed_across_ratios"] for row in parameter_rows),
        full_imports_passed=full_imports_passed,
        adapter_count=adapter_count,
        reference_forward_passed=reference_forward_passed,
    )
    manifest = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "dataset": dataset,
        "ratios_percent": list(ratios),
        "ratio_unit": "percent_of_fit_pool_after_fixed_validation",
        "models": [candidate["model"] for candidate in MODEL_CANDIDATES[dataset]],
        "input_scope": "학습 통계는 정상 학습 구간만 사용하고 test feature는 무결성·정렬만 검사",
        "inputs": [
            {
                "path": f"{path.parent.name}/{path.name}",
                "size_bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_path(path),
            }
            for path in all_paths
        ],
        "features": {
            "count": len(feature_names),
            "names": feature_names,
            "excluded_metadata": ["timestamp", "Label", "label"],
        },
        "scope_definitions": {
            "fit_pool": "전체 정상 학습 구간에서 고정 validation을 제외한 앞 구간",
            "fit_subset": "fit pool에서 ceil(ratio*pool_length)로 남긴 시간순 누적 구간",
            "fixed_validation": "전체 정상 학습 구간의 마지막 ceil(10%) 구간",
        },
        "model_input_windows": summary["model_windows"],
        "fixed_across_ratios": [
            "model_structure", "input_window", "hidden_or_latent_dimension", "learning_rate",
            "batch_size", "max_epochs_and_early_stopping", "optimizer", "anomaly_score",
            "preprocessing_and_normalization",
            "fixed_validation_rows",
        ],
        "split_rule": "전체 정상 구간의 마지막 10%를 고정 validation으로 먼저 분리하고 나머지 fit pool에 비율 적용",
        "scaler_rule": "현재 비율의 fit subset에만 fit하고 고정 validation과 test에는 transform만 적용",
        "window_rule": "모델별 값을 모든 비율에 고정하고 파일·세션·fit/validation 경계를 넘지 않음",
        "vus_l_max_status": "계산하지 않음; 지우 담당",
        "source_commits": SOURCE_COMMITS,
        "project_commit": project_commit,
        "worktree_dirty": worktree_dirty,
        "dependencies": dependency_versions,
        "runtime_contract_status": environment_gate,
        "runtime_contract_error": environment_error,
        "selected_model_imports_passed": full_imports_passed,
        "config_sha256": hash_repository_files(CONFIG_FILES),
        "generation_code_sha256": hash_repository_files(GENERATION_CODE_FILES),
        "command": command,
        "generated_files": [
            {"path": path, "sha256": artifact_hashes[path]} for path in ARTIFACT_FILES
        ],
        "manifest_path": MANIFEST_PATH,
        "manifest_self_hash": "not_applicable",
        "row_counts": actual_row_counts,
        "expected_row_counts": expected_row_counts,
        "label_values_used": False,
        "model_training_executed": False,
        "data_integrity_passed": all(row["integrity_passed"] for row in integrity_rows),
        "scaler_execution_ready": all(
            row["scaler_execution_ready"] for row in activity_rows
        ),
        "fixed_validation_invariant": fixed_validation_invariant,
        "fit_subsets_time_ordered_and_nested": fit_subsets_nested,
        "boundary_crossing_window_count": boundary_crossing_count,
        "core_forward_passed": all(row["forward_passed"] for row in forward_rows),
        "selected_reference_forward_passed": reference_forward_passed,
        "parameter_decisions_requiring_source_choice": source_choice_count,
        "implementation_adapter_rows": adapter_count,
        "model_applicability": summary["model_applicability"],
        "tier2_execution_blockers": {
            "full_module_import_pending": [
                {"model": row["model"], "error": row["full_module_import_error"]}
                for row in forward_rows if not row["full_module_import_ready"]
            ],
            "implementation_adapter_rows": adapter_count,
            "selected_reference_forward_pending": [
                row["model"] for row in forward_rows if not row["reference_forward_passed"]
            ],
        },
        **statuses,
    }
    (output_dir / "json" / "eda_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"tier2-eda-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    datasets = ("ghl", "hai") if args.dataset == "both" else (args.dataset,)
    for dataset_name in datasets:
        dataset = dataset_name.upper()
        output_dir = args.output_root / dataset_name / "tier2_eda"
        execute(dataset, output_dir, " ".join(sys.argv), run_id)
        print(f"{dataset}: {output_dir}")


if __name__ == "__main__":
    main()
