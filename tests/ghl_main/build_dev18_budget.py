"""Dev18 feasibility support를 equal-trial budget manifest로 봉인한다."""

import argparse
import json
from pathlib import Path

from src.common.equal_trial_budget import build_equal_trial_budget
from src.common.execution_identity import file_sha256, sealed_crlf_text_sha256
from src.common.model_registry import load_model_registry_with_sha
from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEASIBILITY_DIRECTORY = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "snapshots"
    / "dev18_selection"
)
DEFAULT_OUTPUT_PATH = DEFAULT_FEASIBILITY_DIRECTORY / "dev18_budget_manifest.json"


def _load_current_feasibility(repository_root: Path, directory: Path, registry_sha: str, *, config_count: int):
    ledger_path = directory / "dev18_feasibility_ledger.csv"
    summary_path = directory / "dev18_feasibility_summary.json"
    audit_root = (
        repository_root / "experiments" / "checks" / "datasets" / "dev18"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        "config_registry_sha256": registry_sha,
        "input_manifest_sha256": file_sha256(
            repository_root / "configs" / "input_manifest.yaml"
        ),
        "inventory_sha256": sealed_crlf_text_sha256(
            audit_root / "logs" / "inventory.csv"
        ),
        "audit_snapshot_sha256": sealed_crlf_text_sha256(
            audit_root / "snapshots" / "audit.json"
        ),
        "ledger_sha256": file_sha256(ledger_path),
        "builder_sha256": file_sha256(
            repository_root / "tests" / "ghl_main" / "build_dev18_feasibility.py"
        ),
        "feasibility_code_sha256": file_sha256(
            repository_root / "src" / "common" / "model_feasibility.py"
        ),
        "split_code_sha256": file_sha256(
            repository_root / "src" / "data_split" / "split_ratio_prefix.py"
        ),
        "data_preprocessing_sha256": file_sha256(
            repository_root / "configs" / "data_preprocessing.yaml"
        ),
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"Dev18 feasibility {field}가 현재 파일과 다르다")
    if summary.get("status") != "complete_with_declared_static_unavailability":
        raise ValueError("Dev18 feasibility summary가 완료 상태가 아니다")
    if (summary.get("config_count") != config_count
            or summary.get("row_count") != config_count * 18 * len(SUPPORTED_RATIO_PERCENTS)
            or summary.get("labels_or_scores_read") is not False):
        raise ValueError("Dev18 feasibility summary 핵심 계약이 잘못됐다")
    return ledger_path, summary_path, summary


def build_dev18_budget_artifact(
    *, repository_root=REPOSITORY_ROOT,
    feasibility_directory=DEFAULT_FEASIBILITY_DIRECTORY,
    output_path=DEFAULT_OUTPUT_PATH,
) -> dict:
    """stable budget ID와 현재 파일 attestation을 한 JSON에 쓴다."""
    repository_root = Path(repository_root)
    feasibility_directory = Path(feasibility_directory)
    output_path = Path(output_path)
    registry, registry_sha = load_model_registry_with_sha(repository_root)
    ledger_path, summary_path, summary = _load_current_feasibility(
        repository_root, feasibility_directory, registry_sha,
        config_count=sum(len(model["candidates"]) for model in registry["models"].values()),
    )
    budget = build_equal_trial_budget(registry, summary)
    selection = registry["selection"]
    seal_matches = (
        selection.get("primary_hpo_regime") == "equal_trial"
        and selection.get("budget_id") == budget["budget_id"]
        and selection.get("selection_status") == "ready"
    )
    selected_models = {
        row["model"] for row in budget["execution_panel"]
    }
    pending_models = [
        model_name for model_name, model in registry["models"].items()
        if model_name in selected_models and model["execution_status"] != "ready"
    ]
    manifest = {
        **budget,
        "seal_status": "sealed" if seal_matches else "pending_registry_seal",
        "execution_readiness_status": "blocked" if pending_models else "ready",
        "pending_execution_models": pending_models,
        "attestation": {
            "config_registry_sha256": registry_sha,
            "feasibility_ledger_sha256": file_sha256(ledger_path),
            "feasibility_summary_sha256": file_sha256(summary_path),
            "equal_trial_budget_code_sha256": file_sha256(
                repository_root / "src" / "common" / "equal_trial_budget.py"
            ),
            "budget_builder_sha256": file_sha256(Path(__file__)),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "budget_path": str(output_path),
        "budget_id": budget["budget_id"],
        "seal_status": manifest["seal_status"],
        "execution_readiness_status": manifest["execution_readiness_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feasibility-directory", type=Path,
                        default=DEFAULT_FEASIBILITY_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    print(json.dumps(build_dev18_budget_artifact(
        feasibility_directory=arguments.feasibility_directory,
        output_path=arguments.output,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
