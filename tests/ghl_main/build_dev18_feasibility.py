"""Role-A snapshot에서 Dev18 label-blind 정적 feasibility를 봉인한다."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import yaml

from src.common.execution_identity import file_sha256
from src.common.model_feasibility import (
    FEASIBILITY_LEDGER_FIELDS,
    build_dev18_feasibility_rows,
    summarize_dev18_feasibility,
)
from src.common.model_registry import load_model_registry_with_sha


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "snapshots"
    / "dev18_selection"
)


def _load_approved_inventory(repository_root: Path, entries: list, manifest_sha: str):
    audit_root = (
        repository_root / "experiments" / "checks" / "datasets" / "dev18"
    )
    snapshot_path = audit_root / "snapshots" / "audit.json"
    inventory_path = audit_root / "logs" / "inventory.csv"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    inventory_sha = file_sha256(inventory_path)
    audit_script = repository_root / "tests" / "checks" / "audit_dev18_inputs.py"
    if not str(snapshot.get("status", "")).startswith("approved"):
        raise ValueError("Dev18 Role-A audit가 승인되지 않았다")
    if snapshot.get("input_manifest_sha256") != manifest_sha:
        raise ValueError("Dev18 audit와 input manifest SHA-256이 다르다")
    if snapshot.get("audit_script_sha256") != file_sha256(audit_script):
        raise ValueError("Dev18 audit script SHA-256이 현재 코드와 다르다")
    if snapshot.get("tables", {}).get("inventory.csv") != inventory_sha:
        raise ValueError("Dev18 inventory SHA-256이 audit snapshot과 다르다")
    expected_tables = {
        "inventory.csv", "feature_schema_and_quality.csv",
        "high_correlation_pairs.csv",
    }
    if set(snapshot.get("tables", {})) != expected_tables:
        raise ValueError("Dev18 audit 원표 목록이 불완전하다")
    for table_name in expected_tables:
        if snapshot["tables"][table_name] != file_sha256(
            audit_root / "logs" / table_name
        ):
            raise ValueError(f"Dev18 {table_name} SHA-256이 audit snapshot과 다르다")
    role_sets = snapshot.get("role_sets", {})
    if (
        snapshot.get("series_count") != 18
        or snapshot.get("family_count") != 10
        or role_sets.get("official20_count") != 20
        or role_sets.get("dev18_count") != 18
        or role_sets.get("ghl25_count") != 25
        or role_sets.get("dev18_ghl25_intersection_count") != 0
        or role_sets.get("official20_ghl25_intersection") != [
            "040_GHL_id_9_Sensor_tr_50000_1st_92001.csv",
            "049_GHL_id_18_Sensor_tr_50000_1st_109001.csv",
        ]
    ):
        raise ValueError("Dev18 audit role set 또는 family 수가 잘못됐다")

    with inventory_path.open(encoding="utf-8-sig", newline="") as file:
        inventory = list(csv.DictReader(file))
    if len(inventory) != len(entries):
        raise ValueError("Dev18 inventory 행 수가 manifest와 다르다")
    for entry, row in zip(entries, inventory):
        expected = {
            "series": entry["series"],
            "order": str(entry["order"]),
            "family": entry["family"],
            "source_directory": entry["source_directory"],
            "name": entry["name"],
            "size_bytes": str(entry["size_bytes"]),
            "sha256": entry["sha256"],
            "row_count": str(entry["row_count"]),
            "feature_count": str(entry["feature_count"]),
            "feature_names_sha256": entry["feature_names_sha256"],
            "training_boundary": str(entry["training_boundary"]),
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError(f"Dev18 inventory와 manifest가 다르다: {entry['series']}")
    return snapshot_path, inventory_path, inventory_sha, snapshot


def _csv_bytes(rows: list[dict]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text, fieldnames=FEASIBILITY_LEDGER_FIELDS, lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def build_dev18_feasibility_artifacts(
    *, repository_root=REPOSITORY_ROOT, output_directory=DEFAULT_OUTPUT_DIRECTORY,
) -> dict:
    """승인된 shape만 읽어 등록 후보의 원표와 support snapshot을 쓴다."""
    repository_root = Path(repository_root)
    output_directory = Path(output_directory)
    manifest_path = repository_root / "configs" / "input_manifest.yaml"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = yaml.safe_load(manifest_bytes.decode("utf-8"))
    entries = manifest["datasets"]["DEV18"]["files"]
    snapshot_path, inventory_path, inventory_sha, audit = _load_approved_inventory(
        repository_root, entries, manifest_sha,
    )
    registry, registry_sha = load_model_registry_with_sha(repository_root)
    rows = build_dev18_feasibility_rows(
        registry,
        entries,
        config_registry_sha256=registry_sha,
        input_manifest_sha256=manifest_sha,
        inventory_sha256=inventory_sha,
    )
    ledger_bytes = _csv_bytes(rows)
    summary = {
        **summarize_dev18_feasibility(rows, registry),
        "status": "complete_with_declared_static_unavailability",
        "config_registry_sha256": registry_sha,
        "input_manifest_sha256": manifest_sha,
        "inventory_sha256": inventory_sha,
        "audit_snapshot_sha256": file_sha256(snapshot_path),
        "audit_status": audit["status"],
        "builder_sha256": file_sha256(Path(__file__)),
        "feasibility_code_sha256": file_sha256(
            repository_root / "src" / "common" / "model_feasibility.py"
        ),
        "split_code_sha256": file_sha256(
            repository_root / "src" / "data_split" / "split_ratio_prefix.py"
        ),
        "data_preprocessing_sha256": file_sha256(
            repository_root / "configs" / "data_preprocessing.yaml"
        ),
        "ledger_fields": list(FEASIBILITY_LEDGER_FIELDS),
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "labels_or_scores_read": False,
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    ledger_path = output_directory / "dev18_feasibility_ledger.csv"
    summary_path = output_directory / "dev18_feasibility_summary.json"
    ledger_path.write_bytes(ledger_bytes)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "ledger_path": str(ledger_path),
        "summary_path": str(summary_path),
        "ledger_sha256": summary["ledger_sha256"],
        "row_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()
    result = build_dev18_feasibility_artifacts(output_directory=arguments.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
