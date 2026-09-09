"""완료한 새 실험의 작은 증거를 묶고 큰 산출물은 위치와 지문으로 인수한다."""

import csv
import io
import json
import os
import tarfile
from pathlib import Path

from src.common.execution_identity import SHA256_PATTERN, file_sha256
from src.common.execution_evidence import FULL_PREFIX_STORAGE_SCHEMA_VERSION
from tests.ghl_main.record_run_history import _load_histories, record_run_history


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def package_recommendation_handoff(report):
    """닫힌 명령·DB를 묶고 인수 시간·실패는 별도 이력에 기록한다."""
    if (report.get("experiment_mode") != "full_prefix_v2" or report.get("status") != "ready_for_handoff"
            or any(report.get(key) != "complete" for key in
                   ("execution_status", "scoring_selection_status", "recommendation_status"))):
        raise ValueError("추천 전달 묶음에는 실행·채점·추천 자료 완료가 모두 필요하다")
    root = REPOSITORY_ROOT.resolve()
    result_directory = Path(report["result_directory"]).resolve()
    result_directory.relative_to(root / "experiments/01_ghl_main/results/dev18_tuning/full_prefix_v2")
    with record_run_history(result_directory / "handoff/run_history", identity={
        "kind": "recommendation_handoff", "budget_id": report["budget_id"],
        "command_run_id": report["run_id"], "cost_scope": "handoff_archive_and_hash",
    }) as history:
        handoff = _package_recommendation_handoff(report, current_history_file=Path(history["history_file"]))
        history["artifacts"] = {name: {key: item[key] for key in ("sha256", "bytes")}
                                for name, item in handoff.items()}
    history_path = Path(history["history_file"])
    handoff["history"] = {"file": str(history_path), "sha256": file_sha256(history_path),
                          "bytes": history_path.stat().st_size}
    return handoff


def _package_recommendation_handoff(report, *, current_history_file):
    root = REPOSITORY_ROOT.resolve()
    result_directory = Path(report["result_directory"]).resolve()
    included, external, fingerprints = {}, {}, {}

    def reference(path, *, sha256=None, size=None, include=True, kind="evidence", allow_external=False):
        path = Path(path)
        path = (path if path.is_absolute() else root / path).resolve()
        if not allow_external:
            path.relative_to(root)
        if not path.is_file():
            raise ValueError(f"인수 파일이 없다: {path}")
        if path not in fingerprints:
            fingerprints[path] = {"file": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        item = fingerprints[path]
        if (sha256 is not None and item["sha256"] != sha256) or (size is not None and item["bytes"] != size):
            raise ValueError(f"인수 파일 지문이 저장 근거와 다르다: {path}")
        destination = included if include else external
        destination[path] = {**item, "kind": kind}
        if include:
            destination[path]["archive_name"] = path.relative_to(root).as_posix()
        return path

    def read_json(path, **kwargs):
        return json.loads(reference(path, **kwargs).read_text(encoding="utf-8"))

    budget_ref, manifest_ref = (report["artifacts"][name] for name in ("budget", "manifest"))
    budget_path = reference(budget_ref["file"], sha256=budget_ref["sha256"])
    budget_path.relative_to(root / "experiments/01_ghl_main/snapshots/dev18_selection/full_prefix_v2")
    budget_artifact = json.loads(budget_path.read_text(encoding="utf-8"))
    budget = budget_artifact["budget"]
    if budget["budget_id"] != report["budget_id"] or budget.get("experiment_mode") != "full_prefix_v2":
        raise ValueError("인수 예산이 완료한 새 실험과 다르다")
    for field, name in (("input_manifest_sha256", "input_manifest.yaml"),
                        ("data_preprocessing_sha256", "data_preprocessing.yaml")):
        reference(root / "configs" / name, sha256=budget[field])
    reference(root / "configs/model_registry.yaml",
              sha256=budget_artifact["attestation"]["config_registry_sha256"])
    recommendation = report["recommendation_evidence"]
    saved_recommendation = read_json(recommendation["receipt_file"], sha256=recommendation["receipt_sha256"])
    if (not saved_recommendation.get("recommendation_complete")
            or saved_recommendation["identity"]["budget_id"] != report["budget_id"]
            or saved_recommendation["identity"]["project_commit"] != report["project_commit"]):
        raise ValueError("추천 영수증이 완료한 실험 신원과 다르다")
    for key in ("database", "exports"):
        if recommendation[key] != saved_recommendation[key]:
            raise ValueError("추천 전달 경로가 최종 영수증과 다르다")
    runtime = saved_recommendation["environment"].get("runtime_snapshot", {})
    for name in ("requirements", "environment_config"):
        if name in runtime:
            reference(runtime[name]["file"], sha256=runtime[name]["sha256"])
    if "sha256" in runtime:
        runtime_path = root / ".runtime/runtime.json"
        if not runtime_path.is_file():
            runtime_path = root / "experiments/checks/reference_code/environment/runtime.json"
        saved_runtime = read_json(runtime_path, sha256=runtime["sha256"])
        if isinstance(saved_runtime.get("pip_freeze"), str):
            reference(runtime_path.parent / saved_runtime["pip_freeze"])
    backup = recommendation["database"]
    backup_path = Path(backup["file"]).resolve()
    if backup_path != result_directory / "recommendation_evidence" / f"recommendation_backup_{backup['sha256']}.sqlite3":
        raise ValueError("전달 DB는 완료 영수증의 일관된 backup이어야 한다")
    for item in (backup, *recommendation["exports"]):
        path = reference(item["file"], sha256=item["sha256"], size=item["bytes"])
        path.relative_to(result_directory / "recommendation_evidence")
    contract = read_json(budget_path.parent / "recommendation_contract.json")
    if contract["identity"] != saved_recommendation["identity"]:
        raise ValueError("추천 계약과 최종 영수증의 실험 신원이 다르다")
    feasibility = read_json(budget_path.parent / "dev18_feasibility_summary.json")
    reference(budget_path.parent / "dev18_feasibility_ledger.csv", sha256=feasibility["ledger_sha256"])
    for name, digest in report["outputs"].items():
        if Path(name).name != name or Path(name).suffix not in {".csv", ".json"}:
            raise ValueError("선택 출력은 결과 폴더의 명시한 작은 원표여야 한다")
        reference(result_directory / name, sha256=digest)
    from tests.ghl_main.run_dev18_tuning import _validate_ell_max, validate_vus_evidence

    ell_max = _validate_ell_max(
        reference(budget_path.parent.parent / "dev18_ell_max.json", kind="ell_max_snapshot"),
        budget["input_manifest_sha256"],
    )
    evaluator = validate_vus_evidence(
        reference(root / "experiments/checks/reference_code/vus_pr/official_tsb_ad_comparison.json",
                  kind="vus_comparison"),
        root / "src/채점기/vus_pr.py",
    )
    ledger_path = reference(result_directory / "dev18_trial_score_ledger.csv",
                            sha256=report["outputs"]["dev18_trial_score_ledger.csv"])
    with ledger_path.open(encoding="utf-8", newline="") as source:
        ledger = list(csv.DictReader(source))
    if not ledger or any(row.get("ell_max_id") != ell_max["ell_max_id"]
                         or row.get("evaluator_sha256") != evaluator["evaluator_sha256"] for row in ledger):
        raise ValueError("인수 채점 원표의 ell_max·evaluator 신원이 전달 근거와 다르다")
    for name in ("selection_complete.json", "execution_complete.json"):
        path = result_directory / name
        if name == "execution_complete.json" and not path.exists():
            continue
        receipt = read_json(path)
        if receipt["budget_id"] != report["budget_id"]:
            raise ValueError("완료 영수증이 다른 예산을 가리킨다")
        if name == "selection_complete.json" and receipt != report:
            raise ValueError("선택 완료 영수증이 현재 전달 결과와 다르다")
    costs = read_json(result_directory / "tuning_cost_history.json")
    if costs["budget_id"] != report["budget_id"]:
        raise ValueError("비용 이력이 다른 예산을 가리킨다")
    for item in costs.get("unassigned_command_wall", {}).get("history_files", []):
        history = read_json(item["file"], sha256=item["sha256"], kind="unassigned_command_history")
        if history["identity"].get("budget_id") is not None or history["identity"].get("kind") != "tuning_command":
            raise ValueError("미배정 명령 이력에 예산이 배정됐거나 명령 종류가 다르다")
    resource_directory = root / "experiments/checks/reference_code/active_models/full_prefix_v2"
    resource_report = read_json(resource_directory / "resource_gate.json", kind="resource_report")
    resource_identity = {"project_commit": report["project_commit"], "budget_id": report["budget_id"],
                         "input_manifest_sha256": budget["input_manifest_sha256"],
                         "environment": {key: value for key, value in saved_recommendation["environment"].items()
                                         if key != "runtime_snapshot"}}
    resource_rows = resource_report.get("results")
    if (any(resource_report.get(key) != value for key, value in resource_identity.items())
            or resource_report.get("status") != "passed"
            or set(resource_report.get("checked_models", [])) != {row["model"] for row in budget["execution_panel"]}
            or not isinstance(resource_rows, list) or not resource_rows
            or any(not isinstance(row, dict) or row.get("status") != "passed" for row in resource_rows)
            or {row["model"] for row in resource_rows if row.get("model")} != set(resource_report["checked_models"])):
        raise ValueError("전달 자원 검사가 완료한 실험의 소스·환경·예산과 다르다")
    probe_directory = resource_directory / "resource_probe_history"
    if Path(resource_report.get("probe_history_directory", "")).resolve() != probe_directory:
        raise ValueError("자원 probe 이력 폴더가 저장 계약과 다르다")
    for result in resource_rows:
        item = result.get("probe_history")
        requires_probe = (result.get("model") in {"PaAno", "GDN", "TimeRCD", "TSPulse"}
                          or (result.get("model") == "PCA_LEGACY" and result.get("measurement_kind") == "process_rss"))
        if item is None and not requires_probe:
            continue
        if (not isinstance(item, dict) or not isinstance(item.get("file"), str) or not item["file"].strip()
                or not isinstance(item.get("sha256"), str) or not SHA256_PATTERN.fullmatch(item["sha256"])):
            raise ValueError("자원 probe 이력의 파일·SHA 연결이 없다")
        path = reference(item["file"], sha256=item["sha256"], kind="resource_probe_history")
        if path.parent != probe_directory:
            raise ValueError("자원 probe가 다른 이력 폴더를 가리킨다")
        history = json.loads(path.read_text(encoding="utf-8"))
        if (history.get("status") != "complete" or history.get("identity", {}).get("kind") != "resource_probe"
                or any(history["identity"].get(key) != value for key, value in resource_identity.items())
                or history.get("result") != {key: value for key, value in result.items() if key != "probe_history"}):
            raise ValueError("자원 probe 원본이 전달 검사의 실행 신원·측정값과 다르다")
    for path, _, history in _load_histories(probe_directory, report["budget_id"]):
        if history["identity"].get("kind") != "resource_probe":
            raise ValueError("자원 probe 폴더에 다른 종류의 실행 이력이 있다")
        reference(path, kind="resource_probe_history")
    for path in sorted((result_directory / "handoff/run_history").glob("*.json")):
        if path.resolve() == current_history_file.resolve():
            continue
        history = json.loads(path.read_text(encoding="utf-8"))
        if history["identity"].get("budget_id") == report["budget_id"]:
            reference(path, kind="handoff_history")
    history_refs = [item for name in ("command_wall", "model_attempt_wall")
                    for item in costs[name]["history_files"]]
    history_refs.append({"file": report["history_file"]})
    for item in history_refs:
        history = read_json(item["file"], sha256=item.get("sha256"))
        if history["identity"].get("budget_id") != report["budget_id"]:
            raise ValueError("다른 예산의 실행 이력을 전달 묶음에 넣지 않는다")
        if Path(item["file"]).resolve() == Path(report["history_file"]).resolve() and (
            history["status"] != "complete" or history["run_id"] != report["run_id"]
        ):
            raise ValueError("현재 명령 이력이 닫힌 뒤 전달 묶음을 만든다")
        for source, files in (("training_complete", history.get("training_complete", {}).get("files", {})),
                              ("result_files", history.get("result_files", {}))):
            for name, artifact in files.items():
                reference(artifact["file"], sha256=artifact["sha256"], size=artifact["bytes"],
                          include=name not in {"checkpoint", "scaler_state"}, kind=f"{source}_{name}")
        previous = history.get("previous_receipt_file")
        if previous:
            receipt = read_json(previous)
            if receipt.get("budget_id") != report["budget_id"]:
                raise ValueError("보존 영수증이 다른 예산을 가리킨다")
        for previous in history.get("prior_failed_checks", []):
            digest = Path(previous).stem.rsplit("__", 1)[-1]
            failed_check = read_json(previous, sha256=digest)
            if (failed_check.get("budget_id", report["budget_id"]) != report["budget_id"]
                    or ("budget_id" not in failed_check and failed_check.get("registry_sha256")
                        != budget_artifact["attestation"]["config_registry_sha256"])):
                raise ValueError("보존한 실패 검사가 현재 실험 신원과 다르다")

    manifest_path = reference(manifest_ref["file"], sha256=manifest_ref["sha256"])
    with manifest_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    expected = {(str(series), panel["model"], panel["config_id"], str(panel["physical_ratio"]),
                 str(panel["seed"]), variant) for panel in budget["execution_panel"]
                for series in panel["series_ids"]
                for variant in (*panel["primary_score_variants"], *panel["diagnostic_score_variants"])}
    keys = [tuple(row[field] for field in ("series", "model", "config_id", "physical_ratio", "seed", "score_variant"))
            for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("전달 manifest의 실행 목록이 새 예산과 다르다")
    checkpoints, seen_metadata = {}, set()
    for row in rows:
        if row["status"] != "complete" or row["budget_id"] != report["budget_id"]:
            raise ValueError("전달 manifest에 미완료 또는 다른 실험 결과가 있다")
        raw = reference(row["score_file"], sha256=row["score_sha256"], include=False, kind="raw_score")
        if not raw.name.endswith("__raw__trainnorm.npy"):
            raise ValueError("전달 manifest의 기본 점수 파일명이 저장 계약과 다르다")
        metadata_path = reference(row["metadata_file"], sha256=row["metadata_sha256"])
        if metadata_path in seen_metadata:
            continue
        seen_metadata.add(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for item in metadata.get("score_files", []):
            reference(item["file"], sha256=item["sha256"], size=item["bytes"], include=False, kind="score")
        if (metadata["model"], metadata["config_id"], int(metadata["series"]), metadata["ratio"],
                metadata["seed"], metadata.get("score_variant") or "") != (
                    row["model"], row["config_id"], int(row["series"]), int(row["physical_ratio"]),
                    int(row["seed"]), row["score_variant"]):
            raise ValueError("전달 metadata의 실행 신원이 manifest와 다르다")
        attempt = metadata["execution_attempt"]
        history = read_json(attempt["history_file"])
        if history["run_id"] != attempt["run_id"] or history["identity"]["budget_id"] != report["budget_id"]:
            raise ValueError("모델 시도 이력이 실제 실행 신원과 다르다")
        for variant in ("raw", "smoothed"):
            score_path = raw.with_name(raw.name.replace("__raw__trainnorm.npy", f"__{variant}__trainnorm.npy"))
            reference(score_path, include=False, kind=f"{variant}_score")
            if metadata["channel_count"] or metadata.get("native_channel_score_shape"):
                reference(score_path.with_name(score_path.stem + "__channels.npy"), include=False, kind=f"{variant}_channel_score")
        snapshot_ref = metadata["run_snapshot"]
        snapshot = read_json(snapshot_ref["file"], sha256=snapshot_ref["sha256"])
        if (snapshot["project_commit"] != report["project_commit"]
                or snapshot["storage_schema_version"] != FULL_PREFIX_STORAGE_SCHEMA_VERSION
                or snapshot["environment"] != saved_recommendation["environment"]):
            raise ValueError("전달 snapshot의 소스·환경·저장 계약이 다르다")
        for name, item in metadata["training_files"].items():
            reference(item["file"], sha256=item["sha256"], size=item.get("bytes"),
                      include=name not in {"checkpoint", "scaler_state"}, kind=name)
        calibration = metadata.get("calibration_reference")
        if calibration:
            reference(metadata_path.parent / calibration["file"], sha256=calibration["sha256"],
                      include=False, kind="calibration_reference")
        specification = snapshot.get("spec", {})
        if row["model"] in {"TimeRCD", "TSPulse"} or specification.get("source_checkpoint_sha256") not in {None, "none"}:
            if (specification.get("model") != row["model"]
                    or not isinstance(specification.get("source_commit"), str) or not specification["source_commit"].strip()
                    or any(not isinstance(specification.get(key), str) or not SHA256_PATTERN.fullmatch(specification[key])
                           for key in ("source_checkpoint_sha256", "checkpoint_config_sha256"))):
                raise ValueError("사전학습 모델 snapshot에 source·checkpoint·config 신원이 없다")
            source = {key: specification[key] for key in ("source_commit", "source_checkpoint_sha256", "checkpoint_config_sha256")}
            if row["model"] in checkpoints and checkpoints[row["model"]] != source:
                raise ValueError("같은 모델의 사전학습 원본 신원이 서로 다르다")
            checkpoints[row["model"]] = source

    if checkpoints:
        from tests.checks.run_checkpoint_smoke import MODEL_DIRECTORIES

        for model, source in checkpoints.items():
            source_report = read_json(root / "experiments/checks/reference_code/active_models/full_prefix_v2/checkpoints"
                                      / MODEL_DIRECTORIES[model] / "dev18_checkpoint_smoke.json")
            if (source_report["status"] != "passed" or source_report["source"]["commit"] != source["source_commit"]
                    or source_report["budget_id"] != budget["budget_id"]
                    or source_report["budget_sha256"] != budget["budget_sha256"]
                    or source_report["checkpoint"]["sha256"] != source["source_checkpoint_sha256"]
                    or source_report["config"]["sha256"] != source["checkpoint_config_sha256"]):
                raise ValueError("사전학습 원본 근거가 실제 실행 소스와 다르다")
            for name in ("checkpoint", "config"):
                item = source_report[name]
                if (not isinstance(item.get("path"), str) or not item["path"].strip()
                        or type(item.get("bytes")) is not int or item["bytes"] <= 0):
                    raise ValueError("사전학습 원본 파일의 경로·크기가 없다")
                reference(item["path"], sha256=item["sha256"], size=item["bytes"],
                          include=False, kind=f"pretrained_{name}", allow_external=True)

    payload = {"schema_version": 1, "budget_id": report["budget_id"], "project_commit": report["project_commit"],
               "status": "complete", "selection_receipt_sha256": fingerprints[result_directory / "selection_complete.json"]["sha256"],
               "included_files": [included[path] for path in sorted(included)],
               "external_artifacts": [external[path] for path in sorted(external)],
               "large_artifact_policy": "retained at recorded paths; file bytes are not included in this archive"}
    content = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output_directory = result_directory / "handoff"
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = output_directory / "recommendation_handoff.tar.gz"
    manifest_path = output_directory / "handoff_manifest.json"
    temporary_archive = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.tmp")
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(temporary_archive, "w:gz") as archive:
            for path, item in sorted(included.items()):
                archive.add(path, arcname=item["archive_name"], recursive=False)
                if file_sha256(path) != item["sha256"]:
                    raise ValueError(f"묶는 동안 인수 파일이 바뀌었다: {path}")
            member = tarfile.TarInfo("handoff_manifest.json")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        temporary_manifest.write_bytes(content)
        temporary_archive.replace(archive_path)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_archive.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return {name: {"file": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for name, path in (("archive", archive_path), ("manifest", manifest_path))}
