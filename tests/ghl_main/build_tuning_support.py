"""완료 튜닝의 관측량·센서 수와 실행 근거를 함께 보존한다."""

import hashlib
import json

from src.common.equal_trial_budget import registry_space_sha256
from src.common.execution_evidence import (
    FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
    FULL_PREFIX_STORAGE_SCHEMA_VERSION,
    validate_execution_evidence_for_run,
)
from src.common.tuning_support import resolve_policy_score_variant
from tests.ghl_main import run_dev18_tuning as tuning
from tests.ghl_main.compare_execution_runtimes import compare_execution_runtime, load_runtime_references
from tests.ghl_main.select_ratio_tuning import _validate_full_prefix_trials


def _digest(value):
    return hashlib.sha256(tuning._json(value).encode()).hexdigest()


def _read_evidence(reference, cache):
    if not isinstance(reference, dict) or not reference.get("file") or not reference.get("sha256"):
        raise ValueError("작은 실행 증거의 파일·SHA 연결이 없다")
    path = (tuning.REPOSITORY_ROOT / reference["file"]).resolve()
    try:
        path.relative_to(tuning.REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError("실행 증거가 저장소 밖을 가리킨다") from error
    if path.suffix != ".json":
        raise ValueError("지원 근거에는 metadata·snapshot JSON만 읽는다")
    key = (str(path), reference["sha256"])
    if key not in cache:
        try:
            serialized = path.read_bytes()
            if hashlib.sha256(serialized).hexdigest() != reference["sha256"]:
                raise ValueError("작은 실행 증거의 SHA가 다르다")
            payload = json.loads(serialized)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("작은 실행 증거를 읽을 수 없다") from error
        if not isinstance(payload, dict):
            raise ValueError("실행 증거 JSON은 객체여야 한다")
        cache[key] = payload
    return cache[key]


def _read_execution(row, registry, budget, entry, cache, runtime_references=None):
    metadata_reference = {"file": row.get("metadata_file"), "sha256": row.get("metadata_sha256")}
    metadata = _read_evidence(metadata_reference, cache)
    snapshot_reference = metadata.get("run_snapshot")
    snapshot = _read_evidence(snapshot_reference, cache)
    expected = {"model": row["model"], "tier": row["tier"], "config_id": row["config_id"],
                "ratio": int(row["physical_ratio"]), "seed": int(row["seed"]),
                "dataset_role": "development", "split_role": "dev18_selection",
                "input_manifest_sha256": budget["input_manifest_sha256"],
                "target_use": registry["models"][row["model"]]["target_use"]}
    if expected["target_use"] not in {"fit_full_prefix", "training_free", "strict_zero_shot"}:
        raise ValueError("이전 분할의 실행을 현재 지원 근거로 쓰지 않는다")
    spec = snapshot.get("spec", {})
    if any(metadata.get(field) != value or spec.get(field) != value for field, value in expected.items()):
        raise ValueError("metadata·snapshot의 물리 실행 신원이 manifest와 다르다")
    details = registry["models"][row["model"]]
    if row["tier"] != details["tier"]:
        raise ValueError("manifest의 Tier가 현재 registry와 다르다")
    recipe = {field: details.get(field) for field in (
        "source_commit", "source_checkpoint_sha256", "checkpoint_config_sha256",
        "checkpoint_revision", "preprocess_recipe",
    )}
    recipe.update(hyperparameters=tuning._hyperparameters(registry, row["model"], row["config_id"]),
                  common_recipe=registry["common_recipe"], common_recipe_id=registry["common_recipe_id"],
                  series=row["series"], series_input_sha256=entry["sha256"])
    if any(field not in spec or spec[field] != value for field, value in recipe.items()):
        raise ValueError("snapshot의 source·입력·recipe가 현재 봉인과 다르다")
    if any(not metadata.get(field) or metadata[field] != spec.get(field)
           for field in ("common_recipe_id", "config_registry_sha256")):
        raise ValueError("metadata의 recipe·registry 신원이 snapshot과 다르다")
    verified = {field: entry[field] for field in ("name", "source_directory", "size_bytes", "sha256")}
    input_identity = snapshot.get("input_identity", {})
    if any(input_identity.get(field) != value for field, value in {
        **verified, "dataset": "DEV18", "input_manifest_sha256": budget["input_manifest_sha256"],
        "verified_files": [verified],
    }.items()):
        raise ValueError("snapshot 입력 신원이 봉인한 series 파일과 다르다")
    ranges = {"source": entry["name"], "source_directory": entry["source_directory"],
              "normal_training": [0, entry["training_boundary"]],
              "test_sessions": [[entry["training_boundary"], entry["row_count"]]]}
    if snapshot.get("source_ranges") != ranges:
        raise ValueError("snapshot의 원본 시점 범위가 봉인 입력과 다르다")
    if (metadata.get("dataset") != "DEV18"
            or str(metadata.get("series")).zfill(2) != row["series"]
            or "score_variant" not in metadata
            or (metadata.get("score_variant") or "") != row["score_variant"]
            or spec.get("common_recipe", {}).get("training_split") != "full_prefix_v2"):
        raise ValueError("metadata의 파일·head 또는 전체 prefix 계약이 다르다")
    evidence = validate_execution_evidence_for_run(
        metadata.get("execution_evidence"), dataset_role="development", target_use=expected["target_use"],
    )
    if (evidence["measurement_protocol_id"] != FULL_PREFIX_MEASUREMENT_PROTOCOL_ID
            or metadata.get("storage_schema_version") != FULL_PREFIX_STORAGE_SCHEMA_VERSION
            or snapshot.get("storage_schema_version") != FULL_PREFIX_STORAGE_SCHEMA_VERSION):
        raise ValueError("현재 튜닝 측정 프로토콜이 아니다")
    if (len(evidence["test_sessions"]) != 1 or evidence["test_sessions"][0]["observation_count"]
            != entry["row_count"] - entry["training_boundary"]):
        raise ValueError("Dev18 지원 근거는 test session 하나여야 한다")
    environment = snapshot.get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("실행 환경 근거가 없다")
    return {
        **{field: row[field] for field in ("series", "family", "model", "tier", "config_id", "score_variant")},
        "physical_ratio": int(row["physical_ratio"]), "seed": int(row["seed"]),
        "metadata_file": metadata_reference["file"], "metadata_sha256": metadata_reference["sha256"],
        "snapshot_file": snapshot_reference["file"], "snapshot_sha256": snapshot_reference["sha256"],
        "physical_execution_id": "p" + _digest([snapshot_reference["file"], snapshot_reference["sha256"]]),
        "measurement_scope": "completed_single_execution",
        "resource_usage": evidence["resource_usage"],
        "input_identity": input_identity, "source_ranges": ranges,
        "environment": environment, "environment_id": _digest(environment), "execution_evidence": evidence,
        **compare_execution_runtime(
            snapshot, evidence["runtime_seconds"], runtime_references or [],
            repository_root=tuning.REPOSITORY_ROOT,
        ),
    }


def _build_tier_support_group(policy, groups, series_ids):
    by_series = {series: {} for series in series_ids}
    for group in groups.values():
        if group["tier"] == policy["tier"] and group["ratio"] == policy["ratio"]:
            for series in group["series_ids"]:
                by_series[series][group["model"]] = group
    signatures = {
        series: {model: group["candidate_ids"] for model, group in models.items() if group["candidate_ids"]}
        for series, models in by_series.items()
    }
    candidates = policy["candidate_ids_by_model"]
    members = [series for series in series_ids if signatures[series] == candidates]
    identity = {"tier": policy["tier"], "ratio": policy["ratio"], "series_ids": members,
                "candidate_ids_by_model": candidates}
    if (not members or policy["series_ids"] != members
            or policy["group_id"] != "g" + _digest(identity)[:12]):
        raise ValueError("Tier 선택 집단이 봉인 예산의 공동 후보 조건과 다르다")
    if not candidates:
        return {**identity, "candidate_ids": [], "support": []}
    model = policy["model"]
    if model not in candidates or policy.get("selected_model") != model:
        raise ValueError("Tier 선택 모델이 공동 후보 집단에 없다")
    return {
        **identity, "model": model, "candidate_ids": candidates[model],
        "support": [next(item for item in by_series[series][model]["support"]
                         if item["series"] == series) for series in members],
    }


def build_tuning_support(selection, registry, budget, ledger, manifest_rows):
    """완료한 선택의 실제 joint point를 모으며 서비스 검증을 주장하지 않는다."""
    if budget.get("experiment_mode") != "full_prefix_v2":
        raise ValueError("지원 근거는 full_prefix_v2 완료 결과만 소비한다")
    if budget.get("registry_space_sha256") != registry_space_sha256(registry):
        raise ValueError("현재 registry 후보 공간이 튜닝 예산과 다르다")
    try:
        input_manifest, input_sha = tuning.load_input_manifest_role(
            tuning.REPOSITORY_ROOT / "configs/input_manifest.yaml", "development", "dev18_selection",
        )
    except OSError as error:
        raise ValueError("봉인한 입력 manifest를 읽을 수 없다") from error
    if input_sha != budget["input_manifest_sha256"]:
        raise ValueError("입력 manifest SHA가 튜닝 예산과 다르다")
    entries = {entry["series"]: entry for entry in input_manifest["datasets"]["DEV18"]["files"]}
    if len(entries) != len(input_manifest["datasets"]["DEV18"]["files"]):
        raise ValueError("입력 manifest의 series가 중복됐다")
    trials = _validate_full_prefix_trials(ledger, budget)
    trial_by_key = {tuple(row[field] for field in (
        "model", "config_id", "ratio", "seed", "series", "score_variant",
    )): row for row in trials}
    manifest_by_key = {}
    for row in manifest_rows:
        key = tuning._manifest_key(row)
        if key in manifest_by_key:
            raise ValueError("지원 근거의 manifest에 중복 물리 key가 있다")
        manifest_by_key[key] = row
    groups = {(panel["model"], group["group_id"]): group for panel in budget["model_panels"]
              for group in panel["groups"]}
    physical_by_logical = {}
    for execution in budget["execution_panel"]:
        for ratio in execution["logical_ratios"]:
            for series in execution["series_ids"]:
                for variant in execution["primary_score_variants"]:
                    key = (execution["model"], execution["config_id"], ratio, series, variant)
                    physical_by_logical.setdefault(key, []).append(execution)
    points, executions, cache = [], {}, {}
    runtime_references = load_runtime_references(
        tuning.REPOSITORY_ROOT / "experiments/01_ghl_main/logs/run_history/model_attempts", budget["budget_id"],
    )
    seen_groups, seen_policies = set(), set()
    policies = [(kind, policy) for kind in ("model_ratio", "tier_adaptive")
                for policy in selection.get(kind, [])]
    for analysis_kind, policy in policies:
        policy_key = (analysis_kind, policy["group_id"])
        if policy_key in seen_policies or policy.get("analysis_kind", analysis_kind) != analysis_kind:
            raise ValueError("선택 정책 종류 또는 조건집단이 중복·불일치한다")
        seen_policies.add(policy_key)
        if analysis_kind == "model_ratio":
            group_key = (policy["model"], policy["group_id"])
            if group_key not in groups:
                raise ValueError("선택 정책에 미등록 조건집단이 있다")
            seen_groups.add(group_key)
            group = groups[group_key]
            fields = ("model", "tier", "ratio", "series_ids", "candidate_ids", "support")
        else:
            group = _build_tier_support_group(policy, groups, budget["series_ids"])
            fields = ("tier", "ratio", "series_ids", "candidate_ids_by_model")
            if group["candidate_ids"]:
                fields += ("model", "candidate_ids", "support")
        if any(policy.get(field) != group[field] for field in fields):
            raise ValueError("선택 정책의 조건집단이 봉인 예산과 다르다")
        if policy["selection_status"] != "selected":
            if group["candidate_ids"]:
                raise ValueError("실행 가능한 조건집단의 선택이 빠졌다")
            continue
        if policy["config_id"] not in group["candidate_ids"]:
            raise ValueError("선택한 설정이 조건집단 후보에 없다")
        resolve_policy_score_variant(policy, None)
        details = registry["models"][policy["model"]]
        if (policy.get("hyperparameters") != tuning._hyperparameters(registry, policy["model"], policy["config_id"])
                or analysis_kind == "model_ratio" and any(
                    policy.get(field) != details[field] for field in ("source_commit", "source_checkpoint_sha256")
                )):
            raise ValueError("선택 정책의 source·설정이 현재 registry와 다르다")
        for support in group["support"]:
            count_field = "observed_row" if budget.get("schema_version", 2) >= 3 else "available_count"
            point = {field: policy[field] for field in ("model", "group_id", "ratio", "config_id")}
            point["score_variant"] = resolve_policy_score_variant(policy, support["family"])
            point["analysis_kind"] = analysis_kind
            point.update({field: support[field] for field in (
                "series", "family", count_field, "feature_count", "training_boundary",
            )})
            entry = entries.get(point["series"])
            if (entry is None or any(support[field] != entry[field] for field in (
                "family", "name", "feature_count", "training_boundary",
            )) or point[count_field] != entry["training_boundary"] * point["ratio"] // 100):
                raise ValueError("지원 지점의 파일·관측량·센서 수가 봉인 입력과 다르다")
            logical_key = (point["model"], point["config_id"], point["ratio"], point["series"], point["score_variant"])
            planned = physical_by_logical.get(logical_key, [])
            if not planned:
                raise ValueError("선택한 head·설정의 실행 예산이 없다")
            execution_ids, trial_scores, environments, evaluation_lengths = [], [], set(), set()
            for physical in sorted(planned, key=lambda row: row["seed"]):
                manifest_key = tuple(map(str, (point["series"], point["model"], point["config_id"],
                                               physical["physical_ratio"], physical["seed"], point["score_variant"])))
                row = manifest_by_key.get(manifest_key)
                trial = trial_by_key[(*logical_key[:3], physical["seed"], *logical_key[3:])]
                if (row is None or row.get("status") != "complete" or row.get("primary_score") != "true"
                        or row.get("budget_id") != budget["budget_id"] or row.get("family") != point["family"]
                        or trial.get("normalization") != "trainnorm"
                        or any(not row.get(field) or row.get(field) != trial.get(field)
                               for field in ("score_file", "score_sha256"))):
                    raise ValueError("선택한 seed의 완료 manifest·ledger 점수 신원이 다르다")
                execution_id = "e" + _digest([row.get("metadata_file"), row.get("metadata_sha256")])
                if execution_id not in executions:
                    executions[execution_id] = _read_execution(
                        row, registry, budget, entry, cache, runtime_references,
                    )
                observed = executions[execution_id]
                if any(str(observed[field]) != str(row[field]) for field in (
                    "series", "model", "config_id", "physical_ratio", "seed", "score_variant",
                )):
                    raise ValueError("같은 metadata가 서로 다른 물리 실행에 연결됐다")
                evidence = observed["execution_evidence"]
                sessions = evidence["training_sessions"]
                target_use = registry["models"][point["model"]]["target_use"]
                if target_use == "fit_full_prefix" and (len(sessions) != 1 or any(
                    sessions[0][field] != value for field, value in (
                        ("observed_row", point.get("observed_row", point.get("available_count"))),
                        ("training_boundary", point["training_boundary"]),
                    )
                )):
                    raise ValueError("실제 전체 prefix 학습량이 지원 지점과 다르다")
                execution_ids.append(execution_id)
                environments.add(observed["environment_id"])
                evaluation_lengths.add(evidence["test_sessions"][0]["observation_count"])
                trial_scores.append({"seed": physical["seed"], **{field: trial[field] for field in (
                    "vus_pr", "score_file", "score_sha256", "evaluator_sha256", "ell_max_id",
                )}})
            if len(environments) != 1 or len(evaluation_lengths) != 1:
                raise ValueError("같은 지원 지점의 seed별 실행 환경·평가 길이가 다르다")
            points.append({**point, "evaluation_rows": evaluation_lengths.pop(),
                           "environment_id": environments.pop(), "execution_ids": execution_ids,
                           "trial_vus_pr": trial_scores,
                           **{field: policy[field] for field in (
                               "family_count", "validation_status", "family_macro_vus_pr",
                           )}, "selection_procedure_lofo_vus_pr": policy["family_lofo_vus_pr"]})
    if seen_groups != groups.keys():
        raise ValueError("지원 근거에서 선택 조건집단이 빠졌다")
    return {
        "schema_version": 2 if budget.get("schema_version", 2) >= 3 else 1,
        "budget_id": budget["budget_id"],
        "selection_rule_id": budget["selection_rule_id"],
        "registry_space_sha256": budget["registry_space_sha256"], "selection_sha256": _digest(selection),
        "input_manifest_sha256": input_sha, "resource_status": "not_configured",
        "observation_status": "complete", "service_status": "unvalidated",
        "points": points, "executions": executions,
    }
