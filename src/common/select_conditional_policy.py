"""개발 지원 범위 안에서 기업 입력에 맞는 설정 후보를 찾는다."""

from collections.abc import Mapping

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from src.common.model_feasibility import assess_candidate
from src.common.model_registry import validate_execution_status
from src.common.tuning_support import load_tuning_support, resolve_policy_score_variant


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}는 양의 정수여야 한다")
    return value


def match_joint_support(support, *, available_count, feature_count, training_boundary):
    """같이 관측한 행 수·채널 수·전체 학습 계획만 지원 범위로 인정한다."""
    fields = ("available_count", "feature_count", "training_boundary")
    query = (available_count, feature_count, training_boundary)
    if not isinstance(support, (list, tuple)) or not support:
        return ["conditional_group_support_missing"]
    points = []
    for entry in support:
        if not isinstance(entry, Mapping):
            return ["conditional_group_support_missing"]
        point = (entry.get("observed_row", entry.get("available_count")),
                 entry.get("feature_count"), entry.get("training_boundary"))
        if any(type(value) is not int or value < 1 for value in point) or point[0] > point[2]:
            return ["conditional_group_support_missing"]
        points.append(point)
    if query in points:
        return []
    reasons = ["joint_shape_not_observed"]
    for index, (field, value) in enumerate(zip(fields, query)):
        observed = [point[index] for point in points]
        if not min(observed) <= value <= max(observed):
            reasons.append(field + "_out_of_dev_support")
    return reasons


def match_conditional_policy(
    selection, registry, *, model, ratio, available_count, feature_count,
    test_length, training_boundary, allow_out_of_support=False,
    feasible_config_ids=None, analysis_kind="model_ratio",
):
    """같은 q·실행 가능 후보 집합을 가진 개발 집단 하나만 매칭한다.

    feasible_config_ids는 다중 세션의 교집합을 이미 검증한 내부 호출에만 쓴다.
    """
    for name, value in (
        ("available_count", available_count), ("feature_count", feature_count),
        ("test_length", test_length), ("training_boundary", training_boundary),
    ):
        _positive_integer(value, name)
    if ratio not in SUPPORTED_RATIO_PERCENTS or type(ratio) is not int:
        raise ValueError("ratio는 등록된 정수 비율이어야 한다")
    if available_count > training_boundary:
        raise ValueError("확보 행 수가 총 정상 학습 계획보다 크다")
    details = registry["models"][model]
    result = {
        "status": "unavailable", "policy": None, "candidate_ids": [],
        "support_status": "unavailable", "reasons": [],
    }
    if validate_execution_status(model, details) != "ready":
        return {**result, "reasons": ["model_not_ready"]}
    if details["target_use"] not in {"fit_full_prefix", "training_free", "strict_zero_shot"}:
        return {**result, "reasons": ["full_prefix_policy_required"]}
    known_ids = {candidate["config_id"] for candidate in details["candidates"]}
    if feasible_config_ids is None:
        feasible = {
            candidate["config_id"] for candidate in details["candidates"]
            if assess_candidate(
                model, candidate["hyperparameters"], available_count, 0,
                test_length, feature_count,
                full_prefix=details["target_use"] == "fit_full_prefix",
                official_protocol=registry.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4",
            )["status"] == "feasible"
        }
    else:
        if isinstance(feasible_config_ids, (str, bytes)):
            raise ValueError("feasible_config_ids는 후보 ID 모음이어야 한다")
        feasible = set(feasible_config_ids)
        if not feasible <= known_ids:
            raise ValueError("feasible_config_ids에 registry 밖 후보가 있다")
    result["candidate_ids"] = sorted(feasible)
    if analysis_kind not in {"model_ratio", "tier_adaptive"}:
        raise ValueError("지원하지 않는 조건부 정책 종류다")
    tier_candidates = None
    if analysis_kind == "tier_adaptive":
        tier_candidates = {
            name: sorted(candidate["config_id"] for candidate in other["candidates"]
                         if assess_candidate(name, candidate["hyperparameters"], available_count, 0,
                                             test_length, feature_count,
                                             full_prefix=other["target_use"] == "fit_full_prefix",
                                             official_protocol=registry.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4")["status"] == "feasible")
            for name, other in registry["models"].items() if other["tier"] == details["tier"]
        }
        tier_candidates = {name: candidates for name, candidates in tier_candidates.items() if candidates}
    matches = [
        policy for policy in selection.get(analysis_kind, ())
        if policy["model"] == model and policy["ratio"] == ratio
        and "group_id" in policy
        and (set(policy["candidate_ids"]) == feasible if analysis_kind == "model_ratio"
             else {name: sorted(ids) for name, ids in policy.get("candidate_ids_by_model", {}).items()} == tier_candidates)
    ]
    if len(matches) != 1:
        reason = "conditional_group_missing" if not matches else "conditional_group_ambiguous"
        return {**result, "reasons": [reason]}
    policy = matches[0]
    result["policy"] = policy
    if not feasible or policy.get("selection_status") != "selected":
        return {**result, "reasons": ["conditional_group_has_no_selected_config"]}
    if policy.get("config_id") not in feasible:
        raise ValueError("선택된 config_id가 해당 집단의 실행 가능 후보에 없다")
    resolve_policy_score_variant(policy, None)
    reasons = match_joint_support(
        policy.get("support"), available_count=available_count,
        feature_count=feature_count, training_boundary=training_boundary,
    )
    if reasons == ["conditional_group_support_missing"]:
        return {**result, "reasons": reasons}
    return {
        **result,
        "status": "matched" if not reasons or allow_out_of_support else "unavailable",
        "support_status": "out_of_dev_support" if reasons else "within_dev_support",
        "reasons": reasons,
    }


def recommend_conditional_candidates(
    selection, registry, *, nrows, dfeatures, planned_rows,
    max_rows=None, max_columns=None, evaluation_rows=None,
    support_path=None, support_sha256=None, environment_id=None,
):
    """완료 튜닝과 정확히 같은 규모의 현장 검증 후보를 반환한다."""
    for name, value in (
        ("nrows", nrows), ("dfeatures", dfeatures), ("planned_rows", planned_rows),
    ):
        _positive_integer(value, name)
    for name, value in (("max_rows", max_rows), ("max_columns", max_columns),
                        ("evaluation_rows", evaluation_rows)):
        if value is not None:
            _positive_integer(value, name)
    result = {
        "status": "unavailable", "candidates": [], "reasons": [],
        "normalized_q": None, "observed_ratio_percent": 100 * nrows / planned_rows,
        "profitability_optimal": False, "domain_generalization_guaranteed": False,
        "service_status": "unvalidated", "plan_observation_status": "unavailable",
        "missing_plan_ratios": [],
        "missing_inputs": [
            "false_positive_cost", "false_negative_cost", "anomaly_prevalence",
            "latency_cost", "tuning_and_inference_cost", "deployment_domain_validation",
            "operating_quality_requirements", "serving_resource_validation",
        ],
        "evaluation_rows_assumed": False,
    }
    if (max_rows is not None and max(nrows, planned_rows) > max_rows
            or max_columns is not None and dfeatures > max_columns):
        return {**result, "reasons": ["declared_capacity_exceeded"]}
    if nrows > planned_rows:
        return {**result, "reasons": ["available_rows_exceed_plan"]}
    observed = result["observed_ratio_percent"]
    exact_ratios = [ratio for ratio in SUPPORTED_RATIO_PERCENTS
                    if planned_rows * ratio // 100 == nrows]
    if not exact_ratios and observed < min(SUPPORTED_RATIO_PERCENTS):
        return {**result, "reasons": ["below_minimum_supported_ratio"]}
    if not exact_ratios:
        return {**result, "reasons": ["registered_prefix_not_observed"]}
    ratio = min(exact_ratios, key=lambda candidate: (abs(candidate - observed), candidate))
    result["normalized_q"] = ratio
    if support_path is None:
        return {**result, "reasons": ["tuning_support_required"]}
    if evaluation_rows is None:
        result["missing_inputs"].append("evaluation_rows")
    if not environment_id:
        result["missing_inputs"].append("environment_id")
    if evaluation_rows is None or not environment_id:
        return {**result, "reasons": ["evaluation_shape_and_environment_required"]}
    support = load_tuning_support(support_path, support_sha256, selection, registry)
    context = [point for point in support["points"]
               if point["feature_count"] == dfeatures and point["training_boundary"] == planned_rows
               and point["evaluation_rows"] == evaluation_rows and point["environment_id"] == environment_id]
    current = [point for point in context if point["ratio"] == ratio
               and point.get("observed_row", point.get("available_count")) == nrows]
    if not current:
        return {**result, "reasons": ["joint_shape_not_observed"]}
    observed_ratios = {point["ratio"] for point in context}
    result["missing_plan_ratios"] = [required for required in SUPPORTED_RATIO_PERCENTS
                                     if required >= ratio and required not in observed_ratios]
    result["plan_observation_status"] = "incomplete_grid" if result["missing_plan_ratios"] else "complete_grid"
    for model, analysis_kind in ((name, kind) for name in registry["models"]
                                 for kind in ("model_ratio", "tier_adaptive")):
        model_points = [point for point in current if point["model"] == model
                        and point.get("analysis_kind", "model_ratio") == analysis_kind]
        if not model_points:
            if analysis_kind == "model_ratio":
                result["reasons"].append(model + ":joint_shape_not_observed")
            continue
        match = match_conditional_policy(
            selection, registry, model=model, ratio=ratio, available_count=nrows,
            feature_count=dfeatures, test_length=evaluation_rows,
            training_boundary=planned_rows,
            analysis_kind=analysis_kind,
        )
        if match["status"] == "matched":
            policy = match["policy"]
            points = [point for point in model_points if all(
                point[field] == policy.get(field, "") for field in ("group_id", "config_id")
            ) and point["score_variant"] == resolve_policy_score_variant(policy, point.get("family"))]
            if not points:
                result["reasons"].append(model + ":selected_recipe_not_observed")
                continue
            model_ratios = {point["ratio"] for point in context if point["model"] == model}
            result["candidates"].append({
                "model": model, "analysis_kind": analysis_kind, "group_id": policy["group_id"],
                "config_id": policy["config_id"],
                "score_variant": policy.get("score_variant", ""),
                "score_variant_by_family": dict(policy.get("score_variant_by_family", {})),
                "score_variant_fallback": policy.get("score_variant_fallback", ""),
                "hyperparameters": dict(policy.get("hyperparameters", {})),
                "support_status": match["support_status"],
                "service_status": "unvalidated", "evidence_scope": "development_observations",
                "observations": points, "support_sha256": support_sha256,
                "missing_plan_ratios": [required for required in SUPPORTED_RATIO_PERCENTS
                                        if required >= ratio and required not in model_ratios],
            })
        else:
            result["reasons"].extend(model + ":" + reason for reason in match["reasons"])
    if result["candidates"]:
        result["status"] = "candidates_for_validation"
    return result
