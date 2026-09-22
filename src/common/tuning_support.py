"""튜닝의 관측 지점을 읽고 모델별 규모를 요약한다."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from src.common.equal_trial_budget import _canonical_bytes, registry_space_sha256
from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS


def resolve_policy_score_variant(policy, family):
    """family별 선택을 저장·실행에 쓰는 실제 head로 바꾼다."""
    variant = policy.get("score_variant", "")
    by_family = policy.get("score_variant_by_family", {})
    fallback = policy.get("score_variant_fallback", "")
    if (not isinstance(variant, str) or not isinstance(by_family, Mapping)
            or not isinstance(fallback, str)):
        raise ValueError("선택 정책의 head·family 매핑·fallback 형식이 잘못됐다")
    native_heads = {"time", "fft", "pred", "ensemble"}
    if variant == "family_selected":
        if (policy.get("model") != "TSPulse" or not by_family or fallback != "time"
                or any(not isinstance(name, str) or not name.strip()
                       or not isinstance(head, str) or head not in native_heads
                       for name, head in by_family.items())
                or family is not None and not isinstance(family, str)):
            raise ValueError("TSPulse의 family별 head 매핑 또는 time fallback이 잘못됐다")
        return by_family.get(family, fallback)
    if (by_family or fallback
            or policy.get("model") == "TSPulse" and variant not in native_heads | {"raw_max"}):
        raise ValueError("선택 정책의 실제 head 또는 불필요한 family 매핑이 잘못됐다")
    return variant


def load_tuning_support(path, expected_sha256, selection, registry):
    """완료 영수증의 지문과 현재 선택표가 일치하는 근거만 읽는다."""
    payload = Path(path).read_bytes()
    if not expected_sha256 or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("튜닝 규모 근거가 완료 영수증의 SHA-256과 다르다")
    report = json.loads(payload)
    if (report.get("schema_version") not in {1, 2} or report.get("observation_status") != "complete"
            or report.get("service_status") != "unvalidated"
            or report.get("registry_space_sha256") != registry_space_sha256(registry)
            or report.get("selection_sha256") != hashlib.sha256(_canonical_bytes(selection)).hexdigest()):
        raise ValueError("튜닝 규모 근거의 버전·완료 상태·선택표·registry가 다르다")
    policies = [(kind, policy) for kind in ("model_ratio", "tier_adaptive")
                for policy in selection.get(kind, [])]
    if not policies or any(policy.get("budget_id") != report.get("budget_id") for _, policy in policies):
        raise ValueError("튜닝 규모 근거의 budget이 현재 선택표와 다르다")
    for _, policy in policies:
        if policy.get("selection_status") == "selected":
            resolve_policy_score_variant(policy, None)
    points, executions = report.get("points"), report.get("executions")
    if not isinstance(points, list) or not isinstance(executions, dict):
        raise ValueError("튜닝 규모 근거의 관측 지점 또는 실행 증거가 없다")
    count_field = "observed_row" if report["schema_version"] == 2 else "available_count"
    for point in points:
        if report["schema_version"] == 2 and any(field in point for field in (
            "available_count", "fit_count", "validation_count", "observed_prefix_count",
        )):
            raise ValueError("새 튜닝 규모 근거에 중복 행 수 필드가 있다")
        for field in (count_field, "feature_count", "training_boundary", "evaluation_rows"):
            if type(point.get(field)) is not int or point[field] < 1:
                raise ValueError("튜닝 규모 근거의 행·센서 수가 잘못됐다")
        ratio = point.get("ratio")
        if (type(ratio) is not int or ratio not in SUPPORTED_RATIO_PERCENTS
                or point[count_field] != point["training_boundary"] * ratio // 100):
            raise ValueError("튜닝 규모 근거의 q와 실제 prefix가 다르다")
        identities = point.get("execution_ids")
        if not identities or not point.get("environment_id") or any(
            identity not in executions
            or executions[identity].get("environment_id") != point["environment_id"]
            for identity in identities
        ):
            raise ValueError("튜닝 규모 근거의 실행 또는 환경 연결이 빠졌다")
        matches = [policy for kind, policy in policies
                   if kind == point.get("analysis_kind", "model_ratio") and all(
            policy.get(field, "") == point.get(field, "")
            for field in ("model", "group_id", "ratio", "config_id")
        ) and policy.get("selection_status") == "selected"
            and resolve_policy_score_variant(policy, point.get("family")) == point.get("score_variant", "")]
        if len(matches) != 1 or not any(all(
            support.get(field) == point[field]
            for field in (count_field, "feature_count", "training_boundary")
        ) and (matches[0].get("score_variant") != "family_selected" or all(
            isinstance(point.get(field), str) and bool(point[field].strip())
            and support.get(field) == point[field] for field in ("series", "family")
        )) for support in matches[0].get("support", [])):
            raise ValueError("튜닝 규모 근거가 선택된 모델·설정·입력 조합과 다르다")
    return report


def summarize_tuning_support(points):
    """관측한 N별 q 격자를 보존한다. 최대값 안의 빈 구간을 지원으로 채우지 않는다."""
    fields = ("model", "feature_count", "training_boundary", "evaluation_rows", "environment_id")
    grouped = {}
    for point in points:
        grouped.setdefault(tuple(point[field] for field in fields), []).append(point)
    summaries = []
    for key, members in sorted(grouped.items()):
        ratios = sorted({point["ratio"] for point in members})
        complete = [ratio for ratio in ratios
                    if all(required in ratios for required in SUPPORTED_RATIO_PERCENTS if required >= ratio)]
        summaries.append({
            **dict(zip(fields, key)), "observed_ratios": ratios,
            "missing_ratios": [ratio for ratio in SUPPORTED_RATIO_PERCENTS if ratio not in ratios],
            "complete_from_ratio": min(complete) if complete else None,
            "observed_min_rows": min(point.get("observed_row", point.get("available_count")) for point in members),
            "observed_max_rows": max(point.get("observed_row", point.get("available_count")) for point in members),
            "series_ids": sorted({point["series"] for point in members}),
            "observation_status": "complete", "service_status": "unvalidated",
        })
    for row in summaries:
        row["observed_max_planned_rows_at_same_columns"] = max(
            other["training_boundary"] for other in summaries
            if all(other[field] == row[field]
                   for field in ("model", "feature_count", "evaluation_rows", "environment_id"))
        )
    return summaries
