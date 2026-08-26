"""Dev18 정적 support에서 equal-trial exact panel을 만든다."""

import hashlib
import json
import re
from copy import deepcopy


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MUTABLE_SELECTION_FIELDS = {
    "primary_hpo_regime", "budget_id", "selection_status",
}
MUTABLE_MODEL_FIELDS = {"execution_status", "status_reason"}
TARGET_FREE_USES = {"training_free", "strict_zero_shot"}


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def registry_space_sha256(registry: dict) -> str:
    """budget seal과 readiness만 뺀 후보 공간의 안정 SHA를 반환한다."""
    stable = deepcopy(registry)
    selection = stable.get("selection")
    models = stable.get("models")
    if not isinstance(selection, dict) or not isinstance(models, dict):
        raise ValueError("registry selection 또는 models가 없다")
    for field in MUTABLE_SELECTION_FIELDS:
        selection.pop(field, None)
    for model in models.values():
        for field in MUTABLE_MODEL_FIELDS:
            model.pop(field, None)
    return hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if count < 1 or count > len(values):
        raise ValueError("equal-trial config 수가 후보 수를 벗어났다")
    if count == 1:
        return values[:1]
    return [values[index * (len(values) - 1) // (count - 1)] for index in range(count)]


def _score_variants(selection: dict, model_name: str) -> tuple[list[str], list[str]]:
    if model_name == "TSPulse":
        primary = selection.get("primary_score_variants", {}).get(model_name)
        diagnostic = selection.get("diagnostic_score_variants", {}).get(model_name)
        if primary != ["raw_max"] or diagnostic != ["time", "fft", "pred"]:
            raise ValueError("TSPulse primary·diagnostic score variant 봉인이 잘못됐다")
        return list(primary), list(diagnostic)
    return [""], []


def build_equal_trial_budget(registry: dict, feasibility_summary: dict) -> dict:
    """점수 없이 model별 config 수와 물리 실행 panel을 고정한다."""
    selection = registry.get("selection")
    models = registry.get("models")
    summary_models = feasibility_summary.get("models")
    if not isinstance(selection, dict) or not isinstance(models, dict):
        raise ValueError("registry selection 또는 models가 없다")
    if not isinstance(summary_models, dict) or tuple(summary_models) != tuple(models):
        raise ValueError("feasibility summary model 순서가 registry와 다르다")
    for field in (
        "input_manifest_sha256", "inventory_sha256",
        "data_preprocessing_sha256", "feasibility_decision_sha256",
    ):
        value = feasibility_summary.get(field)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"feasibility summary {field}가 잘못됐다")
    if selection.get("minimum_primary_ratio_count") != 3:
        raise ValueError("minimum primary ratio count는 3이어야 한다")
    development_seeds = registry.get("seeds", {}).get("development")
    if development_seeds != [0, 1, 2]:
        raise ValueError("development seed는 [0, 1, 2]여야 한다")

    tier_trial_counts = {}
    for tier in selection["tier_q_floor"]:
        eligible = [
            details for details in summary_models.values()
            if details["tier"] == tier
            and details["dev18_selection_support_status"] == "eligible_for_model_fixed_hpo"
        ]
        if not eligible:
            raise ValueError(f"{tier} equal-trial 대상 모델이 없다")
        tier_trial_counts[tier] = min(
            len(details["dev18_common_config_ids_across_supported_ratios"])
            for details in eligible
        )
        if tier_trial_counts[tier] < 1:
            raise ValueError(f"{tier} equal-trial config 수가 0이다")

    model_panels = []
    execution_panel = []
    logical_score_rows = 0
    for model_name, model in models.items():
        support = summary_models[model_name]
        support_status = support["dev18_selection_support_status"]
        logical_ratios = support["dev18_supported_ratios_at_or_above_q_floor"]
        common_ids = support["dev18_common_config_ids_across_supported_ratios"]
        if support_status == "eligible_for_model_fixed_hpo":
            selected_ids = _evenly_spaced(
                common_ids, tier_trial_counts[model["tier"]],
            )
            selection_kind = "equal_trial"
        elif support_status == "insufficient_ratio_support" and common_ids:
            selected_ids = common_ids[:1]
            selection_kind = "canonical_fixed_without_hpo"
        elif support_status == "unavailable":
            selected_ids = []
            selection_kind = "unavailable"
        else:
            raise ValueError(f"{model_name} selection support가 모순된다")

        primary_variants, diagnostic_variants = _score_variants(
            selection, model_name,
        )
        seeds = development_seeds[:1] if model["deterministic"] else development_seeds
        model_panels.append({
            "model": model_name,
            "tier": model["tier"],
            "target_use": model["target_use"],
            "selection_kind": selection_kind,
            "support_status": support_status,
            "logical_ratios": logical_ratios,
            "selected_config_ids": selected_ids,
            "dev18_tier_representative_eligible": support[
                "dev18_tier_representative_eligible"
            ],
            "primary_score_variants": primary_variants,
            "diagnostic_score_variants": diagnostic_variants,
            "seeds": list(seeds),
        })
        logical_score_rows += (
            len(selected_ids) * len(logical_ratios) * len(seeds)
            * len(primary_variants)
        )
        for config_id in selected_ids:
            if model["target_use"] in TARGET_FREE_USES:
                for seed in seeds:
                    execution_panel.append({
                        "model": model_name,
                        "tier": model["tier"],
                        "config_id": config_id,
                        "physical_ratio": 100,
                        "logical_ratios": list(logical_ratios),
                        "seed": seed,
                        "primary_score_variants": primary_variants,
                        "diagnostic_score_variants": diagnostic_variants,
                    })
            else:
                for ratio in logical_ratios:
                    for seed in seeds:
                        execution_panel.append({
                            "model": model_name,
                            "tier": model["tier"],
                            "config_id": config_id,
                            "physical_ratio": ratio,
                            "logical_ratios": [ratio],
                            "seed": seed,
                            "primary_score_variants": primary_variants,
                            "diagnostic_score_variants": diagnostic_variants,
                        })

    budget_core = {
        "schema_version": 1,
        "primary_hpo_regime": "equal_trial",
        "selection_rule_id": selection["selection_rule_id"],
        "registry_space_sha256": registry_space_sha256(registry),
        "input_manifest_sha256": feasibility_summary["input_manifest_sha256"],
        "inventory_sha256": feasibility_summary["inventory_sha256"],
        "data_preprocessing_sha256": feasibility_summary[
            "data_preprocessing_sha256"
        ],
        "feasibility_decision_sha256": feasibility_summary[
            "feasibility_decision_sha256"
        ],
        "tier_config_trial_count": tier_trial_counts,
        "seeds": {"deterministic": [0], "stochastic": list(development_seeds)},
        "model_panels": model_panels,
        "execution_panel": execution_panel,
        "physical_execution_count": len(execution_panel),
        "primary_logical_score_row_count": logical_score_rows,
        "fairness_rules": {
            "config_trial_unit": "one_full_fidelity_config_across_model_supported_ratios",
            "q_floor_change_after_scores": False,
            "score_variant_rule": "one_presealed_primary_variant_per_model",
            "insufficient_support_rule": "canonical_first_without_hpo_or_unavailable",
        },
        "failure_rules": {
            "maximum_transient_retries": 2,
            "maximum_total_attempts": 3,
            "allow_partial_seed_mean": False,
            "zero_fill_failed_trial": False,
            "replacement_config": False,
            "adapter_change": "invalidate_and_rerun_same_model_panel_with_new_code_sha",
        },
        "tie_rule": {
            "tolerance": 1e-6,
            "order": ["model", "config_id", "score_variant"],
        },
    }
    budget_sha = hashlib.sha256(_canonical_bytes(budget_core)).hexdigest()
    return {
        **budget_core,
        "budget_sha256": budget_sha,
        "budget_id": "b" + budget_sha[:12],
    }
