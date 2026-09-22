"""기존 Dev18 예산을 보존하며 비율별 전체 후보 예산을 만든다."""

import hashlib
from copy import deepcopy

from src.common.equal_trial_budget import (
    TARGET_FREE_USES,
    _canonical_bytes,
    _score_variants,
    registry_space_sha256,
)
from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS


def build_full_prefix_budget(registry, feasibility_rows, series_entries, evidence) -> dict:
    """모든 파일에서 가능한 후보를 실행하고 같은 후보를 비교할 집단을 고정한다."""
    entries = list(series_entries)
    series_ids = [str(entry["series"]).zfill(2) for entry in entries]
    if series_ids != [f"{index:02d}" for index in range(1, 19)]:
        raise ValueError("Dev18 원본 18개 파일과 순서를 유지해야 한다")
    entry_by_series = dict(zip(series_ids, entries))
    models = registry["models"]
    development_seeds = registry.get("seeds", {}).get("development")
    if development_seeds != [0, 1, 2]:
        raise ValueError("development seed는 [0, 1, 2]여야 한다")
    ratios = list(SUPPORTED_RATIO_PERCENTS)
    expected = {(name, candidate["config_id"], ratio, series)
                for name, model in models.items() for candidate in model["candidates"]
                for ratio in ratios for series in series_ids}
    decisions = {}
    support_by_key = {}
    excluded = []
    for row in feasibility_rows:
        series = str(row["series"]).zfill(2)
        key = (row["model"], row["config_id"], int(row["logical_ratio"]), series)
        if key not in expected or key in decisions or row["status"] not in {
            "feasible", "structurally_infeasible",
        }:
            raise ValueError("feasibility 원표에 예상하지 않은 키·중복·상태가 있다")
        decisions[key] = row["status"] == "feasible"
        entry = entry_by_series[series]
        support = {"series": series, "family": entry["family"], "name": entry["name"]}
        for field in ("training_boundary", "feature_count"):
            value = int(row[field])
            if value < 1 or value != entry[field]:
                raise ValueError("feasibility의 행 수 또는 센서 수가 잘못됐다")
            support[field] = value
        observed_row = int(row["available_count"])
        if observed_row != support["training_boundary"] * key[2] // 100:
            raise ValueError("feasibility의 관측 행 수가 지정 q-prefix와 다르다")
        if models[row["model"]]["target_use"] == "fit_full_prefix" and (
            int(row["fit_count"]) != observed_row or int(row["validation_count"]) != 0
        ):
            raise ValueError("새 학습형 후보는 현재 prefix 전부를 fit에 써야 한다")
        support["observed_row"] = observed_row
        support_key = (row["model"], int(row["logical_ratio"]), series)
        if support_key in support_by_key and support_by_key[support_key] != support:
            raise ValueError("같은 모델·비율·파일의 입력 크기가 후보마다 다르다")
        support_by_key[support_key] = support
        if not decisions[key]:
            excluded.append({**support, "model": row["model"], "config_id": row["config_id"],
                             "ratio": key[2], "status_reason": row["status_reason"]})
    if decisions.keys() != expected:
        raise ValueError("feasibility 원표가 등록 후보·비율·18개 파일을 모두 포함하지 않는다")

    panels, executions = [], {}
    for name, model in models.items():
        candidate_ids = [candidate["config_id"] for candidate in model["candidates"]]
        primary, diagnostic = _score_variants(registry["selection"], name)
        variants_by_config = {
            candidate["config_id"]: _score_variants(registry["selection"], name, candidate["hyperparameters"])
            for candidate in model["candidates"]
        }
        seeds = development_seeds[:1] if model["deterministic"] else development_seeds
        by_ratio, groups = {}, []
        for ratio in ratios:
            by_ratio[str(ratio)] = [config for config in candidate_ids
                                   if any(decisions[name, config, ratio, series]
                                          for series in series_ids)]
            grouped = {}
            for series in series_ids:
                signature = tuple(config for config in candidate_ids
                                  if decisions[name, config, ratio, series])
                grouped.setdefault(signature, []).append(series)
            for signature, members in grouped.items():
                group = {"model": name, "tier": model["tier"], "ratio": ratio,
                         "series_ids": members, "candidate_ids": list(signature)}
                group["group_id"] = "g" + hashlib.sha256(_canonical_bytes(group)).hexdigest()[:12]
                group["support"] = [support_by_key[name, ratio, series] for series in members]
                for field in ("observed_row", "training_boundary", "feature_count"):
                    values = [support[field] for support in group["support"]]
                    group[field + "_min"], group[field + "_max"] = min(values), max(values)
                groups.append(group)
        for candidate in model["candidates"]:
            config = candidate["config_id"]
            candidate_primary, candidate_diagnostic = variants_by_config[config]
            for ratio in ratios:
                members = [series for series in series_ids if decisions[name, config, ratio, series]]
                if not members:
                    continue
                physical_ratio = 100 if model["target_use"] in TARGET_FREE_USES else ratio
                for seed in seeds:
                    key = (name, config, physical_ratio, seed)
                    execution = executions.setdefault(key, {
                        "model": name, "tier": model["tier"], "config_id": config,
                        "physical_ratio": physical_ratio, "logical_ratios": [], "seed": seed,
                        "series_ids": members,
                        "primary_score_variants": candidate_primary,
                        "diagnostic_score_variants": candidate_diagnostic,
                    })
                    if execution["series_ids"] != members:
                        raise ValueError("target-free 설정의 실행 가능 파일이 비율에 따라 달라졌다")
                    execution["logical_ratios"].append(ratio)
        panels.append({
            "model": name, "tier": model["tier"], "target_use": model["target_use"],
            "selection_kind": "conditional_per_ratio_full_grid", "groups": groups,
            "selected_config_ids": [config for config in candidate_ids
                                    if any(config in values for values in by_ratio.values())],
            "selected_config_ids_by_ratio": by_ratio,
            "logical_ratios": [ratio for ratio in ratios if by_ratio[str(ratio)]],
            "seeds": seeds, "primary_score_variants": primary,
            "diagnostic_score_variants": diagnostic,
            "primary_score_variants_by_config": {config: variants[0] for config, variants in variants_by_config.items()},
            "dev18_tier_representative_eligible": False,
        })
    execution_panel = list(executions.values())
    physical_runs = sum(len(row["series_ids"]) for row in execution_panel)
    logical_rows = sum(len(row["series_ids"]) * len(row["logical_ratios"])
                       * len(row["primary_score_variants"]) for row in execution_panel)
    training_keys = {(row["model"], row["config_id"], row["physical_ratio"], row["seed"], series)
                     for row in execution_panel for series in row["series_ids"]
                     if models[row["model"]]["target_use"] not in TARGET_FREE_USES}
    core = {
        "schema_version": 3, "experiment_mode": "full_prefix_v2",
        "primary_hpo_regime": registry["selection"].get("primary_hpo_regime", "full_prefix_per_ratio"),
        "selection_rule_id": registry["selection"].get("selection_rule_id", "full_prefix_conditional_family_lofo_v2"),
        "registry_space_sha256": registry_space_sha256(registry),
        **{field: evidence[field] for field in (
            "input_manifest_sha256", "inventory_sha256", "data_preprocessing_sha256",
            "feasibility_decision_sha256",
        )},
        "series_ids": series_ids, "model_panels": panels, "execution_panel": execution_panel,
        "structural_exclusions": excluded, "physical_execution_count": len(execution_panel),
        "physical_run_count": physical_runs, "training_run_count": len(training_keys),
        "expected_ledger_rows": logical_rows, "primary_logical_score_row_count": logical_rows,
        "count_scope": "actual_series_execution_pairs",
        "seeds": {"deterministic": development_seeds[:1], "stochastic": list(development_seeds)},
        "fairness_rules": {"config_trial_unit": "one_feasible_config_series_ratio",
                           "insufficient_support_rule": "conditional_common_candidate_panel",
                           "score_variant_rule": "all_presealed_primary_variants_share_inference"},
        "failure_rules": {"maximum_transient_retries": 2, "maximum_total_attempts": 3,
                          "allow_partial_seed_mean": False, "zero_fill_failed_trial": False,
                          "replacement_config": False},
        "tie_rule": {"tolerance": 1e-6, "order": ["model", "config_id", "score_variant"]},
    }
    digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    return {**core, "budget_sha256": digest, "budget_id": "b" + digest[:12]}


def build_ratio_tuning_budget(
    registry, feasibility_summary, legacy_budget,
) -> dict:
    """각 q의 공통 후보를 실행하고 target-free 결과는 물리 실행 하나로 접는다."""
    models = registry["models"]
    summaries = feasibility_summary["models"]
    ratios = list(SUPPORTED_RATIO_PERCENTS)
    series_ids = [f"{series:02d}" for series in range(1, 19)]
    if feasibility_summary.get("series_count") != 18 or (
        tuple(summaries) != tuple(models)
        or feasibility_summary.get("ratio_order") != ratios
        or legacy_budget.get("primary_hpo_regime") != "equal_trial"
        or [panel["model"] for panel in legacy_budget["model_panels"]] != list(models)
    ):
        raise ValueError("Dev18 registry·feasibility·기존 예산의 범위가 다르다")
    space_sha256 = registry_space_sha256(registry)
    if space_sha256 != legacy_budget["registry_space_sha256"] or any(
        feasibility_summary[field] != legacy_budget[field]
        for field in (
            "input_manifest_sha256", "inventory_sha256", "data_preprocessing_sha256",
            "feasibility_decision_sha256",
        )
    ):
        raise ValueError("기존 예산과 registry 또는 feasibility 입력 신원이 다르다")
    model_panels = deepcopy(legacy_budget["model_panels"])
    execution_by_key = {}
    for panel in model_panels:
        model_name = panel["model"]
        model = models[model_name]
        candidate_ids = [candidate["config_id"] for candidate in model["candidates"]]
        floor = registry["selection"]["tier_q_floor"][model["tier"]]
        by_ratio = {}
        for ratio in ratios:
            if ratio < floor:
                selected = []
            elif model["tier"] == "t1":
                selected = (
                    panel["selected_config_ids"] if ratio in panel["logical_ratios"] else []
                )
            else:
                selected = summaries[model_name]["fully_feasible_config_ids_by_ratio"][str(ratio)]
            if len(set(selected)) != len(selected) or not set(selected) <= set(candidate_ids):
                raise ValueError(f"{model_name} q{ratio} 후보가 registry와 다르다")
            by_ratio[str(ratio)] = list(selected)
        selected_union = {
            config_id for selected in by_ratio.values() for config_id in selected
        }
        panel.update(
            selected_config_ids_by_ratio=by_ratio,
            selected_config_ids=[
                config_id for config_id in candidate_ids if config_id in selected_union
            ],
            logical_ratios=[ratio for ratio in ratios if by_ratio[str(ratio)]],
        )
        if not selected_union:
            panel.update(selection_kind="unavailable", support_status="unavailable",
                         dev18_tier_representative_eligible=False)
        elif model["tier"] != "t1":
            panel.update(selection_kind="per_ratio_full_grid",
                         support_status="eligible_for_ratio_hpo")
        for config_id in panel["selected_config_ids"]:
            for ratio in panel["logical_ratios"]:
                if config_id not in by_ratio[str(ratio)]:
                    continue
                physical_ratio = 100 if model["target_use"] in TARGET_FREE_USES else ratio
                for seed in panel["seeds"]:
                    key = (model_name, config_id, physical_ratio, seed)
                    row = execution_by_key.setdefault(key, {
                        "model": model_name, "tier": model["tier"], "config_id": config_id,
                        "physical_ratio": physical_ratio, "logical_ratios": [], "seed": seed,
                        "primary_score_variants": list(panel["primary_score_variants"]),
                        "diagnostic_score_variants": list(panel["diagnostic_score_variants"]),
                    })
                    row["logical_ratios"].append(ratio)

    core = deepcopy(legacy_budget)
    for field in (
        "budget_id", "budget_sha256", "tier_config_trial_count", "seal_status",
        "execution_readiness_status", "pending_execution_models", "attestation",
    ):
        core.pop(field, None)
    execution_panel = list(execution_by_key.values())
    core.update(
        primary_hpo_regime="per_ratio_full_grid",
        selection_rule_id="tier_adaptive_config_family_lofo_v2",
        registry_space_sha256=space_sha256,
        feasibility_decision_sha256=feasibility_summary["feasibility_decision_sha256"],
        legacy_budget=deepcopy(legacy_budget),
        baseline_budget_id=legacy_budget["budget_id"],
        baseline_budget_sha256=legacy_budget["budget_sha256"],
        series_ids=series_ids, model_panels=model_panels,
        execution_panel=execution_panel, physical_execution_count=len(execution_panel),
        primary_logical_score_row_count=sum(
            len(row["logical_ratios"]) * len(row["primary_score_variants"])
            for row in execution_panel
        ),
    )
    core["fairness_rules"].update(
        config_trial_unit="one_full_fidelity_config_at_one_supported_ratio",
        insufficient_support_rule="all_feasible_configs_on_the_fixed_series_panel",
    )
    digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    return {**core, "budget_sha256": digest, "budget_id": "b" + digest[:12]}
