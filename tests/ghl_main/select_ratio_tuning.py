"""기존 통제표를 보존하고 각 비율에서 recipe와 Tier 대표를 고른다."""

import hashlib
import math
from itertools import combinations
from statistics import mean

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS


def select_tspulse_official_heads(seed_rows, registry, *, budget_id, evaluator_sha256,
                                  evaluation_families=("GHL", "HAI"), tolerance=1e-6):
    """q별 파일 평균으로 공통 창을 고른 뒤 공식 family별 head 선택을 적용한다."""
    model = registry["models"].get("TSPulse")
    if model is None:
        return []
    configs = {candidate["hyperparameters"]["aggregation_window"]: candidate["config_id"]
               for candidate in model["candidates"]}
    if set(configs) != {64, 96, 128} or len(model["candidates"]) != 3 or len(set(configs.values())) != 3:
        raise ValueError("TSPulse 공통 창 선택에는 64·96·128 설정이 각각 하나씩 필요하다")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("TSPulse 창 선택의 동률 허용 오차가 잘못됐다")
    heads = ("time", "fft", "pred", "ensemble")
    source_modes = {"time": "time", "fft": "fft", "pred": "forecast", "ensemble": "time+fft+forecast"}
    grouped, seen, file_families = {}, set(), {}
    for row in seed_rows:
        if row["model"] != "TSPulse":
            continue
        ratio, config = int(row["ratio"]), row["config_id"]
        head, family, series = row["score_variant"], row["family"], row["series"]
        value = float(row["vus_pr"])
        key = (ratio, config, series, head)
        if (ratio not in SUPPORTED_RATIO_PERCENTS or config not in configs.values()
                or head not in heads or key in seen or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("공식 TSPulse head 선택 원표의 점수 또는 고유 키가 잘못됐다")
        seen.add(key)
        if file_families.setdefault((ratio, series), family) != family:
            raise ValueError("TSPulse 파일의 family가 설정별로 다르다")
        config_rows = grouped.setdefault(ratio, {}).setdefault(config, {head: {} for head in heads})
        config_rows[head][series] = value

    result = []
    for ratio, ratio_rows in sorted(grouped.items()):
        series_ids = sorted(series for current_ratio, series in file_families if current_ratio == ratio)
        if (set(ratio_rows) != set(configs.values()) or any(
                set(values) != set(series_ids) for config_rows in ratio_rows.values()
                for values in config_rows.values())):
            raise ValueError("TSPulse 창·head별 파일 구성이 다르다")
        window_scores = {str(window): mean(
            (ratio_rows[config]["time"][series] + ratio_rows[config]["fft"][series]) / 2
            for series in series_ids) for window, config in sorted(configs.items())}
        best = max(window_scores.values())
        window = min(window for window in configs if best - window_scores[str(window)] <= tolerance)
        config = configs[window]
        families = sorted({file_families[ratio, series] for series in series_ids})
        for family in sorted(set(families) | set(evaluation_families)):
            members = [series for series in series_ids if file_families[ratio, series] == family]
            means = {head: (mean(float(f"{ratio_rows[config][head][series]:.5f}") for series in members)
                            if members else None) for head in heads}
            selected = max(heads, key=means.get) if members else "time"
            result.append({
                "analysis_kind": "tspulse_official_family_head", "family": family, "ratio": ratio,
                "model": "TSPulse", "config_id": config, "aggregation_window": window,
                "score_variant": selected, "official_mode": source_modes[selected],
                "family_mean_vus_pr": means[selected],
                **{head + "_vus_pr": means[head] for head in heads},
                "series_ids": members, "tuning_file_count": len(members),
                "selection_status": "selected" if members else "unseen_family_time_fallback",
                "selection_scope": "common_window_then_family_file_mean_head",
                "window_rule_id": "tspulse_file_mean_time_fft_v1",
                "window_rule_source": "user_approved_project_supplement",
                "window_scores": window_scores, "window_series_ids": series_ids, "window_families": families,
                "window_metric_decimal_places": None,
                "window_tie_rule": "within_tolerance_then_smallest_window", "window_tolerance": tolerance,
                "tuning_logical_ratio": ratio, "metric_decimal_places": 5,
                "head_order": list(heads), "tie_rule": "first_in_readme_mode_order",
                "unseen_family_fallback": "time", "budget_id": budget_id,
                "evaluator_sha256": evaluator_sha256, "source_commit": model["source_commit"],
            })
    return result


def _validate_full_prefix_trials(rows, budget):
    expected = {
        (execution["model"], execution["config_id"], ratio, seed, series, variant)
        for execution in budget["execution_panel"] for ratio in execution["logical_ratios"]
        for seed in [execution["seed"]] for series in execution["series_ids"]
        for variant in execution["primary_score_variants"]
    }
    families = {support["series"]: support["family"] for panel in budget["model_panels"]
                for group in panel["groups"] for support in group["support"]}
    tiers = {panel["model"]: panel["tier"] for panel in budget["model_panels"]}
    seen, normalized = set(), []
    for row in rows:
        row = {**row, "ratio": int(row["ratio"]), "seed": int(row["seed"]),
               "series": str(row["series"]).zfill(2), "vus_pr": float(row["vus_pr"])}
        key = tuple(row[field] for field in ("model", "config_id", "ratio", "seed", "series", "score_variant"))
        if key not in expected or key in seen or row["status"] != "complete":
            raise ValueError("조건별 선택에는 중복 없는 전체 완료 trial 원표가 필요하다")
        if (row["family"] != families[row["series"]] or row["tier"] != tiers[row["model"]]
                or not math.isfinite(row["vus_pr"])
                or not 0 <= row["vus_pr"] <= 1):
            raise ValueError("trial의 family 또는 VUS-PR이 잘못됐다")
        seen.add(key)
        normalized.append(row)
    if seen != expected:
        raise ValueError("조건별 선택 원표에서 파일·설정·비율·seed 점수가 빠졌다")
    return normalized


def _select_full_prefix_policies(rows, registry, budget, evaluator_sha256):
    from tests.ghl_main import run_dev18_tuning as tuning

    rows = _validate_full_prefix_trials(rows, budget)
    seed_rows = tuning._seed_means(rows)
    tolerance = float(budget["tie_rule"]["tolerance"])
    selection = {key: [] for key in (
        "model_fixed", "tier_fixed", "lofo", "model_ratio", "tier_adaptive",
        "adaptive_lofo", "candidate_audit", "policy_transitions", "model_comparison",
    )}
    if registry.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4":
        selection["tspulse_official_heads"] = select_tspulse_official_heads(
            seed_rows, registry, budget_id=budget["budget_id"], evaluator_sha256=evaluator_sha256,
            tolerance=tolerance,
        )
    panels = {panel["model"]: panel for panel in budget["model_panels"]}
    tspulse_by_ratio, tspulse_folds = {}, {}
    for row in selection.get("tspulse_official_heads", []):
        tspulse_by_ratio.setdefault(row["ratio"], []).append(row)

    def select_tspulse_rows(group_rows, ratio, families=None):
        if families is None:
            return tspulse_by_ratio[ratio]
        training = [row for row in group_rows if row["model"] == "TSPulse"
                    and row["ratio"] == ratio and row["family"] in families]
        key = (ratio, tuple(sorted({row["series"] for row in training})))
        if key not in tspulse_folds:
            tspulse_folds[key] = select_tspulse_official_heads(
                training, registry, budget_id=budget["budget_id"], evaluator_sha256=evaluator_sha256,
                evaluation_families=(), tolerance=tolerance,
            )
        if not tspulse_folds[key]:
            raise ValueError("TSPulse fold의 학습 파일이 없다")
        return tspulse_folds[key]

    def head_fields(group_rows, model, variant, ratio, families=None):
        fields = {"score_variant_by_family": {}, "score_variant_fallback": ""}
        if model == "TSPulse" and variant == "family_selected":
            fields.update(score_variant_by_family={row["family"]: row["score_variant"]
                          for row in select_tspulse_rows(group_rows, ratio, families)},
                          score_variant_fallback="time")
        return fields

    def scoring_rows(group_rows, model, config, variant, ratio, training_families=None):
        fields = head_fields(group_rows, model, variant, ratio, training_families)
        return [{**row, "score_variant": variant} for row in group_rows
                if row["model"] == model and row["config_id"] == config and row["ratio"] == ratio
                and row["score_variant"] == (fields["score_variant_by_family"].get(
                    row["family"], fields["score_variant_fallback"])
                    if variant == "family_selected" else variant)]

    def candidate_score(group_rows, *, model, config_id, score_variant, ratio,
                        included_families=None, training_families=None):
        return tuning._candidate_score(
            scoring_rows(group_rows, model, config_id, score_variant, ratio, training_families),
            model=model, config_id=config_id, score_variant=score_variant, ratios=[ratio],
            included_families=included_families,
        )

    def native_candidates(group_rows, panel, config_ids, ratio, families=None):
        return [(
            tuning._candidate_score(group_rows, model=panel["model"], config_id=config,
                                    score_variant=variant, ratios=[ratio],
                                    included_families=families),
            panel["model"], config, variant,
        ) for config in config_ids for variant in panel.get("primary_score_variants_by_config", {}).get(
            config, panel["primary_score_variants"],
        )]

    def candidates(group_rows, panel, config_ids, ratio, families=None):
        if panel["model"] != "TSPulse" or "tspulse_official_heads" not in selection:
            return native_candidates(group_rows, panel, config_ids, ratio, families)
        config = select_tspulse_rows(group_rows, ratio, families)[0]["config_id"]
        if config not in config_ids:
            raise ValueError("TSPulse 공통 창이 조건집단의 실행 가능 후보에 없다")
        return [(candidate_score(
            group_rows, model="TSPulse", config_id=config, score_variant="family_selected", ratio=ratio,
            included_families=families, training_families=families,
        ), "TSPulse", config, "family_selected")]

    def native_tspulse_audit(group_rows, panel, config_ids, ratio):
        return (native_candidates(group_rows, panel, config_ids, ratio)
                if panel["model"] == "TSPulse" and "tspulse_official_heads" in selection else [])

    for panel in panels.values():
        model = registry["models"][panel["model"]]
        for group in panel["groups"]:
            ratio = group["ratio"]
            group_rows = [row for row in seed_rows if row["model"] == panel["model"]
                          and row["ratio"] == ratio and row["series"] in group["series_ids"]]
            families = sorted({support["family"] for support in group["support"]})
            policy = {
                **group, "config_id": "", "hyperparameters": {}, "score_variant": "",
                "score_variant_by_family": {}, "score_variant_fallback": "",
                "analysis_kind": "model_ratio",
                "family_macro_vus_pr": None, "family_lofo_vus_pr": None, "family_count": len(families),
                "selection_status": "unavailable", "selection_reason": "실행 가능한 등록 후보가 없음",
                "validation_status": "insufficient_families" if len(families) < 2 else "family_lofo",
                "hpo_regime": budget["primary_hpo_regime"], "budget_id": budget["budget_id"],
                "evaluator_sha256": evaluator_sha256, "source_commit": model["source_commit"],
                "source_checkpoint_sha256": model["source_checkpoint_sha256"],
            }
            if group["candidate_ids"]:
                scores = candidates(group_rows, panel, group["candidate_ids"], ratio)
                score, _, config, variant = tuning._pick(scores, tolerance)
                policy.update(config_id=config, hyperparameters=tuning._hyperparameters(
                    registry, panel["model"], config,
                ), score_variant=variant, family_macro_vus_pr=score, selection_status="selected",
                              selection_reason="동일 조건 집단의 family 동일 가중 VUS-PR로 선택")
                policy.update(head_fields(group_rows, panel["model"], variant, ratio))
                for value, _, candidate_config, candidate_variant in (
                        scores + native_tspulse_audit(group_rows, panel, group["candidate_ids"], ratio)):
                    candidate_rows = scoring_rows(group_rows, panel["model"], candidate_config, candidate_variant, ratio)
                    selection["candidate_audit"].append({
                        "analysis_kind": "model_ratio", "tier": panel["tier"],
                        "group_id": group["group_id"], "model": panel["model"], "ratio": ratio,
                        "series_ids": group["series_ids"], "candidate_ids": group["candidate_ids"],
                        "config_id": candidate_config, "score_variant": candidate_variant,
                        "family_macro_vus_pr": value,
                        **head_fields(group_rows, panel["model"], candidate_variant, ratio),
                        "series_macro_vus_pr": mean(row["vus_pr"] for row in candidate_rows),
                        "file_count": len(candidate_rows),
                        "family_count": len({row["family"] for row in candidate_rows}),
                        "selected": (candidate_config, candidate_variant) == (config, variant),
                    })
                if len(families) >= 2:
                    for holdout in families:
                        train_families = [family for family in families if family != holdout]
                        training_score, _, selected_config, selected_variant = tuning._pick(
                            candidates(group_rows, panel, group["candidate_ids"], ratio, train_families),
                            tolerance,
                        )
                        holdout_score = candidate_score(
                            group_rows, model=panel["model"], config_id=selected_config,
                            score_variant=selected_variant, ratio=ratio, included_families=[holdout],
                            training_families=train_families,
                        )
                        selection["adaptive_lofo"].append({
                            "analysis_kind": "model_ratio",
                            "group_id": group["group_id"], "model": panel["model"],
                            "tier": panel["tier"], "ratio": ratio, "series_ids": group["series_ids"],
                            "holdout_family": holdout, "training_families": train_families,
                            "selected_config_id": selected_config, "score_variant": selected_variant,
                            **head_fields(group_rows, panel["model"], selected_variant, ratio, train_families),
                            "training_vus_pr": training_score, "holdout_vus_pr": holdout_score,
                        })
            folds = [row["holdout_vus_pr"] for row in selection["adaptive_lofo"]
                     if row["group_id"] == group["group_id"]]
            if folds:
                policy["family_lofo_vus_pr"] = mean(folds)
            selection["model_ratio"].append(policy)

    from src.common.equal_trial_budget import _canonical_bytes

    for tier in sorted({panel["tier"] for panel in panels.values()}):
        tier_panels = [panel for panel in panels.values() if panel["tier"] == tier]
        for ratio in SUPPORTED_RATIO_PERCENTS:
            groups_by_series = {
                (panel["model"], series): group for panel in tier_panels
                for group in panel["groups"] if group["ratio"] == ratio
                for series in group["series_ids"]
            }
            members_by_signature = {}
            for series in budget["series_ids"]:
                signature = tuple((panel["model"], tuple(groups_by_series[panel["model"], series]["candidate_ids"]))
                                  for panel in tier_panels
                                  if groups_by_series[panel["model"], series]["candidate_ids"])
                members_by_signature.setdefault(signature, []).append(series)
            for signature, members in members_by_signature.items():
                candidate_ids_by_model = {model: list(configs) for model, configs in signature}
                group_identity = {"tier": tier, "ratio": ratio, "series_ids": members,
                                  "candidate_ids_by_model": candidate_ids_by_model}
                group_id = "g" + hashlib.sha256(_canonical_bytes(group_identity)).hexdigest()[:12]
                shared = [row for row in seed_rows if row["tier"] == tier
                          and row["ratio"] == ratio and row["series"] in members]

                def joint_candidates(included_families=None):
                    return [candidate for model, configs in signature for candidate in candidates(
                        shared, panels[model], configs, ratio, included_families,
                    )]

                policy = {**group_identity, "group_id": group_id, "analysis_kind": "tier_adaptive",
                          "selected_model": "", "model": "", "config_id": "", "candidate_ids": [],
                          "hyperparameters": {}, "score_variant": "", "family_macro_vus_pr": None,
                          "score_variant_by_family": {}, "score_variant_fallback": "",
                          "family_lofo_vus_pr": None,
                          "selection_status": "unavailable", "selection_reason": "실행 가능한 등록 후보가 없음",
                          "hpo_regime": budget["primary_hpo_regime"], "budget_id": budget["budget_id"],
                          "evaluator_sha256": evaluator_sha256}
                scores = joint_candidates()
                selected_model = tier_panels[0]["model"]
                if scores:
                    score, selected_model, config, variant = tuning._pick(scores, tolerance)
                    policy.update(selected_model=selected_model, model=selected_model, config_id=config,
                                  candidate_ids=candidate_ids_by_model[selected_model], score_variant=variant,
                                  hyperparameters=tuning._hyperparameters(registry, selected_model, config),
                                   family_macro_vus_pr=score, selection_status="selected",
                                   selection_reason="같은 파일과 후보 조건에서 모델·설정을 공동 선택")
                    policy.update(head_fields(shared, selected_model, variant, ratio))
                    audit = scores + [candidate for model, configs in signature
                                      for candidate in native_tspulse_audit(shared, panels[model], configs, ratio)]
                    for value, model, candidate_config, candidate_variant in audit:
                        candidate_rows = scoring_rows(shared, model, candidate_config, candidate_variant, ratio)
                        selection["candidate_audit"].append({
                            "analysis_kind": "tier_adaptive", "tier": tier, "group_id": group_id,
                            "model": model, "ratio": ratio, "series_ids": members,
                            "candidate_ids": candidate_ids_by_model[model], "config_id": candidate_config,
                            "score_variant": candidate_variant, "family_macro_vus_pr": value,
                            **head_fields(shared, model, candidate_variant, ratio),
                            "series_macro_vus_pr": mean(row["vus_pr"] for row in candidate_rows),
                            "file_count": len(candidate_rows),
                            "family_count": len({row["family"] for row in candidate_rows}),
                            "selected": (model, candidate_config, candidate_variant) == (selected_model, config, variant),
                        })
                support = [next(item for item in groups_by_series[selected_model, series]["support"]
                                if item["series"] == series) for series in members]
                families = sorted({item["family"] for item in support})
                policy.update(support=support, family_count=len(families),
                              validation_status="family_lofo" if scores and len(families) >= 2
                              else "insufficient_families")
                count_fields = (("observed_row",) if budget.get("schema_version", 2) >= 3
                                else ("available_count", "fit_count"))
                for field in (*count_fields, "training_boundary", "feature_count"):
                    values = [item[field] for item in support]
                    policy[field + "_min"], policy[field + "_max"] = min(values), max(values)
                if scores and len(families) >= 2:
                    for holdout in families:
                        training_families = [family for family in families if family != holdout]
                        training_score, model, config, variant = tuning._pick(joint_candidates(training_families), tolerance)
                        selection["adaptive_lofo"].append({
                            "analysis_kind": "tier_adaptive", "group_id": group_id,
                            "model": model, "tier": tier, "ratio": ratio, "series_ids": members,
                            "holdout_family": holdout, "training_families": training_families,
                            "selected_config_id": config, "score_variant": variant,
                            **head_fields(shared, model, variant, ratio, training_families),
                            "training_vus_pr": training_score,
                            "holdout_vus_pr": candidate_score(
                                shared, model=model, config_id=config, score_variant=variant,
                                ratio=ratio, included_families=[holdout], training_families=training_families,
                            ),
                        })
                folds = [row["holdout_vus_pr"] for row in selection["adaptive_lofo"]
                         if row["group_id"] == group_id]
                if folds:
                    policy["family_lofo_vus_pr"] = mean(folds)
                selection["tier_adaptive"].append(policy)

    # 서로 다른 지원 패널의 평균을 비교하지 않고 겹치는 파일에서 LOFO를 다시 계산한다.
    for left_panel, right_panel in combinations(panels.values(), 2):
        for left in left_panel["groups"]:
            if not left["candidate_ids"]:
                continue
            for right in right_panel["groups"]:
                if left["ratio"] != right["ratio"] or not right["candidate_ids"]:
                    continue
                members = sorted(set(left["series_ids"]) & set(right["series_ids"]))
                if not members:
                    continue
                ratio = left["ratio"]
                shared = [row for row in seed_rows if row["ratio"] == ratio
                          and row["series"] in members
                          and row["model"] in {left["model"], right["model"]}]
                families = sorted({row["family"] for row in shared})
                comparison = {
                    "ratio": ratio, "left_model": left["model"], "right_model": right["model"],
                    "left_group_id": left["group_id"], "right_group_id": right["group_id"],
                    "left_series_ids": members, "right_series_ids": members,
                    "family_count": len(families), "left_lofo_vus_pr": None,
                    "right_lofo_vus_pr": None, "difference": None,
                    "status": "insufficient_families", "folds": [],
                }
                if len(families) >= 2:
                    for holdout in families:
                        fold = {"holdout_family": holdout}
                        training_families = [family for family in families if family != holdout]
                        for side, group, panel in (("left", left, left_panel), ("right", right, right_panel)):
                            _, _, config, variant = tuning._pick(candidates(
                                shared, panel, group["candidate_ids"], ratio,
                                training_families,
                            ), tolerance)
                            fold[side + "_config_id"] = config
                            fold[side + "_score_variant"] = variant
                            fold.update({side + "_" + field: value for field, value in head_fields(
                                shared, group["model"], variant, ratio, training_families,
                            ).items()})
                            fold[side + "_vus_pr"] = candidate_score(
                                shared, model=group["model"], config_id=config, score_variant=variant,
                                ratio=ratio, included_families=[holdout], training_families=training_families,
                            )
                        comparison["folds"].append(fold)
                    for side in ("left", "right"):
                        comparison[side + "_lofo_vus_pr"] = mean(
                            fold[side + "_vus_pr"] for fold in comparison["folds"]
                        )
                    comparison["difference"] = comparison["left_lofo_vus_pr"] - comparison["right_lofo_vus_pr"]
                    comparison["status"] = "matched_family_lofo"
                selection["model_comparison"].append(comparison)
    return selection


def select_ratio_tuning_policies(
    rows, registry: dict, budget: dict, evaluator_sha256: str,
) -> dict:
    """검증된 원표와 비율별 후보 예산으로 family-LOFO 정책을 만든다."""
    if budget.get("experiment_mode") == "full_prefix_v2":
        return _select_full_prefix_policies(rows, registry, budget, evaluator_sha256)
    from tests.ghl_main import run_dev18_tuning as tuning

    rows = list(rows)
    panels = budget["model_panels"]
    supported_models = {panel["model"] for panel in panels if panel["selected_config_ids"]}
    legacy_budget = budget["legacy_budget"]
    legacy_keys = {
        (panel["model"], config_id, ratio, seed, variant)
        for panel in legacy_budget["model_panels"]
        for config_id in panel["selected_config_ids"]
        for ratio in panel["logical_ratios"]
        for seed in panel["seeds"]
        for variant in panel["primary_score_variants"]
    }
    legacy_rows = [row for row in rows if (
        row["model"], row["config_id"], row["ratio"], row["seed"], row["score_variant"],
    ) in legacy_keys]
    selection = tuning.select_tuning_policies(
        legacy_rows, registry, legacy_budget, evaluator_sha256=evaluator_sha256,
    )

    seed_rows = tuning._seed_means(rows)
    tolerance = float(budget["tie_rule"]["tolerance"])
    selection.update(model_ratio=[], tier_adaptive=[], adaptive_lofo=[],
                     candidate_audit=[], policy_transitions=[])
    for ratio in SUPPORTED_RATIO_PERCENTS:
        ratio_panels = []
        ratio_policies = []
        for panel in panels:
            model_name = panel["model"]
            model = registry["models"][model_name]
            config_ids = panel["selected_config_ids_by_ratio"].get(str(ratio), [])
            ratio_panels.append({
                **panel, "selected_config_ids": config_ids,
                "logical_ratios": [ratio] if config_ids else [],
            })
            policy = {
                "tier": panel["tier"], "model": model_name, "ratio": ratio,
                "q_support": [ratio] if config_ids else [],
                "config_id": "", "hyperparameters": {}, "score_variant": "",
                "family_macro_vus_pr": None, "selection_status": "unavailable",
                "selection_reason": "이 비율에서 전체 비교 패널을 덮는 실행 후보가 없음",
                "hpo_regime": budget["primary_hpo_regime"], "budget_id": budget["budget_id"],
                "evaluator_sha256": evaluator_sha256,
                "source_commit": model["source_commit"],
                "source_checkpoint_sha256": model["source_checkpoint_sha256"],
            }
            if config_ids:
                candidates = [(
                    tuning._candidate_score(
                        seed_rows, model=model_name, config_id=config_id,
                        score_variant=variant, ratios=[ratio],
                    ), model_name, config_id, variant,
                ) for config_id in config_ids for variant in panel["primary_score_variants"]]
                score, _, config_id, variant = tuning._pick(candidates, tolerance)
                policy.update(
                    config_id=config_id, score_variant=variant,
                    hyperparameters=tuning._hyperparameters(registry, model_name, config_id),
                    family_macro_vus_pr=score, selection_status="selected",
                    selection_reason=f"q{ratio}의 family 동일 가중 VUS-PR과 동률 규칙으로 선택",
                )
            ratio_policies.append(policy)
        selection["model_ratio"].extend(ratio_policies)

        # 한 비율만 넘겨 기존 LOFO가 다른 비율의 점수로 recipe를 고르지 않게 한다.
        adaptive = tuning.select_ratio_adaptive_policies(
            [row for row in rows if row["ratio"] == ratio], registry,
            {**budget, "model_panels": ratio_panels}, model_fixed=ratio_policies,
            evaluator_sha256=evaluator_sha256,
        )
        for audit in adaptive["candidate_audit"]:
            if (audit["model"] in supported_models
                    and audit["eligibility"] == "full_panel_config_unavailable"):
                audit.update(
                    eligibility="ratio_unsupported", reason_code="ratio_unsupported",
                    reason="이 비율에서는 비교 패널 전체를 처리할 설정이 없음",
                )
        for name in ("adaptive_lofo", "candidate_audit", "tier_adaptive"):
            selection[name].extend(row for row in adaptive[name] if row["ratio"] == ratio)

    selection["model_ratio"].sort(key=lambda row: (row["tier"], row["model"], row["ratio"]))
    selection["tier_adaptive"].sort(key=lambda row: (row["tier"], row["ratio"]))
    previous_by_tier = {}
    for policy in selection["tier_adaptive"]:
        tier = policy["tier"]
        previous = previous_by_tier.get(tier)
        if policy["selection_status"] != "selected":
            transition = "unavailable"
        elif previous is None:
            transition = "initial"
        elif all(policy[field] == previous[field] for field in (
            "selected_model", "config_id", "score_variant",
        )):
            transition = "keep"
        else:
            transition = "switch"
        row = {
            "tier": tier, "previous_ratio": previous["ratio"] if previous else None,
            "current_ratio": policy["ratio"],
            "previous_model": previous["selected_model"] if previous else "",
            "previous_config_id": previous["config_id"] if previous else "",
            "current_model": policy["selected_model"], "current_config_id": policy["config_id"],
            "transition": transition,
        }
        row["transition_key"] = "tr" + hashlib.sha256(
            tuning._json(row).encode("utf-8"),
        ).hexdigest()[:12]
        selection["policy_transitions"].append(row)
        if policy["selection_status"] == "selected":
            previous_by_tier[tier] = policy
    return selection
