"""라벨과 점수를 읽지 않고 모델 후보의 최소 길이 제약을 판정한다."""

import hashlib
import json
import math
import re
from collections.abc import Mapping

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from src.data_split.split_ratio_prefix import compute_prefix_counts


DECISION_FIELDS = {"model", "config_id", "ratio", "series", "status"}
STATIC_FEASIBILITY_STATUSES = {"feasible", "structurally_infeasible"}
CONFIG_ID_PATTERN = re.compile(r"^c[0-9a-f]{12}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TARGET_FREE_USES = {"training_free", "strict_zero_shot"}
FEASIBILITY_LEDGER_FIELDS = (
    "model", "tier", "execution_status", "target_use", "config_id",
    "config_order", "logical_ratio", "physical_ratio",
    "uses_training_prefix", "series", "series_order", "family",
    "training_boundary", "row_count", "test_length", "feature_count",
    "available_count", "fit_count", "validation_count", "status",
    "status_reason", "derived_json", "config_registry_sha256",
    "input_manifest_sha256", "inventory_sha256",
)
FEASIBILITY_DECISION_FIELDS = tuple(
    field for field in FEASIBILITY_LEDGER_FIELDS[:-3]
    if field != "execution_status"
)


def collect_fully_feasible_keys(rows, required_series) -> frozenset[tuple[str, str, int]]:
    """Dev18 전 시계열에서 정적으로 가능한 후보 키만 반환한다."""
    if isinstance(required_series, (str, bytes)):
        raise ValueError("required_series는 비어 있지 않은 고유 문자열 모음이어야 한다")
    try:
        required_series = tuple(required_series)
    except TypeError as error:
        raise ValueError(
            "required_series는 비어 있지 않은 고유 문자열 모음이어야 한다"
        ) from error
    if (
        not required_series
        or any(not isinstance(series, str) or not series.strip() for series in required_series)
        or len(set(required_series)) != len(required_series)
    ):
        raise ValueError("required_series는 비어 있지 않은 고유 문자열 모음이어야 한다")

    required_series_set = frozenset(required_series)
    grouped = {}
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != DECISION_FIELDS:
            raise ValueError(f"feasibility row fields가 잘못됐다: {row!r}")
        model = row["model"]
        config_id = row["config_id"]
        ratio = row["ratio"]
        series = row["series"]
        status = row["status"]
        if not isinstance(model, str) or not model.strip() or model != model.strip():
            raise ValueError(f"model key가 잘못됐다: {model!r}")
        if not isinstance(config_id, str) or not CONFIG_ID_PATTERN.fullmatch(config_id):
            raise ValueError(f"config_id key가 잘못됐다: {config_id!r}")
        if type(ratio) is not int or ratio not in SUPPORTED_RATIO_PERCENTS:
            raise ValueError(f"ratio가 잘못됐다: {ratio!r}")
        if series not in required_series_set:
            raise ValueError(f"required_series에 없는 series다: {series!r}")
        if status not in STATIC_FEASIBILITY_STATUSES:
            raise ValueError(f"status가 잘못됐다: {status!r}")

        row_key = (model, config_id, ratio, series)
        if row_key in seen:
            raise ValueError(f"duplicate feasibility row다: {row_key!r}")
        seen.add(row_key)
        candidate_key = (model, config_id, ratio)
        grouped.setdefault(candidate_key, {})[series] = status

    return frozenset(
        candidate_key
        for candidate_key, statuses in grouped.items()
        if statuses.keys() == required_series_set
        and all(status == "feasible" for status in statuses.values())
    )


def _validate_dev18_entries(entries) -> tuple[dict, ...]:
    if not isinstance(entries, list) or len(entries) != 18:
        raise ValueError("Dev18 manifest는 공식 순서의 18개 파일이어야 한다")
    entries = tuple(entries)
    for order, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping) or (
            entry.get("series") != f"{order:02d}" or entry.get("order") != order
        ):
            raise ValueError("Dev18 series와 order 순서가 잘못됐다")
        integers = (
            entry.get("training_boundary"), entry.get("row_count"),
            entry.get("feature_count"),
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError(f"Dev18 shape가 잘못됐다: {entry.get('series')!r}")
        if entry["training_boundary"] >= entry["row_count"]:
            raise ValueError(f"Dev18 학습 경계가 잘못됐다: {entry['series']}")
        if not isinstance(entry.get("family"), str) or not entry["family"].strip():
            raise ValueError(f"Dev18 family가 잘못됐다: {entry['series']}")
    return entries


def build_dev18_feasibility_rows(
    registry: dict,
    entries,
    *,
    config_registry_sha256: str,
    input_manifest_sha256: str,
    inventory_sha256: str,
) -> list[dict]:
    """봉인 registry와 Role-A shape만으로 Dev18 정적 원표를 만든다."""
    entries = _validate_dev18_entries(entries)
    for name, digest in (
        ("config_registry_sha256", config_registry_sha256),
        ("input_manifest_sha256", input_manifest_sha256),
        ("inventory_sha256", inventory_sha256),
    ):
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"{name}는 64자리 소문자 hex여야 한다")
    try:
        models = registry["models"]
    except (KeyError, TypeError) as error:
        raise ValueError("registry models가 없다") from error

    rows = []
    for model_name, model in models.items():
        target_use = model["target_use"]
        uses_training_prefix = target_use not in TARGET_FREE_USES
        for config_order, candidate in enumerate(model["candidates"], start=1):
            for logical_ratio in SUPPORTED_RATIO_PERCENTS:
                physical_ratio = logical_ratio if uses_training_prefix else 100
                for entry in entries:
                    available_count, fit_count, validation_count = (
                        compute_prefix_counts(
                            entry["training_boundary"], logical_ratio,
                        )
                    )
                    result = assess_candidate(
                        model_name,
                        candidate["hyperparameters"],
                        fit_count,
                        validation_count,
                        entry["row_count"] - entry["training_boundary"],
                        entry["feature_count"],
                    )
                    rows.append({
                        "model": model_name,
                        "tier": model["tier"],
                        "execution_status": model["execution_status"],
                        "target_use": target_use,
                        "config_id": candidate["config_id"],
                        "config_order": config_order,
                        "logical_ratio": logical_ratio,
                        "physical_ratio": physical_ratio,
                        "uses_training_prefix": uses_training_prefix,
                        "series": entry["series"],
                        "series_order": entry["order"],
                        "family": entry["family"],
                        "training_boundary": entry["training_boundary"],
                        "row_count": entry["row_count"],
                        "test_length": entry["row_count"] - entry["training_boundary"],
                        "feature_count": entry["feature_count"],
                        "available_count": available_count,
                        "fit_count": fit_count,
                        "validation_count": validation_count,
                        "status": result["status"],
                        "status_reason": result["reason"],
                        "derived_json": json.dumps(
                            result["derived"], sort_keys=True,
                            separators=(",", ":"), ensure_ascii=False,
                        ),
                        "config_registry_sha256": config_registry_sha256,
                        "input_manifest_sha256": input_manifest_sha256,
                        "inventory_sha256": inventory_sha256,
                    })
    return rows


def _validate_dev18_feasibility_rows(rows: tuple[Mapping, ...], registry: dict) -> None:
    models = registry["models"]
    candidates = {
        (model_name, candidate["config_id"]): (order, model, candidate)
        for model_name, model in models.items()
        for order, candidate in enumerate(model["candidates"], start=1)
    }
    identity = {
        field: rows[0][field]
        for field in FEASIBILITY_LEDGER_FIELDS[-3:]
    }
    if any(
        not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
        for value in identity.values()
    ):
        raise ValueError("Dev18 feasibility SHA-256 신원이 잘못됐다")
    series_metadata = {}
    for row in rows:
        if row["status"] not in STATIC_FEASIBILITY_STATUSES:
            raise ValueError("Dev18 feasibility status가 잘못됐다")
        try:
            config_order, model, candidate = candidates[
                (row["model"], row["config_id"])
            ]
        except KeyError as error:
            raise ValueError("Dev18 feasibility config가 registry에 없다") from error
        target_use = model["target_use"]
        uses_training_prefix = target_use not in TARGET_FREE_USES
        expected_static = {
            "tier": model["tier"],
            "execution_status": model["execution_status"],
            "target_use": target_use,
            "config_order": config_order,
            "physical_ratio": row["logical_ratio"] if uses_training_prefix else 100,
            "uses_training_prefix": uses_training_prefix,
        }
        if any(row[field] != value for field, value in expected_static.items()):
            raise ValueError("Dev18 feasibility registry 결정값이 다르다")
        if type(row["logical_ratio"]) is not int or (
            row["logical_ratio"] not in SUPPORTED_RATIO_PERCENTS
        ):
            raise ValueError("Dev18 feasibility logical ratio가 잘못됐다")
        if (
            not isinstance(row["series"], str)
            or type(row["series_order"]) is not int
            or row["series"] != f"{row['series_order']:02d}"
            or row["series_order"] not in range(1, 19)
            or not isinstance(row["family"], str)
            or not row["family"].strip()
        ):
            raise ValueError("Dev18 feasibility series 신원이 잘못됐다")
        metadata = (
            row["series_order"], row["family"], row["training_boundary"],
            row["row_count"], row["test_length"], row["feature_count"],
        )
        if row["series"] in series_metadata and series_metadata[row["series"]] != metadata:
            raise ValueError("Dev18 feasibility series shape가 행마다 다르다")
        series_metadata.setdefault(row["series"], metadata)
        lengths = (
            row["training_boundary"], row["row_count"], row["test_length"],
            row["feature_count"], row["available_count"], row["fit_count"],
            row["validation_count"],
        )
        if any(type(value) is not int or value < 0 for value in lengths):
            raise ValueError("Dev18 feasibility 길이값이 잘못됐다")
        if not 0 < row["training_boundary"] < row["row_count"] or (
            row["test_length"] != row["row_count"] - row["training_boundary"]
        ) or row["feature_count"] < 1:
            raise ValueError("Dev18 feasibility 경계나 shape가 잘못됐다")
        expected_counts = compute_prefix_counts(
            row["training_boundary"], row["logical_ratio"],
        )
        if (
            row["available_count"], row["fit_count"], row["validation_count"]
        ) != expected_counts:
            raise ValueError("Dev18 feasibility prefix count가 잘못됐다")
        result = assess_candidate(
            row["model"], candidate["hyperparameters"], row["fit_count"],
            row["validation_count"], row["test_length"], row["feature_count"],
        )
        expected_derived = json.dumps(
            result["derived"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        )
        if (
            row["status"] != result["status"]
            or row["status_reason"] != result["reason"]
            or row["derived_json"] != expected_derived
        ):
            raise ValueError("Dev18 feasibility 재계산 결과가 원표와 다르다")
        if any(row[field] != value for field, value in identity.items()):
            raise ValueError("Dev18 feasibility SHA-256 신원이 행마다 다르다")


def summarize_dev18_feasibility(rows, registry: dict) -> dict:
    """18개 모두 가능한 config를 model·q별로 집계한다."""
    rows = tuple(rows)
    if not rows or any(
        not isinstance(row, Mapping)
        or tuple(row) != FEASIBILITY_LEDGER_FIELDS
        for row in rows
    ):
        raise ValueError("Dev18 feasibility 원표 열이 계약과 다르다")
    _validate_dev18_feasibility_rows(rows, registry)
    series = tuple(dict.fromkeys(row["series"] for row in rows))
    if series != tuple(f"{order:02d}" for order in range(1, 19)):
        raise ValueError("Dev18 feasibility series 순서가 잘못됐다")

    try:
        tier_q_floor = registry["selection"]["tier_q_floor"]
        minimum_ratio_count = registry["selection"]["minimum_primary_ratio_count"]
    except (KeyError, TypeError) as error:
        raise ValueError("registry selection support 계약이 없다") from error
    models = {}
    physical_keys = set()
    for model_name, model in registry["models"].items():
        by_ratio = {}
        by_ratio_ids = {}
        for ratio in SUPPORTED_RATIO_PERCENTS:
            feasible_ids = []
            for candidate in model["candidates"]:
                key = (model_name, candidate["config_id"], ratio)
                candidate_rows = [
                    row for row in rows
                    if (row["model"], row["config_id"], row["logical_ratio"]) == key
                ]
                if len(candidate_rows) != len(series):
                    raise ValueError(f"Dev18 feasibility 조합이 빠졌다: {key!r}")
                if all(row["status"] == "feasible" for row in candidate_rows):
                    feasible_ids.append(candidate["config_id"])
                    physical_keys.add((
                        model_name,
                        candidate["config_id"],
                        100 if model["target_use"] in TARGET_FREE_USES else ratio,
                    ))
            by_ratio[str(ratio)] = len(feasible_ids)
            by_ratio_ids[str(ratio)] = feasible_ids
        required_ratios = [
            ratio for ratio in SUPPORTED_RATIO_PERCENTS
            if ratio >= tier_q_floor[model["tier"]]
        ]
        supported_ratios = [
            ratio for ratio in required_ratios if by_ratio[str(ratio)]
        ]
        common_supported_ids = set(by_ratio_ids[str(supported_ratios[0])]) if (
            supported_ratios
        ) else set()
        for ratio in supported_ratios[1:]:
            common_supported_ids &= set(by_ratio_ids[str(ratio)])
        representative_ids = set(by_ratio_ids[str(required_ratios[0])])
        for ratio in required_ratios[1:]:
            representative_ids &= set(by_ratio_ids[str(ratio)])
        candidate_order = [candidate["config_id"] for candidate in model["candidates"]]
        common_supported_ids = [
            config_id for config_id in candidate_order
            if config_id in common_supported_ids
        ]
        representative_ids = [
            config_id for config_id in candidate_order
            if config_id in representative_ids
        ]
        if not supported_ratios:
            support_status = "unavailable"
        elif len(supported_ratios) < minimum_ratio_count or not common_supported_ids:
            support_status = "insufficient_ratio_support"
        else:
            support_status = "eligible_for_model_fixed_hpo"
        models[model_name] = {
            "tier": model["tier"],
            "execution_status": model["execution_status"],
            "target_use": model["target_use"],
            "candidate_count": len(model["candidates"]),
            "fully_feasible_config_count_by_ratio": by_ratio,
            "fully_feasible_config_ids_by_ratio": by_ratio_ids,
            "q_floor": tier_q_floor[model["tier"]],
            "dev18_supported_ratios_at_or_above_q_floor": supported_ratios,
            "dev18_common_config_ids_across_supported_ratios": common_supported_ids,
            "dev18_minimum_primary_ratio_count_met": (
                len(supported_ratios) >= minimum_ratio_count
            ),
            "dev18_tier_representative_required_ratios": required_ratios,
            "dev18_tier_representative_config_ids": representative_ids,
            "dev18_tier_representative_eligible": bool(representative_ids),
            "dev18_selection_support_status": support_status,
        }

    actual_keys = [
        (row["model"], row["config_id"], row["logical_ratio"], row["series"])
        for row in rows
    ]
    expected_keys = [
        (model_name, candidate["config_id"], ratio, item)
        for model_name, model in registry["models"].items()
        for candidate in model["candidates"]
        for ratio in SUPPORTED_RATIO_PERCENTS
        for item in series
    ]
    if actual_keys != expected_keys:
        raise ValueError("Dev18 feasibility 원표 순서나 중복이 잘못됐다")
    tiers = {
        tier: {
            "q_floor": floor,
            "minimum_primary_ratio_count": minimum_ratio_count,
            "dev18_model_fixed_hpo_models": [
                name for name, details in models.items()
                if details["tier"] == tier
                and details["dev18_selection_support_status"] == "eligible_for_model_fixed_hpo"
            ],
            "dev18_tier_representative_models": [
                name for name, details in models.items()
                if details["tier"] == tier
                and details["dev18_tier_representative_eligible"]
            ],
        }
        for tier, floor in tier_q_floor.items()
    }
    fully_feasible_logical_key_count = sum(
        count
        for model in models.values()
        for count in model["fully_feasible_config_count_by_ratio"].values()
    )
    decision_bytes = json.dumps(
        [
            {field: row[field] for field in FEASIBILITY_DECISION_FIELDS}
            for row in rows
        ],
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "row_count": len(rows),
        "series_count": len(series),
        "model_count": len(models),
        "config_count": sum(model["candidate_count"] for model in models.values()),
        "ratio_order": list(SUPPORTED_RATIO_PERCENTS),
        "feasible_row_count": sum(row["status"] == "feasible" for row in rows),
        "structurally_infeasible_row_count": sum(
            row["status"] == "structurally_infeasible" for row in rows
        ),
        "fully_feasible_logical_key_count": fully_feasible_logical_key_count,
        "fully_feasible_ledger_row_count": (
            fully_feasible_logical_key_count * len(series)
        ),
        "feasibility_decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "models": models,
        "tiers": tiers,
        "fully_feasible_physical_keys": [
            {"model": model, "config_id": config_id, "physical_ratio": ratio}
            for model, config_id, ratio in sorted(physical_keys)
        ],
    }


def _result(reason="", **derived):
    return {
        "status": "feasible" if not reason else "structurally_infeasible",
        "reason": reason,
        "derived": derived,
    }


def _require_lengths(requirements: dict[str, tuple[int, int]], **derived):
    for split, (actual, minimum) in requirements.items():
        if actual < minimum:
            return _result(f"{split} length {actual} < {minimum}", **derived)
    return None


def assess_candidate(
    model: str,
    hyperparameters: dict,
    fit_length: int,
    validation_length: int,
    test_length: int,
    channel_count: int,
) -> dict:
    """봉인한 구조 제약만으로 한 `(model,config,series,q)`를 판정한다."""
    lengths = (fit_length, validation_length, test_length, channel_count)
    if any(not isinstance(value, int) or value < 0 for value in lengths):
        raise ValueError("길이와 채널 수는 0 이상의 정수여야 한다")

    if model == "MWVAR":
        window = hyperparameters["window"]
        derived = {
            "test_window_count": max(0, test_length - window + 1),
            "native_continuous_score_count": max(0, test_length - window + 1),
            "aligned_score_count": test_length,
        }
        return _require_lengths(
            {"test": (test_length, window)}, **derived,
        ) or _result(**derived)
    if model == "SQDIFF_LAST3":
        window = hyperparameters["window"]
        derived = {
            "test_window_count": max(0, test_length - window + 1),
            "native_continuous_score_count": max(0, test_length - window + 1),
            "aligned_score_count": test_length,
        }
        return _require_lengths(
            {"test": (test_length, window)}, **derived,
        ) or _result(**derived)
    if model == "PCA_LEGACY":
        window = hyperparameters["window"]
        derived = {
            "fit_window_count": max(0, fit_length - window + 1),
            "validation_window_count": max(0, validation_length - window + 1),
            "test_window_count": max(0, test_length - window + 1),
            "native_continuous_score_count": max(0, test_length - window + 1),
            "aligned_score_count": test_length,
        }
        return _require_lengths({
            "fit": (fit_length, window + 1),
            "validation": (validation_length, window),
            "test": (test_length, window),
        }, **derived) or _result(**derived)
    if model == "PaAno":
        patch = hyperparameters["patch_size"]
        fit_patches = max(0, fit_length - patch + 1)
        derived = {
            "fit_patch_count": fit_patches,
            "validation_patch_count": max(0, validation_length - patch + 1),
            "test_patch_count": max(0, test_length - patch + 1),
            "native_continuous_score_count": max(0, test_length - patch + 1),
            "aligned_score_count": test_length,
            "memory_size": math.floor(
                fit_patches * hyperparameters["memory_fraction"]
            ),
        }
        unavailable = _require_lengths({
            "fit": (fit_length, patch),
            "validation": (validation_length, patch),
            "test": (test_length, patch),
        }, **derived)
        if unavailable:
            return unavailable
        memory_size = derived["memory_size"]
        if memory_size < hyperparameters["neighbors"]:
            return _result(
                f"memory size {memory_size} < neighbors {hyperparameters['neighbors']}",
                **derived,
            )
        if fit_patches <= patch:
            return _result(
                f"pretext patch count {fit_patches} <= patch step {patch}",
                **derived,
            )
        return _result(**derived)
    if model == "ALoRa":
        window = hyperparameters["window"]
        pair_count = min(
            hyperparameters["max_pairs"], channel_count * (channel_count - 1) // 2,
        )
        derived = {
            "fit_window_count": max(0, fit_length - window + 1),
            "validation_window_count": max(0, validation_length - window + 1),
            "test_window_count": max(0, test_length - window + 1),
            "native_continuous_score_count": max(0, test_length - window + 1),
            "aligned_score_count": test_length,
            "pair_count": pair_count,
        }
        unavailable = _require_lengths({
            "fit": (fit_length, window),
            "validation": (validation_length, window),
            "test": (test_length, window),
        }, **derived)
        if unavailable:
            return unavailable
        if pair_count < hyperparameters["heads"]:
            return _result(
                f"pair count {pair_count} < heads {hyperparameters['heads']}",
                **derived,
            )
        return _result(**derived)
    if model == "GDN":
        window = hyperparameters["window"]
        topk = max(
            1, min(channel_count - 1, math.floor(hyperparameters["rho"] * channel_count)),
        ) if channel_count >= 2 else 0
        derived = {
            "fit_forecast_count": max(0, fit_length - window),
            "validation_forecast_count": max(0, validation_length - window),
            "test_forecast_count": max(0, test_length - window),
            "native_continuous_score_count": max(0, test_length - window),
            "aligned_score_count": max(0, test_length - window),
            "topk": topk,
        }
        if channel_count < 2:
            return _result(
                "GDN은 self-edge 외 이웃을 위해 채널 2개 이상이 필요하다",
                **derived,
            )
        unavailable = _require_lengths({
            "fit": (fit_length, window + 1),
            "validation": (validation_length, window + 1),
            "test": (test_length, window + 1),
        }, **derived)
        if unavailable:
            return unavailable
        return _result(**derived)
    if model == "TimeRCD":
        context = hyperparameters["context_length"]
        derived = {
            "test_context_block_count": math.ceil(test_length / context),
            "native_continuous_score_count": test_length,
            "aligned_score_count": test_length,
        }
        if hyperparameters.get("checkpoint_variant") == "multi" and channel_count < 2:
            return _result(
                "TimeRCD multi checkpoint에는 채널 2개 이상이 필요하다", **derived,
            )
        return _require_lengths(
            {"test": (test_length, 1)}, **derived,
        ) or _result(**derived)
    if model == "TSPulse":
        context = hyperparameters["context_length"]
        half_aggregation = hyperparameters["aggregation_window"] // 2
        minimum = 3 * context
        derived = {
            "test_context_window_count": max(0, test_length - context),
            "native_continuous_score_count": max(
                0, test_length - context - half_aggregation,
            ),
            "aligned_score_count": test_length,
            "minimum_test_length": minimum,
        }
        return _require_lengths(
            {"test": (test_length, minimum)}, **derived,
        ) or _result(**derived)
    raise ValueError(f"registry에 없는 모델이다: {model}")
