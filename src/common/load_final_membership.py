"""최종 실행 요청 CSV를 registry의 물리 실행으로 바꾼다."""

import csv
import hashlib
from pathlib import Path

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from src.common.model_registry import validate_execution_status


FIELDS = (
    "analysis_kind", "split_role", "tier", "model", "evaluation_ratio",
    "physical_ratio", "config_id", "score_variant", "status", "status_reason",
)
CONDITIONAL_FIELDS = FIELDS + ("series", "group_id", "support_status")
SUPPORT_STATUSES = {"within_dev_support", "out_of_dev_support", "unavailable"}
FINAL_SPLIT_ROLES = {
    "ghl25_final", "train1_to_test1", "train1_train2_to_test2",
}
ANALYSIS_KINDS = {"model_fixed", "tier_fixed", "tier_adaptive", "model_ratio"}
TSPULSE_VARIANTS = {"time", "fft", "pred", "raw_max", "ensemble"}
TARGET_FREE = {"training_free", "strict_zero_shot"}


def normalize_membership_series(series) -> str:
    text = str(series).strip()
    if not text.isascii() or not text.isdecimal() or not 1 <= int(text) <= 99:
        raise ValueError("membership series는 1~99의 정수여야 한다")
    return f"{int(text):02d}"


def _parse_ratio(value: str, field: str) -> int:
    try:
        ratio = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field}는 정수 비율이어야 한다") from error
    if ratio not in SUPPORTED_RATIO_PERCENTS:
        raise ValueError(f"{field}는 지원 비율이어야 한다: {ratio}")
    return ratio


def load_final_membership(path, registry: dict) -> tuple[str, tuple[dict, ...]]:
    """membership 한 파일만 검증하고 파일 SHA와 행을 반환한다."""
    membership_path = Path(path)
    serialized = membership_path.read_bytes()
    text = serialized.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    fieldnames = tuple(reader.fieldnames or ())
    if fieldnames not in (FIELDS, CONDITIONAL_FIELDS):
        raise ValueError("final membership 열이 계약과 다르다")
    conditional = fieldnames == CONDITIONAL_FIELDS

    rows = []
    logical_keys = set()
    for line_number, source in enumerate(reader, start=2):
        if None in source or any(value is None for value in source.values()):
            raise ValueError(f"final membership 행의 열 수가 다르다: {line_number}")
        analysis_kind = source["analysis_kind"]
        split_role = source["split_role"]
        model_name = source["model"]
        if analysis_kind not in ANALYSIS_KINDS:
            raise ValueError(f"지원하지 않는 analysis_kind이다: {analysis_kind!r}")
        if conditional:
            if analysis_kind not in {"model_ratio", "tier_adaptive"}:
                raise ValueError("조건부 membership에는 model_ratio와 tier_adaptive만 허용한다")
            source["series"] = normalize_membership_series(source["series"])
            source["group_id"] = source["group_id"].strip()
            if source["support_status"] not in SUPPORT_STATUSES:
                raise ValueError("조건부 membership의 support_status가 잘못됐다")
        if split_role not in FINAL_SPLIT_ROLES:
            raise ValueError(f"지원하지 않는 final split_role이다: {split_role!r}")
        try:
            model = registry["models"][model_name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"registry에 없는 model이다: {model_name!r}") from error
        if source["tier"] != model["tier"]:
            raise ValueError(f"membership tier가 registry와 다르다: {model_name}")
        known_config_ids = {
            candidate["config_id"] for candidate in model["candidates"]
        }
        no_config = conditional and source["status"] == "unavailable" and not source["config_id"]
        if source["config_id"] not in known_config_ids and not no_config:
            raise ValueError(f"registry에 없는 config_id이다: {source['config_id']!r}")

        evaluation_ratio = _parse_ratio(source["evaluation_ratio"], "evaluation_ratio")
        status = source["status"]
        status_reason = source["status_reason"].strip()
        score_variant = source["score_variant"].strip()
        if model_name == "TSPulse":
            official = registry.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"
            variants = TSPULSE_VARIANTS - ({"raw_max"} if official else {"ensemble"})
            if score_variant not in variants and not (no_config and not score_variant):
                raise ValueError("TSPulse score_variant가 잘못됐다")
            if (conditional and not no_config
                    and registry.get("common_recipe", {}).get("methodology_revision")
                    in {"source_faithful_v3", "paper_tuning_v4"}):
                from src.common.equal_trial_budget import _score_variants

                candidate = next(candidate for candidate in model["candidates"]
                                 if candidate["config_id"] == source["config_id"])
                primary, _ = _score_variants(registry["selection"], model_name, candidate["hyperparameters"])
                if score_variant not in primary:
                    raise ValueError("조건부 TSPulse 정책에 진단용 head를 선택할 수 없다")
        elif score_variant:
            raise ValueError(f"{model_name}에는 score_variant를 지정하지 않는다")

        if status == "runnable":
            if conditional and (
                not source["group_id"] or source["support_status"] == "unavailable"
            ):
                raise ValueError("조건부 runnable 행에는 group_id와 검증 범위가 필요하다")
            if status_reason:
                raise ValueError("runnable 행의 status_reason은 비어 있어야 한다")
            if validate_execution_status(model_name, model) != "ready":
                raise ValueError(f"ready가 아닌 모델은 실행할 수 없다: {model_name}")
            physical_ratio = _parse_ratio(source["physical_ratio"], "physical_ratio")
            expected_ratio = 100 if model["target_use"] in TARGET_FREE else evaluation_ratio
            if physical_ratio != expected_ratio:
                raise ValueError(
                    f"membership의 물리 비율이 모델 계약과 다르다: {model_name}"
                )
        elif status == "unavailable":
            if source["physical_ratio"].strip() or not status_reason:
                raise ValueError("unavailable 행은 물리 비율 없이 이유를 남겨야 한다")
            physical_ratio = None
        else:
            raise ValueError(f"지원하지 않는 membership status이다: {status!r}")

        logical_key = (
            analysis_kind, split_role, model_name, evaluation_ratio, score_variant,
        )
        if conditional:
            logical_key = (
                analysis_kind, split_role, source["series"], source["tier"],
                model_name, evaluation_ratio,
            )
        if logical_key in logical_keys:
            raise ValueError(f"final membership에 중복 실행 요청이 있다: {line_number}")
        logical_keys.add(logical_key)
        rows.append({
            **source,
            "evaluation_ratio": evaluation_ratio,
            "physical_ratio": physical_ratio,
            "score_variant": score_variant,
            "status_reason": status_reason,
        })

    if not rows:
        raise ValueError("final membership은 비어 있을 수 없다")
    if conditional:
        if registry.get("selection", {}).get("primary_hpo_regime") == "full_prefix_per_ratio":
            expected_groups = ({("model_ratio", name) for name in registry["models"]}
                               | {("tier_adaptive", model["tier"]) for model in registry["models"].values()})
            for split_role, series in {(row["split_role"], row["series"]) for row in rows}:
                actual_groups = {
                    (row["analysis_kind"], row["model"] if row["analysis_kind"] == "model_ratio" else row["tier"])
                    for row in rows if row["split_role"] == split_role and row["series"] == series
                }
                if actual_groups != expected_groups:
                    raise ValueError("조건부 membership의 모델·Tier 집단 구성이 완전하지 않다")
        ratio_groups = {}
        for row in rows:
            key = (row["analysis_kind"], row["split_role"], row["series"], row["tier"])
            if row["analysis_kind"] == "model_ratio":
                key += (row["model"],)
            ratio_groups.setdefault(key, []).append(row["evaluation_ratio"])
        for key, ratios in ratio_groups.items():
            if set(ratios) != set(SUPPORTED_RATIO_PERCENTS) or len(ratios) != len(SUPPORTED_RATIO_PERCENTS):
                raise ValueError(f"조건부 membership의 비율 구성이 완전하지 않다: {key}")
        return hashlib.sha256(serialized).hexdigest(), tuple(rows)
    if {row["split_role"] for row in rows} != FINAL_SPLIT_ROLES:
        raise ValueError("final membership의 split 구성이 완전하지 않다")
    model_ratio_keys = {
        (model_name, ratio)
        for model_name in registry["models"]
        for ratio in SUPPORTED_RATIO_PERCENTS
    }
    tier_ratio_keys = {
        (tier, ratio)
        for tier in {model["tier"] for model in registry["models"].values()}
        for ratio in SUPPORTED_RATIO_PERCENTS
    }
    for split_role in FINAL_SPLIT_ROLES:
        kinds = {
            row["analysis_kind"] for row in rows if row["split_role"] == split_role
        }
        if kinds not in (ANALYSIS_KINDS, ANALYSIS_KINDS - {"model_ratio"}):
            raise ValueError(
                f"{split_role}에 model_fixed, tier_fixed와 tier_adaptive가 모두 필요하다"
            )
        model_rows = [
            row for row in rows
            if row["split_role"] == split_role and row["analysis_kind"] == "model_fixed"
        ]
        if {
            (row["model"], row["evaluation_ratio"]) for row in model_rows
        } != model_ratio_keys or len(model_rows) != len(model_ratio_keys):
            raise ValueError(f"{split_role}의 model_fixed 구성이 완전하지 않다")
        if "model_ratio" in kinds:
            ratio_rows = [row for row in rows if row["split_role"] == split_role
                          and row["analysis_kind"] == "model_ratio"]
            if {(row["model"], row["evaluation_ratio"]) for row in ratio_rows} != model_ratio_keys \
                    or len(ratio_rows) != len(model_ratio_keys):
                raise ValueError(f"{split_role}의 model_ratio 구성이 완전하지 않다")
        tier_rows = [
            row for row in rows
            if row["split_role"] == split_role and row["analysis_kind"] == "tier_fixed"
        ]
        if {
            (row["tier"], row["evaluation_ratio"]) for row in tier_rows
        } != tier_ratio_keys or len(tier_rows) != len(tier_ratio_keys):
            raise ValueError(f"{split_role}의 tier_fixed 구성이 완전하지 않다")
        adaptive_rows = [
            row for row in rows
            if row["split_role"] == split_role
            and row["analysis_kind"] == "tier_adaptive"
        ]
        if {
            (row["tier"], row["evaluation_ratio"]) for row in adaptive_rows
        } != tier_ratio_keys or len(adaptive_rows) != len(tier_ratio_keys):
            raise ValueError(f"{split_role}의 tier_adaptive 구성이 완전하지 않다")
    return hashlib.sha256(serialized).hexdigest(), tuple(rows)


def build_final_execution_union(
    membership: tuple[dict, ...], split_role: str, *, series=None,
) -> dict:
    """논리 분석 행을 중복 없는 물리 실행과 score variant로 합친다."""
    union = {}
    conditional = any("series" in row for row in membership)
    if conditional:
        if series is None:
            raise ValueError("조건부 final membership에는 series를 지정해야 한다")
        series = normalize_membership_series(series)
        if not any(row["split_role"] == split_role and row.get("series") == series for row in membership):
            raise ValueError(f"조건부 membership에 해당 split/series가 없다: {split_role}/{series}")
    for row in membership:
        if conditional and row.get("series") != series:
            continue
        if row["split_role"] != split_role or row["status"] != "runnable":
            continue
        key = (row["model"], row["config_id"], row["physical_ratio"])
        union.setdefault(key, set()).add(row["score_variant"])
    return {key: tuple(sorted(variants)) for key, variants in union.items()}
