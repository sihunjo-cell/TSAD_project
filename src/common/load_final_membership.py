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
FINAL_SPLIT_ROLES = {
    "ghl25_final", "train1_to_test1", "train1_train2_to_test2",
}
ANALYSIS_KINDS = {"model_fixed", "tier_fixed"}
TSPULSE_VARIANTS = {"time", "fft", "pred", "raw_max"}
TARGET_FREE = {"training_free", "strict_zero_shot"}


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
    if tuple(reader.fieldnames or ()) != FIELDS:
        raise ValueError("final membership 열이 계약과 다르다")

    rows = []
    logical_keys = set()
    for line_number, source in enumerate(reader, start=2):
        analysis_kind = source["analysis_kind"]
        split_role = source["split_role"]
        model_name = source["model"]
        if analysis_kind not in ANALYSIS_KINDS:
            raise ValueError(f"지원하지 않는 analysis_kind이다: {analysis_kind!r}")
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
        if source["config_id"] not in known_config_ids:
            raise ValueError(f"registry에 없는 config_id이다: {source['config_id']!r}")

        evaluation_ratio = _parse_ratio(source["evaluation_ratio"], "evaluation_ratio")
        status = source["status"]
        status_reason = source["status_reason"].strip()
        score_variant = source["score_variant"].strip()
        if model_name == "TSPulse":
            if score_variant not in TSPULSE_VARIANTS:
                raise ValueError("TSPulse score_variant가 잘못됐다")
        elif score_variant:
            raise ValueError(f"{model_name}에는 score_variant를 지정하지 않는다")

        if status == "runnable":
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
        if kinds != ANALYSIS_KINDS:
            raise ValueError(f"{split_role}에 model_fixed와 tier_fixed가 모두 필요하다")
        model_rows = [
            row for row in rows
            if row["split_role"] == split_role and row["analysis_kind"] == "model_fixed"
        ]
        if {
            (row["model"], row["evaluation_ratio"]) for row in model_rows
        } != model_ratio_keys or len(model_rows) != len(model_ratio_keys):
            raise ValueError(f"{split_role}의 model_fixed 구성이 완전하지 않다")
        tier_rows = [
            row for row in rows
            if row["split_role"] == split_role and row["analysis_kind"] == "tier_fixed"
        ]
        if {
            (row["tier"], row["evaluation_ratio"]) for row in tier_rows
        } != tier_ratio_keys or len(tier_rows) != len(tier_ratio_keys):
            raise ValueError(f"{split_role}의 tier_fixed 구성이 완전하지 않다")
    return hashlib.sha256(serialized).hexdigest(), tuple(rows)


def build_final_execution_union(membership: tuple[dict, ...], split_role: str) -> dict:
    """논리 분석 행을 중복 없는 물리 실행과 score variant로 합친다."""
    union = {}
    for row in membership:
        if row["split_role"] != split_role or row["status"] != "runnable":
            continue
        key = (row["model"], row["config_id"], row["physical_ratio"])
        union.setdefault(key, set()).add(row["score_variant"])
    return {key: tuple(sorted(variants)) for key, variants in union.items()}
