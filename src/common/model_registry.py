"""봉인한 모델 registry를 읽고 후보별 config_id를 붙인다."""

import hashlib
import itertools
import math
import re
from copy import deepcopy
from pathlib import Path

import yaml

from src.common.build_config_id import build_common_recipe_id, build_config_id


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXECUTION_STATUSES = {"ready", "pending_checkpoint_smoke", "unavailable"}
HPO_REGIMES = {"equal_trial", "runtime_matched"}
BUDGET_ID_PATTERN = re.compile(r"^b[0-9a-f]{12}$")


def _registry_path(repository_root=REPOSITORY_ROOT) -> Path:
    return Path(repository_root) / "configs" / "model_registry.yaml"


def model_registry_sha256(repository_root=REPOSITORY_ROOT) -> str:
    """실행에서 읽는 registry 파일의 SHA-256을 반환한다."""
    return hashlib.sha256(_registry_path(repository_root).read_bytes()).hexdigest()


def resolve_gdn_topk(channel_count: int, *, rho=None, topk=None) -> int:
    """고정 top-k는 그대로 쓰고 기존 rho 후보의 변환 규칙은 보존한다."""
    if channel_count < 2:
        raise ValueError("GDN requires at least two channels")
    if (rho is None) == (topk is None):
        raise ValueError("GDN requires exactly one of rho or topk")
    if topk is not None:
        if isinstance(topk, bool) or not isinstance(topk, int) or topk < 1:
            raise ValueError("GDN topk must be a positive integer")
        if topk > channel_count:
            raise ValueError(f"GDN topk {topk} > channel count {channel_count}")
        return topk
    if isinstance(rho, bool) or not isinstance(rho, (int, float)) or not math.isfinite(rho) or rho <= 0:
        raise ValueError("GDN rho must be positive and finite")
    return max(1, min(channel_count - 1, math.floor(rho * channel_count)))


def _expand_candidates(
    model_name: str, model: dict, common_recipe: dict,
) -> list[dict]:
    fixed = model.get("fixed", {})
    if "candidates" in model and "grid" in model:
        raise ValueError(f"{model_name} candidates와 grid를 함께 지정할 수 없다")
    if not isinstance(fixed, dict):
        raise ValueError(f"{model_name} fixed는 mapping이어야 한다")
    if "candidates" in model:
        varying = model["candidates"]
        if not isinstance(varying, list) or not varying:
            raise ValueError(f"{model_name} candidates는 비어 있지 않은 목록이어야 한다")
    else:
        grid = model.get("grid", {})
        if not isinstance(grid, dict) or any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError(f"{model_name} grid의 각 축은 비어 있지 않은 목록이어야 한다")
        varying = [
            dict(zip(grid, values))
            for values in itertools.product(*(grid[key] for key in grid))
        ] or [{}]

    candidates = []
    seen = set()
    for values in varying:
        if not isinstance(values, dict):
            raise ValueError(f"{model_name} 후보는 mapping이어야 한다")
        overlap = fixed.keys() & values.keys()
        if overlap:
            raise ValueError(
                f"{model_name} fixed 파라미터를 후보가 덮어쓴다: {sorted(overlap)}"
            )
        hyperparameters = {**fixed, **values}
        candidate = {"hyperparameters": hyperparameters}
        candidate["config_id"] = build_config_id(
            model=model_name,
            source_commit=model["source_commit"],
            source_checkpoint_sha256=model["source_checkpoint_sha256"],
            checkpoint_config_sha256=model.get("checkpoint_config_sha256"),
            hyperparameters=hyperparameters,
            preprocess_recipe=model["preprocess_recipe"],
            common_recipe=common_recipe,
        )
        if candidate["config_id"] in seen:
            raise ValueError(f"{model_name} 중복 후보가 있다: {candidate['config_id']}")
        seen.add(candidate["config_id"])
        candidates.append(candidate)
    return candidates


def _validate_checkpoint_identity(model_name: str, model: dict) -> None:
    if model["source_checkpoint_sha256"] == "none":
        return
    for field in ("source_checkpoint_sha256", "checkpoint_config_sha256"):
        if not SHA256_PATTERN.fullmatch(str(model.get(field, ""))):
            raise ValueError(f"{model_name} {field}는 64자리 소문자 hex여야 한다")


def validate_execution_status(model_name: str, model: dict) -> str:
    if "execution_status" not in model:
        raise ValueError(f"{model_name} execution_status가 빠졌다")
    status = model["execution_status"]
    if status not in EXECUTION_STATUSES:
        raise ValueError(f"{model_name} execution_status가 잘못됐다: {status!r}")
    if status != "ready" and not str(model.get("status_reason", "")).strip():
        raise ValueError(f"{model_name} {status}에는 status_reason이 필요하다")
    return status


def load_model_registry_with_sha(repository_root=REPOSITORY_ROOT) -> tuple[dict, str]:
    """같은 파일 snapshot에서 확장한 registry와 SHA-256을 함께 반환한다."""
    path = _registry_path(repository_root)
    serialized = path.read_bytes()
    registry = deepcopy(yaml.safe_load(serialized.decode("utf-8")))
    common_recipe = registry.get("common_recipe")
    registry["common_recipe_id"] = build_common_recipe_id(common_recipe)
    for model_name, model in registry["models"].items():
        model["execution_status"] = validate_execution_status(model_name, model)
        _validate_checkpoint_identity(model_name, model)
        model["candidates"] = _expand_candidates(
            model_name, model, common_recipe,
        )
    return registry, hashlib.sha256(serialized).hexdigest()


def load_model_registry(repository_root=REPOSITORY_ROOT) -> dict:
    return load_model_registry_with_sha(repository_root)[0]


def validate_primary_hpo_seal(registry: dict, *, budget=None) -> None:
    """Dev18 실실행 전에 HPO 예산과 선택 규칙이 봉인됐는지 확인한다."""
    if budget is not None and budget.get("experiment_mode") == "full_prefix_v2":
        from src.common.equal_trial_budget import _canonical_bytes, _score_variants, registry_space_sha256

        if (registry.get("common_recipe", {}).get("methodology_revision") == "source_faithful_v3"
                and registry["selection"].get("tspulse_prediction_aggregation_window") != 96):
            raise ValueError("source_faithful_v3는 TSPulse pred를 aggregation 96에서만 선택한다")
        _score_variants(registry["selection"], "TSPulse")
        for candidate in registry["models"].get("TSPulse", {}).get("candidates", []):
            _score_variants(registry["selection"], "TSPulse", candidate["hyperparameters"])
        core = {key: value for key, value in budget.items() if key not in {"budget_id", "budget_sha256"}}
        digest = hashlib.sha256(_canonical_bytes(core)).hexdigest()
        if (
            budget.get("budget_sha256") != digest or budget.get("budget_id") != "b" + digest[:12]
            or budget.get("registry_space_sha256") != registry_space_sha256(registry)
            or registry["selection"].get("primary_hpo_regime") != "full_prefix_per_ratio"
            or any(model.get("execution_status") != "ready" for model in registry["models"].values())
        ):
            raise ValueError("full-prefix 예산 또는 후보 공간의 봉인이 다르다")
        return
    try:
        selection = registry["selection"]
        regime = selection["primary_hpo_regime"]
        budget_id = selection["budget_id"]
        rule_id = selection["selection_rule_id"]
    except (KeyError, TypeError) as error:
        raise ValueError("registry selection 봉인이 불완전하다") from error
    if selection.get("selection_status") != "ready":
        raise ValueError("primary HPO regime과 예산 봉인이 ready가 아니다")
    if (
        regime not in HPO_REGIMES
        or not isinstance(budget_id, str)
        or not BUDGET_ID_PATTERN.fullmatch(budget_id)
        or not str(rule_id).strip()
    ):
        raise ValueError("primary HPO regime과 예산을 먼저 봉인해야 한다")
    if selection.get("primary_score_variants") != {"TSPulse": ["raw_max"]}:
        raise ValueError("primary score variant 봉인이 잘못됐다")
