"""봉인한 모델 registry를 읽고 후보별 config_id를 붙인다."""

import hashlib
import itertools
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


def _expand_candidates(
    model_name: str, model: dict, common_recipe: dict,
) -> list[dict]:
    fixed = model.get("fixed", {})
    if "candidates" in model:
        varying = model["candidates"]
    else:
        grid = model.get("grid", {})
        varying = [
            dict(zip(grid, values))
            for values in itertools.product(*(grid[key] for key in grid))
        ] or [{}]

    candidates = []
    for values in varying:
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


def validate_primary_hpo_seal(registry: dict) -> None:
    """Dev18 실실행 전에 HPO 예산과 선택 규칙이 봉인됐는지 확인한다."""
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
