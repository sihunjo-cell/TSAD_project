"""모델 recipe의 데이터 독립 식별자를 만든다."""

import hashlib
import json


def _canonical_sha_prefix(payload: dict, prefix: str) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return prefix + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def build_common_recipe_id(common_recipe: dict) -> str:
    """점수 의미를 정하는 공통 recipe의 데이터 독립 식별자를 만든다."""
    if not isinstance(common_recipe, dict) or not common_recipe:
        raise ValueError("common_recipe는 비어 있지 않은 mapping이어야 한다")
    return _canonical_sha_prefix(common_recipe, "r")


def build_config_id(
    *,
    model: str,
    source_commit: str,
    source_checkpoint_sha256: str,
    hyperparameters: dict,
    preprocess_recipe: dict,
    common_recipe: dict,
    checkpoint_config_sha256: str | None = None,
) -> str:
    """봉인 대상만 canonical JSON으로 해시해 `c`와 12자리 hex를 반환한다."""
    payload = {
        "model": model,
        "source_commit": source_commit,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "hyperparameters": hyperparameters,
        "preprocess_recipe": preprocess_recipe,
        "common_recipe": common_recipe,
    }
    if checkpoint_config_sha256 is not None:
        payload["checkpoint_config_sha256"] = checkpoint_config_sha256
    return _canonical_sha_prefix(payload, "c")
