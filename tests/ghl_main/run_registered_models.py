"""활성 registry의 실행 조합을 만들고 현재 실데이터 gate를 강제한다."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import re

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from src.common.execution_identity import (
    expected_dataset_for_identity,
    file_sha256,
    load_input_manifest_role,
    validate_execution_identity,
    validate_input_manifest_role,
)
from src.common.model_registry import (
    load_model_registry_with_sha,
    validate_primary_hpo_seal,
    validate_execution_status,
)
from src.common.load_final_membership import (
    build_final_execution_union,
    load_final_membership,
    normalize_membership_series,
)
from src.data_split.load_dev18_series import load_dev18_registered_inputs
from src.data_split.load_ghl_series import load_ghl_registered_inputs
from src.data_split.load_hai_sessions import load_hai_registered_inputs


class RealDataExecutionBlocked(RuntimeError):
    """사전 검증이 닫히기 전에 실데이터 실행을 요청했다."""


DEFAULT_SPLIT_ROLES = {
    "development": "dev18_selection", "final": "ghl25_final",
}
OUTPUT_SPLITS = {
    "dev18_selection": ("dev18",),
    "ghl25_final": ("ghl25",),
    "train1_to_test1": ("hai", "train1_to_test1"),
    "train1_train2_to_test2": ("hai", "train1_train2_to_test2"),
}
SCORE_VARIANTS = {"time", "fft", "pred", "raw_max", "ensemble"}
REAL_DATA_EXECUTION_ENABLED = True
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "input_manifest.yaml"


def _dataset_files(manifest: dict, dataset: str) -> list:
    try:
        files = manifest["datasets"][dataset]["files"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"입력 manifest에 {dataset} files가 없다") from error
    if not isinstance(files, list):
        raise ValueError(f"입력 manifest의 {dataset} files는 목록이어야 한다")
    return files


def _unique_manifest_entry(manifest: dict, dataset: str, name: str) -> dict:
    matches = [
        entry for entry in _dataset_files(manifest, dataset)
        if isinstance(entry, dict) and entry.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(f"입력 파일은 manifest에서 유일해야 한다: {dataset}/{name}")
    return matches[0]


def _verify_manifest_file(path: Path, entry: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일이 없다: {path}")
    try:
        expected_size = entry["size_bytes"]
        expected_sha256 = entry["sha256"]
    except KeyError as error:
        raise ValueError(f"입력 manifest 파일 지문이 불완전하다: {path.name}") from error
    size_bytes = path.stat().st_size
    if size_bytes != expected_size:
        raise ValueError(
            f"입력 파일 크기가 manifest와 다르다: "
            f"{path.name} {size_bytes} != {expected_size}"
        )
    sha256 = file_sha256(path)
    if sha256 != expected_sha256:
        raise ValueError(f"입력 파일 SHA-256이 manifest와 다르다: {path.name}")
    identity = {"name": path.name, "size_bytes": size_bytes, "sha256": sha256}
    if "source_directory" in entry:
        identity["source_directory"] = entry["source_directory"]
    return identity


def _merge_input_identity(
    inputs: Mapping, *, dataset: str, manifest_sha256: str, verified_files: tuple,
) -> dict:
    existing = inputs.get("input_identity", {})
    if not isinstance(existing, Mapping):
        raise ValueError("input loader의 input_identity는 mapping이어야 한다")
    if len(verified_files) == 1:
        for field, value in verified_files[0].items():
            if field in existing and existing[field] != value:
                raise ValueError(f"input_identity와 검증 파일의 {field}가 다르다")
    additions = {
        "dataset": dataset,
        "input_manifest_sha256": manifest_sha256,
        "verified_files": verified_files,
    }
    for field, value in additions.items():
        if field in existing and existing[field] != value:
            raise ValueError(f"input_identity의 {field}가 봉인 신원과 다르다")
    return {**existing, **additions}


def _load_dev18_manifest_entry(series, manifest: dict) -> dict:
    if not isinstance(series, str):
        raise ValueError("development series는 두 자리 문자열이어야 한다")
    matches = [
        entry for entry in _dataset_files(manifest, "DEV18")
        if isinstance(entry, dict) and entry.get("series") == series
    ]
    if len(matches) != 1:
        raise ValueError(f"DEV18 series는 manifest에서 유일해야 한다: {series}")
    entry = matches[0]
    order = entry.get("order")
    if type(order) is not int or series != f"{order:02d}":
        raise ValueError(f"DEV18 series와 order가 일치하지 않는다: {series}")
    return entry


def load_registered_inputs(
    *, spec, input_manifest_path=None, csv_path=None, series=None, data_root=None,
):
    """봉인 spec과 일치하는 label-free 원시 입력만 읽는다."""
    identity = validate_execution_identity(spec)
    manifest_path = Path(input_manifest_path or DEFAULT_INPUT_MANIFEST_PATH)
    manifest, manifest_sha256 = load_input_manifest_role(
        manifest_path, identity["dataset_role"], identity["split_role"],
    )
    if manifest_sha256 != identity["input_manifest_sha256"]:
        raise ValueError("현재 input manifest SHA-256이 spec과 다르다")
    dataset = expected_dataset_for_identity(identity)
    split_role = identity["split_role"]
    bound_series = (
        normalize_membership_series(spec["series"]) if "series" in spec else None
    )

    if split_role == "dev18_selection":
        if csv_path is not None or series is None or data_root is None:
            raise ValueError("development 입력에는 series와 data root가 필요하다")
        if bound_series is not None and normalize_membership_series(series) != bound_series:
            raise ValueError("입력 series가 봉인 spec과 다르다")
        entry = _load_dev18_manifest_entry(series, manifest)
        root = Path(data_root).resolve()
        source_path = (root / entry["source_directory"] / entry["name"]).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as error:
            raise ValueError("Dev18 입력 경로가 data root 밖을 가리킨다") from error
        verified_files = (_verify_manifest_file(source_path, entry),)
        inputs = load_dev18_registered_inputs(entry, data_root)
    elif split_role == "ghl25_final":
        if csv_path is None or series is not None or data_root is not None:
            raise ValueError("GHL final 입력에는 CSV 경로 하나가 필요하다")
        csv_path = Path(csv_path)
        if bound_series is not None:
            match = re.search(r"_GHL_id_(\d+)_", csv_path.name)
            if match is None or normalize_membership_series(match.group(1)) != bound_series:
                raise ValueError("GHL CSV series가 봉인 spec과 다르다")
        entry = _unique_manifest_entry(manifest, dataset, csv_path.name)
        verified_files = (_verify_manifest_file(csv_path, entry),)
        inputs = load_ghl_registered_inputs(csv_path)
    elif split_role in {"train1_to_test1", "train1_train2_to_test2"}:
        if csv_path is not None or series is not None or data_root is None:
            raise ValueError("HAI final 입력에는 data root와 split_role이 필요하다")
        expected_series = "01" if split_role == "train1_to_test1" else "02"
        if bound_series is not None and bound_series != expected_series:
            raise ValueError("HAI split의 series가 봉인 spec과 다르다")
        role = manifest["roles"][split_role]
        filenames = (*role["normal_training_files"], *role["test_files"])
        root = Path(data_root)
        verified_files = tuple(
            _verify_manifest_file(
                root / filename,
                _unique_manifest_entry(manifest, dataset, filename),
            )
            for filename in filenames
        )
        inputs = load_hai_registered_inputs(data_root, split_role)
    else:
        raise ValueError(f"지원하지 않는 split_role이다: {split_role!r}")

    if not isinstance(inputs, Mapping):
        raise ValueError("등록 input loader 결과는 mapping이어야 한다")
    return {
        **inputs,
        "input_identity": _merge_input_identity(
            inputs, dataset=dataset, manifest_sha256=manifest_sha256,
            verified_files=verified_files,
        ),
    }


def _model_ratios(model: dict) -> tuple[int, ...]:
    if model["target_use"] in {"training_free", "strict_zero_shot"}:
        return (100,)
    return SUPPORTED_RATIO_PERCENTS


def _validate_feasible_keys(registry: dict, feasible_keys) -> frozenset:
    ready_models = {
        name: model for name, model in registry["models"].items()
        if validate_execution_status(name, model) == "ready"
    }
    if isinstance(feasible_keys, (str, bytes)):
        raise ValueError("feasibility allowlist는 key 모음이어야 한다")
    try:
        keys = tuple(feasible_keys)
    except TypeError as error:
        raise ValueError("feasibility allowlist는 key 모음이어야 한다") from error
    if not keys:
        raise ValueError("feasibility allowlist는 비어 있을 수 없다")
    for key in keys:
        if not isinstance(key, tuple) or len(key) != 3:
            raise ValueError(f"feasibility 키는 (model, config_id, ratio)여야 한다: {key!r}")
        model_name, config_id, ratio = key
        if model_name not in ready_models:
            raise ValueError(f"feasibility model이 ready registry에 없다: {model_name!r}")
        if not isinstance(config_id, str):
            raise ValueError(f"feasibility config_id는 문자열이어야 한다: {key!r}")
        known_config_ids = {
            candidate["config_id"] for candidate in ready_models[model_name]["candidates"]
        }
        if config_id not in known_config_ids:
            raise ValueError(f"registry에 없는 feasibility config_id이다: {key!r}")
        if type(ratio) is not int or ratio not in _model_ratios(ready_models[model_name]):
            raise ValueError(f"feasibility ratio를 지원하지 않는다: {key!r}")
    return frozenset(keys)


def build_specs(
    dataset_role: str, *, split_role=None, input_manifest_path=None,
    final_policy_membership_path=None, include_pending: bool = False,
    feasible_keys=None, series=None,
) -> list[dict]:
    registry, registry_sha = load_model_registry_with_sha()
    if dataset_role not in registry["seeds"]:
        raise ValueError(f"지원하지 않는 dataset_role이다: {dataset_role}")
    split_role = split_role or DEFAULT_SPLIT_ROLES.get(dataset_role)
    if dataset_role == "final" and final_policy_membership_path is None:
        raise ValueError("final 실행에는 봉인한 membership이 필요하다")
    if dataset_role == "development" and final_policy_membership_path is not None:
        raise ValueError("development 탐색에는 membership 경로를 넘기지 않는다")
    manifest_path = Path(input_manifest_path or DEFAULT_INPUT_MANIFEST_PATH)
    input_manifest_sha256 = validate_input_manifest_role(
        manifest_path, dataset_role, split_role,
    )
    final_policy_membership_sha256 = None
    final_execution_union = None
    conditional_membership = False
    if dataset_role == "final":
        final_policy_membership_sha256, membership = load_final_membership(
            final_policy_membership_path, registry,
        )
        conditional_membership = any("series" in row for row in membership)
        final_execution_union = build_final_execution_union(membership, split_role, series=series)
        if not final_execution_union and not conditional_membership:
            raise ValueError("final membership의 runnable 실행 합집합이 비어 있다")
    if feasible_keys is not None:
        feasible_keys = _validate_feasible_keys(registry, feasible_keys)
    if dataset_role == "final" and feasible_keys is not None:
        missing = set(final_execution_union) - feasible_keys
        if missing:
            raise ValueError("final membership의 물리 실행이 feasibility allowlist에 없다")
    role_seeds = registry["seeds"][dataset_role]
    specs = []
    for model_name, model in registry["models"].items():
        status = validate_execution_status(model_name, model)
        if status == "unavailable" or (status != "ready" and not include_pending):
            continue
        ratios = _model_ratios(model)
        seeds = role_seeds[:1] if model["deterministic"] else role_seeds
        for candidate in model["candidates"]:
            for ratio in ratios:
                if feasible_keys is not None and (
                    model_name, candidate["config_id"], ratio
                ) not in feasible_keys:
                    continue
                physical_key = (model_name, candidate["config_id"], ratio)
                if final_execution_union is not None and physical_key not in final_execution_union:
                    continue
                for seed in seeds:
                    specs.append({
                        **({"series": normalize_membership_series(series)} if series is not None else {}),
                        "model": model_name,
                        "tier": model["tier"],
                        "config_id": candidate["config_id"],
                        "hyperparameters": deepcopy(candidate["hyperparameters"]),
                        "ratio": ratio,
                        "seed": seed,
                        "dataset_role": dataset_role,
                        "split_role": split_role,
                        "input_manifest_sha256": input_manifest_sha256,
                        "final_policy_membership_sha256": final_policy_membership_sha256,
                        "config_registry_sha256": registry_sha,
                        "target_use": model["target_use"],
                        "source_commit": model["source_commit"],
                        "source_checkpoint_sha256": model["source_checkpoint_sha256"],
                        "checkpoint_config_sha256": model.get(
                            "checkpoint_config_sha256"
                        ),
                        "checkpoint_revision": model.get("checkpoint_revision"),
                        "preprocess_recipe": deepcopy(model["preprocess_recipe"]),
                        "common_recipe": deepcopy(registry["common_recipe"]),
                        "common_recipe_id": registry["common_recipe_id"],
                        "score_variants": (
                            final_execution_union[physical_key]
                            if final_execution_union is not None
                            else ("time", "fft", "pred", "ensemble") if model_name == "TSPulse"
                            and registry["common_recipe"].get("methodology_revision") == "paper_tuning_v4"
                            else tuple(sorted(SCORE_VARIANTS - {"ensemble"})) if model_name == "TSPulse" else ("",)
                        ),
                    })
    return specs


def build_output_directory(
    experiment_dir, spec: dict, *, score_variant: str | None = None,
) -> Path:
    tier_name = {"t1": "tier1", "t2": "tier2", "t3": "tier3"}[spec["tier"]]
    try:
        split_parts = OUTPUT_SPLITS[spec["split_role"]]
    except KeyError as error:
        raise ValueError(f"지원하지 않는 split_role이다: {spec.get('split_role')!r}") from error
    output_directory = Path(experiment_dir) / "scores"
    for part in split_parts:
        output_directory /= part
    output_directory /= (
        Path(tier_name) / spec["model"] / spec["config_id"]
        / f"r{spec['ratio']:03d}" / f"s{spec['seed']}"
    )
    if score_variant is None:
        return output_directory
    official = spec.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"
    variants = SCORE_VARIANTS - ({"raw_max"} if official else {"ensemble"})
    if spec["model"] != "TSPulse" or score_variant not in variants:
        raise ValueError(f"지원하지 않는 score_variant이다: {score_variant}")
    return output_directory / score_variant


def run_batch(
    *, dataset_role: str, allow_real_data: bool = False, executor=None,
    split_role=None, input_manifest_path=None, final_policy_membership_path=None,
    feasible_keys=None, series=None,
):
    """승인 전에는 차단하고, 승인 뒤에는 주입한 실행기로 봉인 spec만 넘긴다."""
    if not REAL_DATA_EXECUTION_ENABLED or not allow_real_data:
        raise RealDataExecutionBlocked(
            "현재 게이트에서는 실제 Dev18·GHL25 모델 실행을 허용하지 않는다"
        )
    if feasible_keys is None:
        raise ValueError("실데이터 실행에는 feasibility allowlist를 명시해야 한다")
    if executor is None:
        raise ValueError("실데이터 gate가 열린 뒤에는 executor를 명시해야 한다")
    if dataset_role == "development":
        validate_primary_hpo_seal(load_model_registry_with_sha()[0])
    return [
        executor(spec)
        for spec in build_specs(
            dataset_role, split_role=split_role,
            input_manifest_path=input_manifest_path,
            final_policy_membership_path=final_policy_membership_path,
            feasible_keys=feasible_keys,
            series=series,
        )
    ]
