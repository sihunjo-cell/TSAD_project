"""공통 데이터 비율과 validation 설정을 읽는다."""

from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_RATIO_PERCENTS = (5, 10, 20, 40, 60, 80, 100)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset_ratios(
    dataset: str, repository_root=REPOSITORY_ROOT, ratio_key: str = "ratios",
) -> tuple[int, ...]:
    config_dir = Path(repository_root) / "configs"
    preprocessing = load_yaml(config_dir / "data_preprocessing.yaml")
    try:
        ratios = tuple(preprocessing["datasets"][dataset][ratio_key])
    except (KeyError, TypeError) as error:
        raise ValueError(f"ratio 설정이 없다: {dataset}.{ratio_key}") from error
    if not ratios or any(not isinstance(value, int) or value <= 0 for value in ratios):
        raise ValueError(f"ratio는 중복 없는 양의 정수여야 한다: {dataset}.{ratio_key}")
    if len(set(ratios)) != len(ratios):
        raise ValueError(f"ratio는 중복 없는 양의 정수여야 한다: {dataset}.{ratio_key}")
    if ratios != SUPPORTED_RATIO_PERCENTS:
        raise ValueError(
            f"비율 계약은 {SUPPORTED_RATIO_PERCENTS}이다: {dataset}.{ratio_key}={ratios}"
        )
    return ratios


def load_validation_fraction(repository_root=REPOSITORY_ROOT) -> float:
    preprocessing = load_yaml(Path(repository_root) / "configs" / "data_preprocessing.yaml")
    try:
        fraction = preprocessing["common"]["validation_fraction"]
    except (KeyError, TypeError) as error:
        raise ValueError("고정 validation 비율 설정이 없다") from error
    if not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError(f"validation_fraction은 0과 1 사이여야 한다: {fraction!r}")
    return float(fraction)
