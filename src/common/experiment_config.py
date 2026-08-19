"""YAML에서 GDN 실험 조합을 읽고 고정 pipeline 계약을 검증한다."""

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


def load_planned_ratios_and_seeds(dataset: str, repository_root=REPOSITORY_ROOT):
    config_dir = Path(repository_root) / "configs"
    ratios = load_dataset_ratios(dataset, repository_root)
    hyperparameters = load_yaml(config_dir / "gdn_hyperparams.yaml")
    try:
        seeds = tuple(hyperparameters["train_params"]["seeds_by_dataset"][dataset])
    except (KeyError, TypeError) as error:
        raise ValueError(f"ratio·seed 설정이 없다: {dataset}") from error
    if (
        not ratios or not seeds
        or any(not isinstance(value, int) or value <= 0 for value in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError(f"ratio·seed는 중복 없는 양의 정수여야 한다: {dataset}")
    return ratios, seeds


def read_value(config: dict, path: str):
    value = config
    try:
        for key in path.split("."):
            value = value[key]
    except (KeyError, TypeError) as error:
        raise ValueError(f"실행 계약 설정이 없다: {path}") from error
    return value


def require_values(config: dict, expected_values: dict) -> None:
    for path, expected in expected_values.items():
        actual = read_value(config, path)
        if actual != expected:
            raise ValueError(f"지원하지 않는 실행 계약 {path}: {actual!r} != {expected!r}")


def validate_pipeline_contract(
    scoring: dict, preprocessing: dict, dataset: str, feature_count: int,
) -> None:
    require_values(scoring, {
        "error.type": "absolute",
        "error.gragod_post_process_scores": False,
        "normalization.method": "median_iqr",
        "normalization.estimation_split": "train_val",
        "normalization.epsilon": 0.01,
        "normalization.author_original_appendix": True,
        "smoothing.implementation": "ours",
        "smoothing.kind": "trailing_mean",
        "smoothing.window": 4,
        "smoothing.boundary": "first_three_timesteps_zero",
        "smoothing.axis": "time",
        "smoothing.per_channel": True,
        "aggregation.mode": "max",
        "pipeline_order.raw": ["normalize", "aggregate_max"],
        "pipeline_order.smoothed": ["normalize", "smooth_per_channel", "aggregate_max"],
    })
    require_values(preprocessing, {
        "common.downsample_factor": 1,
        "common.validation_fraction": 0.1,
        "common.validation_position": "tail",
        "common.validation_split_order": "before_ratio",
        "common.ratio_base": "fit_pool",
        "common.timestamp_as_feature": False,
        "common.label_as_feature": False,
        "common.label_aggregation": None,
        "common.input_scaling.method": "minmax",
        "common.input_scaling.fit_split": "train",
        "common.input_scaling.transform_splits": ["train", "validation", "test"],
        "common.input_scaling.clip": False,
    })
    if dataset == "SYNTH":
        return
    require_values(preprocessing, {
        f"datasets.{dataset}.ratio_application": "per_training_session",
        f"datasets.{dataset}.feature_count": feature_count,
    })
    if dataset == "HAI":
        require_values(preprocessing, {
            "datasets.HAI.training_sessions": 4,
            "datasets.HAI.test_sessions": 2,
            "datasets.HAI.scaler_fit_scope": "all_training_sessions_train",
            "datasets.HAI.cross_session_windows": False,
        })
    elif dataset == "GHL":
        require_values(preprocessing, {
            "datasets.GHL.scaler_fit_scope": "current_series_train",
        })
    else:
        raise ValueError(f"지원하지 않는 dataset이다: {dataset}")


def validate_gdn_training_contract(contract: dict) -> None:
    require_values(contract, {
        "loss": "mse",
        "forecast_horizon": 1,
        "n_workers": 0,
        "validation_shuffle": False,
        "optimizer": "Adam",
        "scheduler": "none",
        "gradient_clip": "none",
    })
