"""GHL 주 실행 또는 뒷자르기 통제군을 순서대로 실행한다."""

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.exp02_gdn_ghl.check_completeness import (
    back_trim_run_directory,
    build_back_trim_specs,
    build_specs,
    is_back_trim_run_complete,
    is_run_complete,
    run_directory,
)
from src.data_split.load_ghl_series import load_ghl_series
from src.gdn_runner.run_gdn_single import run_gdn_sessions


def build_run_config(series: int, ratio: int) -> dict:
    with (REPOSITORY_ROOT / "configs" / "gdn_hyperparams.yaml").open(encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file)
    model_params = dict(hyperparameters["model_params"])
    model_params["topk"] = model_params.pop("topk_by_dataset")["GHL"]
    train_params = dict(hyperparameters["train_params"])
    train_params.pop("seeds_by_dataset")
    return {
        "fork_path": str(REPOSITORY_ROOT.parent / "gragod-fork"),
        "model_params": model_params,
        "train_params": train_params,
        "naming": {"dataset": "GHL", "series": series, "tier": "t2", "ratio": ratio},
    }  # GraGOD models/train.py:143-145의 모델 인자 보강은 공통 러너가 맡는다.


def find_series_file(data_dir: Path, series: int) -> Path:
    matches = list(data_dir.glob(f"*_GHL_id_{series}_Sensor_tr_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"GHL series {series} 파일은 1개여야 한다: {matches}")
    return matches[0]


def load_inputs(
    data_dir: Path, series: int, ratio: int, config: dict, trim_direction: str = "front",
) -> dict:
    train_params = config["train_params"]
    return load_ghl_series(
        find_series_file(data_dir, series),
        ratio_percent=ratio,
        val_fraction=train_params["val_size"],
        min_train_length=train_params["min_train_length"],
        trim_direction=trim_direction,
    )


def append_failure(
    experiment_dir: Path, spec, error: Exception, trim_direction: str = "front",
) -> None:
    log_dir = "logs" if trim_direction == "front" else "back_trim_logs"
    path = experiment_dir / log_dir / "failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("dataset", "series", "ratio", "seed", "error"))
        if write_header:
            writer.writeheader()
        series, ratio, seed = spec
        writer.writerow({
            "dataset": "GHL", "series": series, "ratio": ratio, "seed": seed,
            "error": f"{type(error).__name__}: {error}",
        })


def run_batch(
    experiment_dir=None,
    data_dir=None,
    specs=None,
    input_loader=load_inputs,
    model_runner=run_gdn_sessions,
    trim_direction="front",
) -> dict:
    experiment_dir = Path(experiment_dir or Path(__file__).resolve().parent)
    data_dir = Path(data_dir or REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M")
    if trim_direction == "front":
        spec_builder, completeness_checker, directory_builder = (
            build_specs, is_run_complete, run_directory,
        )
    elif trim_direction == "back":
        spec_builder, completeness_checker, directory_builder = (
            build_back_trim_specs, is_back_trim_run_complete, back_trim_run_directory,
        )
    else:
        raise ValueError(f"trim_direction은 front 또는 back이어야 한다: {trim_direction}")
    specs = spec_builder() if specs is None else specs
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    cached_key = None
    cached_inputs = None

    for spec in specs:
        if completeness_checker(experiment_dir, spec):
            counts["skipped"] += 1
            continue
        series, ratio, seed = spec
        config = build_run_config(series, ratio)
        config["trim_direction"] = trim_direction
        try:
            if cached_key != (series, ratio):
                cached_inputs = input_loader(
                    data_dir, series, ratio, config, trim_direction=trim_direction,
                )
                cached_key = (series, ratio)
            model_runner(
                config=config,
                train_sessions=cached_inputs["train_sessions"],
                validation_sessions=cached_inputs["validation_sessions"],
                test_sessions=cached_inputs["test_sessions"],
                output_dir=str(directory_builder(experiment_dir, spec)),
                seed=seed,
                input_metadata={
                    "path": str(data_dir.resolve()),
                    "feature_names": cached_inputs["feature_names"],
                    "session_splits": cached_inputs["session_splits"],
                },
            )
            if not completeness_checker(experiment_dir, spec):
                raise RuntimeError("실행이 끝났지만 필수 산출물이 빠졌다")
            counts["completed"] += 1
        except Exception as error:
            append_failure(experiment_dir, spec, error, trim_direction)
            counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M",
    )
    parser.add_argument("--trim-direction", choices=("front", "back"), default="front")
    arguments = parser.parse_args()
    print(run_batch(
        arguments.experiment_dir, arguments.data_dir,
        trim_direction=arguments.trim_direction,
    ))


if __name__ == "__main__":
    main()
