"""GHL 주 실행을 순서대로 실행한다."""

import argparse
import csv
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.ghl_main.check_gdn_outputs import (
    build_specs,
    is_run_complete,
    run_directory,
)
from src.data_split.load_ghl_series import load_ghl_series
from src.models.tier2.gdn.run_gdn_single import run_gdn_sessions
from src.common.run_completion import clear_completion_marker, write_completion_marker
from src.common.verify_input_files import verify_input_files
from src.common.verify_run_context import verify_run_context


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
        "fork_contract": hyperparameters["fork_contract"],
        "naming": {"dataset": "GHL", "series": series, "tier": "t2", "ratio": ratio},
    }


def find_series_file(data_dir: Path, series: int) -> Path:
    matches = list(data_dir.glob(f"*_GHL_id_{series}_Sensor_tr_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"GHL series {series} 파일은 1개여야 한다: {matches}")
    return matches[0]


def load_inputs(data_dir: Path, series: int, ratio: int, config: dict) -> dict:
    train_params = config["train_params"]
    return load_ghl_series(
        find_series_file(data_dir, series),
        ratio_percent=ratio,
        val_fraction=train_params["val_size"],
        min_train_length=train_params["min_train_length"],
    )


def append_failure(experiment_dir: Path, spec, error: Exception) -> None:
    path = experiment_dir / "logs" / "failures.csv"
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
    input_verifier=verify_input_files,
    context_verifier=verify_run_context,
) -> dict:
    experiment_dir = Path(
        experiment_dir or REPOSITORY_ROOT / "experiments" / "01_ghl_main"
    )
    data_dir = Path(data_dir or REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M")
    specs = list(build_specs() if specs is None else specs)
    expected_git_hashes = context_verifier(
        REPOSITORY_ROOT, REPOSITORY_ROOT.parent / "gragod-fork",
    )
    pending_specs = [
        spec for spec in specs
        if not is_run_complete(
            experiment_dir, spec, expected_git_hashes=expected_git_hashes,
        )
    ]
    counts = {"completed": 0, "skipped": len(specs) - len(pending_specs), "failed": 0}
    if not pending_specs:
        return counts
    input_files = input_verifier("GHL", data_dir)
    cached_key = None
    cached_inputs = None

    for spec in pending_specs:
        series, ratio, seed = spec
        config = build_run_config(series, ratio)
        run_dir = run_directory(experiment_dir, spec)
        clear_completion_marker(run_dir)
        try:
            if cached_key != (series, ratio):
                cached_inputs = input_loader(data_dir, series, ratio, config)
                cached_key = (series, ratio)
            model_runner(
                config=config,
                train_sessions=cached_inputs["train_sessions"],
                validation_sessions=cached_inputs["validation_sessions"],
                test_sessions=cached_inputs["test_sessions"],
                output_dir=str(run_dir),
                seed=seed,
                input_metadata={
                    "path": str(data_dir.resolve()),
                    "feature_names": cached_inputs["feature_names"],
                    "session_splits": cached_inputs["session_splits"],
                    "files": tuple(
                        item for item in input_files
                        if item["name"] == cached_inputs["session_splits"][0]["source"]
                    ),
                },
            )
            write_completion_marker(run_dir)
            if not is_run_complete(
                experiment_dir, spec, expected_git_hashes=expected_git_hashes,
            ):
                raise RuntimeError("실행이 끝났지만 필수 산출물이 빠졌다")
            counts["completed"] += 1
        except Exception as error:
            append_failure(experiment_dir, spec, error)
            counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=REPOSITORY_ROOT / "experiments" / "01_ghl_main",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M",
    )
    arguments = parser.parse_args()
    print(run_batch(arguments.experiment_dir, arguments.data_dir))


if __name__ == "__main__":
    main()
