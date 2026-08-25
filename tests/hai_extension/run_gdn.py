"""HAI의 두 시간 조건·seed 10개를 실행하고 학습 그래프를 저장한다."""

import argparse
import csv
import sys
from pathlib import Path

import numpy
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.hai_extension.check_gdn_outputs import (
    adjacency_path,
    build_specs,
    is_run_complete,
    run_directory,
)
from src.data_split.load_hai_sessions import load_hai_temporal_condition
from src.models.tier2.gdn.extract_adjacency import extract_best_adjacency
from src.models.tier2.gdn.run_gdn_single import run_gdn_sessions
from src.common.run_completion import clear_completion_marker, write_completion_marker
from src.common.verify_input_files import verify_input_files
from src.common.verify_run_context import verify_run_context


def build_run_config(condition: int) -> dict:
    with (REPOSITORY_ROOT / "configs" / "gdn_hyperparams.yaml").open(encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file)
    model_params = dict(hyperparameters["model_params"])
    model_params["topk"] = model_params.pop("topk_by_dataset")["HAI"]
    train_params = dict(hyperparameters["train_params"])
    train_params.pop("seeds_by_dataset")
    return {
        "fork_path": str(REPOSITORY_ROOT.parent / "gragod-fork"),
        "model_params": model_params,
        "train_params": train_params,
        "fork_contract": hyperparameters["fork_contract"],
        "naming": {"dataset": "HAI", "series": condition, "tier": "t2", "ratio": 100},
    }


def load_inputs(data_dir: Path, condition: int, config: dict) -> dict:
    train_params = config["train_params"]
    return load_hai_temporal_condition(
        data_dir,
        condition=condition,
        val_fraction=train_params["val_size"],
        min_train_length=train_params["min_train_length"],
    )


def save_adjacency(experiment_dir: Path, condition: int, seed: int, edge_sets) -> None:
    for self_edges, edges in zip((True, False), edge_sets):
        path = adjacency_path(experiment_dir, condition, seed, self_edges)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = numpy.asarray(sorted(edges), dtype=int).T if edges else numpy.empty((2, 0), dtype=int)
        numpy.save(path, array)
    # 저장한 edge 방향은 (이웃, 대상)이다.


def append_failure(experiment_dir: Path, spec, error: Exception) -> None:
    path = experiment_dir / "logs" / "failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("dataset", "condition", "seed", "error"))
        if write_header:
            writer.writeheader()
        condition, seed = spec
        writer.writerow({
            "dataset": "HAI", "condition": condition, "seed": seed,
            "error": f"{type(error).__name__}: {error}",
        })


def run_batch(
    experiment_dir=None,
    data_dir=None,
    specs=None,
    input_loader=load_inputs,
    model_runner=run_gdn_sessions,
    adjacency_extractor=extract_best_adjacency,
    input_verifier=verify_input_files,
    context_verifier=verify_run_context,
) -> dict:
    experiment_dir = Path(
        experiment_dir or REPOSITORY_ROOT / "experiments" / "02_hai_extension"
    )
    data_dir = Path(data_dir or REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "HAI-23.05")
    specs = list(build_specs() if specs is None else specs)
    expected_git_hashes = context_verifier(
        REPOSITORY_ROOT, REPOSITORY_ROOT.parent / "gragod-fork",
    )
    pending_specs = [
        spec for spec in specs
        if not is_run_complete(experiment_dir, spec, expected_git_hashes)
    ]
    counts = {"completed": 0, "skipped": len(specs) - len(pending_specs), "failed": 0}
    if not pending_specs:
        return counts
    input_files = input_verifier("HAI", data_dir)
    cached_condition = None
    cached_inputs = None

    for spec in pending_specs:
        condition, seed = spec
        config = build_run_config(condition)
        run_dir = run_directory(experiment_dir, spec)
        clear_completion_marker(run_dir)
        try:
            if cached_condition != condition:
                cached_inputs = input_loader(data_dir, condition, config)
                cached_condition = condition
            result = model_runner(
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
                    "files": input_files,
                },
            )
            edge_sets = adjacency_extractor(
                result["best_checkpoint_path"], config["model_params"]["topk"],
            )
            save_adjacency(experiment_dir, condition, seed, edge_sets)
            write_completion_marker(run_dir)
            if not is_run_complete(experiment_dir, spec, expected_git_hashes):
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
        default=REPOSITORY_ROOT / "experiments" / "02_hai_extension",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "HAI-23.05",
    )
    arguments = parser.parse_args()
    print(run_batch(arguments.experiment_dir, arguments.data_dir))


if __name__ == "__main__":
    main()
