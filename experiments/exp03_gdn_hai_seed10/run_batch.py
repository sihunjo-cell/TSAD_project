"""HAI 비율 2개·시드 10개를 실행하고 학습 그래프를 저장한다."""

import argparse
import csv
import sys
from pathlib import Path

import numpy
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.exp03_gdn_hai_seed10.check_completeness import (
    adjacency_path,
    build_specs,
    is_run_complete,
    run_directory,
)
from src.data_split.load_hai_sessions import load_hai_sessions
from src.gdn_runner.extract_adjacency import extract_best_adjacency
from src.gdn_runner.run_gdn_single import run_gdn_sessions


def build_run_config(ratio: int) -> dict:
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
        "naming": {"dataset": "HAI", "series": (1, 2), "tier": "t2", "ratio": ratio},
    }  # GraGOD models/train.py:143-145의 모델 인자 보강은 공통 러너가 맡는다.


def load_inputs(data_dir: Path, ratio: int, config: dict) -> dict:
    train_params = config["train_params"]
    return load_hai_sessions(
        data_dir,
        ratio_percent=ratio,
        val_fraction=train_params["val_size"],
        min_train_length=train_params["min_train_length"],
    )


def save_adjacency(experiment_dir: Path, ratio: int, seed: int, edge_sets) -> None:
    for self_edges, edges in zip((True, False), edge_sets):
        path = adjacency_path(experiment_dir, ratio, seed, self_edges)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = numpy.asarray(sorted(edges), dtype=int).T if edges else numpy.empty((2, 0), dtype=int)
        numpy.save(path, array)
    # GraGOD models/gdn/model.py:139-155와 같은 cos-sim topk·(이웃, 대상) 방향이다.


def append_failure(experiment_dir: Path, spec, error: Exception) -> None:
    path = experiment_dir / "logs" / "failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("dataset", "ratio", "seed", "error"))
        if write_header:
            writer.writeheader()
        ratio, seed = spec
        writer.writerow({
            "dataset": "HAI", "ratio": ratio, "seed": seed,
            "error": f"{type(error).__name__}: {error}",
        })


def run_batch(
    experiment_dir=None,
    data_dir=None,
    specs=None,
    input_loader=load_inputs,
    model_runner=run_gdn_sessions,
    adjacency_extractor=extract_best_adjacency,
) -> dict:
    experiment_dir = Path(experiment_dir or Path(__file__).resolve().parent)
    data_dir = Path(data_dir or REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "HAI-23.05")
    specs = build_specs() if specs is None else specs
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    cached_ratio = None
    cached_inputs = None

    for spec in specs:
        if is_run_complete(experiment_dir, spec):
            counts["skipped"] += 1
            continue
        ratio, seed = spec
        config = build_run_config(ratio)
        try:
            if cached_ratio != ratio:
                cached_inputs = input_loader(data_dir, ratio, config)
                cached_ratio = ratio
            result = model_runner(
                config=config,
                train_sessions=cached_inputs["train_sessions"],
                validation_sessions=cached_inputs["validation_sessions"],
                test_sessions=cached_inputs["test_sessions"],
                output_dir=str(run_directory(experiment_dir, spec)),
                seed=seed,
                input_metadata={
                    "path": str(data_dir.resolve()),
                    "feature_names": cached_inputs["feature_names"],
                    "session_splits": cached_inputs["session_splits"],
                },
            )
            edge_sets = adjacency_extractor(
                result["best_checkpoint_path"], config["model_params"]["topk"],
            )
            save_adjacency(experiment_dir, ratio, seed, edge_sets)
            if not is_run_complete(experiment_dir, spec):
                raise RuntimeError("실행이 끝났지만 필수 산출물이 빠졌다")
            counts["completed"] += 1
        except Exception as error:
            append_failure(experiment_dir, spec, error)
            counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "HAI-23.05",
    )
    arguments = parser.parse_args()
    print(run_batch(arguments.experiment_dir, arguments.data_dir))


if __name__ == "__main__":
    main()
