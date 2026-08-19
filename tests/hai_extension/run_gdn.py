"""HAI GDN의 비율 7개·시드 10개 조합을 실행한다."""

import argparse
import csv
import sys
from pathlib import Path

import numpy
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.run_completion import clear_completion_marker, write_completion_marker
from src.common.experiment_config import validate_gdn_training_contract
from src.common.save_tier2_artifacts import save_tier2_artifacts
from src.common.set_seed import set_seed
from src.common.verify_input_files import verify_input_files
from src.common.verify_run_context import (
    verify_run_context,
    verify_source_identity_unchanged,
)
from src.data_split.load_hai_sessions import load_hai_sessions
from src.models.tier2.GDN.adapter import GDN_CONFIG, run_gdn_sessions
from src.models.tier2.GDN.extract_adjacency import extract_best_adjacency
from tests.hai_extension.check_gdn_outputs import (
    adjacency_path,
    build_specs,
    is_run_complete,
    run_directory,
)


def build_run_config(ratio: int, seed: int) -> dict:
    with (REPOSITORY_ROOT / "configs" / "gdn_hyperparams.yaml").open(encoding="utf-8") as file:
        hyperparameters = yaml.safe_load(file)
    with (REPOSITORY_ROOT / "configs" / "scoring_pipeline.yaml").open(encoding="utf-8") as file:
        scoring = yaml.safe_load(file)
    validate_gdn_training_contract(hyperparameters["local_training_contract"])
    return {
        **GDN_CONFIG,
        "topk": hyperparameters["model_params"]["topk_by_dataset"]["HAI"],
        "validation_fraction": hyperparameters["train_params"]["val_size"],
        "normalization_epsilon": scoring["normalization"]["epsilon"],
        "smoothing_window": scoring["smoothing"]["window"],
        "training_contract": hyperparameters["local_training_contract"],
        "naming": {
            "dataset": "HAI", "series": (1, 2), "model": "GDN",
            "tier": "t2", "ratio": ratio,
        },
        "seed": seed,
    }


def load_inputs(data_dir: Path, ratio: int, config: dict) -> dict:
    return load_hai_sessions(
        data_dir, ratio, config["validation_fraction"],
        GDN_CONFIG["window_size"] + 1,
    )


def execute_model(inputs: dict, config: dict, device: str) -> dict:
    return run_gdn_sessions(
        inputs["train_sessions"], inputs["validation_sessions"],
        inputs["test_sessions"], config["topk"], device,
    )


def write_artifacts(result, output_dir, config, source_identity, input_metadata, inputs):
    return save_tier2_artifacts(
        result, output_dir, "HAI", (1, 2), "GDN",
        config["naming"]["ratio"], config["seed"],
        tuple(len(session) for session in inputs["test_sessions"]),
        config, source_identity, input_metadata,
        config["normalization_epsilon"], config["smoothing_window"],
    )


def save_adjacency(experiment_dir: Path, ratio: int, seed: int, edge_sets) -> None:
    for self_edges, edges in zip((True, False), edge_sets):
        path = adjacency_path(experiment_dir, ratio, seed, self_edges)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = numpy.asarray(sorted(edges), dtype=int).T if edges else numpy.empty((2, 0), dtype=int)
        numpy.save(path, array)


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
    model_executor=execute_model,
    artifact_writer=write_artifacts,
    adjacency_extractor=extract_best_adjacency,
    input_verifier=verify_input_files,
    context_verifier=verify_run_context,
    identity_verifier=verify_source_identity_unchanged,
) -> dict:
    experiment_dir = Path(experiment_dir or REPOSITORY_ROOT / "experiments" / "02_hai_extension")
    data_dir = Path(data_dir or REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "HAI-23.05")
    specs = list(build_specs() if specs is None else specs)
    source_identity = context_verifier(REPOSITORY_ROOT)
    pending = [
        spec for spec in specs
        if not is_run_complete(experiment_dir, spec, source_identity)
    ]
    counts = {"completed": 0, "skipped": len(specs) - len(pending), "failed": 0}
    if not pending:
        return counts
    input_files = input_verifier("HAI", data_dir)
    cached_ratio = None
    cached_inputs = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in pending:
        ratio, seed = spec
        config = build_run_config(ratio, seed)
        run_dir = run_directory(experiment_dir, spec)
        clear_completion_marker(run_dir)
        try:
            if cached_ratio != ratio:
                cached_inputs = input_loader(data_dir, ratio, config)
                cached_ratio = ratio
            input_metadata = {
                "path": str(data_dir.resolve()),
                "feature_names": cached_inputs["feature_names"],
                "session_splits": cached_inputs["session_splits"],
                "files": input_files,
            }
            set_seed(seed)
            result = model_executor(inputs=cached_inputs, config=config, device=device)
            artifacts = artifact_writer(
                result=result, output_dir=run_dir, config=config,
                source_identity=source_identity, input_metadata=input_metadata,
                inputs=cached_inputs,
            )
            edge_sets = adjacency_extractor(artifacts["checkpoint_path"], config["topk"])
            save_adjacency(experiment_dir, ratio, seed, edge_sets)
            identity_verifier(source_identity, REPOSITORY_ROOT)
            write_completion_marker(run_dir)
            if not is_run_complete(experiment_dir, spec, source_identity):
                raise RuntimeError("실행이 끝났지만 필수 산출물이 빠졌다")
            counts["completed"] += 1
        except Exception as error:
            clear_completion_marker(run_dir)
            append_failure(experiment_dir, spec, error)
            counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=REPOSITORY_ROOT / "experiments" / "02_hai_extension",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "HAI-23.05",
    )
    arguments = parser.parse_args()
    if (arguments.ratio is None) != (arguments.seed is None):
        parser.error("한 조합만 실행하려면 --ratio와 --seed를 함께 지정해야 한다")
    specs = [(arguments.ratio, arguments.seed)] if arguments.ratio is not None else None
    print(run_batch(arguments.experiment_dir, arguments.data_dir, specs))


if __name__ == "__main__":
    main()
