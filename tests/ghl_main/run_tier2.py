"""GHL Tier 2 네 모델의 고정 조합을 실행한다."""

import argparse
import csv
import sys
from pathlib import Path

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
from src.data_split.load_ghl_series import load_ghl_series
from src.models.tier2.AE.adapter import CI_AE_CONFIG, run_ci_ae_sessions
from src.models.tier2.GDN.adapter import GDN_CONFIG, run_gdn_sessions
from src.models.tier2.LSTMAD.adapter import LSTM_AD_CONFIG, run_lstm_ad_sessions
from src.models.tier2.USAD.adapter import USAD_CONFIG, run_usad_sessions
from tests.ghl_main.check_tier2_outputs import (
    MODELS,
    build_specs,
    is_run_complete,
    run_directory,
)


MODEL_CONFIGS = {
    "CI-AE": CI_AE_CONFIG,
    "LSTM-AD": LSTM_AD_CONFIG,
    "USAD": USAD_CONFIG,
    "GDN": GDN_CONFIG,
}


def build_run_config(model: str, series: int, ratio: int, seed: int) -> dict:
    config = dict(MODEL_CONFIGS[model])
    if model == "GDN":
        with (REPOSITORY_ROOT / "configs" / "gdn_hyperparams.yaml").open(encoding="utf-8") as file:
            gdn_config = yaml.safe_load(file)
        validate_gdn_training_contract(gdn_config["local_training_contract"])
        config["topk"] = gdn_config["model_params"]["topk_by_dataset"]["GHL"]
        config["training_contract"] = gdn_config["local_training_contract"]
    with (REPOSITORY_ROOT / "configs" / "scoring_pipeline.yaml").open(encoding="utf-8") as file:
        scoring = yaml.safe_load(file)
    config.update({
        "validation_fraction": 0.1,
        "normalization_epsilon": scoring["normalization"]["epsilon"],
        "smoothing_window": scoring["smoothing"]["window"],
        "naming": {
            "dataset": "GHL", "series": series, "model": model,
            "tier": "t2", "ratio": ratio,
        },
        "seed": seed,
    })
    return config


def find_series_file(data_dir: Path, series: int) -> Path:
    matches = list(data_dir.glob(f"*_GHL_id_{series}_Sensor_tr_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"GHL series {series} 파일은 1개여야 한다: {matches}")
    return matches[0]


def load_inputs(data_dir: Path, series: int, ratio: int, config: dict) -> dict:
    minimum = {
        "CI-AE": 227, "LSTM-AD": 101, "USAD": 10, "GDN": 6,
    }[config["naming"]["model"]]
    return load_ghl_series(
        find_series_file(data_dir, series), ratio,
        config["validation_fraction"], minimum,
    )


def execute_model(model: str, inputs: dict, config: dict, device: str) -> dict:
    arguments = (
        inputs["train_sessions"], inputs["validation_sessions"],
        inputs["test_sessions"], device,
    )
    if model == "CI-AE":
        return run_ci_ae_sessions(*arguments)
    if model == "LSTM-AD":
        return run_lstm_ad_sessions(*arguments)
    if model == "USAD":
        return run_usad_sessions(*arguments)
    return run_gdn_sessions(*arguments[:3], config["topk"], device)


def write_artifacts(result, output_dir, config, source_identity, input_metadata, inputs):
    naming = config["naming"]
    return save_tier2_artifacts(
        result, output_dir, "GHL", (naming["series"],), naming["model"],
        naming["ratio"], config["seed"],
        tuple(len(session) for session in inputs["test_sessions"]),
        config, source_identity, input_metadata,
        config["normalization_epsilon"], config["smoothing_window"],
    )


def append_failure(experiment_dir: Path, spec, error: Exception) -> None:
    path = experiment_dir / "logs" / "failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("dataset", "model", "series", "ratio", "seed", "error"),
        )
        if write_header:
            writer.writeheader()
        model, series, ratio, seed = spec
        writer.writerow({
            "dataset": "GHL", "model": model, "series": series,
            "ratio": ratio, "seed": seed,
            "error": f"{type(error).__name__}: {error}",
        })


def run_batch(
    experiment_dir=None,
    data_dir=None,
    specs=None,
    input_loader=load_inputs,
    model_executor=execute_model,
    artifact_writer=write_artifacts,
    input_verifier=verify_input_files,
    context_verifier=verify_run_context,
    identity_verifier=verify_source_identity_unchanged,
) -> dict:
    experiment_dir = Path(experiment_dir or REPOSITORY_ROOT / "experiments" / "01_ghl_main")
    data_dir = Path(data_dir or REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M")
    specs = list(build_specs() if specs is None else specs)
    source_identity = context_verifier(REPOSITORY_ROOT)
    pending = [
        spec for spec in specs
        if not is_run_complete(experiment_dir, spec, source_identity)
    ]
    counts = {"completed": 0, "skipped": len(specs) - len(pending), "failed": 0}
    if not pending:
        return counts
    input_files = input_verifier("GHL", data_dir)
    cached_key = None
    cached_inputs = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in pending:
        model, series, ratio, seed = spec
        config = build_run_config(model, series, ratio, seed)
        run_dir = run_directory(experiment_dir, spec)
        clear_completion_marker(run_dir)
        try:
            if cached_key != (model, series, ratio):
                cached_inputs = input_loader(data_dir, series, ratio, config)
                cached_key = (model, series, ratio)
            input_metadata = {
                "path": str(data_dir.resolve()),
                "feature_names": cached_inputs["feature_names"],
                "session_splits": cached_inputs["session_splits"],
                "files": tuple(
                    item for item in input_files
                    if item["name"] == cached_inputs["session_splits"][0]["source"]
                ),
            }
            set_seed(seed)
            result = model_executor(
                model=model, inputs=cached_inputs, config=config, device=device,
            )
            artifact_writer(
                result=result, output_dir=run_dir, config=config,
                source_identity=source_identity, input_metadata=input_metadata,
                inputs=cached_inputs,
            )
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
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--series", type=int)
    parser.add_argument("--ratio", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=REPOSITORY_ROOT / "experiments" / "01_ghl_main",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M",
    )
    arguments = parser.parse_args()
    selectors = (arguments.series, arguments.ratio, arguments.seed)
    if any(value is not None for value in selectors) and not all(value is not None for value in selectors):
        parser.error("한 조합만 실행하려면 --series, --ratio, --seed를 모두 지정해야 한다")
    specs = (
        [(arguments.model, *selectors)] if all(value is not None for value in selectors)
        else build_specs(arguments.model)
    )
    print(run_batch(arguments.experiment_dir, arguments.data_dir, specs))


if __name__ == "__main__":
    main()
