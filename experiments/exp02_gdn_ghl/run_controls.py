"""GHL −TOPK 또는 TopK 민감도 팔을 실행한다."""

import argparse
import csv
from pathlib import Path

from experiments.exp02_gdn_ghl.check_controls import (
    build_notopk_specs,
    build_topk_sensitivity_specs,
    is_notopk_complete,
    is_topk_sensitivity_complete,
    load_arm_config,
    notopk_run_directory,
    sensitivity_run_directory,
)
from experiments.exp02_gdn_ghl.run_batch import build_run_config, load_inputs
from src.common.run_completion import clear_completion_marker, write_completion_marker
from src.common.verify_input_files import verify_input_file_state, verify_input_files
from src.common.verify_run_context import verify_run_context
from src.gdn_runner.run_gdn_single import run_gdn_sessions


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def append_failure(experiment_dir: Path, arm: str, spec, error: Exception) -> None:
    path = experiment_dir / "control_logs" / "failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    series, ratio, *rest = spec
    topk, seed = (None, rest[0]) if arm == "notopk" else rest
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=("arm", "series", "ratio", "topk", "seed", "error"),
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "arm": arm, "series": series, "ratio": ratio, "topk": topk, "seed": seed,
            "error": f"{type(error).__name__}: {error}",
        })


def run_controls(
    arm: str,
    experiment_dir=None,
    data_dir=None,
    specs=None,
    input_loader=load_inputs,
    model_runner=run_gdn_sessions,
    input_verifier=verify_input_files,
    context_verifier=verify_run_context,
) -> dict:
    experiment_dir = Path(experiment_dir or Path(__file__).resolve().parent)
    data_dir = Path(data_dir or REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M")
    if arm == "notopk":
        specs = list(build_notopk_specs() if specs is None else specs)
        completeness_checker = is_notopk_complete
        directory_builder = notopk_run_directory
    elif arm == "topk_sensitivity":
        specs = list(build_topk_sensitivity_specs() if specs is None else specs)
        completeness_checker = is_topk_sensitivity_complete
        directory_builder = sensitivity_run_directory
    else:
        raise ValueError(f"arm은 notopk 또는 topk_sensitivity여야 한다: {arm}")

    arm_config = load_arm_config()
    expected_git_hashes = context_verifier(
        REPOSITORY_ROOT, REPOSITORY_ROOT.parent / "gragod-fork",
    )
    pending_specs = [
        spec for spec in specs
        if not completeness_checker(
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
        if arm == "notopk":
            series, ratio, seed = spec
            topk = None
            model = arm_config["notopk_model"]
        else:
            series, ratio, topk, seed = spec
            if topk == 5:
                append_failure(
                    experiment_dir, arm, spec,
                    RuntimeError("k=5 주 실행이 없어 민감도 기준을 재사용할 수 없다"),
                )
                counts["failed"] += 1
                continue
            model = f"GDN_K{topk}"
        config = build_run_config(series, ratio)
        if topk is None:
            topk = config["model_params"]["topk"]
        config["experiment_arm"] = arm
        config["naming"]["model"] = model
        config["model_params"]["learn_graph"] = arm != "notopk"
        config["model_params"]["topk"] = topk
        run_dir = directory_builder(experiment_dir, spec)
        clear_completion_marker(run_dir)
        try:
            if cached_key != (series, ratio):
                verify_input_file_state(data_dir, input_files)
                cached_inputs = input_loader(data_dir, series, ratio, config)
                verify_input_file_state(data_dir, input_files)
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
            if not completeness_checker(
                experiment_dir, spec, expected_git_hashes=expected_git_hashes,
            ):
                raise RuntimeError("실행이 끝났지만 필수 산출물이 빠졌다")
            counts["completed"] += 1
        except Exception as error:
            append_failure(experiment_dir, arm, spec, error)
            counts["failed"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("notopk", "topk_sensitivity"), required=True)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M",
    )
    arguments = parser.parse_args()
    print(run_controls(arguments.arm, arguments.experiment_dir, arguments.data_dir))


if __name__ == "__main__":
    main()
