"""GHL Tier 2 실행에서 빠진 조합과 산출물을 찾는다."""

import argparse
import itertools
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.experiment_config import load_planned_ratios_and_seeds
from src.common.naming import build_score_filename
from src.common.run_completion import has_completion_marker, snapshot_matches_source_identity
from src.common.verify_run_context import verify_run_context


MODELS = ("CI-AE", "LSTM-AD", "USAD", "GDN")


def build_specs(model=None):
    if model is not None and model not in MODELS:
        raise ValueError(f"지원하지 않는 GHL Tier 2 모델이다: {model}")
    ratios, seeds = load_planned_ratios_and_seeds("GHL")
    models = (model,) if model else MODELS
    return list(itertools.product(models, range(1, 26), ratios, seeds))


def run_directory(experiment_dir, spec) -> Path:
    model, series, ratio, seed = spec
    return (
        Path(experiment_dir) / "scores" / "tier2" / model.lower().replace("-", "_")
        / f"series_{series:02d}" / f"r{ratio:03d}" / f"s{seed}"
    )


def expected_score_paths(run_dir: Path, spec) -> list[Path]:
    model, series, ratio, seed = spec
    arguments = {
        "dataset": "GHL", "series": series, "model": model, "tier": "t2",
        "ratio": ratio, "seed": seed,
    }
    paths = [
        run_dir / "scores" / build_score_filename(
            smoothing_kind=smoothing, norm_kind=norm, channels=channels,
            **arguments,
        )
        for smoothing in ("raw", "smoothed")
        for norm in ("trainnorm", "testnorm")
        for channels in (False, True)
    ]
    metadata = build_score_filename(
        smoothing_kind="raw", norm_kind="trainnorm", channels=False, **arguments,
    ).replace(".npy", ".meta.json")
    return paths + [run_dir / "scores" / metadata]


def is_run_complete(experiment_dir, spec, expected_identity=None) -> bool:
    run_dir = run_directory(experiment_dir, spec)
    model = spec[0]
    fixed_paths = expected_score_paths(run_dir, spec) + [
        run_dir / "early_stopping_log.json",
        run_dir / "timing.json",
        run_dir / "snapshots" / "config_snapshot.json",
        run_dir / "training" / model.lower().replace("-", "_") / "checkpoint.pt",
    ]
    return (
        has_completion_marker(run_dir)
        and snapshot_matches_source_identity(run_dir, expected_identity)
        and all(path.is_file() and path.stat().st_size > 0 for path in fixed_paths)
    )


def find_missing_runs(experiment_dir, specs=None, expected_identity=None):
    specs = build_specs() if specs is None else specs
    return [
        spec for spec in specs
        if not is_run_complete(experiment_dir, spec, expected_identity)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=REPOSITORY_ROOT / "experiments" / "01_ghl_main",
    )
    arguments = parser.parse_args()
    identity = verify_run_context(REPOSITORY_ROOT)
    missing = find_missing_runs(
        arguments.experiment_dir, build_specs(arguments.model), identity,
    )
    for model, series, ratio, seed in missing:
        print(f"model={model}, series={series:02d}, ratio={ratio:03d}, seed={seed}")
    print(f"missing={len(missing)}/{len(build_specs(arguments.model))}")
    raise SystemExit(bool(missing))


if __name__ == "__main__":
    main()
