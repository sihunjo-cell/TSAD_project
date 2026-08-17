"""exp02의 GHL 주 실행과 뒷자르기 통제군에서 빠진 조합을 찾는다."""

import argparse
import itertools
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.naming import build_score_filename
from src.common.experiment_config import load_dataset_ratios, load_planned_ratios_and_seeds
from src.common.run_completion import has_completion_marker, snapshot_matches_git_hashes
from src.common.verify_run_context import verify_run_context


def build_specs():
    ratios, seeds = load_planned_ratios_and_seeds("GHL")
    return list(itertools.product(range(1, 26), ratios, seeds))


def run_directory(experiment_dir, spec) -> Path:
    series, ratio, seed = spec
    return Path(experiment_dir) / "runs" / f"series_{series:02d}" / f"r{ratio:03d}" / f"s{seed}"


def build_back_trim_specs():
    ratios = load_dataset_ratios("GHL", ratio_key="back_trim_ratios")
    _, seeds = load_planned_ratios_and_seeds("GHL")
    return list(itertools.product(range(1, 26), ratios, seeds))


def back_trim_run_directory(experiment_dir, spec) -> Path:
    series, ratio, seed = spec
    if ratio == 100:
        return run_directory(experiment_dir, spec)  # D-30: 앞·뒤 입력이 같아 한 번만 저장한다.
    return (
        Path(experiment_dir) / "back_trim_runs"
        / f"series_{series:02d}" / f"r{ratio:03d}" / f"s{seed}"
    )


def expected_score_paths(
    run_dir: Path, series: int, ratio: int, seed: int, model: str = "GDN",
) -> list[Path]:
    arguments = {
        "dataset": "GHL", "series": series, "model": model, "tier": "t2",
        "ratio": ratio, "seed": seed,
    }
    paths = [
        run_dir / "scores" / build_score_filename(
            smoothing_kind=smoothing_kind,
            norm_kind=norm_kind,
            channels=channels,
            **arguments,
        )
        for smoothing_kind in ("raw", "smoothed")
        for norm_kind in ("trainnorm", "testnorm")
        for channels in (False, True)
    ]
    metadata_name = build_score_filename(
        smoothing_kind="raw", norm_kind="trainnorm", channels=False, **arguments,
    ).replace(".npy", ".meta.json")
    return paths + [run_dir / "scores" / metadata_name]


def is_run_directory_complete(
    run_dir: Path, spec, model: str = "GDN", expected_git_hashes=None,
) -> bool:
    series, ratio, seed = spec
    fixed_paths = expected_score_paths(run_dir, series, ratio, seed, model) + [
        run_dir / "early_stopping_log.json",
        run_dir / "timing.json",
        run_dir / "snapshots" / "config_snapshot.json",
    ]
    checkpoints = (run_dir / "training" / "gdn").glob("version_*/best.ckpt")
    return has_completion_marker(run_dir) and snapshot_matches_git_hashes(
        run_dir, expected_git_hashes,
    ) and all(
        path.is_file() and path.stat().st_size > 0 for path in fixed_paths
    ) and any(path.is_file() and path.stat().st_size > 0 for path in checkpoints)


def is_run_complete(experiment_dir, spec, expected_git_hashes=None) -> bool:
    return is_run_directory_complete(
        run_directory(experiment_dir, spec), spec,
        expected_git_hashes=expected_git_hashes,
    )


def is_back_trim_run_complete(experiment_dir, spec, expected_git_hashes=None) -> bool:
    return is_run_directory_complete(
        back_trim_run_directory(experiment_dir, spec), spec,
        expected_git_hashes=expected_git_hashes,
    )


def find_missing_runs(
    experiment_dir, specs=None, expected_git_hashes=None,
) -> list[tuple[int, int, int]]:
    specs = build_specs() if specs is None else specs
    return [
        spec for spec in specs
        if not is_run_complete(experiment_dir, spec, expected_git_hashes)
    ]


def find_missing_back_trim_runs(
    experiment_dir, specs=None, expected_git_hashes=None,
) -> list[tuple[int, int, int]]:
    specs = build_back_trim_specs() if specs is None else specs
    return [
        spec for spec in specs
        if not is_back_trim_run_complete(experiment_dir, spec, expected_git_hashes)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--trim-direction", choices=("front", "back"), default="front")
    arguments = parser.parse_args()
    expected_git_hashes = verify_run_context(
        REPOSITORY_ROOT, REPOSITORY_ROOT.parent / "gragod-fork",
    )
    if arguments.trim_direction == "front":
        missing = find_missing_runs(
            arguments.experiment_dir, expected_git_hashes=expected_git_hashes,
        )
        total = len(build_specs())
    else:
        missing = find_missing_back_trim_runs(
            arguments.experiment_dir, expected_git_hashes=expected_git_hashes,
        )
        total = len(build_back_trim_specs())
    for series, ratio, seed in missing:
        print(f"series={series:02d}, ratio={ratio:03d}, seed={seed}")
    print(f"missing={len(missing)}/{total}")
    raise SystemExit(bool(missing))


if __name__ == "__main__":
    main()
