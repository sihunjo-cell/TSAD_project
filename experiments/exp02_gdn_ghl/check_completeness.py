"""exp02의 GHL 주 실행과 뒷자르기 통제군에서 빠진 조합을 찾는다."""

import argparse
import itertools
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.naming import build_score_filename

RATIOS = (5, 10, 20, 50, 100)
BACK_TRIM_RATIOS = (5, 20, 100)  # docs/plan_v4.md:213
SEEDS = (1, 2, 3)


def build_specs():
    return list(itertools.product(range(1, 26), RATIOS, SEEDS))


def run_directory(experiment_dir, spec) -> Path:
    series, ratio, seed = spec
    return Path(experiment_dir) / "runs" / f"series_{series:02d}" / f"r{ratio:03d}" / f"s{seed}"


def build_back_trim_specs():
    return list(itertools.product(range(1, 26), BACK_TRIM_RATIOS, SEEDS))


def back_trim_run_directory(experiment_dir, spec) -> Path:
    series, ratio, seed = spec
    if ratio == 100:
        return run_directory(experiment_dir, spec)  # D-30: 앞·뒤 입력이 같아 한 번만 저장한다.
    return (
        Path(experiment_dir) / "back_trim_runs"
        / f"series_{series:02d}" / f"r{ratio:03d}" / f"s{seed}"
    )


def expected_score_paths(run_dir: Path, series: int, ratio: int, seed: int) -> list[Path]:
    arguments = {
        "dataset": "GHL", "series": series, "model": "GDN", "tier": "t2",
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


def is_run_directory_complete(run_dir: Path, spec) -> bool:
    series, ratio, seed = spec
    fixed_paths = expected_score_paths(run_dir, series, ratio, seed) + [
        run_dir / "early_stopping_log.json",
        run_dir / "snapshots" / "config_snapshot.json",
    ]
    checkpoints = (run_dir / "training" / "gdn").glob("version_*/best.ckpt")
    return all(path.is_file() for path in fixed_paths) and any(checkpoints)


def is_run_complete(experiment_dir, spec) -> bool:
    return is_run_directory_complete(run_directory(experiment_dir, spec), spec)


def is_back_trim_run_complete(experiment_dir, spec) -> bool:
    return is_run_directory_complete(back_trim_run_directory(experiment_dir, spec), spec)


def find_missing_runs(experiment_dir, specs=None) -> list[tuple[int, int, int]]:
    specs = build_specs() if specs is None else specs
    return [spec for spec in specs if not is_run_complete(experiment_dir, spec)]


def find_missing_back_trim_runs(experiment_dir, specs=None) -> list[tuple[int, int, int]]:
    specs = build_back_trim_specs() if specs is None else specs
    return [spec for spec in specs if not is_back_trim_run_complete(experiment_dir, spec)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--trim-direction", choices=("front", "back"), default="front")
    arguments = parser.parse_args()
    if arguments.trim_direction == "front":
        missing = find_missing_runs(arguments.experiment_dir)
        total = len(build_specs())
    else:
        missing = find_missing_back_trim_runs(arguments.experiment_dir)
        total = len(build_back_trim_specs())
    for series, ratio, seed in missing:
        print(f"series={series:02d}, ratio={ratio:03d}, seed={seed}")
    print(f"missing={len(missing)}/{total}")
    raise SystemExit(bool(missing))


if __name__ == "__main__":
    main()
