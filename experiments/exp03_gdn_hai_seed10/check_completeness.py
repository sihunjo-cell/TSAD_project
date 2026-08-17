"""exp03의 HAI 비율 2개·시드 10개에서 빠진 실행을 찾는다."""

import argparse
import itertools
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.naming import build_score_filename
from src.common.experiment_config import load_planned_ratios_and_seeds
from src.common.run_completion import has_completion_marker, snapshot_matches_git_hashes
from src.common.verify_run_context import verify_run_context


def build_specs():
    ratios, seeds = load_planned_ratios_and_seeds("HAI")
    return list(itertools.product(ratios, seeds))


def run_directory(experiment_dir, spec) -> Path:
    ratio, seed = spec
    return Path(experiment_dir) / "runs" / f"r{ratio:03d}" / f"s{seed}"


def adjacency_path(experiment_dir, ratio: int, seed: int, self_edges: bool) -> Path:
    suffix = "with_self" if self_edges else "without_self"
    name = f"HAI__GDN__t2__r{ratio:03d}__s{seed}__edges_{suffix}.npy"
    return Path(experiment_dir) / "adjacency" / name


def expected_score_paths(run_dir: Path, ratio: int, seed: int) -> list[Path]:
    paths = []
    for series in (1, 2):
        arguments = {
            "dataset": "HAI", "series": series, "model": "GDN", "tier": "t2",
            "ratio": ratio, "seed": seed,
        }
        paths.extend(
            run_dir / "scores" / build_score_filename(
                smoothing_kind=smoothing_kind,
                norm_kind=norm_kind,
                channels=channels,
                **arguments,
            )
            for smoothing_kind in ("raw", "smoothed")
            for norm_kind in ("trainnorm", "testnorm")
            for channels in (False, True)
        )
        metadata_name = build_score_filename(
            smoothing_kind="raw", norm_kind="trainnorm", channels=False, **arguments,
        ).replace(".npy", ".meta.json")
        paths.append(run_dir / "scores" / metadata_name)
    return paths


def is_run_complete(experiment_dir, spec, expected_git_hashes=None) -> bool:
    ratio, seed = spec
    run_dir = run_directory(experiment_dir, spec)
    fixed_paths = expected_score_paths(run_dir, ratio, seed) + [
        run_dir / "early_stopping_log.json",
        run_dir / "timing.json",
        run_dir / "snapshots" / "config_snapshot.json",
        adjacency_path(experiment_dir, ratio, seed, self_edges=True),
        adjacency_path(experiment_dir, ratio, seed, self_edges=False),
    ]
    checkpoints = (run_dir / "training" / "gdn").glob("version_*/best.ckpt")
    return has_completion_marker(run_dir) and snapshot_matches_git_hashes(
        run_dir, expected_git_hashes,
    ) and all(
        path.is_file() and path.stat().st_size > 0 for path in fixed_paths
    ) and any(path.is_file() and path.stat().st_size > 0 for path in checkpoints)


def find_missing_runs(experiment_dir, specs=None, expected_git_hashes=None) -> list[tuple[int, int]]:
    specs = build_specs() if specs is None else specs
    return [
        spec for spec in specs
        if not is_run_complete(experiment_dir, spec, expected_git_hashes)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    expected_git_hashes = verify_run_context(
        REPOSITORY_ROOT, REPOSITORY_ROOT.parent / "gragod-fork",
    )
    missing = find_missing_runs(
        arguments.experiment_dir, expected_git_hashes=expected_git_hashes,
    )
    for ratio, seed in missing:
        print(f"ratio={ratio:03d}, seed={seed}")
    print(f"missing={len(missing)}/{len(build_specs())}")
    raise SystemExit(bool(missing))


if __name__ == "__main__":
    main()
