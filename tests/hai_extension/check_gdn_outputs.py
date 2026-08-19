"""HAI GDN 확장 실행에서 빠진 조합과 산출물을 찾는다."""

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


def build_specs():
    ratios, seeds = load_planned_ratios_and_seeds("HAI")
    return list(itertools.product(ratios, seeds))


def run_directory(experiment_dir, spec) -> Path:
    ratio, seed = spec
    return Path(experiment_dir) / "scores" / "tier2" / "gdn" / f"r{ratio:03d}" / f"s{seed}"


def adjacency_path(experiment_dir, ratio: int, seed: int, self_edges: bool) -> Path:
    suffix = "with_self" if self_edges else "without_self"
    return Path(experiment_dir) / "adjacency" / f"HAI__GDN__t2__r{ratio:03d}__s{seed}__edges_{suffix}.npy"


def expected_score_paths(run_dir: Path, ratio: int, seed: int) -> list[Path]:
    paths = []
    for series in (1, 2):
        arguments = {
            "dataset": "HAI", "series": series, "model": "GDN", "tier": "t2",
            "ratio": ratio, "seed": seed,
        }
        paths.extend(
            run_dir / "scores" / build_score_filename(
                smoothing_kind=smoothing, norm_kind=norm, channels=channels,
                **arguments,
            )
            for smoothing in ("raw", "smoothed")
            for norm in ("trainnorm", "testnorm")
            for channels in (False, True)
        )
        metadata = build_score_filename(
            smoothing_kind="raw", norm_kind="trainnorm", channels=False, **arguments,
        ).replace(".npy", ".meta.json")
        paths.append(run_dir / "scores" / metadata)
    return paths


def is_run_complete(experiment_dir, spec, expected_identity=None) -> bool:
    ratio, seed = spec
    run_dir = run_directory(experiment_dir, spec)
    paths = expected_score_paths(run_dir, ratio, seed) + [
        run_dir / "early_stopping_log.json",
        run_dir / "timing.json",
        run_dir / "snapshots" / "config_snapshot.json",
        run_dir / "training" / "gdn" / "checkpoint.pt",
        adjacency_path(experiment_dir, ratio, seed, True),
        adjacency_path(experiment_dir, ratio, seed, False),
    ]
    return (
        has_completion_marker(run_dir)
        and snapshot_matches_source_identity(run_dir, expected_identity)
        and all(path.is_file() and path.stat().st_size > 0 for path in paths)
    )


def find_missing_runs(experiment_dir, specs=None, expected_identity=None):
    specs = build_specs() if specs is None else specs
    return [
        spec for spec in specs
        if not is_run_complete(experiment_dir, spec, expected_identity)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=REPOSITORY_ROOT / "experiments" / "02_hai_extension",
    )
    arguments = parser.parse_args()
    identity = verify_run_context(REPOSITORY_ROOT)
    missing = find_missing_runs(arguments.experiment_dir, expected_identity=identity)
    for ratio, seed in missing:
        print(f"ratio={ratio:03d}, seed={seed}")
    print(f"missing={len(missing)}/{len(build_specs())}")
    raise SystemExit(bool(missing))


if __name__ == "__main__":
    main()
