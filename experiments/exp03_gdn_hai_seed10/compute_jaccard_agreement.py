"""HAI 시드 10개의 self-edge 제거 그래프를 전수 쌍별 비교한다."""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.exp03_gdn_hai_seed10.check_completeness import (
    adjacency_path,
    build_specs,
    find_missing_runs,
)
from src.common.verify_run_context import verify_run_context


def calculate_jaccard(first_edges: set, second_edges: set) -> float:
    union = first_edges | second_edges
    return len(first_edges & second_edges) / len(union) if union else 1.0


def load_edges(path: Path) -> set[tuple[int, int]]:
    array = numpy.load(path)
    if array.ndim != 2 or array.shape[0] != 2:
        raise ValueError(f"인접행렬은 (2, edge_count)여야 한다: {path}, {array.shape}")
    return {
        (int(source), int(target))
        for source, target in array.T
        if source != target
    }  # D-08: 모든 시드에 공통인 self-edge는 일치도를 부풀리므로 제외한다.


def compute_pairwise_rows(experiment_dir, ratio: int, seeds=range(1, 11)) -> list[dict]:
    edges_by_seed = {
        seed: load_edges(adjacency_path(experiment_dir, ratio, seed, self_edges=False))
        for seed in seeds
    }
    return [
        {
            "ratio": ratio,
            "seed_a": first_seed,
            "seed_b": second_seed,
            "jaccard": calculate_jaccard(
                edges_by_seed[first_seed], edges_by_seed[second_seed],
            ),
        }
        for first_seed, second_seed in itertools.combinations(edges_by_seed, 2)
    ]  # D-18: seed 10개에서 기준 그래프를 고르지 않고 45쌍을 모두 비교한다.


def summarize_rows(rows: list[dict]) -> dict:
    values = numpy.asarray([row["jaccard"] for row in rows], dtype=float)
    return {
        "ratio": rows[0]["ratio"],
        "pair_count": len(rows),
        "minimum": float(values.min()),
        "q1": float(numpy.quantile(values, 0.25)),
        "median": float(numpy.median(values)),
        "q3": float(numpy.quantile(values, 0.75)),
        "maximum": float(values.max()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    specs = build_specs()
    expected_git_hashes = verify_run_context(
        REPOSITORY_ROOT, REPOSITORY_ROOT.parent / "gragod-fork",
    )
    missing = find_missing_runs(
        arguments.experiment_dir, specs, expected_git_hashes,
    )
    if missing:
        raise RuntimeError(f"현재 commit에서 끝나지 않은 HAI 실행이 있다: {len(missing)}개")
    pairwise_rows = []
    summary_rows = []
    for ratio in sorted({ratio for ratio, _seed in specs}):
        seeds = (seed for spec_ratio, seed in specs if spec_ratio == ratio)
        ratio_rows = compute_pairwise_rows(arguments.experiment_dir, ratio, seeds)
        pairwise_rows.extend(ratio_rows)
        summary_rows.append(summarize_rows(ratio_rows))
    for row in pairwise_rows + summary_rows:
        row.update({
            "tsad_commit": expected_git_hashes["tsad_project"],
            "gragod_commit": expected_git_hashes["gragod_fork"],
        })
    analysis_dir = arguments.experiment_dir / "analysis"
    write_csv(analysis_dir / "jaccard_pairs.csv", pairwise_rows)
    write_csv(analysis_dir / "jaccard_summary.csv", summary_rows)


if __name__ == "__main__":
    main()
