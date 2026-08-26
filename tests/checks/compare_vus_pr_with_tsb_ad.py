"""고정 TSB-AD 구현과 프로젝트 VUS-PR을 작은 fixture로 대조한다."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy

from src.채점기.vus_pr import vus_pr


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_COMMIT = "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48"
OFFICIAL_PATH = "TSB_AD/evaluation/basic_metrics.py"
OFFICIAL_URL = (
    "https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/"
    f"{OFFICIAL_COMMIT}/{OFFICIAL_PATH}"
)
OUTPUT_PATH = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
    / "vus_pr" / "official_tsb_ad_comparison.json"
)
ABSOLUTE_TOLERANCE = 1e-12
FIXTURES = (
    {
        "name": "two_ranges_even_window",
        "score": [0.1, 0.2, 0.3, 0.8, 0.9, 0.2, 0.1, 0.7, 0.3, 0.1],
        "label": [0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
        "l_max": 2,
    },
    {
        "name": "two_ranges_odd_window",
        "score": [
            0.4, 0.1, 0.7, 0.3, 0.9, 0.8, 0.2, 0.6,
            0.5, 0.15, 0.75, 0.35, 0.95, 0.45, 0.25,
        ],
        "label": [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        "l_max": 5,
    },
    {
        "name": "tied_scores",
        "score": [0.1, 0.7, 0.7, 0.2, 0.2, 0.9, 0.4, 0.4, 0.8, 0.3, 0.6, 0.5],
        "label": [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        "l_max": 4,
    },
    {
        "name": "constant_scores",
        "score": [0.5] * 20,
        "label": [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "l_max": 4,
    },
)


def _download_official_source() -> bytes:
    with urllib.request.urlopen(OFFICIAL_URL, timeout=30) as response:
        return response.read()


def _load_generate_curve(source: bytes):
    namespace = {"__name__": "tsb_ad_pinned_basic_metrics"}
    exec(compile(source, OFFICIAL_URL, "exec"), namespace)
    return namespace["generate_curve"]


def compare_vus_pr_with_tsb_ad(
    *, output_path=OUTPUT_PATH, source_loader=_download_official_source,
) -> dict:
    """공식 소스를 메모리에서 실행하고 배열을 저장하지 않은 비교 보고서를 쓴다."""
    source = source_loader()
    generate_curve = _load_generate_curve(source)
    rows = []
    for fixture in FIXTURES:
        score = numpy.asarray(fixture["score"], dtype=float)
        label = numpy.asarray(fixture["label"], dtype=int)
        official = float(generate_curve(
            label, score, fixture["l_max"], version="opt", thre=250,
        )[-1])
        current = vus_pr(score, label, fixture["l_max"], n_thresholds=250)
        rows.append({
            **fixture,
            "official_vus_pr": official,
            "project_vus_pr": current,
            "absolute_difference": abs(official - current),
        })

    evaluator_path = REPOSITORY_ROOT / "src" / "채점기" / "vus_pr.py"
    maximum_difference = max(row["absolute_difference"] for row in rows)
    report = {
        "schema_version": 1,
        "status": (
            "passed" if maximum_difference <= ABSOLUTE_TOLERANCE else "failed"
        ),
        "official": {
            "repository": "TheDatumOrg/TSB-AD",
            "commit": OFFICIAL_COMMIT,
            "path": OFFICIAL_PATH,
            "url": OFFICIAL_URL,
            "sha256": hashlib.sha256(source).hexdigest(),
            "function": "generate_curve",
            "version": "opt",
        },
        "evaluator_sha256": hashlib.sha256(evaluator_path.read_bytes()).hexdigest(),
        "n_thresholds": 250,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "maximum_absolute_difference": maximum_difference,
        "fixtures": rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=Path)
    arguments = parser.parse_args()
    source_loader = (
        (lambda: arguments.source_file.read_bytes())
        if arguments.source_file else _download_official_source
    )
    report = compare_vus_pr_with_tsb_ad(source_loader=source_loader)
    if report["status"] != "passed":
        raise RuntimeError("공식 TSB-AD와 VUS-PR 결과가 다르다")
    print(json.dumps({
        "status": report["status"],
        "maximum_absolute_difference": report["maximum_absolute_difference"],
        "output": str(OUTPUT_PATH),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
