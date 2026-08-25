"""계약을 통과한 점수 파일을 VUS-PR·AUPRC 원표로 변환한다."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

from src.common.experiment_config import REPOSITORY_ROOT, load_yaml
from src.common.naming import parse_score_filename
from src.채점기.parser import load_and_validate_score
from src.채점기.vus_pr import vus_pr


GHL_BOUNDARY = re.compile(r"_tr_(\d+)_")


def load_evaluation_settings() -> dict:
    """사전 고정된 채점 설정을 읽고 최소 계약을 확인한다."""
    scoring = load_yaml(REPOSITORY_ROOT / "configs" / "scoring_pipeline.yaml")
    try:
        settings = scoring["evaluation"]
        l_max_by_dataset = settings["l_max_samples"]
        if settings["n_thresholds"] != 250:
            raise ValueError("n_thresholds는 사전 고정값 250이어야 한다.")
        if settings["main_result"] != {"smoothing_kind": "raw", "norm_kind": "trainnorm"}:
            raise ValueError("본 결과는 raw/trainnorm이어야 한다.")
        if settings["sensitivity"]["l_max_multipliers"] != [0.5, 1.0, 2.0]:
            raise ValueError("민감도는 ℓ/2, ℓ, 2ℓ이어야 한다.")
        if settings["sensitivity"]["half_rounding"] != "floor":
            raise ValueError("ℓ/2 정수화는 floor여야 한다.")
        if settings["f1"]["threshold_source"] != "metadata.validation_threshold":
            raise ValueError("F1 threshold는 metadata validation_threshold여야 한다.")
        if settings["f1"]["quantile"] != 0.99:
            raise ValueError("F1 validation quantile은 사전 고정값 0.99여야 한다.")
        if settings["f1"]["score_source"] != "raw_trainnorm_aggregated_validation":
            raise ValueError("F1 threshold는 raw/trainnorm validation 집계 점수에서 산출해야 한다.")
        if settings["f1"]["test_optimized_main_result"] is not False:
            raise ValueError("본 결과에서 test 최적 F1은 금지한다.")
    except (KeyError, TypeError) as error:
        raise ValueError("evaluation 설정 계약이 불완전하다.") from error
    if set(l_max_by_dataset) != {"GHL", "HAI"} or any(
        not isinstance(value, int) or value <= 0 for value in l_max_by_dataset.values()
    ):
        raise ValueError("GHL·HAI의 양의 정수 l_max가 필요하다.")
    return settings


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """동점은 함께 처리하는 표준 point-wise average precision을 계산한다."""
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_label = label[order]
    positive_total = int(sorted_label.sum())
    if positive_total == 0:
        raise ValueError("AUPRC에는 최소 하나의 이상 label이 필요합니다.")

    cumulative_true = np.cumsum(sorted_label)
    distinct_last = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    true_at_threshold = cumulative_true[distinct_last]
    prediction_count = np.flatnonzero(distinct_last) + 1
    precision = true_at_threshold / prediction_count
    recall = true_at_threshold / positive_total
    return float(np.dot(recall - np.r_[0.0, recall[:-1]], precision))


def f1_at_threshold(score: np.ndarray, label: np.ndarray, threshold: float) -> float:
    """point-adjust 없이 고정 validation threshold의 point-wise F1을 계산한다."""
    if not np.isfinite(threshold):
        raise ValueError("validation_threshold는 유한한 수여야 한다.")
    prediction = np.asarray(score) >= threshold
    label = np.asarray(label, dtype=bool)
    true_positive = int(np.sum(prediction & label))
    false_positive = int(np.sum(prediction & ~label))
    false_negative = int(np.sum(~prediction & label))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def align_labels(labels: np.ndarray, label_slice: list[int | None], score_length: int) -> np.ndarray:
    """metadata의 label_slice를 적용하고 score 길이와 정확히 일치하는지 확인한다."""
    start, end = label_slice
    aligned = np.asarray(labels, dtype=int)[slice(start, end)]
    if len(aligned) != score_length:
        raise ValueError(
            "label_slice를 적용한 라벨 길이가 score 길이와 다르다: "
            f"labels={len(labels)}, slice={label_slice}, scores={score_length}"
        )
    return aligned


def _read_binary_csv_column(path: Path, column: str) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path.name}에 {column!r} 열이 없다.")
        values = [row[column] for row in reader]
    try:
        labels = np.asarray(values, dtype=int)
    except ValueError as error:
        raise ValueError(f"{path.name}의 {column} 열은 정수 binary여야 한다.") from error
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(f"{path.name}의 {column} 열은 0/1 binary여야 한다.")
    return labels


def load_test_labels(dataset: str, series: int, dataset_dir: str | Path) -> np.ndarray:
    """파일명 series에 대응하는 원본 test label을 읽는다."""
    dataset_dir = Path(dataset_dir)
    if dataset == "GHL":
        files = sorted(dataset_dir.glob("*_GHL_*.csv"))
        if len(files) != 25 or not 1 <= series <= len(files):
            raise ValueError(f"GHL series는 1..{len(files)}여야 한다: {series}")
        path = files[series - 1]
        boundary = GHL_BOUNDARY.search(path.name)
        if boundary is None:
            raise ValueError(f"GHL 파일명에서 train 경계를 찾지 못했다: {path.name}")
        return _read_binary_csv_column(path, "Label")[int(boundary.group(1)):]
    if dataset == "HAI":
        if series not in (1, 2):
            raise ValueError(f"HAI series는 1 또는 2여야 한다: {series}")
        return _read_binary_csv_column(dataset_dir / f"label-test{series}.csv", "label")
    raise ValueError(f"지원하지 않는 dataset: {dataset}")


def evaluate_score_file(
    score_path: str | Path, labels: np.ndarray, settings: dict | None = None,
) -> dict:
    """점수 하나를 평가해 원표의 한 행을 반환한다."""
    scores, info, metadata = load_and_validate_score(score_path)
    settings = load_evaluation_settings() if settings is None else settings
    l_max_by_dataset = settings["l_max_samples"]
    if info["dataset"] not in l_max_by_dataset:
        raise ValueError(f"l_max가 확정되지 않은 dataset: {info['dataset']}")
    aligned_label = align_labels(labels, metadata["label_slice"], len(scores))
    try:
        validation_threshold = float(metadata["validation_threshold"])
        validation_quantile = float(metadata["validation_threshold_quantile"])
        validation_score_count = int(metadata["validation_score_count"])
        validation_score_source = metadata["validation_score_source"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "본 결과의 F1에는 metadata.validation_threshold가 필요하다."
        ) from error
    if not np.isfinite(validation_threshold):
        raise ValueError("metadata.validation_threshold는 유한한 수여야 한다.")
    if validation_quantile != settings["f1"]["quantile"]:
        raise ValueError("metadata의 validation threshold quantile이 사전 고정값과 다르다.")
    if validation_score_count <= 0:
        raise ValueError("metadata.validation_score_count는 양수여야 한다.")
    if validation_score_source != settings["f1"]["score_source"]:
        raise ValueError("metadata의 validation score source가 사전 고정값과 다르다.")
    l_max = l_max_by_dataset[info["dataset"]]
    sensitivity_l_max = (math.floor(l_max / 2), l_max, 2 * l_max)
    return {
        **info,
        "score_file": Path(score_path).name,
        "score_length": len(scores),
        "label_positive_count": int(aligned_label.sum()),
        "l_max": l_max,
        "n_thresholds": settings["n_thresholds"],
        "validation_threshold": validation_threshold,
        "validation_threshold_quantile": validation_quantile,
        "validation_score_count": validation_score_count,
        "vus_pr": vus_pr(scores, aligned_label, l_max, settings["n_thresholds"]),
        "auprc": average_precision(scores, aligned_label),
        "f1": f1_at_threshold(scores, aligned_label, validation_threshold),
        "vus_pr_lmax_half": vus_pr(scores, aligned_label, sensitivity_l_max[0], settings["n_thresholds"]),
        "vus_pr_lmax_double": vus_pr(scores, aligned_label, sensitivity_l_max[2], settings["n_thresholds"]),
    }


def evaluate_directory(scores_dir: str | Path, dataset_dir: str | Path, output_csv: str | Path) -> list[dict]:
    """본 채점 대상 trainnorm 집계본을 모두 읽어 결정적 순서의 CSV 원표를 쓴다."""
    rows = []
    settings = load_evaluation_settings()
    for score_path in sorted(Path(scores_dir).glob("*.npy")):
        info = parse_score_filename(score_path.name)
        if (
            info["channels"]
            or info["norm_kind"] != settings["main_result"]["norm_kind"]
            or info["smoothing_kind"] != settings["main_result"]["smoothing_kind"]
        ):
            continue
        labels = load_test_labels(info["dataset"], info["series"], dataset_dir)
        rows.append(evaluate_score_file(score_path, labels, settings))
    if not rows:
        raise ValueError("본 채점 대상 trainnorm 집계 점수 파일이 없다.")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    arguments = parser.parse_args()
    rows = evaluate_directory(**vars(arguments))
    print(f"{len(rows)}개 점수 파일을 채점했습니다: {arguments.output_csv}")


if __name__ == "__main__":
    main()
