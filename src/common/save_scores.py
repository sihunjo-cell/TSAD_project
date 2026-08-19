"""정규화한 채널별 점수의 raw·smoothed 배열과 config 스냅숏을 저장한다."""

import datetime
import json
import os

import numpy

from src.common.naming import build_score_filename
from src.common.normalization import apply_median_iqr, estimate_median_iqr
from src.common.smoothing import trailing_average_smoothing


def validate_channel_scores(values, name: str) -> numpy.ndarray:
    matrix = numpy.asarray(values, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name}는 비어 있지 않은 2차원 배열이어야 한다")
    if not numpy.isfinite(matrix).all():
        raise ValueError(f"{name}는 모두 finite여야 한다")
    return matrix


def save_score_arrays(
    channel_scores: numpy.ndarray,
    output_dir: str,
    dataset: str,
    series: int,
    model: str,
    tier: str,
    ratio: int,
    seed: int,
    norm_kind: str,
    smoothing_window: int = 4,
) -> list[str]:
    """정규화된 채널별 점수 (n_samples, n_features) 하나에서 4개 파일을 저장한다.

    (a) raw __channels: 채널별, smoothing 전
    (b) raw: 채널 max 집계
    (c) smoothed: 채널별 후행 smoothing 후 max 집계
    (d) smoothed __channels: smoothing 후 채널별 (회수 분석용)
    """
    channel_scores = validate_channel_scores(channel_scores, "channel_scores")
    smoothed_channel_scores = trailing_average_smoothing(channel_scores, window=smoothing_window)
    validate_channel_scores(smoothed_channel_scores, "smoothed_channel_scores")

    naming_arguments = {
        "dataset": dataset, "series": series, "model": model, "tier": tier,
        "ratio": ratio, "seed": seed, "norm_kind": norm_kind,
    }
    arrays_by_name = {
        build_score_filename(smoothing_kind="raw", channels=True, **naming_arguments): channel_scores,
        build_score_filename(smoothing_kind="raw", channels=False, **naming_arguments): channel_scores.max(axis=1),
        build_score_filename(smoothing_kind="smoothed", channels=False, **naming_arguments): smoothed_channel_scores.max(axis=1),
        build_score_filename(smoothing_kind="smoothed", channels=True, **naming_arguments): smoothed_channel_scores,
    }

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for filename, array in arrays_by_name.items():
        path = os.path.join(output_dir, filename)
        numpy.save(path, array)
        saved_paths.append(path)
    return saved_paths


def save_score_bundle(
    reference_error_sessions,
    test_errors,
    output_dir: str,
    dataset: str,
    series: int,
    model: str,
    tier: str,
    ratio: int,
    seed: int,
    epsilon: float,
    smoothing_window: int = 4,
) -> list[str]:
    """분리해 만든 학습·validation 오차로 test 점수 8개를 저장한다."""
    if epsilon <= 0:
        raise ValueError("epsilon은 0보다 커야 한다")
    reference_matrices = [
        numpy.abs(validate_channel_scores(errors, "reference_errors"))
        for errors in reference_error_sessions
    ]
    if not reference_matrices:
        raise ValueError("reference_error_sessions가 비어 있다")
    feature_counts = {matrix.shape[1] for matrix in reference_matrices}
    test_matrix = numpy.abs(validate_channel_scores(test_errors, "test_errors"))
    feature_counts.add(test_matrix.shape[1])
    if len(feature_counts) != 1:
        raise ValueError("reference와 test 오차의 채널 수가 다르다")

    reference_errors = numpy.concatenate(reference_matrices, axis=0)
    train_median, train_iqr = estimate_median_iqr(reference_errors)
    test_median, test_iqr = estimate_median_iqr(test_matrix)
    normalized = {
        "trainnorm": apply_median_iqr(test_matrix, train_median, train_iqr, epsilon),
        "testnorm": apply_median_iqr(test_matrix, test_median, test_iqr, epsilon),
    }
    for name, matrix in normalized.items():
        validate_channel_scores(matrix, name)

    paths = []
    for norm_kind, matrix in normalized.items():
        paths.extend(save_score_arrays(
            matrix, output_dir, dataset, series, model, tier, ratio, seed,
            norm_kind, smoothing_window,
        ))
    return paths


def save_score_metadata(
    output_dir: str,
    dataset: str, series: int, model: str, tier: str, ratio: int, seed: int,
    window_size: int, test_length: int, score_length: int, label_slice: tuple,
    source_start: int, source_end_exclusive: int, alignment: str,
) -> str:
    """점수 길이와 라벨 offset을 `.meta.json` 사이드카로 저장한다."""
    if tuple(label_slice) != (source_start, source_end_exclusive):
        raise ValueError("label_slice는 source 범위와 같아야 한다")
    if not 0 <= source_start <= source_end_exclusive <= test_length:
        raise ValueError("source 범위가 test 범위를 벗어났다")
    if score_length != source_end_exclusive - source_start:
        raise ValueError("score_length와 source 범위 길이가 다르다")
    if not alignment:
        raise ValueError("alignment가 비어 있다")
    anchor_filename = build_score_filename(
        dataset, series, model, tier, ratio, seed, "raw", "trainnorm", channels=False)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, anchor_filename[: -len(".npy")] + ".meta.json")
    with open(path, "w", encoding="utf-8") as metadata_file:
        json.dump({
            "window_size": window_size,
            "test_length": test_length,
            "score_length": score_length,
            "label_slice": list(label_slice),
            "source_start": source_start,
            "source_end_exclusive": source_end_exclusive,
            "alignment": alignment,
        }, metadata_file, ensure_ascii=False, indent=2)
    return path


def snapshot_config(config_dict: dict, git_hash: str, output_dir: str) -> str:
    """실행 시점의 config와 소스 버전을 JSON으로 저장한다."""
    os.makedirs(output_dir, exist_ok=True)
    snapshot = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit_hash": git_hash,
        "config": config_dict,
    }
    path = os.path.join(output_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as snapshot_file:
        json.dump(snapshot, snapshot_file, ensure_ascii=False, indent=2)
    return path
