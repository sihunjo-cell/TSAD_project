"""정규화한 채널별 점수의 raw·smoothed 배열과 config 스냅숏을 저장한다."""

import datetime
import json
import os

import numpy

from src.common.naming import build_score_filename
from src.common.smoothing import trailing_average_smoothing


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
    channel_scores = numpy.asarray(channel_scores, dtype=float)
    smoothed_channel_scores = trailing_average_smoothing(channel_scores, window=smoothing_window)

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


def save_score_metadata(
    output_dir: str,
    dataset: str, series: int, model: str, tier: str, ratio: int, seed: int,
    window_size: int, test_length: int, score_length: int, label_slice: tuple,
) -> str:
    """점수 길이와 라벨 offset을 `.meta.json` 사이드카로 저장한다."""
    anchor_filename = build_score_filename(
        dataset, series, model, tier, ratio, seed, "raw", "trainnorm", channels=False)
    path = os.path.join(output_dir, anchor_filename[: -len(".npy")] + ".meta.json")
    with open(path, "w", encoding="utf-8") as metadata_file:
        json.dump({
            "window_size": window_size,
            "test_length": test_length,
            "score_length": score_length,
            "label_slice": list(label_slice),  # labels[label_slice[0]:label_slice[1]]
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
