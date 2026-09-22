"""정규화한 채널별 점수의 raw·smoothed 배열과 config 스냅숏을 저장한다."""

import datetime
import json
import os

import numpy

from src.common.naming import build_score_filename
from src.common.smoothing import trailing_average_smoothing


def validate_scores(values, name: str) -> numpy.ndarray:
    scores = numpy.asarray(values, dtype=float)
    if scores.ndim not in (1, 2) or not scores.shape[0]:
        raise ValueError(f"{name}는 비어 있지 않은 1차원 또는 2차원 배열이어야 한다")
    if scores.ndim == 2 and not scores.shape[1]:
        raise ValueError(f"{name}의 채널 축이 비어 있다")
    if not numpy.isfinite(scores).all():
        raise ValueError(f"{name}는 모두 finite여야 한다")
    return scores


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
    *,
    native_postprocessing: bool = False,
) -> list[str]:
    """정규화된 scalar 또는 채널 점수의 raw·smoothed 배열을 저장한다.

    scalar는 2개 파일, 채널 점수는 max 집계와 `__channels`를 합쳐 4개 파일이다.
    """
    channel_scores = validate_scores(channel_scores, "scores")
    smoothed_channel_scores = (channel_scores.copy() if native_postprocessing else
                               trailing_average_smoothing(channel_scores, window=smoothing_window))
    validate_scores(smoothed_channel_scores, "smoothed_scores")

    naming_arguments = {
        "dataset": dataset, "series": series, "model": model, "tier": tier,
        "ratio": ratio, "seed": seed, "norm_kind": norm_kind,
    }
    if channel_scores.ndim == 1:
        arrays_by_name = {
            build_score_filename(smoothing_kind="raw", channels=False, **naming_arguments): channel_scores,
            build_score_filename(smoothing_kind="smoothed", channels=False, **naming_arguments): smoothed_channel_scores,
        }
    else:
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
        temporary_path = os.path.join(output_dir, f".{filename}.tmp")
        try:
            with open(temporary_path, "wb") as destination:
                numpy.save(destination, array)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        saved_paths.append(path)
    return saved_paths


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
