"""Tier 2 모델 결과를 공통 점수·metadata·실행 기록으로 저장한다."""

import json
from pathlib import Path

import torch

from src.common.save_scores import (
    save_score_bundle,
    save_score_metadata,
    snapshot_config,
    validate_channel_scores,
)


def _score_range(model: str, test_length: int) -> tuple[int, int, str]:
    if model == "CI-AE":
        return 50, test_length - 49, "window_center_right"
    if model == "LSTM-AD":
        return 100, test_length, "next_step"
    if model == "USAD":
        return 5, test_length - 4, "window_center_right"
    if model == "GDN":
        return 5, test_length, "next_step"
    raise ValueError(f"지원하지 않는 Tier 2 모델이다: {model}")


def save_tier2_artifacts(
    result: dict,
    output_dir,
    dataset: str,
    test_series,
    model: str,
    ratio: int,
    seed: int,
    test_lengths,
    run_config: dict,
    source_identity: dict,
    input_metadata: dict,
    epsilon: float,
    smoothing_window: int,
) -> dict:
    """모든 채널 오차를 먼저 검증한 뒤 한 실행의 산출물을 저장한다."""
    reference_errors = tuple(result["reference_errors"])
    test_errors = tuple(result["test_errors"])
    test_series = tuple(test_series)
    test_lengths = tuple(test_lengths)
    if not reference_errors or len(test_errors) != len(test_series) or len(test_errors) != len(test_lengths):
        raise ValueError("reference 또는 test 세션 수가 산출 계약과 다르다")

    matrices = [
        validate_channel_scores(errors, "reference_errors")
        for errors in reference_errors
    ] + [
        validate_channel_scores(errors, "test_errors")
        for errors in test_errors
    ]
    if len({matrix.shape[1] for matrix in matrices}) != 1:
        raise ValueError("reference와 test 오차의 채널 수가 다르다")
    ranges = []
    for errors, test_length in zip(test_errors, test_lengths):
        source_start, source_end, alignment = _score_range(model, test_length)
        if len(errors) != source_end - source_start:
            raise ValueError(f"{model} 점수 길이가 source 범위와 다르다")
        ranges.append((source_start, source_end, alignment))

    output_dir = Path(output_dir)
    checkpoint_path = (
        output_dir / "training" / model.lower().replace("-", "_") / "checkpoint.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result["checkpoint"], checkpoint_path)

    scores_dir = output_dir / "scores"
    score_paths = []
    metadata_paths = []
    for series, errors, test_length, score_range in zip(
        test_series, test_errors, test_lengths, ranges,
    ):
        score_paths.extend(save_score_bundle(
            reference_errors, errors, str(scores_dir), dataset, series, model,
            "t2", ratio, seed, epsilon, smoothing_window,
        ))
        source_start, source_end, alignment = score_range
        metadata_paths.append(save_score_metadata(
            str(scores_dir), dataset, series, model, "t2", ratio, seed,
            run_config["window_size"], test_length, len(errors),
            (source_start, source_end), source_start, source_end, alignment,
        ))

    early_stopping_path = output_dir / "early_stopping_log.json"
    timing_path = output_dir / "timing.json"
    early_stopping_path.write_text(
        json.dumps(result["training_log"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    timing_path.write_text(
        json.dumps(result["timing"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    snapshot_path = snapshot_config({
        "run_config": run_config,
        "seed": seed,
        "source_identity": source_identity,
        "input": input_metadata,
    }, source_identity["project_commit"], str(output_dir / "snapshots"))
    return {
        "checkpoint_path": str(checkpoint_path),
        "score_paths": score_paths,
        "metadata_paths": metadata_paths,
        "early_stopping_log_path": str(early_stopping_path),
        "timing_path": str(timing_path),
        "snapshot_path": snapshot_path,
    }
