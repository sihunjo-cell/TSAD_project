"""라벨 값을 읽지 않고 점수 배열의 예상 시간축만 계산한다."""

import math
from datetime import datetime, timedelta

import numpy


def restore_score_alignment(scores: numpy.ndarray, original_length: int, first_valid_index: int) -> numpy.ndarray:
    scores = numpy.asarray(scores)
    if first_valid_index < 0 or first_valid_index + len(scores) > original_length:
        raise ValueError("점수 길이와 offset이 원 시간축 범위를 벗어납니다.")
    restored = numpy.zeros(original_length, dtype=scores.dtype)
    restored[first_valid_index:first_valid_index + len(scores)] = scores
    return restored


def compute_core_score_length(test_length: int, candidate: dict) -> int:
    return max(
        test_length - candidate["model_input_window_size"] - candidate["target_horizon"] + 1,
        0,
    )


def build_score_alignment_rows(
    dataset: str,
    source: str,
    test_length: int,
    candidates: tuple[dict, ...],
    timestamp_start: str = "",
    sampling_interval_seconds: float | None = None,
) -> list[dict]:
    rows = []
    for candidate in candidates:
        model = candidate["model"]
        window_size = candidate["model_input_window_size"]
        core_length = compute_core_score_length(test_length, candidate)
        if model in {"CI-AE", "USAD"}:
            source_start = math.ceil((window_size - 1) / 2)
            raw_left_padding = source_start
            raw_right_padding = math.floor((window_size - 1) / 2)
            raw_wrapper_length = test_length
            alignment_rule = "window reconstruction score를 중앙 시점에 맞춘 core만 저장"
        elif model == "LSTM-AD":
            source_start = window_size
            raw_left_padding = window_size
            raw_right_padding = 0
            raw_wrapper_length = test_length
            alignment_rule = "1-step target인 labels[W:]와 core를 직접 대응"
        else:
            source_start = window_size
            raw_left_padding = 0
            raw_right_padding = 0
            raw_wrapper_length = core_length
            alignment_rule = "1-step target인 labels[W:]와 core를 직접 대응"

        first_timestamp = ""
        if timestamp_start and sampling_interval_seconds is not None:
            first_timestamp = (
                datetime.fromisoformat(timestamp_start)
                + timedelta(seconds=source_start * sampling_interval_seconds)
            ).isoformat()
        source_end = source_start + core_length
        rows.append({
            "dataset": dataset,
            "file_or_session": source,
            "model": model,
            "test_length": test_length,
            "model_input_window_size": window_size,
            "target_horizon": candidate["target_horizon"],
            "first_valid_score_index": source_start,
            "source_time_index_of_first_core_score": source_start,
            "source_time_index_after_last_core_score": source_end,
            "source_timestamp_of_first_core_score": first_timestamp,
            "core_score_length": core_length,
            "expected_saved_score_length": core_length,
            "project_saved_axis": f"source[{source_start}:{source_end}]",
            "raw_wrapper_left_padding_count": raw_left_padding,
            "raw_wrapper_right_padding_count": raw_right_padding,
            "raw_wrapper_saved_score_length": raw_wrapper_length,
            "project_padding_count": 0,
            "alignment_rule": alignment_rule,
            "alignment_valid": source_end <= test_length and core_length >= 0,
            "label_values_used": False,
        })
    return rows
