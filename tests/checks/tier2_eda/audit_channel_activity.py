"""학습 구간 scope별로 constant한 파일·채널 조합을 센다."""

from collections import defaultdict

import numpy


def is_constant_series(values: numpy.ndarray) -> bool:
    values = numpy.asarray(values)
    return bool(values.size and numpy.all(values == values.flat[0]))


def summarize_channel_activity(
    values: numpy.ndarray,
    channel_names: list[str] | tuple[str, ...],
    ratio: int,
    dataset: str = "",
    file_or_session: str = "",
    scope: str = "fit_subset",
) -> list[dict]:
    values = numpy.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(channel_names):
        raise ValueError("채널 활동성 입력은 채널 수가 맞는 2차원 배열이어야 합니다.")

    rows = []
    for index, channel in enumerate(channel_names):
        channel_values = values[:, index]
        finite_mask = numpy.isfinite(channel_values)
        finite_values = channel_values[finite_mask]
        nan_count = int(numpy.isnan(channel_values).sum())
        infinite_count = int(numpy.isinf(channel_values).sum())
        unique_value_count = int(numpy.unique(finite_values).size)
        minimum = float(finite_values.min()) if finite_values.size else numpy.nan
        maximum = float(finite_values.max()) if finite_values.size else numpy.nan
        value_range = maximum - minimum
        constant = bool(finite_values.size and finite_mask.all() and unique_value_count == 1)
        variance = (
            0.0 if constant else float(numpy.var(finite_values))
            if finite_values.size else numpy.nan
        )
        iqr = (
            float(numpy.quantile(finite_values, 0.75) - numpy.quantile(finite_values, 0.25))
            if finite_values.size else numpy.nan
        )
        if finite_mask.all() and not (constant == (unique_value_count == 1) == (value_range == 0.0)):
            raise ValueError(f"constant 판정 조건이 일치하지 않습니다: {file_or_session}/{channel}/{scope}")
        nonfinite = bool(nan_count or infinite_count)
        zero_range = bool(finite_values.size and value_range == 0.0)
        scaler_execution_ready = bool(channel_values.size and not nonfinite)
        rows.append({
            "dataset": dataset,
            "file_or_session": file_or_session,
            "ratio": ratio,
            "stage": "normal",
            "scope": scope,
            "channel": channel,
            "sample_count": len(channel_values),
            "unique_value_count": unique_value_count,
            "minimum": minimum,
            "maximum": maximum,
            "variance": variance,
            "variance_zero": variance == 0.0,
            "iqr": iqr,
            "iqr_zero": iqr == 0.0,
            "range": value_range,
            "zero_range": zero_range,
            "constant": constant,
            "nan_count": nan_count,
            "infinite_count": infinite_count,
            "nonfinite_count": nan_count + infinite_count,
            "nonfinite": nonfinite,
            "finite": not nonfinite,
            "scaler_execution_ready": scaler_execution_ready,
            "variation_available": not constant,
        })
    return rows


def build_scope_summary_rows(activity_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in activity_rows:
        grouped[(row["dataset"], row["ratio"], row["scope"])].append(row)

    summaries = []
    for (dataset, ratio, scope), rows in sorted(grouped.items()):
        constant_count = sum(bool(row["constant"]) for row in rows)
        by_source = defaultdict(int)
        for row in rows:
            by_source[row["file_or_session"]] += int(row["constant"])
        pooled_constant_count = "not_applicable"
        combination_unit = "series_channel"
        if dataset == "HAI":
            combination_unit = "training_session_channel"
            by_channel = defaultdict(list)
            for row in rows:
                by_channel[row["channel"]].append(row)
            pooled_constant_count = sum(
                all(row["finite"] for row in channel_rows)
                and max(row["maximum"] for row in channel_rows)
                == min(row["minimum"] for row in channel_rows)
                for channel_rows in by_channel.values()
            )
        summaries.append({
            "dataset": dataset,
            "ratio": ratio,
            "scope": scope,
            "total_file_channel_combinations": len(rows),
            "combination_unit": combination_unit,
            "constant_file_channel_combinations": constant_count,
            "constant_by_source": ";".join(
                f"{source}={count}" for source, count in sorted(by_source.items())
            ),
            "pooled_constant_channel_count": pooled_constant_count,
            "iqr_zero_file_channel_combinations": sum(bool(row["iqr_zero"]) for row in rows),
            "variance_zero_file_channel_combinations": sum(
                bool(row["variance_zero"]) for row in rows
            ),
            "zero_range_file_channel_combinations": sum(
                bool(row["zero_range"]) for row in rows
            ),
            "nonfinite_file_channel_combinations": sum(
                bool(row["nonfinite"]) for row in rows
            ),
            "scaler_execution_not_ready_file_channel_combinations": sum(
                not bool(row["scaler_execution_ready"]) for row in rows
            ),
            "variation_available_file_channel_combinations": sum(
                bool(row["variation_available"]) for row in rows
            ),
            "nan_file_channel_combinations": sum(row["nan_count"] > 0 for row in rows),
            "infinite_file_channel_combinations": sum(
                row["infinite_count"] > 0 for row in rows
            ),
        })
    return summaries


def validate_nested_constant_sets(activity_rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in activity_rows:
        if row["scope"] == "fit_subset":
            key = (row["dataset"], row["file_or_session"], row["channel"], row["scope"])
            grouped[key].append(row)

    for key, rows in grouped.items():
        rows.sort(key=lambda row: row["ratio"])
        for earlier, later in zip(rows, rows[1:]):
            if not earlier["constant"] and later["constant"]:
                raise ValueError(
                    f"nested scope에서 active였던 조합이 constant로 바뀌었습니다: {key}, "
                    f"{earlier['ratio']}%->{later['ratio']}%"
                )
