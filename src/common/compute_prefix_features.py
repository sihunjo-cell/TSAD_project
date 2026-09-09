"""모델별 전처리 전 float32 prefix의 최소 통계를 계산한다."""

import hashlib
import json
import math
import re

import numpy


EXTRACTOR_CONTRACT = {
    "version": "prefix_features.v2", "input_dtype": "float32", "calculation_dtype": "float64",
    "standard_deviation_ddof": 0, "quantile_method": "linear",
    "interquartile_range": "finite_Q75-Q25; minimum_finite_count=1; zero_is_valid",
    "difference_q90_iqr_ratio": "Q90(abs(diff(original_channel)))/interquartile_range; minimum_rows=2",
    "median_shift_iqr_ratio": "abs(median(column[floor(n/2):])-median(column[:floor(n/2)]))/interquartile_range; minimum_rows=4",
    "ratio_requirements": "fully_finite_original_channel; positive_interquartile_range; no_time_squeezing",
    "spectral_entropy": "-sum(p[p>0]*log(p[p>0]))/log(m); clipped_to_[0,1]; minimum_rows=4; fully_finite_nonconstant_channel",
    "spectral_power": "abs(rfft(column-mean))[1:]**2; no_window_or_padding; no_bin_doubling; m=all_non_DC_bins>1; p=power/sum(power)",
    "spectral_power_scaling": "divide_positive_bin_magnitudes_by_their_maximum_before_squaring",
    "temporal_null_priority": "too_few_rows_then_nonfinite_input_then_zero_interquartile_range_or_constant_channel",
    "channel_summary": "linear_median_of_defined_channel_values_with_valid_channel_count",
    "acf": "adjacent_centered_product_sum/full_centered_square_sum",
    "correlation": "absolute_pearson_unique_fully_finite_nonconstant_pairs; finite_values_clipped_to_[0,1]",
}


def prefix_feature_identity(*, csv_id, source_sha256, ordered_schema_sha256,
                            q_percent, source_start, source_end_exclusive) -> dict:
    """시각·모델·seed와 무관한 입력 범위와 산식 신원을 만든다."""
    if not isinstance(csv_id, str) or not csv_id:
        raise ValueError("prefix CSV ID is required")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in (source_sha256, ordered_schema_sha256)):
        raise ValueError("prefix source and ordered schema SHA-256 are required")
    if type(q_percent) is not int or q_percent not in (5, 10, 20, 40, 60, 80, 100):
        raise ValueError("prefix q must use the registered percentages")
    if (type(source_start) is not int or type(source_end_exclusive) is not int
            or not 0 <= source_start <= source_end_exclusive):
        raise ValueError("invalid prefix source range")
    identity = dict(csv_id=csv_id, source_sha256=source_sha256,
                    ordered_schema_sha256=ordered_schema_sha256, q_percent=q_percent,
                    source_start=source_start, source_end_exclusive=source_end_exclusive,
                    extractor=EXTRACTOR_CONTRACT)
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"prefix_feature_id": "p" + hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "identity_json": serialized}


def _finite_statistic(value, reason=None):
    if value is None:
        return None, reason
    value = float(value)
    return (0.0 if value == 0 else value, None) if math.isfinite(value) else (None, "nonfinite_computation")


def _median(values):
    if not values:
        return None, "no_valid_values"
    return _finite_statistic(numpy.quantile(values, 0.5, method="linear"))


def compute_prefix_features(prefix, channel_names) -> tuple[dict, list[dict]]:
    """현재 prefix만 읽고 유효 통계와 미정의 사유를 반환한다."""
    if not isinstance(prefix, numpy.ndarray) or prefix.ndim != 2 or prefix.dtype != numpy.float32:
        raise ValueError("prefix must be a two-dimensional float32 loader array")
    channel_names = tuple(channel_names)
    if len(channel_names) != prefix.shape[1] or not channel_names or any(
        not isinstance(name, str) for name in channel_names
    ):
        raise ValueError("ordered channel names must match the loader array")
    values = prefix.astype(numpy.float64, copy=True)
    channels, correlation_columns, square_sums = [], [], {}
    with numpy.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for index, name in enumerate(channel_names):
            column = values[:, index]
            finite = column[numpy.isfinite(column)]
            count = len(finite)
            constant = int(numpy.all(finite == finite[0])) if count else None
            channel = {"channel_index": index, "channel_name": name,
                       "valid_value_count": count, "is_constant": constant,
                       "constant_reason": None if count else "no_finite_values"}
            quartiles = numpy.quantile(finite, [0.25, 0.75], method="linear") if count else None
            for field, value in (("mean", numpy.mean(finite) if count else None),
                                 ("std", numpy.std(finite, ddof=0) if count else None),
                                 ("median", numpy.quantile(finite, 0.5, method="linear") if count else None),
                                 ("interquartile_range", quartiles[1] - quartiles[0] if count else None)):
                channel[field], channel[field + "_reason"] = _finite_statistic(value, "no_finite_values")
            interquartile_range = channel["interquartile_range"]
            difference_reason = ("fewer_than_two_rows" if len(column) < 2 else
                                 "nonfinite_input" if count != len(column) else
                                 "zero_interquartile_range" if interquartile_range == 0 else None)
            difference_ratio = (numpy.quantile(numpy.abs(numpy.diff(column)), 0.9, method="linear")
                                / interquartile_range if difference_reason is None else None)
            channel["difference_q90_iqr_ratio"], channel["difference_q90_iqr_ratio_reason"] = _finite_statistic(
                difference_ratio, difference_reason,
            )
            four_row_reason = ("fewer_than_four_rows" if len(column) < 4 else
                               "nonfinite_input" if count != len(column) else None)
            shift_reason = four_row_reason or ("zero_interquartile_range" if interquartile_range == 0 else None)
            split = len(column) // 2
            shift_ratio = (abs(numpy.median(column[split:]) - numpy.median(column[:split]))
                           / interquartile_range if shift_reason is None else None)
            channel["median_shift_iqr_ratio"], channel["median_shift_iqr_ratio_reason"] = _finite_statistic(
                shift_ratio, shift_reason,
            )
            entropy_reason = four_row_reason or ("constant_channel" if constant else None)
            entropy = None
            if entropy_reason is None:
                magnitudes = numpy.abs(numpy.fft.rfft(column - channel["mean"]))[1:]
                power = (magnitudes / numpy.max(magnitudes)) ** 2
                probabilities = power / numpy.sum(power)
                if numpy.all(numpy.isfinite(probabilities)):
                    positive = probabilities[probabilities > 0]
                    entropy = numpy.clip(-numpy.sum(positive * numpy.log(positive)) / numpy.log(len(power)), 0.0, 1.0)
                else:
                    entropy_reason = "nonfinite_computation"
            channel["spectral_entropy"], channel["spectral_entropy_reason"] = _finite_statistic(entropy, entropy_reason)
            reason = ("fewer_than_two_rows" if len(column) < 2 else
                      "nonfinite_input" if count != len(column) else
                      "constant_channel" if constant else None)
            autocorrelation = None
            if reason is None:
                column -= channel["mean"]
                square_sum = float(numpy.dot(column, column))
                if not math.isfinite(square_sum) or square_sum <= 0:
                    reason = "nonfinite_computation"
                else:
                    autocorrelation = numpy.dot(column[:-1], column[1:]) / square_sum
                    correlation_columns.append(index)
                    square_sums[index] = square_sum
            channel["acf_lag1"], channel["acf_lag1_reason"] = _finite_statistic(autocorrelation, reason)
            channels.append(channel)
        correlations = []
        for position, first in enumerate(correlation_columns):
            for second in correlation_columns[position + 1:]:
                correlation, _ = _finite_statistic(abs(numpy.dot(values[:, first], values[:, second])
                    / math.sqrt(square_sums[first]) / math.sqrt(square_sums[second])))
                if correlation is not None:
                    correlations.append(min(correlation, 1.0))
    evaluable = [channel for channel in channels if channel["is_constant"] is not None]
    summary = {
        "constant_channel_count": sum(channel["is_constant"] for channel in evaluable) if evaluable else None,
        "constant_evaluable_channel_count": len(evaluable),
        "constant_channel_count_reason": ("no_evaluable_channels" if not evaluable else
                                          "partially_evaluable_channels" if len(evaluable) < len(channels) else None),
    }
    for source in ("std", "acf_lag1", "interquartile_range", "difference_q90_iqr_ratio",
                   "median_shift_iqr_ratio", "spectral_entropy"):
        field = "channel_" + source
        valid = [channel[source] for channel in channels if channel[source] is not None]
        summary[field + "_median"], summary[field + "_median_reason"] = _median(valid)
        summary[field + "_valid_channel_count"] = len(valid)
    summary["absolute_correlation_median"], summary["absolute_correlation_median_reason"] = _median(correlations)
    summary["absolute_correlation_valid_pair_count"] = len(correlations)
    summary["absolute_correlation_eligible_pair_count"] = len(correlation_columns) * (len(correlation_columns) - 1) // 2
    if not correlations and summary["absolute_correlation_eligible_pair_count"]:
        summary["absolute_correlation_median_reason"] = "nonfinite_computation"
    return summary, channels
