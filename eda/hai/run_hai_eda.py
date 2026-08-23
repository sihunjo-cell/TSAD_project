"""Reproducible exploratory data analysis for the local HAI-23.05 files.

The script intentionally relies only on information present in the distributed
files.  It does not attach domain meanings to tags or use assumptions from the
paper.  Training files are therefore called "unlabelled train" rather than
"normal" throughout the outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


TIMESTAMP_COLUMN = "timestamp"
LABEL_COLUMN = "label"
AUTOCORRELATION_LAGS = (1, 10, 60, 300)
CARDINALITY_CUTOFF = 200
EDA_SAMPLE_FRACTION = 0.02
EDA_MINIMUM_SAMPLE_ROWS = 1000


@dataclass
class RunningStats:
    """Exact streaming first/second moments and data-quality counters."""

    features: Sequence[str]

    def __post_init__(self) -> None:
        width = len(self.features)
        self.rows = 0
        self.count = np.zeros(width, dtype=np.int64)
        self.missing = np.zeros(width, dtype=np.int64)
        self.infinite = np.zeros(width, dtype=np.int64)
        self.zeros = np.zeros(width, dtype=np.int64)
        self.mean_values = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        values = frame.loc[:, self.features].to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        self.rows += len(frame)
        batch_count = finite.sum(axis=0).astype(np.int64)
        old_count = self.count.copy()
        self.missing += np.isnan(values).sum(axis=0)
        self.infinite += np.isinf(values).sum(axis=0)
        self.zeros += ((values == 0) & finite).sum(axis=0)

        clean = np.where(finite, values, np.nan)
        with np.errstate(all="ignore"):
            batch_mean = np.nanmean(clean, axis=0)
            batch_variance = np.nanvar(clean, axis=0, ddof=0)
            current_min = np.nanmin(clean, axis=0)
            current_max = np.nanmax(clean, axis=0)
        batch_m2 = batch_variance * batch_count
        new_count = old_count + batch_count
        valid = batch_count > 0
        delta = np.zeros_like(self.mean_values)
        delta[valid] = batch_mean[valid] - self.mean_values[valid]
        self.mean_values[valid] += (
            delta[valid] * batch_count[valid] / new_count[valid]
        )
        self.m2[valid] += batch_m2[valid] + (
            delta[valid] ** 2
            * old_count[valid]
            * batch_count[valid]
            / new_count[valid]
        )
        self.count = new_count
        self.minimum = np.fmin(self.minimum, current_min)
        self.maximum = np.fmax(self.maximum, current_max)

    def finish(self, group: str) -> pd.DataFrame:
        count = self.count.astype(np.float64)
        mean = self.mean_values.copy()
        mean[count == 0] = np.nan
        variance = np.divide(
            np.maximum(self.m2, 0.0),
            count - 1,
            out=np.full_like(self.mean_values, np.nan),
            where=count > 1,
        )
        minimum = self.minimum.copy()
        maximum = self.maximum.copy()
        minimum[~np.isfinite(minimum)] = np.nan
        maximum[~np.isfinite(maximum)] = np.nan
        exactly_constant = (
            np.isfinite(minimum) & np.isfinite(maximum) & (minimum == maximum)
        )
        variance[exactly_constant] = 0.0
        denominator = max(self.rows, 1)
        return pd.DataFrame(
            {
                "group": group,
                "feature": self.features,
                "rows": self.rows,
                "finite_count": self.count,
                "missing_count": self.missing,
                "missing_rate": self.missing / denominator,
                "infinite_count": self.infinite,
                "infinite_rate": self.infinite / denominator,
                "zero_count": self.zeros,
                "zero_rate": self.zeros / denominator,
                "mean": mean,
                "std": np.sqrt(variance),
                "min": minimum,
                "max": maximum,
            }
        )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parents[1]
    parser = argparse.ArgumentParser(description="Run EDA for HAI-23.05")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "HAI-23.05",
        help="Directory containing HAI CSV and summary files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
        help="Directory for generated tables, figures, and report",
    )
    parser.add_argument(
        "--sample-per-file",
        type=int,
        default=5000,
        help="Maximum rows sampled from each file and label group for plots",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def natural_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    return int(match.group(1)) if match else 0


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def seconds_to_hms(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    seconds = int(round(value))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def ensure_numeric(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    numeric = frame.loc[:, features]
    object_columns = numeric.select_dtypes(exclude=[np.number]).columns
    if len(object_columns):
        numeric = numeric.copy()
        for column in object_columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def timestamp_audit(timestamps: pd.Series) -> Dict[str, object]:
    valid = timestamps.dropna()
    result: Dict[str, object] = {
        "timestamp_missing_count": int(timestamps.isna().sum()),
        "timestamp_duplicate_count": int(timestamps.duplicated().sum()),
        "timestamp_monotonic_increasing": bool(timestamps.is_monotonic_increasing),
        "start_time": valid.min() if not valid.empty else pd.NaT,
        "end_time": valid.max() if not valid.empty else pd.NaT,
        "duration_seconds": np.nan,
        "expected_interval_seconds": np.nan,
        "irregular_interval_count": 0,
        "nonpositive_interval_count": 0,
        "estimated_missing_intervals": 0,
        "max_interval_seconds": np.nan,
    }
    if len(valid) < 2:
        return result
    delta = timestamps.diff().dt.total_seconds().dropna()
    positive = delta[delta > 0]
    expected = positive.mode().iloc[0] if not positive.empty else np.nan
    result["duration_seconds"] = float((valid.max() - valid.min()).total_seconds())
    result["expected_interval_seconds"] = float(expected)
    result["nonpositive_interval_count"] = int((delta <= 0).sum())
    result["max_interval_seconds"] = float(delta.max())
    if np.isfinite(expected) and expected > 0:
        tolerance = max(1e-9, expected * 1e-6)
        result["irregular_interval_count"] = int(
            (~np.isclose(delta.to_numpy(), expected, atol=tolerance, rtol=0)).sum()
        )
        large = delta[delta > expected + tolerance]
        result["estimated_missing_intervals"] = int(
            np.maximum(np.rint(large / expected).astype(np.int64) - 1, 0).sum()
        )
    return result


def update_cardinality(
    trackers: Dict[str, Optional[set]], numeric: pd.DataFrame, features: Sequence[str]
) -> None:
    for feature in features:
        current = trackers[feature]
        if current is None:
            continue
        values = pd.unique(numeric[feature].dropna())
        finite_values = values[np.isfinite(values)]
        current.update(finite_values.tolist())
        if len(current) > CARDINALITY_CUTOFF:
            trackers[feature] = None


def classify_cardinality(tracker: Optional[set]) -> Tuple[str, int, bool]:
    if tracker is None:
        return "high_cardinality", CARDINALITY_CUTOFF + 1, False
    count = len(tracker)
    if count <= 1:
        category = "constant"
    elif count == 2:
        category = "binary"
    elif count <= 20:
        category = "low_cardinality"
    else:
        category = "medium_cardinality"
    return category, count, True


def feature_prefix(feature: str) -> str:
    return feature.split("_", 1)[0] if "_" in feature else feature


def sample_rows(
    numeric: pd.DataFrame,
    group: str,
    source_file: str,
    limit: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if numeric.empty or limit <= 0:
        return pd.DataFrame()
    count = min(len(numeric), limit)
    if count == len(numeric):
        positions = np.arange(len(numeric))
    else:
        positions = np.sort(rng.choice(len(numeric), size=count, replace=False))
    sample = numeric.iloc[positions].copy()
    sample.insert(0, "source_file", source_file)
    sample.insert(0, "group", group)
    return sample


def feature_file_statistics(
    numeric: pd.DataFrame, source_file: str, split: str
) -> pd.DataFrame:
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    infinite = np.isinf(values).sum(axis=0)
    clean = numeric.replace([np.inf, -np.inf], np.nan)
    description = clean.describe(
        percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]
    ).T.reset_index(names="feature")
    description = description.rename(
        columns={"1%": "q01", "25%": "q25", "50%": "median", "75%": "q75", "99%": "q99"}
    )
    description.insert(0, "split", split)
    description.insert(0, "source_file", source_file)
    description["missing_count"] = numeric.isna().sum().to_numpy()
    description["missing_rate"] = description["missing_count"] / max(len(numeric), 1)
    description["infinite_count"] = infinite
    description["infinite_rate"] = infinite / max(len(numeric), 1)
    description["zero_count"] = ((values == 0) & np.isfinite(values)).sum(axis=0)
    description["zero_rate"] = description["zero_count"] / max(len(numeric), 1)
    description["n_unique"] = clean.nunique(dropna=True).to_numpy()
    return description


def transition_statistics(
    numeric: pd.DataFrame, source_file: str, split: str, features: Sequence[str]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for feature in features:
        values = numeric[feature].to_numpy(dtype=np.float64, copy=False)
        if len(values) < 2:
            comparable = 0
            changes = 0
        else:
            comparable_mask = np.isfinite(values[:-1]) & np.isfinite(values[1:])
            comparable = int(comparable_mask.sum())
            changes = int(((values[:-1] != values[1:]) & comparable_mask).sum())
        rows.append(
            {
                "source_file": source_file,
                "split": split,
                "feature": feature,
                "comparable_pairs": comparable,
                "transition_count": changes,
                "transition_rate": changes / comparable if comparable else np.nan,
            }
        )
    return rows


def lagged_correlation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return np.nan
    left = values[:-lag]
    right = values[lag:]
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return np.nan
    left = left[mask]
    right = right[mask]
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def autocorrelation_statistics(
    numeric: pd.DataFrame, source_file: str, features: Sequence[str]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for feature in features:
        values = numeric[feature].to_numpy(dtype=np.float64, copy=False)
        for lag in AUTOCORRELATION_LAGS:
            rows.append(
                {
                    "source_file": source_file,
                    "feature": feature,
                    "lag_samples": lag,
                    "autocorrelation": lagged_correlation(values, lag),
                }
            )
    return rows


def extract_events(
    timestamps: pd.Series, labels: pd.Series, source_file: str
) -> List[Dict[str, object]]:
    label_values = pd.to_numeric(labels, errors="coerce").fillna(0).to_numpy()
    mask = label_values == 1
    if not mask.any():
        return []
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    events: List[Dict[str, object]] = []
    for number, (start_idx, end_idx) in enumerate(zip(starts, ends), start=1):
        start_time = timestamps.iloc[start_idx]
        end_time = timestamps.iloc[end_idx]
        span_seconds = (
            float((end_time - start_time).total_seconds())
            if pd.notna(start_time) and pd.notna(end_time)
            else np.nan
        )
        events.append(
            {
                "source_file": source_file,
                "event_number": number,
                "start_index": int(start_idx),
                "end_index": int(end_idx),
                "start_time": start_time,
                "end_time": end_time,
                "n_points": int(end_idx - start_idx + 1),
                "span_seconds": span_seconds,
            }
        )
    return events


SUMMARY_EVENT_PATTERN = re.compile(
    r"\[\s*(\d+)\]\s+"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(\d{2}:\d{2}:\d{2})"
)
SUMMARY_ATTACK_COUNT_PATTERN = re.compile(r"Attacks\s*:\s*(\d+)\s+times")
SUMMARY_TOTAL_PATTERN = re.compile(
    r"Total\s*:\s*(\d{2}:\d{2}:\d{2})\s*\(([\d.]+)\s+percent\)"
)


def hms_to_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def parse_summary_events(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in SUMMARY_EVENT_PATTERN.finditer(text):
        rows.append(
            {
                "event_number": int(match.group(1)),
                "start_time": pd.Timestamp(match.group(2)),
                "end_time": pd.Timestamp(match.group(3)),
                "reported_duration": match.group(4),
            }
        )
    return rows


def parse_summary_header(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    attack_match = SUMMARY_ATTACK_COUNT_PATTERN.search(text)
    total_match = SUMMARY_TOTAL_PATTERN.search(text)
    return {
        "summary_header_attack_count": int(attack_match.group(1)) if attack_match else np.nan,
        "summary_header_total_hms": total_match.group(1) if total_match else "",
        "summary_header_total_seconds": hms_to_seconds(total_match.group(1)) if total_match else np.nan,
        "summary_header_total_percent": float(total_match.group(2)) if total_match else np.nan,
    }


def compare_summary_events(
    actual: Sequence[Dict[str, object]],
    reported: Sequence[Dict[str, object]],
    header: Dict[str, object],
) -> Dict[str, object]:
    count_match = len(actual) == len(reported)
    boundary_match = count_match and all(
        pd.Timestamp(a["start_time"]) == pd.Timestamp(r["start_time"])
        and pd.Timestamp(a["end_time"]) == pd.Timestamp(r["end_time"])
        for a, r in zip(actual, reported)
    )
    duration_match = count_match and all(
        seconds_to_hms(float(a["span_seconds"]))[-8:] == r["reported_duration"]
        for a, r in zip(actual, reported)
    )
    actual_total_seconds = float(sum(float(event["span_seconds"]) for event in actual))
    header_count = header["summary_header_attack_count"]
    header_total_seconds = header["summary_header_total_seconds"]
    header_count_match = bool(np.isfinite(header_count) and int(header_count) == len(actual))
    header_total_match = bool(
        np.isfinite(header_total_seconds) and float(header_total_seconds) == actual_total_seconds
    )
    return {
        "actual_event_count": len(actual),
        "summary_table_event_count": len(reported),
        "summary_table_event_count_match": count_match,
        "summary_table_boundaries_match": boundary_match,
        "summary_table_durations_match": duration_match,
        "all_summary_table_events_match": count_match and boundary_match and duration_match,
        "actual_event_span_total_seconds": actual_total_seconds,
        **header,
        "summary_header_attack_count_match": header_count_match,
        "summary_header_total_match": header_total_match,
        "summary_header_aggregate_match": header_count_match and header_total_match,
    }


def event_feature_means(
    numeric: pd.DataFrame,
    events: Sequence[Dict[str, object]],
    features: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for event in events:
        start_idx = int(event["start_index"])
        end_idx = int(event["end_index"])
        length = end_idx - start_idx + 1
        pre_start = max(0, start_idx - length)
        post_end = min(len(numeric), end_idx + 1 + length)
        during = numeric.iloc[start_idx : end_idx + 1]
        before = numeric.iloc[pre_start:start_idx]
        after = numeric.iloc[end_idx + 1 : post_end]
        during_mean = during.replace([np.inf, -np.inf], np.nan).mean()
        before_mean = before.replace([np.inf, -np.inf], np.nan).mean()
        after_mean = after.replace([np.inf, -np.inf], np.nan).mean()
        for feature in features:
            rows.append(
                {
                    "source_file": event["source_file"],
                    "event_number": event["event_number"],
                    "start_time": event["start_time"],
                    "end_time": event["end_time"],
                    "n_points": event["n_points"],
                    "feature": feature,
                    "pre_mean": before_mean[feature],
                    "during_mean": during_mean[feature],
                    "post_mean": after_mean[feature],
                }
            )
    return rows


def first_sustained_true(mask: np.ndarray, required_length: int) -> Optional[int]:
    if required_length <= 0 or len(mask) < required_length:
        return None
    rolling_count = np.convolve(
        mask.astype(np.int16), np.ones(required_length, dtype=np.int16), mode="valid"
    )
    matches = np.flatnonzero(rolling_count >= required_length)
    return int(matches[0]) if len(matches) else None


def event_response_metrics(
    numeric: pd.DataFrame,
    events: Sequence[Dict[str, object]],
    features: Sequence[str],
    normal_std: pd.Series,
    normal_min: pd.Series,
    normal_max: pd.Series,
    reaction_sigma: float = 3.0,
    reaction_sustain_samples: int = 3,
    recovery_sigma: float = 1.0,
    recovery_sustain_samples: int = 10,
    recovery_search_samples: int = 600,
) -> List[Dict[str, object]]:
    """Compute explicitly operational, domain-free event response metrics.

    The local baseline is the median of the 300 rows immediately before an
    event. A continuous response is the first sustained during-event excursion
    beyond ``reaction_sigma`` file-normal SD. A feature that was already beyond
    the threshold immediately before the label boundary is flagged separately.
    When file-normal SD is zero, a sustained change from the pre-window mode is
    used instead. Recovery is searched only until the next event or the stated
    maximum horizon.
    """

    values = numeric.loc[:, features].to_numpy(dtype=np.float64, copy=False)
    normal_std_values = normal_std.reindex(features).to_numpy(dtype=np.float64)
    normal_min_values = normal_min.reindex(features).to_numpy(dtype=np.float64)
    normal_max_values = normal_max.reindex(features).to_numpy(dtype=np.float64)
    rows: List[Dict[str, object]] = []
    next_starts = [
        int(events[index + 1]["start_index"]) if index + 1 < len(events) else len(values)
        for index in range(len(events))
    ]
    for event_index, event in enumerate(events):
        start_idx = int(event["start_index"])
        end_idx = int(event["end_index"])
        pre_start = max(0, start_idx - 300)
        pre_values = values[pre_start:start_idx]
        during_values = values[start_idx : end_idx + 1]
        post_values = values[
            end_idx + 1 : min(
                len(values),
                next_starts[event_index],
                end_idx + 1 + recovery_search_samples,
            )
        ]
        with np.errstate(all="ignore"):
            pre_center = np.nanmedian(pre_values, axis=0)

        for column_number, feature in enumerate(features):
            scale = normal_std_values[column_number]
            center = pre_center[column_number]
            pre_column = pre_values[:, column_number]
            finite_pre = pre_column[np.isfinite(pre_column)]
            if len(finite_pre):
                unique_values, unique_counts = np.unique(finite_pre, return_counts=True)
                pre_mode = float(unique_values[np.argmax(unique_counts)])
            else:
                pre_mode = np.nan
            file_normal_constant = bool(
                np.isfinite(normal_min_values[column_number])
                and np.isfinite(normal_max_values[column_number])
                and normal_min_values[column_number] == normal_max_values[column_number]
            )
            continuous_rule = bool(
                not file_normal_constant
                and np.isfinite(scale)
                and scale > 0
                and np.isfinite(center)
            )
            mode_rule = bool(file_normal_constant and np.isfinite(pre_mode))
            threshold_defined = continuous_rule or mode_rule
            response_rule = "3_sigma_from_pre_median" if continuous_rule else (
                "change_from_pre_mode" if mode_rule else "undefined"
            )
            reaction_delay = np.nan
            recovery_delay = np.nan
            peak_deviation_sd = np.nan
            response_detected = False
            recovery_detected = False
            preexisting_excursion = False
            recovery_not_observed_within_horizon = False
            if threshold_defined:
                during_column = during_values[:, column_number]
                finite_during = np.isfinite(during_column)
                recent_pre = pre_column[-reaction_sustain_samples:]
                if continuous_rule:
                    deviation_sd = np.abs(during_column - center) / scale
                    if finite_during.any():
                        peak_deviation_sd = float(np.nanmax(deviation_sd))
                    reaction_mask = finite_during & (deviation_sd > reaction_sigma)
                    preexisting_excursion = bool(
                        len(recent_pre) == reaction_sustain_samples
                        and np.isfinite(recent_pre).all()
                        and (np.abs(recent_pre - center) > reaction_sigma * scale).all()
                    )
                else:
                    reaction_mask = finite_during & (during_column != pre_mode)
                    preexisting_excursion = bool(
                        len(recent_pre) == reaction_sustain_samples
                        and np.isfinite(recent_pre).all()
                        and (recent_pre != pre_mode).all()
                    )
                reaction_start = None if preexisting_excursion else first_sustained_true(
                    reaction_mask, reaction_sustain_samples
                )
                if reaction_start is not None:
                    response_detected = True
                    reaction_delay = float(reaction_start)
                    post_column = post_values[:, column_number]
                    if continuous_rule:
                        recovery_mask = np.isfinite(post_column) & (
                            np.abs(post_column - center) <= recovery_sigma * scale
                        )
                    else:
                        recovery_mask = np.isfinite(post_column) & (post_column == pre_mode)
                    recovery_start = first_sustained_true(
                        recovery_mask, recovery_sustain_samples
                    )
                    if recovery_start is not None:
                        recovery_detected = True
                        # The first post-event row is one second after event end.
                        recovery_delay = float(recovery_start + 1)
                    else:
                        recovery_not_observed_within_horizon = True
            rows.append(
                {
                    "source_file": event["source_file"],
                    "event_number": event["event_number"],
                    "feature": feature,
                    "threshold_defined": threshold_defined,
                    "response_rule": response_rule,
                    "pre_event_median": center,
                    "pre_event_mode": pre_mode,
                    "pre_event_window_samples": len(pre_values),
                    "file_normal_std": scale,
                    "file_normal_constant": file_normal_constant,
                    "reaction_threshold_sigma": reaction_sigma,
                    "reaction_sustain_samples": reaction_sustain_samples,
                    "response_detected": response_detected,
                    "preexisting_excursion": preexisting_excursion,
                    "reaction_delay_seconds": reaction_delay,
                    "peak_abs_deviation_from_pre_median_sd": peak_deviation_sd,
                    "recovery_threshold_sigma": recovery_sigma,
                    "recovery_sustain_samples": recovery_sustain_samples,
                    "recovery_search_seconds": recovery_search_samples,
                    "recovery_search_actual_samples": len(post_values),
                    "recovery_detected": recovery_detected,
                    "recovery_not_observed_within_horizon": recovery_not_observed_within_horizon,
                    "recovery_right_censored": recovery_not_observed_within_horizon,
                    "recovery_delay_seconds": recovery_delay,
                }
            )
    return rows


def add_sample_quantiles(
    summaries: pd.DataFrame,
    samples: Dict[str, pd.DataFrame],
    features: Sequence[str],
) -> pd.DataFrame:
    pieces = []
    for group, group_summary in summaries.groupby("group", sort=False):
        piece = group_summary.copy()
        sample = samples.get(group, pd.DataFrame())
        if not sample.empty:
            quantiles = sample.loc[:, features].replace([np.inf, -np.inf], np.nan).quantile(
                [0.01, 0.25, 0.50, 0.75, 0.99]
            ).T
            quantiles.columns = ["sample_q01", "sample_q25", "sample_median", "sample_q75", "sample_q99"]
            piece = piece.merge(quantiles, left_on="feature", right_index=True, how="left")
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def comparison_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_sample: pd.DataFrame,
    right_sample: pd.DataFrame,
    features: Sequence[str],
    left_name: str,
    right_name: str,
    ks_basis: str,
) -> pd.DataFrame:
    left_index = left.set_index("feature")
    right_index = right.set_index("feature")
    rows = []
    for feature in features:
        left_mean = float(left_index.at[feature, "mean"])
        right_mean = float(right_index.at[feature, "mean"])
        left_std = float(left_index.at[feature, "std"])
        left_constant = bool(
            np.isfinite(left_index.at[feature, "min"])
            and np.isfinite(left_index.at[feature, "max"])
            and left_index.at[feature, "min"] == left_index.at[feature, "max"]
        )
        raw_absolute_difference = abs(right_mean - left_mean)
        means_numerically_equal = bool(
            np.isclose(
                left_mean,
                right_mean,
                rtol=1e-12,
                atol=1e-12 * max(1.0, abs(left_mean), abs(right_mean)),
            )
        )
        absolute_difference = 0.0 if means_numerically_equal else raw_absolute_difference
        if not left_constant and np.isfinite(left_std) and left_std > 0:
            standardized = absolute_difference / left_std
            zero_variance_change = False
        elif np.isfinite(absolute_difference) and absolute_difference > 0:
            standardized = np.inf
            zero_variance_change = True
        else:
            standardized = 0.0
            zero_variance_change = False

        left_values = left_sample[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        right_values = right_sample[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(left_values) >= 2 and len(right_values) >= 2:
            ks = ks_2samp(left_values, right_values, alternative="two-sided", method="auto")
            ks_statistic = float(ks.statistic)
            ks_pvalue = float(ks.pvalue)
        else:
            ks_statistic = np.nan
            ks_pvalue = np.nan
        rows.append(
            {
                "feature": feature,
                f"{left_name}_mean": left_mean,
                f"{left_name}_std": left_std,
                f"{right_name}_mean": right_mean,
                f"{right_name}_std": float(right_index.at[feature, "std"]),
                "absolute_mean_difference": absolute_difference,
                f"absolute_mean_difference_in_{left_name}_sd": standardized,
                f"{left_name}_zero_variance_but_mean_changed": zero_variance_change,
                "ks_statistic": ks_statistic,
                "ks_pvalue": ks_pvalue,
                "ks_basis": ks_basis,
            }
        )
    table = pd.DataFrame(rows)
    score_column = f"absolute_mean_difference_in_{left_name}_sd"
    table["ks_rank"] = table["ks_statistic"].rank(
        method="min", ascending=False, na_option="bottom"
    )
    table["standardized_mean_shift_rank"] = table[score_column].rank(
        method="min", ascending=False, na_option="bottom"
    )
    return table.sort_values(
        ["ks_statistic", score_column], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_file_timeline(file_summary: pd.DataFrame, path: Path) -> None:
    ordered = file_summary.sort_values("start_time").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"train_unlabeled": "#4C78A8", "test": "#F58518"}
    for position, row in ordered.iterrows():
        start = pd.Timestamp(row["start_time"])
        end = pd.Timestamp(row["end_time"])
        ax.plot([start, end], [position, position], linewidth=8, solid_capstyle="butt", color=colors[row["split"]])
    ax.set_yticks(range(len(ordered)), ordered["source_file"])
    ax.set_title("HAI file coverage on the timestamp axis")
    ax.set_xlabel("Timestamp")
    ax.grid(axis="x", alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.tight_layout()
    save_figure(fig, path)


def plot_missing_rates(feature_file: pd.DataFrame, path: Path) -> None:
    rates = feature_file.groupby("feature")["missing_rate"].max().sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(11, 6))
    if rates.max() == 0:
        ax.text(0.5, 0.5, "No missing feature values detected", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    else:
        rates.sort_values().plot.barh(ax=ax, color="#E45756")
        ax.set_xlabel("Maximum missing rate across files")
        ax.set_ylabel("")
    ax.set_title("Feature missingness audit")
    save_figure(fig, path)


def plot_label_timeline(
    label_series: Sequence[Tuple[str, pd.Series, pd.Series]], path: Path
) -> None:
    fig, axes = plt.subplots(len(label_series), 1, figsize=(14, 2.4 * len(label_series)), squeeze=False)
    for ax, (name, timestamps, labels) in zip(axes[:, 0], label_series):
        ax.fill_between(timestamps, 0, labels, step="post", color="#E45756", alpha=0.85)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[0, 0].set_title("Test anomaly labels")
    axes[-1, 0].set_xlabel("Timestamp")
    fig.tight_layout()
    save_figure(fig, path)


def plot_event_durations(events: pd.DataFrame, path: Path) -> None:
    frame = events.copy()
    frame["event_id"] = frame["source_file"].str.replace("hai-test", "T", regex=False).str.replace(".csv", "", regex=False) + "-E" + frame["event_number"].astype(str)
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = np.where(frame["source_file"].eq("hai-test1.csv"), "#F58518", "#E45756")
    ax.bar(frame["event_id"], frame["span_seconds"] / 60.0, color=colors)
    ax.set_ylabel("Event span (minutes)")
    ax.set_xlabel("Event")
    ax.set_title("Anomaly event durations")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def finite_plot_scores(frame: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    selected = frame.head(top_n).copy()
    finite = selected.loc[np.isfinite(selected[column]), column]
    replacement = (finite.max() * 1.15) if not finite.empty and finite.max() > 0 else 1.0
    selected["plot_score"] = selected[column].replace([np.inf, -np.inf], replacement).fillna(0)
    return selected


def plot_ranked_effect(
    frame: pd.DataFrame, score_column: str, title: str, xlabel: str, path: Path
) -> None:
    selected = finite_plot_scores(frame, score_column)
    fig, ax = plt.subplots(figsize=(11, 7))
    selected.iloc[::-1].plot.barh(x="feature", y="plot_score", legend=False, ax=ax, color="#4C78A8")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, path)


def plot_correlation(correlation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 13))
    image = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    step = max(1, math.ceil(len(correlation) / 22))
    ticks = np.arange(0, len(correlation), step)
    labels = correlation.index[ticks]
    ax.set_xticks(ticks, labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_title("Unlabelled-train sample Pearson correlation")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    save_figure(fig, path)


def pca_projection(
    samples: Dict[str, pd.DataFrame], features: Sequence[str], table_path: Path, figure_path: Path
) -> Dict[str, object]:
    train = samples["train_unlabeled"].loc[:, features].replace([np.inf, -np.inf], np.nan)
    usable = train.columns[train.notna().mean().gt(0.5) & train.std().gt(0)].tolist()
    medians = train[usable].median()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[usable].fillna(medians))
    pca = PCA(n_components=2, random_state=0)
    pca.fit(train_scaled)
    projected_pieces = []
    for group in ("train_unlabeled", "test_normal", "test_anomaly"):
        frame = samples[group]
        scaled = scaler.transform(frame[usable].replace([np.inf, -np.inf], np.nan).fillna(medians))
        projected = pca.transform(scaled)
        piece = pd.DataFrame({"group": group, "PC1": projected[:, 0], "PC2": projected[:, 1]})
        if len(piece) > 8000:
            piece = piece.sample(8000, random_state=42)
        projected_pieces.append(piece)
    projection = pd.concat(projected_pieces, ignore_index=True)
    write_csv(projection, table_path)

    colors = {"train_unlabeled": "#4C78A8", "test_normal": "#54A24B", "test_anomaly": "#E45756"}
    fig, ax = plt.subplots(figsize=(10, 8))
    for group in ("train_unlabeled", "test_normal", "test_anomaly"):
        piece = projection[projection["group"] == group]
        ax.scatter(piece["PC1"], piece["PC2"], s=5, alpha=0.22, label=group, color=colors[group], rasterized=True)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA fitted on unlabelled-train sample")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)
    save_figure(fig, figure_path)
    return {
        "usable_feature_count": len(usable),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
    }


def plot_top_distributions(
    samples: Dict[str, pd.DataFrame], ranked_features: Sequence[str], path: Path
) -> None:
    features = list(ranked_features[:6])
    normal = samples["test_normal"]
    anomaly = samples["test_anomaly"]
    source_files = sorted(set(normal["source_file"]) | set(anomaly["source_file"]))
    fig, axes = plt.subplots(
        len(features), len(source_files), figsize=(14, 2.5 * len(features)), squeeze=False
    )
    for row_number, feature in enumerate(features):
        for column_number, source_file in enumerate(source_files):
            ax = axes[row_number, column_number]
            left = normal.loc[normal["source_file"] == source_file, feature].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            right = anomaly.loc[anomaly["source_file"] == source_file, feature].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            combined = pd.concat([left, right])
            if combined.empty:
                ax.set_axis_off()
                continue
            low, high = combined.quantile([0.01, 0.99])
            if low == high:
                low, high = combined.min(), combined.max()
            ax.hist(left.clip(low, high), bins=40, density=True, alpha=0.55, label="normal", color="#54A24B")
            ax.hist(right.clip(low, high), bins=40, density=True, alpha=0.55, label="anomaly", color="#E45756")
            if row_number == 0:
                ax.set_title(source_file)
            if column_number == 0:
                ax.set_ylabel(feature, rotation=0, ha="right", va="center")
            ax.grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.99), ncol=2)
    fig.suptitle("Top within-test anomaly distribution shifts", y=0.995)
    fig.tight_layout(rect=[0.02, 0.01, 1, 0.97])
    save_figure(fig, path)


def plot_event_feature_heatmap(
    event_effects: pd.DataFrame, ranked_features: Sequence[str], path: Path
) -> None:
    features = list(ranked_features[:15])
    subset = event_effects[event_effects["feature"].isin(features)].copy()
    subset["event_id"] = subset["source_file"].str.replace("hai-test", "T", regex=False).str.replace(".csv", "", regex=False) + "-E" + subset["event_number"].astype(str)
    event_order = (
        subset[["source_file", "event_number", "event_id"]]
        .drop_duplicates()
        .sort_values(["source_file", "event_number"])["event_id"]
        .tolist()
    )
    matrix = (
        subset.pivot(index="feature", columns="event_id", values="during_vs_normal_sd")
        .reindex(index=features, columns=event_order)
    )
    clipped = matrix.clip(-10, 10)
    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(clipped.to_numpy(), cmap="coolwarm", vmin=-10, vmax=10, aspect="auto")
    ax.set_xticks(range(len(clipped.columns)), clipped.columns, rotation=75, fontsize=7)
    ax.set_yticks(range(len(clipped.index)), clipped.index, fontsize=8)
    ax.set_title("Per-event mean shift vs test-normal baseline (clipped at ±10 SD)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Normal SD")
    save_figure(fig, path)


def plot_autocorrelation(autocorrelation: pd.DataFrame, path: Path) -> None:
    averaged = autocorrelation.groupby(["feature", "lag_samples"])["autocorrelation"].mean().unstack()
    selected = averaged.abs().mean(axis=1).nlargest(20).index.tolist()
    matrix = averaged.reindex(selected)
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [str(value) for value in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    ax.set_xlabel("Lag (samples)")
    ax.set_title("Mean autocorrelation across train files (strongest mean absolute persistence)")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    save_figure(fig, path)


def file_mean_shift_table(
    feature_file: pd.DataFrame, global_stats: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    baseline = global_stats[global_stats["group"] == "train_unlabeled"].set_index("feature")
    rows = []
    for _, row in feature_file.iterrows():
        feature = row["feature"]
        baseline_std = float(baseline.at[feature, "std"])
        if np.isfinite(baseline_std) and baseline_std > 0:
            shift = (float(row["mean"]) - float(baseline.at[feature, "mean"])) / baseline_std
        else:
            shift = np.nan
        rows.append(
            {
                "source_file": row["source_file"],
                "split": row["split"],
                "feature": feature,
                "file_mean": row["mean"],
                "train_unlabeled_mean": baseline.at[feature, "mean"],
                "train_unlabeled_std": baseline_std,
                "file_mean_vs_train_unlabeled_sd": shift,
            }
        )
    return pd.DataFrame(rows)


def plot_file_mean_shift(file_shift: pd.DataFrame, file_summary: pd.DataFrame, path: Path) -> None:
    pivot = file_shift.pivot(index="feature", columns="source_file", values="file_mean_vs_train_unlabeled_sd")
    ordered_files = file_summary.sort_values("start_time")["source_file"].tolist()
    pivot = pivot.reindex(columns=ordered_files)
    spread = pivot.max(axis=1) - pivot.min(axis=1)
    selected = spread.nlargest(25).index
    matrix = pivot.loc[selected].clip(-5, 5)
    fig, ax = plt.subplots(figsize=(11, 10))
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-5, vmax=5, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    ax.set_title("File-level feature means vs pooled unlabelled-train baseline (clipped at ±5 SD)")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03, label="Train-unlabelled SD")
    save_figure(fig, path)


def plot_test_feature_timelines(
    test_files: Sequence[Path],
    features: Sequence[str],
    events: pd.DataFrame,
    output_dir: Path,
) -> None:
    selected_features = list(features[:6])
    for test_path in test_files:
        frame = pd.read_csv(test_path, usecols=[TIMESTAMP_COLUMN] + selected_features)
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
        stride = max(1, math.ceil(len(frame) / 7000))
        sampled = frame.iloc[::stride]
        file_events = events[events["source_file"] == test_path.name]
        fig, axes = plt.subplots(len(selected_features), 1, figsize=(15, 2.0 * len(selected_features)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, feature in zip(axes, selected_features):
            ax.plot(sampled[TIMESTAMP_COLUMN], sampled[feature], linewidth=0.65, color="#4C78A8")
            for _, event in file_events.iterrows():
                ax.axvspan(pd.Timestamp(event["start_time"]), pd.Timestamp(event["end_time"]), color="#E45756", alpha=0.16)
            ax.set_ylabel(feature, rotation=0, ha="right", va="center", fontsize=8)
            ax.grid(alpha=0.16)
        axes[0].set_title(f"{test_path.name}: top distribution-shift features (red = anomaly interval)")
        axes[-1].set_xlabel("Timestamp")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        fig.tight_layout()
        save_figure(fig, output_dir / f"13_{test_path.stem}_top_feature_timelines.png")


def high_correlation_pairs(correlation: pd.DataFrame) -> pd.DataFrame:
    values = correlation.to_numpy()
    rows = []
    for left in range(len(correlation)):
        for right in range(left + 1, len(correlation)):
            value = values[left, right]
            if np.isfinite(value) and abs(value) >= 0.95:
                rows.append(
                    {
                        "feature_1": correlation.index[left],
                        "feature_2": correlation.columns[right],
                        "pearson_correlation": value,
                        "absolute_correlation": abs(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("absolute_correlation", ascending=False) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "pearson_correlation", "absolute_correlation"]
    )


def test_correlation_changes(
    samples: Dict[str, pd.DataFrame], features: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normal = samples["test_normal"]
    anomaly = samples["test_anomaly"]
    source_files = sorted(set(normal["source_file"]) | set(anomaly["source_file"]))
    rows: List[Dict[str, object]] = []
    for source_file in source_files:
        normal_corr = normal.loc[
            normal["source_file"] == source_file, features
        ].replace([np.inf, -np.inf], np.nan).corr()
        anomaly_corr = anomaly.loc[
            anomaly["source_file"] == source_file, features
        ].replace([np.inf, -np.inf], np.nan).corr()
        for left in range(len(features)):
            for right in range(left + 1, len(features)):
                normal_value = normal_corr.iat[left, right]
                anomaly_value = anomaly_corr.iat[left, right]
                if not (np.isfinite(normal_value) and np.isfinite(anomaly_value)):
                    continue
                difference = float(anomaly_value - normal_value)
                rows.append(
                    {
                        "source_file": source_file,
                        "feature_1": features[left],
                        "feature_2": features[right],
                        "normal_sample_correlation": float(normal_value),
                        "anomaly_correlation": float(anomaly_value),
                        "correlation_change": difference,
                        "absolute_correlation_change": abs(difference),
                    }
                )
    changes = pd.DataFrame(rows)
    macro = (
        changes.groupby(["feature_1", "feature_2"], as_index=False)
        .agg(
            mean_absolute_correlation_change=("absolute_correlation_change", "mean"),
            max_absolute_correlation_change=("absolute_correlation_change", "max"),
            mean_signed_correlation_change=("correlation_change", "mean"),
            contributing_test_files=("source_file", "nunique"),
        )
        .sort_values("mean_absolute_correlation_change", ascending=False)
        .reset_index(drop=True)
    )
    return changes, macro


def plot_correlation_change(
    macro_changes: pd.DataFrame, features: Sequence[str], path: Path
) -> None:
    feature_scores = Counter()
    for _, row in macro_changes.iterrows():
        feature_scores[row["feature_1"]] += row["mean_absolute_correlation_change"]
        feature_scores[row["feature_2"]] += row["mean_absolute_correlation_change"]
    selected = [feature for feature, _ in feature_scores.most_common(30)]
    matrix = pd.DataFrame(np.nan, index=selected, columns=selected)
    np.fill_diagonal(matrix.values, 0.0)
    indexed = macro_changes.set_index(["feature_1", "feature_2"])
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            key = (left, right) if (left, right) in indexed.index else (right, left)
            if key in indexed.index:
                value = float(indexed.at[key, "mean_absolute_correlation_change"])
                matrix.at[left, right] = value
                matrix.at[right, left] = value
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(matrix.to_numpy(), cmap="magma", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(selected)), selected, rotation=90, fontsize=7)
    ax.set_yticks(range(len(selected)), selected, fontsize=7)
    ax.set_title("Mean absolute normal/anomaly correlation change within test files")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03, label="|Δ Pearson r|")
    save_figure(fig, path)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in selected.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + rows)


def build_report(
    output_dir: Path,
    file_summary: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    global_stats: pd.DataFrame,
    label_validation: pd.DataFrame,
    events: pd.DataFrame,
    anomaly_effect: pd.DataFrame,
    anomaly_effect_by_test: pd.DataFrame,
    drift: pd.DataFrame,
    high_corr: pd.DataFrame,
    file_mean_shift: pd.DataFrame,
    response_summary: pd.DataFrame,
    correlation_changes_macro: pd.DataFrame,
    prefix_statistics: pd.DataFrame,
    pca_info: Dict[str, object],
) -> None:
    total_rows = int(file_summary["n_rows"].sum())
    train_rows = int(file_summary.loc[file_summary["split"] == "train_unlabeled", "n_rows"].sum())
    test_rows = int(file_summary.loc[file_summary["split"] == "test", "n_rows"].sum())
    missing = int(file_summary["feature_missing_count"].sum())
    infinite = int(file_summary["feature_infinite_count"].sum())
    duplicate_timestamps = int(file_summary["timestamp_duplicate_count"].sum())
    duplicate_rows = int(file_summary["duplicate_row_count"].sum())
    irregular = int(file_summary["irregular_interval_count"].sum())
    category_counts = feature_catalog["cardinality_category"].value_counts().to_dict()
    constant_features = feature_catalog.loc[
        feature_catalog["cardinality_category"] == "constant", "feature"
    ].tolist()
    train_constant_features = feature_catalog.loc[
        feature_catalog["train_cardinality_category"] == "constant", "feature"
    ].tolist()
    train_only_constant_features = sorted(set(train_constant_features) - set(constant_features))
    group_stats_index = global_stats.set_index(["group", "feature"])
    train_only_change_notes = []
    for feature in train_only_constant_features:
        train_value = float(group_stats_index.at[("train_unlabeled", feature), "mean"])
        anomaly_mean = float(group_stats_index.at[("test_anomaly", feature), "mean"])
        anomaly_count = int(group_stats_index.at[("test_anomaly", feature), "finite_count"])
        if np.isclose(train_value, 0.0):
            changed_rows = int(round(anomaly_mean * anomaly_count))
        elif np.isclose(train_value, 1.0):
            changed_rows = int(round((1.0 - anomaly_mean) * anomaly_count))
        else:
            changed_rows = -1
        train_only_change_notes.append(f"{feature}: 이상 라벨 구간 {changed_rows:,}행에서 학습 고정값과 다름")
    prefix_counts = feature_catalog["prefix"].value_counts().sort_index().to_dict()
    file_summary["previous_file"] = file_summary["source_file"].shift(1)
    gaps = file_summary[file_summary["unobserved_seconds_since_previous_file"] > 0]
    gap_text = ", ".join(
        f"{row['previous_file']} → {row['source_file']}: {row['unobserved_seconds_since_previous_file'] / 3600:.0f}시간"
        for _, row in gaps.iterrows()
    )

    label_columns = [
        "test_file",
        "n_rows",
        "positive_points",
        "positive_rate",
        "actual_event_count",
        "label_timestamp_alignment_mode",
        "all_summary_table_events_match",
        "summary_header_aggregate_match",
    ]
    file_table = file_summary.copy()
    file_table["duration"] = file_table["duration_seconds"].map(seconds_to_hms)
    file_columns = ["source_file", "split", "n_rows", "start_time", "end_time", "duration"]

    anomaly_score = "mean_within_test_abs_mean_shift_sd"
    drift_score = "absolute_mean_difference_in_train_unlabeled_sd"
    top_anomaly = anomaly_effect.head(10).copy()
    top_anomaly_by_test = (
        anomaly_effect_by_test.sort_values(
            ["source_file", "ks_statistic"], ascending=[True, False]
        )
        .groupby("source_file", as_index=False, group_keys=False)
        .head(5)
    )
    top_drift = drift.head(10).copy()
    top_anomaly_mean = (
        anomaly_effect[np.isfinite(anomaly_effect[anomaly_score])]
        .sort_values(anomaly_score, ascending=False)
        .head(5)
    )
    zero_variance_changes = anomaly_effect[
        anomaly_effect["test_files_with_zero_normal_variance_change"] > 0
    ]["feature"].tolist()
    shift_spread = (
        file_mean_shift.groupby("feature")["file_mean_vs_train_unlabeled_sd"]
        .agg(lambda values: values.max() - values.min())
        .sort_values(ascending=False)
        .head(10)
        .rename("file_mean_range_in_train_sd")
        .reset_index()
    )
    second_label = label_validation[label_validation["test_file"] == "hai-test2.csv"].iloc[0]
    normal_event_gaps = events["normal_gap_points_before"].dropna()
    top_responses = response_summary.head(10).copy()
    top_correlation_changes = correlation_changes_macro.head(10).copy()

    report = f"""# HAI-23.05 파일 기반 EDA 보고서

생성 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}

## 분석 범위

이 보고서는 현재 제공된 HAI-23.05 파일에서 직접 확인할 수 있는 구조, 품질, 시간축, 라벨, 통계적 패턴만 다룬다. 논문의 공정 설명이나 공격 시나리오 의미는 사용하지 않았다. 라벨이 없는 네 개의 학습 파일은 정상이라고 단정하지 않고 `train_unlabeled`로 표기했다.

## 데이터 구조

- 데이터 파일: {len(file_summary)}개 (학습 {sum(file_summary['split'] == 'train_unlabeled')}개, 테스트 {sum(file_summary['split'] == 'test')}개)
- 특성: {len(feature_catalog)}개, timestamp: 1개
- 전체 관측치: {total_rows:,}개 (학습 {train_rows:,}, 테스트 {test_rows:,})
- subsystem prefix별 특성 수: {json.dumps(prefix_counts, ensure_ascii=False)}

{markdown_table(file_table, file_columns)}

- 모든 파일 내부는 정확한 1초 간격이며 시간 범위가 서로 겹치지 않는다.
- 파일 사이 미관측 구간: {gap_text}
- `hai-test1` → `hai-train2`, `hai-test2` → `hai-train3` 경계는 1초 차이로 연속된다.

## 데이터 품질

- 특성 결측값: {missing:,}개
- 특성 무한값: {infinite:,}개
- 중복 timestamp: {duplicate_timestamps:,}개
- 파일 내부 완전 중복 행: {duplicate_rows:,}개
- 최빈 샘플링 간격과 다른 인접 구간: {irregular:,}개
- cardinality 분류: {json.dumps(category_counts, ensure_ascii=False)}
- 전체 기간 동안 값이 변하지 않는 {len(constant_features)}개 특성: {', '.join(constant_features)}
- 학습 파일에서 값이 변하지 않는 특성은 {len(train_constant_features)}개다. 그중 학습에서만 고정된 특성은 {', '.join(train_only_constant_features)}이며, 테스트 이상 라벨 구간에서는 {', '.join(train_only_change_notes)}.

## 테스트 라벨 정합성

{markdown_table(label_validation, label_columns)}

- 전체 이상 포인트: {int(label_validation['positive_points'].sum()):,}개 / {int(label_validation['n_rows'].sum()):,}개 ({label_validation['positive_points'].sum() / label_validation['n_rows'].sum():.3%})
- 연속 이상 이벤트: {len(events)}개
- 이벤트 span: 최소 {events['span_seconds'].min():.0f}초, 중앙값 {events['span_seconds'].median():.0f}초, 최대 {events['span_seconds'].max():.0f}초
- `label-test2.csv`의 timestamp는 초 정보가 제거되어 230,400행 중 고유 timestamp가 {int(second_label['label_timestamp_unique_count']):,}개뿐이다. 다만 `hai-test2.csv` timestamp를 분 단위로 내린 값과 전 행이 순서대로 일치하므로 원본 test timestamp를 유지한 positional alignment를 사용했다.
- `summary_label2.txt`의 이벤트 표 38개는 CSV와 모두 일치하지만, 상단 집계의 공격 수 {int(second_label['summary_header_attack_count'])}회와 총 span {second_label['summary_header_total_hms']}는 실제 연속 이벤트 {int(second_label['actual_event_count'])}개와 총 span {seconds_to_hms(float(second_label['actual_event_span_total_seconds']))}에 일치하지 않는다.
- 각 이벤트의 `끝-시작` span 합은 양 끝을 포함한 양성 포인트 수보다 이벤트당 1초씩 짧다. 모델 평가용 클래스 비율은 CSV의 양성 포인트 수를 기준으로 해야 한다.
- 연속 이벤트 사이 정상 포인트 수는 최소 {normal_event_gaps.min():.0f}, 중앙값 {normal_event_gaps.median():.0f}, 최대 {normal_event_gaps.max():.0f}개다.

## 정상 라벨 구간과 이상 라벨 구간의 차이가 큰 특성

각 test 파일 안에서 정상/이상 라벨의 전체 행을 사용해 KS 통계를 계산한 뒤, 두 test 파일을 동일 가중 평균했다. 따라서 test 파일 자체의 분포 차이가 이상 효과로 섞이는 문제를 줄였다. 인과관계나 공격 대상 변수를 뜻하지 않는다.

{markdown_table(top_anomaly, ['feature', 'mean_within_test_ks', 'min_within_test_ks', 'max_within_test_ks', anomaly_score])}

파일별 상위 특성은 다음과 같아 두 test 파일의 이상 패턴이 완전히 같지는 않음을 보여준다.

{markdown_table(top_anomaly_by_test, ['source_file', 'feature', 'ks_statistic', 'absolute_mean_difference_in_test_normal_sd'])}

파일 내부의 표준화 평균 이동량만 보면 다음 특성이 크다. 정상 라벨 구간에서 분산이 0인데 이상 구간에서 값이 변한 상태 변수({', '.join(zero_variance_changes)})는 무한한 표준화 값이 되므로 이 순위에서 제외했다.

{markdown_table(top_anomaly_mean, ['feature', anomaly_score, 'mean_within_test_ks'])}

## 학습 파일과 테스트 정상 라벨 구간의 분포 차이

아래 순위는 파일 행 수의 2%를 기본으로 하되 파일당 1,000–5,000행으로 제한한 결정론적 표본의 KS 통계 기준이며, 전체 행 평균의 차이를 train-unlabelled 표준편차 단위로 함께 표시한다. 이후 모델 실험의 데이터 분할을 결정한 결과가 아니다.

{markdown_table(top_drift, ['feature', 'ks_statistic', drift_score])}

파일별 평균이 크게 달라지는 특성은 다음과 같다. 값은 여섯 파일 평균의 최대–최소 범위를 pooled train-unlabelled 표준편차로 환산한 탐색 지표다.

{markdown_table(shift_spread, ['feature', 'file_mean_range_in_train_sd'])}

## 이벤트 반응 및 복귀의 탐색적 지표

반응 지연은 이벤트 직전 300초 median에서 file-normal SD의 3배를 벗어난 상태가 3초 지속되는 최초 시점이다. 이벤트 직전부터 이미 임계값을 벗어난 경우는 `preexisting_excursion`으로 분리했다. file-normal SD가 0이면 직전 300초 mode에서 3초 연속 달라지는 규칙을 사용했다. 복귀는 반응한 변수에 한해 이벤트 종료 후 local baseline의 1 SD 이내가 10초 지속되는 최초 시점이며, 다음 이벤트 전 또는 최대 600초까지만 검색했다.

{markdown_table(top_responses, ['feature', 'eligible_events', 'detected_responses', 'response_rate', 'median_reaction_delay_seconds', 'recovery_rate_among_responses', 'median_recovery_delay_seconds', 'preexisting_excursion_events'])}

이 지표는 큰 수준 변화를 찾는 운영적 정의다. 라벨 경계가 물리적 반응 경계라는 보장이 없고, slope나 분산만 변하는 반응은 놓칠 수 있으므로 공격 메커니즘으로 해석하지 않는다.

## prefix별 파일 기반 패턴

{markdown_table(prefix_statistics, ['prefix', 'feature_count', 'global_constant_count', 'train_constant_count', 'median_within_test_anomaly_ks', 'max_within_test_anomaly_ks', 'median_train_test_normal_sample_ks'])}

## 다변량 및 시간 특성

- 상관행렬은 학습 파일에서 고정된 {len(train_constant_features)}개 특성을 제외한 {len(feature_catalog) - len(train_constant_features)}개 변동 특성으로 계산했다.
- 학습 표본에서 |Pearson r| ≥ 0.95인 특성 쌍: {len(high_corr):,}개
- PCA 사용 가능 특성: {pca_info['usable_feature_count']}개
- PCA 설명분산: PC1 {pca_info['pc1_explained_variance_ratio']:.2%}, PC2 {pca_info['pc2_explained_variance_ratio']:.2%}
- 1, 10, 60, 300 sample lag 자기상관은 `tables/autocorrelation_train.csv`에 기록했다.

각 test 파일 내부에서 정상 표본 상관과 이상 전체행 상관의 차이를 계산했다. 변화가 큰 특성 쌍은 다음과 같으며, 이는 인과관계가 아니라 라벨 구간별 선형 관계 변화다.

{markdown_table(top_correlation_changes, ['feature_1', 'feature_2', 'mean_absolute_correlation_change', 'max_absolute_correlation_change', 'contributing_test_files'])}

`contributing_test_files=1`인 쌍은 다른 test 파일에서 한 변수의 분산이 0이라 상관을 정의할 수 없었던 경우이므로 두 파일에 공통된 변화로 일반화하지 않는다.

## 해석 시 주의사항

- EDA의 그룹 차이는 연관성이다. 공격 원인이나 센서의 물리적 의미로 해석하지 않는다.
- 학습 파일에는 제공된 라벨이 없으므로 파일 기반 EDA만으로 정상성을 검증할 수 없다.
- test 파일 내부 정상/이상 KS는 전체 행으로 계산했다. train/test drift의 KS, 분위수, 상관관계 및 PCA는 행 수의 2%를 기본으로 파일당 1,000–5,000행으로 제한한 결정론적 표본을 사용했고 평균·표준편차·결측 통계는 전체 행으로 계산했다.
- KS p-value는 강한 자기상관과 이산형 변수 때문에 독립·동일분포 및 연속분포 가정을 충족하지 않을 수 있으므로 유의성 추론에 사용하지 않고 서술적 참고값으로만 기록했다.
- 시간 순서를 보존했으며 서로 다른 파일 경계에 걸쳐 lag 또는 이벤트를 연결하지 않았다.
- 학습 데이터 5–100% 부분집합 구성과 모델링은 이 EDA에 포함하지 않았다.

## 산출물

- `tables/file_summary.csv`: 파일별 크기와 시간축 품질
- `tables/feature_catalog.csv`: 컬럼 prefix와 cardinality 분류
- `tables/feature_file_statistics.csv`: 파일·특성별 기술통계
- `tables/feature_group_statistics.csv`: 학습/테스트 정상/테스트 이상 전체행 통계
- `tables/label_validation.csv`: 테스트와 라벨 timestamp 및 summary 검증
- `tables/anomaly_events.csv`: 연속 이상 이벤트 목록
- `tables/anomaly_feature_effect.csv`: test 파일 내부 전수 KS의 macro 요약
- `tables/anomaly_feature_effect_by_test.csv`: test 파일별 전수 정상/이상 비교
- `tables/anomaly_feature_effect_pooled.csv`: 보조적인 pooled 표본 비교
- `tables/train_vs_test_normal_drift.csv`: 학습/테스트 정상 분포 차이
- `tables/event_feature_effects.csv`: 이벤트별 전·중·후 특성 평균
- `tables/event_response_metrics.csv`: 명시적 임계값 기준 이벤트·특성별 반응/복귀
- `tables/event_response_summary.csv`: 특성별 반응/복귀 요약
- `tables/autocorrelation_train.csv`: 파일 경계를 보존한 자기상관
- `tables/state_transitions.csv`: 저 cardinality 변수 상태 전환
- `tables/train_sample_correlation.csv`: 학습 표본 상관행렬
- `tables/high_correlation_pairs.csv`: 고상관 특성 쌍
- `tables/correlation_change_by_test.csv`: test 파일별 정상/이상 상관 변화
- `tables/correlation_change_macro.csv`: 상관 변화의 test 파일 macro 요약
- `tables/file_feature_mean_shift.csv`: 파일별 평균의 학습 기준 표준화 차이
- `tables/feature_prefix_summary.csv`: prefix·cardinality별 특성 수
- `tables/prefix_statistics.csv`: prefix별 품질·이상 차이·분포 변화 요약
- `tables/pca_projection_sample.csv`: PCA 시각화용 표본 좌표
- `run_metadata.json`: 실행 옵션과 입력 파일 기록
- `figures/`: 핵심 시각화
"""
    (output_dir / "HAI_EDA_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(data_dir.glob("hai-train*.csv"), key=natural_number)
    test_files = sorted(data_dir.glob("hai-test*.csv"), key=natural_number)
    label_files = sorted(data_dir.glob("label-test*.csv"), key=natural_number)
    summary_files = sorted(data_dir.glob("summary_label*.txt"), key=natural_number)
    if not train_files or not test_files:
        raise FileNotFoundError(f"HAI train/test CSVs not found under {data_dir}")
    if len(test_files) != len(label_files) or len(test_files) != len(summary_files):
        raise ValueError("Each test file must have one label CSV and one summary text file")

    headers = {}
    for path in train_files + test_files:
        headers[path.name] = pd.read_csv(path, nrows=0).columns.tolist()
    reference_header = headers[train_files[0].name]
    mismatched_headers = [name for name, header in headers.items() if header != reference_header]
    if mismatched_headers:
        raise ValueError(f"Feature schema differs in: {mismatched_headers}")
    if reference_header[0] != TIMESTAMP_COLUMN:
        raise ValueError(f"Expected first column '{TIMESTAMP_COLUMN}'")
    features = reference_header[1:]

    rng = np.random.default_rng(args.seed)
    stats = {
        group: RunningStats(features)
        for group in ("train_unlabeled", "test_all", "test_normal", "test_anomaly", "all_data")
    }
    sample_parts: Dict[str, List[pd.DataFrame]] = {
        "train_unlabeled": [],
        "test_all": [],
        "test_normal": [],
        "test_anomaly": [],
    }
    cardinality_trackers: Dict[str, Optional[set]] = {feature: set() for feature in features}
    train_cardinality_trackers: Dict[str, Optional[set]] = {feature: set() for feature in features}
    file_rows: List[Dict[str, object]] = []
    feature_file_parts: List[pd.DataFrame] = []
    transition_rows: List[Dict[str, object]] = []
    autocorrelation_rows: List[Dict[str, object]] = []
    label_validation_rows: List[Dict[str, object]] = []
    event_rows: List[Dict[str, object]] = []
    event_feature_rows: List[Dict[str, object]] = []
    event_response_rows: List[Dict[str, object]] = []
    label_plot_series: List[Tuple[str, pd.Series, pd.Series]] = []
    per_test_effect_parts: List[pd.DataFrame] = []

    def proportional_sample_limit(row_count: int) -> int:
        target = max(EDA_MINIMUM_SAMPLE_ROWS, math.ceil(row_count * EDA_SAMPLE_FRACTION))
        return min(args.sample_per_file, target)

    def load_and_audit(path: Path, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"Reading {path.name} ...", flush=True)
        frame = pd.read_csv(path)
        frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
        numeric = ensure_numeric(frame, features)
        audit = timestamp_audit(frame[TIMESTAMP_COLUMN])
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        row = {
            "source_file": path.name,
            "split": split,
            "file_size_bytes": path.stat().st_size,
            "n_rows": len(frame),
            "n_features": len(features),
            "feature_missing_count": int(np.isnan(values).sum()),
            "feature_infinite_count": int(np.isinf(values).sum()),
            "duplicate_row_count": int(frame.duplicated().sum()),
            **audit,
        }
        file_rows.append(row)
        feature_file_parts.append(feature_file_statistics(numeric, path.name, split))
        transition_rows.extend(transition_statistics(numeric, path.name, split, features))
        update_cardinality(cardinality_trackers, numeric, features)
        return frame, numeric

    for path in train_files:
        frame, numeric = load_and_audit(path, "train_unlabeled")
        stats["train_unlabeled"].update(numeric)
        stats["all_data"].update(numeric)
        update_cardinality(train_cardinality_trackers, numeric, features)
        sample_parts["train_unlabeled"].append(
            sample_rows(
                numeric,
                "train_unlabeled",
                path.name,
                proportional_sample_limit(len(numeric)),
                rng,
            )
        )
        autocorrelation_rows.extend(autocorrelation_statistics(numeric, path.name, features))
        del frame, numeric

    for test_path, label_path, summary_path in zip(test_files, label_files, summary_files):
        frame, numeric = load_and_audit(test_path, "test")
        print(f"Reading and validating {label_path.name} ...", flush=True)
        label_frame = pd.read_csv(label_path)
        if list(label_frame.columns) != [TIMESTAMP_COLUMN, LABEL_COLUMN]:
            raise ValueError(f"Unexpected label schema in {label_path.name}: {list(label_frame.columns)}")
        label_frame[TIMESTAMP_COLUMN] = pd.to_datetime(label_frame[TIMESTAMP_COLUMN], errors="coerce")
        label_frame[LABEL_COLUMN] = pd.to_numeric(label_frame[LABEL_COLUMN], errors="coerce")
        exact_alignment = len(frame) == len(label_frame) and frame[TIMESTAMP_COLUMN].equals(label_frame[TIMESTAMP_COLUMN])
        minute_floor_alignment = (
            len(frame) == len(label_frame)
            and frame[TIMESTAMP_COLUMN].dt.floor("min").equals(label_frame[TIMESTAMP_COLUMN])
        )
        if exact_alignment:
            labels = label_frame[LABEL_COLUMN].reset_index(drop=True)
            alignment_mode = "exact_timestamp_positional"
        elif minute_floor_alignment:
            # label-test2.csv preserves row order but its timestamp seconds were
            # truncated. Keep the second-resolution times from hai-test2.csv.
            labels = label_frame[LABEL_COLUMN].reset_index(drop=True)
            alignment_mode = "minute_floor_timestamp_positional"
        else:
            if label_frame[TIMESTAMP_COLUMN].duplicated().any():
                raise ValueError(
                    f"Cannot safely align {label_path.name}: timestamps are neither exact nor "
                    "minute-floor positional matches, and label timestamps are duplicated"
                )
            merged = frame[[TIMESTAMP_COLUMN]].merge(
                label_frame, on=TIMESTAMP_COLUMN, how="left", validate="one_to_one"
            )
            labels = merged[LABEL_COLUMN]
            alignment_mode = "unique_timestamp_merge"
        label_values = labels.to_numpy(dtype=np.float64, copy=False)
        valid_label_values = sorted(pd.unique(labels.dropna()).tolist())
        normal_mask = label_values == 0
        anomaly_mask = label_values == 1
        events = extract_events(frame[TIMESTAMP_COLUMN], labels, test_path.name)
        reported_events = parse_summary_events(summary_path)
        summary_header = parse_summary_header(summary_path)
        summary_comparison = compare_summary_events(events, reported_events, summary_header)
        label_audit = timestamp_audit(label_frame[TIMESTAMP_COLUMN])
        label_validation_rows.append(
            {
                "test_file": test_path.name,
                "label_file": label_path.name,
                "summary_file": summary_path.name,
                "n_rows": len(frame),
                "label_rows": len(label_frame),
                "row_count_match": len(frame) == len(label_frame),
                "timestamps_exactly_aligned": exact_alignment,
                "timestamps_minute_floor_aligned": minute_floor_alignment,
                "label_timestamp_alignment_mode": alignment_mode,
                "label_timestamp_missing_count": label_audit["timestamp_missing_count"],
                "label_timestamp_duplicate_count": label_audit["timestamp_duplicate_count"],
                "label_timestamp_unique_count": int(label_frame[TIMESTAMP_COLUMN].nunique()),
                "label_missing_count": int(labels.isna().sum()),
                "label_values": ",".join(str(value) for value in valid_label_values),
                "positive_points": int(anomaly_mask.sum()),
                "positive_rate": float(anomaly_mask.mean()),
                **summary_comparison,
            }
        )
        event_rows.extend(events)
        event_feature_rows.extend(event_feature_means(numeric, events, features))
        label_plot_series.append((test_path.name, frame[TIMESTAMP_COLUMN], labels.fillna(0)))

        normal_numeric = numeric.loc[normal_mask]
        anomaly_numeric = numeric.loc[anomaly_mask]
        stats["test_all"].update(numeric)
        stats["test_normal"].update(normal_numeric)
        stats["test_anomaly"].update(anomaly_numeric)
        stats["all_data"].update(numeric)
        normal_sample = sample_rows(
            normal_numeric,
            "test_normal",
            test_path.name,
            proportional_sample_limit(len(normal_numeric)),
            rng,
        )
        anomaly_sample = sample_rows(
            anomaly_numeric,
            "test_anomaly",
            test_path.name,
            len(anomaly_numeric),
            rng,
        )
        sample_parts["test_normal"].append(normal_sample)
        sample_parts["test_anomaly"].append(anomaly_sample)
        sample_parts["test_all"].append(
            sample_rows(
                numeric,
                "test_all",
                test_path.name,
                proportional_sample_limit(len(numeric)),
                rng,
            )
        )

        local_normal = RunningStats(features)
        local_anomaly = RunningStats(features)
        local_normal.update(normal_numeric)
        local_anomaly.update(anomaly_numeric)
        local_normal_summary = local_normal.finish("test_normal")
        local_anomaly_summary = local_anomaly.finish("test_anomaly")
        local_effect = comparison_table(
            local_normal_summary,
            local_anomaly_summary,
            normal_numeric,
            anomaly_numeric,
            features,
            "test_normal",
            "test_anomaly",
            "all_rows_within_test_file",
        )
        local_effect.insert(0, "source_file", test_path.name)
        per_test_effect_parts.append(local_effect)
        event_response_rows.extend(
            event_response_metrics(
                numeric,
                events,
                features,
                local_normal_summary.set_index("feature")["std"],
                local_normal_summary.set_index("feature")["min"],
                local_normal_summary.set_index("feature")["max"],
            )
        )
        del frame, numeric, label_frame, normal_numeric, anomaly_numeric

    samples = {
        group: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["group", "source_file"] + features)
        for group, parts in sample_parts.items()
    }
    samples["all_data"] = pd.concat(
        [samples["train_unlabeled"], samples["test_all"]], ignore_index=True
    )

    file_summary = pd.DataFrame(file_rows).sort_values("start_time").reset_index(drop=True)
    file_summary["seconds_from_previous_end"] = (
        pd.to_datetime(file_summary["start_time"])
        - pd.to_datetime(file_summary["end_time"]).shift(1)
    ).dt.total_seconds()
    file_summary["unobserved_seconds_since_previous_file"] = np.maximum(
        file_summary["seconds_from_previous_end"] - 1.0, 0.0
    )
    file_summary["contiguous_with_previous_file"] = file_summary[
        "seconds_from_previous_end"
    ].eq(1.0)
    feature_file = pd.concat(feature_file_parts, ignore_index=True)
    global_stats = pd.concat([group_stats.finish(group) for group, group_stats in stats.items()], ignore_index=True)
    global_stats = add_sample_quantiles(global_stats, samples, features)
    label_validation = pd.DataFrame(label_validation_rows)
    events_frame = pd.DataFrame(event_rows).sort_values(
        ["source_file", "event_number"]
    ).reset_index(drop=True)
    previous_end = events_frame.groupby("source_file")["end_time"].shift(1)
    next_start = events_frame.groupby("source_file")["start_time"].shift(-1)
    events_frame["normal_gap_points_before"] = (
        (pd.to_datetime(events_frame["start_time"]) - pd.to_datetime(previous_end))
        .dt.total_seconds()
        .sub(1)
    )
    events_frame["normal_gap_points_after"] = (
        (pd.to_datetime(next_start) - pd.to_datetime(events_frame["end_time"]))
        .dt.total_seconds()
        .sub(1)
    )
    event_effects = pd.DataFrame(event_feature_rows)
    event_responses = pd.DataFrame(event_response_rows)
    transitions = pd.DataFrame(transition_rows)
    autocorrelation = pd.DataFrame(autocorrelation_rows)

    catalog_rows = []
    all_stats = global_stats[global_stats["group"] == "all_data"].set_index("feature")
    for feature in features:
        category, unique_lower_bound, exact = classify_cardinality(cardinality_trackers[feature])
        train_category, train_unique_lower_bound, train_exact = classify_cardinality(
            train_cardinality_trackers[feature]
        )
        catalog_rows.append(
            {
                "feature": feature,
                "prefix": feature_prefix(feature),
                "cardinality_category": category,
                "global_unique_count_or_lower_bound": unique_lower_bound,
                "global_unique_count_exact": exact,
                "train_cardinality_category": train_category,
                "train_unique_count_or_lower_bound": train_unique_lower_bound,
                "train_unique_count_exact": train_exact,
                "global_mean": all_stats.at[feature, "mean"],
                "global_std": all_stats.at[feature, "std"],
                "global_min": all_stats.at[feature, "min"],
                "global_max": all_stats.at[feature, "max"],
                "global_missing_rate": all_stats.at[feature, "missing_rate"],
                "global_zero_rate": all_stats.at[feature, "zero_rate"],
            }
        )
    feature_catalog = pd.DataFrame(catalog_rows)
    low_cardinality_features = feature_catalog.loc[
        feature_catalog["cardinality_category"].isin(["constant", "binary", "low_cardinality"]), "feature"
    ]
    transitions = transitions[transitions["feature"].isin(low_cardinality_features)].reset_index(drop=True)

    normal_stats = global_stats[global_stats["group"] == "test_normal"]
    anomaly_stats = global_stats[global_stats["group"] == "test_anomaly"]
    train_stats = global_stats[global_stats["group"] == "train_unlabeled"]
    pooled_anomaly_effect = comparison_table(
        normal_stats,
        anomaly_stats,
        samples["test_normal"],
        samples["test_anomaly"],
        features,
        "test_normal",
        "test_anomaly",
        "proportional_normal_sample_and_all_anomaly_rows_pooled_across_test_files",
    )
    drift = comparison_table(
        train_stats,
        normal_stats,
        samples["train_unlabeled"],
        samples["test_normal"],
        features,
        "train_unlabeled",
        "test_normal",
        "proportional_random_sample_across_files",
    )
    anomaly_effect_by_test = pd.concat(per_test_effect_parts, ignore_index=True)
    anomaly_effect = (
        anomaly_effect_by_test.groupby("feature", as_index=False)
        .agg(
            mean_within_test_ks=("ks_statistic", "mean"),
            min_within_test_ks=("ks_statistic", "min"),
            max_within_test_ks=("ks_statistic", "max"),
            mean_within_test_abs_mean_shift_sd=(
                "absolute_mean_difference_in_test_normal_sd",
                "mean",
            ),
            test_files_with_zero_normal_variance_change=(
                "test_normal_zero_variance_but_mean_changed",
                "sum",
            ),
        )
    )
    pooled_columns = pooled_anomaly_effect[
        [
            "feature",
            "absolute_mean_difference_in_test_normal_sd",
            "test_normal_zero_variance_but_mean_changed",
            "ks_statistic",
        ]
    ].rename(columns={"ks_statistic": "pooled_sample_ks_statistic"})
    anomaly_effect = anomaly_effect.merge(pooled_columns, on="feature", how="left")
    anomaly_effect["within_test_ks_rank"] = anomaly_effect["mean_within_test_ks"].rank(
        method="min", ascending=False
    )
    anomaly_effect = anomaly_effect.sort_values(
        ["mean_within_test_ks", "max_within_test_ks"], ascending=False
    ).reset_index(drop=True)

    file_baselines = anomaly_effect_by_test[
        ["source_file", "feature", "test_normal_mean", "test_normal_std"]
    ].rename(
        columns={"test_normal_mean": "normal_mean", "test_normal_std": "normal_std"}
    )
    event_effects = event_effects.merge(
        file_baselines, on=["source_file", "feature"], how="left", validate="many_to_one"
    )
    event_effects = event_effects.merge(
        event_responses,
        on=["source_file", "event_number", "feature"],
        how="left",
        validate="one_to_one",
    )
    valid_std = event_effects["normal_std"].gt(0)
    event_effects["during_vs_normal_sd"] = np.where(
        valid_std,
        (event_effects["during_mean"] - event_effects["normal_mean"]) / event_effects["normal_std"],
        np.nan,
    )
    event_effects["during_vs_pre_sd"] = np.where(
        valid_std,
        (event_effects["during_mean"] - event_effects["pre_mean"]) / event_effects["normal_std"],
        np.nan,
    )
    event_effects["post_vs_normal_sd"] = np.where(
        valid_std,
        (event_effects["post_mean"] - event_effects["normal_mean"]) / event_effects["normal_std"],
        np.nan,
    )

    response_eligible = event_responses[
        event_responses["threshold_defined"] & ~event_responses["preexisting_excursion"]
    ]
    response_summary = (
        response_eligible.groupby("feature", as_index=False)
        .agg(
            eligible_events=("event_number", "count"),
            detected_responses=("response_detected", "sum"),
            median_reaction_delay_seconds=("reaction_delay_seconds", "median"),
            median_peak_abs_deviation_sd=(
                "peak_abs_deviation_from_pre_median_sd",
                "median",
            ),
            detected_recoveries=("recovery_detected", "sum"),
            median_recovery_delay_seconds=("recovery_delay_seconds", "median"),
            right_censored_recoveries=("recovery_right_censored", "sum"),
        )
    )
    response_summary["response_rate"] = (
        response_summary["detected_responses"] / response_summary["eligible_events"]
    )
    response_summary["recovery_rate_among_responses"] = np.divide(
        response_summary["detected_recoveries"],
        response_summary["detected_responses"],
        out=np.full(len(response_summary), np.nan),
        where=response_summary["detected_responses"].to_numpy() > 0,
    )
    preexisting_counts = (
        event_responses.groupby("feature")["preexisting_excursion"]
        .sum()
        .rename("preexisting_excursion_events")
    )
    response_summary = response_summary.merge(
        preexisting_counts, left_on="feature", right_index=True, how="left"
    ).sort_values(
        ["response_rate", "median_peak_abs_deviation_sd"],
        ascending=[False, False],
        na_position="last",
    )

    train_varying_features = feature_catalog.loc[
        feature_catalog["train_cardinality_category"] != "constant", "feature"
    ].tolist()
    train_sample = samples["train_unlabeled"].loc[:, train_varying_features].replace(
        [np.inf, -np.inf], np.nan
    )
    correlation = train_sample.corr(method="pearson")
    high_corr = high_correlation_pairs(correlation)
    file_mean_shift = file_mean_shift_table(feature_file, global_stats, features)
    correlation_changes_by_test, correlation_changes_macro = test_correlation_changes(
        samples, train_varying_features
    )
    file_mean_ranges = (
        file_mean_shift.groupby("feature")["file_mean_vs_train_unlabeled_sd"]
        .agg(lambda values: values.max() - values.min())
        .rename("file_mean_range_in_train_sd")
    )
    prefix_feature_frame = (
        feature_catalog.merge(
            anomaly_effect[["feature", "mean_within_test_ks"]], on="feature", how="left"
        )
        .merge(
            drift[["feature", "ks_statistic"]].rename(
                columns={"ks_statistic": "train_test_normal_sample_ks"}
            ),
            on="feature",
            how="left",
        )
        .merge(file_mean_ranges, left_on="feature", right_index=True, how="left")
    )
    prefix_feature_frame["global_constant"] = prefix_feature_frame[
        "cardinality_category"
    ].eq("constant")
    prefix_feature_frame["train_constant"] = prefix_feature_frame[
        "train_cardinality_category"
    ].eq("constant")
    prefix_statistics = (
        prefix_feature_frame.groupby("prefix", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            global_constant_count=("global_constant", "sum"),
            train_constant_count=("train_constant", "sum"),
            median_within_test_anomaly_ks=("mean_within_test_ks", "median"),
            max_within_test_anomaly_ks=("mean_within_test_ks", "max"),
            median_train_test_normal_sample_ks=("train_test_normal_sample_ks", "median"),
            median_file_mean_range_in_train_sd=("file_mean_range_in_train_sd", "median"),
        )
        .sort_values("prefix")
    )

    write_csv(file_summary, table_dir / "file_summary.csv")
    write_csv(feature_catalog, table_dir / "feature_catalog.csv")
    write_csv(feature_file, table_dir / "feature_file_statistics.csv")
    write_csv(global_stats, table_dir / "feature_group_statistics.csv")
    write_csv(label_validation, table_dir / "label_validation.csv")
    write_csv(events_frame.drop(columns=["start_index", "end_index"]), table_dir / "anomaly_events.csv")
    write_csv(anomaly_effect, table_dir / "anomaly_feature_effect.csv")
    write_csv(anomaly_effect_by_test, table_dir / "anomaly_feature_effect_by_test.csv")
    write_csv(pooled_anomaly_effect, table_dir / "anomaly_feature_effect_pooled.csv")
    write_csv(drift, table_dir / "train_vs_test_normal_drift.csv")
    write_csv(event_effects, table_dir / "event_feature_effects.csv")
    write_csv(event_responses, table_dir / "event_response_metrics.csv")
    write_csv(response_summary, table_dir / "event_response_summary.csv")
    write_csv(transitions, table_dir / "state_transitions.csv")
    write_csv(autocorrelation, table_dir / "autocorrelation_train.csv")
    correlation.to_csv(table_dir / "train_sample_correlation.csv", encoding="utf-8-sig")
    write_csv(high_corr, table_dir / "high_correlation_pairs.csv")
    write_csv(file_mean_shift, table_dir / "file_feature_mean_shift.csv")
    write_csv(correlation_changes_by_test, table_dir / "correlation_change_by_test.csv")
    write_csv(correlation_changes_macro, table_dir / "correlation_change_macro.csv")
    prefix_summary = feature_catalog.groupby(["prefix", "cardinality_category"]).size().rename("feature_count").reset_index()
    write_csv(prefix_summary, table_dir / "feature_prefix_summary.csv")
    write_csv(prefix_statistics, table_dir / "prefix_statistics.csv")

    plot_file_timeline(file_summary, figure_dir / "01_file_timeline.png")
    plot_missing_rates(feature_file, figure_dir / "02_missingness.png")
    plot_label_timeline(label_plot_series, figure_dir / "03_label_timeline.png")
    plot_event_durations(events_frame, figure_dir / "04_event_durations.png")
    plot_ranked_effect(
        anomaly_effect,
        "mean_within_test_ks",
        "Largest within-test distribution shifts: normal vs anomaly",
        "Mean exact KS statistic across test files",
        figure_dir / "05_anomaly_feature_effect.png",
    )
    plot_ranked_effect(
        drift,
        "ks_statistic",
        "Largest distribution shifts: train unlabelled vs test normal",
        "Proportional-sample KS statistic",
        figure_dir / "06_train_test_normal_drift.png",
    )
    plot_correlation(correlation, figure_dir / "07_train_sample_correlation.png")
    pca_info = pca_projection(
        samples,
        features,
        table_dir / "pca_projection_sample.csv",
        figure_dir / "08_pca_projection.png",
    )
    plot_top_distributions(
        samples,
        anomaly_effect["feature"].tolist(),
        figure_dir / "09_top_anomaly_feature_distributions.png",
    )
    plot_event_feature_heatmap(
        event_effects,
        anomaly_effect["feature"].tolist(),
        figure_dir / "10_event_feature_heatmap.png",
    )
    plot_autocorrelation(autocorrelation, figure_dir / "11_autocorrelation.png")
    plot_file_mean_shift(file_mean_shift, file_summary, figure_dir / "12_file_feature_mean_shift.png")
    plot_test_feature_timelines(
        test_files,
        anomaly_effect["feature"].tolist(),
        events_frame,
        figure_dir,
    )
    plot_correlation_change(
        correlation_changes_macro,
        train_varying_features,
        figure_dir / "14_test_correlation_change.png",
    )

    build_report(
        output_dir,
        file_summary,
        feature_catalog,
        global_stats,
        label_validation,
        events_frame,
        anomaly_effect,
        anomaly_effect_by_test,
        drift,
        high_corr,
        file_mean_shift,
        response_summary,
        correlation_changes_macro,
        prefix_statistics,
        pca_info,
    )

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_directory": str(data_dir),
        "output_directory": str(output_dir),
        "seed": args.seed,
        "sample_per_file": args.sample_per_file,
        "sample_fraction_before_cap": EDA_SAMPLE_FRACTION,
        "minimum_sample_rows_per_file": EDA_MINIMUM_SAMPLE_ROWS,
        "train_files": [path.name for path in train_files],
        "test_files": [path.name for path in test_files],
        "label_files": [path.name for path in label_files],
        "summary_files": [path.name for path in summary_files],
        "feature_count": len(features),
        "notes": [
            "All exact moments and data-quality counters use every row.",
            "Within-test normal/anomaly KS statistics use every row in each test file.",
            "Quantiles, train/test drift, correlation, and PCA use deterministic proportional samples.",
            "Event response/recovery metrics use the explicit operational thresholds documented in the report.",
            "No paper-derived domain semantics were used.",
            "No 5-100% training subsets were created during EDA.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"EDA complete. Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
