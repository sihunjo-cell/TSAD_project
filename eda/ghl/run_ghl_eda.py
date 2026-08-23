# -*- coding: utf-8 -*-
"""Reproducible file-based exploratory analysis for the local GHL dataset.

The distributed GHL files have no explicit timestamp. Row order is therefore
the only time coordinate, and files are never concatenated into one sequence.
The script uses all rows for data-quality counters, moments, event extraction,
and within-file normal/anomaly comparisons. Explicitly documented,
deterministic samples are used for PCA, correlations, and train/post-train
distribution comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


LABEL_COLUMN = "Label"
FILE_PATTERN = re.compile(
    r"(?P<prefix>\d+)_GHL_id_(?P<series_id>\d+)_Sensor_"
    r"tr_(?P<train_length>\d+)_1st_(?P<first_anomaly>\d+)\.csv$"
)
CARDINALITY_CUTOFF = 200
AUTOCORRELATION_LAGS = (1, 10, 100, 1000)
DEFAULT_SAMPLE_PER_GROUP = 3000


@dataclass
class RunningStats:
    """Exact streaming first/second moments and quality counters."""

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
            batch_var = np.nanvar(clean, axis=0, ddof=0)
            batch_min = np.nanmin(clean, axis=0)
            batch_max = np.nanmax(clean, axis=0)
        new_count = old_count + batch_count
        valid = batch_count > 0
        delta = np.zeros_like(self.mean_values)
        delta[valid] = batch_mean[valid] - self.mean_values[valid]
        self.mean_values[valid] += (
            delta[valid] * batch_count[valid] / new_count[valid]
        )
        self.m2[valid] += batch_var[valid] * batch_count[valid] + (
            delta[valid] ** 2
            * old_count[valid]
            * batch_count[valid]
            / new_count[valid]
        )
        self.count = new_count
        self.minimum = np.fmin(self.minimum, batch_min)
        self.maximum = np.fmax(self.maximum, batch_max)

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
        constant = np.isfinite(minimum) & np.isfinite(maximum) & (minimum == maximum)
        variance[constant] = 0.0
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
    parser = argparse.ArgumentParser(description="Run EDA for local GHL files")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "TSB-AD-M",
        help="Directory containing *_GHL_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--sample-per-group",
        type=int,
        default=DEFAULT_SAMPLE_PER_GROUP,
        help="Maximum deterministic sample rows per file and analysis group",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_filename(path: Path) -> Dict[str, int]:
    match = FILE_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected GHL filename: {path.name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in selected.iterrows():
        rendered = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                rendered.append("NA" if not np.isfinite(value) else f"{value:.6g}")
            elif isinstance(value, (bool, np.bool_)):
                rendered.append(str(bool(value)))
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, separator] + rows)


def update_cardinality(
    trackers: Dict[str, Optional[set]], frame: pd.DataFrame, features: Sequence[str]
) -> None:
    for feature in features:
        current = trackers[feature]
        if current is None:
            continue
        values = pd.unique(frame[feature].dropna())
        values = values[np.isfinite(values)]
        current.update(values.tolist())
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


def feature_family(feature: str) -> str:
    lowered = feature.lower()
    if "temperature" in lowered:
        return "temperature"
    if "level" in lowered:
        return "level"
    if "valve" in lowered or "heater_act" in lowered or lowered.endswith(".active"):
        return "actuator_or_state"
    if "rand" in lowered:
        return "random_input"
    if "flow" in lowered:
        return "flow"
    if "limiter" in lowered:
        return "limiter"
    return "other"


def deterministic_sample(
    numeric: pd.DataFrame,
    mask: np.ndarray,
    source_file: str,
    series_id: int,
    group: str,
    limit: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = np.flatnonzero(mask)
    if not len(candidates) or limit <= 0:
        return pd.DataFrame()
    if len(candidates) > limit:
        candidates = np.sort(rng.choice(candidates, size=limit, replace=False))
    sample = numeric.iloc[candidates].copy()
    sample.insert(0, "sample_index_0based", candidates)
    sample.insert(0, "series_id", series_id)
    sample.insert(0, "source_file", source_file)
    sample.insert(0, "group", group)
    return sample


def exact_feature_statistics(
    numeric: pd.DataFrame, source_file: str, series_id: int
) -> pd.DataFrame:
    values = numeric.to_numpy(dtype=np.float64, copy=False)
    clean = numeric.replace([np.inf, -np.inf], np.nan)
    description = clean.describe(
        percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]
    ).T.reset_index(names="feature")
    description = description.rename(
        columns={
            "1%": "q01",
            "25%": "q25",
            "50%": "median",
            "75%": "q75",
            "99%": "q99",
        }
    )
    description.insert(0, "series_id", series_id)
    description.insert(0, "source_file", source_file)
    description["missing_count"] = numeric.isna().sum().to_numpy()
    description["missing_rate"] = description["missing_count"] / max(len(numeric), 1)
    description["infinite_count"] = np.isinf(values).sum(axis=0)
    description["infinite_rate"] = description["infinite_count"] / max(len(numeric), 1)
    description["zero_count"] = ((values == 0) & np.isfinite(values)).sum(axis=0)
    description["zero_rate"] = description["zero_count"] / max(len(numeric), 1)
    description["n_unique"] = clean.nunique(dropna=True).to_numpy()
    return description


def extract_events(labels: np.ndarray, source_file: str, series_id: int) -> List[Dict[str, object]]:
    mask = labels == 1
    if not mask.any():
        return []
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1
    rows: List[Dict[str, object]] = []
    previous_end = -1
    for number, (start, end) in enumerate(zip(starts, ends), start=1):
        rows.append(
            {
                "source_file": source_file,
                "series_id": series_id,
                "event_number": number,
                "start_index_0based": int(start),
                "end_index_0based": int(end),
                "start_position_1based": int(start + 1),
                "end_position_1based": int(end + 1),
                "n_points": int(end - start + 1),
                "normal_gap_points_before": int(start - previous_end - 1),
            }
        )
        previous_end = int(end)
    return rows


def clean_values(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=np.float64, copy=False)
    return values[np.isfinite(values)]


def compare_groups(
    left: pd.DataFrame,
    right: pd.DataFrame,
    features: Sequence[str],
    source_file: str,
    series_id: int,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature in features:
        left_values = clean_values(left[feature])
        right_values = clean_values(right[feature])
        left_mean = float(np.mean(left_values)) if len(left_values) else np.nan
        right_mean = float(np.mean(right_values)) if len(right_values) else np.nan
        left_std = float(np.std(left_values, ddof=1)) if len(left_values) > 1 else np.nan
        if len(left_values) and len(right_values):
            ks = float(ks_2samp(left_values, right_values, method="asymp").statistic)
        else:
            ks = np.nan
        changed_with_zero_variance = bool(
            np.isfinite(left_std)
            and left_std == 0
            and np.isfinite(left_mean)
            and np.isfinite(right_mean)
            and left_mean != right_mean
        )
        standardized = (
            abs(right_mean - left_mean) / left_std
            if np.isfinite(left_std) and left_std > 0
            else np.nan
        )
        rows.append(
            {
                "source_file": source_file,
                "series_id": series_id,
                "feature": feature,
                "left_group": left_name,
                "right_group": right_name,
                "left_count": len(left_values),
                "right_count": len(right_values),
                "left_mean": left_mean,
                "right_mean": right_mean,
                "left_std": left_std,
                "mean_difference": right_mean - left_mean,
                "absolute_mean_difference_in_left_sd": standardized,
                "left_zero_variance_but_mean_changed": changed_with_zero_variance,
                "ks_statistic": ks,
            }
        )
    return pd.DataFrame(rows)


def lagged_correlation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return np.nan
    left = values[:-lag]
    right = values[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    left = left[valid]
    right = right[valid]
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def temporal_feature_rows(
    train: pd.DataFrame, source_file: str, series_id: int, features: Sequence[str]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    transitions: List[Dict[str, object]] = []
    autocorrelation: List[Dict[str, object]] = []
    for feature in features:
        values = train[feature].to_numpy(dtype=np.float64, copy=False)
        if len(values) > 1:
            valid = np.isfinite(values[:-1]) & np.isfinite(values[1:])
            comparable = int(valid.sum())
            changes = int(((values[:-1] != values[1:]) & valid).sum())
        else:
            comparable = changes = 0
        transitions.append(
            {
                "source_file": source_file,
                "series_id": series_id,
                "feature": feature,
                "comparable_pairs": comparable,
                "transition_count": changes,
                "transition_rate": changes / comparable if comparable else np.nan,
            }
        )
        for lag in AUTOCORRELATION_LAGS:
            autocorrelation.append(
                {
                    "source_file": source_file,
                    "series_id": series_id,
                    "feature": feature,
                    "lag_samples": lag,
                    "autocorrelation": lagged_correlation(values, lag),
                }
            )
    return transitions, autocorrelation


def event_feature_effects(
    numeric: pd.DataFrame,
    events: Sequence[Dict[str, object]],
    features: Sequence[str],
    max_context: int = 2000,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    clean = numeric.replace([np.inf, -np.inf], np.nan)
    for event in events:
        start = int(event["start_index_0based"])
        end = int(event["end_index_0based"])
        context = min(int(event["n_points"]), max_context)
        pre = clean.iloc[max(0, start - context) : start]
        during = clean.iloc[start : end + 1]
        post = clean.iloc[end + 1 : min(len(clean), end + 1 + context)]
        pre_mean = pre.mean()
        during_mean = during.mean()
        post_mean = post.mean()
        for feature in features:
            rows.append(
                {
                    "source_file": event["source_file"],
                    "series_id": event["series_id"],
                    "event_number": event["event_number"],
                    "start_index_0based": start,
                    "end_index_0based": end,
                    "n_points": event["n_points"],
                    "feature": feature,
                    "pre_window_points": len(pre),
                    "post_window_points": len(post),
                    "pre_mean": pre_mean[feature],
                    "during_mean": during_mean[feature],
                    "post_mean": post_mean[feature],
                }
            )
    return rows


def macro_comparison(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    grouped = frame.groupby("feature", as_index=False).agg(
        contributing_files=("source_file", "nunique"),
        mean_ks_statistic=("ks_statistic", "mean"),
        median_ks_statistic=("ks_statistic", "median"),
        min_ks_statistic=("ks_statistic", "min"),
        max_ks_statistic=("ks_statistic", "max"),
        mean_absolute_mean_shift_sd=("absolute_mean_difference_in_left_sd", "mean"),
        median_absolute_mean_shift_sd=("absolute_mean_difference_in_left_sd", "median"),
        zero_variance_change_files=("left_zero_variance_but_mean_changed", "sum"),
    )
    grouped.insert(0, "comparison", prefix)
    return grouped.sort_values(
        ["mean_ks_statistic", "max_ks_statistic"], ascending=False
    ).reset_index(drop=True)


def high_correlation_pairs(correlation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = correlation.columns.tolist()
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            value = correlation.iat[left, right]
            if np.isfinite(value) and abs(value) >= 0.95:
                rows.append(
                    {
                        "feature_1": columns[left],
                        "feature_2": columns[right],
                        "pearson_r": value,
                        "absolute_pearson_r": abs(value),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "absolute_pearson_r", ascending=False
    ) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "pearson_r", "absolute_pearson_r"]
    )


def pca_projection(
    samples: Dict[str, pd.DataFrame],
    features: Sequence[str],
    table_path: Path,
    figure_path: Path,
    seed: int,
) -> Dict[str, object]:
    combined = pd.concat(
        [samples[group] for group in ("train_prefix", "post_train_normal", "anomaly")],
        ignore_index=True,
    )
    train = samples["train_prefix"]
    train_values = train.loc[:, features].replace([np.inf, -np.inf], np.nan)
    medians = train_values.median()
    usable = [
        feature
        for feature in features
        if np.isfinite(medians[feature])
        and train_values[feature].fillna(medians[feature]).std(ddof=0) > 0
    ]
    if len(usable) < 2:
        write_csv(pd.DataFrame(), table_path)
        return {
            "usable_feature_count": len(usable),
            "pc1_explained_variance_ratio": np.nan,
            "pc2_explained_variance_ratio": np.nan,
        }
    scaler = StandardScaler()
    scaler.fit(train_values.loc[:, usable].fillna(medians[usable]))
    matrix = combined.loc[:, usable].replace([np.inf, -np.inf], np.nan).fillna(medians[usable])
    transformed = scaler.transform(matrix)
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(transformed)
    projection = combined.loc[
        :, ["group", "source_file", "series_id", "sample_index_0based"]
    ].copy()
    projection["PC1"] = coordinates[:, 0]
    projection["PC2"] = coordinates[:, 1]
    write_csv(projection, table_path)

    plot_frame = projection
    if len(plot_frame) > 50000:
        plot_frame = plot_frame.sample(50000, random_state=seed)
    colors = {
        "train_prefix": "#4C78A8",
        "post_train_normal": "#9C9C9C",
        "anomaly": "#E45756",
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    for group in ("train_prefix", "post_train_normal", "anomaly"):
        part = plot_frame[plot_frame["group"] == group]
        ax.scatter(
            part["PC1"],
            part["PC2"],
            s=7,
            alpha=0.25 if group != "anomaly" else 0.45,
            c=colors[group],
            label=group,
            rasterized=True,
        )
    ax.set_title("GHL PCA fitted on training-prefix samples")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, figure_path)
    return {
        "usable_feature_count": len(usable),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
    }


def plot_dataset_overview(
    file_summary: pd.DataFrame, events: pd.DataFrame, path: Path
) -> None:
    ordered = file_summary.sort_values("series_id")
    fig, ax = plt.subplots(figsize=(13, 10))
    for row_number, (_, row) in enumerate(ordered.iterrows()):
        y = row_number
        ax.broken_barh([(0, row["train_length"])], (y - 0.35, 0.7), color="#4C78A8")
        ax.broken_barh(
            [(row["train_length"], row["n_rows"] - row["train_length"])],
            (y - 0.35, 0.7),
            color="#D7D7D7",
        )
        subset = events[events["series_id"] == row["series_id"]]
        for _, event in subset.iterrows():
            ax.broken_barh(
                [(event["start_index_0based"], event["n_points"])],
                (y - 0.35, 0.7),
                color="#E45756",
            )
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([f"id {value}" for value in ordered["series_id"]])
    ax.set_xlabel("Sample index (zero based)")
    ax.set_title("GHL series layout: training prefix, evaluation region, anomaly events")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#4C78A8", label="training prefix"),
            plt.Rectangle((0, 0), 1, 1, color="#D7D7D7", label="post-train normal"),
            plt.Rectangle((0, 0), 1, 1, color="#E45756", label="anomaly"),
        ],
        loc="lower right",
    )
    save_figure(fig, path)


def plot_anomaly_rates(file_summary: pd.DataFrame, path: Path) -> None:
    ordered = file_summary.sort_values("series_id")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(ordered["series_id"].astype(str), ordered["anomaly_rate"] * 100, color="#E45756")
    ax.axhline(
        ordered["anomaly_count"].sum() / ordered["n_rows"].sum() * 100,
        color="black",
        linestyle="--",
        linewidth=1,
        label="pooled rate",
    )
    ax.set_xlabel("Series id")
    ax.set_ylabel("Anomaly points (%)")
    ax.set_title("GHL anomaly rate by separate series file")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, path)


def plot_label_timelines(
    file_summary: pd.DataFrame, events: pd.DataFrame, path: Path
) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(16, 12), sharex=False, sharey=True)
    for ax, (_, row) in zip(axes.flat, file_summary.sort_values("series_id").iterrows()):
        ax.axvspan(0, row["train_length"], color="#4C78A8", alpha=0.15)
        for _, event in events[events["series_id"] == row["series_id"]].iterrows():
            ax.axvspan(
                event["start_index_0based"],
                event["end_index_0based"] + 1,
                color="#E45756",
                alpha=0.9,
            )
        ax.set_xlim(0, row["n_rows"])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"id {int(row['series_id'])}", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Anomaly-event positions (red); training prefix shaded blue", y=1.01)
    fig.tight_layout()
    save_figure(fig, path)


def plot_event_durations(events: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(events["n_points"], bins=min(25, max(5, len(events) // 2)), color="#E45756")
    axes[0].set_xlabel("Event length (samples)")
    axes[0].set_ylabel("Event count")
    axes[0].set_title("Contiguous anomaly-event lengths")
    axes[0].grid(alpha=0.2)
    per_file = events.groupby("series_id")["n_points"].sum()
    axes[1].bar(per_file.index.astype(str), per_file.values, color="#F58518")
    axes[1].set_xlabel("Series id")
    axes[1].set_ylabel("Anomaly samples")
    axes[1].set_title("Total anomaly samples by series")
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].grid(axis="y", alpha=0.2)
    save_figure(fig, path)


def plot_missingness(feature_file: pd.DataFrame, path: Path) -> None:
    pivot = feature_file.pivot(index="series_id", columns="feature", values="missing_rate")
    fig, ax = plt.subplots(figsize=(13, 7))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"id {value}" for value in pivot.index])
    ax.set_title("Feature missing rate by series")
    fig.colorbar(image, ax=ax, label="Missing rate")
    if np.nanmax(pivot.to_numpy()) == 0:
        ax.text(
            0.5,
            0.5,
            "No missing feature values",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7, "pad": 8},
        )
    save_figure(fig, path)


def plot_ranked_effect(frame: pd.DataFrame, title: str, path: Path) -> None:
    top = frame.head(15).sort_values("mean_ks_statistic")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["mean_ks_statistic"], color="#E45756")
    ax.set_xlabel("Mean within-series KS statistic")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, path)


def plot_correlation(correlation: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(correlation.index)))
    ax.set_yticklabels(correlation.index, fontsize=8)
    ax.set_title("Pearson correlation on pooled training-prefix samples")
    fig.colorbar(image, ax=ax, label="Pearson r")
    save_figure(fig, path)


def plot_top_distributions(
    samples: Dict[str, pd.DataFrame], top_features: Sequence[str], path: Path
) -> None:
    selected = list(top_features[:6])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {
        "train_prefix": "#4C78A8",
        "post_train_normal": "#9C9C9C",
        "anomaly": "#E45756",
    }
    for ax, feature in zip(axes.flat, selected):
        combined_values = np.concatenate(
            [clean_values(samples[group][feature]) for group in colors]
        )
        if not len(combined_values):
            continue
        low, high = np.nanquantile(combined_values, [0.01, 0.99])
        if low == high:
            low, high = float(np.nanmin(combined_values)), float(np.nanmax(combined_values))
        for group, color in colors.items():
            values = clean_values(samples[group][feature])
            if len(values):
                ax.hist(
                    values,
                    bins=50,
                    range=(low, high) if low < high else None,
                    density=True,
                    histtype="step",
                    linewidth=1.5,
                    color=color,
                    label=group,
                )
        ax.set_title(feature, fontsize=9)
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Sampled distributions of top anomaly-associated features", y=1.02)
    fig.tight_layout()
    save_figure(fig, path)


def plot_event_feature_heatmap(event_effects: pd.DataFrame, path: Path) -> None:
    matrix = event_effects.pivot_table(
        index=["series_id", "event_number"],
        columns="feature",
        values="during_vs_pre_train_sd",
        aggfunc="first",
    )
    clipped = matrix.clip(-5, 5)
    fig, ax = plt.subplots(figsize=(13, 10))
    image = ax.imshow(clipped.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-5, vmax=5)
    ax.set_xticks(np.arange(len(clipped.columns)))
    ax.set_xticklabels(clipped.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(clipped.index)))
    ax.set_yticklabels(
        [f"{series}-{event}" for series, event in clipped.index], fontsize=6
    )
    ax.set_title("Event mean minus pre-event mean, scaled by pooled training SD")
    fig.colorbar(image, ax=ax, label="Standard deviations (clipped to +/-5)")
    save_figure(fig, path)


def plot_autocorrelation(autocorrelation: pd.DataFrame, path: Path) -> None:
    macro = autocorrelation.groupby(["feature", "lag_samples"])["autocorrelation"].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(macro.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(macro.columns)))
    ax.set_xticklabels(macro.columns)
    ax.set_yticks(np.arange(len(macro.index)))
    ax.set_yticklabels(macro.index, fontsize=8)
    ax.set_xlabel("Lag (samples)")
    ax.set_title("Mean training-prefix autocorrelation across series")
    fig.colorbar(image, ax=ax, label="Autocorrelation")
    save_figure(fig, path)


def plot_file_mean_shift(file_shift: pd.DataFrame, path: Path) -> None:
    matrix = file_shift.pivot(index="series_id", columns="feature", values="file_train_mean_vs_global_train_sd")
    clipped = matrix.clip(-4, 4)
    fig, ax = plt.subplots(figsize=(13, 8))
    image = ax.imshow(clipped.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
    ax.set_xticks(np.arange(len(clipped.columns)))
    ax.set_xticklabels(clipped.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(clipped.index)))
    ax.set_yticklabels([f"id {value}" for value in clipped.index])
    ax.set_title("Training-prefix file means relative to pooled training mean/SD")
    fig.colorbar(image, ax=ax, label="Standard deviations (clipped to +/-4)")
    save_figure(fig, path)


def build_reports(
    output_dir: Path,
    file_summary: pd.DataFrame,
    schema_validation: pd.DataFrame,
    feature_catalog: pd.DataFrame,
    group_stats: pd.DataFrame,
    events: pd.DataFrame,
    anomaly_macro: pd.DataFrame,
    anomaly_by_file: pd.DataFrame,
    drift_macro: pd.DataFrame,
    file_shift: pd.DataFrame,
    high_corr: pd.DataFrame,
    autocorrelation: pd.DataFrame,
    pca_info: Dict[str, object],
    sample_per_group: int,
) -> None:
    total_rows = int(file_summary["n_rows"].sum())
    train_rows = int(file_summary["train_length"].sum())
    anomaly_points = int(file_summary["anomaly_count"].sum())
    post_rows = total_rows - train_rows
    missing = int(file_summary["feature_missing_count"].sum())
    infinite = int(file_summary["feature_infinite_count"].sum())
    training_positives = int(file_summary["train_positive_count"].sum())
    constants = feature_catalog.loc[
        feature_catalog["global_cardinality_category"] == "constant", "feature"
    ].tolist()
    train_constants = feature_catalog.loc[
        feature_catalog["train_cardinality_category"] == "constant", "feature"
    ].tolist()
    order_mismatches = schema_validation.loc[
        ~schema_validation["column_order_matches_reference"], "series_id"
    ].astype(int).tolist()
    first_index_matches = int(file_summary["first_anomaly_matches_filename_zero_based"].sum())
    event_lengths = events["n_points"]
    file_table = file_summary.copy()
    file_table["anomaly_rate_pct"] = file_table["anomaly_rate"] * 100
    top_rate = file_summary.nlargest(5, "anomaly_rate").copy()
    top_rate["anomaly_rate_pct"] = top_rate["anomaly_rate"] * 100
    top_anomaly = anomaly_macro.head(10)
    top_drift = drift_macro.head(10)
    top_file_effect = anomaly_by_file.nlargest(12, "ks_statistic")
    shift_range = (
        file_shift.groupby("feature")["file_train_mean_vs_global_train_sd"]
        .agg(lambda values: values.max() - values.min())
        .sort_values(ascending=False)
        .head(10)
        .rename("file_mean_range_in_global_train_sd")
        .reset_index()
    )
    autocorr_macro = (
        autocorrelation.groupby(["feature", "lag_samples"], as_index=False)["autocorrelation"]
        .mean()
        .sort_values("autocorrelation", ascending=False)
    )
    strongest_lag1 = autocorr_macro[autocorr_macro["lag_samples"] == 1].head(10)
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    concise = f"""# GHL 파일 기반 EDA 보고서

생성 시각: {generated}

## 분석 범위

- 별도 시계열 파일: {len(file_summary)}개
- 입력 특성: {len(feature_catalog)}개, 라벨 열: `Label`
- 명시적 timestamp는 없으며 각 파일의 0-based 행 인덱스를 시간축으로 사용했다.
- 전체 관측치: {total_rows:,}개, 파일명 기준 학습 prefix: {train_rows:,}개, 이후 평가 구간: {post_rows:,}개
- 모든 파일을 별도 시계열 경계로 처리했으며 파일 경계를 가로지르는 window는 만들지 않았다.
- 파일들은 동일한 GHL 공정 모델과 실험 설계에서 유래하므로 통계적으로 독립적인 iid 표본으로 가정하지 않았다.

## 핵심 품질 및 라벨 결과

- 특성 결측값: {missing:,}개, 무한값: {infinite:,}개
- 학습 prefix 안의 양성 라벨: {training_positives:,}개
- 이상 포인트: {anomaly_points:,}개 ({anomaly_points / total_rows:.3%}), 연속 이상 이벤트: {len(events)}개
- 파일명의 `1st_*`가 실제 첫 이상 0-based index와 일치: {first_index_matches}/{len(file_summary)}개
- 기준 파일과 열 이름 집합은 전부 같지만 온도 열 순서가 다른 파일: {len(order_mismatches)}개 (id {', '.join(map(str, order_mismatches))})
- 전체 구간 상수 특성: {len(constants)}개 ({', '.join(constants) if constants else '없음'})
- 학습 prefix 상수 특성: {len(train_constants)}개 ({', '.join(train_constants) if train_constants else '없음'})

## 파일별 구조

{markdown_table(file_table, ['series_id', 'n_rows', 'train_length', 'evaluation_rows', 'first_anomaly_index_0based', 'anomaly_count', 'anomaly_rate_pct', 'event_count'])}

## 정상/이상 분포 차이

각 파일의 학습 prefix 이후 구간 안에서 `Label=0`과 `Label=1`을 비교한 KS 통계량을 파일 동일 가중치로 평균했다. 이는 연관성 탐색 지표이며 이상 원인이나 검출 성능을 의미하지 않는다.

{markdown_table(top_anomaly, ['feature', 'contributing_files', 'mean_ks_statistic', 'median_ks_statistic', 'max_ks_statistic', 'mean_absolute_mean_shift_sd'])}

## 학습 prefix와 이후 정상 구간의 drift

파일별 두 그룹에서 각각 최대 {sample_per_group:,}개를 결정론적으로 표본 추출해 KS를 계산했다.

{markdown_table(top_drift, ['feature', 'contributing_files', 'mean_ks_statistic', 'median_ks_statistic', 'max_ks_statistic'])}

## 다변량 및 시간 특성

- 학습 prefix 표본에서 |Pearson r| >= 0.95인 특성 쌍: {len(high_corr):,}개
- PCA 사용 가능 특성: {pca_info['usable_feature_count']}개
- PCA 설명분산: PC1 {pca_info['pc1_explained_variance_ratio']:.2%}, PC2 {pca_info['pc2_explained_variance_ratio']:.2%}
- 1, 10, 100, 1000 sample lag 자기상관은 `tables/autocorrelation_train.csv`에 기록했다.

## 해석 및 모델링 주의사항

- timestamp와 실제 샘플링 주기가 없으므로 이벤트 길이는 시간 단위가 아니라 sample 수다.
- 파일은 계산상 분리된 시계열 경계다. 파일 간 연결, lag, window 생성은 허용되지 않는다.
- 동일한 GHL 공정 모델에서 생성된 반복 시계열이므로 파일 사이의 통계적 독립성은 가정하지 않는다.
- 파일명 `tr_n`은 앞쪽 n개 행의 학습 prefix 길이로 사용했다. 이 구간에는 양성 라벨이 없다.
- 파일명 `1st_n`은 1-based 위치가 아니라 0-based index다.
- 열 순서 변형이 있으므로 NumPy 위치 기반 로딩 전에 반드시 열 이름으로 재정렬해야 한다.
- 정상/이상 분포 차이는 시점·운전 상태·파일 차이의 영향을 받을 수 있어 인과관계로 해석하면 안 된다.
- 스케일과 파일별 baseline 차이가 있으므로 학습 prefix만으로 fit한 scaling과 파일 단위 평가를 권장한다.

## 주요 산출물

- `tables/file_summary.csv`: 파일명 메타정보, 크기, 라벨, 첫 이상 정합성
- `tables/schema_validation.csv`: 열 집합 및 열 순서 검증
- `tables/feature_catalog.csv`: 특성 유형과 cardinality
- `tables/feature_file_statistics.csv`: 파일·특성별 전체 행 기술통계
- `tables/feature_group_statistics.csv`: 학습/이후 정상/이상 그룹의 정확 통계
- `tables/anomaly_events.csv`: 연속 이상 이벤트 경계와 길이
- `tables/anomaly_feature_effect_by_file.csv`: 파일 내부 정상/이상 비교
- `tables/anomaly_feature_effect_macro.csv`: 파일 동일 가중치 정상/이상 비교 요약
- `tables/train_postnormal_drift_*.csv`: 학습 prefix/이후 정상 drift
- `tables/event_feature_effects.csv`: 이벤트 직전·중·직후 평균 변화
- `tables/autocorrelation_train.csv`: 학습 prefix 자기상관
- `tables/train_sample_correlation.csv`: 학습 prefix 표본 상관행렬
- `tables/file_feature_mean_shift.csv`: 파일별 학습 baseline 차이
- `tables/pca_projection_sample.csv`: PCA 표본 좌표
- `figures/`: 핵심 시각화
"""
    (output_dir / "GHL_EDA_REPORT.md").write_text(concise, encoding="utf-8")

    notion = f"""# GHL 데이터셋 탐색적 데이터 분석(EDA) 보고서

> **분석 대상:** 로컬 `TSB-AD-M`의 GHL 25개 CSV  
> **분석 축:** 데이터 구조, 파일명 메타데이터, 품질, 라벨 이벤트, 분포 이동, 다변량·시간 특성  
> **중요 전제:** timestamp가 없으므로 sample index를 시간축으로 쓰며, 각 파일은 연결하지 않는 별도 시계열 경계로 처리한다. 이는 통계적 독립성을 뜻하지 않는다.

---

## 1. 핵심 요약

GHL은 19개 입력 특성과 하나의 이진 `Label`을 가진 별도 다변량 시계열 파일 25개다. 전체 {total_rows:,}개 관측치 중 파일명으로 지정된 학습 prefix는 {train_rows:,}개이고, 이후 평가 구간은 {post_rows:,}개다. 이상 포인트는 {anomaly_points:,}개({anomaly_points / total_rows:.3%})이며 연속된 이상 구간으로 묶으면 {len(events)}개 이벤트다. 각 파일은 계산 경계로는 분리하지만, 동일한 GHL 공정 모델과 실험 설계에서 유래하므로 통계적으로 독립적인 iid 표본으로 가정하지 않는다.

데이터 품질 측면에서 특성 결측값은 {missing:,}개, 무한값은 {infinite:,}개다. 모든 학습 prefix의 라벨은 0이다. 25개 파일 모두 파일명의 `1st_*` 값이 실제 첫 이상 포인트의 **0-based index**와 정확히 일치한다. 사람이 행 번호를 1부터 셀 경우 실제 위치는 파일명 값보다 1 크다.

모든 파일의 열 이름 집합은 같지만 id {', '.join(map(str, order_mismatches))}의 {len(order_mismatches)}개 파일은 `C_temperature.T`와 `HT_temperature.T`의 열 순서가 기준 파일과 반대다. Pandas처럼 이름으로 선택하면 문제가 없지만, 원시 배열의 열 위치를 고정해서 읽으면 두 온도 특성이 뒤바뀔 수 있다.

### 핵심 수치

| 항목 | 결과 |
| --- | ---: |
| 별도 시계열 파일 | {len(file_summary)}개 |
| 입력 특성 | {len(feature_catalog)}개 |
| 전체 관측치 | {total_rows:,}개 |
| 학습 prefix | {train_rows:,}개 |
| 평가 구간 | {post_rows:,}개 |
| 이상 포인트 | {anomaly_points:,}개 ({anomaly_points / total_rows:.3%}) |
| 연속 이상 이벤트 | {len(events)}개 |
| 이벤트 길이 | 최소 {event_lengths.min():.0f}, 중앙값 {event_lengths.median():.0f}, 최대 {event_lengths.max():.0f} samples |
| 결측값 / 무한값 | {missing:,} / {infinite:,} |
| 열 순서 변형 파일 | {len(order_mismatches)}개 |

![Dataset overview](figures/01_dataset_overview.png)

---

## 2. 데이터 구성과 index 의미

각 파일명은 `GHL_id_<id>_Sensor_tr_<n>_1st_<m>.csv` 형식이다.

- `id`: 별도 시계열 파일 식별자
- `tr_n`: 앞에서부터 n개 행이 학습 prefix
- `1st_m`: 첫 이상 포인트의 0-based index
- CSV 행 자체에는 timestamp가 없으므로 sample 간 실제 시간 간격은 파일만으로 알 수 없다.

따라서 `n_points=1000`인 이벤트를 1000초 또는 특정 시간으로 바꾸어 해석할 근거는 없다. 본 보고서는 모든 길이와 lag를 sample 단위로 기록한다.

{markdown_table(file_table, ['series_id', 'n_rows', 'train_length', 'evaluation_rows', 'first_anomaly_index_0based', 'clean_gap_after_train', 'anomaly_count', 'anomaly_rate_pct', 'event_count'])}

id 19를 제외한 파일은 200,001행이고 id 19는 175,001행이다. 학습 prefix는 대부분 50,000행이며 id 12는 39,938행, id 19는 43,750행이다.

![Anomaly rates](figures/02_anomaly_rates.png)

---

## 3. 스키마 및 데이터 품질

### 3.1 스키마

19개 특성 이름과 `Label` 열은 모든 파일에 존재한다. 단, {len(order_mismatches)}개 파일에서 두 온도 열의 순서가 바뀐다. 본 EDA는 기준 열 이름 목록으로 모든 프레임을 재정렬한 뒤 계산했다.

### 3.2 결측·무한값·상수 특성

- 특성 결측값: {missing:,}개
- 특성 무한값: {infinite:,}개
- 전체 범위 상수 특성: {len(constants)}개 — {', '.join(constants) if constants else '없음'}
- 학습 prefix 상수 특성: {len(train_constants)}개 — {', '.join(train_constants) if train_constants else '없음'}

`feature_catalog.csv`의 cardinality는 최대 200개까지 정확히 추적하며, 그보다 많으면 `high_cardinality`로 분류한다. 상수나 저카디널리티 특성은 모델에 무조건 제거할 대상이라기보다 상태 신호일 수 있으므로, 학습 구간과 평가 구간의 support 차이를 함께 확인해야 한다.

![Missingness](figures/05_missingness.png)

---

## 4. 라벨과 이상 이벤트 구조

학습 prefix 안의 양성 라벨은 {training_positives:,}개로, 모든 파일에서 0이다. 이상 비율은 파일마다 다르며 가장 높은 5개 파일은 다음과 같다.

{markdown_table(top_rate, ['series_id', 'anomaly_count', 'anomaly_rate_pct', 'event_count', 'first_anomaly_index_0based'])}

연속된 `Label=1`을 하나의 이벤트로 묶으면 총 {len(events)}개다. 이벤트 길이는 최소 {event_lengths.min():.0f}, 중앙값 {event_lengths.median():.0f}, 최대 {event_lengths.max():.0f} samples다. 포인트 단위 클래스 불균형과 이벤트 단위 불균형은 다르므로 이후 평가는 둘을 구분해야 한다.

![Label timelines](figures/03_label_timelines.png)

![Event durations](figures/04_event_durations.png)

---

## 5. 이후 정상 구간과 이상 구간의 특성 차이

파일별로 학습 prefix 이후의 `Label=0`과 `Label=1` 전체 행을 비교하고, KS 통계량을 25개 파일 동일 가중치로 요약했다. pooled 비교만 사용하면 특정 파일의 크기와 baseline이 결과를 지배할 수 있어 macro 평균을 주 결과로 사용했다.

{markdown_table(top_anomaly, ['feature', 'contributing_files', 'mean_ks_statistic', 'median_ks_statistic', 'min_ks_statistic', 'max_ks_statistic', 'mean_absolute_mean_shift_sd'])}

파일·특성 조합 중 KS가 가장 큰 사례는 다음과 같다. 동일 특성이 모든 파일에서 같은 정도로 반응한다고 가정하면 안 된다.

{markdown_table(top_file_effect, ['series_id', 'feature', 'left_count', 'right_count', 'ks_statistic', 'absolute_mean_difference_in_left_sd'])}

![Anomaly feature effect](figures/06_anomaly_feature_effect.png)

![Top distributions](figures/10_top_feature_distributions.png)

이 분석은 라벨과의 분포 연관성을 보여주지만, 해당 특성이 이상을 유발했다거나 라벨 시작 즉시 반응했다는 의미는 아니다. `event_feature_effects.csv`에는 각 이벤트 길이와 같은 크기(최대 2,000 samples)의 직전·직후 window를 이용한 평균 변화가 별도로 기록되어 있다.

![Event feature heatmap](figures/11_event_feature_heatmap.png)

---

## 6. 학습 prefix와 이후 정상 구간의 분포 이동

각 파일 안에서 학습 prefix와 이후 `Label=0` 구간을 비교했다. 계산량과 파일 크기 영향의 균형을 위해 그룹별 최대 {sample_per_group:,}행의 결정론적 표본을 사용했고, 파일별 KS를 동일 가중치로 집계했다.

{markdown_table(top_drift, ['feature', 'contributing_files', 'mean_ks_statistic', 'median_ks_statistic', 'min_ks_statistic', 'max_ks_statistic'])}

![Train post-normal drift](figures/07_train_postnormal_drift.png)

파일별 학습 prefix 평균도 서로 다를 수 있다. 아래 특성들은 파일 평균의 최대-최소 범위가 pooled 학습 표준편차 단위로 큰 순서다.

{markdown_table(shift_range, ['feature', 'file_mean_range_in_global_train_sd'])}

![File mean shift](figures/13_file_mean_shift.png)

이 결과는 global scaler보다 학습 prefix에만 fit한 scaler, 파일 단위 분할, leave-one-series-out 검증을 고려할 이유가 된다. 단, 실제 벤치마크 프로토콜은 별도로 확인해야 한다.

---

## 7. 다변량 구조와 시간 의존성

학습 prefix의 결정론적 표본으로 Pearson 상관을 계산했다. |r| ≥ 0.95인 특성 쌍은 {len(high_corr):,}개다.

{markdown_table(high_corr.head(10), ['feature_1', 'feature_2', 'pearson_r', 'absolute_pearson_r']) if len(high_corr) else '강한 상관 특성 쌍이 없다.'}

![Training correlation](figures/08_train_correlation.png)

PCA는 학습 prefix 표본으로 scaling과 축을 fit한 뒤 이후 정상 및 이상 표본을 투영했다. 사용 가능 특성은 {pca_info['usable_feature_count']}개이며 PC1과 PC2 설명분산은 각각 {pca_info['pc1_explained_variance_ratio']:.2%}, {pca_info['pc2_explained_variance_ratio']:.2%}다.

![PCA](figures/09_pca_projection.png)

학습 prefix에서 계산한 lag별 평균 자기상관 상위 특성은 다음과 같다. lag는 실제 시간이 아닌 sample 수다.

{markdown_table(strongest_lag1, ['feature', 'lag_samples', 'autocorrelation'])}

![Autocorrelation](figures/12_autocorrelation.png)

---

## 8. 모델링 전 권고사항

1. **파일 경계 유지:** 서로 다른 id를 연결해 window를 만들지 않는다. 이는 통계적 독립성 가정과 구분한다.
2. **열 이름 기준 정렬:** 온도 열 순서 변형 때문에 배열 위치를 그대로 신뢰하지 않는다.
3. **학습 prefix로만 전처리 fit:** 전체 파일에 fit하면 미래 평가 구간 정보가 누출될 수 있다.
4. **sample과 event 평가 병행:** point-wise metric만으로 긴 이벤트에 유리한 결과를 만들지 않는다.
5. **파일별 결과 보고:** pooled 점수와 함께 id별 점수 및 macro 평균을 제시한다.
6. **timestamp 해석 금지:** 샘플링 주기 근거가 확보되기 전에는 sample을 초·분으로 바꾸지 않는다.
7. **첫 이상 index 기준 명시:** `1st_*`를 0-based로 처리하고, 1-based 행 위치와 혼동하지 않는다.

---

## 9. 재현성과 산출물

분석은 `eda/ghl/run_ghl_eda.py`로 재실행할 수 있다. 정확 통계·결측·무한값·이벤트·파일 내부 정상/이상 KS는 전체 행을 사용했다. PCA·상관·학습/이후 정상 drift는 seed 고정 표본을 사용한다. 구체적인 옵션과 입력 파일은 `run_metadata.json`에 기록한다.

핵심 CSV와 그림 목록은 간결 보고서 `GHL_EDA_REPORT.md`의 산출물 절을 참고한다.
"""
    # Keep the generated report separate from the manually edited Notion
    # handoff so rerunning EDA does not overwrite editorial changes.
    (output_dir / "GHL_EDA_NOTION_REPORT_AUTO.md").write_text(notion, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        data_dir.glob("*_GHL_id_*_Sensor_tr_*_1st_*.csv"),
        key=lambda path: parse_filename(path)["series_id"],
    )
    if not files:
        raise FileNotFoundError(f"No GHL CSV files found under {data_dir}")

    headers = {path.name: pd.read_csv(path, nrows=0).columns.tolist() for path in files}
    reference = headers[files[0].name]
    if LABEL_COLUMN not in reference:
        raise ValueError(f"Reference file has no {LABEL_COLUMN!r} column")
    features = [column for column in reference if column != LABEL_COLUMN]
    reference_set = set(reference)
    schema_rows: List[Dict[str, object]] = []
    for path in files:
        metadata = parse_filename(path)
        header = headers[path.name]
        schema_rows.append(
            {
                "source_file": path.name,
                "series_id": metadata["series_id"],
                "column_count": len(header),
                "column_set_matches_reference": set(header) == reference_set,
                "column_order_matches_reference": header == reference,
                "missing_columns": ", ".join(sorted(reference_set - set(header))),
                "extra_columns": ", ".join(sorted(set(header) - reference_set)),
                "column_order": " | ".join(header),
            }
        )
    schema_validation = pd.DataFrame(schema_rows).sort_values("series_id")
    if not schema_validation["column_set_matches_reference"].all():
        bad = schema_validation.loc[
            ~schema_validation["column_set_matches_reference"], "source_file"
        ].tolist()
        raise ValueError(f"GHL feature schema differs in: {bad}")

    rng = np.random.default_rng(args.seed)
    stats = {
        group: RunningStats(features)
        for group in (
            "all_data",
            "train_prefix",
            "post_train_all",
            "post_train_normal",
            "anomaly",
        )
    }
    samples_parts: Dict[str, List[pd.DataFrame]] = {
        "train_prefix": [],
        "post_train_normal": [],
        "anomaly": [],
    }
    cardinality: Dict[str, Optional[set]] = {feature: set() for feature in features}
    train_cardinality: Dict[str, Optional[set]] = {feature: set() for feature in features}
    file_rows: List[Dict[str, object]] = []
    feature_file_parts: List[pd.DataFrame] = []
    event_rows: List[Dict[str, object]] = []
    event_effect_rows: List[Dict[str, object]] = []
    anomaly_parts: List[pd.DataFrame] = []
    drift_parts: List[pd.DataFrame] = []
    transition_rows: List[Dict[str, object]] = []
    autocorrelation_rows: List[Dict[str, object]] = []
    file_train_stats: List[pd.DataFrame] = []

    for path in files:
        metadata = parse_filename(path)
        series_id = metadata["series_id"]
        print(f"Loading GHL id {series_id}: {path.name}", flush=True)
        raw = pd.read_csv(path)
        raw = raw.loc[:, features + [LABEL_COLUMN]]
        numeric = raw.loc[:, features].apply(pd.to_numeric, errors="coerce")
        labels_series = pd.to_numeric(raw[LABEL_COLUMN], errors="coerce")
        if labels_series.isna().any():
            raise ValueError(f"Missing or non-numeric labels in {path.name}")
        label_values = labels_series.to_numpy(dtype=np.int64)
        unexpected = sorted(set(np.unique(label_values)) - {0, 1})
        if unexpected:
            raise ValueError(f"Unexpected labels in {path.name}: {unexpected}")
        n_rows = len(numeric)
        train_length = metadata["train_length"]
        if not 0 < train_length < n_rows:
            raise ValueError(f"Invalid training prefix in {path.name}: {train_length}")
        train_mask = np.arange(n_rows) < train_length
        post_mask = ~train_mask
        normal_mask = post_mask & (label_values == 0)
        anomaly_mask = label_values == 1
        train = numeric.iloc[:train_length]
        post_normal = numeric.loc[normal_mask]
        anomaly = numeric.loc[anomaly_mask]

        stats["all_data"].update(numeric)
        stats["train_prefix"].update(train)
        stats["post_train_all"].update(numeric.iloc[train_length:])
        stats["post_train_normal"].update(post_normal)
        stats["anomaly"].update(anomaly)
        update_cardinality(cardinality, numeric, features)
        update_cardinality(train_cardinality, train, features)

        samples = {
            "train_prefix": deterministic_sample(
                numeric,
                train_mask,
                path.name,
                series_id,
                "train_prefix",
                args.sample_per_group,
                rng,
            ),
            "post_train_normal": deterministic_sample(
                numeric,
                normal_mask,
                path.name,
                series_id,
                "post_train_normal",
                args.sample_per_group,
                rng,
            ),
            "anomaly": deterministic_sample(
                numeric,
                anomaly_mask,
                path.name,
                series_id,
                "anomaly",
                args.sample_per_group,
                rng,
            ),
        }
        for group, sample in samples.items():
            samples_parts[group].append(sample)

        events = extract_events(label_values, path.name, series_id)
        event_rows.extend(events)
        event_effect_rows.extend(event_feature_effects(numeric, events, features))
        feature_file_parts.append(exact_feature_statistics(numeric, path.name, series_id))
        anomaly_parts.append(
            compare_groups(
                post_normal,
                anomaly,
                features,
                path.name,
                series_id,
                "post_train_normal",
                "anomaly",
            )
        )
        drift_parts.append(
            compare_groups(
                samples["train_prefix"].loc[:, features],
                samples["post_train_normal"].loc[:, features],
                features,
                path.name,
                series_id,
                "train_prefix_sample",
                "post_train_normal_sample",
            )
        )
        transitions, autocorrelations = temporal_feature_rows(
            train, path.name, series_id, features
        )
        transition_rows.extend(transitions)
        autocorrelation_rows.extend(autocorrelations)
        train_stats = exact_feature_statistics(train, path.name, series_id)
        file_train_stats.append(
            train_stats.loc[:, ["source_file", "series_id", "feature", "mean", "std"]]
        )

        anomaly_indices = np.flatnonzero(anomaly_mask)
        first_actual = int(anomaly_indices[0]) if len(anomaly_indices) else -1
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        file_rows.append(
            {
                "source_file": path.name,
                "series_id": series_id,
                "filename_prefix_number": metadata["prefix"],
                "n_rows": n_rows,
                "feature_count": len(features),
                "train_length": train_length,
                "evaluation_rows": n_rows - train_length,
                "filename_first_anomaly": metadata["first_anomaly"],
                "first_anomaly_index_0based": first_actual,
                "first_anomaly_position_1based": first_actual + 1 if first_actual >= 0 else -1,
                "first_anomaly_matches_filename_zero_based": first_actual == metadata["first_anomaly"],
                "first_anomaly_matches_filename_one_based": first_actual + 1 == metadata["first_anomaly"],
                "clean_gap_after_train": first_actual - train_length if first_actual >= 0 else np.nan,
                "train_positive_count": int((label_values[:train_length] == 1).sum()),
                "normal_count": int((label_values == 0).sum()),
                "anomaly_count": int(anomaly_mask.sum()),
                "anomaly_rate": float(anomaly_mask.mean()),
                "event_count": len(events),
                "label_missing_count": int(labels_series.isna().sum()),
                "label_unique_values": "|".join(map(str, sorted(np.unique(label_values)))),
                "feature_missing_count": int(np.isnan(values).sum()),
                "feature_infinite_count": int(np.isinf(values).sum()),
                "duplicate_full_row_count": int(raw.duplicated().sum()),
                "column_order_matches_reference": headers[path.name] == reference,
            }
        )
        del raw, numeric, train, post_normal, anomaly

    file_summary = pd.DataFrame(file_rows).sort_values("series_id").reset_index(drop=True)
    feature_file = pd.concat(feature_file_parts, ignore_index=True)
    events_frame = pd.DataFrame(event_rows).sort_values(
        ["series_id", "event_number"]
    ).reset_index(drop=True)
    event_effects = pd.DataFrame(event_effect_rows)
    anomaly_by_file = pd.concat(anomaly_parts, ignore_index=True)
    anomaly_macro = macro_comparison(anomaly_by_file, "post_train_normal_vs_anomaly")
    drift_by_file = pd.concat(drift_parts, ignore_index=True)
    drift_macro = macro_comparison(drift_by_file, "train_prefix_vs_post_train_normal")
    transitions = pd.DataFrame(transition_rows)
    autocorrelation = pd.DataFrame(autocorrelation_rows)
    group_stats = pd.concat(
        [stats[group].finish(group) for group in stats], ignore_index=True
    )
    samples = {
        group: pd.concat(parts, ignore_index=True)
        for group, parts in samples_parts.items()
    }

    feature_rows = []
    for feature in features:
        global_category, global_nunique, global_exact = classify_cardinality(cardinality[feature])
        train_category, train_nunique, train_exact = classify_cardinality(train_cardinality[feature])
        feature_rows.append(
            {
                "feature": feature,
                "feature_family": feature_family(feature),
                "global_cardinality_category": global_category,
                "global_n_unique_or_lower_bound": global_nunique,
                "global_n_unique_exact": global_exact,
                "train_cardinality_category": train_category,
                "train_n_unique_or_lower_bound": train_nunique,
                "train_n_unique_exact": train_exact,
            }
        )
    feature_catalog = pd.DataFrame(feature_rows)

    train_correlation = samples["train_prefix"].loc[:, features].replace(
        [np.inf, -np.inf], np.nan
    ).corr()
    high_corr = high_correlation_pairs(train_correlation)
    global_train = group_stats[group_stats["group"] == "train_prefix"].set_index("feature")
    file_shift = pd.concat(file_train_stats, ignore_index=True)
    file_shift = file_shift.rename(columns={"mean": "file_train_mean", "std": "file_train_std"})
    file_shift["global_train_mean"] = file_shift["feature"].map(global_train["mean"])
    file_shift["global_train_std"] = file_shift["feature"].map(global_train["std"])
    file_shift["file_train_mean_vs_global_train_sd"] = np.divide(
        file_shift["file_train_mean"] - file_shift["global_train_mean"],
        file_shift["global_train_std"],
        out=np.full(len(file_shift), np.nan),
        where=file_shift["global_train_std"].to_numpy() > 0,
    )
    event_effects["global_train_std"] = event_effects["feature"].map(global_train["std"])
    event_effects["during_vs_pre_train_sd"] = np.divide(
        event_effects["during_mean"] - event_effects["pre_mean"],
        event_effects["global_train_std"],
        out=np.full(len(event_effects), np.nan),
        where=event_effects["global_train_std"].to_numpy() > 0,
    )
    event_effects["post_vs_pre_train_sd"] = np.divide(
        event_effects["post_mean"] - event_effects["pre_mean"],
        event_effects["global_train_std"],
        out=np.full(len(event_effects), np.nan),
        where=event_effects["global_train_std"].to_numpy() > 0,
    )

    write_csv(file_summary, table_dir / "file_summary.csv")
    write_csv(schema_validation, table_dir / "schema_validation.csv")
    write_csv(feature_catalog, table_dir / "feature_catalog.csv")
    write_csv(feature_file, table_dir / "feature_file_statistics.csv")
    write_csv(group_stats, table_dir / "feature_group_statistics.csv")
    write_csv(events_frame, table_dir / "anomaly_events.csv")
    write_csv(anomaly_by_file, table_dir / "anomaly_feature_effect_by_file.csv")
    write_csv(anomaly_macro, table_dir / "anomaly_feature_effect_macro.csv")
    write_csv(drift_by_file, table_dir / "train_postnormal_drift_by_file.csv")
    write_csv(drift_macro, table_dir / "train_postnormal_drift_macro.csv")
    write_csv(event_effects, table_dir / "event_feature_effects.csv")
    write_csv(transitions, table_dir / "state_transitions_train.csv")
    write_csv(autocorrelation, table_dir / "autocorrelation_train.csv")
    train_correlation.to_csv(table_dir / "train_sample_correlation.csv", encoding="utf-8-sig")
    write_csv(high_corr, table_dir / "high_correlation_pairs.csv")
    write_csv(file_shift, table_dir / "file_feature_mean_shift.csv")

    plot_dataset_overview(file_summary, events_frame, figure_dir / "01_dataset_overview.png")
    plot_anomaly_rates(file_summary, figure_dir / "02_anomaly_rates.png")
    plot_label_timelines(file_summary, events_frame, figure_dir / "03_label_timelines.png")
    plot_event_durations(events_frame, figure_dir / "04_event_durations.png")
    plot_missingness(feature_file, figure_dir / "05_missingness.png")
    plot_ranked_effect(
        anomaly_macro,
        "Largest within-series shifts: post-train normal vs anomaly",
        figure_dir / "06_anomaly_feature_effect.png",
    )
    plot_ranked_effect(
        drift_macro,
        "Largest within-series shifts: training prefix vs post-train normal",
        figure_dir / "07_train_postnormal_drift.png",
    )
    plot_correlation(train_correlation, figure_dir / "08_train_correlation.png")
    pca_info = pca_projection(
        samples,
        features,
        table_dir / "pca_projection_sample.csv",
        figure_dir / "09_pca_projection.png",
        args.seed,
    )
    plot_top_distributions(
        samples,
        anomaly_macro["feature"].tolist(),
        figure_dir / "10_top_feature_distributions.png",
    )
    plot_event_feature_heatmap(event_effects, figure_dir / "11_event_feature_heatmap.png")
    plot_autocorrelation(autocorrelation, figure_dir / "12_autocorrelation.png")
    plot_file_mean_shift(file_shift, figure_dir / "13_file_mean_shift.png")

    build_reports(
        output_dir,
        file_summary,
        schema_validation,
        feature_catalog,
        group_stats,
        events_frame,
        anomaly_macro,
        anomaly_by_file,
        drift_macro,
        file_shift,
        high_corr,
        autocorrelation,
        pca_info,
        args.sample_per_group,
    )

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "data_directory": str(data_dir),
        "output_directory": str(output_dir),
        "seed": args.seed,
        "sample_per_group_per_file": args.sample_per_group,
        "file_count": len(files),
        "feature_count": len(features),
        "input_files": [path.name for path in files],
        "analysis_groups": list(stats),
        "notes": [
            "Files are separate time-series boundaries with no explicit timestamp; statistical independence is not assumed.",
            "All files are reordered by column name before analysis.",
            "Exact moments, quality counts, events, and within-file normal/anomaly KS use all rows.",
            "PCA, correlation, and train/post-train drift use deterministic capped samples.",
            "Filename first-anomaly metadata is interpreted and validated as a zero-based index.",
            "No model training or reduced training subsets were created.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"GHL EDA complete. Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
