"""TSB 비-GHL tuning 18개 파일의 Role-A 근거를 다시 만든다."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy
import pandas
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.execution_identity import file_sha256


HIGH_CORRELATION_THRESHOLD = 0.995
ACF_MAXIMUM_LAG = 400
ACF_FALLBACK_PERIOD = 125
DEFAULT_DATA_ROOT = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project"
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "checks" / "datasets" / "dev18"
)
CSV_CHUNK_SIZE = 20_000


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _anomaly_lengths(labels: numpy.ndarray) -> list[int]:
    padded = numpy.pad(labels.astype(numpy.int8), (1, 1))
    changes = numpy.flatnonzero(numpy.diff(padded))
    return (changes[1::2] - changes[::2]).tolist()


def estimate_training_period(values) -> int:
    """공식 TSB-AD period 규칙을 training-only 첫 feature에 적용한다."""
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 1 or not len(values) or not numpy.isfinite(values).all():
        raise ValueError("period 입력은 비어 있지 않은 유한 1차원 배열이어야 한다")
    if len(values) < 4:
        return ACF_FALLBACK_PERIOD
    centered = values[: min(20_000, len(values))] - numpy.mean(
        values[: min(20_000, len(values))]
    )
    if numpy.ptp(centered) == 0:
        return ACF_FALLBACK_PERIOD
    transform_length = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = numpy.fft.rfft(centered, transform_length)
    autocovariance = numpy.fft.irfft(
        spectrum * numpy.conjugate(spectrum), transform_length,
    )[: min(ACF_MAXIMUM_LAG, len(centered) - 1) + 1]
    autocorrelation = autocovariance / autocovariance[0]
    candidates = numpy.flatnonzero(
        (autocorrelation[3:-1] > autocorrelation[2:-2])
        & (autocorrelation[3:-1] > autocorrelation[4:])
    ) + 3
    if not len(candidates):
        return ACF_FALLBACK_PERIOD
    period = int(candidates[numpy.argmax(autocorrelation[candidates])])
    return ACF_FALLBACK_PERIOD if period > 300 else period


def _summarize_dev18(
    file_name: str,
    feature_names: list,
    training: pandas.DataFrame,
    labels: numpy.ndarray,
    *,
    training_boundary: int,
    row_count: int,
    column_count: int,
    missing_count: int,
    nonfinite_count: int,
    duplicate_timestamp_count,
) -> tuple[dict, list[dict], list[dict]]:
    if not 0 < training_boundary < row_count:
        raise ValueError(f"학습 경계가 행 범위를 벗어났다: {file_name}")
    training_labels = labels[:training_boundary]
    training_anomaly_count = int(training_labels.sum())
    test_labels = labels[training_boundary:]
    anomaly_lengths = _anomaly_lengths(test_labels)
    anomaly_positions = numpy.flatnonzero(labels)
    observed_first_anomaly = (
        int(anomaly_positions[0]) if len(anomaly_positions) else None
    )
    first_anomaly_token = int(file_name.split("_1st_")[1].split(".csv")[0])
    acf_candidate_lag = estimate_training_period(
        training[feature_names[0]].to_numpy(dtype=numpy.float64)
    )

    channel_rows = []
    constant_count = 0
    iqr_zero_count = 0
    for position, feature_name in enumerate(feature_names, start=1):
        series = training[feature_name]
        first_quartile = float(series.quantile(0.25))
        third_quartile = float(series.quantile(0.75))
        interquartile_range = third_quartile - first_quartile
        constant = int(series.nunique(dropna=False)) <= 1
        iqr_zero = interquartile_range == 0.0
        constant_count += int(constant)
        iqr_zero_count += int(iqr_zero)
        channel_rows.append({
            "position": position,
            "feature_name": str(feature_name),
            "minimum": float(series.min()),
            "first_quartile": first_quartile,
            "median": float(series.median()),
            "third_quartile": third_quartile,
            "maximum": float(series.max()),
            "interquartile_range": interquartile_range,
            "unique_value_count": int(series.nunique()),
            "constant": constant,
            "iqr_zero": iqr_zero,
        })

    correlation = training.corr(method="pearson").abs()
    correlation_rows = []
    for left_index, left in enumerate(feature_names):
        for right in feature_names[left_index + 1:]:
            value = correlation.at[left, right]
            if pandas.notna(value) and value >= HIGH_CORRELATION_THRESHOLD:
                correlation_rows.append({
                    "left": str(left),
                    "right": str(right),
                    "absolute_correlation": round(float(value), 12),
                })

    timestamp_names = [
        name for name in feature_names if str(name).strip().lower() == "timestamp"
    ]
    duplicate_timestamp_count = (
        duplicate_timestamp_count if timestamp_names else None
    )
    inventory = {
        "row_count": row_count,
        "column_count": column_count,
        "feature_count": len(feature_names),
        "feature_names_json": canonical_json(feature_names),
        "feature_names_sha256": sha256_text(canonical_json(feature_names)),
        "training_boundary": training_boundary,
        "test_length": row_count - training_boundary,
        "training_anomaly_count": training_anomaly_count,
        "training_anomaly_ratio": training_anomaly_count / training_boundary,
        "training_label_contaminated": bool(training_anomaly_count),
        "label_row_count_matches_data": len(labels) == row_count,
        "test_label_length": len(test_labels),
        "test_anomaly_count": int(test_labels.sum()),
        "test_anomaly_segment_count": len(anomaly_lengths),
        "test_anomaly_length_min": min(anomaly_lengths) if anomaly_lengths else 0,
        "test_anomaly_length_median": (
            float(numpy.median(anomaly_lengths)) if anomaly_lengths else 0.0
        ),
        "test_anomaly_length_max": max(anomaly_lengths) if anomaly_lengths else 0,
        "filename_first_anomaly_index": first_anomaly_token,
        "observed_first_anomaly_index": observed_first_anomaly,
        "first_anomaly_matches_filename": observed_first_anomaly == first_anomaly_token,
        "missing_value_count": missing_count,
        "nonfinite_value_count": nonfinite_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "constant_channel_count": constant_count,
        "iqr_zero_channel_count": iqr_zero_count,
        "high_correlation_threshold": HIGH_CORRELATION_THRESHOLD,
        "high_correlation_pair_count": len(correlation_rows),
        "acf_source": "designated_training_prefix_first_feature",
        "acf_lag_cap": min(ACF_MAXIMUM_LAG, training_boundary - 1),
        "acf_candidate_lag": acf_candidate_lag,
        "audit_status": "approved",
        "status_reason": "",
    }
    return inventory, channel_rows, correlation_rows


def analyze_dev18_frame(
    file_name: str, frame: pandas.DataFrame, *, training_boundary: int,
) -> tuple[dict, list[dict], list[dict]]:
    """작은 합성 frame의 feature 순서·품질과 라벨 정렬을 판정한다."""
    if "Label" not in frame.columns:
        raise ValueError(f"Label 열이 없다: {file_name}")
    feature_names = [column for column in frame.columns if column != "Label"]
    if not feature_names:
        raise ValueError(f"feature 열이 없다: {file_name}")
    if any(
        not pandas.api.types.is_numeric_dtype(frame[column])
        for column in frame.columns
    ):
        raise ValueError(f"숫자가 아닌 값이 있다: {file_name}")
    labels = frame["Label"]
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Label은 결측 없는 0/1이어야 한다: {file_name}")
    features = frame[feature_names]
    missing_count = int(features.isna().to_numpy().sum())
    if missing_count:
        raise ValueError(f"feature 결측값이 있다: {file_name}")
    values = features.to_numpy(dtype=numpy.float64)
    nonfinite_count = int((~numpy.isfinite(values)).sum())
    if nonfinite_count:
        raise ValueError(f"feature 값은 모두 유한해야 한다: {file_name}")
    timestamp_names = [
        name for name in feature_names if str(name).strip().lower() == "timestamp"
    ]
    return _summarize_dev18(
        file_name,
        feature_names,
        features.iloc[:training_boundary],
        labels.to_numpy(dtype=numpy.int8),
        training_boundary=training_boundary,
        row_count=len(frame),
        column_count=len(frame.columns),
        missing_count=missing_count,
        nonfinite_count=nonfinite_count,
        duplicate_timestamp_count=(
            int(frame[timestamp_names[0]].duplicated().sum())
            if timestamp_names else None
        ),
    )


def analyze_dev18_file(
    path: Path, *, training_boundary: int, chunk_size: int = CSV_CHUNK_SIZE,
) -> tuple[dict, list[dict], list[dict]]:
    """큰 CSV는 training prefix만 보관하며 순차 처리한다."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size는 양의 정수여야 한다")
    path = Path(path)
    feature_names = None
    column_count = 0
    row_count = 0
    missing_count = 0
    nonfinite_count = 0
    duplicate_timestamp_count = None
    seen_timestamps = set()
    training_chunks = []
    label_chunks = []

    for chunk in pandas.read_csv(path, chunksize=chunk_size, low_memory=False):
        if feature_names is None:
            if "Label" not in chunk.columns:
                raise ValueError(f"Label 열이 없다: {path.name}")
            feature_names = [column for column in chunk.columns if column != "Label"]
            if not feature_names:
                raise ValueError(f"feature 열이 없다: {path.name}")
            column_count = len(chunk.columns)
            duplicate_timestamp_count = 0 if any(
                str(name).strip().lower() == "timestamp" for name in feature_names
            ) else None
        if any(
            not pandas.api.types.is_numeric_dtype(chunk[column])
            for column in chunk.columns
        ):
            raise ValueError(f"숫자가 아닌 값이 있다: {path.name}")
        labels = chunk["Label"]
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError(f"Label은 결측 없는 0/1이어야 한다: {path.name}")
        features = chunk[feature_names]
        chunk_missing_count = int(features.isna().to_numpy().sum())
        if chunk_missing_count:
            raise ValueError(f"feature 결측값이 있다: {path.name}")
        values = features.to_numpy(dtype=numpy.float64)
        chunk_nonfinite_count = int((~numpy.isfinite(values)).sum())
        if chunk_nonfinite_count:
            raise ValueError(f"feature 값은 모두 유한해야 한다: {path.name}")

        remaining_training_rows = max(0, training_boundary - row_count)
        if remaining_training_rows:
            training_chunks.append(features.iloc[:remaining_training_rows].copy())
        label_chunks.append(labels.to_numpy(dtype=numpy.int8))
        if duplicate_timestamp_count is not None:
            timestamp_name = next(
                name for name in feature_names
                if str(name).strip().lower() == "timestamp"
            )
            for timestamp in features[timestamp_name].to_numpy():
                duplicate_timestamp_count += timestamp in seen_timestamps
                seen_timestamps.add(timestamp)
        row_count += len(chunk)
        missing_count += chunk_missing_count
        nonfinite_count += chunk_nonfinite_count

    if feature_names is None or not label_chunks:
        raise ValueError(f"비어 있는 CSV다: {path.name}")
    training = pandas.concat(training_chunks, ignore_index=True)
    labels = numpy.concatenate(label_chunks)
    return _summarize_dev18(
        path.name,
        feature_names,
        training,
        labels,
        training_boundary=training_boundary,
        row_count=row_count,
        column_count=column_count,
        missing_count=missing_count,
        nonfinite_count=nonfinite_count,
        duplicate_timestamp_count=duplicate_timestamp_count,
    )


def audit_dev18_inputs(
    *, data_root=DEFAULT_DATA_ROOT, manifest_path=DEFAULT_MANIFEST_PATH,
    output_root=DEFAULT_OUTPUT_ROOT,
) -> dict:
    """봉인 manifest의 Dev18 18개를 감사하고 재사용할 원표를 저장한다."""
    data_root = Path(data_root)
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["datasets"]["DEV18"]["files"]
    if len(entries) != 18 or [entry["series"] for entry in entries] != [
        f"{index:02d}" for index in range(1, 19)
    ]:
        raise ValueError("Dev18 manifest는 공식 순서의 18개 series여야 한다")

    inventory_rows = []
    channel_rows = []
    correlation_rows = []
    for position, entry in enumerate(entries, start=1):
        path = data_root / entry["source_directory"] / entry["name"]
        if not path.is_file():
            raise FileNotFoundError(f"Dev18 입력 파일이 없다: {path}")
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"Dev18 파일 크기가 manifest와 다르다: {path.name}")
        digest = file_sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"Dev18 SHA-256이 manifest와 다르다: {path.name}")
        print(f"[{position:02d}/18] {path.name}", flush=True)
        inventory, file_channels, file_correlations = analyze_dev18_file(
            path, training_boundary=entry["training_boundary"],
        )
        identity = {
            "series": entry["series"],
            "order": entry["order"],
            "family": entry["family"],
            "source_directory": entry["source_directory"],
            "name": entry["name"],
            "size_bytes": entry["size_bytes"],
            "sha256": digest,
        }
        inventory_rows.append({**identity, **inventory})
        channel_rows.extend({**identity, **row} for row in file_channels)
        correlation_rows.extend({**identity, **row} for row in file_correlations)

    logs = output_root / "logs"
    snapshots = output_root / "snapshots"
    logs.mkdir(parents=True, exist_ok=True)
    snapshots.mkdir(parents=True, exist_ok=True)
    inventory_path = logs / "inventory.csv"
    feature_path = logs / "feature_schema_and_quality.csv"
    correlation_path = logs / "high_correlation_pairs.csv"
    pandas.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    pandas.DataFrame(channel_rows).to_csv(feature_path, index=False)
    pandas.DataFrame(correlation_rows, columns=(
        "series", "order", "family", "source_directory", "name", "size_bytes",
        "sha256", "left", "right", "absolute_correlation",
    )).to_csv(correlation_path, index=False)

    contaminated = [
        row["series"] for row in inventory_rows if row["training_anomaly_count"]
    ]
    dev18_names = {entry["name"] for entry in entries}
    ghl_entries = manifest["datasets"]["GHL"]["files"]
    ghl_names = {entry["name"] for entry in ghl_entries}
    official_names = []
    for reference in manifest["roles"]["official20_provenance"]["ordered_members"]:
        if reference["dataset"] == "DEV18":
            official_names.append(next(
                entry["name"] for entry in entries
                if entry["series"] == reference["series"]
            ))
        else:
            official_names.append(reference["name"])
    if (
        len(official_names) != 20 or len(set(official_names)) != 20
        or dev18_names & ghl_names
        or set(official_names) & ghl_names != {
            "040_GHL_id_9_Sensor_tr_50000_1st_92001.csv",
            "049_GHL_id_18_Sensor_tr_50000_1st_109001.csv",
        }
    ):
        raise ValueError("공식20·Dev18·GHL25 역할 교집합이 계약과 다르다")

    snapshot = {
        "schema_version": 1,
        "status": (
            "approved_with_disclosed_source_training_contamination"
            if contaminated else "approved"
        ),
        "dataset_role": "TSB_non_GHL_tuning_panel",
        "series_count": 18,
        "family_count": len({row["family"] for row in inventory_rows}),
        "uses_test_labels_only_for_integrity_summary": True,
        "uses_test_labels_for_model_or_parameter_choice": False,
        "training_prefix_label_filtering": False,
        "training_contaminated_series": contaminated,
        "high_correlation_threshold": HIGH_CORRELATION_THRESHOLD,
        "acf_rule": {
            "source": "designated_training_prefix_first_feature",
            "maximum_observations": 20_000,
            "maximum_lag": ACF_MAXIMUM_LAG,
            "minimum_candidate_lag": 3,
            "candidate": "highest_strict_local_maximum",
            "fallback_or_above_300": ACF_FALLBACK_PERIOD,
        },
        "role_sets": {
            "official20_count": len(official_names),
            "dev18_count": len(dev18_names),
            "ghl25_count": len(ghl_names),
            "dev18_ghl25_intersection_count": len(dev18_names & ghl_names),
            "official20_ghl25_intersection": sorted(set(official_names) & ghl_names),
        },
        "input_manifest_sha256": file_sha256(manifest_path),
        "audit_script_sha256": file_sha256(Path(__file__)),
        "tables": {
            path.name: file_sha256(path)
            for path in (inventory_path, feature_path, correlation_path)
        },
    }
    snapshot_path = snapshots / "audit.json"
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return {**snapshot, "snapshot_path": str(snapshot_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    audit_dev18_inputs(
        data_root=arguments.data_root,
        manifest_path=arguments.manifest,
        output_root=arguments.output,
    )


if __name__ == "__main__":
    main()
