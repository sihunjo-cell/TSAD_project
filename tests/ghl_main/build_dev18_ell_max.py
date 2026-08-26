"""TSB 비-GHL 튜닝 패널의 파일별 ell_max를 봉인한다."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.execution_identity import file_sha256
from src.data_split.load_dev18_series import load_dev18_registered_inputs


REFERENCE_BRANCH = "feature/vus-pr-jiwoo"
REFERENCE_COMMIT = "7a9b085a9e9b6d8b53382ecf795e52094ddb9934"
DEFAULT_DATA_ROOT = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project"
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "input_manifest.yaml"
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT / "experiments" / "01_ghl_main" / "snapshots"
    / "dev18_selection" / "dev18_ell_max.json"
)


def first_strict_local_peak(values, maximum_lag=500) -> int | None:
    """정규화 FFT ACF의 lag 1..maximum_lag에서 첫 strict local peak를 찾는다."""
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.ndim != 1 or not len(values) or not numpy.isfinite(values).all():
        raise ValueError("ACF 입력은 비어 있지 않은 유한 1차원 배열이어야 한다")
    if type(maximum_lag) is not int or maximum_lag < 1:
        raise ValueError("maximum_lag는 양의 정수여야 한다")

    centered = values - values.mean()
    if numpy.ptp(centered) == 0:
        return None
    lag_limit = min(maximum_lag, len(centered) - 2)
    if lag_limit < 1:
        return None

    transform_length = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = numpy.fft.rfft(centered, transform_length)
    autocorrelation = numpy.fft.irfft(
        spectrum * numpy.conjugate(spectrum), transform_length,
    )[:lag_limit + 2]
    autocorrelation /= autocorrelation[0]
    peaks = numpy.flatnonzero(
        (autocorrelation[1:lag_limit + 1] > autocorrelation[:lag_limit])
        & (autocorrelation[1:lag_limit + 1] > autocorrelation[2:lag_limit + 2])
    ) + 1
    return int(peaks[0]) if len(peaks) else None


def summarize_training_periods(training, feature_names, maximum_lag=500) -> dict:
    """한 파일의 비상수 채널 첫 peak lag 중앙값을 ell_max로 만든다."""
    training = numpy.asarray(training, dtype=numpy.float64)
    feature_names = tuple(str(name) for name in feature_names)
    if training.ndim != 2 or training.shape[1] != len(feature_names):
        raise ValueError("training 열 수와 feature_names 길이가 같아야 한다")
    if not len(training) or not numpy.isfinite(training).all():
        raise ValueError("training은 비어 있지 않은 유한 2차원 배열이어야 한다")

    first_peaks = []
    nonconstant_count = 0
    for position, feature_name in enumerate(feature_names):
        if feature_name.strip().lower() in {"label", "timestamp"}:
            continue
        values = training[:, position]
        if numpy.ptp(values) == 0:
            continue
        nonconstant_count += 1
        peak = first_strict_local_peak(values, maximum_lag)
        if peak is not None:
            first_peaks.append(peak)

    if not first_peaks:
        raise ValueError("비상수 채널에서 first strict local peak를 찾지 못했다")
    median = float(numpy.median(first_peaks))
    return {
        "nonconstant_channel_count": nonconstant_count,
        "detected_peak_channel_count": len(first_peaks),
        "missing_peak_channel_count": nonconstant_count - len(first_peaks),
        "first_peak_lag_median": median,
        "l_max_samples": int(numpy.floor(median)),
    }


def build_dev18_ell_max(data_root, manifest_path, output_path) -> dict:
    """봉인 manifest의 18개 파일을 순서대로 읽어 ell_max snapshot을 만든다."""
    manifest_path = Path(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    entries = sorted(
        manifest["datasets"]["DEV18"]["files"], key=lambda entry: entry["order"],
    )
    if [entry["order"] for entry in entries] != list(range(1, 19)):
        raise ValueError("Dev18 order는 1부터 18까지 한 번씩 있어야 한다")
    if [entry["series"] for entry in entries] != [f"{index:02d}" for index in range(1, 19)]:
        raise ValueError("Dev18 series는 01부터 18까지 순서대로 있어야 한다")

    series_results = []
    for entry in entries:
        loaded = load_dev18_registered_inputs(entry, data_root)
        series_results.append({
            "series": entry["series"],
            "order": entry["order"],
            "family": entry["family"],
            "name": entry["name"],
            "input_sha256": entry["sha256"],
            "training_boundary": entry["training_boundary"],
            **summarize_training_periods(
                loaded["normal_training"], loaded["feature_names"],
            ),
        })

    payload = {
        "schema_version": 1,
        "dataset_role": "TSB_non_GHL_tuning_panel",
        "reference_branch": REFERENCE_BRANCH,
        "reference_commit": REFERENCE_COMMIT,
        "protocol": {
            "input": "designated training prefix; Label and timestamp excluded",
            "label_row_filtering": False,
            "acf": "normalized autocorrelation via FFT",
            "lag_search": [1, 500],
            "peak_rule": "first strict local peak",
            "channel_filter": "all nonconstant channels",
            "file_selection": "median of detected channel first-peak lags",
            "integer_conversion": "floor after median",
            "missing_peak": "exclude channel; fail if no channel peak is detected",
            "reuse": "one value per series for every model, config, q, and seed",
        },
        "series_count": len(series_results),
        "input_manifest_sha256": file_sha256(manifest_path),
        "generator_sha256": file_sha256(Path(__file__)),
        "series": series_results,
    }
    ell_max_id = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    snapshot = {**payload, "ell_max_id": ell_max_id}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    snapshot = build_dev18_ell_max(
        arguments.data_root, arguments.manifest, arguments.output,
    )
    print(json.dumps({
        "output": str(arguments.output),
        "series_count": snapshot["series_count"],
        "ell_max_id": snapshot["ell_max_id"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
