"""Dev18 파일별 ell_max 산출 규칙을 검증한다."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy
import pandas
import yaml

from tests.ghl_main.build_dev18_ell_max import (
    build_dev18_ell_max,
    first_strict_local_peak,
    summarize_training_periods,
)


class TestDev18EllMax(unittest.TestCase):
    def test_finds_first_strict_local_peak(self):
        timestep = numpy.arange(240, dtype=numpy.float64)
        values = numpy.sin(2 * numpy.pi * timestep / 12)

        self.assertEqual(first_strict_local_peak(values), 12)

    def test_uses_all_nonconstant_nontimestamp_channels_and_floors_median(self):
        timestep = numpy.arange(600, dtype=numpy.float64)
        training = numpy.column_stack((
            numpy.sin(2 * numpy.pi * timestep / 10),
            numpy.sin(2 * numpy.pi * timestep / 21),
            timestep,
            numpy.ones_like(timestep),
        ))

        result = summarize_training_periods(
            training,
            ("period_10", "period_21", "timestamp", "constant"),
        )

        self.assertEqual(result["nonconstant_channel_count"], 2)
        self.assertEqual(result["detected_peak_channel_count"], 2)
        self.assertEqual(result["missing_peak_channel_count"], 0)
        self.assertEqual(result["first_peak_lag_median"], 15.5)
        self.assertEqual(result["l_max_samples"], 15)

    def test_rejects_file_without_detected_channel_peak(self):
        with self.assertRaisesRegex(ValueError, "strict local peak"):
            summarize_training_periods(
                numpy.ones((20, 2), dtype=numpy.float64),
                ("first", "second"),
            )

    def test_builds_one_reproducible_result_for_each_of_18_manifest_files(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            tuning_directory = root / "tuning"
            tuning_directory.mkdir()
            entries = []
            for order in range(1, 19):
                period = order + 5
                training_boundary = 240
                timestep = numpy.arange(260, dtype=numpy.float64)
                frame = pandas.DataFrame({
                    "signal": numpy.sin(2 * numpy.pi * timestep / period),
                    "Label": numpy.zeros(len(timestep), dtype=int),
                })
                name = (
                    f"{order:03d}_TEST_id_{order}_Sensor_"
                    f"tr_{training_boundary}_1st_241.csv"
                )
                path = tuning_directory / name
                frame.to_csv(path, index=False)
                feature_names = ["signal"]
                feature_names_json = json.dumps(
                    feature_names,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                entries.append({
                    "series": f"{order:02d}",
                    "order": order,
                    "family": "TEST",
                    "source_directory": "tuning",
                    "name": name,
                    "training_boundary": training_boundary,
                    "row_count": len(frame),
                    "feature_count": 1,
                    "feature_names_sha256": hashlib.sha256(
                        feature_names_json.encode("utf-8")
                    ).hexdigest(),
                    "size_bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })

            manifest_path = root / "input_manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump({"datasets": {"DEV18": {"files": entries}}}),
                encoding="utf-8",
            )
            output_path = root / "dev18_ell_max.json"

            snapshot = build_dev18_ell_max(root, manifest_path, output_path)

            self.assertEqual(snapshot["series_count"], 18)
            self.assertEqual(len(snapshot["series"]), 18)
            self.assertEqual(snapshot["series"][0]["l_max_samples"], 6)
            self.assertEqual(snapshot["series"][-1]["l_max_samples"], 23)
            self.assertEqual(len(snapshot["ell_max_id"]), 64)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")), snapshot,
            )


if __name__ == "__main__":
    unittest.main()
