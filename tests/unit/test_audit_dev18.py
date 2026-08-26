"""Dev18 Role-A 감사가 누수 없이 구조·품질 근거를 만드는지 검증한다."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy
import pandas

from tests.checks.audit_dev18_inputs import (
    analyze_dev18_frame,
    analyze_dev18_file,
    estimate_training_period,
)


class TestAnalyzeDev18Frame(unittest.TestCase):
    def test_file_audit_reads_csv_in_chunks_and_matches_frame_analysis(self):
        frame = pandas.DataFrame({
            "timestamp": [0.0, 1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0],
            "second": numpy.arange(8, dtype=float) * 2,
            "Label": [0, 0, 1, 0, 0, 1, 1, 0],
        })
        expected = analyze_dev18_frame(
            "004_MSL_id_3_Sensor_tr_4_1st_2.csv", frame, training_boundary=4,
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "004_MSL_id_3_Sensor_tr_4_1st_2.csv"
            frame.to_csv(path, index=False)
            original_read_csv = pandas.read_csv

            def require_chunks(*args, **kwargs):
                self.assertGreater(kwargs.get("chunksize", 0), 0)
                return original_read_csv(*args, **kwargs)

            with patch(
                "tests.checks.audit_dev18_inputs.pandas.read_csv",
                side_effect=require_chunks,
            ):
                actual = analyze_dev18_file(path, training_boundary=4, chunk_size=3)

        self.assertEqual(actual, expected)
        self.assertEqual(actual[0]["duplicate_timestamp_count"], 1)

    def test_estimates_period_from_training_values_only(self):
        timestep = numpy.arange(240, dtype=float)
        values = numpy.sin(2 * numpy.pi * timestep / 12)
        self.assertEqual(estimate_training_period(values), 12)

    def test_reports_training_quality_and_test_label_segments(self):
        frame = pandas.DataFrame({
            "first": numpy.arange(8, dtype=float),
            "second": numpy.arange(8, dtype=float) * 2,
            "constant": numpy.ones(8),
            "Label": [0, 0, 0, 0, 0, 1, 1, 0],
        })

        inventory, channel_rows, correlation_rows = analyze_dev18_frame(
            "004_MSL_id_3_Sensor_tr_4_1st_5.csv", frame, training_boundary=4,
        )

        self.assertEqual(inventory["row_count"], 8)
        self.assertEqual(inventory["feature_count"], 3)
        self.assertEqual(inventory["test_length"], 4)
        self.assertEqual(inventory["training_anomaly_count"], 0)
        self.assertEqual(inventory["test_anomaly_count"], 2)
        self.assertEqual(inventory["test_anomaly_segment_count"], 1)
        self.assertEqual(inventory["test_anomaly_length_min"], 2)
        self.assertEqual(inventory["test_anomaly_length_median"], 2.0)
        self.assertEqual(inventory["test_anomaly_length_max"], 2)
        self.assertEqual(inventory["missing_value_count"], 0)
        self.assertEqual(inventory["nonfinite_value_count"], 0)
        self.assertEqual(inventory["constant_channel_count"], 1)
        self.assertEqual(inventory["iqr_zero_channel_count"], 1)
        self.assertEqual(inventory["high_correlation_pair_count"], 1)
        self.assertEqual(
            [(row["position"], row["feature_name"]) for row in channel_rows],
            [(1, "first"), (2, "second"), (3, "constant")],
        )
        self.assertEqual(
            correlation_rows,
            [{"left": "first", "right": "second", "absolute_correlation": 1.0}],
        )

    def test_records_source_training_contamination_without_label_filtering(self):
        base = pandas.DataFrame({
            "feature": [0.0, 1.0, 2.0, 3.0],
            "Label": [0, 0, 1, 0],
        })
        inventory, _, _ = analyze_dev18_frame(
            "004_MSL_id_3_Sensor_tr_3_1st_2.csv", base, training_boundary=3,
        )
        self.assertEqual(inventory["training_anomaly_count"], 1)
        self.assertAlmostEqual(inventory["training_anomaly_ratio"], 1 / 3)

    def test_rejects_invalid_values(self):

        cases = (
            (pandas.DataFrame({"feature": [0.0, numpy.nan], "Label": [0, 1]}), "결측"),
            (pandas.DataFrame({"feature": [0.0, numpy.inf], "Label": [0, 1]}), "유한"),
            (pandas.DataFrame({"feature": [0.0, 1.0], "Label": [0, 2]}), "0/1"),
            (pandas.DataFrame({"feature": [0.0, "bad"], "Label": [0, 1]}), "숫자"),
        )
        for frame, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    analyze_dev18_frame(
                        "004_MSL_id_3_Sensor_tr_1_1st_1.csv",
                        frame,
                        training_boundary=1,
                    )


if __name__ == "__main__":
    unittest.main()
