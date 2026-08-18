import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas

from tests.checks import audit_ghl_inputs
from tests.checks.audit_ghl_inputs import (
    analyze_series_frame,
    build_ratio_channel_quality_rows,
    build_ratio_feasibility_rows,
    parse_ghl_filename,
)


class TestParseGhlFilename(unittest.TestCase):
    def test_parses_official_filename_fields(self):
        parsed = parse_ghl_filename(
            "043_GHL_id_12_Sensor_tr_39938_1st_40038.csv"
        )

        self.assertEqual(parsed["archive_index"], 43)
        self.assertEqual(parsed["series"], 12)
        self.assertEqual(parsed["train_boundary"], 39938)
        self.assertEqual(parsed["first_anomaly_index"], 40038)

    def test_rejects_non_ghl_filename(self):
        with self.assertRaisesRegex(ValueError, "GHL 파일명 규약 위반"):
            parse_ghl_filename("057_SMD_id_1_Facility_tr_4529_1st_4629.csv")


class TestAnalyzeSeriesFrame(unittest.TestCase):
    def test_reports_boundary_labels_and_zero_iqr_channel(self):
        frame = pandas.DataFrame({
            "constant_sensor": [1.0] * 6,
            "moving_sensor": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "Label": [0, 0, 0, 0, 1, 0],
        })
        parsed = parse_ghl_filename(
            "032_GHL_id_1_Sensor_tr_4_1st_4.csv"
        )

        inventory, channel_rows = analyze_series_frame(
            "032_GHL_id_1_Sensor_tr_4_1st_4.csv", frame, parsed
        )

        self.assertEqual(inventory["feature_count"], 2)
        self.assertEqual(inventory["train_length"], 4)
        self.assertEqual(inventory["test_length"], 2)
        self.assertEqual(inventory["train_anomaly_count"], 0)
        self.assertEqual(inventory["test_anomaly_count"], 1)
        self.assertEqual(inventory["observed_first_anomaly_index"], 4)
        self.assertTrue(inventory["first_anomaly_matches_filename"])

        by_channel = {row["channel"]: row for row in channel_rows}
        self.assertTrue(by_channel["constant_sensor"]["iqr_zero"])
        self.assertTrue(by_channel["constant_sensor"]["standard_deviation_zero"])
        self.assertFalse(by_channel["moving_sensor"]["iqr_zero"])


class TestBuildRatioFeasibilityRows(unittest.TestCase):
    def test_shortest_ghl_five_percent_checks_only_current_window(self):
        rows = build_ratio_feasibility_rows(
            series=12,
            train_length=39938,
            ratios=(0.05,),
            validation_fraction=0.1,
        )

        self.assertEqual([row["window_size"] for row in rows], [5])
        self.assertEqual(rows[0]["kept_length"], 1997)
        self.assertEqual(rows[0]["validation_length"], 200)
        self.assertEqual(rows[0]["model_train_length"], 1797)
        self.assertEqual(rows[0]["train_window_count"], 1792)
        self.assertEqual(rows[0]["normalization_sample_count"], 1992)
        self.assertTrue(rows[0]["feasible"])


class TestBuildRatioChannelQualityRows(unittest.TestCase):
    def test_detects_channel_frozen_only_in_front_half(self):
        train_features = pandas.DataFrame({
            "late_moving_sensor": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        })

        rows = build_ratio_channel_quality_rows(
            train_features, series=1, ratios=(0.5, 1.0)
        )

        by_ratio = {row["ratio"]: row for row in rows}
        self.assertTrue(by_ratio[0.5]["iqr_zero"])
        self.assertTrue(by_ratio[0.5]["standard_deviation_zero"])
        self.assertFalse(by_ratio[1.0]["iqr_zero"])
        self.assertFalse(by_ratio[1.0]["standard_deviation_zero"])


class TestRunGhlPreflight(unittest.TestCase):
    def test_writes_only_prefix_local_audit_outputs(self):
        frame = pandas.DataFrame({
            **{f"sensor_{index}": range(40) for index in range(19)},
            "Label": [0] * 20 + [1] + [0] * 19,
        })

        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            data_dir = temporary_path / "data"
            experiment_dir = temporary_path / "experiment"
            logs_dir = experiment_dir / "logs"
            data_dir.mkdir()
            for archive_index, series in zip(range(32, 57), range(1, 26)):
                file_name = (
                    f"{archive_index:03d}_GHL_id_{series}_Sensor_tr_20_1st_20.csv"
                )
                frame.to_csv(data_dir / file_name, index=False)

            with (
                patch.object(audit_ghl_inputs, "DATA_DIR", data_dir),
                patch.object(audit_ghl_inputs, "EXPERIMENT_DIR", experiment_dir),
                patch.object(audit_ghl_inputs, "LOGS_DIR", logs_dir),
            ):
                audit_ghl_inputs.run_ghl_preflight()

            self.assertEqual(
                {path.name for path in logs_dir.glob("*.csv")},
                {
                    "inventory.csv",
                    "ratio_feasibility.csv",
                    "train_channel_quality.csv",
                },
            )


if __name__ == "__main__":
    unittest.main()
