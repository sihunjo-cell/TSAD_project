import unittest

import pandas

from experiments.exp01b_ghl_preflight.run_ghl_preflight import (
    analyze_series_frame,
    build_ratio_channel_quality_rows,
    build_ratio_feasibility_rows,
    build_ratio_representativeness_rows,
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
    def test_shortest_ghl_five_percent_supports_source_window_candidates(self):
        rows = build_ratio_feasibility_rows(
            series=12,
            train_length=39938,
            ratios=(0.05,),
            window_sizes=(5, 155),
            validation_fraction=0.1,
        )

        by_window = {row["window_size"]: row for row in rows}
        self.assertEqual(by_window[5]["kept_length"], 1997)
        self.assertEqual(by_window[5]["validation_length"], 200)
        self.assertEqual(by_window[5]["model_train_length"], 1797)
        self.assertEqual(by_window[5]["train_window_count"], 1792)
        self.assertEqual(by_window[5]["normalization_sample_count"], 1992)
        self.assertEqual(by_window[155]["validation_window_count"], 45)
        self.assertEqual(by_window[155]["normalization_sample_count"], 1842)
        self.assertTrue(by_window[155]["feasible"])


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


class TestBuildRatioRepresentativenessRows(unittest.TestCase):
    def test_does_not_count_long_integer_constants_as_active(self):
        train_features = pandas.DataFrame({
            **{f"constant_{index}": [1] * 28080 for index in range(84)},
            "moving_a": list(range(28080)),
            "moving_b": list(range(28080, 56160)),
        })

        row = build_ratio_representativeness_rows(
            train_features, series=1, ratios=(0.5,)
        )[0]

        self.assertEqual(row["active_channel_count"], 2)
        self.assertGreater(row["median_standardized_mean_gap"], 0.0)

    def test_marks_prefix_with_missing_channel_activity_and_relationships(self):
        train_features = pandas.DataFrame({
            "always_moving": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "late_moving": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        })

        rows = build_ratio_representativeness_rows(
            train_features, series=1, ratios=(0.5, 1.0)
        )

        by_ratio = {row["ratio"]: row for row in rows}
        self.assertEqual(by_ratio[0.5]["active_channel_count"], 1)
        self.assertEqual(by_ratio[0.5]["variable_iqr_channel_count"], 1)
        self.assertEqual(by_ratio[0.5]["interquartile_support_coverage"], 0.0)
        self.assertEqual(by_ratio[0.5]["relationship_similarity"], 0.0)
        self.assertGreater(by_ratio[0.5]["median_standardized_mean_gap"], 0.0)

        self.assertEqual(by_ratio[1.0]["active_channel_count"], 2)
        self.assertEqual(by_ratio[1.0]["variable_iqr_channel_count"], 2)
        self.assertEqual(by_ratio[1.0]["interquartile_support_coverage"], 1.0)
        self.assertEqual(by_ratio[1.0]["relationship_similarity"], 1.0)
        self.assertEqual(by_ratio[1.0]["median_standardized_mean_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
