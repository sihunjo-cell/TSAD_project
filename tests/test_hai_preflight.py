import unittest

import pandas

from experiments.exp01c_hai_preflight.run_hai_preflight import (
    analyze_training_session,
    build_ratio_channel_activity_rows,
    build_ratio_feasibility_rows,
    build_ratio_representativeness_rows,
)


class TestAnalyzeTrainingSession(unittest.TestCase):
    def test_excludes_timestamp_and_accepts_one_second_continuity(self):
        frame = pandas.DataFrame({
            "timestamp": pandas.date_range("2023-01-01", periods=4, freq="s"),
            "sensor_a": [0.0, 1.0, 2.0, 3.0],
            "sensor_b": [1.0, 1.0, 1.0, 1.0],
        })

        inventory = analyze_training_session("hai-train1.csv", frame)

        self.assertEqual(inventory["row_count"], 4)
        self.assertEqual(inventory["feature_count"], 2)
        self.assertEqual(inventory["timestamp_gap_count"], 0)
        self.assertEqual(inventory["missing_value_count"], 0)
        self.assertEqual(inventory["nonfinite_value_count"], 0)

    def test_rejects_a_gap_inside_a_continuous_session(self):
        frame = pandas.DataFrame({
            "timestamp": pandas.to_datetime([
                "2023-01-01 00:00:00",
                "2023-01-01 00:00:01",
                "2023-01-01 00:00:03",
            ]),
            "sensor": [0.0, 1.0, 2.0],
        })

        with self.assertRaisesRegex(ValueError, "1초 연속성 위반"):
            analyze_training_session("hai-train1.csv", frame)


class TestBuildRatioFeasibilityRows(unittest.TestCase):
    def test_shortest_hai_session_ten_percent_supports_window_155(self):
        row = build_ratio_feasibility_rows(
            session=3,
            train_length=126000,
            ratios=(0.10,),
            window_sizes=(155,),
            validation_fraction=0.1,
        )[0]

        self.assertEqual(row["kept_length"], 12600)
        self.assertEqual(row["validation_length"], 1260)
        self.assertEqual(row["model_train_length"], 11340)
        self.assertEqual(row["train_window_count"], 11185)
        self.assertEqual(row["validation_window_count"], 1105)
        self.assertEqual(row["normalization_sample_count"], 12445)
        self.assertTrue(row["feasible"])


class TestBuildRatioRepresentativenessRows(unittest.TestCase):
    def test_does_not_count_a_long_constant_one_channel_as_active(self):
        features = pandas.DataFrame({
            **{f"constant_{index}": [1] * 28080 for index in range(84)},
            "moving_a": list(range(28080)),
            "moving_b": list(range(28080, 56160)),
        })

        row = build_ratio_representativeness_rows(
            features, session=1, ratios=(0.5,)
        )[0]

        self.assertEqual(row["active_channel_count"], 2)
        self.assertGreater(row["median_standardized_mean_gap"], 0.0)

    def test_reports_structure_missing_from_a_prefix(self):
        features = pandas.DataFrame({
            "always_moving": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "late_moving": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        })

        rows = build_ratio_representativeness_rows(
            features, session=1, ratios=(0.5, 1.0)
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


class TestBuildRatioChannelActivityRows(unittest.TestCase):
    def test_marks_a_channel_active_only_after_it_changes(self):
        features = pandas.DataFrame({
            "always_moving": [0, 1, 2, 3, 4, 5, 6, 7],
            "late_moving": [0, 0, 0, 0, 0, 1, 2, 3],
        })

        rows = build_ratio_channel_activity_rows(
            features, session=1, ratios=(0.5, 1.0)
        )

        by_ratio_channel = {
            (row["ratio"], row["channel"]): row["active"] for row in rows
        }
        self.assertTrue(by_ratio_channel[(0.5, "always_moving")])
        self.assertFalse(by_ratio_channel[(0.5, "late_moving")])
        self.assertTrue(by_ratio_channel[(1.0, "late_moving")])


if __name__ == "__main__":
    unittest.main()
