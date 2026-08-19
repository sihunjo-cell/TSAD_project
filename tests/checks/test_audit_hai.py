import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas

from tests.checks import audit_hai_inputs
from tests.checks.audit_hai_inputs import (
    analyze_training_session,
    build_ratio_channel_activity_rows,
    build_ratio_feasibility_rows,
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
    def test_shortest_hai_session_checks_only_current_window(self):
        rows = build_ratio_feasibility_rows(
            session=3,
            train_length=126000,
            ratios=(0.10,),
            validation_fraction=0.1,
        )

        self.assertEqual([row["window_size"] for row in rows], [5])
        row = rows[0]
        self.assertEqual(row["fit_pool_length"], 113400)
        self.assertEqual(row["kept_length"], 11340)
        self.assertEqual(row["validation_length"], 12600)
        self.assertEqual(row["model_train_length"], 11340)
        self.assertEqual(row["train_window_count"], 11335)
        self.assertEqual(row["validation_window_count"], 12595)
        self.assertEqual(row["scaler_fit_observation_count"], 11340)
        self.assertEqual(row["validation_transform_observation_count"], 12600)
        self.assertTrue(row["feasible"])


class TestBuildRatioChannelActivityRows(unittest.TestCase):
    def test_marks_a_channel_active_only_after_it_changes(self):
        features = pandas.DataFrame({
            "always_moving": [0, 1, 2, 3, 4, 5, 6, 7],
            "late_moving": [0, 0, 0, 0, 0, 1, 2, 3],
        })

        rows = build_ratio_channel_activity_rows(
            features, session=1, ratios=(0.4, 1.0)
        )

        by_ratio_channel = {
            (row["ratio"], row["channel"]): row["active"] for row in rows
        }
        self.assertTrue(by_ratio_channel[(0.4, "always_moving")])
        self.assertFalse(by_ratio_channel[(0.4, "late_moving")])
        self.assertTrue(by_ratio_channel[(1.0, "late_moving")])


class TestRunHaiPreflight(unittest.TestCase):
    def test_writes_only_prefix_local_audit_outputs(self):
        frame = pandas.DataFrame({
            "timestamp": pandas.date_range("2023-01-01", periods=1200, freq="s"),
            **{f"sensor_{index}": range(1200) for index in range(86)},
        })

        with TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            data_dir = temporary_path / "data"
            experiment_dir = temporary_path / "experiment"
            logs_dir = experiment_dir / "logs"
            data_dir.mkdir()
            for session in range(1, 5):
                frame.to_csv(data_dir / f"hai-train{session}.csv", index=False)

            with (
                patch.object(audit_hai_inputs, "DATA_DIR", data_dir),
                patch.object(audit_hai_inputs, "EXPERIMENT_DIR", experiment_dir),
                patch.object(audit_hai_inputs, "LOGS_DIR", logs_dir),
            ):
                audit_hai_inputs.run_hai_preflight()

            self.assertEqual(
                {path.name for path in logs_dir.glob("*.csv")},
                {
                    "corpus_channel_activity.csv",
                    "inventory.csv",
                    "ratio_channel_activity.csv",
                    "ratio_feasibility.csv",
                },
            )


if __name__ == "__main__":
    unittest.main()
