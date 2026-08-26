"""새 GHL registry 입력 경로가 원시 feature만 읽는지 검증한다."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy
import pandas

from src.data_split.load_ghl_series import load_ghl_registered_inputs


def _write_csv(path: Path, *, finite=True):
    values = {
        f"sensor_{channel:02d}": numpy.arange(15, dtype=float) + channel * 100
        for channel in range(19)
    }
    if not finite:
        values["sensor_00"][3] = numpy.nan
    frame = pandas.DataFrame(values)
    frame["Label"] = ["not-read"] * len(frame)
    frame.to_csv(path, index=False)


class TestLoadGhlRegisteredInputs(unittest.TestCase):
    def test_returns_raw_features_and_ranges_without_labels_or_scaler(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "032_GHL_id_1_Sensor_tr_11_1st_12.csv"
            _write_csv(path)
            inputs = load_ghl_registered_inputs(path)

        self.assertEqual(set(inputs), {
            "feature_names", "normal_training", "test_sessions", "source_ranges",
        })
        self.assertEqual(len(inputs["feature_names"]), 19)
        numpy.testing.assert_array_equal(
            inputs["normal_training"][:, 0], numpy.arange(11, dtype=numpy.float32),
        )
        numpy.testing.assert_array_equal(
            inputs["test_sessions"][0][:, 0], numpy.arange(11, 15, dtype=numpy.float32),
        )
        self.assertEqual(inputs["source_ranges"], {
            "source": path.name,
            "normal_training": (0, 11),
            "test_sessions": ((11, 15),),
        })

    def test_sensor_read_excludes_label_values(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "032_GHL_id_1_Sensor_tr_11_1st_12.csv"
            _write_csv(path)
            calls = []
            original = pandas.read_csv

            def recording_read_csv(*args, **kwargs):
                calls.append(dict(kwargs))
                return original(*args, **kwargs)

            with patch(
                "src.data_split.load_ghl_series.pandas.read_csv",
                side_effect=recording_read_csv,
            ):
                load_ghl_registered_inputs(path)

        value_reads = [call for call in calls if call.get("nrows") != 0]
        self.assertEqual(len(value_reads), 1)
        self.assertNotIn("Label", value_reads[0]["usecols"])
        self.assertEqual(len(value_reads[0]["usecols"]), 19)

    def test_rejects_bad_filename_feature_count_and_nonfinite_sensor(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bad_name = root / "GHL.csv"
            _write_csv(bad_name)
            with self.assertRaisesRegex(ValueError, "tr_ 경계"):
                load_ghl_registered_inputs(bad_name)

            nonfinite = root / "032_GHL_id_1_Sensor_tr_11_1st_12.csv"
            _write_csv(nonfinite, finite=False)
            with self.assertRaisesRegex(ValueError, "결측 또는 비유한"):
                load_ghl_registered_inputs(nonfinite)

            frame = pandas.read_csv(nonfinite).drop(columns=["sensor_18"])
            frame.to_csv(nonfinite, index=False)
            with self.assertRaisesRegex(ValueError, "19개"):
                load_ghl_registered_inputs(nonfinite)


if __name__ == "__main__":
    unittest.main()
