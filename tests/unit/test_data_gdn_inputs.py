"""임시 CSV로 GHL·HAI 입력 경계와 scaler fit 범위를 검증한다."""

import tempfile
import unittest
from pathlib import Path

import numpy
import pandas
from unittest.mock import patch

from src.data_split.load_ghl_series import load_ghl_series
from src.data_split.load_hai_sessions import load_hai_sessions, load_hai_temporal_condition


def feature_names(count: int) -> list[str]:
    return [f"sensor_{channel:02d}" for channel in range(count)]


def write_ghl_csv(path: Path) -> None:
    frame = pandas.DataFrame({
        name: numpy.arange(15, dtype=float) + channel * 100
        for channel, name in enumerate(feature_names(19))
    })
    frame["Label"] = [0] * 11 + [1, 0, 1, 0]
    frame.to_csv(path, index=False)


def write_hai_data(dataset_dir: Path) -> None:
    names = feature_names(86)
    for session, offset in enumerate((0, 20, 40, 60), start=1):
        frame = pandas.DataFrame({
            "timestamp": pandas.date_range(f"2023-01-0{session}", periods=100, freq="s").astype(str),
            **{
                name: offset + numpy.arange(100, dtype=float) + channel * 100
                for channel, name in enumerate(names)
            },
        })
        frame.to_csv(dataset_dir / f"hai-train{session}.csv", index=False)

    for session, (offset, row_count) in enumerate(((80, 3), (100, 4)), start=1):
        timestamps = pandas.date_range(f"2023-02-0{session}", periods=row_count, freq="s").astype(str)
        frame = pandas.DataFrame({
            "timestamp": timestamps,
            **{
                name: offset + numpy.arange(row_count, dtype=float) + channel * 100
                for channel, name in enumerate(names)
            },
        })
        frame.to_csv(dataset_dir / f"hai-test{session}.csv", index=False)
        pandas.DataFrame({
            "timestamp": timestamps,
            "label": numpy.arange(row_count) % 2,
        }).to_csv(dataset_dir / f"label-test{session}.csv", index=False)


class TestLoadGhlSeries(unittest.TestCase):
    def test_uses_filename_boundary_prefix_ratio_and_train_only_scaler(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "032_GHL_id_1_Sensor_tr_11_1st_12.csv"
            write_ghl_csv(csv_path)

            inputs = load_ghl_series(
                csv_path, ratio_percent=50, val_fraction=1 / 3, min_train_length=4,
            )

        self.assertEqual(inputs["feature_names"], tuple(feature_names(19)))
        self.assertEqual([array.shape for array in inputs["train_sessions"]], [(4, 19)])
        self.assertEqual([array.shape for array in inputs["validation_sessions"]], [(2, 19)])
        self.assertEqual([array.shape for array in inputs["test_sessions"]], [(4, 19)])
        self.assertEqual(inputs["test_labels"][0].tolist(), [1, 0, 1, 0])
        self.assertEqual(inputs["session_splits"], ({
            "source": csv_path.name,
            "original_train_range": (0, 11),
            "kept_train_range": (0, 6),
            "train_range": (0, 4),
            "validation_range": (4, 6),
            "test_range": (11, 15),
        },))
        self.assertAlmostEqual(inputs["train_sessions"][0][-1, 0], 1.0)
        self.assertAlmostEqual(inputs["validation_sessions"][0][0, 0], 4 / 3)
        self.assertAlmostEqual(inputs["test_sessions"][0][0, 0], 11 / 3, places=6)

    def test_rejects_nonbinary_ghl_labels(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "032_GHL_id_1_Sensor_tr_11_1st_12.csv"
            write_ghl_csv(csv_path)
            frame = pandas.read_csv(csv_path)
            frame.loc[12, "Label"] = 2
            frame.to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "0 또는 1"):
                load_ghl_series(
                    csv_path, ratio_percent=50, val_fraction=1 / 3, min_train_length=4,
                )

    def test_uses_scaler_return_value_when_transform_copies(self):
        class CopyingScaler:
            def fit(self, _array):
                return self

            def transform(self, array):
                return array + 1000

        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "032_GHL_id_1_Sensor_tr_11_1st_12.csv"
            write_ghl_csv(csv_path)
            with patch("src.data_split.load_ghl_series.MinMaxScaler", return_value=CopyingScaler()):
                inputs = load_ghl_series(
                    csv_path, ratio_percent=50, val_fraction=1 / 3, min_train_length=4,
                )

        self.assertEqual(inputs["train_sessions"][0][0, 0], 1000)
        self.assertEqual(inputs["validation_sessions"][0][0, 0], 1004)
        self.assertEqual(inputs["test_sessions"][0][0, 0], 1011)

    def test_rejects_ratio_outside_ghl_plan(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "032_GHL_id_1_Sensor_tr_11_1st_12.csv"
            write_ghl_csv(csv_path)
            with self.assertRaisesRegex(ValueError, "GHL ratio"):
                load_ghl_series(csv_path, ratio_percent=7, val_fraction=0.1, min_train_length=6)

class TestLoadHaiSessions(unittest.TestCase):
    def test_temporal_conditions_use_only_available_train_sessions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset_dir = Path(temporary_dir)
            write_hai_data(dataset_dir)
            first = load_hai_temporal_condition(
                dataset_dir, condition=1, val_fraction=0.2, min_train_length=4,
            )
            second = load_hai_temporal_condition(
                dataset_dir, condition=2, val_fraction=0.2, min_train_length=4,
            )

        self.assertEqual([array.shape for array in first["train_sessions"]], [(80, 86)])
        self.assertEqual([array.shape for array in second["train_sessions"]], [(80, 86), (80, 86)])
        self.assertEqual([array.shape for array in first["test_sessions"]], [(3, 86)])
        self.assertEqual([array.shape for array in second["test_sessions"]], [(4, 86)])
        self.assertAlmostEqual(first["test_sessions"][0][0, 0], 80 / 79)
        self.assertAlmostEqual(second["test_sessions"][0][0, 0], 100 / 99)

    def test_preserves_sessions_and_fits_one_scaler_on_four_train_parts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset_dir = Path(temporary_dir)
            write_hai_data(dataset_dir)

            inputs = load_hai_sessions(
                dataset_dir, ratio_percent=100, val_fraction=0.2, min_train_length=4,
            )

        self.assertEqual(inputs["feature_names"], tuple(feature_names(86)))
        self.assertEqual([array.shape for array in inputs["train_sessions"]], [(80, 86)] * 4)
        self.assertEqual([array.shape for array in inputs["validation_sessions"]], [(20, 86)] * 4)
        self.assertEqual([array.shape for array in inputs["test_sessions"]], [(3, 86), (4, 86)])
        self.assertEqual(inputs["test_labels"][0].tolist(), [0, 1, 0])
        self.assertEqual(inputs["test_labels"][1].tolist(), [0, 1, 0, 1])
        self.assertEqual(
            [split["kept_train_range"] for split in inputs["session_splits"]],
            [(0, 100)] * 4,
        )
        self.assertEqual(
            [split["validation_range"] for split in inputs["session_splits"]],
            [(80, 100)] * 4,
        )
        self.assertAlmostEqual(inputs["train_sessions"][0][-1, 0], 79 / 139)
        self.assertAlmostEqual(inputs["train_sessions"][3][-1, 0], 1.0)
        self.assertAlmostEqual(inputs["validation_sessions"][3][0, 0], 140 / 139)
        self.assertAlmostEqual(inputs["test_sessions"][0][0, 0], 80 / 139)

    def test_rejects_ratio_outside_hai_plan(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset_dir = Path(temporary_dir)
            write_hai_data(dataset_dir)
            with self.assertRaisesRegex(ValueError, "HAI ratio"):
                load_hai_sessions(dataset_dir, ratio_percent=10, val_fraction=0.1, min_train_length=6)

    def test_rejects_nonbinary_hai_labels(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset_dir = Path(temporary_dir)
            write_hai_data(dataset_dir)
            label_path = dataset_dir / "label-test1.csv"
            labels = pandas.read_csv(label_path)
            labels.loc[1, "label"] = 2
            labels.to_csv(label_path, index=False)

            with self.assertRaisesRegex(ValueError, "0 또는 1"):
                load_hai_sessions(
                    dataset_dir, ratio_percent=100, val_fraction=0.2, min_train_length=4,
                )

    def test_hai_uses_scaler_return_values_when_transform_copies(self):
        class CopyingScaler:
            def partial_fit(self, _array):
                return self

            def transform(self, array):
                return array + 1000

        with tempfile.TemporaryDirectory() as temporary_dir:
            dataset_dir = Path(temporary_dir)
            write_hai_data(dataset_dir)
            with patch("src.data_split.load_hai_sessions.MinMaxScaler", return_value=CopyingScaler()):
                inputs = load_hai_sessions(
                    dataset_dir, ratio_percent=100, val_fraction=0.2, min_train_length=4,
                )

        self.assertEqual(inputs["train_sessions"][0][0, 0], 1000)
        self.assertEqual(inputs["validation_sessions"][0][0, 0], 1080)
        self.assertEqual(inputs["test_sessions"][0][0, 0], 1080)


if __name__ == "__main__":
    unittest.main()
