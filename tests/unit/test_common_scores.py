"""src/common/save_scores.py 단위 테스트."""

import json
import os
import tempfile
import unittest

import numpy

from src.common.save_scores import (
    save_score_arrays,
    snapshot_config,
)


class TestSaveScoreArrays(unittest.TestCase):
    def test_scalar_scores_save_two_files_without_fake_channels(self):
        with tempfile.TemporaryDirectory() as output_dir:
            saved_paths = save_score_arrays(
                numpy.array([1.0, 3.0, 5.0]), output_dir,
                dataset="GHL", series=1, model="TimeRCD", tier="t3",
                ratio=100, seed=3, norm_kind="trainnorm", smoothing_window=2,
            )
            names = sorted(os.path.basename(path) for path in saved_paths)
            self.assertEqual(names, [
                "GHL__01__TimeRCD__t3__r100__s3__raw__trainnorm.npy",
                "GHL__01__TimeRCD__t3__r100__s3__smoothed__trainnorm.npy",
            ])
            self.assertFalse(any("channels" in name for name in names))

    def test_four_files_with_convention_names_and_values(self):
        # 3×2 장난감 배열, smoothing_window=2 로 손계산 가능하게 한다.
        # channel_scores = [[1,5],[2,1],[3,2]]
        # raw 집계(max) = [5, 2, 3]
        # 후행 2-창 smoothing: t0=[0,0], t1=[1.5,3.0], t2=[2.5,1.5] → 집계 = [0, 3, 2.5]
        channel_scores = numpy.array([[1.0, 5.0], [2.0, 1.0], [3.0, 2.0]])
        with tempfile.TemporaryDirectory() as output_dir:
            saved_paths = save_score_arrays(
                channel_scores, output_dir,
                dataset="GHL", series=3, model="GDN", tier="t2", ratio=10, seed=1,
                norm_kind="trainnorm", smoothing_window=2,
            )
            saved_names = sorted(os.path.basename(path) for path in saved_paths)
            self.assertEqual(saved_names, [
                "GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy",
                "GHL__03__GDN__t2__r010__s1__raw__trainnorm__channels.npy",
                "GHL__03__GDN__t2__r010__s1__smoothed__trainnorm.npy",
                "GHL__03__GDN__t2__r010__s1__smoothed__trainnorm__channels.npy",
            ])
            arrays = {os.path.basename(path): numpy.load(path) for path in saved_paths}
            numpy.testing.assert_array_equal(
                arrays["GHL__03__GDN__t2__r010__s1__raw__trainnorm__channels.npy"], channel_scores)
            numpy.testing.assert_array_equal(
                arrays["GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy"], [5.0, 2.0, 3.0])
            numpy.testing.assert_array_equal(
                arrays["GHL__03__GDN__t2__r010__s1__smoothed__trainnorm__channels.npy"],
                [[0.0, 0.0], [1.5, 3.0], [2.5, 1.5]])
            numpy.testing.assert_array_equal(
                arrays["GHL__03__GDN__t2__r010__s1__smoothed__trainnorm.npy"], [0.0, 3.0, 2.5])

    def test_rejects_empty_nonfinite_or_above_two_dimensions(self):
        invalid_arrays = (
            numpy.empty((0, 2)),
            numpy.ones((2, 2, 2)),
            numpy.array([[1.0, numpy.nan]]),
            numpy.array([[1.0, numpy.inf]]),
        )
        with tempfile.TemporaryDirectory() as output_dir:
            for invalid in invalid_arrays:
                with self.subTest(shape=invalid.shape):
                    with self.assertRaises(ValueError):
                        save_score_arrays(
                            invalid, output_dir, dataset="GHL", series=1,
                            model="GDN", tier="t2", ratio=5, seed=1,
                            norm_kind="trainnorm",
                        )
            self.assertEqual(os.listdir(output_dir), [])

class TestSnapshotConfig(unittest.TestCase):
    def test_snapshot_contains_config_and_git_hash(self):
        with tempfile.TemporaryDirectory() as output_dir:
            path = snapshot_config({"window_size": 4}, "485e26b0", output_dir)
            with open(path, encoding="utf-8") as snapshot_file:
                snapshot = json.load(snapshot_file)
            self.assertEqual(snapshot["git_commit_hash"], "485e26b0")
            self.assertEqual(snapshot["config"], {"window_size": 4})
            self.assertIn("saved_at", snapshot)


if __name__ == "__main__":
    unittest.main()
