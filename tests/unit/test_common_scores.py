"""src/common/save_scores.py 단위 테스트."""

import json
import os
import tempfile
import unittest

import numpy

from src.common.save_scores import (
    save_score_arrays,
    save_aggregated_score_arrays,
    save_score_metadata,
    snapshot_config,
    validation_threshold_from_scores,
)


class TestSaveScoreMetadata(unittest.TestCase):
    def test_sidecar_created_with_dryrun_settings(self):
        # 1-step forecast: window_size=8, test 200행 → 점수 길이 192.
        with tempfile.TemporaryDirectory() as output_dir:
            path = save_score_metadata(
                output_dir, dataset="SYNTH", series=0, model="GDN", tier="t0",
                ratio=100, seed=1, window_size=8, test_length=200,
                score_length=192, label_slice=(8, None),
                validation_scores=numpy.arange(100, dtype=float),
            )
            self.assertEqual(os.path.basename(path),
                             "SYNTH__00__GDN__t0__r100__s1__raw__trainnorm.meta.json")
            with open(path, encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(metadata["label_slice"], [8, None])
            self.assertEqual(metadata["score_length"], 192)
            self.assertEqual(metadata["validation_threshold_quantile"], 0.99)
            self.assertEqual(metadata["validation_score_count"], 100)
            self.assertEqual(metadata["validation_score_source"], "raw_trainnorm_aggregated_validation")
            smoothed_path = os.path.join(
                output_dir,
                "SYNTH__00__GDN__t0__r100__s1__smoothed__trainnorm.meta.json",
            )
            self.assertTrue(os.path.isfile(smoothed_path))
            testnorm_path = os.path.join(
                output_dir,
                "SYNTH__00__GDN__t0__r100__s1__raw__testnorm.meta.json",
            )
            self.assertTrue(os.path.isfile(testnorm_path))

    def test_validation_threshold_is_conservative_q99(self):
        scores = numpy.arange(100, dtype=float)
        self.assertEqual(validation_threshold_from_scores(scores), 99.0)

    def test_validation_threshold_rejects_invalid_scores(self):
        with self.assertRaises(ValueError):
            validation_threshold_from_scores(numpy.array([[0.1, 0.2]]))


class TestSaveScoreArrays(unittest.TestCase):
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

    def test_aggregated_scores_do_not_create_fake_channel_arrays(self):
        with tempfile.TemporaryDirectory() as output_dir:
            paths = save_aggregated_score_arrays(
                numpy.array([1.0, 2.0, 3.0, 4.0]), output_dir,
                dataset="GHL", series=3, model="PCA", tier="t1", ratio=10,
                seed=1, norm_kind="trainnorm", smoothing_window=2,
            )
            self.assertEqual(sorted(os.path.basename(path) for path in paths), [
                "GHL__03__PCA__t1__r010__s1__raw__trainnorm.npy",
                "GHL__03__PCA__t1__r010__s1__smoothed__trainnorm.npy",
            ])


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
