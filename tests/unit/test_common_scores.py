"""src/common/save_scores.py 단위 테스트."""

import json
import os
import tempfile
import unittest

import numpy

from src.common.save_scores import (
    save_score_arrays,
    save_score_bundle,
    save_score_metadata,
    snapshot_config,
)


class TestSaveScoreMetadata(unittest.TestCase):
    def test_sidecar_created_with_dryrun_settings(self):
        # 1-step forecast: window_size=8, test 200행 → 점수 길이 192.
        with tempfile.TemporaryDirectory() as output_dir:
            path = save_score_metadata(
                output_dir, dataset="SYNTH", series=0, model="GDN", tier="t0",
                ratio=100, seed=1, window_size=8, test_length=200,
                score_length=192, label_slice=(8, 200), source_start=8,
                source_end_exclusive=200, alignment="next_step",
            )
            self.assertEqual(os.path.basename(path),
                             "SYNTH__00__GDN__t0__r100__s1__raw__trainnorm.meta.json")
            with open(path, encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(metadata["label_slice"], [8, 200])
            self.assertEqual(metadata["score_length"], 192)
            self.assertEqual(metadata["source_start"], 8)
            self.assertEqual(metadata["source_end_exclusive"], 200)
            self.assertEqual(metadata["alignment"], "next_step")

    def test_sidecar_rejects_mismatched_label_slice(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "label_slice"):
                save_score_metadata(
                    output_dir, dataset="SYNTH", series=0, model="GDN", tier="t0",
                    ratio=100, seed=1, window_size=8, test_length=200,
                    score_length=192, label_slice=(0, 192), source_start=8,
                    source_end_exclusive=200, alignment="next_step",
                )


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

    def test_rejects_empty_nonfinite_or_non_matrix_scores(self):
        invalid_arrays = (
            numpy.empty((0, 2)),
            numpy.array([1.0, 2.0]),
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

    def test_bundle_saves_eight_arrays_from_separate_reference_errors(self):
        train_errors = numpy.array([[1.0, -2.0], [2.0, -4.0]])
        validation_errors = numpy.array([[3.0, -6.0]])
        test_errors = numpy.array([[2.0, -3.0], [4.0, -9.0]])
        with tempfile.TemporaryDirectory() as output_dir:
            paths = save_score_bundle(
                (train_errors, validation_errors), test_errors, output_dir,
                dataset="GHL", series=1, model="GDN", tier="t2",
                ratio=5, seed=1, epsilon=1e-8, smoothing_window=2,
            )
            self.assertEqual(len(paths), 8)
            self.assertEqual(len({os.path.basename(path) for path in paths}), 8)
            trainnorm = numpy.load(next(
                path for path in paths
                if path.endswith("raw__trainnorm__channels.npy")
            ))
            testnorm = numpy.load(next(
                path for path in paths
                if path.endswith("raw__testnorm__channels.npy")
            ))
            self.assertEqual(trainnorm.shape, test_errors.shape)
            self.assertEqual(testnorm.shape, test_errors.shape)

    def test_bundle_rejects_nonfinite_before_writing_any_score(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "finite"):
                save_score_bundle(
                    (numpy.ones((2, 2)),), numpy.array([[1.0, numpy.nan]]),
                    output_dir, dataset="GHL", series=1, model="GDN", tier="t2",
                    ratio=5, seed=1, epsilon=1e-8,
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
