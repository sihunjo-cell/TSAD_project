"""Tier 2 공통 산출물 저장 계약."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy
import torch

from src.common.save_tier2_artifacts import save_tier2_artifacts


class TestTier2Artifacts(unittest.TestCase):
    def test_saves_checkpoint_scores_metadata_logs_and_snapshot(self):
        result = {
            "checkpoint": {"weight": torch.tensor([1.0])},
            "reference_errors": [numpy.ones((3, 2)), numpy.ones((2, 2))],
            "test_errors": [numpy.arange(12, dtype=float).reshape(6, 2)],
            "training_log": {"enabled": False, "epochs_completed": 1},
            "timing": {"accelerator": "cpu", "training_seconds": 0.1},
        }
        identity = {
            "project_commit": "a" * 40,
            "local_model_sha256": {"model.py": "b" * 64},
            "upstream_source_commits": {"source": "c" * 40},
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = save_tier2_artifacts(
                result, temporary_dir, dataset="GHL", test_series=(1,),
                model="GDN", ratio=5, seed=1, test_lengths=(11,),
                run_config={"window_size": 5}, source_identity=identity,
                input_metadata={"path": "generated"}, epsilon=0.01,
                smoothing_window=4,
            )
            self.assertEqual(len(output["score_paths"]), 8)
            self.assertTrue(all(Path(path).is_file() for path in output["score_paths"]))
            self.assertEqual(torch.load(output["checkpoint_path"])["weight"].item(), 1.0)
            metadata = json.loads(Path(output["metadata_paths"][0]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["label_slice"], [5, 11])
            self.assertEqual(metadata["source_start"], 5)
            self.assertEqual(metadata["alignment"], "next_step")
            snapshot = json.loads(Path(output["snapshot_path"]).read_text(encoding="utf-8"))
            self.assertEqual(snapshot["config"]["source_identity"], identity)

    def test_nonfinite_test_error_writes_nothing(self):
        result = {
            "checkpoint": {},
            "reference_errors": [numpy.ones((3, 2))],
            "test_errors": [numpy.array([[1.0, numpy.nan]])],
            "training_log": {},
            "timing": {},
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "run"
            with self.assertRaisesRegex(ValueError, "finite"):
                save_tier2_artifacts(
                    result, output_dir, dataset="GHL", test_series=(1,),
                    model="GDN", ratio=5, seed=1, test_lengths=(6,),
                    run_config={}, source_identity={"project_commit": "a" * 40},
                    input_metadata={}, epsilon=0.01, smoothing_window=4,
                )
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
