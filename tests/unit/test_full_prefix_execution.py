"""Full-prefix 분할·실행·정규화의 데이터 경계를 모델 없이 확인한다."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy

from src.common.build_config_id import build_common_recipe_id
from src.common.execution_evidence import (
    FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, build_execution_evidence,
    validate_execution_evidence_for_run,
)
from src.common.model_feasibility import assess_candidate
from src.common.run_registered_model import execute_registered_model, prepare_session_inputs
from src.common.save_model_artifacts import save_model_score
from src.data_split.split_ratio_prefix import compute_prefix_counts, split_ratio_prefix


PROTOCOL = FULL_PREFIX_MEASUREMENT_PROTOCOL_ID
LEGACY_FULL_PREFIX_RECIPE = {
    "methodology_revision": "source_faithful_v3", "training_split": "full_prefix_v2",
    "score_calibration": {
        "fit_validation": "validation_median_iqr", "target_free": "none",
        "full_prefix_scalar": "none", "full_prefix_channels": "fit_median_iqr",
        "epsilon": 0.01,
    },
    "smoothing": {"kind": "trailing_mean", "window": 4, "boundary": "first_three_timesteps_zero"},
    "order": {"raw": "normalize_then_aggregate", "smoothed": "normalize_then_smooth_per_channel_then_aggregate"},
    "aggregation": {"channel_scores": "max", "scalar_scores": "model_native"},
}


def _evidence(model="GDN"):
    _, _, split = split_ratio_prefix(numpy.zeros((20, 2)), 40, full_prefix=True)
    return build_execution_evidence(
        split,
        {name: 0.0 for name in (
            "split_preprocess_seconds", "model_setup_seconds", "training_seconds",
            "validation_inference_seconds", "calibration_inference_seconds",
            "test_inference_seconds",
        )},
        spec={"dataset_role": "development", "target_use": "fit_full_prefix", "ratio": 40},
        measurement_protocol_id=PROTOCOL, retry_count=0,
        training_session_durations=[{"observed_duration_seconds": None, "duration_basis": "unavailable"}],
        test_input_sessions=(numpy.zeros((4, 2)),),
        test_session_durations=[{"observed_duration_seconds": None, "duration_basis": "unavailable"}],
        peak_memory_mb=0.0, model_artifact_bytes=0,
    )


class TestFullPrefixExecution(unittest.TestCase):
    def test_full_prefix_keeps_all_available_rows_and_no_validation(self):
        normal = numpy.arange(200, dtype=float).reshape(100, 2)
        self.assertEqual(compute_prefix_counts(100, 40, full_prefix=True), (40, 40, 0))
        self.assertEqual(compute_prefix_counts(100, 40), (40, 32, 8))
        prepared = prepare_session_inputs(
            normal_training=normal, test_sessions=(normal[:4],),
            ratio_percent=40, scale=True, full_prefix=True,
        )
        self.assertEqual(prepared["fit_sessions"][0].shape, (40, 2))
        self.assertEqual(prepared["validation_sessions"], ())
        self.assertEqual(prepared["scaler_state"]["sample_count"], 40)
        self.assertEqual(prepared["session_splits"][0]["validation_range"], (40, 40))

    def test_feasibility_does_not_require_validation_but_keeps_fit_minimum(self):
        parameters = {"window": 100}
        self.assertEqual(assess_candidate("PCA_LEGACY", parameters, 101, 0, 100, 2, full_prefix=True)["status"], "feasible")
        self.assertEqual(assess_candidate("PCA_LEGACY", parameters, 100, 0, 100, 2, full_prefix=True)["status"], "structurally_infeasible")
        self.assertEqual(assess_candidate("PCA_LEGACY", parameters, 101, 0, 100, 2)["status"], "structurally_infeasible")

    def test_scalar_runner_skips_validation_inference(self):
        calls = []

        class Adapter:
            def fit(self, values):
                calls.append(("fit", len(values)))

            def score(self, values):
                calls.append(("score", len(values)))
                return {"scores": numpy.zeros(len(values)), "calibration_mode": "validation_median_iqr", "normalization_scope": "current_prefix_validation"}

        spec = {
            "model": "PCA_LEGACY", "target_use": "fit_full_prefix", "tier": "t1",
            "ratio": 40, "seed": 1,
            "hyperparameters": {"window": 100, "zero_pruning": False, "n_components": 0.25},
        }
        with patch("src.common.run_registered_model.validate_registered_spec"), patch("src.common.run_registered_model.set_reproducible_seed", return_value={}):
            result = execute_registered_model(
                spec, normal_training=numpy.zeros((100, 2)),
                test_sessions=(numpy.zeros((4, 2)),), device="cpu",
                entrypoint=lambda **kwargs: Adapter(),
            )
        self.assertEqual(calls, [("fit", 40), ("score", 4)])
        self.assertEqual(result["validation_outputs"], ())
        self.assertEqual(result["calibration_outputs"], ())
        self.assertEqual(result["test_outputs"][0]["calibration_mode"], "none")

    def test_evidence_records_full_counts_and_distinct_calibration_time(self):
        evidence = _evidence()
        self.assertEqual(evidence["training_sessions"][0]["observed_row"], 8)
        self.assertNotIn("validation_count", evidence["training_sessions"][0])
        self.assertIn("calibration_inference_seconds", evidence["timing"])
        evidence["training_sessions"][0].update(fit_count=6, validation_count=2)
        with self.assertRaises(ValueError):
            validate_execution_evidence_for_run(evidence, dataset_role="development", target_use="fit_full_prefix")

    def test_saver_binds_fit_reference_and_rejects_fake_validation(self):
        recipe = deepcopy(LEGACY_FULL_PREFIX_RECIPE)
        output = {
            "scores": numpy.array([[1., 3.], [2., 4.]]), "source_start": 2,
            "source_end_exclusive": 4, "alignment": "next_step", "primitive": "absolute_error",
            "calibration_mode": "fit_median_iqr", "normalization_scope": "current_prefix_fit",
            "evaluation_mode": "causal", "lookahead": 0, "maximum_effective_lookahead": 0,
        }
        arguments = dict(
            dataset="DEV18", series=1, model="GDN", target_use="fit_full_prefix", tier="t2",
            ratio=40, seed=1, config_id="cabcdef012345", common_recipe=recipe,
            common_recipe_id=build_common_recipe_id(recipe), normalization_scope="current_prefix_fit",
            config_registry_sha256="a" * 64,
            execution_identity={"dataset_role": "development", "split_role": "dev18_selection", "input_manifest_sha256": "b" * 64, "final_policy_membership_sha256": None},
            execution_evidence=_evidence(), calibration_scores=(numpy.ones((6, 2)),),
            calibration_source_starts=(2,),
        )
        with tempfile.TemporaryDirectory() as directory:
            saved = save_model_score(output, directory, **arguments)
            metadata = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["calibration_reference"]["source"], "fit")
            self.assertEqual(metadata["calibration_reference"]["sample_count"], 6)
            with self.assertRaises(ValueError):
                save_model_score(output, directory, **{**arguments, "calibration_scores": (numpy.ones((2, 2)),)})
            with self.assertRaises(ValueError):
                save_model_score(output, directory, validation_scores=(numpy.ones((6, 2)),), **arguments)


if __name__ == "__main__":
    unittest.main()
