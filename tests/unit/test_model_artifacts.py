"""새 모델의 shape-aware 점수 저장 계약을 검증한다."""

import hashlib
import json
import inspect
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy

from src.common.build_config_id import build_common_recipe_id
from src.common.model_registry import load_model_registry
from src.common.save_model_artifacts import save_model_score as _save_model_score
from src.data_split.split_ratio_prefix import compute_prefix_counts


INPUT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "input_manifest.yaml"
DEVELOPMENT_IDENTITY = {
    "dataset_role": "development",
    "split_role": "dev18_selection",
    "input_manifest_sha256": hashlib.sha256(INPUT_MANIFEST_PATH.read_bytes()).hexdigest(),
    "final_policy_membership_sha256": None,
}


def save_model_score(output, output_dir, **arguments):
    """합성 saver 검사는 Dev18 신원과 합성 실행 증거를 명시한다."""
    arguments["dataset"] = "DEV18"
    arguments["execution_identity"] = DEVELOPMENT_IDENTITY
    arguments.setdefault(
        "target_use",
        "training_free"
        if output.get("calibration_mode") == "none"
        else "fit_validation",
    )
    target_free = arguments["target_use"] in {
        "training_free", "strict_zero_shot",
    }
    training_sessions = []
    if not target_free:
        supplied = arguments.get("validation_scores", ())
        references = (
            (supplied,)
            if isinstance(supplied, numpy.ndarray) and len(supplied)
            else tuple(supplied)
        )
        validation_counts = [len(reference) for reference in references] or [1]
        for validation_count in validation_counts:
            ratio = arguments["ratio"]
            observation_count = (500 * validation_count + ratio - 1) // ratio
            available_count, fit_count, expected_validation = compute_prefix_counts(
                observation_count, ratio,
            )
            training_sessions.append({
                "available_count": available_count,
                "fit_count": fit_count,
                "validation_count": expected_validation,
                "observation_count": observation_count,
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            })
    arguments.setdefault("execution_evidence", {
        "measurement_protocol_id": "synthetic_unit.v1",
        "execution_phase": "development_hpo",
        "status": "complete",
        "retry_count": 0,
        "training_sessions": training_sessions,
        "test_sessions": [{
            "observation_count": output.get("source_end_exclusive", 0),
            "observed_duration_seconds": None,
            "duration_basis": "unavailable",
        }],
        "timing": {
            "split_preprocess_seconds": 0.0,
            "model_setup_seconds": 0.0,
            "training_seconds": 0.0,
            "validation_inference_seconds": 0.0,
            "test_inference_seconds": 0.0,
        },
        "runtime_seconds": 0.0,
        "peak_memory_mb": 0.0,
        "model_artifact_bytes": 0,
    })
    return _save_model_score(output, output_dir, **arguments)


class TestSaveModelScore(unittest.TestCase):
    COMMON_RECIPE = load_model_registry()["common_recipe"]
    COMMON_RECIPE_ID = build_common_recipe_id(COMMON_RECIPE)

    def test_semantic_values_cannot_be_overridden_outside_the_recipe(self):
        parameters = inspect.signature(save_model_score).parameters
        self.assertNotIn("epsilon", parameters)
        self.assertNotIn("smoothing_window", parameters)

    def test_scalar_zero_shot_is_not_target_calibrated(self):
        output = {
            "scores": numpy.array([0.1, 0.4, 0.2]),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "same_timestep",
            "primitive": "probability",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 2,
            "maximum_effective_lookahead": 2,
            "normalization_scope": "none",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            saved = save_model_score(
                output, temporary_dir, dataset="DEV18", series=1,
                model="TimeRCD", tier="t3", ratio=100, seed=3,
                config_id="c123456789abc",
                normalization_scope="none",
                common_recipe=self.COMMON_RECIPE,
                common_recipe_id=self.COMMON_RECIPE_ID,
            )
            raw = numpy.load(next(path for path in saved["score_paths"] if "__raw__" in path))
            metadata = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))

        numpy.testing.assert_array_equal(raw, output["scores"])
        self.assertEqual(len(saved["score_paths"]), 2)
        self.assertEqual(metadata["config_id"], "c123456789abc")
        self.assertEqual(metadata["common_recipe_id"], self.COMMON_RECIPE_ID)
        self.assertEqual(metadata["epsilon"], 0.01)
        self.assertEqual(metadata["smoothing"], {
            "kind": "trailing_mean", "window": 4,
            "boundary": "first_three_timesteps_zero",
        })
        self.assertEqual(metadata["pipeline_order"], self.COMMON_RECIPE["order"])
        self.assertEqual(metadata["channel_count"], 0)
        self.assertEqual(metadata["normalization_scope"], "none")
        self.assertEqual(metadata["primitive"], "probability")
        self.assertEqual(metadata["aggregation_mode"], "model_native_scalar")
        self.assertEqual(metadata["window_size"], 0)
        self.assertEqual(metadata["test_length"], 3)
        self.assertEqual(metadata["evaluation_mode"], "offline_noncausal")
        self.assertEqual(metadata["lookahead"], 2)
        self.assertEqual(metadata["maximum_effective_lookahead"], 2)
        self.assertRegex(metadata["config_registry_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsNone(saved["calibration_reference_path"])
        self.assertIsNone(metadata["calibration_reference"])
        self.assertEqual(
            metadata["execution_evidence"]["measurement_protocol_id"],
            "synthetic_unit.v1",
        )
        self.assertEqual(
            metadata["execution_evidence"]["execution_phase"], "development_hpo",
        )

    def test_score_variant_is_recorded_for_shared_tspulse_forward(self):
        output = {
            "scores": numpy.array([0.1, 0.2, 0.3]),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "same_timestep",
            "primitive": "raw_fft_head_score",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 1,
            "maximum_effective_lookahead": 2,
            "normalization_scope": "model_revin_only",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            saved = save_model_score(
                output, temporary_dir, dataset="DEV18", series=1,
                model="TSPulse", tier="t3", ratio=100, seed=3,
                config_id="c123456789abc", normalization_scope="model_revin_only",
                score_variant="fft",
                common_recipe=self.COMMON_RECIPE,
                common_recipe_id=self.COMMON_RECIPE_ID,
            )
            metadata = json.loads(
                Path(saved["metadata_path"]).read_text(encoding="utf-8")
            )
        self.assertEqual(metadata["score_variant"], "fft")

    def test_preserves_only_supported_native_alignment_metadata(self):
        output = {
            "scores": numpy.array([0.1, 0.2, 0.3]),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "boundary_repeat_from_native",
            "primitive": "raw_prediction_head_score",
            "calibration_mode": "none",
            "native_source_start": 2,
            "native_source_end_exclusive": 3,
            "boundary_repeat": {"left": 2, "right": 0},
            "boundary_policy": "edge_repeat",
            "lookahead": 0,
            "maximum_effective_lookahead": 2,
            "evaluation_mode": "offline_noncausal",
            "normalization_scope": "model_revin_only",
            "unapproved_native_detail": "must not leak",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            saved = save_model_score(
                output, temporary_dir, dataset="DEV18", series=1,
                model="TSPulse", tier="t3", ratio=100, seed=3,
                config_id="c123456789abc", normalization_scope="model_revin_only",
                score_variant="pred",
                common_recipe=self.COMMON_RECIPE,
                common_recipe_id=self.COMMON_RECIPE_ID,
            )
            metadata = json.loads(
                Path(saved["metadata_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(metadata["native_source_start"], 2)
        self.assertEqual(metadata["native_source_end_exclusive"], 3)
        self.assertEqual(metadata["boundary_repeat"], {"left": 2, "right": 0})
        self.assertEqual(metadata["boundary_policy"], "edge_repeat")
        self.assertEqual(metadata["lookahead"], 0)
        self.assertEqual(metadata["maximum_effective_lookahead"], 2)
        self.assertNotIn("unapproved_native_detail", metadata)

    def test_channel_scores_use_only_supplied_validation_reference(self):
        output = {
            "scores": numpy.array([[2.0, 20.0], [4.0, 40.0]]),
            "source_start": 5,
            "source_end_exclusive": 7,
            "alignment": "next_step",
            "primitive": "absolute_error",
            "calibration_mode": "validation_median_iqr",
            "evaluation_mode": "causal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "current_prefix_validation",
        }
        validation = (
            numpy.array([[1.0, 10.0]]),
            numpy.array([[3.0, 30.0]]),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            saved = save_model_score(
                output, temporary_dir, dataset="DEV18", series=2,
                model="GDN", tier="t2", ratio=5, seed=0,
                config_id="cabcdef012345",
                normalization_scope="current_prefix_validation",
                validation_scores=validation,
                common_recipe=self.COMMON_RECIPE,
                common_recipe_id=self.COMMON_RECIPE_ID,
            )
            channel_path = next(
                path for path in saved["score_paths"]
                if "__raw__" in path and path.endswith("__channels.npy")
            )
            channels = numpy.load(channel_path)
            metadata = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))
            reference_path = Path(saved["calibration_reference_path"])
            reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
            with numpy.load(reference_path, allow_pickle=False) as reference:
                reference_keys = reference.files
                reference_sessions = tuple(
                    reference[key].copy() for key in reference.files
                )

        numpy.testing.assert_allclose(
            channels, [[0.0, 0.0], [2.0 / 1.01, 20.0 / 10.01]],
        )
        self.assertEqual(len(saved["score_paths"]), 4)
        self.assertEqual(metadata["channel_count"], 2)
        self.assertEqual(metadata["source_start"], 5)
        self.assertEqual(metadata["source_end_exclusive"], 7)
        self.assertEqual(metadata["alignment"], "next_step")
        self.assertEqual(metadata["aggregation_mode"], "max")
        self.assertEqual(reference_keys, ["session_000", "session_001"])
        for saved_session, expected_session in zip(reference_sessions, validation):
            numpy.testing.assert_array_equal(saved_session, expected_session)
        self.assertEqual(metadata["calibration_reference"], {
            "file": reference_path.name,
            "sha256": reference_sha256,
            "session_shapes": [[1, 2], [1, 2]],
            "source_starts": [0, 0],
            "sample_count": 2,
            "median": [2.0, 20.0],
            "iqr": [1.0, 10.0],
            "score_space": "adapter_native_pre_calibration",
        })

    def test_rejects_adapter_range_mismatch_before_writing(self):
        output = {
            "scores": numpy.ones(3),
            "source_start": 5,
            "source_end_exclusive": 9,
            "alignment": "next_step",
            "primitive": "probability",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "none",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "source 범위"):
                save_model_score(
                    output, temporary_dir, dataset="DEV18", series=1,
                    model="TimeRCD", tier="t3", ratio=100, seed=3,
                    config_id="c123456789abc", normalization_scope="none",
                    common_recipe=self.COMMON_RECIPE,
                    common_recipe_id=self.COMMON_RECIPE_ID,
                )
            self.assertEqual(tuple(Path(temporary_dir).iterdir()), ())

    def test_rejects_execution_evidence_before_accessing_output_directory(self):
        output = {
            "scores": numpy.ones(3),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "same_timestep",
            "primitive": "probability",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "none",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_directory = Path(temporary_dir) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "execution_evidence 필드"):
                save_model_score(
                    output, missing_directory, dataset="DEV18", series=1,
                    model="TimeRCD", tier="t3", ratio=100, seed=3,
                    config_id="c123456789abc", normalization_scope="none",
                    common_recipe=self.COMMON_RECIPE,
                    common_recipe_id=self.COMMON_RECIPE_ID,
                    execution_evidence={},
                )
            self.assertFalse(missing_directory.exists())

    def test_cross_checks_target_use_calibration_and_validation_before_writing(self):
        output = {
            "scores": numpy.ones(3),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "same_timestep",
            "primitive": "probability",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "none",
        }
        cases = (
            (
                "training_free",
                {**output, "calibration_mode": "validation_median_iqr"},
                (numpy.ones(2),),
            ),
            ("strict_zero_shot", output, (numpy.ones(2),)),
            ("fit_validation", output, ()),
            (
                "fit_validation",
                {**output, "calibration_mode": "validation_median_iqr"},
                (),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            for index, (target_use, changed, validation_scores) in enumerate(cases):
                missing_directory = Path(temporary_dir) / f"must-not-exist-{index}"
                with self.subTest(target_use=target_use, index=index):
                    with self.assertRaisesRegex(ValueError, "calibration|validation"):
                        save_model_score(
                            changed,
                            missing_directory,
                            dataset="DEV18",
                            series=1,
                            model="TimeRCD",
                            target_use=target_use,
                            tier="t3",
                            ratio=100,
                            seed=3,
                            config_id="c123456789abc",
                            normalization_scope="none",
                            validation_scores=validation_scores,
                            common_recipe=self.COMMON_RECIPE,
                            common_recipe_id=self.COMMON_RECIPE_ID,
                        )
                    self.assertFalse(missing_directory.exists())

    def test_rejects_test_observation_count_that_disagrees_with_score_range(self):
        output = {
            "scores": numpy.ones(3),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "same_timestep",
            "primitive": "probability",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "none",
        }
        forged = {
            "measurement_protocol_id": "synthetic_unit.v1",
            "execution_phase": "development_hpo",
            "status": "complete",
            "retry_count": 0,
            "training_sessions": [],
            "test_sessions": [{
                "observation_count": 4,
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "timing": {
                "split_preprocess_seconds": 0.0,
                "model_setup_seconds": 0.0,
                "training_seconds": 0.0,
                "validation_inference_seconds": 0.0,
                "test_inference_seconds": 0.0,
            },
            "runtime_seconds": 0.0,
            "peak_memory_mb": 0.0,
            "model_artifact_bytes": 0,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_directory = Path(temporary_dir) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "observation_count"):
                save_model_score(
                    output,
                    missing_directory,
                    dataset="DEV18",
                    series=1,
                    model="TimeRCD",
                    target_use="training_free",
                    tier="t3",
                    ratio=100,
                    seed=3,
                    config_id="c123456789abc",
                    normalization_scope="none",
                    common_recipe=self.COMMON_RECIPE,
                    common_recipe_id=self.COMMON_RECIPE_ID,
                    execution_evidence=forged,
                )
            self.assertFalse(missing_directory.exists())

    def test_binds_learned_counts_to_ratio_and_validation_sessions(self):
        output = {
            "scores": numpy.ones(2),
            "source_start": 0,
            "source_end_exclusive": 2,
            "alignment": "same_timestep",
            "primitive": "absolute_error",
            "calibration_mode": "validation_median_iqr",
            "evaluation_mode": "causal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "current_prefix_validation",
        }
        evidence = {
            "measurement_protocol_id": "synthetic_unit.v1",
            "execution_phase": "development_hpo",
            "status": "complete",
            "retry_count": 0,
            "training_sessions": [{
                "available_count": 5,
                "fit_count": 4,
                "validation_count": 1,
                "observation_count": 5,
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "test_sessions": [{
                "observation_count": 2,
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "timing": {
                "split_preprocess_seconds": 0.0,
                "model_setup_seconds": 0.0,
                "training_seconds": 0.0,
                "validation_inference_seconds": 0.0,
                "test_inference_seconds": 0.0,
            },
            "runtime_seconds": 0.0,
            "peak_memory_mb": 0.0,
            "model_artifact_bytes": 0,
        }
        wrong_counts = deepcopy(evidence)
        wrong_counts["training_sessions"][0].update({
            "available_count": 4,
            "fit_count": 3,
        })
        cases = (
            (wrong_counts, numpy.ones(1), "ratio"),
            (evidence, numpy.ones(2), "validation"),
        )
        for execution_evidence, validation_scores, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                missing_directory = Path(directory) / "must-not-exist"
                with self.assertRaisesRegex(ValueError, message):
                    save_model_score(
                        output,
                        missing_directory,
                        dataset="DEV18",
                        series=1,
                        model="GDN",
                        target_use="fit_validation",
                        tier="t2",
                        ratio=100,
                        seed=0,
                        config_id="cabcdef012345",
                        normalization_scope="current_prefix_validation",
                        validation_scores=validation_scores,
                        common_recipe=self.COMMON_RECIPE,
                        common_recipe_id=self.COMMON_RECIPE_ID,
                        execution_evidence=execution_evidence,
                    )
                self.assertFalse(missing_directory.exists())

    def test_rejects_phase_that_disagrees_with_identity_and_target_use(self):
        output = {
            "scores": numpy.ones(3),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "same_timestep",
            "primitive": "probability",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "none",
        }
        forged = {
            "measurement_protocol_id": "synthetic_unit.v1",
            "execution_phase": "target_free_inference",
            "status": "complete",
            "retry_count": 0,
            "training_sessions": [],
            "test_sessions": [{
                "observation_count": 3,
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "timing": {
                "split_preprocess_seconds": 0.0,
                "model_setup_seconds": 0.0,
                "training_seconds": 0.0,
                "validation_inference_seconds": 0.0,
                "test_inference_seconds": 0.0,
            },
            "runtime_seconds": 0.0,
            "peak_memory_mb": 0.0,
            "model_artifact_bytes": 0,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_directory = Path(temporary_dir) / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "execution_phase"):
                save_model_score(
                    output, missing_directory, dataset="DEV18", series=1,
                    model="TimeRCD", tier="t3", ratio=100, seed=3,
                    config_id="c123456789abc", normalization_scope="none",
                    common_recipe=self.COMMON_RECIPE,
                    common_recipe_id=self.COMMON_RECIPE_ID,
                    target_use="training_free", execution_evidence=forged,
                )
            self.assertFalse(missing_directory.exists())

    def test_rejects_missing_or_mismatched_normalization_scope_before_writing(self):
        base = {
            "scores": numpy.ones(3),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "same_timestep",
            "primitive": "probability",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            for output in (base, {**base, "normalization_scope": "model_revin_only"}):
                with self.subTest(output=output):
                    with self.assertRaisesRegex(ValueError, "normalization_scope"):
                        save_model_score(
                            output, temporary_dir, dataset="DEV18", series=1,
                            model="TimeRCD", tier="t3", ratio=100, seed=3,
                            config_id="c123456789abc", normalization_scope="none",
                            common_recipe=self.COMMON_RECIPE,
                            common_recipe_id=self.COMMON_RECIPE_ID,
                        )
            self.assertEqual(tuple(Path(temporary_dir).iterdir()), ())

    def test_accepts_one_numpy_validation_matrix_without_truth_testing_it(self):
        output = {
            "scores": numpy.array([[2.0], [4.0]]),
            "source_start": 0,
            "source_end_exclusive": 2,
            "alignment": "same_timestep",
            "primitive": "absolute_error",
            "calibration_mode": "validation_median_iqr",
            "evaluation_mode": "causal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "current_prefix_validation",
        }
        validation = numpy.array([[1.0], [3.0]])
        with tempfile.TemporaryDirectory() as temporary_dir:
            saved = save_model_score(
                output, temporary_dir, dataset="DEV18", series=2,
                model="GDN", tier="t2", ratio=5, seed=0,
                config_id="cabcdef012345",
                common_recipe=self.COMMON_RECIPE,
                common_recipe_id=self.COMMON_RECIPE_ID,
                normalization_scope="current_prefix_validation",
                validation_scores=validation,
            )
            channels = numpy.load(next(
                path for path in saved["score_paths"]
                if "__raw__" in path and path.endswith("__channels.npy")
            ))
        numpy.testing.assert_allclose(channels, [[0.0], [2.0 / 1.01]])

    def test_requires_well_formed_common_recipe_id_before_writing(self):
        output = {
            "scores": numpy.ones(3),
            "source_start": 0,
            "source_end_exclusive": 3,
            "alignment": "same_timestep",
            "primitive": "probability",
            "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal",
            "lookahead": 0,
            "maximum_effective_lookahead": 0,
            "normalization_scope": "none",
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "common_recipe_id"):
                save_model_score(
                    output, temporary_dir, dataset="DEV18", series=1,
                    model="TimeRCD", tier="t3", ratio=100, seed=3,
                    config_id="c123456789abc", common_recipe_id="bad",
                    common_recipe=self.COMMON_RECIPE,
                    normalization_scope="none",
                )
            self.assertEqual(tuple(Path(temporary_dir).iterdir()), ())

    def test_rejects_recipe_id_mismatch_before_writing(self):
        output = {
            "scores": numpy.ones(3), "source_start": 0,
            "source_end_exclusive": 3, "alignment": "same_timestep",
            "primitive": "probability", "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal", "lookahead": 0,
            "maximum_effective_lookahead": 0, "normalization_scope": "none",
        }
        changed = deepcopy(self.COMMON_RECIPE)
        changed["score_calibration"]["epsilon"] = 0.02
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "common_recipe.*ID|common_recipe_id"):
                save_model_score(
                    output, temporary_dir, dataset="DEV18", series=1,
                    model="TimeRCD", tier="t3", ratio=100, seed=3,
                    config_id="c123456789abc", common_recipe=changed,
                    common_recipe_id=self.COMMON_RECIPE_ID,
                    normalization_scope="none",
                )
            self.assertEqual(tuple(Path(temporary_dir).iterdir()), ())

    def test_rejects_unsupported_sealed_recipe_before_writing(self):
        output = {
            "scores": numpy.ones(3), "source_start": 0,
            "source_end_exclusive": 3, "alignment": "same_timestep",
            "primitive": "probability", "calibration_mode": "none",
            "evaluation_mode": "offline_noncausal", "lookahead": 0,
            "maximum_effective_lookahead": 0, "normalization_scope": "none",
        }
        changed = deepcopy(self.COMMON_RECIPE)
        changed["order"]["raw"] = "aggregate_then_normalize"
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "order.raw"):
                save_model_score(
                    output, temporary_dir, dataset="DEV18", series=1,
                    model="TimeRCD", tier="t3", ratio=100, seed=3,
                    config_id="c123456789abc", common_recipe=changed,
                    common_recipe_id=build_common_recipe_id(changed),
                    normalization_scope="none",
                )
            self.assertEqual(tuple(Path(temporary_dir).iterdir()), ())


if __name__ == "__main__":
    unittest.main()
