"""공식 튜닝의 호출 인자와 native 점수 저장을 모델 없이 검증한다."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy

from src.common.execution_evidence import FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, build_execution_evidence
from src.common.model_registry import load_model_registry_with_sha
from src.common.run_registered_model import build_entrypoint_arguments, execute_registered_model
from src.common.save_model_artifacts import save_model_score
from src.data_split.split_ratio_prefix import split_ratio_prefix


class PaperTuningPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.registry_sha = load_model_registry_with_sha()

    def spec(self, name, *, ratio=40, seed=2):
        model = self.registry["models"][name]
        return {"model": name, "tier": model["tier"], "target_use": model["target_use"],
                "ratio": ratio, "seed": seed, **model["candidates"][0],
                "common_recipe": self.registry["common_recipe"]}

    def test_every_registered_candidate_reaches_the_official_arguments(self):
        for name, model in self.registry["models"].items():
            for candidate in model["candidates"]:
                spec = {**self.spec(name), **candidate}
                with self.subTest(model=name, config=candidate["config_id"]):
                    arguments = build_entrypoint_arguments(spec, device="cuda", channel_count=64)
                    if model["tier"] == "t2":
                        self.assertTrue(arguments["official_procedure"])
                    if name == "PaAno":
                        self.assertEqual(arguments["memory_policy"], "official_minimum")
                        for key, value in candidate["hyperparameters"].items():
                            self.assertEqual(arguments[key], value, key)
                    if name == "GDN":
                        self.assertEqual(arguments["split_seed"], 2)
                        for source_key, argument_key in {
                            "embedding": "embedding_dimension", "hidden": "hidden_dimension",
                            "topk": "fixed_topk", "window": "window_size",
                            "epochs": "epochs", "patience": "patience", "batch_size": "batch_size",
                            "learning_rate": "learning_rate", "validation_ratio": "validation_ratio",
                        }.items():
                            self.assertEqual(arguments[argument_key], candidate["hyperparameters"][source_key])
                        self.assertEqual(arguments["optimizer_betas"], tuple(candidate["hyperparameters"]["optimizer_betas"]))
                    if name == "PCA_LEGACY":
                        self.assertEqual(arguments["n_components"], candidate["hyperparameters"]["n_components"])
                    if model["tier"] == "t3":
                        self.assertTrue(arguments["official_protocol"])
                        self.assertNotIn("inference_context_normalization", arguments)
                        self.assertEqual(arguments["context_length"], candidate["hyperparameters"]["context_length"])
                    if name == "TSPulse":
                        self.assertEqual(arguments["batch_size"], 128)
                        self.assertEqual(arguments["aggregation_window"], candidate["hyperparameters"]["aggregation_window"])

    def test_gdn_receives_each_q_prefix_before_its_internal_window_split(self):
        calls = []

        def runner(training, validation, tests, **arguments):
            calls.append((len(training[0]), validation, arguments["split_seed"]))
            return {"validation_outputs": (), "test_outputs": (), "calibration_outputs": (),
                    "training_protocol": {"validation_source": "prefix_windows"},
                    "timing": {field: 0.0 for field in (
                        "model_setup_seconds", "training_seconds", "validation_inference_seconds",
                        "calibration_inference_seconds", "test_inference_seconds")}}

        with patch("src.common.run_registered_model.validate_registered_spec"), patch(
            "src.common.run_registered_model.set_reproducible_seed", return_value={},
        ):
            for ratio in (20, 80):
                result = execute_registered_model(
                    self.spec("GDN", ratio=ratio), normal_training=numpy.zeros((100, 64)),
                    test_sessions=(numpy.zeros((20, 64)),), device="cpu", entrypoint=runner,
                )
                self.assertEqual(result["training_protocol"]["validation_source"], "prefix_windows")
                self.assertEqual(result["calibration_scope"], "full_evaluation")
        self.assertEqual(calls, [(20, (), 2), (80, (), 2)])

    def test_tspulse_reports_full_evaluation_calibration_without_model_loading(self):
        with patch("src.common.run_registered_model.validate_registered_spec"), patch(
            "src.common.run_registered_model.set_reproducible_seed", return_value={},
        ), patch("src.common.run_registered_model.load_target_free_setup_entrypoint") as setup:
            result = execute_registered_model(
                self.spec("TSPulse", ratio=100), normal_training=object(),
                test_sessions=(numpy.zeros((20, 2)),), device="cpu",
                entrypoint=lambda values, **arguments: {"scores": values[:, 0]},
            )
        setup.assert_not_called()
        self.assertEqual(result["calibration_scope"], "full_evaluation")

    def test_pca_uses_evaluation_fit_and_keeps_its_checkpoint(self):
        evaluation = numpy.arange(240, dtype=float).reshape(120, 2)

        def scorer(values, **arguments):
            numpy.testing.assert_array_equal(values, evaluation)
            self.assertTrue(arguments["zero_pruning"])
            return {"scores": values[:, 0], "checkpoint": {"fit_rows": len(values)},
                    "fit_source": "full_evaluation"}

        with patch("src.common.run_registered_model.validate_registered_spec"), patch(
            "src.common.run_registered_model.set_reproducible_seed", return_value={},
        ):
            result = execute_registered_model(self.spec("PCA_LEGACY", ratio=100),
                normal_training=object(), test_sessions=(evaluation,), device="cpu", entrypoint=scorer)
        self.assertIsNone(result["split"])
        self.assertEqual(result["checkpoint"]["sessions"], [{"fit_rows": 120}])
        self.assertNotIn("checkpoint", result["test_outputs"][0])

    def test_native_gdn_score_is_saved_without_second_calibration_or_smoothing(self):
        normal = numpy.zeros((100, 2))
        _, _, split = split_ratio_prefix(normal, 40, full_prefix=True)
        duration = {"observed_duration_seconds": None, "duration_basis": "unavailable"}
        evidence = build_execution_evidence(split, {field: 0.0 for field in (
            "split_preprocess_seconds", "model_setup_seconds", "training_seconds",
            "validation_inference_seconds", "calibration_inference_seconds", "test_inference_seconds")},
            spec={"dataset_role": "development", "target_use": "fit_full_prefix", "ratio": 40},
            measurement_protocol_id=FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, retry_count=0,
            training_session_durations=[duration], test_input_sessions=(numpy.zeros((10, 2)),),
            test_session_durations=[duration], peak_memory_mb=0.0, model_artifact_bytes=0)
        output = {"scores": numpy.array([0., 0., 0., 0.7, 1.2]), "source_start": 5,
                  "source_end_exclusive": 10, "alignment": "next_step", "primitive": "official_gdn",
                  "calibration_mode": "official_full_evaluation", "normalization_scope": "full_evaluation",
                  "evaluation_mode": "offline_noncausal", "lookahead": 4, "maximum_effective_lookahead": 4,
                  "native_postprocessing": True, "official_protocol": "paper_tuning_v4",
                  "score_normalization_source_start": 5, "score_normalization_source_end_exclusive": 10,
                  "native_smoothing_window": 4}
        output["native_channel_scores"] = numpy.column_stack((output["scores"], numpy.zeros(5)))
        output["native_calibration"] = {
            "source": "full_evaluation", "source_start": 5, "source_end_exclusive": 10,
            "score_space": "absolute_error", "median": [0.5, 0.0], "iqr": [0.2, 0.0],
            "epsilon": 0.01,
        }
        arguments = dict(dataset="DEV18", series=1, model="GDN", target_use="fit_full_prefix",
            tier="t2", ratio=40, seed=2, config_id=self.spec("GDN")["config_id"],
            common_recipe=self.registry["common_recipe"], common_recipe_id=self.registry["common_recipe_id"],
            normalization_scope="full_evaluation", config_registry_sha256=self.registry_sha,
            execution_identity={"dataset_role": "development", "split_role": "dev18_selection",
                                "input_manifest_sha256": "a" * 64, "final_policy_membership_sha256": None},
            execution_evidence=evidence)
        with tempfile.TemporaryDirectory() as directory:
            saved = save_model_score(output, directory, **arguments)
            self.assertEqual(len(saved["score_paths"]), 4)
            for path in saved["score_paths"]:
                expected = output["native_channel_scores"] if "__channels" in path else output["scores"]
                numpy.testing.assert_array_equal(numpy.load(path), expected)
            metadata = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["score_shape"], [5])
            self.assertEqual(metadata["channel_count"], 0)
            self.assertEqual(metadata["native_channel_score_shape"], [5, 2])
            self.assertEqual(metadata["native_calibration"], output["native_calibration"])
            self.assertIsNone(metadata["calibration_reference"])
            self.assertEqual(metadata["smoothing"]["kind"], "model_native")
            self.assertEqual(metadata["score_normalization_source_end_exclusive"], 10)
            self.assertEqual(metadata["native_smoothing_window"], 4)
            with self.assertRaisesRegex(ValueError, "외부 교정"):
                save_model_score(output, directory, calibration_scores=(numpy.ones(5),), **arguments)
            for change in (
                {"native_channel_scores": None},
                {"native_channel_scores": numpy.ones((5, 2))},
                {"native_calibration": {**output["native_calibration"], "epsilon": 0.02}},
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    save_model_score({**output, **change}, directory, **arguments)


if __name__ == "__main__":
    unittest.main()
