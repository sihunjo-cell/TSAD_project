"""Strict TimeRCD adapter contract tests."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy
import torch

from src.models.tier3.time_rcd import (
    TIME_RCD_CONFIG_FILE,
    TIME_RCD_CONFIG_SHA256,
    TIME_RCD_CHECKPOINT_SHA256,
    build_time_rcd_config,
    get_time_rcd_status,
    load_time_rcd_components,
    load_time_rcd_model,
    score_time_rcd_official,
    score_time_rcd,
    verify_time_rcd_checkpoint,
)


class ConstantProbabilityModel:
    def __init__(self):
        self.calls = []
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, *, time_series, mask):
        self.calls.append((time_series.detach().cpu(), mask.detach().cpu()))
        batch_size, time_count, channel_count = time_series.shape
        logits = torch.zeros(
            batch_size, time_count, channel_count, 2,
            dtype=time_series.dtype, device=time_series.device,
        )
        logits[..., 1] = math.log(3.0)
        return logits


class RecordingTester:
    latest = None

    def __init__(self, checkpoint_path, config):
        type(self).latest = self
        self.arguments = checkpoint_path, config
        self.device = None
        self.model = ConstantProbabilityModel()


class UnequalChannelLogitModel(ConstantProbabilityModel):
    def __call__(self, *, time_series, mask):
        batch_size, time_count, channel_count = time_series.shape
        self.calls.append((time_series.detach().cpu(), mask.detach().cpu()))
        logits = torch.zeros(
            batch_size, time_count, channel_count, 2,
            dtype=time_series.dtype, device=time_series.device,
        )
        logits[..., 1, 1] = 4.0
        return logits


class TestTimeRCD(unittest.TestCase):
    def test_builds_official_multi_config_without_mutating_default(self):
        default = SimpleNamespace(
            win_size=12,
            batch_size=9,
            ts_config=SimpleNamespace(patch_size=3, num_features=1),
        )

        config = build_time_rcd_config(default, channel_count=7)

        self.assertEqual(config.ts_config.patch_size, 16)
        self.assertEqual(config.win_size, 5000)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.ts_config.num_features, 7)
        self.assertEqual(default.ts_config.patch_size, 3)
        self.assertIsNot(config, default)
        self.assertIsNot(config.ts_config, default.ts_config)

    def test_short_session_is_not_padded(self):
        model = ConstantProbabilityModel()
        session = numpy.arange(6, dtype=numpy.float32).reshape(3, 2)

        score_time_rcd(model, session, context_length=4, device="cpu")

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0].shape, (1, 3, 2))
        self.assertEqual(model.calls[0][1][0].tolist(), [True, True, True])

    def test_scores_class_one_probability_and_trims_last_value_padding(self):
        model = ConstantProbabilityModel()
        session = numpy.arange(14, dtype=numpy.float32).reshape(7, 2)

        result = score_time_rcd(model, session, context_length=4, device="cpu")

        numpy.testing.assert_allclose(result["scores"], numpy.full(7, 0.75))
        self.assertEqual(result["scores"].shape, (7,))
        self.assertTrue(model.eval_called)
        self.assertEqual(len(model.calls), 2)
        numpy.testing.assert_array_equal(model.calls[1][0][0, :3], session[4:])
        numpy.testing.assert_array_equal(model.calls[1][0][0, 3], session[-1])
        self.assertEqual(model.calls[1][1][0].tolist(), [True, True, True, False])

    def test_keeps_contexts_separate_and_does_not_normalize_or_mutate_input(self):
        model = ConstantProbabilityModel()
        session = numpy.array(
            [[10.0], [20.0], [30.0], [40.0], [1000.0], [2000.0]],
            dtype=numpy.float32,
        )
        original = session.copy()

        score_time_rcd(model, session, context_length=4, device="cpu")

        numpy.testing.assert_array_equal(session, original)
        self.assertEqual([tuple(values.shape) for values, _ in model.calls], [(1, 4, 1), (1, 4, 1)])
        numpy.testing.assert_array_equal(model.calls[0][0][0, :, 0], original[:4, 0])
        numpy.testing.assert_array_equal(
            model.calls[1][0][0, :, 0], [1000.0, 2000.0, 2000.0, 2000.0],
        )

    def test_metadata_declares_probability_without_target_normalization(self):
        result = score_time_rcd(
            ConstantProbabilityModel(), numpy.ones((3, 2), dtype=numpy.float32),
            context_length=4, device="cpu",
        )

        self.assertEqual(result["source_start"], 0)
        self.assertEqual(result["source_end_exclusive"], 3)
        self.assertEqual(result["alignment"], "same_timestep")
        self.assertEqual(result["primitive"], "probability")
        self.assertEqual(result["calibration_mode"], "none")
        self.assertEqual(result["normalization_scope"], "none")
        self.assertEqual(result["evaluation_mode"], "offline_noncausal")
        self.assertEqual(result["lookahead"], 2)
        self.assertEqual(result["maximum_effective_lookahead"], 2)

    def test_averages_channel_logits_before_softmax_like_official_inference(self):
        result = score_time_rcd(
            UnequalChannelLogitModel(), numpy.ones((2, 2), dtype=numpy.float32),
            context_length=2, device="cpu",
        )

        numpy.testing.assert_allclose(
            result["scores"], numpy.full(2, 1.0 / (1.0 + math.exp(-2.0))),
        )

    def test_checkpoint_verifier_and_loader_do_not_call_official_zero_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.write_bytes(b"strict-time-rcd")
            digest = "1c6236748216844d1d0f3e1682c088015de261977c5feb6ac53347b03445a063"
            self.assertEqual(
                verify_time_rcd_checkpoint(checkpoint, expected_sha256=digest), digest,
            )
            with self.assertRaises(ValueError):
                verify_time_rcd_checkpoint(checkpoint)

            model = load_time_rcd_model(
                checkpoint, config={"win_size": 5000}, tester_class=RecordingTester,
                expected_sha256=digest,
            )

        self.assertIsInstance(model, ConstantProbabilityModel)

    def test_loader_downloads_and_verifies_both_sealed_files_on_requested_device(self):
        default = SimpleNamespace(
            win_size=1, batch_size=2,
            ts_config=SimpleNamespace(patch_size=3, num_features=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            downloads = []
            verifications = []

            def hub_download(*, repo_id, filename, revision):
                downloads.append((repo_id, filename, revision))
                path = Path(directory) / Path(filename).name
                path.write_bytes(filename.encode())
                return str(path)

            def file_verifier(path, *, expected_sha256):
                verifications.append((Path(path).name, expected_sha256))
                return expected_sha256

            tester, model = load_time_rcd_components(
                channel_count=3, device="cpu", hub_download=hub_download,
                file_verifier=file_verifier, tester_class=RecordingTester,
                default_config=default,
            )

        self.assertEqual([item[1] for item in downloads], [
            "best_model/pretrain_checkpoint_best_multi.pth", TIME_RCD_CONFIG_FILE,
        ])
        self.assertEqual(verifications, [
            ("pretrain_checkpoint_best_multi.pth", TIME_RCD_CHECKPOINT_SHA256),
            (TIME_RCD_CONFIG_FILE, TIME_RCD_CONFIG_SHA256),
        ])
        self.assertEqual(tester.device, torch.device("cpu"))
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertTrue(model.eval_called)

    def test_official_entrypoint_connects_loader_and_score(self):
        session = numpy.arange(6, dtype=numpy.float32).reshape(3, 2)
        model = ConstantProbabilityModel()

        def component_loader(**arguments):
            self.assertEqual(arguments, {"channel_count": 2, "device": "cpu"})
            return object(), model

        result = score_time_rcd_official(
            session, context_length=4, device="cpu",
            component_loader=component_loader,
        )

        self.assertEqual(result["scores"].shape, (3,))
        self.assertEqual(model.calls[0][0].shape, (1, 3, 2))

    def test_source_only_smoke_cannot_mark_checkpoint_path_available(self):
        self.assertEqual(len(TIME_RCD_CHECKPOINT_SHA256), 64)
        self.assertEqual(get_time_rcd_status(source_smoke_passed=False)["status"], "unavailable")
        self.assertEqual(get_time_rcd_status(source_smoke_passed=True)["status"], "unavailable")

    def test_checkpoint_status_requires_finite_nonconstant_score_evidence(self):
        complete_evidence = {
            "checkpoint_smoke_passed": True,
            "finite_scores_verified": True,
            "nonconstant_scores_verified": True,
        }
        try:
            available = get_time_rcd_status(**complete_evidence)
        except TypeError as error:
            self.fail(f"checkpoint evidence contract is missing: {error}")
        self.assertEqual(available["status"], "available")
        for missing in complete_evidence:
            with self.subTest(missing=missing):
                evidence = {**complete_evidence, missing: False}
                self.assertEqual(
                    get_time_rcd_status(**evidence)["status"], "unavailable",
                )

    def test_rejects_labels_and_invalid_sessions_at_the_api_boundary(self):
        with self.assertRaises(TypeError):
            score_time_rcd(
                ConstantProbabilityModel(), numpy.ones((3, 2)), labels=numpy.zeros(3),
            )
        for invalid in (
            numpy.ones(3), numpy.empty((0, 2)), numpy.array([[numpy.nan, 1.0]]),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    score_time_rcd(ConstantProbabilityModel(), invalid, device="cpu")


if __name__ == "__main__":
    unittest.main()
