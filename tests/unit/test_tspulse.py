"""Strict TSPulse raw-head adapter contract tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy
import torch

from src.models.tier3.tspulse import (
    TSPULSE_CONFIG_FILE,
    TSPULSE_CONFIG_SHA256,
    TSPULSE_CHECKPOINT_REVISION,
    TSPULSE_CHECKPOINT_SHA256,
    TSPULSE_SOURCE_COMMIT,
    align_tspulse_scores,
    build_tspulse_official_scorer,
    build_tspulse_raw_head_function,
    load_tspulse_components,
    score_tspulse_official,
    score_tspulse,
    verify_tspulse_checkpoint,
)


class RecordingModel:
    loaded = None

    @classmethod
    def from_pretrained(cls, model_name, **arguments):
        cls.loaded = model_name, arguments
        return cls()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class RecordingUtility:
    def __init__(self, model, *, mode, aggregation_length):
        self.model = model
        self.mode = mode
        self.aggregation_length = aggregation_length


class RawScoreUtility:
    def __init__(self):
        self.payloads = []

    def compute_score(self, payload, *, mode, expand_score):
        self.payloads.append((
            {name: value.detach().cpu().clone() for name, value in payload.items()},
            list(mode), expand_score,
        ))
        past = payload["past_values"]
        future = payload["future_values"]
        return {
            "time": past[:, -1].mean(dim=1),
            "fft": past[:, -2].mean(dim=1),
            "forecast": future[:, 0].mean(dim=1),
        }


class TestTSPulse(unittest.TestCase):
    def test_aligns_official_raw_head_boundaries(self):
        raw = numpy.array([1.0, 2.0, 3.0, 4.0])

        time_score = align_tspulse_scores(
            raw, head="time", context_length=8, aggregation_window=4,
        )
        fft_score = align_tspulse_scores(
            raw, head="fft", context_length=8, aggregation_window=4,
        )
        prediction_score = align_tspulse_scores(
            raw, head="pred", context_length=8, aggregation_window=4,
        )

        expected_reconstruction = numpy.array([1.0] * 6 + [1.0, 2.0, 3.0, 4.0] + [4.0] * 2)
        expected_prediction = numpy.array([1.0] * 8 + [1.0, 2.0, 3.0, 4.0])
        numpy.testing.assert_array_equal(time_score, expected_reconstruction)
        numpy.testing.assert_array_equal(fft_score, expected_reconstruction)
        numpy.testing.assert_array_equal(prediction_score, expected_prediction)

    def test_returns_all_raw_heads_and_elementwise_raw_max(self):
        session = numpy.arange(24, dtype=numpy.float32).reshape(12, 2)
        observed = []

        def raw_head_function(values, *, aggregation_window):
            observed.append(values.copy())
            self.assertEqual(aggregation_window, 4)
            return {
                "time": numpy.array([1.0, 2.0, 3.0, 4.0]),
                "fft": numpy.array([4.0, 3.0, 2.0, 1.0]),
                "pred": numpy.array([2.0, 2.0, 2.0, 2.0]),
            }

        result = score_tspulse(
            session, raw_head_function=raw_head_function,
            aggregation_window=4, context_length=8,
        )

        self.assertEqual(set(result), {"time", "fft", "pred", "raw_max"})
        for head in result:
            self.assertEqual(result[head]["scores"].shape, (12,))
            self.assertEqual(result[head]["normalization_scope"], "model_revin_only")
            self.assertEqual(result[head]["calibration_mode"], "none")
        expected_raw_max = numpy.pad(
            numpy.array([3.0, 4.0]), (8, 2), mode="edge",
        )
        numpy.testing.assert_array_equal(result["raw_max"]["scores"], expected_raw_max)
        fully_aligned_max = numpy.maximum.reduce([
            result["time"]["scores"], result["fft"]["scores"],
            result["pred"]["scores"],
        ])
        self.assertFalse(numpy.array_equal(expected_raw_max, fully_aligned_max))
        numpy.testing.assert_array_equal(observed[0], session)

    def test_raw_max_requires_a_nonempty_common_native_range(self):
        session = numpy.arange(20, dtype=numpy.float32).reshape(10, 2)

        def raw_head_function(values, *, aggregation_window):
            del aggregation_window
            raw = numpy.arange(len(values) - 8, dtype=numpy.float64)
            return {head: raw for head in ("time", "fft", "pred")}

        with self.assertRaisesRegex(ValueError, "common native range"):
            score_tspulse(
                session, raw_head_function=raw_head_function,
                aggregation_window=4, context_length=8,
            )

    def test_records_native_ranges_for_each_full_length_head(self):
        session = numpy.arange(24, dtype=numpy.float32).reshape(12, 2)

        def raw_head_function(values, *, aggregation_window):
            del aggregation_window
            return {
                head: numpy.arange(len(values) - 8, dtype=numpy.float64)
                for head in ("time", "fft", "pred")
            }

        result = score_tspulse(
            session, raw_head_function=raw_head_function,
            aggregation_window=4, context_length=8,
        )

        self.assertEqual(result["pred"]["native_source_start"], 8)
        self.assertEqual(result["pred"]["native_source_end_exclusive"], 12)
        self.assertEqual(result["pred"]["boundary_repeat"], {"left": 8, "right": 0})
        self.assertEqual(result["pred"]["lookahead"], 0)
        self.assertEqual(result["pred"]["maximum_effective_lookahead"], 8)
        for head in ("time", "fft"):
            self.assertEqual(result[head]["native_source_start"], 6)
            self.assertEqual(result[head]["native_source_end_exclusive"], 10)
            self.assertEqual(result[head]["boundary_repeat"], {"left": 6, "right": 2})
            self.assertEqual(result[head]["lookahead"], 2)
            self.assertEqual(result[head]["maximum_effective_lookahead"], 7)
        self.assertEqual(result["raw_max"]["native_source_start"], 8)
        self.assertEqual(result["raw_max"]["native_source_end_exclusive"], 10)
        self.assertEqual(result["raw_max"]["boundary_repeat"], {"left": 8, "right": 2})
        self.assertEqual(result["raw_max"]["lookahead"], 2)
        self.assertEqual(result["raw_max"]["maximum_effective_lookahead"], 9)
        self.assertTrue(all(
            output["alignment"] == "boundary_repeat_from_native"
            for output in result.values()
        ))
        self.assertTrue(all(
            output["evaluation_mode"] == "offline_noncausal"
            for output in result.values()
        ))

    def test_never_passes_labels_or_applies_target_preprocessing(self):
        session = numpy.array(
            [[10.0], [20.0], [30.0], [40.0], [50.0], [60.0], [70.0], [80.0],
             [900.0], [1000.0], [1100.0], [1200.0]],
            dtype=numpy.float32,
        )
        original = session.copy()

        def raw_head_function(values, *, aggregation_window):
            numpy.testing.assert_array_equal(values, original)
            return {head: numpy.arange(4, dtype=numpy.float64) for head in ("time", "fft", "pred")}

        score_tspulse(
            session, raw_head_function=raw_head_function,
            aggregation_window=4, context_length=8,
        )
        numpy.testing.assert_array_equal(session, original)
        with self.assertRaises(TypeError):
            score_tspulse(
                session, raw_head_function=raw_head_function,
                aggregation_window=4, context_length=8, labels=numpy.zeros(12),
            )

    def test_loads_pinned_model_and_raw_utility_lazily(self):
        with tempfile.TemporaryDirectory() as directory:
            downloads = []
            verifications = []
            events = []
            snapshot_directory = Path(directory) / "snapshots" / TSPULSE_CHECKPOINT_REVISION

            class EventRecordingModel(RecordingModel):
                @classmethod
                def from_pretrained(cls, model_name, **arguments):
                    events.append("from_pretrained")
                    return super().from_pretrained(model_name, **arguments)

            def hub_download(*, repo_id, filename, revision):
                downloads.append((repo_id, filename, revision))
                snapshot_directory.mkdir(parents=True, exist_ok=True)
                path = snapshot_directory / Path(filename).name
                path.write_bytes(filename.encode())
                return str(path)

            def file_verifier(path, *, expected_sha256):
                verifications.append((Path(path).name, expected_sha256))
                events.append(f"verify:{Path(path).name}")
                return expected_sha256

            model, utility = load_tspulse_components(
                aggregation_window=64, channel_count=2, device="cpu",
                model_class=EventRecordingModel, utility_class=RecordingUtility,
                hub_download=hub_download, file_verifier=file_verifier,
            )

        model_source, arguments = EventRecordingModel.loaded
        self.assertEqual(Path(model_source), snapshot_directory)
        self.assertEqual(arguments["revision"], TSPULSE_CHECKPOINT_REVISION)
        self.assertEqual(arguments["num_input_channels"], 2)
        self.assertEqual(arguments["mask_type"], "user")
        self.assertEqual(downloads, [
            ("ibm-granite/granite-timeseries-tspulse-r1", "model.safetensors", TSPULSE_CHECKPOINT_REVISION),
            ("ibm-granite/granite-timeseries-tspulse-r1", TSPULSE_CONFIG_FILE, TSPULSE_CHECKPOINT_REVISION),
        ])
        self.assertEqual(verifications, [
            ("model.safetensors", TSPULSE_CHECKPOINT_SHA256),
            (TSPULSE_CONFIG_FILE, TSPULSE_CONFIG_SHA256),
        ])
        self.assertEqual(events, [
            "verify:model.safetensors", "verify:config.json", "from_pretrained",
        ])
        self.assertEqual(model.device, "cpu")
        self.assertTrue(model.eval_called)
        self.assertIs(utility.model, model)
        self.assertEqual(utility.mode, ["time", "fft", "forecast"])
        self.assertEqual(utility.aggregation_length, 64)

    def test_rejects_verified_files_from_different_snapshot_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def hub_download(*, repo_id, filename, revision):
                del repo_id, revision
                snapshot = root / ("weights" if filename.endswith("safetensors") else "config")
                snapshot.mkdir()
                path = snapshot / filename
                path.write_bytes(filename.encode())
                return str(path)

            with self.assertRaisesRegex(ValueError, "same snapshot directory"):
                load_tspulse_components(
                    aggregation_window=64, channel_count=2,
                    model_class=RecordingModel, utility_class=RecordingUtility,
                    hub_download=hub_download,
                    file_verifier=lambda path, *, expected_sha256: expected_sha256,
                )

    def test_does_not_resolve_verified_snapshot_paths_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshots" / TSPULSE_CHECKPOINT_REVISION
            snapshot.mkdir(parents=True)
            paths = {}
            for filename in ("model.safetensors", TSPULSE_CONFIG_FILE):
                path = snapshot / filename
                path.write_bytes(filename.encode())
                paths[filename] = str(path)

            with patch.object(
                Path, "resolve",
                side_effect=AssertionError("verified hub paths must not be resolved"),
            ):
                load_tspulse_components(
                    aggregation_window=64, channel_count=2,
                    model_class=RecordingModel, utility_class=RecordingUtility,
                    hub_download=lambda *, repo_id, filename, revision: paths[filename],
                    file_verifier=lambda path, *, expected_sha256: expected_sha256,
                )

        model_source, _ = RecordingModel.loaded
        self.assertEqual(Path(model_source), snapshot)

    def test_builds_strict_stride_one_payloads_and_maps_forecast_to_pred(self):
        utility = RawScoreUtility()
        raw_head_function = build_tspulse_raw_head_function(
            utility, aggregation_window=4, context_length=8,
            batch_size=3, device="cpu",
        )
        values = numpy.arange(24, dtype=numpy.float32).reshape(12, 2)

        heads = raw_head_function(values, aggregation_window=4)

        self.assertEqual(set(heads), {"time", "fft", "pred"})
        self.assertTrue(all(head.shape == (4,) for head in heads.values()))
        self.assertEqual(len(utility.payloads), 2)
        first_payload, mode, expand_score = utility.payloads[0]
        self.assertEqual(mode, ["time", "fft", "forecast"])
        self.assertFalse(expand_score)
        self.assertEqual(first_payload["past_values"].shape, (3, 8, 2))
        self.assertEqual(first_payload["future_values"].shape, (3, 1, 2))
        torch.testing.assert_close(
            first_payload["past_values"][0], torch.from_numpy(values[:8]),
        )
        torch.testing.assert_close(
            first_payload["future_values"][0, 0], torch.from_numpy(values[8]),
        )

    def test_raw_head_scores_are_batch_partition_invariant(self):
        values = numpy.arange(1090, dtype=numpy.float32).reshape(545, 2)
        outputs = []
        for batch_size in (1, 32):
            utility = RawScoreUtility()
            raw_head_function = build_tspulse_raw_head_function(
                utility, aggregation_window=4, context_length=8,
                batch_size=batch_size, device="cpu",
            )
            outputs.append(score_tspulse(
                values, raw_head_function=raw_head_function,
                aggregation_window=4, context_length=8,
            ))
        for head in ("time", "fft", "pred", "raw_max"):
            numpy.testing.assert_array_equal(
                outputs[0][head]["scores"], outputs[1][head]["scores"],
            )

    def test_prepared_scorer_loads_components_once(self):
        utility = RawScoreUtility()
        loader_calls = []

        def component_loader(**arguments):
            loader_calls.append(arguments)
            return object(), utility

        scorer = build_tspulse_official_scorer(
            aggregation_window=4, channel_count=2, context_length=8,
            batch_size=3, device="cpu", component_loader=component_loader,
        )
        for _ in range(2):
            result = scorer(numpy.arange(24, dtype=numpy.float32).reshape(12, 2))
            self.assertEqual(set(result), {"time", "fft", "pred", "raw_max"})
        self.assertEqual(len(loader_calls), 1)

    def test_official_entrypoint_connects_loader_raw_utility_and_alignment(self):
        utility = RawScoreUtility()

        def component_loader(**arguments):
            self.assertEqual(arguments["aggregation_window"], 4)
            self.assertEqual(arguments["device"], "cpu")
            self.assertEqual(arguments["channel_count"], 2)
            return object(), utility

        result = score_tspulse_official(
            numpy.arange(24, dtype=numpy.float32).reshape(12, 2),
            aggregation_window=4, context_length=8, batch_size=3,
            device="cpu", component_loader=component_loader,
        )

        self.assertEqual(set(result), {"time", "fft", "pred", "raw_max"})
        self.assertTrue(all(output["scores"].shape == (12,) for output in result.values()))

    def test_pins_and_verifies_official_source_and_checkpoint(self):
        self.assertEqual(TSPULSE_SOURCE_COMMIT, "9739fa59b61bd9f15cbfb06e5dc3dab28c72ee8d")
        self.assertEqual(TSPULSE_CHECKPOINT_REVISION, "2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2")
        self.assertEqual(TSPULSE_CHECKPOINT_SHA256, "57fa03b67d1473a7253ac37801b24c391d1fa931395db986743e2f59e556b9ac")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.write_bytes(b"strict-tspulse")
            digest = "1ebaffbbb14355df45322e6db14f290c960a9fe764f40ae66e7d04ae71444fe6"
            self.assertEqual(
                verify_tspulse_checkpoint(checkpoint, expected_sha256=digest), digest,
            )
            with self.assertRaises(ValueError):
                verify_tspulse_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
