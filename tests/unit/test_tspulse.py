"""Strict TSPulse raw-head adapter contract tests."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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
    score_tspulse_paper,
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
    def __init__(self, model, *, mode, aggregation_length, **settings):
        self.model = model
        self.mode = mode
        self.aggregation_length = aggregation_length
        self.settings = settings


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
    def test_paper_loader_uses_benchmark_score_calibration_parameters(self):
        _, utility = load_tspulse_components(
            aggregation_window=96, channel_count=2, official_protocol=True,
            model_class=RecordingModel, utility_class=RecordingUtility,
            hub_download=lambda **arguments: str(Path("snapshot") / arguments["filename"]),
            file_verifier=lambda path, **arguments: arguments["expected_sha256"],
        )
        self.assertEqual(utility.settings, {
            "least_significant_scale": 0.0, "least_significant_score": 1.0,
        })

    def test_paper_heads_match_official_postprocess_and_use_full_input_statistics(self):
        import pandas
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from tsfm_public.models.tspulse.utils.ad_helpers import TSPulseADUtility
        from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
            TimeSeriesAnomalyDetectionPipeline, score_smoothing,
        )

        values = numpy.column_stack((numpy.arange(20, dtype=float) ** 2, numpy.full(20, 7.0)))
        original = values.copy()
        input_scaler = StandardScaler().fit(values)
        normalized = input_scaler.transform(values)
        raw_heads = {
            "time": numpy.array([0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
            "fft": numpy.array([0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0], dtype=float),
            "pred": numpy.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float),
        }
        utility = TSPulseADUtility(
            SimpleNamespace(config=SimpleNamespace(context_length=8, patch_length=2)),
            mode=["time", "fft", "forecast"], aggregation_length=4,
            least_significant_scale=0.0, least_significant_score=1.0,
        )

        def raw_head_function(input_values, *, aggregation_window):
            self.assertEqual(aggregation_window, 4)
            numpy.testing.assert_allclose(input_values, normalized)
            self.assertNotAlmostEqual(input_values[:8, 0].mean(), 0)
            return raw_heads

        outputs = score_tspulse_paper(
            values, raw_head_function=raw_head_function, utility=utility,
            aggregation_window=4, context_length=8,
        )
        self.assertEqual(set(outputs), {"time", "fft", "pred", "ensemble"})
        for head in outputs:
            official_raw = {
                "forecast" if name == "pred" else name: scores
                for name, scores in raw_heads.items() if head == "ensemble" or name == head
            }
            frame = pandas.DataFrame(normalized, columns=["first", "constant"])
            pipeline = SimpleNamespace(
                _TimeSeriesAnomalyDetectionPipeline__context_memory={"data": frame, "reference": frame},
                _model_processor=utility, aggr_function=numpy.max,
            )
            expected = TimeSeriesAnomalyDetectionPipeline.postprocess(
                pipeline, official_raw, target_columns=list(frame.columns), smoothing_length=8,
            )["anomaly_score"].to_numpy()
            maximum = float(numpy.nanmax(expected))
            expected /= maximum + 1e-5
            output_scaler = MinMaxScaler().fit(expected.reshape(-1, 1))
            expected = output_scaler.transform(expected.reshape(-1, 1)).ravel()
            numpy.testing.assert_allclose(outputs[head]["scores"], expected)
            self.assertTrue(outputs[head]["native_postprocessing"])
            self.assertEqual(outputs[head]["input_normalization_scope"], "full_evaluation")
            calibration = outputs[head]["native_calibration"]
            self.assertEqual(calibration["score_head"], head)
            self.assertEqual(calibration["source_end_exclusive"], len(values))
            self.assertEqual(calibration["input_standard_scaler"], {
                "mean": input_scaler.mean_.tolist(), "variance": input_scaler.var_.tolist(),
                "scale": input_scaler.scale_.tolist(), "sample_count": len(values),
            })
            self.assertEqual(set(calibration["head_minmax"]),
                             set(raw_heads) if head == "ensemble" else {head})
            for name, state in calibration["head_minmax"].items():
                self.assertEqual(state, {
                    "data_min": float(raw_heads[name].min()), "data_max": float(raw_heads[name].max()),
                    "data_range": float(numpy.ptp(raw_heads[name])), "sample_count": len(values),
                    "transform": "sklearn.preprocessing.MinMaxScaler()",
                    "score_exponent": 1.0, "least_significant_scale": 0.0, "least_significant_score": 1.0,
                })
            self.assertEqual(calibration["output_maximum"], maximum)
            self.assertEqual(calibration["output_maximum_epsilon"], 1e-5)
            self.assertEqual(calibration["output_maximum_divisor"], maximum + 1e-5)
            self.assertEqual(calibration["output_minmax_scaler"], {
                "data_min": output_scaler.data_min_.tolist(), "data_max": output_scaler.data_max_.tolist(),
                "data_range": output_scaler.data_range_.tolist(), "scale": output_scaler.scale_.tolist(),
                "offset": output_scaler.min_.tolist(), "sample_count": len(values),
            })
            input_state = calibration["input_standard_scaler"]
            numpy.testing.assert_allclose((values - input_state["mean"]) / input_state["scale"], normalized)
            restored_heads = []
            for name, state in calibration["head_minmax"].items():
                boundary_scaler = MinMaxScaler().fit(numpy.array([state["data_min"], state["data_max"]]).reshape(-1, 1))
                restored = boundary_scaler.transform(align_tspulse_scores(raw_heads[name], head=name,
                    context_length=8, aggregation_window=4).reshape(-1, 1)).ravel()
                if name != "pred":
                    restored = score_smoothing(restored.reshape(-1, 1), smoothing_window_size=8).reshape(-1)
                restored_heads.append(restored)
            restored = numpy.maximum.reduce(restored_heads) / calibration["output_maximum_divisor"]
            state = calibration["output_minmax_scaler"]
            restored = restored * state["scale"][0] + state["offset"][0]
            numpy.testing.assert_allclose(restored, outputs[head]["scores"])
        self.assertEqual(numpy.count_nonzero(outputs["pred"]["scores"]), 1)
        numpy.testing.assert_array_equal(values, original)
        from src.common.execution_evidence import FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, build_execution_evidence
        from src.common.model_registry import load_model_registry_with_sha
        from src.common.save_model_artifacts import save_model_score

        registry, registry_sha = load_model_registry_with_sha()
        duration = {"observed_duration_seconds": None, "duration_basis": "unavailable"}
        evidence = build_execution_evidence(None, {field: 0.0 for field in (
            "split_preprocess_seconds", "model_setup_seconds", "training_seconds",
            "calibration_inference_seconds", "test_inference_seconds")},
            spec={"dataset_role": "development", "target_use": "strict_zero_shot", "ratio": 100},
            measurement_protocol_id=FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, retry_count=0,
            training_session_durations=[], test_input_sessions=(values,), test_session_durations=[duration],
            peak_memory_mb=0.0, model_artifact_bytes=0)
        with tempfile.TemporaryDirectory() as directory:
            for head, output in outputs.items():
                saved = save_model_score(output, Path(directory) / head,
                    dataset="DEV18", series=1, model="TSPulse", target_use="strict_zero_shot",
                    tier="t3", ratio=100, seed=0, config_id=registry["models"]["TSPulse"]["candidates"][0]["config_id"],
                    common_recipe=registry["common_recipe"], common_recipe_id=registry["common_recipe_id"],
                    normalization_scope="full_evaluation", score_variant=head,
                    config_registry_sha256=registry_sha, execution_evidence=evidence,
                    execution_identity={"dataset_role": "development", "split_role": "dev18_selection",
                                        "input_manifest_sha256": "a" * 64, "final_policy_membership_sha256": None})
                metadata = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))
                self.assertEqual(metadata["native_calibration"], output["native_calibration"])
                for path in saved["score_paths"]:
                    numpy.testing.assert_array_equal(numpy.load(path), output["scores"])

    def test_paper_calibration_reconstructs_constant_and_near_constant_head_minmax(self):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from tsfm_public.models.tspulse.utils.ad_helpers import TSPulseADUtility

        values = numpy.arange(40, dtype=float).reshape(20, 2)
        raw_heads = {"time": numpy.resize([1.0, numpy.nextafter(1.0, 2.0)], 12),
                     "fft": numpy.full(12, 3.0), "pred": numpy.arange(12, dtype=float)}
        utility = TSPulseADUtility(
            SimpleNamespace(config=SimpleNamespace(context_length=8, patch_length=2)),
            mode=["time", "fft", "forecast"], aggregation_length=4,
            least_significant_scale=0.0, least_significant_score=1.0,
        )
        outputs = score_tspulse_paper(values, raw_head_function=lambda *args, **kwargs: raw_heads,
            utility=utility, aggregation_window=4, context_length=8)
        states = outputs["ensemble"]["native_calibration"]["head_minmax"]
        for head, state in states.items():
            scaler = MinMaxScaler().fit(numpy.array([state["data_min"], state["data_max"]]).reshape(-1, 1))
            restored = scaler.transform(align_tspulse_scores(raw_heads[head], head=head,
                context_length=8, aggregation_window=4).reshape(-1, 1))
            expected = utility.adjust_boundary("forecast" if head == "pred" else head, raw_heads[head],
                reference=StandardScaler().fit_transform(values))
            numpy.testing.assert_array_equal(restored, expected)
            if head in {"time", "fft"}:
                self.assertEqual(scaler.scale_.tolist(), [1.0])
        self.assertGreater(states["time"]["data_range"], 0.0)
        self.assertEqual(states["fft"]["data_range"], 0.0)

    def test_paper_builder_forwards_protocol_once_and_rejects_context_normalization(self):
        loader_arguments = []
        utility = RawScoreUtility()

        def component_loader(**arguments):
            loader_arguments.append(arguments)
            return object(), utility

        with patch("src.models.tier3.tspulse.score_tspulse_paper", return_value={"ensemble": {}}) as paper_score:
            scorer = build_tspulse_official_scorer(
                aggregation_window=96, channel_count=2, official_protocol=True,
                component_loader=component_loader,
            )
            for _ in range(2):
                scorer(numpy.ones((520, 2)))
        self.assertEqual(len(loader_arguments), 1)
        self.assertTrue(loader_arguments[0]["official_protocol"])
        self.assertEqual(paper_score.call_count, 2)
        with self.assertRaisesRegex(ValueError, "per-context"):
            build_tspulse_official_scorer(
                aggregation_window=96, channel_count=2, official_protocol=True,
                inference_context_normalization=True, component_loader=component_loader,
            )
        self.assertEqual(len(loader_arguments), 1)

    def test_paper_short_session_has_full_ensemble_without_common_native_range(self):
        from tsfm_public.models.tspulse.utils.ad_helpers import TSPulseADUtility

        utility = TSPulseADUtility(
            SimpleNamespace(config=SimpleNamespace(context_length=8, patch_length=2)),
            mode=["time", "fft", "forecast"], aggregation_length=4,
            least_significant_scale=0.0, least_significant_score=1.0,
        )
        result = score_tspulse_paper(
            numpy.arange(18, dtype=float).reshape(9, 2),
            raw_head_function=lambda values, **arguments: {
                head: numpy.array([1.0]) for head in ("time", "fft", "pred")
            },
            utility=utility, aggregation_window=4, context_length=8,
        )
        ensemble = result["ensemble"]
        self.assertEqual(ensemble["scores"].shape, (9,))
        self.assertLess(ensemble["native_source_start"], ensemble["native_source_end_exclusive"])
        for output in result.values():
            self.assertEqual(output["native_calibration"]["output_minmax_scaler"]["data_range"], [0.0])
            self.assertEqual(output["native_calibration"]["output_minmax_scaler"]["scale"], [1.0])

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

    def test_context_normalization_uses_only_each_past_window_for_future_and_metadata(self):
        values = numpy.array([
            [10, 100], [12, 100], [14, 100], [16, 100],
            [1000, 130], [1200, 160], [1400, 190],
        ], dtype=numpy.float32)
        original = values.copy()
        utility = RawScoreUtility()
        raw_head_function = build_tspulse_raw_head_function(
            utility, aggregation_window=2, context_length=4, batch_size=2,
            inference_context_normalization=True,
        )
        outputs = score_tspulse(
            values, raw_head_function=raw_head_function,
            aggregation_window=2, context_length=4,
        )

        first_payload = utility.payloads[0][0]
        numpy.testing.assert_allclose(
            first_payload["past_values"][0, :, 0], numpy.array([-3, -1, 1, 3]) / numpy.sqrt(5),
            rtol=1e-6,
        )
        numpy.testing.assert_array_equal(first_payload["past_values"][0, :, 1], 0)
        numpy.testing.assert_allclose(
            first_payload["future_values"][0, 0], [987 / numpy.sqrt(5), 30], rtol=1e-6,
        )
        for output in outputs.values():
            self.assertEqual(output["normalization_scope"], "none")
            self.assertEqual(output["input_normalization"], "inference_context_zscore")
            self.assertEqual(output["input_normalization_scope"], "past_context")
            self.assertFalse(output["persistent_target_fit"])
        numpy.testing.assert_array_equal(values, original)

        other_utility = RawScoreUtility()
        other_function = build_tspulse_raw_head_function(
            other_utility, aggregation_window=2, context_length=4, batch_size=1,
            inference_context_normalization=True,
        )
        other_outputs = score_tspulse(
            values, raw_head_function=other_function, aggregation_window=2, context_length=4,
        )
        for head in outputs:
            numpy.testing.assert_array_equal(outputs[head]["scores"], other_outputs[head]["scores"])
        changed = values.copy()
        changed[4:] *= 1000
        other_utility.payloads.clear()
        other_function(changed, aggregation_window=2)
        torch.testing.assert_close(
            first_payload["past_values"][0], other_utility.payloads[0][0]["past_values"][0],
            rtol=0, atol=0,
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

    def test_normalized_prediction_full_range_is_independent_of_aggregation_window(self):
        timeline = numpy.arange(1536, dtype=numpy.float32)
        values = numpy.column_stack((timeline ** 2, 3 * timeline + 7))
        predictions = []
        metadata = []
        for aggregation_window in (64, 96, 128):
            result = score_tspulse_official(
                values, aggregation_window=aggregation_window, batch_size=8,
                inference_context_normalization=True,
                component_loader=lambda **arguments: (object(), RawScoreUtility()),
            )
            prediction = result["pred"]
            predictions.append(prediction["scores"])
            metadata.append({key: value for key, value in prediction.items() if key != "scores"})
        for prediction, details in zip(predictions, metadata):
            numpy.testing.assert_array_equal(prediction, predictions[0])
            self.assertEqual(len(prediction), len(values))
            self.assertEqual(details, metadata[0])
            self.assertEqual(details["native_source_start"], 512)
            self.assertEqual(details["native_source_end_exclusive"], len(values))
            self.assertEqual(details["boundary_repeat"], {"left": 512, "right": 0})

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
            inference_context_normalization=True,
        )

        self.assertEqual(set(result), {"time", "fft", "pred", "raw_max"})
        self.assertTrue(all(output["scores"].shape == (12,) for output in result.values()))
        self.assertTrue(all(output["input_normalization_scope"] == "past_context" for output in result.values()))

    def test_pins_and_verifies_official_source_and_checkpoint(self):
        self.assertEqual(TSPULSE_SOURCE_COMMIT, "fe7a35697723e2a2f5246ae979474bfc554e26c0")
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
