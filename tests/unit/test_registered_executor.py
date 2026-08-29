"""Registry spec가 label-free 공통 실행기로 정확히 전달되는지 검증한다."""

import inspect
import os
import random
import unittest
from copy import deepcopy
from importlib import import_module as real_import_module
from types import SimpleNamespace

import numpy
import torch
from unittest.mock import patch

from src.common.model_registry import load_model_registry
from src.common.run_registered_model import (
    execute_registered_model,
    load_target_free_setup_entrypoint,
)
from src.common.set_reproducible_seed import set_reproducible_seed
from tests.ghl_main.run_registered_models import build_specs


def _score_output(values):
    return {"scores": numpy.arange(len(values), dtype=float)}


class _PoisonTraining:
    def __array__(self, *args, **kwargs):
        raise AssertionError("target-free 실행이 normal_training을 읽었다")

    def __len__(self):
        raise AssertionError("target-free 실행이 normal_training 길이를 읽었다")


class _RecordingAdapter:
    def __init__(self):
        self.fit_calls = []
        self.score_calls = []

    def fit(self, *values):
        self.fit_calls.append(tuple(numpy.array(value, copy=True) for value in values))
        return {"optimizer_updates": 2}

    def score(self, values):
        self.score_calls.append(numpy.array(values, copy=True))
        return _score_output(values)

    def checkpoint(self):
        return {"state": "sealed"}


class TestReproducibleSeed(unittest.TestCase):
    def test_resets_python_numpy_and_torch_streams(self):
        first_state = set_reproducible_seed(7)
        first = (random.random(), numpy.random.random(), torch.rand(3))
        second_state = set_reproducible_seed(7)
        second = (random.random(), numpy.random.random(), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2])
        self.assertEqual(first_state, second_state)
        self.assertEqual(first_state["seed"], 7)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(first_state["deterministic_algorithms"])
        self.assertEqual(first_state["cublas_workspace_config"], ":4096:8")
        if torch.backends.cudnn.is_available():
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(first_state["cudnn_deterministic"])

    def test_fails_if_cuda_was_initialized_before_cublas_was_configured(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ""}), patch(
            "src.common.set_reproducible_seed.torch.cuda.is_initialized",
            return_value=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
                set_reproducible_seed(7)


class TestRegisteredExecutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {}
        for spec in build_specs("development", include_pending=True):
            cls.specs.setdefault(spec["model"], spec)

    def test_target_free_models_never_touch_normal_training(self):
        test = numpy.arange(24, dtype=float).reshape(12, 2)
        for model in ("MWVAR", "SQDIFF_LAST3"):
            calls = []

            def scorer(session, **arguments):
                calls.append((numpy.array(session, copy=True), arguments))
                return _score_output(session)

            with self.subTest(model=model):
                result = execute_registered_model(
                    self.specs[model], normal_training=_PoisonTraining(),
                    test_sessions=(test,), device="cpu", entrypoint=scorer,
                )
                self.assertEqual(len(calls), 1)
                numpy.testing.assert_array_equal(calls[0][0], test)
                self.assertIsNone(result["split"])
                self.assertIsNone(result["scaler_state"])
                self.assertEqual(result["validation_outputs"], ())
                self.assertEqual(result["calibration_scope"], "none")

    def test_tier3_prepares_once_and_times_setup_separately(self):
        sessions = (
            numpy.zeros((12, 2), dtype=float),
            numpy.ones((12, 2), dtype=float),
        )
        for model in ("TimeRCD", "TSPulse"):
            events = []

            def build_scorer(**arguments):
                events.append(("build", arguments))

                def scorer(session):
                    events.append(("score", numpy.array(session, copy=True)))
                    return _score_output(session)

                return scorer

            def load_builder(requested_model):
                events.append(("import", requested_model))
                return build_scorer

            with self.subTest(model=model), patch(
                "src.common.run_registered_model.load_target_free_setup_entrypoint",
                side_effect=load_builder,
            ), patch(
                "src.common.run_registered_model.time.perf_counter",
                side_effect=(0.0, 2.0, 5.0, 9.0),
            ), patch(
                "src.common.run_registered_model._synchronize_cuda",
                side_effect=lambda device: events.append(("sync", device)),
            ):
                result = execute_registered_model(
                    self.specs[model], test_sessions=sessions, device="cpu",
                )

            self.assertEqual(events[0], ("import", model))
            self.assertEqual(events[1][0], "build")
            self.assertEqual(events[1][1]["channel_count"], 2)
            self.assertEqual([event[0] for event in events], [
                "import", "build", "sync", "score", "score", "sync",
            ])
            self.assertEqual(result["timing"]["model_setup_seconds"], 2.0)
            self.assertEqual(result["timing"]["test_inference_seconds"], 4.0)
            self.assertGreaterEqual(result["timing"]["model_setup_seconds"], 0.0)
            self.assertGreaterEqual(result["timing"]["test_inference_seconds"], 0.0)
            self.assertEqual(len(result["test_outputs"]), 2)

    def test_target_free_setup_loader_maps_both_tier3_builders(self):
        time_rcd_builder = lambda **arguments: arguments
        tspulse_builder = lambda **arguments: arguments
        modules = {
            "src.models.tier3.time_rcd": SimpleNamespace(
                build_time_rcd_official_scorer=time_rcd_builder,
            ),
            "src.models.tier3.tspulse": SimpleNamespace(
                build_tspulse_official_scorer=tspulse_builder,
            ),
        }
        with patch(
            "src.common.run_registered_model.importlib.import_module",
            side_effect=modules.__getitem__,
        ) as importer:
            self.assertIs(load_target_free_setup_entrypoint("TimeRCD"), time_rcd_builder)
            self.assertIs(load_target_free_setup_entrypoint("TSPulse"), tspulse_builder)
            self.assertIsNone(load_target_free_setup_entrypoint("MWVAR"))
        self.assertEqual([call.args[0] for call in importer.call_args_list], list(modules))

    def test_tier3_honors_injected_legacy_entrypoint(self):
        sessions = (
            numpy.zeros((12, 2), dtype=float),
            numpy.ones((12, 2), dtype=float),
        )
        for model in ("TimeRCD", "TSPulse"):
            injected_calls = []

            def injected_entrypoint(session, **arguments):
                injected_calls.append((numpy.array(session, copy=True), arguments))
                return _score_output(session)

            with self.subTest(model=model), patch(
                "src.common.run_registered_model.load_target_free_setup_entrypoint",
                side_effect=AssertionError("prepared builder must not load"),
            ):
                result = execute_registered_model(
                    self.specs[model], test_sessions=sessions, device="cpu",
                    entrypoint=injected_entrypoint,
                )
            self.assertEqual(len(injected_calls), 2)
            self.assertEqual(len(result["test_outputs"]), 2)

    def test_rejects_forged_specs_before_seed_data_or_injected_code(self):
        base = self.specs["MWVAR"]
        mutations = {
            "config_registry_sha256": "0" * 64,
            "model": "Unknown",
            "config_id": "c000000000000",
            "hyperparameters": {**base["hyperparameters"], "window": 95},
            "tier": "t9",
            "target_use": "fit_validation",
            "source_commit": "0" * 40,
            "source_checkpoint_sha256": "forged",
            "checkpoint_config_sha256": "forged",
            "checkpoint_revision": "forged",
            "preprocess_recipe": {},
            "common_recipe": {},
            "common_recipe_id": "r000000000000",
        }
        for field, value in mutations.items():
            events = []

            class RecordingSessions:
                def __iter__(self):
                    events.append("test_sessions")
                    return iter((numpy.zeros((100, 2)),))

            def scorer(values):
                events.append("entrypoint")
                return _score_output(values)

            spec = deepcopy(base)
            spec[field] = value
            with self.subTest(field=field), patch(
                "src.common.run_registered_model.set_reproducible_seed",
                side_effect=lambda seed: events.append(("seed", seed)),
            ):
                with self.assertRaisesRegex(ValueError, "registry|spec|model|config"):
                    execute_registered_model(
                        spec,
                        normal_training=_PoisonTraining(),
                        test_sessions=RecordingSessions(),
                        device="cpu",
                        entrypoint=scorer,
                    )
            self.assertEqual(events, [])

    def test_rejects_nonready_models_before_injected_code(self):
        for status in ("pending_checkpoint_smoke", "unavailable"):
            spec = deepcopy(self.specs["TimeRCD"])
            registry = deepcopy(load_model_registry())
            registry["models"]["TimeRCD"]["execution_status"] = status
            registry["models"]["TimeRCD"]["status_reason"] = "unit test"
            calls = []
            with self.subTest(status=status), patch(
                "src.common.run_registered_model.load_model_registry_with_sha",
                return_value=(registry, spec["config_registry_sha256"]),
            ), patch(
                "src.common.run_registered_model.set_reproducible_seed",
                side_effect=lambda seed: calls.append(seed),
            ):
                with self.assertRaisesRegex(ValueError, "ready"):
                    execute_registered_model(
                        spec,
                        normal_training=numpy.zeros((100, 2)),
                        test_sessions=(numpy.zeros((100, 2)),),
                        device="cpu",
                        entrypoint=lambda *args, **kwargs: calls.append("entrypoint"),
                    )
            self.assertEqual(calls, [])

    def test_rejects_stale_spec_before_resolving_lazy_entrypoint(self):
        spec = {**self.specs["MWVAR"], "config_registry_sha256": "0" * 64}
        calls = []

        def load_entrypoint(model):
            calls.append(model)
            return lambda values: _score_output(values)

        with patch(
            "src.common.run_registered_model.load_model_entrypoint",
            side_effect=load_entrypoint,
        ):
            with self.assertRaisesRegex(ValueError, "registry"):
                execute_registered_model(
                    spec,
                    normal_training=_PoisonTraining(),
                    test_sessions=(numpy.zeros((100, 2)),),
                    device="cpu",
                )
        self.assertEqual(calls, [])

    def test_ratio_prefix_and_tier2_scaler_depend_only_on_fit(self):
        base = numpy.column_stack((numpy.arange(500), numpy.arange(500) + 100)).astype(float)
        changed = base.copy()
        changed[25:] = 1e12
        test = numpy.array([[1000.0, 2000.0], [2000.0, 3000.0]])
        runs = []
        for normal_training in (base, changed):
            adapter = _RecordingAdapter()
            result = execute_registered_model(
                self.specs["PaAno"], normal_training=normal_training,
                test_sessions=(test,), device="cpu",
                entrypoint=lambda **arguments: adapter,
            )
            runs.append((adapter, result))

        for adapter, result in runs:
            self.assertEqual(result["split"]["available_range"], (0, 25))
            self.assertEqual(result["split"]["fit_range"], (0, 20))
            self.assertEqual(result["split"]["validation_range"], (20, 25))
            self.assertEqual(result["scaler_state"]["data_min"], [0.0, 100.0])
            self.assertEqual(result["scaler_state"]["data_max"], [19.0, 119.0])
            self.assertGreater(adapter.score_calls[0][0, 0], 1.0)
            self.assertGreater(adapter.score_calls[1][0, 0], 1.0)
        numpy.testing.assert_array_equal(runs[0][0].fit_calls[0][0], runs[1][0].fit_calls[0][0])
        self.assertEqual(runs[0][1]["scaler_state"], runs[1][1]["scaler_state"])

    def test_pca_receives_raw_split_without_external_scaler(self):
        normal = numpy.column_stack((numpy.arange(500), numpy.arange(500) + 100)).astype(float)
        test = normal[:120]
        adapter = _RecordingAdapter()

        result = execute_registered_model(
            self.specs["PCA_LEGACY"], normal_training=normal,
            test_sessions=(test,), device="cpu",
            entrypoint=lambda **arguments: adapter,
        )

        numpy.testing.assert_array_equal(adapter.fit_calls[0][0], normal[:20])
        numpy.testing.assert_array_equal(adapter.score_calls[0], normal[20:25])
        self.assertIsNone(result["scaler_state"])
        self.assertIsNone(result["training_log"])

    def test_adapter_and_session_runner_signatures_are_model_specific(self):
        normal = numpy.arange(500 * 2, dtype=float).reshape(500, 2)
        test = numpy.arange(40, dtype=float).reshape(20, 2)

        paano = _RecordingAdapter()
        execute_registered_model(
            self.specs["PaAno"], normal_training=normal,
            test_sessions=(test,), device="cpu",
            entrypoint=lambda **arguments: paano,
        )
        self.assertEqual(len(paano.fit_calls[0]), 1)
        self.assertEqual(len(paano.score_calls), 2)

        for model in ("ALoRa", "GDN"):
            calls = []

            def session_runner(fit_sessions, validation_sessions, test_sessions, **arguments):
                calls.append((fit_sessions, validation_sessions, test_sessions, arguments))
                return {
                    "checkpoint": {"state": model},
                    "validation_outputs": (_score_output(validation_sessions[0]),),
                    "test_outputs": tuple(_score_output(values) for values in test_sessions),
                    "training_log": {"optimizer_updates": 1},
                    "timing": {
                        "model_setup_seconds": 0.4,
                        "training_seconds": 0.1,
                        "validation_inference_seconds": 0.1,
                        "test_inference_seconds": 0.1,
                    },
                }

            with self.subTest(model=model):
                result = execute_registered_model(
                    self.specs[model], normal_training=normal,
                    test_sessions=(test,), device="cpu", entrypoint=session_runner,
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(calls[0][0]), 1)
                self.assertEqual(len(calls[0][1]), 1)
                self.assertEqual(len(calls[0][2]), 1)
                numpy.testing.assert_allclose(
                    calls[0][2][0], (test - numpy.array([0.0, 1.0])) / 38.0,
                )
                self.assertEqual(result["training_log"]["optimizer_updates"], 1)
                self.assertEqual(result["timing"]["model_setup_seconds"], 0.4)

    def test_tier2_runner_synchronizes_before_closing_each_cuda_phase(self):
        class FakeModel:
            def state_dict(self):
                return {}

        sessions = (numpy.zeros((4, 2), dtype=float),)
        targets = (
            (
                "src.models.tier2.alora.adapter", "run_alora_sessions",
                "score_alora_sessions",
                {"window_size": 2, "device": "cpu", "epochs": 1},
            ),
            (
                "src.models.tier2.gdn_official.adapter", "run_gdn_sessions",
                "score_gdn_sessions",
                {
                    "embedding_dimension": 2, "hidden_dimension": 2,
                    "rho": 0.5, "device": "cpu",
                },
            ),
        )
        for module_name, runner_name, scorer_name, arguments in targets:
            module = real_import_module(module_name)
            events = []

            def trainer(*args, **kwargs):
                events.append("training")
                if runner_name == "run_alora_sessions":
                    return FakeModel(), {}, {}
                return FakeModel(), None, {"topk": 1}

            def scorer(*args, **kwargs):
                events.append("validation" if "validation" not in events else "test")
                return ()

            with self.subTest(runner=runner_name), patch.object(
                module, scorer_name, side_effect=scorer,
            ), patch.object(
                module, "_synchronize_cuda",
                side_effect=lambda device: events.append(("sync", device)),
                create=True,
            ), patch.object(
                module.time, "perf_counter",
                side_effect=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            ):
                result = getattr(module, runner_name)(
                    sessions, sessions, sessions, trainer=trainer, **arguments,
                )

            self.assertEqual(events, [
                "training", ("sync", "cpu"),
                "validation", ("sync", "cpu"),
                "test", ("sync", "cpu"),
            ])
            self.assertEqual(result["timing"]["training_seconds"], 1.0)
            self.assertEqual(result["timing"]["validation_inference_seconds"], 1.0)
            self.assertEqual(result["timing"]["test_inference_seconds"], 1.0)

    def test_tier2_runners_time_model_construction_as_setup(self):
        class FakeModel:
            def state_dict(self):
                return {}

        sessions = (numpy.zeros((4, 2), dtype=float),)
        targets = (
            (
                "src.models.tier2.alora.adapter", "run_alora_sessions",
                "build_alora_model", "train_alora", "score_alora_sessions",
                {"window_size": 2, "device": "cpu", "epochs": 1},
            ),
            (
                "src.models.tier2.gdn_official.adapter", "run_gdn_sessions",
                "build_gdn_model", "train_gdn", "score_gdn_sessions",
                {
                    "embedding_dimension": 2, "hidden_dimension": 2,
                    "rho": 0.5, "device": "cpu",
                },
            ),
        )
        for module_name, runner_name, builder_name, trainer_name, scorer_name, arguments in targets:
            module = real_import_module(module_name)
            events = []

            def builder(*args, **kwargs):
                events.append("construction")
                if runner_name == "run_alora_sessions":
                    return FakeModel(), {"prepared": True}
                return FakeModel(), "edge", 1

            def trainer(*args, **kwargs):
                events.append("optimization")
                self.assertIsInstance(kwargs["model"], FakeModel)
                if runner_name == "run_alora_sessions":
                    return kwargs["model"], kwargs["recipe"], {}
                return kwargs["model"], kwargs["edge_index"], {"topk": kwargs["topk"]}

            with self.subTest(runner=runner_name), patch.object(
                module, builder_name, side_effect=builder,
            ), patch.object(
                module, trainer_name, side_effect=trainer,
            ), patch.object(
                module, scorer_name, return_value=(),
            ), patch.object(
                module, "_synchronize_cuda",
                side_effect=lambda device: events.append(("sync", device)),
            ), patch.object(
                module.time, "perf_counter",
                side_effect=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            ):
                result = getattr(module, runner_name)(
                    sessions, sessions, sessions, trainer=None, **arguments,
                )

            self.assertEqual(events[:4], [
                "construction", ("sync", "cpu"),
                "optimization", ("sync", "cpu"),
            ])
            self.assertEqual(result["timing"]["model_setup_seconds"], 1.0)
            self.assertEqual(result["timing"]["training_seconds"], 1.0)

    def test_adapter_prepare_hook_runs_inside_model_setup_phase(self):
        events = []

        class PreparedAdapter(_RecordingAdapter):
            def prepare_model(self, channel_count):
                events.append(("construction", channel_count))

            def fit(self, *values):
                events.append("optimization")
                return super().fit(*values)

        adapter = PreparedAdapter()
        normal = numpy.arange(500 * 2, dtype=float).reshape(500, 2)
        test = numpy.arange(40, dtype=float).reshape(20, 2)
        with patch(
            "src.common.run_registered_model.time.perf_counter",
            side_effect=map(float, range(10)),
        ):
            result = execute_registered_model(
                self.specs["PaAno"], normal_training=normal,
                test_sessions=(test,), device="cpu",
                entrypoint=lambda **arguments: adapter,
            )

        self.assertEqual(events[:2], [("construction", 2), "optimization"])
        self.assertEqual(result["timing"]["model_setup_seconds"], 1.0)
        self.assertEqual(result["timing"]["training_seconds"], 1.0)

    def test_result_contains_complete_frozen_identity_and_timing(self):
        spec = self.specs["MWVAR"]
        result = execute_registered_model(
            spec, normal_training=_PoisonTraining(),
            test_sessions=(numpy.zeros((100, 2)),), device="cpu",
            entrypoint=lambda session: _score_output(session),
        )
        required = {
            "model", "config_id", "config_registry_sha256", "source_commit",
            "source_checkpoint_sha256", "checkpoint_config_sha256",
            "checkpoint_revision", "preprocess_recipe", "common_recipe",
            "common_recipe_id", "split", "scaler_state", "validation_outputs",
            "test_outputs", "checkpoint", "training_log", "timing", "seed_state",
        }
        self.assertTrue(required <= result.keys())
        self.assertEqual(result["common_recipe_id"], spec["common_recipe_id"])
        self.assertEqual(result["preprocess_recipe"], spec["preprocess_recipe"])
        self.assertEqual(result["common_recipe"], spec["common_recipe"])
        self.assertEqual(set(result["timing"]), {
            "split_preprocess_seconds", "model_setup_seconds", "training_seconds",
            "validation_inference_seconds", "test_inference_seconds", "accelerator",
        })

    def test_public_executor_and_entrypoint_receive_no_labels(self):
        self.assertNotIn("labels", inspect.signature(execute_registered_model).parameters)
        received = []

        def scorer(session):
            received.append(session)
            return _score_output(session)

        execute_registered_model(
            self.specs["MWVAR"], normal_training=_PoisonTraining(),
            test_sessions=(numpy.zeros((100, 2)),), device="cpu", entrypoint=scorer,
        )
        self.assertEqual(len(received), 1)


if __name__ == "__main__":
    unittest.main()
