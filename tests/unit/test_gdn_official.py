"""공식 GDN 코어와 1-step 점수 정렬의 합성 검증."""

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy
import torch

from src.models.tier2.gdn_official.adapter import (
    build_forecast_arrays,
    build_gdn_model,
    run_gdn_sessions,
    score_gdn_sessions,
    split_official_windows,
    topk_from_rho,
    train_gdn,
)
from src.models.tier2.gdn_official.official import (
    GDN,
    GNNLayer,
    build_fully_connected_edge_index,
)


class ZeroPredictor(torch.nn.Module):
    def forward(self, inputs, original_edge_index=None):
        return torch.zeros(
            (inputs.shape[0], inputs.shape[1]),
            dtype=inputs.dtype,
            device=inputs.device,
        )


class TestGdnRecipe(unittest.TestCase):
    def test_inspection_attention_does_not_retain_autograd_graph(self):
        source = inspect.getsource(GNNLayer.forward)
        self.assertIn("self.attention_weights = attention.detach()", source)

    def test_topk_formula(self):
        self.assertEqual(topk_from_rho(19, 0.30), 5)
        self.assertEqual(topk_from_rho(86, 0.25), 21)
        self.assertEqual(topk_from_rho(2, 0.01), 1)
        self.assertEqual(topk_from_rho(3, 10.0), 2)

    def test_complete_graph_and_learned_graph_preserve_self_edges(self):
        edge_index = build_fully_connected_edge_index(4, "cpu")
        self.assertEqual(edge_index.shape, (2, 16))
        self.assertTrue(all(
            ((edge_index[0] == node) & (edge_index[1] == node)).any()
            for node in range(4)
        ))
        model = GDN(
            [edge_index], node_num=4, dim=8, out_layer_inter_dim=8,
            input_dim=5, out_layer_num=1, topk=1,
        )
        prediction = model(torch.rand(2, 4, 5), edge_index)
        self.assertEqual(prediction.shape, (2, 4))
        self.assertTrue(torch.equal(model.learned_graph[:, 0], torch.arange(4)))
        self.assertIn("gnn_layers.0.gnn.lin.weight", model.state_dict())
        self.assertIn("gnn_layers.0.bn.weight", model.state_dict())

    def test_fixed_topk_two_connects_both_channels_without_replacing_legacy_rho(self):
        model, edge_index, topk = build_gdn_model(
            2, embedding_dimension=8, hidden_dimension=8,
            fixed_topk=2, window_size=5, device="cpu",
        )
        self.assertEqual(topk, 2)
        prediction = model(torch.rand(2, 2, 5), edge_index)
        self.assertEqual(prediction.shape, (2, 2))
        for neighbors in model.learned_graph:
            self.assertEqual(set(neighbors.tolist()), {0, 1})
        self.assertEqual(topk_from_rho(2, 0.3), 1)

    def test_fixed_topk_larger_than_channel_count_is_not_clamped(self):
        with self.assertRaisesRegex(ValueError, "topk 5 > channel count 3"):
            build_gdn_model(
                3, embedding_dimension=8, hidden_dimension=8,
                fixed_topk=5, window_size=5, device="cpu",
            )


class TestGdnWindowsAndScores(unittest.TestCase):
    def test_official_split_preserves_source_rounding_and_contiguous_holdout(self):
        with patch("src.models.tier2.gdn_official.adapter.random.Random") as seeded:
            seeded.return_value.randrange.return_value = 3
            training, validation = split_official_windows(11, 0.2, 7)
        seeded.assert_called_once_with(7)
        seeded.return_value.randrange.assert_called_once_with(8)
        self.assertEqual(training, [0, 1, 2, 5, 6, 7, 8, 9, 10])
        self.assertEqual(validation, [3, 4])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            split_official_windows(9, 0.1, 7)

    def test_official_score_normalizes_full_evaluation_then_smooths_then_maxes(self):
        values = numpy.column_stack((numpy.arange(6), numpy.zeros(6))).astype(numpy.float32)
        output = score_gdn_sessions(
            ZeroPredictor(), (values,), window_size=1,
            edge_index=torch.empty((2, 0), dtype=torch.long), device="cpu",
            batch_size=2, full_prefix=True, official_procedure=True,
        )[0]
        numpy.testing.assert_allclose(output["scores"], [0, 0, 0, 0, 0.5 / 2.01])
        numpy.testing.assert_allclose(output["native_channel_scores"], numpy.column_stack((
            [0, 0, 0, -0.5 / 2.01, 0.5 / 2.01], numpy.zeros(5),
        )))
        numpy.testing.assert_array_equal(output["scores"], output["native_channel_scores"].max(axis=1))
        self.assertEqual(output["native_calibration"], {
            "source": "full_evaluation", "source_start": 1, "source_end_exclusive": 6,
            "score_space": "absolute_error", "median": [3.0, 0.0], "iqr": [2.0, 0.0],
            "epsilon": 0.01,
        })
        self.assertTrue(output["native_postprocessing"])
        self.assertEqual(output["calibration_mode"], "official_full_evaluation")
        self.assertEqual(output["evaluation_mode"], "offline_noncausal")
        self.assertEqual(output["maximum_effective_lookahead"], 4)

    def test_official_validation_uses_batch_means_and_restores_best_epoch(self):
        class ScheduledPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.zeros(()))
                self.register_buffer("validation_batches", torch.zeros((), dtype=torch.long))

            def forward(self, inputs, original_edge_index=None):
                prediction = 0.0
                if not self.training:
                    prediction = (0.0, 3.0, 2.0, 2.0, 3.0, 3.0)[self.validation_batches.item()]
                    self.validation_batches.add_(1)
                return (self.offset + prediction).expand(inputs.shape[0], inputs.shape[1])

        with patch("torch.optim.Adam", wraps=torch.optim.Adam) as optimizer:
            model, _, log = train_gdn(
                (numpy.zeros((11, 3), dtype=numpy.float32),), (),
                embedding_dimension=8, hidden_dimension=8, fixed_topk=2,
                device="cpu", window_size=1, epochs=5, patience=1, batch_size=2,
                learning_rate=0.0, full_prefix=True, official_procedure=True,
                validation_ratio=0.3, split_seed=7, optimizer_betas=(0.9, 0.99),
                model=ScheduledPredictor(), edge_index=torch.empty((2, 0), dtype=torch.long), topk=2,
            )
        self.assertEqual(optimizer.call_args.kwargs["betas"], (0.9, 0.99))
        self.assertEqual(log["selected_epoch"], 2)
        self.assertEqual(log["epochs_completed"], 3)
        self.assertEqual(log["best_validation_loss"], 4.0)
        self.assertEqual(model.validation_batches.item(), 4)
        self.assertEqual(log["model_internal_validation"]["window_count"], 3)
        self.assertEqual(log["model_internal_validation"]["training_window_count"], 7)
        self.assertEqual(log["loss_history"], [
            {"epoch": 1, "training_batch_mse_sum": 0.0, "training_batch_mse_mean": 0.0,
             "validation_loss": 4.5},
            {"epoch": 2, "training_batch_mse_sum": 0.0, "training_batch_mse_mean": 0.0,
             "validation_loss": 4.0},
            {"epoch": 3, "training_batch_mse_sum": 0.0, "training_batch_mse_mean": 0.0,
             "validation_loss": 9.0},
        ])

    def test_official_runner_scores_only_test_with_native_procedure(self):
        fit = numpy.zeros((11, 3), dtype=numpy.float32)
        test = numpy.ones((9, 3), dtype=numpy.float32)

        def trainer(fit_sessions, validation_sessions, **arguments):
            self.assertEqual(validation_sessions, ())
            self.assertTrue(arguments["official_procedure"])
            self.assertEqual(arguments["split_seed"], 7)
            return ZeroPredictor(), None, {"topk": 2}

        with patch("src.models.tier2.gdn_official.adapter.score_gdn_sessions", return_value=()) as scorer:
            result = run_gdn_sessions(
                (fit,), (), (test,), embedding_dimension=8, hidden_dimension=8,
                fixed_topk=2, device="cpu", full_prefix=True, official_procedure=True,
                split_seed=7, trainer=trainer,
            )
        self.assertEqual(scorer.call_count, 1)
        self.assertIs(scorer.call_args.args[1][0], test)
        self.assertTrue(scorer.call_args.kwargs["official_procedure"])
        self.assertEqual(result["calibration_outputs"], ())
        self.assertEqual(result["timing"]["calibration_inference_seconds"], 0.0)

    def test_full_prefix_calibration_comes_only_from_fit_scores(self):
        fit = numpy.ones((8, 3), dtype=numpy.float32)
        test = numpy.full((9, 3), 100.0, dtype=numpy.float32)
        calls = []

        def trainer(fit_sessions, validation_sessions, **arguments):
            self.assertEqual(validation_sessions, ())
            self.assertTrue(arguments["full_prefix"])
            self.assertEqual(arguments["fixed_topk"], 2)
            self.assertNotIn("rho", arguments)
            return ZeroPredictor(), None, {"topk": 2}

        def scorer(_model, sessions, **arguments):
            calls.append(sessions)
            self.assertTrue(arguments["full_prefix"])
            return ({"scores": sessions[0]},)

        with patch("src.models.tier2.gdn_official.adapter.score_gdn_sessions", side_effect=scorer):
            result = run_gdn_sessions(
                (fit,), (), (test,), embedding_dimension=8, hidden_dimension=8,
                fixed_topk=2, device="cpu", full_prefix=True, trainer=trainer,
            )
        self.assertIs(calls[0][0], fit)
        self.assertIs(calls[1][0], test)
        self.assertEqual(result["validation_outputs"], ())
        self.assertIs(result["calibration_outputs"][0]["scores"], fit)
        self.assertEqual(result["timing"]["validation_inference_seconds"], 0.0)
        self.assertEqual(result["checkpoint"]["model_config"]["fixed_topk"], 2)
        self.assertEqual(result["checkpoint"]["training_log"], result["training_log"])

    def test_training_callback_precedes_inference_and_reuses_checkpoint(self):
        sessions = (numpy.zeros((11, 3), dtype=numpy.float32),)
        for failure in (None, "scoring failed", "persistence failed"):
            events = []
            captured = []

            class TrainedModel:
                def state_dict(self):
                    events.append("checkpoint")
                    return {"weight": "trained"}

            def trainer(*args, **kwargs):
                events.append("fit")
                return TrainedModel(), None, {"topk": 2, "selected_epoch": 1}

            def persist_training(partial):
                events.append("persist")
                captured.append(partial)
                if failure == "persistence failed":
                    raise OSError(failure)

            def scorer(*args, **kwargs):
                events.append("score")
                if failure == "scoring failed":
                    raise OSError(failure)
                return ()

            with self.subTest(failure=failure), patch(
                "src.models.tier2.gdn_official.adapter.score_gdn_sessions", side_effect=scorer,
            ), patch(
                "src.models.tier2.gdn_official.adapter._synchronize_cuda",
                side_effect=lambda device: events.append("sync"),
            ), patch(
                "src.models.tier2.gdn_official.adapter.time.perf_counter", side_effect=map(float, range(20)),
            ):
                def run():
                    return run_gdn_sessions(
                        sessions, (), sessions, embedding_dimension=8, hidden_dimension=8,
                        fixed_topk=2, device="cpu", full_prefix=True, official_procedure=True,
                        trainer=trainer, on_training_complete=persist_training,
                    )

                if failure is None:
                    result = run()
                    self.assertIs(result["checkpoint"], captured[0]["checkpoint"])
                else:
                    with self.assertRaisesRegex(OSError, failure):
                        run()
            self.assertEqual(events[:4], ["fit", "sync", "checkpoint", "persist"])
            self.assertEqual(events.count("checkpoint"), 1)
            self.assertEqual(events.count("score"), 0 if failure == "persistence failed" else 1)
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["training_log"]["selected_epoch"], 1)
            self.assertEqual(captured[0]["timing"]["training_seconds"], 1.0)
            self.assertEqual(captured[0]["checkpoint"]["model_state_dict"], {"weight": "trained"})

    def test_full_prefix_training_never_builds_a_validation_loader(self):
        fit = numpy.random.default_rng(9).normal(size=(9, 3)).astype(numpy.float32)
        from src.models.tier2.gdn_official import adapter
        with patch.object(adapter, "_make_dataset", wraps=adapter._make_dataset) as make_dataset:
            _, _, log = train_gdn(
                (fit,), (), embedding_dimension=8, hidden_dimension=8,
                rho=0.5, device="cpu", window_size=5, epochs=2,
                batch_size=2, full_prefix=True,
            )
        self.assertEqual(make_dataset.call_count, 1)
        self.assertEqual(log["epochs_completed"], 2)
        self.assertIsNone(log["best_validation_loss"])
        self.assertEqual(log["checkpoint_selection"], "best_training_loss")

    def test_full_prefix_restores_minimum_batch_loss_sum_after_full_cap(self):
        class ScheduledPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.zeros(()))
                self.register_buffer("batches_seen", torch.zeros((), dtype=torch.long))

            def forward(self, inputs, original_edge_index=None):
                prediction = (0.0, 3.0, 2.0, 2.0, 3.0, 3.0)[self.batches_seen.item()]
                self.batches_seen.add_(1)
                return (self.offset + prediction).expand(inputs.shape[0], inputs.shape[1])

        fit = numpy.zeros((8, 3), dtype=numpy.float32)
        model, _, log = train_gdn(
            (fit,), (), embedding_dimension=8, hidden_dimension=8,
            fixed_topk=2, device="cpu", window_size=5, epochs=3, patience=1,
            batch_size=2, learning_rate=0.0, full_prefix=True,
            model=ScheduledPredictor(), edge_index=torch.empty((2, 0), dtype=torch.long),
            topk=2,
        )
        # The uneven batches give epoch sums 9, 8, 18; row weighting would choose epoch 1.
        self.assertEqual(log["selected_epoch"], 2)
        self.assertEqual(log["best_training_loss"], 8.0)
        self.assertEqual(log["training_loss_reduction"], "sum_batch_mean_mse")
        self.assertEqual(model.batches_seen.item(), 4)
        self.assertEqual(log["epoch_cap"], 3)
        self.assertEqual(log["epochs_completed"], 3)
        self.assertEqual(log["optimizer_updates"], 6)
        self.assertEqual(log["training_examples_seen"], 9)
        self.assertFalse(log["stopped_early"])
        self.assertEqual(log["loss_history"], [
            {"epoch": 1, "training_batch_mse_sum": 9.0, "training_batch_mse_mean": 4.5,
             "validation_loss": None},
            {"epoch": 2, "training_batch_mse_sum": 8.0, "training_batch_mse_mean": 4.0,
             "validation_loss": None},
            {"epoch": 3, "training_batch_mse_sum": 18.0, "training_batch_mse_mean": 9.0,
             "validation_loss": None},
        ])

    def test_full_prefix_rejects_later_nonfinite_loss_despite_an_earlier_checkpoint(self):
        class DivergingPredictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.zeros(()))
                self.calls = 0

            def forward(self, inputs, original_edge_index=None):
                self.calls += 1
                prediction = self.offset if self.calls == 1 else self.offset + float("nan")
                return prediction.expand(inputs.shape[0], inputs.shape[1])

        model = DivergingPredictor()
        with self.assertRaisesRegex(RuntimeError, "training loss.*finite"):
            train_gdn(
                (numpy.zeros((7, 3), dtype=numpy.float32),), (),
                embedding_dimension=8, hidden_dimension=8, fixed_topk=2,
                device="cpu", window_size=5, epochs=2, batch_size=2,
                full_prefix=True, model=model,
                edge_index=torch.empty((2, 0), dtype=torch.long), topk=2,
            )
        self.assertEqual(model.calls, 2)
        self.assertIsNone(model.offset.grad)

    def test_sessions_are_windowed_separately_without_boundary_crossing(self):
        first_column = numpy.arange(7, dtype=numpy.float32)
        second_column = 100 + numpy.arange(8, dtype=numpy.float32)
        first = numpy.column_stack((first_column, first_column + 10))
        second = numpy.column_stack((second_column, second_column + 10))
        arrays = build_forecast_arrays((first, second), window_size=5)
        self.assertEqual([len(targets) for _, targets in arrays], [2, 3])
        numpy.testing.assert_array_equal(arrays[0][0][-1, 0], [1, 2, 3, 4, 5])
        numpy.testing.assert_array_equal(arrays[1][0][0, 0], [100, 101, 102, 103, 104])
        self.assertLess(arrays[0][0].max(), 50)
        self.assertGreater(arrays[1][0].min(), 50)

    def test_zero_predictor_returns_absolute_next_step_error(self):
        values = numpy.arange(21, dtype=numpy.float32).reshape(7, 3)
        edge_index = build_fully_connected_edge_index(3, "cpu")
        output = score_gdn_sessions(
            ZeroPredictor(), (values,), window_size=5, edge_index=edge_index,
            device="cpu", batch_size=2,
        )[0]
        numpy.testing.assert_array_equal(output["scores"], numpy.abs(values[5:]))
        self.assertEqual(output["scores"].shape, (2, 3))
        self.assertEqual(output["source_start"], 5)
        self.assertEqual(output["source_end_exclusive"], 7)
        self.assertEqual(output["alignment"], "next_step")
        self.assertEqual(output["primitive"], "absolute_error")
        self.assertEqual(output["calibration_mode"], "validation_median_iqr")
        self.assertEqual(output["evaluation_mode"], "causal")
        self.assertEqual(output["lookahead"], 0)
        self.assertEqual(output["maximum_effective_lookahead"], 0)
        self.assertEqual(output["normalization_scope"], "current_prefix_validation")

    def test_official_package_keeps_license(self):
        self.assertIn("MIT License", Path("src/models/tier2/gdn_official/LICENSE").read_text())

    def test_adapter_interfaces_do_not_accept_labels(self):
        for function in (train_gdn, score_gdn_sessions, run_gdn_sessions):
            self.assertNotIn("labels", inspect.signature(function).parameters)

    def test_runner_exposes_registry_learning_rate(self):
        self.assertIn("learning_rate", inspect.signature(run_gdn_sessions).parameters)

    def test_project_trainer_makes_an_optimizer_update(self):
        random_generator = numpy.random.default_rng(7)
        fit = random_generator.normal(size=(9, 3)).astype(numpy.float32)
        validation = random_generator.normal(size=(8, 3)).astype(numpy.float32)
        _, _, training_log = train_gdn(
            (fit,),
            (validation,),
            embedding_dimension=8,
            hidden_dimension=8,
            rho=0.5,
            device="cpu",
            window_size=5,
            epochs=1,
            patience=1,
            batch_size=2,
        )
        self.assertGreater(training_log["optimizer_updates"], 0)
        self.assertEqual(training_log["training_examples_seen"], 4)
        self.assertEqual(training_log["topk"], 1)


if __name__ == "__main__":
    unittest.main()
