"""공식 GDN 코어와 1-step 점수 정렬의 합성 검증."""

import inspect
import unittest
from pathlib import Path

import numpy
import torch

from src.models.tier2.gdn_official.adapter import (
    build_forecast_arrays,
    run_gdn_sessions,
    score_gdn_sessions,
    topk_from_rho,
    train_gdn,
)
from src.models.tier2.gdn_official.official import (
    GDN,
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


class TestGdnWindowsAndScores(unittest.TestCase):
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
