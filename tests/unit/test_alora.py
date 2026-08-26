"""ALoRa 공식 코어와 프로젝트 점수 계약의 합성 검증."""

import inspect
import unittest
from unittest.mock import patch

import numpy
import torch

from src.models.tier2.alora.adapter import (
    build_alora_model,
    run_alora_sessions,
    score_alora_sessions,
    select_spearman_pairs,
    stitch_alora_scores,
    train_alora,
)
from src.models.tier2.alora.official import (
    ATTENTION_RANK_THRESHOLD,
    ALoRaT,
    calculate_attention_rank,
)


class IdentityAttentionZeroReconstructor(torch.nn.Module):
    def forward(self, inputs):
        batch_size, window_size, _ = inputs.shape
        attention = torch.eye(window_size, device=inputs.device).expand(
            batch_size, 1, window_size, window_size,
        )
        return torch.zeros_like(inputs), (attention,), inputs


class TestSpearmanPairSelection(unittest.TestCase):
    def test_86_channels_select_exactly_512_fit_pairs(self):
        fit_values = numpy.random.default_rng(3).normal(size=(80, 86))
        pairs = select_spearman_pairs(fit_values)
        self.assertEqual(pairs.shape, (512, 2))
        self.assertEqual(len({tuple(pair) for pair in pairs}), 512)
        self.assertTrue(numpy.all(pairs[:, 0] < pairs[:, 1]))

    def test_constant_channel_nan_pairs_are_last_and_lexicographic(self):
        fit_values = numpy.column_stack((
            numpy.ones(8),
            numpy.arange(8),
            numpy.array([0, 1, 0, 1, 0, 1, 0, 1]),
            numpy.array([7, 2, 6, 1, 5, 0, 4, 3]),
        ))
        pairs = select_spearman_pairs(fit_values, top_k=6)
        self.assertEqual(
            pairs[-3:].tolist(),
            [[0, 1], [0, 2], [0, 3]],
        )

    def test_pair_selection_has_no_validation_or_test_input(self):
        parameters = inspect.signature(select_spearman_pairs).parameters
        self.assertEqual(tuple(parameters), ("fit_values", "top_k"))


class TestOfficialCore(unittest.TestCase):
    def test_rank_threshold_is_fixed_at_h1(self):
        attention = torch.diag(torch.tensor([1.0, 0.02, 0.009])).reshape(1, 1, 3, 3)
        rank = calculate_attention_rank(attention)
        self.assertEqual(ATTENTION_RANK_THRESHOLD, 0.01)
        self.assertEqual(rank.tolist(), [2.0])

    def test_86_channel_forward_uses_selected_pairs(self):
        fit_values = numpy.random.default_rng(4).normal(size=(40, 86))
        pairs = select_spearman_pairs(fit_values)
        model = ALoRaT(
            window_size=20,
            channel_count=86,
            output_channels=86,
            selected_pairs=pairs,
            pair_embedding_dimension=512,
            attention_heads=8,
            encoder_layers=3,
        )
        reconstruction, attentions, embedding = model(torch.rand(1, 20, 86))
        self.assertEqual(reconstruction.shape, (1, 20, 86))
        self.assertEqual(embedding.shape, (1, 20, 512))
        self.assertEqual(len(attentions), 3)
        self.assertEqual(attentions[-1].shape, (1, 8, 20, 20))
        state_keys = model.state_dict()
        self.assertIn("embedding.value_embedding.weights", state_keys)
        self.assertIn("embedding.value_embedding.pairs_idx", state_keys)
        self.assertIn("encoder.attn_layers.0.attention.out_projection.weight", state_keys)


class TestScoreStitching(unittest.TestCase):
    def test_first_window_full_then_only_each_last_point(self):
        reconstruction_mse = numpy.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        ranks = numpy.array([10.0, 20.0, 30.0])
        stitched = stitch_alora_scores(reconstruction_mse, ranks)
        numpy.testing.assert_array_equal(stitched, [10.0, 20.0, 30.0, 120.0, 270.0])

    def test_all_registered_windows_preserve_source_length(self):
        model = IdentityAttentionZeroReconstructor()
        for window_size in (20, 40, 60, 80, 100):
            with self.subTest(window_size=window_size):
                length = window_size + 7
                session = numpy.ones((length, 3), dtype=numpy.float32)
                output = score_alora_sessions(
                    model, (session,), window_size=window_size, device="cpu", batch_size=3,
                )[0]
                self.assertEqual(output["scores"].shape, (length,))
                numpy.testing.assert_array_equal(output["scores"], window_size)
                self.assertEqual(output["source_start"], 0)
                self.assertEqual(output["source_end_exclusive"], length)
                self.assertEqual(output["alignment"], "same_timestep")
                self.assertEqual(output["calibration_mode"], "validation_median_iqr")
                self.assertEqual(output["evaluation_mode"], "offline_noncausal")
                self.assertEqual(output["lookahead"], 0)
                self.assertEqual(
                    output["maximum_effective_lookahead"], window_size - 1,
                )
                self.assertEqual(
                    output["normalization_scope"], "current_prefix_validation",
                )

    def test_model_builder_uses_fit_values_only(self):
        fit_values = numpy.random.default_rng(5).normal(size=(30, 8))
        model, checkpoint_recipe = build_alora_model(
            (fit_values,), window_size=20, device="cpu",
            pair_embedding_dimension=16, attention_heads=8, encoder_layers=1,
        )
        self.assertIsInstance(model, ALoRaT)
        numpy.testing.assert_array_equal(
            checkpoint_recipe["selected_pairs"],
            select_spearman_pairs(fit_values, top_k=16),
        )

    def test_training_and_scoring_interfaces_do_not_accept_labels(self):
        for function in (train_alora, score_alora_sessions):
            self.assertNotIn("labels", inspect.signature(function).parameters)

    def test_runner_exposes_every_registry_training_default(self):
        parameters = inspect.signature(run_alora_sessions).parameters
        for name in (
            "epochs", "batch_size", "learning_rate", "low_rank_weight",
            "pair_embedding_dimension", "attention_heads", "encoder_layers", "patience",
        ):
            self.assertIn(name, parameters)

    def test_project_trainer_makes_an_optimizer_update(self):
        random_generator = numpy.random.default_rng(6)
        fit = random_generator.normal(size=(22, 8)).astype(numpy.float32)
        validation = random_generator.normal(size=(21, 8)).astype(numpy.float32)
        with patch(
            "torch.randperm",
            side_effect=AssertionError("ALoRa 학습은 고정 batch 순서를 써야 한다"),
        ):
            _, recipe, training_log = train_alora(
                (fit,),
                (validation,),
                window_size=20,
                device="cpu",
                epochs=1,
                batch_size=2,
                pair_embedding_dimension=16,
                attention_heads=8,
                encoder_layers=1,
            )
        self.assertGreater(training_log["optimizer_updates"], 0)
        self.assertEqual(training_log["training_examples_seen"], 3)
        self.assertEqual(recipe["rank_threshold"], 0.01)
        self.assertEqual(numpy.asarray(recipe["selected_pairs"]).shape, (16, 2))


if __name__ == "__main__":
    unittest.main()
