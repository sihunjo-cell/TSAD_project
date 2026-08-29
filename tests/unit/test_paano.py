"""PaAno source-compatible core and project adapter contracts."""

import inspect
import unittest
from unittest.mock import patch

import numpy
import torch
from torch import nn

from src.common.model_registry import load_model_registry
from src.models.tier2.paano.adapter import PaAnoAdapter, stitch_patch_scores
from src.models.tier2.paano.official import (
    PatchEncoder,
    choose_training_batch_size,
    select_memory_bank,
    train_encoder,
)


class MeanEncoder(nn.Module):
    """Fast deterministic encoder used to expose session leakage."""

    def embedding(self, patches):
        return torch.cat((patches.mean(dim=2), patches.std(dim=2)), dim=1)


class RecordingTrainer:
    def __init__(self):
        self.fit_patches = None

    def __call__(self, model, fit_patches, **parameters):
        self.fit_patches = fit_patches.detach().clone()
        return {"optimizer_updates": 0, "parameters": parameters}


class RecordingClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self.last_batch_size = None

    def forward(self, features):
        self.last_batch_size = len(features)
        return self.linear(features)


class TinyEncoder(nn.Module):
    def __init__(self, input_width=2):
        super().__init__()
        self.embedding_layer = nn.Linear(input_width, 2)
        self.projection_head = nn.Linear(2, 2)
        self.classification_head = RecordingClassificationHead()

    def embedding(self, patches):
        return self.embedding_layer(patches.flatten(start_dim=1))

    def projection(self, embeddings):
        return self.projection_head(embeddings)


class TestPaAnoOfficialCore(unittest.TestCase):
    def test_patch_encoder_keeps_official_embedding_width(self):
        encoder = PatchEncoder(in_channels=2)
        embeddings = encoder.embedding(torch.randn(3, 2, 32))
        projections = encoder.projection(embeddings)

        self.assertIsNotNone(encoder.revin)
        self.assertEqual(embeddings.shape, (3, 64))
        self.assertEqual(projections.shape, (3, 256))

    def test_memory_bank_is_exactly_ten_percent(self):
        generator = torch.Generator().manual_seed(3)
        embeddings = torch.randn(40, 6, generator=generator)

        memory = select_memory_bank(embeddings, fraction=0.1, random_seed=42)

        self.assertEqual(memory.shape, (4, 6))

    def test_memory_bank_rejects_fewer_than_three_representatives(self):
        with self.assertRaisesRegex(ValueError, "3"):
            select_memory_bank(torch.randn(29, 4), fraction=0.1)

    def test_training_batch_size_never_leaves_a_singleton(self):
        self.assertEqual(choose_training_batch_size(513, 512), 511)
        self.assertEqual(choose_training_batch_size(514, 512), 512)
        self.assertEqual(choose_training_batch_size(30, 512), 30)

    def test_pretext_uses_five_nonadjacent_patches_per_anchor(self):
        torch.manual_seed(4)
        model = TinyEncoder()
        patches = torch.randn(30, 1, 2)

        train_encoder(
            model,
            patches,
            iterations=6,
            batch_size=30,
            learning_rate=1e-4,
            weight_decay=1e-4,
            device="cpu",
        )

        valid_prior_count = len(patches) - patches.shape[-1]
        self.assertEqual(
            model.classification_head.last_batch_size,
            valid_prior_count + 5 * len(patches),
        )

    def test_batch_without_prior_pretext_stays_finite(self):
        model = TinyEncoder(4)
        patches = torch.randn(3, 1, 4)

        training_log = train_encoder(
            model,
            patches,
            iterations=6,
            batch_size=3,
            learning_rate=1e-4,
            weight_decay=1e-4,
            device="cpu",
        )

        self.assertTrue(numpy.isfinite(training_log["best_training_loss"]))
        self.assertTrue(all(
            torch.isfinite(parameter).all() for parameter in model.parameters()
        ))


class TestPaAnoAdapter(unittest.TestCase):
    def test_registry_candidate_maps_directly_to_constructor(self):
        candidate = load_model_registry()["models"]["PaAno"]["candidates"][0]

        adapter = PaAnoAdapter(device="cpu", **candidate["hyperparameters"])

        self.assertEqual(adapter.top_k, 3)
        self.assertEqual(adapter.memory_seed, 42)

    def test_patch_scores_are_overlap_means(self):
        actual = stitch_patch_scores([1.0, 2.0, 3.0], patch_size=2, length=4)
        numpy.testing.assert_allclose(actual, [1.0, 1.5, 2.5, 3.0])

    def test_memory_is_fit_only_and_scores_keep_session_length(self):
        trainer = RecordingTrainer()
        adapter = PaAnoAdapter(
            patch_size=2,
            learning_rate=1e-4,
            device="cpu",
            encoder_factory=lambda channel_count: MeanEncoder(),
            trainer=trainer,
        )
        fit = numpy.arange(62, dtype=numpy.float32).reshape(31, 2)
        validation = numpy.full((7, 2), 10_000.0, dtype=numpy.float32)
        test = numpy.full((8, 2), -10_000.0, dtype=numpy.float32)

        training_log = adapter.fit(fit)
        memory_before = adapter.memory_bank.clone()
        validation_output = adapter.score(validation)
        test_output = adapter.score(test)

        self.assertEqual(trainer.fit_patches.shape[0], 30)
        self.assertEqual(adapter.memory_bank.shape[0], 3)
        torch.testing.assert_close(adapter.memory_bank, memory_before)
        self.assertEqual(validation_output["scores"].shape, (7,))
        self.assertEqual(test_output["scores"].shape, (8,))
        self.assertTrue(numpy.isfinite(validation_output["scores"]).all())
        self.assertEqual(validation_output["source_start"], 0)
        self.assertEqual(validation_output["source_end_exclusive"], 7)
        self.assertEqual(validation_output["alignment"], "overlap_mean")
        self.assertEqual(validation_output["calibration_mode"], "validation_median_iqr")
        self.assertEqual(validation_output["evaluation_mode"], "offline_noncausal")
        self.assertEqual(validation_output["lookahead"], 1)
        self.assertEqual(validation_output["maximum_effective_lookahead"], 1)
        self.assertEqual(
            validation_output["normalization_scope"], "current_prefix_validation",
        )
        self.assertEqual(training_log["parameters"]["iterations"], 100)
        self.assertEqual(training_log["parameters"]["batch_size"], 512)
        self.assertEqual(training_log["parameters"]["weight_decay"], 1e-4)
        self.assertEqual(training_log["training_examples_seen"], 0)

    def test_training_examples_seen_counts_anchor_patches_in_main_updates(self):
        def two_update_trainer(_model, _fit_patches, **_parameters):
            return {"optimizer_updates": 2}

        adapter = PaAnoAdapter(
            patch_size=2, learning_rate=1e-4, device="cpu",
            encoder_factory=lambda _channel_count: MeanEncoder(),
            trainer=two_update_trainer,
        )

        training_log = adapter.fit(
            numpy.arange(62, dtype=numpy.float32).reshape(31, 2),
        )

        self.assertEqual(training_log["training_examples_seen"], 60)

    def test_prepared_model_is_not_reconstructed_during_fit(self):
        constructions = []

        def build_encoder(channel_count):
            constructions.append(channel_count)
            return MeanEncoder()

        adapter = PaAnoAdapter(
            patch_size=2, learning_rate=1e-4, device="cpu",
            encoder_factory=build_encoder, trainer=RecordingTrainer(),
        )
        adapter.prepare_model(2)
        with patch(
            "src.models.tier2.paano.adapter.encode_patches",
            return_value=torch.zeros((30, 2)),
        ), patch(
            "src.models.tier2.paano.adapter.select_memory_bank",
            return_value=torch.zeros((3, 2)),
        ):
            adapter.fit(numpy.arange(62, dtype=numpy.float32).reshape(31, 2))

        self.assertEqual(constructions, [2])

    def test_checkpoint_restores_encoder_memory_and_recipe(self):
        adapter = PaAnoAdapter(
            patch_size=2,
            learning_rate=1e-4,
            device="cpu",
            encoder_factory=lambda channel_count: MeanEncoder(),
            trainer=RecordingTrainer(),
        )
        fit = numpy.arange(62, dtype=numpy.float32).reshape(31, 2)
        test = numpy.arange(16, dtype=numpy.float32).reshape(8, 2)
        adapter.fit(fit)
        expected = adapter.score(test)["scores"]

        restored = PaAnoAdapter.from_checkpoint(
            adapter.checkpoint(),
            device="cpu",
            encoder_factory=lambda channel_count: MeanEncoder(),
            trainer=RecordingTrainer(),
        )

        numpy.testing.assert_allclose(restored.score(test)["scores"], expected)
        torch.testing.assert_close(restored.memory_bank, adapter.memory_bank)

    def test_public_methods_do_not_accept_labels(self):
        fit_parameters = inspect.signature(PaAnoAdapter.fit).parameters
        score_parameters = inspect.signature(PaAnoAdapter.score).parameters

        self.assertNotIn("labels", fit_parameters)
        self.assertNotIn("labels", score_parameters)


if __name__ == "__main__":
    unittest.main()
