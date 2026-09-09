"""PaAno source-compatible core and project adapter contracts."""

import inspect
import unittest
from unittest.mock import patch

import numpy
import torch
from torch import nn

from src.common.model_registry import load_model_registry
from src.models.tier2.paano import official as paano_official
from src.models.tier2.paano.adapter import PaAnoAdapter, make_patch_tensor, stitch_patch_scores
from src.models.tier2.paano.official import (
    PatchEncoder,
    choose_training_batch_size,
    score_embeddings,
    score_patches,
    select_memory_bank,
    select_memory_count,
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
    def test_scoring_streams_source_batches_on_the_requested_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batches = []

        class RecordingEncoder(nn.Module):
            def embedding(self, values):
                batches.append((len(values), values.device.type))
                return values[:, 0, :]

        patches = torch.tensor([[1.0, 0.0]] * 513).unsqueeze(1)
        patches[0, 0] = torch.tensor([float("inf"), 1.0])
        patches[1, 0] = torch.tensor([float("-inf"), 1.0])
        patches[2, 0] = torch.tensor([float("nan"), 1.0])
        memory = torch.tensor([[0.0, 1.0]] * 3)
        with patch(
            "src.models.tier2.paano.official._score_batch", wraps=paano_official._score_batch,
        ) as scorer:
            scores = score_patches(RecordingEncoder(), patches, memory, device=device)

        self.assertEqual(batches, [(512, device), (1, device)])
        self.assertEqual([len(call.args[0]) for call in scorer.call_args_list], [512, 1])
        self.assertTrue(all(
            call.args[0].device.type == call.args[1].device.type == device
            for call in scorer.call_args_list
        ))
        self.assertEqual(scores.device.type, "cpu")
        torch.testing.assert_close(scores[:3], torch.zeros(3))
        torch.testing.assert_close(scores[3:], torch.ones(510))
        torch.testing.assert_close(
            score_embeddings(patches[:3, 0], memory), torch.zeros(3),
        )

    def test_memory_distance_blocks_preserve_source_representatives_and_ties(self):
        embeddings = torch.tensor([
            [1.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-2.0, 0.0],
            [2.0, 0.0], [0.0, -3.0], [0.0, 4.0], [-4.0, 0.0],
        ])
        centers = numpy.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=numpy.float32)
        for candidates, fraction in ((embeddings, 0.5), (embeddings.repeat(4, 1), 0.125)):
            source_indices = torch.cdist(
                torch.nn.functional.normalize(candidates, dim=1), torch.from_numpy(centers),
            ).argmin(dim=0)
            with self.subTest(patch_count=len(candidates)), patch(
                "src.models.tier2.paano.official.MiniBatchKMeans",
            ) as clustering, patch(
                "src.models.tier2.paano.official._MEMORY_DISTANCE_BLOCK_SIZE", 3,
            ), patch("src.models.tier2.paano.official.torch.cdist", wraps=torch.cdist) as distances:
                clustering.return_value.fit.return_value.cluster_centers_ = centers
                memory = select_memory_bank(candidates, fraction=fraction)

            torch.testing.assert_close(source_indices, torch.tensor([0, 2, 3, 5]))
            torch.testing.assert_close(memory, candidates[source_indices])
            self.assertGreater(len(distances.call_args_list), 1)
            self.assertTrue(all(
                len(call.args[0]) <= 3 and len(call.args[1]) <= 3
                for call in distances.call_args_list
            ))
            self.assertEqual(tuple(len(values) for values in distances.call_args.args), (2, 1))
            self.assertEqual(
                {call.kwargs["compute_mode"] for call in distances.call_args_list},
                {"use_mm_for_euclid_dist" if len(candidates) > 25 else "donot_use_mm_for_euclid_dist"},
            )

    def test_official_memory_minimum_keeps_short_prefix_candidates(self):
        for patch_count, expected in ((4, 3), (30, 29), (501, 500), (6000, 600)):
            with self.subTest(patch_count=patch_count):
                self.assertEqual(
                    select_memory_count(patch_count, memory_policy="official_minimum"),
                    expected,
                )
        with self.assertRaisesRegex(ValueError, "3"):
            select_memory_count(3, memory_policy="official_minimum")

    def test_paper_memory_fraction_uses_source_rounding_without_minimum(self):
        for patch_count, expected in ((26, 3), (35, 4), (45, 4), (1000, 100), (6000, 600)):
            with self.subTest(patch_count=patch_count):
                self.assertEqual(
                    select_memory_count(patch_count, memory_policy="paper_fraction"), expected,
                )
        self.assertEqual(select_memory_count(4, fraction=1, memory_policy="paper_fraction"), 3)
        with self.assertRaisesRegex(ValueError, "3"):
            select_memory_count(25, memory_policy="paper_fraction")

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

    def test_official_loss_history_identifies_the_first_best_checkpoint_iteration(self):
        class CountingEncoder(TinyEncoder):
            def __init__(self):
                super().__init__()
                self.register_buffer("iterations_seen", torch.zeros((), dtype=torch.long))

            def embedding(self, patches):
                self.iterations_seen.add_(1)
                return super().embedding(patches)

        torch.manual_seed(4)
        model = CountingEncoder()
        with torch.no_grad():
            model.projection_head.weight.zero_()
            model.projection_head.bias.fill_(1)
        log = train_encoder(
            model, torch.randn(30, 1, 2), iterations=6, batch_size=30,
            learning_rate=0.0, weight_decay=0.0, device="cpu", official_procedure=True,
        )

        history = log["loss_history"]
        self.assertEqual([row["iteration"] for row in history], [1, 2, 3, 4, 5, 6])
        self.assertEqual(log["selected_iteration"], 2)
        self.assertEqual(model.iterations_seen.item(), 2)
        self.assertEqual(log["best_training_loss"], history[1]["total_loss"])
        self.assertGreater(history[0]["total_loss"], history[1]["total_loss"])
        self.assertAlmostEqual(history[0]["pretext_weight"], 1 / 6)
        for row in history:
            self.assertAlmostEqual(
                row["total_loss"], row["triplet_loss"] + row["pretext_weight"] * row["pretext_loss"],
                places=6,
            )
            self.assertEqual(row["learning_rate"], 0.0)
        for row in history[1:]:
            self.assertEqual(row["pretext_weight"], 0.0)
            self.assertEqual(row["pretext_loss"], 0.0)
            self.assertEqual(row["total_loss"], history[1]["total_loss"])


class TestPaAnoAdapter(unittest.TestCase):
    def test_patch_tensor_keeps_a_view_instead_of_copying_all_windows(self):
        values = numpy.arange(60, dtype=numpy.float32).reshape(20, 3)
        patches = make_patch_tensor(values, 4)

        self.assertEqual(patches.untyped_storage().data_ptr(), values.ctypes.data)
        numpy.testing.assert_array_equal(patches[2].numpy(), values[2:6].T)

    def test_overlap_scores_match_source_nonfinite_replacement_and_float32(self):
        scores = stitch_patch_scores([float("nan"), float("inf"), 3.0], 2, 4)

        self.assertEqual(scores.dtype, numpy.float32)
        numpy.testing.assert_array_equal(scores, [0.0, 0.0, 1.5, 3.0])

    def test_official_initialization_follows_the_source_loader_rng_consumption(self):
        from torch.utils.data import DataLoader, TensorDataset

        fit = numpy.arange(20, dtype=numpy.float32).reshape(10, 2)
        torch.manual_seed(7)
        source_loader = DataLoader(TensorDataset(torch.zeros(9, 2, 2)), batch_size=512, shuffle=True)
        next(iter(source_loader))
        expected_state = torch.get_rng_state().clone()
        initialization_states = []

        def build_encoder(_channels):
            initialization_states.append(torch.get_rng_state().clone())
            return MeanEncoder()

        adapter = PaAnoAdapter(
            patch_size=2, learning_rate=1e-4, device="cpu", official_procedure=True,
            encoder_factory=build_encoder, trainer=RecordingTrainer(),
        )
        torch.manual_seed(7)
        with patch("src.models.tier2.paano.adapter.encode_patches", return_value=torch.zeros((9, 2))), patch(
            "src.models.tier2.paano.adapter.select_memory_bank", return_value=torch.zeros((3, 2)),
        ):
            adapter.fit(fit)
        self.assertEqual(len(initialization_states), 1)
        torch.testing.assert_close(initialization_states[0], expected_state)

    def test_official_fit_rejects_an_encoder_initialized_before_loader_warmup(self):
        adapter = PaAnoAdapter(
            patch_size=2, learning_rate=1e-4, device="cpu", official_procedure=True,
            encoder_factory=lambda _channels: MeanEncoder(), trainer=RecordingTrainer(),
        )
        adapter.prepare_model(2)
        with self.assertRaisesRegex(ValueError, "loader warmup"):
            adapter.fit(numpy.zeros((10, 2), dtype=numpy.float32))

    def test_official_checkpoint_restores_cross_boundary_context_and_native_score(self):
        adapter = PaAnoAdapter(
            patch_size=2, learning_rate=1e-4, device="cpu",
            official_procedure=True,
            encoder_factory=lambda _channels: MeanEncoder(), trainer=RecordingTrainer(),
        )
        fit = numpy.arange(62, dtype=numpy.float32).reshape(31, 2)
        with patch("src.models.tier2.paano.adapter.encode_patches", return_value=torch.zeros((30, 2))) as encoder, patch(
            "src.models.tier2.paano.adapter.select_memory_bank", return_value=torch.zeros((29, 2)),
        ) as memory_selector:
            log = adapter.fit(fit)
        self.assertEqual(memory_selector.call_args.kwargs["memory_policy"], "official_minimum")
        self.assertEqual(log["memory_policy"], "official_minimum")
        self.assertEqual(log["memory_count"], 29)
        self.assertTrue(encoder.call_args.kwargs["shuffle"])
        self.assertEqual(encoder.call_args.kwargs["batch_size"], 512)
        self.assertEqual(log["effective_batch_size"], 512)
        restored = PaAnoAdapter.from_checkpoint(
            adapter.checkpoint(), device="cpu", encoder_factory=lambda _channels: MeanEncoder(),
        )
        self.assertEqual(restored.memory_policy, "official_minimum")
        test = numpy.zeros((3, 2), dtype=numpy.float32)
        with patch(
            "src.models.tier2.paano.adapter.score_patches", return_value=torch.tensor([1.0, 2.0, 3.0]),
        ) as scorer:
            output = restored.score(test)
        numpy.testing.assert_array_equal(scorer.call_args.args[1][0, :, 0].numpy(), fit[-1])
        self.assertEqual(scorer.call_args.kwargs["batch_size"], 512)
        self.assertEqual(scorer.call_args.kwargs["device"], "cpu")
        numpy.testing.assert_allclose(output["scores"], [1.5, 2.5, 3.0])
        self.assertEqual(output["source_end_exclusive"], 3)
        self.assertTrue(output["native_postprocessing"])
        self.assertEqual(output["calibration_mode"], "none")
        self.assertEqual(output["training_context_rows"], 1)

    def test_official_path_rejects_reached_singleton_without_changing_batch(self):
        trainer = RecordingTrainer()
        adapter = PaAnoAdapter(
            patch_size=2, learning_rate=1e-4, device="cpu", batch_size=4,
            official_procedure=True, encoder_factory=lambda _channels: MeanEncoder(),
            trainer=trainer,
        )
        with self.assertRaisesRegex(ValueError, "singleton"):
            adapter.fit(numpy.zeros((10, 2), dtype=numpy.float32))
        self.assertIsNone(trainer.fit_patches)

    def test_checkpoint_keeps_default_minimum_and_explicit_historical_fraction(self):
        values = numpy.arange(62, dtype=numpy.float32).reshape(31, 2)
        for policy, expected_policy, memory_count in (
            (None, "official_minimum", 29), ("paper_fraction", "paper_fraction", 3),
        ):
            with self.subTest(policy=policy):
                adapter = PaAnoAdapter(
                    patch_size=2, learning_rate=1e-4, device="cpu",
                    memory_policy=policy, full_prefix=True, official_procedure=True,
                    encoder_factory=lambda _channels: MeanEncoder(),
                    trainer=RecordingTrainer(),
                )
                adapter.fit(values)
                self.assertEqual(len(adapter.memory_bank), memory_count)
                restored = PaAnoAdapter.from_checkpoint(
                    adapter.checkpoint(), device="cpu",
                    encoder_factory=lambda _channels: MeanEncoder(),
                )
                self.assertEqual(restored.memory_policy, expected_policy)
                self.assertEqual(len(restored.memory_bank), memory_count)
                self.assertTrue(restored.full_prefix)
                output = restored.score(values)
                self.assertEqual(output["calibration_mode"], "none")
                self.assertEqual(output["normalization_scope"], "none")

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
