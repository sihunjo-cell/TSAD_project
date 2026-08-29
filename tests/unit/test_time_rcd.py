"""Strict TimeRCD adapter contract tests."""

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy
import torch

from src.models.tier3.time_rcd import (
    TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE,
    TIME_RCD_CONFIG_FILE,
    TIME_RCD_CONFIG_SHA256,
    TIME_RCD_CHECKPOINT_SHA256,
    build_time_rcd_official_scorer,
    build_time_rcd_config,
    get_time_rcd_status,
    install_time_rcd_chunked_attention,
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


class FixtureBinaryAttentionBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emd = torch.nn.Embedding(2, 2)
        with torch.no_grad():
            self.emd.weight.copy_(torch.tensor([[-0.25, 0.5], [0.75, -0.5]]))

    def forward(self, query_id, kv_id):
        same_channel = query_id.unsqueeze(-1).eq(kv_id.unsqueeze(-2)).unsqueeze(1)
        weights = self.emd.weight[:, :, None, None]
        return torch.where(same_channel, weights[1], weights[0])


class FixtureAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 4
        self.num_heads = 2
        self.head_dim = 2
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.v_proj = torch.nn.Linear(4, 4, bias=False)
        self.out_proj = torch.nn.Linear(4, 4, bias=False)
        self.binary_attention_bias = FixtureBinaryAttentionBias()

        with torch.no_grad():
            for index, projection in enumerate((
                self.q_proj, self.k_proj, self.v_proj, self.out_proj,
            )):
                projection.weight.copy_(
                    torch.eye(4) + 0.1 * (index + 1) * torch.ones((4, 4)),
                )

    def apply_rope(self, values, freqs):
        batch_size, token_count, embedding_size = values.shape
        pairs = values.view(batch_size, token_count, embedding_size // 2, 2)
        cosine = freqs.cos().unsqueeze(0)
        sine = freqs.sin().unsqueeze(0)
        return torch.stack((
            pairs[..., 0] * cosine - pairs[..., 1] * sine,
            pairs[..., 0] * sine + pairs[..., 1] * cosine,
        ), dim=-1).view_as(values)

    def forward(self, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
        raise AssertionError("the chunked forward should replace this fixture method")


class FixtureAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FixtureAttention()


class OfficialScoringFixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FixtureAttention()
        self.calls = []

    def forward(self, *, time_series, mask):
        self.calls.append((time_series.detach().cpu(), mask.detach().cpu()))
        batch_size, time_count, channel_count = time_series.shape
        logits = torch.zeros(
            batch_size, time_count, channel_count, 2,
            dtype=time_series.dtype, device=time_series.device,
        )
        logits[..., 1] = math.log(3.0)
        return logits


def dense_attention_reference(attention, query, key, value, freqs, query_id, kv_id, attn_mask):
    batch_size, token_count, embedding_size = query.shape
    query_rotated = attention.apply_rope(attention.q_proj(query), freqs)
    key_rotated = attention.apply_rope(attention.k_proj(key), freqs)
    query_rotated = query_rotated.view(
        batch_size, token_count, attention.num_heads, attention.head_dim,
    ).transpose(1, 2)
    key_rotated = key_rotated.view(
        batch_size, token_count, attention.num_heads, attention.head_dim,
    ).transpose(1, 2)
    values = attention.v_proj(value).view(
        batch_size, token_count, attention.num_heads, attention.head_dim,
    ).transpose(1, 2)
    same_channel = query_id.unsqueeze(-1).eq(kv_id.unsqueeze(-2)).unsqueeze(1)
    bias_weights = attention.binary_attention_bias.emd.weight[:, :, None, None]
    bias = torch.where(same_channel, bias_weights[1], bias_weights[0])
    scores = query_rotated @ key_rotated.transpose(-2, -1) / math.sqrt(attention.head_dim)
    scores = scores + bias
    scores = scores.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
    attended = torch.softmax(scores, dim=-1) @ values
    return attention.out_proj(
        attended.transpose(1, 2).contiguous().view(batch_size, token_count, embedding_size),
    )


def install_fixture_attention_import():
    package = ModuleType("time_rcd")
    package.__path__ = []
    core = ModuleType("time_rcd._core")
    core.__path__ = []
    encoder = ModuleType("time_rcd._core.ts_encoder_bi_bias")
    encoder.MultiheadAttentionWithRoPE = FixtureAttention
    return patch.dict(sys.modules, {
        "time_rcd": package,
        "time_rcd._core": core,
        "time_rcd._core.ts_encoder_bi_bias": encoder,
    })


class TestTimeRCD(unittest.TestCase):
    def test_chunked_attention_matches_dense_global_attention(self):
        model = FixtureAttentionModel()
        attention = model.attention
        query = torch.tensor([[
            [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0],
        ]])
        freqs = torch.tensor([
            [0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9],
        ])
        query_id = torch.tensor([[0, 1, 0, 1, 0]])
        kv_id = torch.tensor([[1, 0, 1, 0, 1]])
        attn_mask = torch.tensor([[True, True, True, False, True]])
        dense = dense_attention_reference(
            attention, query, query, query, freqs, query_id, kv_id, attn_mask,
        )

        self.assertEqual(
            install_time_rcd_chunked_attention(
                model, query_chunk_size=2, attention_class=FixtureAttention,
            ),
            1,
        )
        chunked = attention(query, query, query, freqs, query_id, kv_id, attn_mask)

        torch.testing.assert_close(chunked, dense, rtol=1e-6, atol=1e-7)
        with self.assertRaisesRegex(ValueError, "no MultiheadAttentionWithRoPE"):
            install_time_rcd_chunked_attention(
                torch.nn.Identity(), attention_class=FixtureAttention,
            )

    def test_chunked_attention_keeps_official_no_id_attention_path(self):
        model = FixtureAttentionModel()
        attention = model.attention
        query = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4) / 10
        freqs = torch.tensor([
            [0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9],
        ])
        attn_mask = torch.tensor([[True, True, True, False, True]])
        query_rotated = attention.apply_rope(attention.q_proj(query), freqs)
        key_rotated = attention.apply_rope(attention.k_proj(query), freqs)
        query_rotated = query_rotated.view(1, 5, 2, 2).transpose(1, 2)
        key_rotated = key_rotated.view(1, 5, 2, 2).transpose(1, 2)
        values = attention.v_proj(query).view(1, 5, 2, 2).transpose(1, 2)
        dense = torch.nn.functional.scaled_dot_product_attention(
            query_rotated, key_rotated, values, attn_mask=attn_mask[:, None, None, :],
            is_causal=False,
        )
        dense = attention.out_proj(dense.transpose(1, 2).contiguous().view(1, 5, 4))

        install_time_rcd_chunked_attention(
            model, query_chunk_size=2, attention_class=FixtureAttention,
        )

        torch.testing.assert_close(
            attention(query, query, query, freqs, attn_mask=attn_mask),
            dense, rtol=1e-6, atol=1e-7,
        )

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

    def test_build_official_scorer_connects_loader_and_score(self):
        session = numpy.arange(6, dtype=numpy.float32).reshape(3, 2)
        model = OfficialScoringFixtureModel()

        def component_loader(**arguments):
            self.assertEqual(arguments, {"channel_count": 2, "device": "cpu"})
            return object(), model

        with install_fixture_attention_import():
            scorer = build_time_rcd_official_scorer(
                channel_count=2, context_length=4, device="cpu",
                component_loader=component_loader,
            )
            result = scorer(session)

        self.assertEqual(result["scores"].shape, (3,))
        self.assertEqual(model.calls[0][0].shape, (1, 3, 2))
        self.assertEqual(model.attention._time_rcd_query_chunk_size, TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE)

    def test_official_entrypoint_uses_prepared_scorer_contract(self):
        session = numpy.arange(6, dtype=numpy.float32).reshape(3, 2)
        model = OfficialScoringFixtureModel()

        def component_loader(**arguments):
            self.assertEqual(arguments, {"channel_count": 2, "device": "cpu"})
            return object(), model

        with install_fixture_attention_import():
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
