"""Strict target-free TimeRCD anomaly-head adapter."""

import hashlib
import math
from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import Callable

import numpy
import torch
import torch.nn.functional as functional


TIME_RCD_SOURCE_COMMIT = "372bb980426b2f67007311c6f3165ab789c79bef"
TIME_RCD_MODEL_NAME = "thu-sail-lab/Time-RCD"
TIME_RCD_CHECKPOINT_REVISION = "0880420070a524efe14d7e163b34dcca90d2be1b"
TIME_RCD_CHECKPOINT_FILE = "best_model/pretrain_checkpoint_best_multi.pth"
TIME_RCD_CHECKPOINT_SHA256 = (
    "a497ac910f24f8fb347d690a21b60d74bc04785ebc777880553c158dea4434d0"
)
TIME_RCD_CONFIG_FILE = "config.json"
TIME_RCD_CONFIG_SHA256 = (
    "5d576f85660cc62268566f6281f5840e505672606b51bbb6c93e397a40ebc11f"
)
TIME_RCD_CONTEXT_LENGTH = 5000
TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE = 64


def verify_time_rcd_checkpoint(
    checkpoint_path, *, expected_sha256=TIME_RCD_CHECKPOINT_SHA256,
):
    """Return the checkpoint digest after matching the sealed identity."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"TimeRCD checkpoint not found: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"TimeRCD checkpoint SHA-256 mismatch: {digest} != {expected_sha256}",
        )
    return digest


def _download_from_hub(*, repo_id, filename, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def build_time_rcd_config(default_config, *, channel_count):
    """공식 multi-checkpoint inference 설정을 독립 복사본에 적용한다."""
    if not isinstance(channel_count, int) or channel_count < 1:
        raise ValueError("channel_count must be a positive integer")
    config = deepcopy(default_config)
    config.ts_config.patch_size = 16
    config.win_size = TIME_RCD_CONTEXT_LENGTH
    config.batch_size = 1
    config.ts_config.num_features = channel_count
    return config


def load_time_rcd_model(
    checkpoint_path, config, *, tester_class=None,
    expected_sha256=TIME_RCD_CHECKPOINT_SHA256, device="cpu",
):
    """Load only the official model; scoring bypasses its normalizing dataset."""
    verify_time_rcd_checkpoint(
        checkpoint_path, expected_sha256=expected_sha256,
    )
    if tester_class is None:
        try:
            from time_rcd._inference import TimeRCDPretrainTester
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Install the pinned Time-RCD source before loading TimeRCD",
            ) from error
        tester_class = TimeRCDPretrainTester
    resolved_device = _resolve_device(device)
    tester = tester_class(str(checkpoint_path), config)
    tester.device = resolved_device
    return tester.model.to(resolved_device).eval()


def load_time_rcd_components(
    *, channel_count, device="cpu", hub_download=None,
    file_verifier=verify_time_rcd_checkpoint, tester_class=None,
    default_config=None,
):
    """봉인 파일을 검증하고 공식 multi tester를 요청 device에 올린다."""
    downloader = hub_download or _download_from_hub
    checkpoint_path = downloader(
        repo_id=TIME_RCD_MODEL_NAME, filename=TIME_RCD_CHECKPOINT_FILE,
        revision=TIME_RCD_CHECKPOINT_REVISION,
    )
    config_path = downloader(
        repo_id=TIME_RCD_MODEL_NAME, filename=TIME_RCD_CONFIG_FILE,
        revision=TIME_RCD_CHECKPOINT_REVISION,
    )
    file_verifier(checkpoint_path, expected_sha256=TIME_RCD_CHECKPOINT_SHA256)
    file_verifier(config_path, expected_sha256=TIME_RCD_CONFIG_SHA256)
    if default_config is None or tester_class is None:
        try:
            from time_rcd._core.time_rcd_config import default_config as official_config
            from time_rcd._inference import TimeRCDPretrainTester
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Install the pinned Time-RCD source before loading TimeRCD",
            ) from error
        default_config = default_config or official_config
        tester_class = tester_class or TimeRCDPretrainTester
    config = build_time_rcd_config(default_config, channel_count=channel_count)
    resolved_device = _resolve_device(device)
    tester = tester_class(str(checkpoint_path), config)
    tester.device = resolved_device
    model = tester.model.to(resolved_device).eval()
    return tester, model


def _chunked_attention_forward(
    self, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None,
):
    """Run TimeRCD's binary-bias attention one query slice at a time."""
    batch_size, token_count, embedding_size = query.shape
    assert key.shape == (batch_size, token_count, embedding_size)
    assert value.shape == (batch_size, token_count, embedding_size)
    query_rotated = self.apply_rope(self.q_proj(query), freqs)
    key_rotated = self.apply_rope(self.k_proj(key), freqs)
    query_rotated = query_rotated.view(
        batch_size, token_count, self.num_heads, self.head_dim,
    ).transpose(1, 2)
    key_rotated = key_rotated.view(
        batch_size, token_count, self.num_heads, self.head_dim,
    ).transpose(1, 2)
    projected_values = self.v_proj(value).view(
        batch_size, token_count, self.num_heads, self.head_dim,
    ).transpose(1, 2)

    if query_id is None or kv_id is None:
        attended = functional.scaled_dot_product_attention(
            query_rotated, key_rotated, projected_values,
            attn_mask=(attn_mask[:, None, None, :] if attn_mask is not None else None),
            is_causal=False,
        )
    else:
        key_mask = attn_mask[:, None, None, :] if attn_mask is not None else None
        attended_chunks = []
        for start in range(0, token_count, self._time_rcd_query_chunk_size):
            stop = min(start + self._time_rcd_query_chunk_size, token_count)
            scores = query_rotated[:, :, start:stop] @ key_rotated.transpose(-2, -1)
            scores = scores / math.sqrt(self.head_dim)
            scores = scores + self.binary_attention_bias(query_id[:, start:stop], kv_id)
            if key_mask is not None:
                scores = scores.masked_fill(~key_mask, float("-inf"))
            attended_chunks.append(torch.softmax(scores, dim=-1) @ projected_values)
        attended = torch.cat(attended_chunks, dim=-2)

    output = attended.transpose(1, 2).contiguous().view(
        batch_size, token_count, embedding_size,
    )
    return self.out_proj(output)


def install_time_rcd_chunked_attention(
    model, *, query_chunk_size=TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE,
    attention_class=None,
):
    """Patch loaded official TimeRCD attention instances without editing its source."""
    if not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    if attention_class is None:
        try:
            from time_rcd._core.ts_encoder_bi_bias import MultiheadAttentionWithRoPE
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Install the pinned Time-RCD source before patching attention",
            ) from error
        attention_class = MultiheadAttentionWithRoPE
    patched = 0
    for module in model.modules():
        if isinstance(module, attention_class):
            module._time_rcd_query_chunk_size = query_chunk_size
            module.forward = MethodType(_chunked_attention_forward, module)
            patched += 1
    if patched == 0:
        raise ValueError("no MultiheadAttentionWithRoPE module found")
    return patched


def build_time_rcd_official_scorer(
    *, channel_count, context_length=TIME_RCD_CONTEXT_LENGTH,
    query_chunk_size=TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE, device="cpu",
    component_loader=load_time_rcd_components,
) -> Callable:
    """Load and patch one official model, then return its session scorer."""
    _, model = component_loader(channel_count=channel_count, device=device)
    install_time_rcd_chunked_attention(model, query_chunk_size=query_chunk_size)

    def scorer(session):
        return score_time_rcd(
            model, session, context_length=context_length, device=device,
        )

    return scorer


def get_time_rcd_status(
    *, source_smoke_passed=False, checkpoint_smoke_passed=False,
    finite_scores_verified=False, nonconstant_scores_verified=False,
):
    """Open availability only after complete no-normalization checkpoint evidence."""
    del source_smoke_passed
    if (
        checkpoint_smoke_passed
        and finite_scores_verified
        and nonconstant_scores_verified
    ):
        return {"status": "available", "normalization_scope": "none"}
    return {
        "status": "unavailable",
        "reason": "checkpoint smoke with finite nonconstant scores is incomplete",
    }


def _validate_session(session):
    values = numpy.asarray(session, dtype=numpy.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("TimeRCD input must be a nonempty time-by-channel matrix")
    if not numpy.isfinite(values).all():
        raise ValueError("TimeRCD input must contain only finite values")
    return values


def _resolve_device(device):
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TimeRCD requires an available CUDA device")
    return resolved


def _extract_logits(model, time_series, mask):
    output = model(time_series=time_series, mask=mask)
    if not isinstance(output, torch.Tensor) or output.ndim != 4 or output.shape[-1] != 2:
        if not hasattr(model, "anomaly_head"):
            raise RuntimeError("TimeRCD model did not expose anomaly-head logits")
        output = model.anomaly_head(output)
    if output.ndim != 4 or output.shape[:3] != time_series.shape or output.shape[-1] != 2:
        raise RuntimeError(
            "TimeRCD logits must have shape (batch,time,channel,2)",
        )
    return output


def score_time_rcd(
    model, session, *, context_length=TIME_RCD_CONTEXT_LENGTH, device="cpu",
):
    """Score independent, unnormalized contexts with the official anomaly head."""
    values = _validate_session(session)
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    resolved_device = _resolve_device(device)
    model.to(resolved_device).eval()
    score_parts = []

    with torch.inference_mode():
        for start in range(0, len(values), context_length):
            fragment = values[start:start + context_length]
            valid_length = len(fragment)
            if len(values) > context_length and valid_length < context_length:
                fragment = numpy.concatenate([
                    fragment,
                    numpy.repeat(fragment[-1:], context_length - valid_length, axis=0),
                ])
            time_series = torch.from_numpy(fragment.copy()).unsqueeze(0).to(resolved_device)
            mask = torch.zeros((1, len(fragment)), dtype=torch.bool, device=resolved_device)
            mask[:, :valid_length] = True
            logits = _extract_logits(model, time_series, mask)
            channel_mean_logits = logits.mean(dim=-2)
            probabilities = torch.softmax(channel_mean_logits, dim=-1)[..., 1]
            score_parts.append(probabilities[0, :valid_length].cpu().numpy())

    scores = numpy.concatenate(score_parts).astype(numpy.float64, copy=False)
    if not numpy.isfinite(scores).all():
        raise RuntimeError("TimeRCD produced nonfinite scores")
    return {
        "scores": scores,
        "source_start": 0,
        "source_end_exclusive": len(values),
        "alignment": "same_timestep",
        "primitive": "probability",
        "calibration_mode": "none",
        "normalization_scope": "none",
        "evaluation_mode": "offline_noncausal",
        "lookahead": min(context_length, len(values)) - 1,
        "maximum_effective_lookahead": min(context_length, len(values)) - 1,
    }


def score_time_rcd_official(
    session, *, context_length=TIME_RCD_CONTEXT_LENGTH, device="cpu",
    query_chunk_size=TIME_RCD_ATTENTION_QUERY_CHUNK_SIZE,
    component_loader=load_time_rcd_components,
):
    """봉인 download·config·model load와 strict 점수 생성을 잇는다."""
    values = _validate_session(session)
    return build_time_rcd_official_scorer(
        channel_count=values.shape[1], context_length=context_length,
        query_chunk_size=query_chunk_size, device=device,
        component_loader=component_loader,
    )(values)
