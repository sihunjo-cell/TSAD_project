"""Strict target-free TSPulse raw-head adapter."""

import hashlib
from pathlib import Path

import numpy
import torch


TSPULSE_SOURCE_COMMIT = "9739fa59b61bd9f15cbfb06e5dc3dab28c72ee8d"
TSPULSE_MODEL_NAME = "ibm-granite/granite-timeseries-tspulse-r1"
TSPULSE_CHECKPOINT_REVISION = "2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2"
TSPULSE_CHECKPOINT_FILE = "model.safetensors"
TSPULSE_CHECKPOINT_SHA256 = (
    "57fa03b67d1473a7253ac37801b24c391d1fa931395db986743e2f59e556b9ac"
)
TSPULSE_CONFIG_FILE = "config.json"
TSPULSE_CONFIG_SHA256 = (
    "e33230f47c91dfe4e848ee743924a62a26f710fe7f194566d92835f27c1c6d8b"
)
TSPULSE_CONTEXT_LENGTH = 512
TSPULSE_PATCH_SIZE = 8
TSPULSE_AGGREGATION_WINDOWS = (64, 96, 128)
TSPULSE_HEADS = ("time", "fft", "pred")
TSPULSE_OFFICIAL_HEADS = ("time", "fft", "forecast")


def verify_tspulse_checkpoint(
    checkpoint_path, *, expected_sha256=TSPULSE_CHECKPOINT_SHA256,
):
    """Return the checkpoint digest after matching the sealed identity."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"TSPulse checkpoint not found: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"TSPulse checkpoint SHA-256 mismatch: {digest} != {expected_sha256}",
        )
    return digest


def _download_from_hub(*, repo_id, filename, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def load_tspulse_components(
    *, aggregation_window, channel_count, device="cpu", model_class=None,
    utility_class=None, hub_download=None,
    file_verifier=verify_tspulse_checkpoint,
):
    """Lazily load the pinned reconstruction model and its raw-score utility."""
    if not isinstance(channel_count, int) or channel_count < 1:
        raise ValueError("channel_count must be a positive integer")
    downloader = hub_download or _download_from_hub
    checkpoint_path = downloader(
        repo_id=TSPULSE_MODEL_NAME, filename=TSPULSE_CHECKPOINT_FILE,
        revision=TSPULSE_CHECKPOINT_REVISION,
    )
    config_path = downloader(
        repo_id=TSPULSE_MODEL_NAME, filename=TSPULSE_CONFIG_FILE,
        revision=TSPULSE_CHECKPOINT_REVISION,
    )
    file_verifier(checkpoint_path, expected_sha256=TSPULSE_CHECKPOINT_SHA256)
    file_verifier(config_path, expected_sha256=TSPULSE_CONFIG_SHA256)
    checkpoint_path = Path(checkpoint_path).absolute()
    config_path = Path(config_path).absolute()
    if checkpoint_path.parent != config_path.parent:
        raise ValueError("TSPulse files must share the same snapshot directory")
    snapshot_directory = checkpoint_path.parent
    if model_class is None:
        try:
            from tsfm_public.models.tspulse import TSPulseForReconstruction
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Install the pinned granite-tsfm source before loading TSPulse",
            ) from error
        model_class = TSPulseForReconstruction
    if utility_class is None:
        from tsfm_public.models.tspulse.utils.ad_helpers import TSPulseADUtility
        utility_class = TSPulseADUtility

    model = model_class.from_pretrained(
        str(snapshot_directory), revision=TSPULSE_CHECKPOINT_REVISION,
        num_input_channels=channel_count, mask_type="user",
    ).to(device).eval()
    utility = utility_class(
        model, mode=list(TSPULSE_OFFICIAL_HEADS),
        aggregation_length=aggregation_window,
    )
    return model, utility


def build_tspulse_raw_head_function(
    utility, *, aggregation_window, context_length=TSPULSE_CONTEXT_LENGTH,
    batch_size=128, device="cpu",
):
    """공식 compute_score만 호출하는 stride-1 raw-head 함수를 만든다."""
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TSPulse requires an available CUDA device")

    def compute_raw_heads(session, *, aggregation_window: int):
        if aggregation_window != aggregation_window_value:
            raise ValueError("TSPulse aggregation_window changed after utility creation")
        values = _validate_session(session)
        raw_length = len(values) - context_length
        if raw_length < 1:
            raise ValueError("TSPulse input must be longer than its context")
        accumulated = {head: [] for head in TSPULSE_OFFICIAL_HEADS}
        with torch.inference_mode():
            for batch_start in range(0, raw_length, batch_size):
                starts = range(batch_start, min(raw_length, batch_start + batch_size))
                past_values = numpy.stack([
                    values[start:start + context_length] for start in starts
                ])
                starts = range(batch_start, min(raw_length, batch_start + batch_size))
                future_values = numpy.stack([
                    values[start + context_length:start + context_length + 1]
                    for start in starts
                ])
                payload = {
                    "past_values": torch.from_numpy(past_values).to(resolved_device),
                    "future_values": torch.from_numpy(future_values).to(resolved_device),
                }
                scores = utility.compute_score(
                    payload, mode=list(TSPULSE_OFFICIAL_HEADS), expand_score=False,
                )
                missing = set(TSPULSE_OFFICIAL_HEADS) - scores.keys()
                if missing:
                    raise RuntimeError(f"TSPulse utility omitted raw heads: {sorted(missing)}")
                for head in TSPULSE_OFFICIAL_HEADS:
                    values_batch = scores[head]
                    if not isinstance(values_batch, torch.Tensor):
                        values_batch = torch.as_tensor(values_batch)
                    accumulated[head].append(
                        values_batch.detach().cpu().numpy().reshape(-1)
                    )
        raw_heads = {
            head: numpy.concatenate(parts).astype(numpy.float64, copy=False)
            for head, parts in accumulated.items()
        }
        if any(len(scores) != raw_length for scores in raw_heads.values()):
            raise RuntimeError("TSPulse utility returned an unexpected raw score length")
        return {
            "time": raw_heads["time"],
            "fft": raw_heads["fft"],
            "pred": raw_heads["forecast"],
        }

    aggregation_window_value = aggregation_window
    return compute_raw_heads


def align_tspulse_scores(
    raw_scores, *, head, context_length=TSPULSE_CONTEXT_LENGTH,
    aggregation_window,
):
    """Restore the official raw head boundary repeats without postprocessing."""
    scores = numpy.asarray(raw_scores, dtype=numpy.float64)
    if scores.ndim != 1 or len(scores) == 0 or not numpy.isfinite(scores).all():
        raise ValueError("raw TSPulse scores must be a nonempty finite vector")
    if head not in TSPULSE_HEADS:
        raise ValueError(f"unsupported TSPulse head: {head}")
    if not isinstance(context_length, int) or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    if (
        not isinstance(aggregation_window, int)
        or aggregation_window < 2
        or aggregation_window % 2
        or aggregation_window > 2 * context_length
    ):
        raise ValueError("aggregation_window must be a positive even boundary window")

    if head == "pred":
        left, right = context_length, 0
    else:
        half_window = aggregation_window // 2
        left, right = context_length - half_window, half_window
    if left < 0:
        raise ValueError("aggregation_window exceeds the supported context boundary")
    return numpy.pad(scores, (left, right), mode="edge")


def _validate_session(session):
    values = numpy.asarray(session, dtype=numpy.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("TSPulse input must be a nonempty time-by-channel matrix")
    if not numpy.isfinite(values).all():
        raise ValueError("TSPulse input must contain only finite values")
    return values


def _make_score_output(
    scores, primitive, *, native_source_start, native_source_end_exclusive,
    boundary_repeat, lookahead, maximum_effective_lookahead,
):
    return {
        "scores": scores,
        "source_start": 0,
        "source_end_exclusive": len(scores),
        "alignment": "boundary_repeat_from_native",
        "native_source_start": native_source_start,
        "native_source_end_exclusive": native_source_end_exclusive,
        "boundary_repeat": boundary_repeat,
        "lookahead": lookahead,
        "maximum_effective_lookahead": maximum_effective_lookahead,
        "evaluation_mode": "offline_noncausal",
        "primitive": primitive,
        "calibration_mode": "none",
        "normalization_scope": "model_revin_only",
    }


def score_tspulse(
    session, *, raw_head_function, aggregation_window,
    context_length=TSPULSE_CONTEXT_LENGTH,
):
    """Return aligned time, FFT, prediction and raw-head maximum scores."""
    values = _validate_session(session)
    if len(values) <= context_length:
        raise ValueError("TSPulse input must be longer than its context")
    raw_heads = raw_head_function(
        values.copy(), aggregation_window=aggregation_window,
    )
    missing = set(TSPULSE_HEADS) - raw_heads.keys()
    if missing:
        raise ValueError(f"TSPulse raw heads are missing: {sorted(missing)}")

    outputs = {}
    half_window = aggregation_window // 2
    for head in TSPULSE_HEADS:
        aligned = align_tspulse_scores(
            raw_heads[head], head=head, context_length=context_length,
            aggregation_window=aggregation_window,
        )
        if len(aligned) != len(values):
            raise ValueError(
                f"TSPulse {head} score length {len(aligned)} != input length {len(values)}",
            )
        if head == "pred":
            native_start, native_end = context_length, len(values)
            boundary_repeat = {"left": context_length, "right": 0}
            lookahead = 0
        else:
            native_start = context_length - half_window
            native_end = len(values) - half_window
            boundary_repeat = {"left": native_start, "right": half_window}
            lookahead = half_window
        outputs[head] = _make_score_output(
            aligned, f"raw_{head}_head_score",
            native_source_start=native_start,
            native_source_end_exclusive=native_end,
            boundary_repeat=boundary_repeat,
            lookahead=lookahead,
            maximum_effective_lookahead=(
                context_length if head == "pred" else context_length - 1
            ),
        )

    common_native_heads = (
        numpy.asarray(raw_heads["pred"], dtype=numpy.float64)[:-half_window],
        numpy.asarray(raw_heads["time"], dtype=numpy.float64)[half_window:],
        numpy.asarray(raw_heads["fft"], dtype=numpy.float64)[half_window:],
    )
    common_native_length = len(common_native_heads[0])
    if common_native_length < 1 or any(
        len(scores) != common_native_length for scores in common_native_heads
    ):
        raise ValueError("TSPulse raw heads have no valid common native range")
    raw_max = numpy.pad(
        numpy.maximum.reduce(common_native_heads),
        (context_length, half_window), mode="edge",
    )
    if len(raw_max) != len(values):
        raise ValueError("TSPulse raw_max length does not match the input")
    outputs["raw_max"] = _make_score_output(
        raw_max, "elementwise_max_raw_head_score",
        native_source_start=context_length,
        native_source_end_exclusive=len(values) - half_window,
        boundary_repeat={"left": context_length, "right": half_window},
        lookahead=half_window,
        maximum_effective_lookahead=context_length + half_window - 1,
    )
    return outputs


def score_tspulse_official(
    session, *, aggregation_window, context_length=TSPULSE_CONTEXT_LENGTH,
    batch_size=128, device="cpu", component_loader=load_tspulse_components,
):
    """고정 source의 raw utility를 strict 전처리 경로에 연결한다."""
    values = _validate_session(session)
    _, utility = component_loader(
        aggregation_window=aggregation_window, channel_count=values.shape[1],
        device=device,
    )
    raw_head_function = build_tspulse_raw_head_function(
        utility, aggregation_window=aggregation_window,
        context_length=context_length, batch_size=batch_size, device=device,
    )
    return score_tspulse(
        values, raw_head_function=raw_head_function,
        aggregation_window=aggregation_window, context_length=context_length,
    )
