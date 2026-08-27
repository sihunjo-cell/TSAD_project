"""Leakage-free project controller for the official ALoRa score core."""

import copy
import time

import numpy
import torch
from scipy.stats import rankdata

from src.common.validate_multivariate_session import as_finite_multivariate_session

from .official import (
    ATTENTION_RANK_THRESHOLD,
    ALoRaT,
    calculate_attention_rank,
    calculate_low_rank_loss,
)


SOURCE_COMMIT = "97dcc4a337710e6dc72c1a67893717c9538bae1a"


def select_spearman_pairs(fit_values, top_k: int = 512) -> numpy.ndarray:
    """Select fit-normal channel pairs by absolute Spearman correlation."""
    fit = as_finite_multivariate_session(fit_values, "fit_values")
    if len(fit) < 2:
        raise ValueError("ALoRa Spearman selection needs at least two fit rows")
    if top_k < 1:
        raise ValueError("ALoRa top_k must be positive")
    ranked = numpy.apply_along_axis(rankdata, 0, fit)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        standardized = (
            ranked - ranked.mean(axis=0)
        ) / ranked.std(axis=0, ddof=1)
        correlations = standardized.T @ standardized / (len(ranked) - 1)
    left, right = numpy.triu_indices(fit.shape[1], k=1)
    pair_correlations = correlations[left, right]
    missing = numpy.isnan(pair_correlations)
    descending_magnitude = -numpy.where(missing, -numpy.inf, numpy.abs(pair_correlations))
    order = numpy.lexsort((right, left, descending_magnitude, missing.astype(int)))
    count = min(top_k, len(order))
    return numpy.column_stack((left[order[:count]], right[order[:count]])).astype(
        numpy.int64,
    )


def _make_windows(values, window_size: int) -> numpy.ndarray:
    session = as_finite_multivariate_session(values, "session")
    if window_size < 1 or len(session) < window_size:
        raise ValueError("ALoRa session is shorter than window_size")
    return numpy.stack([
        session[start : start + window_size]
        for start in range(len(session) - window_size + 1)
    ])


def stitch_alora_scores(
    reconstruction_mse,
    attention_ranks,
) -> numpy.ndarray:
    errors = numpy.asarray(reconstruction_mse, dtype=numpy.float64)
    ranks = numpy.asarray(attention_ranks, dtype=numpy.float64)
    if errors.ndim != 2 or ranks.shape != (len(errors),) or len(errors) == 0:
        raise ValueError("ALoRa window scores have incompatible shapes")
    weighted = errors * ranks[:, None]
    return numpy.concatenate((weighted[0], weighted[1:, -1]))


def build_alora_model(
    fit_sessions,
    *,
    window_size: int,
    device: str,
    pair_embedding_dimension: int = 512,
    attention_heads: int = 8,
    encoder_layers: int = 3,
):
    sessions = tuple(
        as_finite_multivariate_session(values, "fit_session")
        for values in fit_sessions
    )
    if not sessions:
        raise ValueError("ALoRa needs at least one fit session")
    channel_count = sessions[0].shape[1]
    if any(session.shape[1] != channel_count for session in sessions):
        raise ValueError("ALoRa fit sessions have different channel counts")
    selected_pairs = select_spearman_pairs(
        numpy.concatenate(sessions, axis=0),
        top_k=pair_embedding_dimension,
    )
    model = ALoRaT(
        window_size=window_size,
        channel_count=channel_count,
        output_channels=channel_count,
        selected_pairs=selected_pairs,
        pair_embedding_dimension=pair_embedding_dimension,
        attention_heads=attention_heads,
        encoder_layers=encoder_layers,
    ).to(device)
    recipe = {
        "selected_pairs": selected_pairs.tolist(),
        "rank_threshold": ATTENTION_RANK_THRESHOLD,
        "window_size": window_size,
        "channel_count": channel_count,
        "pair_embedding_dimension": len(selected_pairs),
        "attention_heads": attention_heads,
        "encoder_layers": encoder_layers,
    }
    return model, recipe


def score_alora_sessions(
    model,
    sessions,
    *,
    window_size: int,
    device: str,
    batch_size: int = 256,
):
    if batch_size < 1:
        raise ValueError("ALoRa batch_size must be positive")
    outputs = []
    model.eval()
    with torch.no_grad():
        for values in sessions:
            session = as_finite_multivariate_session(values, "score_session")
            windows = _make_windows(session, window_size)
            reconstruction_errors = []
            attention_ranks = []
            for start in range(0, len(windows), batch_size):
                inputs = torch.as_tensor(
                    windows[start : start + batch_size], device=device,
                )
                reconstruction, attentions, _ = model(inputs)
                reconstruction_errors.append(
                    torch.mean((reconstruction - inputs) ** 2, dim=-1).cpu().numpy(),
                )
                attention_ranks.append(
                    calculate_attention_rank(attentions[-1]).cpu().numpy(),
                )
            scores = stitch_alora_scores(
                numpy.concatenate(reconstruction_errors),
                numpy.concatenate(attention_ranks),
            )
            outputs.append({
                "scores": scores,
                "source_start": 0,
                "source_end_exclusive": len(session),
                "alignment": "same_timestep",
                "primitive": "reconstruction_mse_times_attention_rank",
                "calibration_mode": "validation_median_iqr",
                "evaluation_mode": "offline_noncausal",
                "lookahead": 0,
                "maximum_effective_lookahead": window_size - 1,
                "normalization_scope": "current_prefix_validation",
            })
    return tuple(outputs)


def train_alora(
    fit_sessions,
    validation_sessions,
    *,
    window_size: int,
    device: str,
    epochs: int,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    low_rank_weight: float = 10.0,
    pair_embedding_dimension: int = 512,
    attention_heads: int = 8,
    encoder_layers: int = 3,
    patience: int = 3,
):
    """Train without labels and select the checkpoint on normal validation MSE."""
    if epochs < 1 or patience < 1:
        raise ValueError("ALoRa epochs and patience must be positive")
    model, recipe = build_alora_model(
        fit_sessions,
        window_size=window_size,
        device=device,
        pair_embedding_dimension=pair_embedding_dimension,
        attention_heads=attention_heads,
        encoder_layers=encoder_layers,
    )
    fit_windows = numpy.concatenate([
        _make_windows(session, window_size) for session in fit_sessions
    ])
    validation_windows = numpy.concatenate([
        _make_windows(session, window_size) for session in validation_sessions
    ])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    updates = 0
    completed_epochs = 0
    training_examples_seen = 0

    for epoch in range(epochs):
        model.train()
        for start in range(0, len(fit_windows), batch_size):
            inputs = torch.as_tensor(
                fit_windows[start : start + batch_size], device=device,
            )
            optimizer.zero_grad()
            reconstruction, attentions, _ = model(inputs)
            reconstruction_loss = torch.mean((reconstruction - inputs) ** 2)
            loss = reconstruction_loss + low_rank_weight * calculate_low_rank_loss(attentions)
            loss.backward()
            optimizer.step()
            updates += 1
            training_examples_seen += len(inputs)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for start in range(0, len(validation_windows), batch_size):
                inputs = torch.as_tensor(
                    validation_windows[start : start + batch_size], device=device,
                )
                reconstruction, _, _ = model(inputs)
                validation_loss_sum += torch.sum((reconstruction - inputs) ** 2).item()
                validation_count += inputs.numel()
        validation_loss = validation_loss_sum / validation_count
        completed_epochs = epoch + 1
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("ALoRa did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, recipe, {
        "epochs_completed": completed_epochs,
        "optimizer_updates": updates,
        "training_examples_seen": training_examples_seen,
        "best_validation_loss": best_loss,
        "stopped_early": completed_epochs < epochs,
    }


def run_alora_sessions(
    fit_sessions,
    validation_sessions,
    test_sessions,
    *,
    window_size: int,
    device: str,
    epochs: int,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    low_rank_weight: float = 10.0,
    pair_embedding_dimension: int = 512,
    attention_heads: int = 8,
    encoder_layers: int = 3,
    patience: int = 3,
    trainer=train_alora,
):
    training_started = time.perf_counter()
    model, recipe, training_log = trainer(
        fit_sessions,
        validation_sessions,
        window_size=window_size,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        low_rank_weight=low_rank_weight,
        pair_embedding_dimension=pair_embedding_dimension,
        attention_heads=attention_heads,
        encoder_layers=encoder_layers,
        patience=patience,
    )
    training_seconds = time.perf_counter() - training_started
    validation_started = time.perf_counter()
    validation_outputs = score_alora_sessions(
        model,
        validation_sessions,
        window_size=window_size,
        device=device,
        batch_size=batch_size,
    )
    validation_seconds = time.perf_counter() - validation_started
    test_started = time.perf_counter()
    test_outputs = score_alora_sessions(
        model,
        test_sessions,
        window_size=window_size,
        device=device,
        batch_size=batch_size,
    )
    return {
        "checkpoint": {
            "model_state_dict": model.state_dict(),
            "model_config": recipe,
            "source_commit": SOURCE_COMMIT,
        },
        "validation_outputs": validation_outputs,
        "test_outputs": test_outputs,
        "training_log": training_log,
        "timing": {
            "training_seconds": training_seconds,
            "validation_inference_seconds": validation_seconds,
            "test_inference_seconds": time.perf_counter() - test_started,
            "accelerator": device,
        },
    }
