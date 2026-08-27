"""Project split, training, and continuous-score controller for official GDN."""

import copy
import math
import time

import numpy
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from src.common.validate_multivariate_session import as_finite_multivariate_session

from .official import GDN, SOURCE_COMMIT, build_fully_connected_edge_index

def topk_from_rho(channel_count: int, rho: float) -> int:
    if channel_count < 2 or not math.isfinite(rho) or rho <= 0:
        raise ValueError("GDN needs at least two channels and positive finite rho")
    return max(1, min(channel_count - 1, math.floor(rho * channel_count)))


def build_forecast_arrays(sessions, window_size: int = 5):
    """Build each session independently, then return channel-first windows."""
    if window_size < 1:
        raise ValueError("GDN window_size must be positive")
    arrays = []
    for values in sessions:
        session = as_finite_multivariate_session(values, "session")
        if len(session) <= window_size:
            raise ValueError("GDN session needs at least one next-step target")
        inputs = numpy.stack([
            session[start : start + window_size].T
            for start in range(len(session) - window_size)
        ])
        arrays.append((inputs, session[window_size:]))
    if not arrays:
        raise ValueError("GDN needs at least one session")
    return tuple(arrays)


def _make_dataset(sessions, window_size: int):
    arrays = build_forecast_arrays(sessions, window_size)
    return ConcatDataset([
        TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
        for inputs, targets in arrays
    ])


def build_gdn_model(
    channel_count: int,
    *,
    embedding_dimension: int,
    hidden_dimension: int,
    rho: float,
    window_size: int,
    device: str,
):
    topk = topk_from_rho(channel_count, rho)
    edge_index = build_fully_connected_edge_index(channel_count, device)
    model = GDN(
        [edge_index],
        node_num=channel_count,
        dim=embedding_dimension,
        out_layer_inter_dim=hidden_dimension,
        input_dim=window_size,
        out_layer_num=1,
        topk=topk,
    ).to(device)
    return model, edge_index, topk


def score_gdn_sessions(
    model,
    sessions,
    *,
    window_size: int,
    edge_index: torch.Tensor,
    device: str,
    batch_size: int = 128,
):
    if batch_size < 1:
        raise ValueError("GDN batch_size must be positive")
    edge_index = edge_index.to(device)
    arrays = build_forecast_arrays(sessions, window_size)
    outputs = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in arrays:
            score_batches = []
            for start in range(0, len(inputs), batch_size):
                input_batch = torch.as_tensor(
                    inputs[start : start + batch_size], device=device,
                )
                target_batch = torch.as_tensor(
                    targets[start : start + batch_size], device=device,
                )
                predictions = model(input_batch, edge_index)
                score_batches.append(torch.abs(predictions - target_batch).cpu().numpy())
            scores = numpy.concatenate(score_batches)
            source_end = len(targets) + window_size
            outputs.append({
                "scores": scores,
                "source_start": window_size,
                "source_end_exclusive": source_end,
                "alignment": "next_step",
                "primitive": "absolute_error",
                "calibration_mode": "validation_median_iqr",
                "evaluation_mode": "causal",
                "lookahead": 0,
                "maximum_effective_lookahead": 0,
                "normalization_scope": "current_prefix_validation",
            })
    return tuple(outputs)


def train_gdn(
    fit_sessions,
    validation_sessions,
    *,
    embedding_dimension: int,
    hidden_dimension: int,
    rho: float,
    device: str,
    window_size: int = 5,
    epochs: int = 30,
    patience: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
):
    """Train on fit windows and select a checkpoint on separate normal validation."""
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("GDN epochs, patience, and batch_size must be positive")
    fit_sessions = tuple(
        as_finite_multivariate_session(values, "fit_session")
        for values in fit_sessions
    )
    validation_sessions = tuple(
        as_finite_multivariate_session(values, "validation_session")
        for values in validation_sessions
    )
    if not fit_sessions or not validation_sessions:
        raise ValueError("GDN needs fit and validation sessions")
    channel_count = fit_sessions[0].shape[1]
    if any(session.shape[1] != channel_count for session in fit_sessions + validation_sessions):
        raise ValueError("GDN sessions have different channel counts")
    model, edge_index, topk = build_gdn_model(
        channel_count,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        rho=rho,
        window_size=window_size,
        device=device,
    )
    fit_loader = DataLoader(
        _make_dataset(fit_sessions, window_size),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        _make_dataset(validation_sessions, window_size),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    updates = 0
    completed_epochs = 0
    training_examples_seen = 0

    for epoch in range(epochs):
        model.train()
        for inputs, targets in fit_loader:
            optimizer.zero_grad()
            predictions = model(inputs.to(device), edge_index)
            loss = torch.mean((predictions - targets.to(device)) ** 2)
            loss.backward()
            optimizer.step()
            updates += 1
            training_examples_seen += len(inputs)

        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                predictions = model(inputs.to(device), edge_index)
                differences = predictions - targets.to(device)
                validation_sum += torch.sum(differences ** 2).item()
                validation_count += differences.numel()
        validation_loss = validation_sum / validation_count
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
        raise RuntimeError("GDN did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return model, edge_index, {
        "epochs_completed": completed_epochs,
        "optimizer_updates": updates,
        "training_examples_seen": training_examples_seen,
        "best_validation_loss": best_loss,
        "stopped_early": completed_epochs < epochs,
        "topk": topk,
    }


def run_gdn_sessions(
    fit_sessions,
    validation_sessions,
    test_sessions,
    *,
    embedding_dimension: int,
    hidden_dimension: int,
    rho: float,
    device: str,
    window_size: int = 5,
    epochs: int = 30,
    patience: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    trainer=train_gdn,
):
    training_started = time.perf_counter()
    model, edge_index, training_log = trainer(
        fit_sessions,
        validation_sessions,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        rho=rho,
        device=device,
        window_size=window_size,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    training_seconds = time.perf_counter() - training_started
    validation_started = time.perf_counter()
    validation_outputs = score_gdn_sessions(
        model,
        validation_sessions,
        window_size=window_size,
        edge_index=edge_index,
        device=device,
        batch_size=batch_size,
    )
    validation_seconds = time.perf_counter() - validation_started
    test_started = time.perf_counter()
    test_outputs = score_gdn_sessions(
        model,
        test_sessions,
        window_size=window_size,
        edge_index=edge_index,
        device=device,
        batch_size=batch_size,
    )
    channel_count = as_finite_multivariate_session(
        fit_sessions[0], "fit_session",
    ).shape[1]
    return {
        "checkpoint": {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "channel_count": channel_count,
                "embedding_dimension": embedding_dimension,
                "hidden_dimension": hidden_dimension,
                "rho": rho,
                "topk": training_log["topk"],
                "window_size": window_size,
            },
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
