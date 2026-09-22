"""Project split, training, and continuous-score controller for official GDN."""

import copy
import random
import time
from collections.abc import Callable

import numpy
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from src.common.validate_multivariate_session import as_finite_multivariate_session
from src.common.model_registry import resolve_gdn_topk
from src.common.smoothing import trailing_average_smoothing

from .official import GDN, SOURCE_COMMIT, build_fully_connected_edge_index


def _synchronize_cuda(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def topk_from_rho(channel_count: int, rho: float) -> int:
    return resolve_gdn_topk(channel_count, rho=rho)


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


def split_official_windows(window_count: int, validation_ratio: float, split_seed: int):
    """Hold out the contiguous window block used by the source controller."""
    if (type(window_count) is not int or window_count < 2
            or not numpy.isfinite(validation_ratio) or not 0 < validation_ratio < 1
            or type(split_seed) is not int):
        raise ValueError("GDN official window split arguments are invalid")
    training_length = int(window_count * (1 - validation_ratio))
    validation_length = int(window_count * validation_ratio)
    if training_length < 1 or validation_length < 1:
        raise ValueError("GDN official split needs nonempty training and validation windows")
    start = random.Random(split_seed).randrange(training_length)
    stop = start + validation_length
    return (
        list(range(start)) + list(range(stop, window_count)),
        list(range(start, stop)),
    )


def build_gdn_model(
    channel_count: int,
    *,
    embedding_dimension: int,
    hidden_dimension: int,
    rho: float | None = None,
    fixed_topk: int | None = None,
    window_size: int,
    device: str,
):
    topk = resolve_gdn_topk(channel_count, rho=rho, topk=fixed_topk)
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
    full_prefix: bool = False,
    official_procedure: bool = False,
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
                differences = (predictions.double() - target_batch.double()
                               if official_procedure else predictions - target_batch)
                score_batches.append(torch.abs(differences).cpu().numpy())
            scores = numpy.concatenate(score_batches)
            if official_procedure:
                median = numpy.median(scores, axis=0)
                quartiles = numpy.percentile(scores, (25, 75), axis=0)
                normalized = (scores - median) / (numpy.abs(quartiles[1] - quartiles[0]) + 0.01)
                native_channel_scores = trailing_average_smoothing(normalized, window=4)
                scores = native_channel_scores.max(axis=1)
            source_end = len(targets) + window_size
            outputs.append({
                "scores": scores,
                "source_start": window_size,
                "source_end_exclusive": source_end,
                "alignment": "next_step",
                "primitive": "absolute_error",
                "calibration_mode": "fit_median_iqr" if full_prefix else "validation_median_iqr",
                "evaluation_mode": "causal",
                "lookahead": 0,
                "maximum_effective_lookahead": 0,
                "normalization_scope": "current_prefix_fit" if full_prefix else "current_prefix_validation",
                **({
                    "calibration_mode": "official_full_evaluation",
                    "normalization_scope": "full_evaluation",
                    "native_postprocessing": True,
                    "official_protocol": "paper_tuning_v4",
                    "native_postprocessing_recipe": "full_evaluation_median_iqr_trailing4_max",
                    "native_channel_scores": native_channel_scores,
                    "native_calibration": {
                        "source": "full_evaluation", "source_start": window_size,
                        "source_end_exclusive": source_end, "score_space": "absolute_error",
                        "median": median.tolist(), "iqr": numpy.abs(quartiles[1] - quartiles[0]).tolist(),
                        "epsilon": 0.01,
                    },
                    "evaluation_mode": "offline_noncausal",
                    "maximum_effective_lookahead": len(targets) - 1,
                } if official_procedure else {}),
            })
    return tuple(outputs)


def train_gdn(
    fit_sessions,
    validation_sessions,
    *,
    embedding_dimension: int,
    hidden_dimension: int,
    rho: float | None = None,
    fixed_topk: int | None = None,
    device: str,
    window_size: int = 5,
    epochs: int = 30,
    patience: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    model=None,
    edge_index=None,
    topk=None,
    full_prefix: bool = False,
    official_procedure: bool = False,
    optimizer_betas=(0.9, 0.999),
    validation_ratio: float = 0.2,
    split_seed: int = 0,
):
    """Select by training loss for a full prefix, otherwise by normal validation."""
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
    if (full_prefix or official_procedure) and validation_sessions:
        raise ValueError("GDN full-prefix 학습에는 validation을 따로 주지 않는다")
    if not fit_sessions or (not full_prefix and not official_procedure and not validation_sessions):
        raise ValueError("GDN needs fit and validation sessions")
    channel_count = fit_sessions[0].shape[1]
    if any(session.shape[1] != channel_count for session in fit_sessions + validation_sessions):
        raise ValueError("GDN sessions have different channel counts")
    prepared = (model, edge_index, topk)
    if any(value is None for value in prepared) and not all(
        value is None for value in prepared
    ):
        raise ValueError("GDN 준비 모델, edge_index와 topk를 함께 넘겨야 한다")
    if model is None:
        model, edge_index, topk = build_gdn_model(
            channel_count,
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            rho=rho,
            fixed_topk=fixed_topk,
            window_size=window_size,
            device=device,
        )
    fit_dataset = _make_dataset(fit_sessions, window_size)
    internal_validation = None
    if official_procedure:
        training_indices, validation_indices = split_official_windows(
            len(fit_dataset), validation_ratio, split_seed,
        )
        validation_dataset = Subset(fit_dataset, validation_indices)
        internal_validation = {
            "source": "current_prefix_windows",
            "validation_ratio": validation_ratio,
            "split_seed": split_seed,
            "window_start": validation_indices[0],
            "window_end_exclusive": validation_indices[-1] + 1,
            "window_count": len(validation_indices),
            "training_window_count": len(training_indices),
            "loss_reduction": "mean_batch_mse",
        }
        fit_dataset = Subset(fit_dataset, training_indices)
    elif not full_prefix:
        validation_dataset = _make_dataset(validation_sessions, window_size)
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = None if full_prefix and not official_procedure else DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=tuple(optimizer_betas))
    best_loss = float("inf")
    best_state = None
    selected_epoch = None
    stale_epochs = 0
    updates = 0
    completed_epochs = 0
    training_examples_seen = 0
    internal_validation_seconds = 0.0
    loss_history = []

    for epoch in range(epochs):
        model.train()
        training_loss_sum = 0.0
        for inputs, targets in fit_loader:
            optimizer.zero_grad()
            predictions = model(inputs.to(device), edge_index)
            loss = torch.mean((predictions - targets.to(device)) ** 2)
            if (full_prefix or official_procedure) and not torch.isfinite(loss):
                raise RuntimeError("GDN training loss must remain finite through the epoch cap")
            loss.backward()
            optimizer.step()
            updates += 1
            training_examples_seen += len(inputs)
            training_loss_sum += loss.item()

        completed_epochs = epoch + 1
        loss_history.append({
            "epoch": completed_epochs,
            "training_batch_mse_sum": training_loss_sum,
            "training_batch_mse_mean": training_loss_sum / len(fit_loader),
            "validation_loss": None,
        })
        if full_prefix and not official_procedure:
            if training_loss_sum < best_loss:
                best_loss = training_loss_sum
                best_state = copy.deepcopy(model.state_dict())
                selected_epoch = completed_epochs
            continue
        model.eval()
        _synchronize_cuda(device)
        validation_started = time.perf_counter()
        validation_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                predictions = model(inputs.to(device), edge_index)
                differences = predictions - targets.to(device)
                if official_procedure:
                    validation_sum += torch.mean(differences ** 2).item()
                    validation_count += 1
                else:
                    validation_sum += torch.sum(differences ** 2).item()
                    validation_count += differences.numel()
        validation_loss = validation_sum / validation_count
        if official_procedure and not numpy.isfinite(validation_loss):
            raise RuntimeError("GDN official validation loss must remain finite")
        _synchronize_cuda(device)
        internal_validation_seconds += time.perf_counter() - validation_started
        completed_epochs = epoch + 1
        loss_history[-1]["validation_loss"] = validation_loss
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            selected_epoch = completed_epochs
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError(
            "GDN did not produce a training checkpoint" if full_prefix
            else "GDN did not produce a validation checkpoint"
        )
    model.load_state_dict(best_state)
    return model, edge_index, {
        "epoch_cap": epochs,
        "epochs_completed": completed_epochs,
        "selected_epoch": selected_epoch,
        "optimizer_updates": updates,
        "training_examples_seen": training_examples_seen,
        "loss_history": loss_history,
        "best_training_loss": best_loss if full_prefix and not official_procedure else None,
        "training_loss_reduction": "sum_batch_mean_mse" if full_prefix and not official_procedure else None,
        "best_validation_loss": None if full_prefix and not official_procedure else best_loss,
        "stopped_early": completed_epochs < epochs,
        "topk": topk,
        "checkpoint_selection": "best_training_loss" if full_prefix else "normal_validation_mse",
        **({
            "checkpoint_selection": "source_prefix_window_validation_mse",
            "official_protocol": "paper_tuning_v4",
            "training_scope": "current_prefix_training_windows",
            "optimizer_betas": list(optimizer_betas),
            "model_internal_validation": internal_validation,
            "model_internal_validation_seconds": internal_validation_seconds,
        } if official_procedure else {}),
    }


def run_gdn_sessions(
    fit_sessions,
    validation_sessions,
    test_sessions,
    *,
    embedding_dimension: int,
    hidden_dimension: int,
    rho: float | None = None,
    fixed_topk: int | None = None,
    device: str,
    window_size: int = 5,
    epochs: int = 30,
    patience: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    full_prefix: bool = False,
    official_procedure: bool = False,
    optimizer_betas=(0.9, 0.999),
    validation_ratio: float = 0.2,
    split_seed: int = 0,
    trainer=None,
    on_training_complete: Callable[[dict], None] | None = None,
):
    if (full_prefix or official_procedure) and len(validation_sessions):
        raise ValueError("GDN full-prefix 학습에는 validation을 따로 주지 않는다")
    default_trainer = trainer is None
    trainer = trainer or train_gdn
    model_setup_seconds = 0.0
    if default_trainer:
        channel_count = as_finite_multivariate_session(
            fit_sessions[0], "fit_session",
        ).shape[1]
        setup_started = time.perf_counter()
        model, edge_index, topk = build_gdn_model(
            channel_count,
            embedding_dimension=embedding_dimension,
            hidden_dimension=hidden_dimension,
            rho=rho,
            fixed_topk=fixed_topk,
            window_size=window_size,
            device=device,
        )
        _synchronize_cuda(device)
        model_setup_seconds = time.perf_counter() - setup_started
    training_started = time.perf_counter()
    training_arguments = {
        "embedding_dimension": embedding_dimension,
        "hidden_dimension": hidden_dimension,
        **({"fixed_topk": fixed_topk} if fixed_topk is not None else {"rho": rho}),
        "device": device,
        "window_size": window_size,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if default_trainer:
        training_arguments.update(
            model=model, edge_index=edge_index, topk=topk,
        )
    if full_prefix:
        training_arguments["full_prefix"] = True
    if official_procedure:
        training_arguments.update(
            official_procedure=True, optimizer_betas=optimizer_betas,
            validation_ratio=validation_ratio, split_seed=split_seed,
        )
    elif tuple(optimizer_betas) != (0.9, 0.999):
        training_arguments["optimizer_betas"] = optimizer_betas
    model, edge_index, training_log = trainer(
        fit_sessions, validation_sessions, **training_arguments,
    )
    _synchronize_cuda(device)
    training_seconds = time.perf_counter() - training_started
    channel_count = as_finite_multivariate_session(
        fit_sessions[0], "fit_session",
    ).shape[1]
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "channel_count": channel_count,
            "embedding_dimension": embedding_dimension,
            "hidden_dimension": hidden_dimension,
            **({"fixed_topk": fixed_topk} if fixed_topk is not None else {"rho": rho}),
            "topk": training_log["topk"],
            "window_size": window_size,
            "full_prefix": full_prefix,
            "official_procedure": official_procedure,
            "optimizer_betas": list(optimizer_betas),
            **({"validation_ratio": validation_ratio, "split_seed": split_seed}
               if official_procedure else {}),
        },
        "source_commit": SOURCE_COMMIT,
        "training_log": dict(training_log),
    }
    training_protocol = {"training_protocol": {
        "official_protocol": "paper_tuning_v4",
        "training_scope": "current_prefix_training_windows",
        "checkpoint_selection": "source_prefix_window_validation_mse",
        "model_internal_validation": training_log.get("model_internal_validation"),
    }} if official_procedure else {}
    if on_training_complete is not None:
        on_training_complete({
            "checkpoint": checkpoint, "training_log": training_log, **training_protocol,
            "timing": {
                "model_setup_seconds": model_setup_seconds,
                "training_seconds": training_seconds,
                "validation_inference_seconds": 0.0,
                "calibration_inference_seconds": 0.0,
                "test_inference_seconds": 0.0,
                "accelerator": device,
            },
        })
    validation_started = time.perf_counter()
    reference_outputs = () if official_procedure else score_gdn_sessions(
        model,
        fit_sessions if full_prefix else validation_sessions,
        window_size=window_size,
        edge_index=edge_index,
        device=device,
        batch_size=batch_size,
        full_prefix=full_prefix,
        official_procedure=official_procedure,
    )
    _synchronize_cuda(device)
    validation_seconds = 0.0 if official_procedure else time.perf_counter() - validation_started
    test_started = time.perf_counter()
    test_outputs = score_gdn_sessions(
        model,
        test_sessions,
        window_size=window_size,
        edge_index=edge_index,
        device=device,
        batch_size=batch_size,
        full_prefix=full_prefix,
        official_procedure=official_procedure,
    )
    _synchronize_cuda(device)
    return {
        "checkpoint": checkpoint,
        "validation_outputs": () if full_prefix else reference_outputs,
        "calibration_outputs": reference_outputs if full_prefix else (),
        "test_outputs": test_outputs,
        "training_log": training_log,
        **training_protocol,
        "timing": {
            "model_setup_seconds": model_setup_seconds,
            "training_seconds": training_seconds,
            "validation_inference_seconds": 0.0 if full_prefix else validation_seconds,
            "calibration_inference_seconds": validation_seconds if full_prefix else 0.0,
            "test_inference_seconds": time.perf_counter() - test_started,
            "accelerator": device,
        },
    }
