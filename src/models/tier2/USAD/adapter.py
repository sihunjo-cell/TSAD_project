"""공식 USAD 코어를 외부 MinMax 입력과 채널별 점수에 연결한다."""

import numpy
import time
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .official import UsadModel
from ..utils.dataset import ReconstructDataset


USAD_CONFIG = {
    "window_size": 10,
    "latent_size": 100,
    "batch_size": 128,
    "n_epochs": 70,
    "learning_rate": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "drop_last": False,
}


def _window_dataset(session):
    return ReconstructDataset(
        numpy.asarray(session, dtype=numpy.float32),
        window_size=USAD_CONFIG["window_size"], normalize=False,
    )


def train_usad(train_sessions, device: str, epochs: int | None = None):
    datasets = [_window_dataset(session) for session in train_sessions]
    if not datasets or any(not len(dataset) for dataset in datasets):
        raise ValueError("USAD 학습 window가 필요하다")
    feature_count = numpy.asarray(train_sessions[0]).shape[1]
    model = UsadModel(
        USAD_CONFIG["window_size"] * feature_count,
        USAD_CONFIG["latent_size"],
    ).to(device)
    optimizer_arguments = {
        "lr": USAD_CONFIG["learning_rate"], "betas": USAD_CONFIG["betas"],
        "eps": USAD_CONFIG["eps"], "weight_decay": USAD_CONFIG["weight_decay"],
    }
    optimizer1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        **optimizer_arguments,
    )
    optimizer2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        **optimizer_arguments,
    )
    loader = DataLoader(
        ConcatDataset(datasets), batch_size=USAD_CONFIG["batch_size"],
        shuffle=True, drop_last=False,
    )
    epoch_count = USAD_CONFIG["n_epochs"] if epochs is None else epochs
    batch_iterations = 0
    for epoch in range(1, epoch_count + 1):
        model.train()
        for windows, _ in loader:
            batch = windows.flatten(1).to(device)
            loss1, _ = model.training_step(batch, epoch)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            _, loss2 = model.training_step(batch, epoch)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            batch_iterations += 1
    return model, {
        "enabled": False,
        "epochs_completed": epoch_count,
        "batch_iterations": batch_iterations,
        "optimizer_updates": 2 * batch_iterations,
    }


def score_usad(model, sessions, device: str):
    results = []
    model.eval()
    with torch.no_grad():
        for session in sessions:
            dataset = _window_dataset(session)
            if not len(dataset):
                raise ValueError("USAD 점수 window가 0개다")
            batches = []
            for windows, _ in DataLoader(
                dataset, batch_size=USAD_CONFIG["batch_size"], shuffle=False,
            ):
                windows = windows.to(device)
                reconstruction1, _, reconstruction3 = model(windows.flatten(1))
                batches.append((
                    0.5 * torch.abs(windows - reconstruction1.view_as(windows))
                    + 0.5 * torch.abs(windows - reconstruction3.view_as(windows))
                ).mean(dim=1).cpu())
            results.append(torch.cat(batches).numpy())
    return results


def load_usad_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["model_config"]
    model = UsadModel(config["input_dimension"], config["latent_size"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_usad_sessions(train_sessions, validation_sessions, test_sessions, device, epochs=None):
    training_started = time.perf_counter()
    model, training_log = train_usad(train_sessions, device, epochs)
    training_seconds = time.perf_counter() - training_started
    reference_started = time.perf_counter()
    reference_errors = score_usad(
        model, (*train_sessions, *validation_sessions), device,
    )
    reference_seconds = time.perf_counter() - reference_started
    test_started = time.perf_counter()
    test_errors = score_usad(model, test_sessions, device)
    return {
        "checkpoint": {
            "model_state_dict": model.state_dict(),
            "model_config": {
                **USAD_CONFIG,
                "input_dimension": model.encoder.linear1.in_features,
            },
        },
        "reference_errors": reference_errors,
        "test_errors": test_errors,
        "training_log": training_log,
        "timing": {
            "accelerator": device,
            "training_seconds": training_seconds,
            "train_reference_inference_seconds": reference_seconds,
            "test_inference_seconds": time.perf_counter() - test_started,
        },
    }
