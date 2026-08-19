"""TSB-AD LSTMModel 코어를 고정 validation과 외부 MinMax 입력에 연결한다."""

import copy
import time

import numpy
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .LSTMAD_raw import LSTMModel
from ..utils.dataset import ForecastDataset


LSTM_AD_CONFIG = {
    "window_size": 100,
    "hidden_dim": 20,
    "num_layers": 2,
    "forecast_horizon": 1,
    "batch_size": 128,
    "n_epochs": 50,
    "learning_rate": 0.0008,
    "scheduler_step_size": 5,
    "scheduler_gamma": 0.75,
    "early_stop_patience": 3,
    "early_stop_delta": 0.0001,
    "drop_last": False,
}


def _forecast_dataset(session):
    return ForecastDataset(
        numpy.asarray(session, dtype=numpy.float32),
        window_size=LSTM_AD_CONFIG["window_size"],
        pred_len=LSTM_AD_CONFIG["forecast_horizon"],
        normalize=False,
    )


def _new_model(feature_count, device):
    return LSTMModel(
        LSTM_AD_CONFIG["window_size"], feature_count,
        LSTM_AD_CONFIG["hidden_dim"], LSTM_AD_CONFIG["forecast_horizon"],
        LSTM_AD_CONFIG["num_layers"], LSTM_AD_CONFIG["batch_size"], device,
    ).to(device)


def train_lstm_ad(train_sessions, validation_sessions, device: str, epochs: int | None = None):
    train_sets = [_forecast_dataset(session) for session in train_sessions]
    validation_sets = [_forecast_dataset(session) for session in validation_sessions]
    if not train_sets or not validation_sets or any(not len(data) for data in train_sets + validation_sets):
        raise ValueError("LSTM-AD train·validation window가 필요하다")
    feature_count = numpy.asarray(train_sessions[0]).shape[1]
    model = _new_model(feature_count, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_AD_CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LSTM_AD_CONFIG["scheduler_step_size"],
        gamma=LSTM_AD_CONFIG["scheduler_gamma"],
    )
    train_loader = DataLoader(
        ConcatDataset(train_sets), batch_size=LSTM_AD_CONFIG["batch_size"],
        shuffle=True, drop_last=False,
    )
    validation_loader = DataLoader(
        ConcatDataset(validation_sets), batch_size=LSTM_AD_CONFIG["batch_size"],
        shuffle=False, drop_last=False,
    )
    epoch_count = LSTM_AD_CONFIG["n_epochs"] if epochs is None else epochs
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    optimizer_updates = 0
    epochs_completed = 0
    for epoch in range(1, epoch_count + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs).permute(1, 0, 2)
            nn.functional.mse_loss(predictions, targets).backward()
            optimizer.step()
            optimizer_updates += 1
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                predictions = model(inputs.to(device)).permute(1, 0, 2)
                validation_losses.append(nn.functional.mse_loss(
                    predictions, targets.to(device), reduction="mean",
                ).item())
        validation_loss = float(numpy.mean(validation_losses))
        scheduler.step()
        epochs_completed = epoch
        if validation_loss < best_loss - LSTM_AD_CONFIG["early_stop_delta"]:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= LSTM_AD_CONFIG["early_stop_patience"]:
                break
    if best_state is None:
        raise RuntimeError("LSTM-AD best validation 상태를 만들지 못했다")
    model.load_state_dict(best_state)
    return model, {
        "enabled": True,
        "epochs_completed": epochs_completed,
        "optimizer_updates": optimizer_updates,
        "best_validation_loss": best_loss,
        "stopped_early": epochs_completed < epoch_count,
    }


def score_lstm_ad(model, sessions, device: str):
    results = []
    model.eval()
    with torch.no_grad():
        for session in sessions:
            dataset = _forecast_dataset(session)
            if not len(dataset):
                raise ValueError("LSTM-AD 점수 window가 0개다")
            batches = []
            for inputs, targets in DataLoader(
                dataset, batch_size=LSTM_AD_CONFIG["batch_size"], shuffle=False,
            ):
                predictions = model(inputs.to(device)).permute(1, 0, 2)
                batches.append(torch.abs(
                    predictions - targets.to(device)
                )[:, 0, :].cpu())
            results.append(torch.cat(batches).numpy())
    return results


def load_lstm_ad_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = _new_model(checkpoint["model_config"]["feature_count"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_lstm_ad_sessions(train_sessions, validation_sessions, test_sessions, device, epochs=None):
    training_started = time.perf_counter()
    model, training_log = train_lstm_ad(
        train_sessions, validation_sessions, device, epochs,
    )
    training_seconds = time.perf_counter() - training_started
    reference_started = time.perf_counter()
    reference_errors = score_lstm_ad(
        model, (*train_sessions, *validation_sessions), device,
    )
    reference_seconds = time.perf_counter() - reference_started
    test_started = time.perf_counter()
    test_errors = score_lstm_ad(model, test_sessions, device)
    return {
        "checkpoint": {
            "model_state_dict": model.state_dict(),
            "model_config": {**LSTM_AD_CONFIG, "feature_count": model.feats},
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
