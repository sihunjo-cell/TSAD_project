"""TSB-AD InnerAutoencoder 코어를 채널별 CI-AE로 실행한다."""

import importlib.util
import time
from pathlib import Path

import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CI_AE_CONFIG = {
    "window_size": 100,
    "hidden_neurons": (64, 32),
    "dropout": 0.2,
    "batch_size": 128,
    "n_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "drop_last": True,
}


def _load_inner_autoencoder():
    source = Path(__file__).with_name("AE_raw(TSB).py")
    spec = importlib.util.spec_from_file_location(
        "src.models.tier2.AE._tsb_source", source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"TSB-AD AE 코어를 읽지 못했다: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InnerAutoencoder


InnerAutoencoder = _load_inner_autoencoder()


def _new_model(device):
    return InnerAutoencoder(
        CI_AE_CONFIG["window_size"], CI_AE_CONFIG["hidden_neurons"],
        CI_AE_CONFIG["dropout"], True, "relu",
    ).to(device)


def _channel_windows(session: numpy.ndarray, channel: int) -> numpy.ndarray:
    values = numpy.asarray(session, dtype=numpy.float32)
    window = CI_AE_CONFIG["window_size"]
    if values.ndim != 2 or len(values) < window:
        raise ValueError("CI-AE 입력은 window 이상 길이의 2차원 배열이어야 한다")
    return numpy.lib.stride_tricks.sliding_window_view(values[:, channel], window)


def train_ci_ae(train_sessions, device: str, epochs: int | None = None):
    sessions = tuple(numpy.asarray(session, dtype=numpy.float32) for session in train_sessions)
    if not sessions or {session.shape[1] for session in sessions} != {19}:
        raise ValueError("CI-AE는 센서 19개인 학습 세션이 필요하다")
    epoch_count = CI_AE_CONFIG["n_epochs"] if epochs is None else epochs
    models = []
    optimizer_updates = 0
    for channel in range(19):
        windows = numpy.concatenate([
            _channel_windows(session, channel) for session in sessions
        ])
        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows.copy())),
            batch_size=CI_AE_CONFIG["batch_size"], shuffle=True,
            drop_last=CI_AE_CONFIG["drop_last"],
        )
        if not len(loader):
            raise ValueError("CI-AE 학습 batch가 0개다")
        model = _new_model(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=CI_AE_CONFIG["learning_rate"],
            weight_decay=CI_AE_CONFIG["weight_decay"],
        )
        for _ in range(epoch_count):
            model.train()
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                nn.functional.mse_loss(model(batch), batch).backward()
                optimizer.step()
                optimizer_updates += 1
        models.append(model)
    return tuple(models), {
        "enabled": False,
        "epochs_completed": epoch_count,
        "optimizer_updates": optimizer_updates,
    }


def score_ci_ae(models, sessions, device: str):
    if len(models) != 19:
        raise ValueError("CI-AE 모델은 채널별 19개여야 한다")
    results = []
    for session in sessions:
        columns = []
        for channel, model in enumerate(models):
            loader = DataLoader(
                TensorDataset(torch.from_numpy(_channel_windows(session, channel).copy())),
                batch_size=CI_AE_CONFIG["batch_size"], shuffle=False,
            )
            model.eval()
            batches = []
            with torch.no_grad():
                for (windows,) in loader:
                    windows = windows.to(device)
                    batches.append(torch.abs(model(windows) - windows).mean(dim=1).cpu())
            columns.append(torch.cat(batches).numpy())
        results.append(numpy.column_stack(columns))
    return results


def load_ci_ae_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    models = tuple(_new_model(device) for _ in checkpoint["model_state_dicts"])
    for model, state_dict in zip(models, checkpoint["model_state_dicts"]):
        model.load_state_dict(state_dict)
    return models


def run_ci_ae_sessions(train_sessions, validation_sessions, test_sessions, device, epochs=None):
    training_started = time.perf_counter()
    models, training_log = train_ci_ae(train_sessions, device, epochs)
    training_seconds = time.perf_counter() - training_started
    reference_started = time.perf_counter()
    reference_errors = score_ci_ae(
        models, (*train_sessions, *validation_sessions), device,
    )
    reference_seconds = time.perf_counter() - reference_started
    test_started = time.perf_counter()
    test_errors = score_ci_ae(models, test_sessions, device)
    return {
        "checkpoint": {
            "model_state_dicts": [model.state_dict() for model in models],
            "model_config": {**CI_AE_CONFIG, "feature_count": len(models)},
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
