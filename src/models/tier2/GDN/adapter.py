"""로컬 GDN 코어를 원 저자 학습 제어와 외부 MinMax 입력에 연결한다."""

import copy
import time

import numpy
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from .graph import build_fully_connected_edge_index
from .model import GDN
from ..utils.dataset import ForecastDataset


GDN_CONFIG = {
    "window_size": 5,
    "embed_dim": 64,
    "out_layer_inter_dim": 128,
    "out_layer_num": 1,
    "heads": 1,
    "attention_dropout": 0.0,
    "output_dropout": 0.2,
    "batch_size": 32,
    "n_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "betas": (0.9, 0.99),
    "eps": 1e-8,
    "early_stop_patience": 10,
    "early_stop_delta": 0.0,
    "forecast_horizon": 1,
}


def _forecast_dataset(session):
    return ForecastDataset(
        numpy.asarray(session, dtype=numpy.float32),
        window_size=GDN_CONFIG["window_size"], pred_len=1, normalize=False,
    )


def _new_model(feature_count, topk, device):
    edge_index = build_fully_connected_edge_index(
        torch.empty((1, feature_count)), device,
    )
    model = GDN(
        [edge_index], n_features=feature_count,
        embed_dim=GDN_CONFIG["embed_dim"],
        out_layer_inter_dim=GDN_CONFIG["out_layer_inter_dim"],
        window_size=GDN_CONFIG["window_size"],
        out_layer_num=GDN_CONFIG["out_layer_num"], topk=topk,
        heads=GDN_CONFIG["heads"], dropout=GDN_CONFIG["attention_dropout"],
        learn_graph=True,
    ).to(device)
    model.dp = nn.Dropout(GDN_CONFIG["output_dropout"])
    return model


def train_gdn(train_sessions, validation_sessions, topk: int, device: str, epochs: int | None = None):
    train_sets = [_forecast_dataset(session) for session in train_sessions]
    validation_sets = [_forecast_dataset(session) for session in validation_sessions]
    if not train_sets or not validation_sets or any(not len(data) for data in train_sets + validation_sets):
        raise ValueError("GDN train·validation window가 필요하다")
    feature_count = numpy.asarray(train_sessions[0]).shape[1]
    if not 0 < topk <= feature_count:
        raise ValueError("GDN topk가 채널 수 범위를 벗어났다")
    model = _new_model(feature_count, topk, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=GDN_CONFIG["learning_rate"],
        weight_decay=GDN_CONFIG["weight_decay"], betas=GDN_CONFIG["betas"],
        eps=GDN_CONFIG["eps"],
    )
    train_loader = DataLoader(
        ConcatDataset(train_sets), batch_size=GDN_CONFIG["batch_size"],
        shuffle=True, drop_last=False,
    )
    validation_loader = DataLoader(
        ConcatDataset(validation_sets), batch_size=GDN_CONFIG["batch_size"],
        shuffle=False, drop_last=False,
    )
    epoch_count = GDN_CONFIG["n_epochs"] if epochs is None else epochs
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    optimizer_updates = 0
    epochs_completed = 0
    for epoch in range(1, epoch_count + 1):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(inputs.to(device).permute(0, 2, 1).contiguous())
            loss = nn.functional.mse_loss(predictions, targets[:, 0, :].to(device))
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                predictions = model(inputs.to(device).permute(0, 2, 1).contiguous())
                validation_losses.append(nn.functional.mse_loss(
                    predictions, targets[:, 0, :].to(device), reduction="mean",
                ).item())
        validation_loss = float(numpy.mean(validation_losses))
        epochs_completed = epoch
        if validation_loss < best_loss - GDN_CONFIG["early_stop_delta"]:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= GDN_CONFIG["early_stop_patience"]:
                break
    if best_state is None:
        raise RuntimeError("GDN best validation 상태를 만들지 못했다")
    model.load_state_dict(best_state)
    return model, {
        "enabled": True,
        "epochs_completed": epochs_completed,
        "optimizer_updates": optimizer_updates,
        "best_validation_loss": best_loss,
        "stopped_early": epochs_completed < epoch_count,
        "scheduler": "none",
        "gradient_clip": "none",
    }


def score_gdn(model, sessions, device: str):
    results = []
    model.eval()
    with torch.no_grad():
        for session in sessions:
            dataset = _forecast_dataset(session)
            if not len(dataset):
                raise ValueError("GDN 점수 window가 0개다")
            batches = []
            for inputs, targets in DataLoader(
                dataset, batch_size=GDN_CONFIG["batch_size"], shuffle=False,
            ):
                inputs = inputs.to(device).permute(0, 2, 1).contiguous()
                targets = targets[:, 0, :].to(device)
                batches.append(torch.abs(model(inputs) - targets).cpu())
            results.append(torch.cat(batches).numpy())
    return results


def load_gdn_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["model_config"]
    model = _new_model(config["feature_count"], config["topk"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_gdn_sessions(
    train_sessions, validation_sessions, test_sessions, topk, device, epochs=None,
):
    training_started = time.perf_counter()
    model, training_log = train_gdn(
        train_sessions, validation_sessions, topk, device, epochs,
    )
    training_seconds = time.perf_counter() - training_started
    reference_started = time.perf_counter()
    reference_errors = score_gdn(
        model, (*train_sessions, *validation_sessions), device,
    )
    reference_seconds = time.perf_counter() - reference_started
    test_started = time.perf_counter()
    test_errors = score_gdn(model, test_sessions, device)
    return {
        "checkpoint": {
            "model_state_dict": model.state_dict(),
            "model_config": {
                **GDN_CONFIG,
                "topk": topk,
                "feature_count": model.embedding.num_embeddings,
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
