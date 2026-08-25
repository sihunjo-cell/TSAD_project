"""분할과 전처리가 끝난 세션으로 GDN을 학습하고 점수를 저장한다.

데이터 분할은 호출자가 맡는다. 정규화와 smoothing은 이 프로젝트의 공통 코드를 사용하며
topk, epsilon과 smoothing 창은 설정에서 받는다. 학습기는 best checkpoint를 만들 수 있도록
최소 2 epoch가 필요하다.
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.normalization import apply_median_iqr, estimate_median_iqr
from src.common.save_scores import save_score_arrays, save_score_metadata, snapshot_config
from src.common.experiment_config import validate_fork_contract, validate_pipeline_contract
from src.common.verify_run_context import (
    verify_git_hashes_unchanged,
    verify_run_context,
    verify_runtime_versions,
)
from src.data_split.validation_split import validation_split


def load_gdn_dependencies(fork_path: str) -> SimpleNamespace:
    """고정 GraGOD 포크의 학습 의존성을 한 경계에서 불러온다."""
    if fork_path not in sys.path:
        sys.path.insert(0, fork_path)

    import pytorch_lightning
    import torch
    from torch import nn
    from pytorch_lightning.loggers import TensorBoardLogger

    from torch.utils.data import ConcatDataset, DataLoader

    from datasets.dataset import SlidingWindowDataset, get_data_loader
    from datasets.graph import build_fully_connected_edge_index
    from gragod import CleanMethods, Models
    from gragod.models import get_model_and_module
    from gragod.training import set_seeds
    from gragod.training.callbacks import get_training_callbacks
    from gragod.training.trainer import TrainerPL
    from gragod.utils import set_device

    return SimpleNamespace(
        torch=torch,
        MSELoss=nn.MSELoss,
        TensorBoardLogger=TensorBoardLogger,
        get_data_loader=get_data_loader,
        SlidingWindowDataset=SlidingWindowDataset,
        ConcatDataset=ConcatDataset,
        DataLoader=DataLoader,
        build_fully_connected_edge_index=build_fully_connected_edge_index,
        CleanMethods=CleanMethods,
        Models=Models,
        get_model_and_module=get_model_and_module,
        set_seeds=set_seeds,
        get_training_callbacks=get_training_callbacks,
        TrainerPL=TrainerPL,
        set_device=set_device,
        PredictionTrainer=pytorch_lightning.Trainer,
    )


def compute_absolute_errors(
    lightning_module, series_tensor, edge_index, window_size, batch_size, device, dependencies,
):
    """한 구간의 1-step forecast 절대 오차 `(L-W, n_features)`를 계산한다.

    마지막 forecast까지 포함해 `L-W`개 예측을 반환한다.
    """
    loader = dependencies.get_data_loader(
        X=series_tensor, edge_index=edge_index,
        y=dependencies.torch.zeros(series_tensor.shape[0]),
        window_size=window_size, clean=dependencies.CleanMethods.NONE,
        batch_size=batch_size, n_workers=0, shuffle=False,
    )

    predict_trainer = dependencies.PredictionTrainer(
        accelerator=device, logger=False, enable_checkpointing=False, enable_progress_bar=False,
    )
    output = predict_trainer.predict(lightning_module, loader)
    predictions = dependencies.torch.cat(output)
    truth = series_tensor[window_size:]
    if predictions.shape != truth.shape:
        raise ValueError(f"GDN forecast와 정답 shape가 다르다: {predictions.shape} != {truth.shape}")
    errors = dependencies.torch.abs(predictions - truth).detach().cpu().numpy()
    if not numpy.isfinite(errors).all():
        raise ValueError("GDN forecast 절대 오차는 모두 유한해야 한다")
    return errors


def build_session_loader(
    session_tensors, edge_index, window_size, batch_size, shuffle, dependencies,
):
    """원시 배열을 잇지 않고 세션별 window dataset만 합친다."""
    datasets = [
        dependencies.SlidingWindowDataset(
            data=session_tensor,
            edge_index=edge_index,
            window_size=window_size,
            labels=dependencies.torch.zeros(session_tensor.shape[0]),
            drop=False,
        )
        for session_tensor in session_tensors
    ]
    return dependencies.DataLoader(
        dependencies.ConcatDataset(datasets),
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        persistent_workers=False,
    )


def run_gdn_single(
    config: dict,
    train_array: numpy.ndarray,
    test_array: numpy.ndarray,
    output_dir: str,
    seed: int,
    input_metadata: dict,
) -> dict:
    """기존 단일 세션 계약. validation을 자른 뒤 다중 세션 본체에 위임한다."""
    train_params = config["train_params"]
    train_part, validation_part = validation_split(
        numpy.asarray(train_array),
        val_fraction=train_params["val_size"],
        min_train_length=train_params["min_train_length"],
    )
    return run_gdn_sessions(
        config,
        (train_part,),
        (validation_part,),
        (numpy.asarray(test_array),),
        output_dir,
        seed,
        input_metadata,
    )


def run_gdn_sessions(
    config: dict,
    train_sessions,
    validation_sessions,
    test_sessions,
    output_dir: str,
    seed: int,
    input_metadata: dict,
) -> dict:
    """세션 경계를 보존해 모델을 한 번 학습하고 test 세션별 점수를 저장한다."""
    train_sessions = tuple(numpy.asarray(session) for session in train_sessions)
    validation_sessions = tuple(numpy.asarray(session) for session in validation_sessions)
    test_sessions = tuple(numpy.asarray(session) for session in test_sessions)
    if not train_sessions or len(train_sessions) != len(validation_sessions):
        raise ValueError("train·validation 세션 수가 같고 1개 이상이어야 한다")
    if not test_sessions:
        raise ValueError("test 세션이 1개 이상이어야 한다")

    feature_count = train_sessions[0].shape[1]
    all_sessions = train_sessions + validation_sessions + test_sessions
    if any(session.ndim != 2 or session.shape[1] != feature_count for session in all_sessions):
        raise ValueError("모든 세션은 같은 feature 수를 가진 2차원 배열이어야 한다")
    if any(not numpy.isfinite(session).all() for session in all_sessions):
        raise ValueError("모든 세션 값은 유한해야 한다")
    window_size = config["model_params"]["window_size"]
    if any(len(session) < window_size + 1 for session in all_sessions):
        raise ValueError(f"모든 세션 길이는 window_size+1={window_size + 1} 이상이어야 한다")

    naming = config["naming"]
    series_value = naming["series"]
    test_series = (series_value,) if isinstance(series_value, int) else tuple(series_value)
    if len(test_series) != len(test_sessions):
        raise ValueError("naming.series 수와 test 세션 수가 다르다")
    if naming["dataset"] in {"GHL", "HAI"}:
        files = input_metadata.get("files") if isinstance(input_metadata, dict) else None
        if not isinstance(files, (list, tuple)) or not files:
            raise ValueError("GHL·HAI 실행의 input_metadata.files가 비었거나 없다")
        for file in files:
            if (
                not isinstance(file, dict)
                or not isinstance(file.get("name"), str)
                or not file["name"]
                or not isinstance(file.get("size_bytes"), int)
                or file["size_bytes"] < 0
            ):
                raise ValueError("input_metadata.files의 name·size_bytes 형식이 잘못됐다")
            sha256 = file.get("sha256")
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise ValueError("input_metadata.files의 SHA-256 형식이 잘못됐다")

    fork_path = str(config["fork_path"])
    git_hashes = verify_run_context(REPOSITORY_ROOT, fork_path)
    with open(REPOSITORY_ROOT / "configs" / "scoring_pipeline.yaml", encoding="utf-8") as scoring_file:
        scoring_config = yaml.safe_load(scoring_file)
    with open(REPOSITORY_ROOT / "configs" / "data_preprocessing.yaml", encoding="utf-8") as preprocessing_file:
        preprocessing_config = yaml.safe_load(preprocessing_file)
    with open(REPOSITORY_ROOT / "configs" / "environment.yaml", encoding="utf-8") as environment_file:
        environment_config = yaml.safe_load(environment_file)
    validate_pipeline_contract(
        scoring_config, preprocessing_config, naming["dataset"], feature_count,
    )
    validate_fork_contract(config["fork_contract"])
    runtime_versions = verify_runtime_versions(environment_config)
    output_dir = Path(output_dir)
    snapshot_path = snapshot_config(
        {"run_config": config,
         "data_preprocessing": preprocessing_config,
         "scoring_pipeline": scoring_config,
         "environment": environment_config,
         "runtime_versions": runtime_versions,
         "input": input_metadata,
         "seed": seed,
         "git_hashes": git_hashes},
        git_hash=git_hashes["tsad_project"],
        output_dir=str(output_dir / "snapshots"),
    )

    dependencies = load_gdn_dependencies(fork_path)
    dependencies.set_seeds(seed)
    device = dependencies.set_device()
    train_params = config["train_params"]
    train_tensors = tuple(
        dependencies.torch.tensor(session, dtype=dependencies.torch.float32)
        for session in train_sessions
    )
    validation_tensors = tuple(
        dependencies.torch.tensor(session, dtype=dependencies.torch.float32)
        for session in validation_sessions
    )
    test_tensors = tuple(
        dependencies.torch.tensor(session, dtype=dependencies.torch.float32)
        for session in test_sessions
    )

    # learned graph를 쓰더라도 모델 생성에는 초기 edge_index가 필요하다.
    edge_index = dependencies.build_fully_connected_edge_index(train_tensors[0], device)

    model_params = dict(config["model_params"])
    model_params["edge_index"] = [edge_index]
    model_params["n_features"] = feature_count
    model_params["out_dim"] = feature_count

    loader_arguments = dict(
        edge_index=edge_index,
        window_size=window_size,
        batch_size=train_params["batch_size"],
        shuffle=train_params["shuffle"],
        dependencies=dependencies,
    )
    train_loader = build_session_loader(train_tensors, **loader_arguments)
    val_loader = build_session_loader(
        validation_tensors, **{**loader_arguments, "shuffle": False},
    )

    model_class, module_class = dependencies.get_model_and_module(dependencies.Models.GDN)
    logger = dependencies.TensorBoardLogger(
        save_dir=str(output_dir / "training"), name="gdn", default_hp_metric=False,
    )
    callback_dict = dependencies.get_training_callbacks(
        log_dir=logger.log_dir, model_name="gdn",
        monitor="Loss/val", monitor_mode="min",
        early_stop_patience=train_params["early_stop_patience"],
        early_stop_delta=train_params["early_stop_delta"],
        save_top_k=1,
    )
    model_instance = model_class(**model_params).to(device)
    trainer = dependencies.TrainerPL(
        model=model_instance, model_pl=module_class, model_params=model_params,
        criterion=dependencies.MSELoss(),
        batch_size=train_params["batch_size"], n_epochs=train_params["n_epochs"],
        init_lr=train_params["init_lr"], device=device,
        log_dir=str(output_dir / "training"),
        callbacks=list(callback_dict.values()),
        checkpoint_cb=callback_dict["checkpoint"], logger=logger,
        target_dims=None, log_every_n_steps=train_params["log_every_n_steps"],
        weight_decay=train_params["weight_decay"], eps=train_params["eps"],
        betas=tuple(train_params["betas"]),
    )
    training_started = time.perf_counter()
    trainer.fit(train_loader, val_loader, args_summary={"seed": seed})
    training_seconds = time.perf_counter() - training_started

    # 콜백이 기록한 best checkpoint를 사용한다.
    best_checkpoint_path = callback_dict["checkpoint"].best_model_path
    best_module = module_class.load_from_checkpoint(best_checkpoint_path, map_location=device)
    best_module.eval()

    # trainnorm 통계는 세션 안에서 train+validation을 복원해 계산한 뒤 행축으로만 합친다.
    error_arguments = dict(edge_index=edge_index, window_size=window_size,
                           batch_size=train_params["batch_size"], device=device,
                           dependencies=dependencies)
    train_reference_started = time.perf_counter()
    train_val_errors = numpy.concatenate([
        compute_absolute_errors(
            best_module,
            dependencies.torch.cat((train_tensor, validation_tensor), dim=0),
            **error_arguments,
        )
        for train_tensor, validation_tensor in zip(train_tensors, validation_tensors)
    ])
    train_reference_inference_seconds = time.perf_counter() - train_reference_started
    test_inference_started = time.perf_counter()
    test_errors_by_session = [
        compute_absolute_errors(best_module, test_tensor, **error_arguments)
        for test_tensor in test_tensors
    ]
    test_inference_seconds = time.perf_counter() - test_inference_started
    timing_path = output_dir / "timing.json"
    with timing_path.open("w", encoding="utf-8") as timing_file:
        json.dump({
            "accelerator": device,
            "training_seconds": training_seconds,
            "train_reference_inference_seconds": train_reference_inference_seconds,
            "test_inference_seconds": test_inference_seconds,
        }, timing_file, ensure_ascii=False, indent=2)

    epsilon = scoring_config["normalization"]["epsilon"]
    smoothing_window = scoring_config["smoothing"]["window"]

    scores_dir = str(output_dir / "scores")

    # 기본 점수는 학습·validation 오차 통계로 정규화한다.
    median, iqr = estimate_median_iqr(train_val_errors)
    validation_scores = numpy.concatenate([
        apply_median_iqr(
            compute_absolute_errors(best_module, validation_tensor, **error_arguments),
            median,
            iqr,
            epsilon,
        ).max(axis=1)
        for validation_tensor in validation_tensors
    ])
    score_paths = []
    metadata_paths = []
    for series, test_tensor, test_errors in zip(test_series, test_tensors, test_errors_by_session):
        naming_arguments = dict(
            dataset=naming["dataset"], series=series, model=naming.get("model", "GDN"),
            tier=naming["tier"], ratio=naming["ratio"], seed=seed,
        )
        score_paths.extend(save_score_arrays(
            apply_median_iqr(test_errors, median, iqr, epsilon), scores_dir,
            norm_kind="trainnorm", smoothing_window=smoothing_window,
            **naming_arguments,
        ))

        # 부록 대조본은 테스트 세션 자체 통계를 쓴다.
        test_median, test_iqr = estimate_median_iqr(test_errors)
        score_paths.extend(save_score_arrays(
            apply_median_iqr(test_errors, test_median, test_iqr, epsilon), scores_dir,
            norm_kind="testnorm", smoothing_window=smoothing_window,
            **naming_arguments,
        ))
        metadata_paths.append(save_score_metadata(
            scores_dir,
            window_size=window_size,
            test_length=int(test_tensor.shape[0]),
            score_length=int(test_errors.shape[0]),
            label_slice=(window_size, None),
            validation_scores=validation_scores,
            **naming_arguments,
        ))

    # early stopping 스텝 로그
    early_stop_callback = callback_dict["early_stop"]
    early_stopping_log_path = output_dir / "early_stopping_log.json"
    with open(early_stopping_log_path, "w", encoding="utf-8") as log_file:
        json.dump({
            "stopped_epoch": early_stop_callback.stopped_epoch,
            "best_score": float(early_stop_callback.best_score),
            "wait_count": early_stop_callback.wait_count,
            "patience": train_params["early_stop_patience"],
            "min_delta": train_params["early_stop_delta"],
            "max_epochs": train_params["n_epochs"],
        }, log_file, ensure_ascii=False, indent=2)

    verify_git_hashes_unchanged(git_hashes, REPOSITORY_ROOT, fork_path)

    result = {
        "score_paths": score_paths,
        "metadata_paths": tuple(metadata_paths),
        "best_checkpoint_path": best_checkpoint_path,
        "early_stopping_log_path": str(early_stopping_log_path),
        "timing_path": str(timing_path),
        "snapshot_path": snapshot_path,
    }
    if len(metadata_paths) == 1:
        result["metadata_path"] = metadata_paths[0]
    return result
