"""GDN 실행 러너 — 분할 완료 세션을 받아 학습→점수 저장까지.

근거: ORCHESTRATION.md 조립 절차 표(models/train.py:115-204 대조), D-02(자체 러너,
get_data_loader 직접 주입), D-05·D-06(GraGOD post_process_scores 미호출 — 정규화·
smoothing 은 src/common 소유), D-11(set_seeds 를 우리 시드로 직접 호출), D-15(trainnorm/
testnorm 병행 저장). GraGOD 는 패치 포크(../gragod-fork)를 임포트로만 사용한다.

이 함수는 front_trim 을 호출하지 않는다 — 입력은 이미 분할·전처리된 배열이다.
분할은 호출자(추후 배치 스크립트) 소관이며, T의 정의가 EDA 의존(E1·E4)이기 때문이다.

epsilon·smoothing 창은 configs/scoring_pipeline.yaml 소유(D-07). topk 는 규칙
문자열이 아니라 호출 시점에 계산된 정수를 받는다 — 계산 책임은 호출자(D-17).

확인된 제약(ORCHESTRATION.md): TrainerPL.fit 은 n_epochs >= 2 를 요구한다
(trainer.py:223-227 — 1 epoch 이면 best_metrics 미기록으로 실패).
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.normalization import apply_median_iqr, estimate_median_iqr
from src.common.save_scores import save_score_arrays, save_score_metadata, snapshot_config
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
    from datasets.graph import build_fully_connected_edge_index  # graph.py:46-66
    from gragod import CleanMethods, Models
    from gragod.models import get_model_and_module  # gragod/models.py:7-31
    from gragod.training import set_seeds  # gragod/training/main.py:18-28
    from gragod.training.callbacks import get_training_callbacks  # callbacks.py:11-59
    from gragod.training.trainer import TrainerPL  # trainer.py:141-233
    from gragod.utils import set_device  # utils.py:83-92

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


def read_git_hash(repository_dir) -> str:
    repository_dir = Path(repository_dir).resolve()
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repository_dir}", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=repository_dir, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown(커밋 없음)"


def compute_absolute_errors(
    lightning_module, series_tensor, edge_index, window_size, batch_size, device, dependencies,
):
    """한 구간의 채널별 절대 오차 (n, n_features) — models/predict.py:55-68·121-139 절차.

    post_process_scores 는 호출하지 않는다(D-05·D-06) — |오차|(model.py:313-316)까지만.
    """
    X_true = series_tensor  # start_index = window_size(최솟값) → 그대로 (predict.py:121-124)
    loader = dependencies.get_data_loader(  # 포크 predict.py:127-136과 동일 인자 구성
        X=X_true, edge_index=edge_index, y=dependencies.torch.zeros(X_true.shape[0]),
        window_size=window_size, clean=dependencies.CleanMethods.NONE,
        batch_size=batch_size, n_workers=0, shuffle=False,
    )
    X_true = X_true[window_size:-1, :]  # predict.py:139

    predict_trainer = dependencies.PredictionTrainer(  # predict.py:55 상당
        accelerator=device, logger=False, enable_checkpointing=False, enable_progress_bar=False,
    )
    output = predict_trainer.predict(lightning_module, loader)
    errors = lightning_module.calculate_anomaly_score(predict_output=output, X_true=X_true)
    return errors.detach().cpu().numpy()


def build_session_loader(
    session_tensors, edge_index, window_size, batch_size, shuffle, dependencies,
):
    """세션별 window dataset만 합친다. raw tensor 연결은 금지한다(D-24)."""
    datasets = [
        dependencies.SlidingWindowDataset(
            data=session_tensor,
            edge_index=edge_index,
            window_size=window_size,
            labels=dependencies.torch.zeros(session_tensor.shape[0]),
            drop=False,
        )
        for session_tensor in session_tensors
    ]  # GraGOD datasets/dataset.py:7-77
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

    naming = config["naming"]
    series_value = naming["series"]
    test_series = (series_value,) if isinstance(series_value, int) else tuple(series_value)
    if len(test_series) != len(test_sessions):
        raise ValueError("naming.series 수와 test 세션 수가 다르다")

    fork_path = str(config["fork_path"])
    dependencies = load_gdn_dependencies(fork_path)
    dependencies.set_seeds(seed)  # D-11: train.py:223의 42 하드코딩은 실행되지 않는다
    device = dependencies.set_device()
    output_dir = Path(output_dir)
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

    # 초기 그래프 — learn_graph=True 여도 필수 (ORCHESTRATION.md, model.py:44-78)
    edge_index = dependencies.build_fully_connected_edge_index(train_tensors[0], device)

    model_params = dict(config["model_params"])
    window_size = model_params["window_size"]
    model_params["edge_index"] = [edge_index]  # train.py:143-145
    model_params["n_features"] = feature_count
    model_params["out_dim"] = feature_count

    loader_arguments = dict(
        edge_index=edge_index,
        window_size=window_size,
        batch_size=train_params["batch_size"],
        shuffle=train_params["shuffle"],  # GraGOD train.py:119-139와 같은 설정
        dependencies=dependencies,
    )
    train_loader = build_session_loader(train_tensors, **loader_arguments)
    val_loader = build_session_loader(validation_tensors, **loader_arguments)

    model_class, module_class = dependencies.get_model_and_module(dependencies.Models.GDN)  # train.py:141
    logger = dependencies.TensorBoardLogger(  # train.py:147-149
        save_dir=str(output_dir / "training"), name="gdn", default_hp_metric=False,
    )
    callback_dict = dependencies.get_training_callbacks(  # train.py:151-160
        log_dir=logger.log_dir, model_name="gdn",
        monitor="Loss/val", monitor_mode="min",
        early_stop_patience=train_params["early_stop_patience"],
        early_stop_delta=train_params["early_stop_delta"],
        save_top_k=1,
    )
    model_instance = model_class(**model_params).to(device)  # train.py:171-173
    trainer = dependencies.TrainerPL(  # train.py:175-193 인자 전부
        model=model_instance, model_pl=module_class, model_params=model_params,
        criterion=dependencies.MSELoss(),  # train.py:162-169 — GDN은 단일 MSELoss
        batch_size=train_params["batch_size"], n_epochs=train_params["n_epochs"],
        init_lr=train_params["init_lr"], device=device,
        log_dir=str(output_dir / "training"),
        callbacks=list(callback_dict.values()),
        checkpoint_cb=callback_dict["checkpoint"], logger=logger,
        target_dims=None, log_every_n_steps=train_params["log_every_n_steps"],
        weight_decay=train_params["weight_decay"], eps=train_params["eps"],
        betas=tuple(train_params["betas"]),
    )
    trainer.fit(train_loader, val_loader, args_summary={"seed": seed})  # train.py:204

    # best.ckpt — 경로 규칙 대신 콜백 속성으로 취득 (ORCHESTRATION.md)
    best_checkpoint_path = callback_dict["checkpoint"].best_model_path
    best_module = module_class.load_from_checkpoint(best_checkpoint_path, map_location=device)  # predict.py:363-366
    best_module.eval()

    # trainnorm 통계는 세션 안에서 train+validation을 복원해 계산한 뒤 행축으로만 합친다.
    error_arguments = dict(edge_index=edge_index, window_size=window_size,
                           batch_size=train_params["batch_size"], device=device,
                           dependencies=dependencies)
    train_val_errors = numpy.concatenate([
        compute_absolute_errors(
            best_module,
            dependencies.torch.cat((train_tensor, validation_tensor), dim=0),
            **error_arguments,
        )
        for train_tensor, validation_tensor in zip(train_tensors, validation_tensors)
    ])
    test_errors_by_session = [
        compute_absolute_errors(best_module, test_tensor, **error_arguments)
        for test_tensor in test_tensors
    ]

    with open(REPOSITORY_ROOT / "configs" / "scoring_pipeline.yaml", encoding="utf-8") as scoring_file:
        scoring_config = yaml.safe_load(scoring_file)
    with open(REPOSITORY_ROOT / "configs" / "data_preprocessing.yaml", encoding="utf-8") as preprocessing_file:
        preprocessing_config = yaml.safe_load(preprocessing_file)
    epsilon = scoring_config["normalization"]["epsilon"]  # D-07
    smoothing_window = scoring_config["smoothing"]["window"]  # D-04

    scores_dir = str(output_dir / "scores")

    # trainnorm(기본, D-06): 학습/검증 구간 통계로 test 점수 정규화
    median, iqr = estimate_median_iqr(train_val_errors)
    score_paths = []
    metadata_paths = []
    for series, test_tensor, test_errors in zip(test_series, test_tensors, test_errors_by_session):
        naming_arguments = dict(
            dataset=naming["dataset"], series=series, model="GDN",
            tier=naming["tier"], ratio=naming["ratio"], seed=seed,
        )
        score_paths.extend(save_score_arrays(
            apply_median_iqr(test_errors, median, iqr, epsilon), scores_dir,
            norm_kind="trainnorm", smoothing_window=smoothing_window,
            **naming_arguments,
        ))

        # testnorm(부록 대조, D-15): test 세션 자체 통계
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
            label_slice=(window_size, -1),
            **naming_arguments,
        ))  # docs/score_interface.md (i)

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

    # 스냅숏: 실행에 쓰인 config 전체 + 입력 출처 + 두 저장소 git hash (AGENTS.md 재현성)
    snapshot_path = snapshot_config(
        {"run_config": config,
         "data_preprocessing": preprocessing_config,
         "scoring_pipeline": scoring_config,
         "input": input_metadata,
         "seed": seed,
         "git_hashes": {"tsad_project": read_git_hash(REPOSITORY_ROOT),
                        "gragod_fork": read_git_hash(fork_path)}},
        git_hash=read_git_hash(REPOSITORY_ROOT),
        output_dir=str(output_dir / "snapshots"),
    )

    result = {
        "score_paths": score_paths,
        "metadata_paths": tuple(metadata_paths),
        "best_checkpoint_path": best_checkpoint_path,
        "early_stopping_log_path": str(early_stopping_log_path),
        "snapshot_path": snapshot_path,
    }
    if len(metadata_paths) == 1:
        result["metadata_path"] = metadata_paths[0]
    return result
