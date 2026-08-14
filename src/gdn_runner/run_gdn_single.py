"""GDN 단일 실행 러너 — 분할 완료 배열 1개를 받아 학습→점수 저장까지.

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

import numpy
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.normalization import apply_median_iqr, estimate_median_iqr
from src.common.save_scores import save_score_arrays, save_score_metadata, snapshot_config
from src.data_split.validation_split import validation_split


def read_git_hash(repository_dir) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=repository_dir, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown(커밋 없음)"


def compute_absolute_errors(lightning_module, series_tensor, edge_index, window_size, batch_size, device):
    """한 구간의 채널별 절대 오차 (n, n_features) — models/predict.py:55-68·121-139 절차.

    post_process_scores 는 호출하지 않는다(D-05·D-06) — |오차|(model.py:313-316)까지만.
    """
    import pytorch_lightning
    import torch
    from datasets.dataset import get_data_loader  # 포크 datasets/dataset.py:97-145
    from gragod import CleanMethods

    X_true = series_tensor  # start_index = window_size(최솟값) → 그대로 (predict.py:121-124)
    loader = get_data_loader(  # predict.py:127-136 과 동일 인자 구성
        X=X_true, edge_index=edge_index, y=torch.zeros(X_true.shape[0]),
        window_size=window_size, clean=CleanMethods.NONE,
        batch_size=batch_size, n_workers=0, shuffle=False,
    )
    X_true = X_true[window_size:-1, :]  # predict.py:139

    predict_trainer = pytorch_lightning.Trainer(  # predict.py:55 상당 (로그·ckpt 비활성만 추가)
        accelerator=device, logger=False, enable_checkpointing=False, enable_progress_bar=False,
    )
    output = predict_trainer.predict(lightning_module, loader)
    errors = lightning_module.calculate_anomaly_score(predict_output=output, X_true=X_true)
    return errors.detach().cpu().numpy()


def run_gdn_single(config: dict, train_array: numpy.ndarray, test_array: numpy.ndarray,
                   output_dir: str, seed: int) -> dict:
    """분할 완료 학습 배열 + 테스트 배열 1쌍으로 GDN 1회 실행. 산출 경로 dict 반환."""
    fork_path = str(config["fork_path"])
    if fork_path not in sys.path:
        sys.path.insert(0, fork_path)

    import torch
    from torch import nn
    from pytorch_lightning.loggers import TensorBoardLogger

    from datasets.dataset import get_data_loader
    from datasets.graph import build_fully_connected_edge_index  # graph.py:46-66
    from gragod import CleanMethods, Models
    from gragod.models import get_model_and_module  # gragod/models.py:7-31
    from gragod.training import set_seeds  # gragod/training/main.py:18-28
    from gragod.training.callbacks import get_training_callbacks  # callbacks.py:11-59
    from gragod.training.trainer import TrainerPL  # trainer.py:141-233
    from gragod.utils import set_device  # utils.py:83-92

    set_seeds(seed)  # D-11: train.py:223 의 42 하드코딩은 실행되지 않는다
    device = set_device()
    output_dir = Path(output_dir)
    train_params = config["train_params"]

    # validation 은 축소된 학습 구간 내부에서 (S2 모듈, 계획서 7-2)
    train_part, val_part = validation_split(
        numpy.asarray(train_array),
        val_fraction=train_params["val_size"],
        min_train_length=train_params["min_train_length"],
    )
    train_tensor = torch.tensor(train_part, dtype=torch.float32)
    val_tensor = torch.tensor(val_part, dtype=torch.float32)
    reduced_tensor = torch.tensor(numpy.asarray(train_array), dtype=torch.float32)
    test_tensor = torch.tensor(numpy.asarray(test_array), dtype=torch.float32)

    # 초기 그래프 — learn_graph=True 여도 필수 (ORCHESTRATION.md, model.py:44-78)
    edge_index = build_fully_connected_edge_index(train_tensor, device)

    model_params = dict(config["model_params"])
    window_size = model_params["window_size"]
    model_params["edge_index"] = [edge_index]  # train.py:143-145
    model_params["n_features"] = train_tensor.shape[1]
    model_params["out_dim"] = train_tensor.shape[1]

    loader_arguments = dict(
        edge_index=edge_index, window_size=window_size, clean=CleanMethods.NONE,
        batch_size=train_params["batch_size"], n_workers=0,
        shuffle=train_params["shuffle"],  # train.py:119-139 — 같은 값이 train·val 양쪽에
    )
    train_loader = get_data_loader(X=train_tensor, y=torch.zeros(train_tensor.shape[0]), **loader_arguments)
    val_loader = get_data_loader(X=val_tensor, y=torch.zeros(val_tensor.shape[0]), **loader_arguments)

    model_class, module_class = get_model_and_module(Models.GDN)  # train.py:141
    logger = TensorBoardLogger(  # train.py:147-149
        save_dir=str(output_dir / "training"), name="gdn", default_hp_metric=False,
    )
    callback_dict = get_training_callbacks(  # train.py:151-160
        log_dir=logger.log_dir, model_name="gdn",
        monitor="Loss/val", monitor_mode="min",
        early_stop_patience=train_params["early_stop_patience"],
        early_stop_delta=train_params["early_stop_delta"],
        save_top_k=1,
    )
    model_instance = model_class(**model_params).to(device)  # train.py:171-173
    trainer = TrainerPL(  # train.py:175-193 인자 전부
        model=model_instance, model_pl=module_class, model_params=model_params,
        criterion=nn.MSELoss(),  # train.py:162-169 — GDN 은 단일 MSELoss
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

    # 채널별 절대 오차: 학습/검증 구간(축소 배열 전체)과 test
    error_arguments = dict(edge_index=edge_index, window_size=window_size,
                           batch_size=train_params["batch_size"], device=device)
    train_val_errors = compute_absolute_errors(best_module, reduced_tensor, **error_arguments)
    test_errors = compute_absolute_errors(best_module, test_tensor, **error_arguments)

    with open(REPOSITORY_ROOT / "configs" / "scoring_pipeline.yaml", encoding="utf-8") as scoring_file:
        scoring_config = yaml.safe_load(scoring_file)
    epsilon = scoring_config["normalization"]["epsilon"]  # D-07
    smoothing_window = scoring_config["smoothing"]["window"]  # D-04

    naming = config["naming"]
    naming_arguments = dict(dataset=naming["dataset"], series=naming["series"], model="GDN",
                            tier=naming["tier"], ratio=naming["ratio"], seed=seed)
    scores_dir = str(output_dir / "scores")

    # trainnorm(기본, D-06): 학습/검증 구간 통계로 test 점수 정규화
    median, iqr = estimate_median_iqr(train_val_errors)
    trainnorm_paths = save_score_arrays(
        apply_median_iqr(test_errors, median, iqr, epsilon), scores_dir,
        norm_kind="trainnorm", smoothing_window=smoothing_window, **naming_arguments)

    # testnorm(부록 대조, D-15): 저자 원본 방식 — test 구간 자체 통계 (RECON [D])
    test_median, test_iqr = estimate_median_iqr(test_errors)
    testnorm_paths = save_score_arrays(
        apply_median_iqr(test_errors, test_median, test_iqr, epsilon), scores_dir,
        norm_kind="testnorm", smoothing_window=smoothing_window, **naming_arguments)

    save_score_metadata(scores_dir, window_size=window_size, test_length=int(test_tensor.shape[0]), score_length=int(test_errors.shape[0]), label_slice=(window_size, -1), **naming_arguments)  # docs/score_interface.md (i)

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

    # 스냅숏: config 전체 + 두 저장소 git hash (CLAUDE.md 재현성)
    snapshot_path = snapshot_config(
        {"run_config": config, "seed": seed,
         "git_hashes": {"tsad_project": read_git_hash(REPOSITORY_ROOT),
                        "gragod_fork": read_git_hash(fork_path)}},
        git_hash=read_git_hash(REPOSITORY_ROOT),
        output_dir=str(output_dir / "snapshots"),
    )

    return {
        "score_paths": trainnorm_paths + testnorm_paths,
        "best_checkpoint_path": best_checkpoint_path,
        "early_stopping_log_path": str(early_stopping_log_path),
        "snapshot_path": snapshot_path,
    }
