# exp00 — 오케스트레이션 정찰 보강 (S3b 1부)

- 조사일: 2026-08-14
- 대상: 패치 포크 `../gragod-fork/` (브랜치 `fix/gdn-input-transform`, 커밋 `485e26b0`) —
  `models/train.py:115-204`를 실제로 열어 확인했다. 패치는 `models/gdn/model.py` 271·304행뿐이라
  train.py·trainer.py·callbacks.py·graph.py는 베이스 `ec8cd452`와 같다는 것도 확인했다.
- 목적: 우리 러너(`src/gdn_runner/run_gdn_single.py`)가 복제할 조립 절차의 명세.

## 조립 절차 표 (models/train.py:115-204)

| train.py 행 | 하는 일 | 호출 함수와 위치 | 우리 러너에서의 대체 방식 |
|---|---|---|---|
| 95 | device 결정 (cuda→mps→cpu 자동) | `set_device` — gragod/utils.py:83-92 | 동일 함수 호출 |
| 100-111 | 데이터 로드 (Datasets enum → 파일) | `load_training_data` — gragod/training/main.py:52-114 | **사용 안 함(D-02)** — 호출자가 넘긴 분할 완료 배열을 직접 사용. validation 분할은 S2 `validation_split` |
| 115-117 | 초기 그래프 생성 | `get_edge_index` — datasets/graph.py:5-43 (edge_index_path 없으면 fully-connected, 42-43행 → `build_fully_connected_edge_index` 46-66행) | `build_fully_connected_edge_index` 직접 호출. learn_graph=True여도 초기 그래프는 필수다 — GDN 생성자가 `edge_index` 리스트를 요구하고(models/gdn/model.py:44-61) `edge_set_num=len(edge_index)`가 GNNLayer 개수를 정한다(66-78행). forward에서 topk 그래프로 대체되지만(136-156행) 구조 정의에 초기 그래프가 쓰인다. False면 이 그래프가 그대로 사용된다(158-159행) |
| 119-139 | train/val 로더 생성 | `get_data_loader` — datasets/dataset.py:97-145 | 동일 함수에 우리 텐서 직접 주입(D-02). 주의: train.py는 같은 `shuffle` 값을 train·val 로더 양쪽에 전달한다(127, 138행) — 원형 그대로 복제 |
| 123 | window_size 주입 | `model_params["window_size"]` — yaml `model_params.window_size` | config에서 동일 키 |
| 141 | 모델·PL모듈 클래스 획득 | `get_model_and_module` — gragod/models.py:7-31 (GDN이면 24-27행: `models/gdn/model.py`의 GDN, GDN_PLModule) | 동일 함수 호출 (인자 `Models.GDN`) |
| 143-145 | model_params 보강 | `edge_index=[edge_index]`, `n_features=X_train.shape[1]`, `out_dim=X_train.shape[1]` | 동일하게 조립. 모델 생성 하이퍼파라미터의 yaml 출처: `model_params.{window_size, embed_dim, out_layer_num, out_layer_inter_dim, topk, heads, dropout, negative_slope, learn_graph}` (models/gdn/params_swat.yaml:1-11 형태) — 우리는 configs/gdn_hyperparams.yaml의 동일 키. **k는 호출자가 계산한 정수**(규칙 문자열 아님) |
| 147-149 | TensorBoardLogger 생성 | `TensorBoardLogger(save_dir=log_dir, name=model_name, default_hp_metric=False)` | 동일 — save_dir는 output_dir 아래로. `logger.log_dir` = `{save_dir}/{name}/version_{n}` (자동 버저닝) |
| 151-160 | 콜백 3종 구성 | `get_training_callbacks` — gragod/training/callbacks.py:11-59. EarlyStopping이 존재한다(35-41행): monitor·min_delta=`early_stop_delta`·patience=`early_stop_patience`(yaml `train_params.early_stop_{patience,delta}`). ModelCheckpoint(44-50행)는 monitor="Loss/val"(train.py 기본 인자 45행 `monitor: str = "Loss/val"`), dirpath=`logger.log_dir`, filename="best" → best.ckpt 경로 = `{log_dir}/{model_name}/version_{n}/best.ckpt`. LearningRateMonitor(53행) | 동일 함수 호출. best.ckpt 경로는 규칙 대신 `checkpoint_cb.best_model_path` 속성으로 취득한다(버저닝 오차 원천 차단) |
| 162-169 | 손실 함수 | GDN은 else 분기(168행): `nn.MSELoss()` 단일 | 동일 |
| 171-173 | 모델 인스턴스화 | `model_class(**model_params).to(device)` | 동일 |
| 175-193 | TrainerPL 생성 | `TrainerPL` — gragod/training/trainer.py:141-204. 인자 전부: model, model_pl, model_params, criterion, batch_size, n_epochs, init_lr, device, log_dir, callbacks, checkpoint_cb, logger, target_dims, log_every_n_steps, weight_decay, eps, betas. LR 스케줄러는 인자가 아니라 PLBaseModule.configure_optimizers에 하드코딩돼 있다(trainer.py:119-138: Adam(lr·weight_decay·eps·betas는 yaml `train_params` 출처) + ReduceLROnPlateau factor=0.5 patience=8 monitor="Loss/val"). gradient_clip_val=1.0도 TrainerPL.fit 하드코딩이다(trainer.py:218) | 동일하게 생성 — 스케줄러·클립은 자동 상속(수정 불가·불필요) |
| 195-196 | 체크포인트 이어 학습 | `trainer.load(ckpt_path_resume)` | 사용 안 함 (신규 학습만) |
| 198-204 | args_summary 조립 + 학습 | `trainer.fit(train_loader, val_loader, args_summary)` — trainer.py:206-228 | 동일. 확인된 제약: `n_epochs=1`이면 fit 말미(trainer.py:223-227)가 `best_metrics=None`으로 죽는다 — best_metrics는 epoch 시작 훅에서만 세팅되는데(models/gdn/model.py:241-248 + trainer.py:101-117) 첫 epoch 시작 시점엔 best_model_score가 없다. 러너는 n_epochs ≥ 2를 요구한다 |
| 207-210 | args_summary JSON 저장 | `json.dump(...)` — logger.log_dir | 우리는 `snapshot_config`(src/common) — config + 두 저장소 git hash |
| (train.py 밖) 222-223 | 시드 고정 — train()이 아니라 main()에서 `set_seeds(RANDOM_SEED=42)` (train.py:22, 223) | `set_seeds` — gragod/training/main.py:18-28 (torch·cuda·cudnn.deterministic·pl.seed_everything) | D-11: 러너 첫 줄에서 `set_seeds(우리 시드)`를 직접 호출한다 — train()을 우회하므로 42 하드코딩은 아예 실행되지 않는다 |

## 예측 경로 보강 (러너의 점수 산출부, models/predict.py 대조)

| predict.py 행 | 하는 일 | 우리 러너에서의 대체 방식 |
|---|---|---|
| 363-366 | `GDN_PLModule.load_from_checkpoint(best_ckpt, map_location=device)` | 동일 (checkpoint_cb.best_model_path 사용) |
| 121-124 | `start_index` 앞자름 | start_index = window_size(최솟값, predict.py:441-444) → 원배열 그대로 |
| 127-136 | 점수용 로더 (clean=NONE, shuffle=False) | 동일 — 학습/검증 구간(축소 배열 전체)과 test 각각 |
| 139 | `X_true = X_true[window_size:-1, :]` | 동일 슬라이스 |
| 55-68 | `trainer.predict` → `calculate_anomaly_score` → (post_process_scores 분기) | predict 후 `calculate_anomaly_score`(models/gdn/model.py:313-316, \|오차\|)까지만. post_process_scores는 호출하지 않는다(D-05·D-06 금지) — 정규화·smoothing은 src/common 소유 |

## 확인 불가 항목

- 없다 — 위 표의 전 항목을 포크 파일에서 직접 확인했다.
