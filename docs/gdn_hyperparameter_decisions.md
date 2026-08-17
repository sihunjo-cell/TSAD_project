# GDN 하이퍼파라미터 결정

3b-0B에서는 실제 학습을 돌리지 않았다. 원 논문, d-ailin/GDN `9853899da860682669a134e4af315d036aab4eca`, 고정 GraGOD 포크 `485e26b0c6b1d63f4f3531c8d05597db82e9db29`의 값이 다른 항목을 대조한 뒤 모든 데이터 비율에 같은 설정을 쓰도록 봉인했다. 데이터 양만 바꾸려는 실험에서 비율마다 구조나 최적화 조건을 조정하지 않는다.

## 확인한 원본

- d-ailin/GDN `run.sh:3-18`: batch 32, window 5, embedding 64, output layer 1개, output hidden 128, topk 5, weight decay 0, epoch 30.
- d-ailin/GDN `main.py:183-197`: 기본값은 batch 128, epoch 100, window 15, embedding 64, output layer 1개, output hidden 256, validation 0.1, topk 20이다.
- d-ailin/GDN `train.py:27, 44, 86-95`: Adam 학습률 0.001과 early stopping patience 15를 코드에 고정한다. `min_delta`는 없어서 validation loss가 조금이라도 낮아지면 개선으로 센다.
- d-ailin/GDN `models/GDN.py:76-105, 138-150`: heads 1, dropout 0.2, learned TopK 그래프를 사용한다.
- GDN 논문 §4.4 Implementation Details: window 5, Adam 학습률 0.001, betas `(0.9, 0.99)`, 최대 50 epoch, patience 10이다. embedding·k·hidden은 WADI에서 `128·30·128`, SWaT에서 `64·15·64`로 달리 둔다.
- 고정 GraGOD 포크 `models/gdn/params.yaml:1-32`: 기본 설정은 batch 32, validation 0.1, 학습률 0.001, weight decay 0, Adam epsilon `1e-8`, betas `(0.9, 0.99)`, embedding 64, output layer 1개, output hidden 128, heads 1, negative slope 0.2다.
- 고정 GraGOD 포크 `models/gdn/params_swat.yaml:1-36`, `params_telco.yaml:1-32`, `params_ute.yaml:1-32`: 데이터셋별 튜닝값은 window 5·155·55, batch 512·512·32, epoch 200처럼 서로 다르다.
- 고정 GraGOD 포크 `models/gdn/model.py:44-56, 136-159`: `learn_graph=True`가 기본이며 `topk`는 forward에서 학습된 embedding 유사도 그래프를 만든다.
- 고정 GraGOD 포크 `gragod/training/trainer.py:119-138, 206-228`: Adam 뒤에
  `ReduceLROnPlateau(factor=0.5, patience=8)`를 붙이고 gradient clipping `1.0`을
  적용한다. `datasets/dataset.py:23-37`의 forecast horizon 기본값은 1이다. 이 값은
  `configs/gdn_hyperparams.yaml`의 `fork_contract`에 기록하고 포크 hash로 보증한다.

## 확정값

| 항목 | 값 | 선택 이유 |
| --- | ---: | --- |
| `window_size` | 5 | 원 논문 두 데이터셋과 d-ailin `run.sh`가 일치한다. |
| `embed_dim` | 64 | d-ailin 기본·실행값과 GraGOD 기본값이다. 데이터셋별로 구조를 바꾸지 않는다. |
| `out_layer_num` | 1 | d-ailin 기본·실행값과 GraGOD 기본값이다. |
| `out_layer_inter_dim` | 128 | d-ailin `run.sh`와 GraGOD 기본값이다. |
| `heads` | 1 | d-ailin 구현과 GraGOD 기본값이다. |
| `dropout` | 0.2 | d-ailin 구현의 고정값이며 GraGOD SWaT 설정과 같다. |
| `negative_slope` | 0.2 | GraGOD GDN 생성자와 모든 제공 설정이 같다. |
| `learn_graph` | true | 원 GDN은 항상 그래프를 학습한다. `false`는 별도 −TOPK 절제 조건에서만 쓴다. |
| GHL·HAI `topk` | 5·22 | D-17의 `max(5, ceil(0.25N))`에 `N=19·86`을 대입했다. 0.25는 원 논문의 WADI `30/127=23.6%`와 SWaT `15/51=29.4%` 사이에 둔 사전 기준이고, 하한 5는 d-ailin `run.sh`의 실행값이다. 데이터에서 고른 최적값이 아니며 GHL `k={2,5,10}` 민감도로 의존도를 따로 확인한다. |
| `batch_size` | 32 | d-ailin 실행 스크립트와 GraGOD 기본 설정이 일치한다. |
| `n_epochs` | 50 | 원 논문의 최대 epoch다. patience 10으로 먼저 멈출 수 있다. |
| `init_lr` | 0.001 | 논문, d-ailin, GraGOD가 모두 같다. |
| `weight_decay` | 0 | d-ailin과 GraGOD 기본 설정이다. 데이터셋 튜닝값 `1e-5`는 옮기지 않는다. |
| `eps` | 1e-8 | GraGOD Adam 기본 설정이다. |
| `betas` | `(0.9, 0.99)` | 원 논문과 GraGOD 기본 설정이 같다. |
| `early_stop_patience` | 10 | 원 논문과 GraGOD 제공 데이터셋 설정이 같다. |
| `early_stop_delta` | 0.0 | d-ailin의 “loss가 조금이라도 낮으면 개선” 판정을 Lightning에서 그대로 표현한다. |
| `val_size` | 0.1 | 계획서 7-2, d-ailin 기본, GraGOD 기본값이 같다. |
| `min_train_length` | 6 | `W=5`에서 train window가 하나 이상 생기는 최소 길이 `W+1`이다. 3b-0A에서 모든 조건이 이를 넘었다. |
| train `shuffle` | true | d-ailin train loader와 GraGOD train 설정이 같다. 시계열 순서는 각 window 안에서 보존된다. |
| validation `shuffle` | false | GraGOD `models/train.py:130-138`은 train 값을 그대로 넘긴다. 평가 순서를 고정하려고 우리 러너에서 false로 분리한다. |
| `log_every_n_steps` | 1 | GraGOD 제공 설정의 공통값이다. |
| GHL·HAI seed | `1~3`·`1~10` | 계획서의 GHL seed 3회와 HAI seed 10회 규격이다. |

## 적용 경계

`configs/gdn_hyperparams.yaml`은 값의 원본이다. 3b-1에서 단일 러너 호출 config를 만들 때 공통 `model_params`를 복사하고 `topk_by_dataset[dataset]`을 정수 `topk`로 바꿔 넣는다. `seeds_by_dataset`은 배치 단계만 읽으며 모델 생성자에는 전달하지 않는다.

대조 팔도 같은 파일의 `analysis_arms.GHL`이 원본이다. −TOPK는 GHL 25개 전수의
10%·100%, seed 1~3에서 `learn_graph=false`로 실행한다. TopK 민감도는 같은 조건의
`k={2,5,10}`이며 `k=5`는 주 실행을 재사용한다. 모델 식별자는 각각
`GDN_NOTOPK`, `GDN_K2`, `GDN_K10`이다.

GraGOD의 데이터셋별 200 epoch·batch 512·window 155 같은 값은 해당 데이터셋 튜닝 결과라 GHL·HAI에 옮기지 않는다. 실험 결과를 본 뒤 이 값을 고르는 것도 금지한다. 설정을 바꿀 근거가 새로 나오면 모델 결과를 보기 전에 D-22를 개정한다.
