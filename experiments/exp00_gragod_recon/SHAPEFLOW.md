# exp00 — reshape→permute 두 줄 교체의 완전성 판정 (shape 흐름 추적)

**종합 판정: 불충분(추가 수정 지점: `models/gdn/model.py:117`).** 271·304행을 문자 그대로 `permute(0,2,1)`로만 바꾸면 forward 진입 직후 117행의 `x.view(-1, all_feature)`가 비연속 텐서를 거부해 RuntimeError로 죽는다(2부 e1 실증). 연속성만 보장되면 — 2부 e2에서 `.contiguous()` 부가로 확인 — 그 이후 전체 경로(forward 내부, y·loss, X_true·점수)는 (batch, n_features, window) 전제로 전부 정합하다(1부 정적 추적 + 2부 e2·f·g 실증). 어느 방식으로 연속성을 보장할지는 패치 결정 사항이라 여기서 정하지 않는다.

- 판정일: 2026-08-14
- 검증 대상: GraGODs/GraGOD develop `ec8cd452a410ba903a31beb097a010ba0448c095` / d-ailin/GDN main `9853899da860682669a134e4af315d036aab4eca` (로컬 클론)
- 검증 스크립트: `experiments/exp00_gragod_recon/trace_shape_flow.py`
- GraGOD·RECON.md·VERIFICATION.md·DECISIONS.md 무수정.

표기: `B`=batch, `N`=n_features(=노드 수), `W`=window_size, `D`=embed_dim, `H`=heads, `E`=엣지 수, `k`=topk, `L`=시계열 길이.

---

## 1부 — 정적 추적

### a. GraGOD `GDN.forward` (models/gdn/model.py:100-200)

입력 전제: docstring(105행)은 `[batch_size, node_num, feature_dim]`이라 쓰는데, 여기서 "feature_dim"은 윈도 길이다. GNNLayer가 `GNNLayer(window_size, embed_dim, ...)`로 생성되므로(67-78행) 각 노드의 입력 특징은 그 노드의 윈도이고, 전제는 (B, N, W)가 된다.

| 행 | 코드 요약 | 입력 shape | 출력 shape | 축의 의미 |
|---|---|---|---|---|
| 111 | `x = data.clone().detach()` | (B, N, W) | (B, N, W) | 축 유지. clone은 스트라이드 보존(preserve_format) — 입력이 비연속이면 결과도 비연속 (2부 e1의 실패 원인 연결점) |
| 116 | `batch_num, node_num, all_feature = x.shape` | (B, N, W) | — | node_num=둘째 축=N, all_feature=셋째 축=W |
| 117 | `x = x.view(-1, all_feature).contiguous()` | (B, N, W) | (B·N, W) | 행 b·N+n = 배치 b, 노드 n의 윈도. **view는 연속 텐서 전제** — permute 직후 비연속이면 여기서 RuntimeError (2부 e1 실증) |
| 121-129 | 배치 엣지 캐시 `_get_batch_edge_index` (202-223행) | (2, E) | (2, E·B) | 배치 b 블록에 b·N 오프셋 |
| 131 | `all_embeddings = self.embedding(arange(N))` | — | (N, D) | 행 n = 노드 n의 embedding |
| 133 | `weights_arr = all_embeddings.detach().clone()` | (N, D) | (N, D) | 동일 |
| 134 | `all_embeddings.repeat(batch_num, 1)` | (N, D) | (B·N, D) | repeat는 [0..N-1] 순서를 B번 이어붙임 → 행 b·N+n ↔ 노드 n — 117행의 x 행 배치와 일치 |
| 137-143 | cos-sim 행렬 (learn_graph=True) | (N, D) | (N, N) | [i,j] = 노드 i·j embedding 유사도 |
| 144 | `torch.topk(cos_ji_mat, k, dim=-1)[1]` | (N, N) | (N, k) | 행 i = 노드 i의 이웃 k개(자기 포함 — RECON [F]) |
| 146-155 | gated_edge_index 구성 | (N, k) | (2, N·k) | [0]=j(출발), [1]=i(도착) |
| 158-159 | learn_graph=False 분기 | — | (2, E) | 주입 그래프 그대로 |
| 161-163 | 배치 확장 | (2, N·k) | (2, N·k·B) | 위와 동일한 오프셋 규칙 |
| 165-170 | `gnn_layers[i](x, edge, node_num=N·B, embedding)` | x:(B·N, W), emb:(B·N, D) | (B·N, H·D) | modules.py:105-121(GNNLayer.forward)→GraphLayer: lin W→H·D(169, 210행), self-loop 제거·재추가(213-214행), 메시지 집계(217-222행), view(226행), bias(229행). 행 = (배치, 노드) 유지 |
| 174 | `x = cat(gcn_outs, dim=1)` | (B·N, H·D)×1 | (B·N, H·D) | edge_set_num=1 |
| 175 | `x.view(batch_num, node_num, -1)` | (B·N, H·D) | (B, N, H·D) | 117행 배치의 역변환 — 정합 |
| 177-178 | `node_embeddings = embedding(arange(N))` | — | (N, D) | 행 n = 노드 n |
| 180-183 | `x_reshaped = x.view(B, N, H, D)` | (B, N, H·D) | (B, N, H, D) | H·D를 (H, D)로 분해 — 연속 텐서라 정합 |
| 185-186 | `embeddings_expanded = view(N,1,D).expand(-1,H,-1)` | (N, D) | (N, H, D) | 노드 축 유지 |
| 189-190 | `out = mul(...)` 후 `view(B, N, H·D)` | (B,N,H,D)×(1,N,H,D) | (B, N, H·D) | 노드 n의 GNN 출력 × 노드 n의 embedding — 노드 축 일치 |
| 192-194 | `permute(0,2,1)` → BatchNorm1d(H·D) → relu → `permute(0,2,1)` | (B, N, H·D) | (B, N, H·D) | BN은 채널 축(H·D)에 작용. **permute를 올바르게 쓴 전례가 forward 안에 이미 있음** |
| 196-197 | dropout → `out_layer` | (B, N, H·D) | (B, N, 1) | OutLayer 마지막 Linear→1 (modules.py:38행) |
| 198 | `out = out.view(-1, node_num)` | (B, N, 1) | (B, N) | 열 n = 노드 n의 1-스텝 예측. B·N·1 원소의 행우선 재배열 — 정합 |

축의 의미가 코드만으로 확정되지 않는 행: 없다. 유일한 모호점(105행 docstring의 "feature_dim" 명칭)은 67-78행의 GNNLayer 생성 인자로 W임이 확정된다.

결론(a): forward 본문은 입력이 값까지 올바른 (B, N, W)라는 전제 아래 자기일관적이다. 입력 쪽에 요구하는 것은 117행 view의 연속성 하나뿐이다.

### b. `GDN_PLModule.shared_step` (models/gdn/model.py:265-278)

| 행 | 코드 요약 | 입력 shape | 출력 shape | 축의 의미 |
|---|---|---|---|---|
| 267 | `x, y, _, edge_index = batch` | — | x:(B, W, N), y:(B, 1, N) | datasets/dataset.py:47(x), 48-50(y, horizon=1). 셋째 축 = 채널 |
| 271 | `x.reshape(-1, x.size(2), x.size(1))` | (B, W, N) | (B, N, W) | **교체 대상.** 모양은 (B,N,W)지만 값은 축 의미를 만족하지 않음(VERIFICATION 의혹 1 = 참) |
| 268-275 | `.float().to(device)` | — | 동일 | dtype이 이미 float32면 no-op(레이아웃 보존) |
| 272 | `y.squeeze(1)` | (B, 1, N) | (B, N) | horizon 축 제거뿐 — 채널·시점 혼합 없음 (2부 g 실증) |
| 276 | `out = self(x)` | (B, N, W) | (B, N) | forward 표 a |
| 277 | `loss = criterion(out, y)` | (B,N) vs (B,N) | 스칼라 | 열 n ↔ 노드 n ↔ 채널 n — permute 교체 시 정렬 (2부 g 실증) |

### c. predict 경로 (models/gdn/model.py:292-316 + models/predict.py:121-152)

| 행 | 코드 요약 | 입력 shape | 출력 shape | 축의 의미 |
|---|---|---|---|---|
| model.py:303 | `x = batch[0]` | — | (B, W, N) | 데이터로더 배치의 x |
| model.py:304 | `self(x.reshape(-1, x.size(2), x.size(1)))` | (B, W, N) | (B, N) | **교체 대상.** 271행과 동일 문제 |
| model.py:309 | `predictions = torch.cat(predictions)` | (Bᵢ, N)×배치들 | (L−W, N) | 행 i = 윈도 i의 예측 = 원시계열 시점 W+i의 예측 (dataset.py:71-77: 윈도 수 = L−W) |
| model.py:310 | `predictions[:-1, :]` | (L−W, N) | (L−W−1, N) | 마지막 예측 1개 폐기 (predict.py:141 주석: recon 호환) |
| predict.py:121-123 | `X_true = X_true[start_index − window_size:]` | (L₀, N) | (L, N) | 앞부분 절단. start_index ≥ window_size 강제(predict.py:441-444) |
| predict.py:127-136 | 같은 X_true로 loader 생성 | (L, N) | 윈도 (L−W)개 | 표 b의 x·y 생성과 동일 규칙 |
| predict.py:139 | `X_true = X_true[window_size:−1, :]` | (L, N) | (L−W−1, N) | 행 i = 원시계열 시점 W+i = predictions 행 i의 타깃(dataset.py:48-50) — **행 정렬 성립** |
| model.py:313-316 | `torch.abs(predictions − X_true)` | (L−W−1, N)−(L−W−1, N) | (L−W−1, N) | 열 = 채널. X_true 열 = 데이터 열 = 노드 → permute 교체 시 predictions 열과 일치 (2부 f 실증) |

커밋 `e0c8846`("fix: X_true shape", 2025-04-03) 한 줄 요약: X_true의 마지막 행 절단을 `calculate_anomaly_score` 내부(구 `X_true = X_true[:-1, :]`)에서 `process_dataset`의 슬라이스(`X_true[window_size:-1, :]`, predict.py:139)로 옮기고 shape assert에 X_true를 추가한 커밋 — 행(시간) 정렬 수정이며 채널 축과는 무관.

### d. d-ailin 원본 `GDN.forward` (gdn/models/GDN.py:122-187) 대조

입력: (B, N, W)를 원생성으로 받는다. TimeDataset이 `ft = data[:, i−slide_win:i]`로 (N, W) 윈도를 직접 만들어 stack하므로(gdn/datasets/TimeDataset.py:48, 57-58) 전치 지점 자체가 없고, 저장된 텐서는 연속이다.

| d-ailin 행 | 코드 요약 | shape | GraGOD 대응 행 | 차이 |
|---|---|---|---|---|
| 124 | `x = data.clone().detach()` | (B, N, W) | 111 | 동일 |
| 129 | `batch_num, node_num, all_feature = x.shape` | — | 116 | 동일 |
| 130 | `x = x.view(-1, all_feature).contiguous()` | (B·N, W) | 117 | **동일한 view — 원본은 입력이 항상 연속이라 문제가 표면화되지 않음** |
| 138-139 | 배치 엣지 캐시 | (2, E·B) | 121-129 | 동일 로직 |
| 143-146 | embedding / repeat | (N,D)→(B·N,D) | 131-134 | 동일 |
| 148-157 | cos-sim → topk | (N,N)→(N,k) | 137-144 | 동일 (learn_graph 스위치 없음 — RECON [E]) |
| 161-163 | gated_edge_index | (2, N·k) | 146-155 | 동일 |
| 165-166 | GNNLayer 호출 | (B·N, D) | 165-170 | 동일 (H=1 고정) |
| 171-172 | cat → view(B, N, −1) | (B, N, D) | 174-175 | 동일 |
| 175-176 | `mul(x, embedding(indexes))` | (B, N, D) | 177-190 | GraGOD는 H 일반화만 추가 |
| 178-180 | permute → BN → relu → permute | (B, N, D) | 192-194 | 동일 |
| 182-184 | dp → out_layer → `view(-1, node_num)` | (B, N) | 196-198 | 동일 |

대조 결론(d): GraGOD forward는 heads 일반화를 빼면 원본의 충실한 포팅이고, 원본이 (B, N, W)를 받으므로 포팅 의도는 "forward에 (B, N, W)를 넣는 것"이 맞다. 271·304행이 그 변환을 담당하는 유일한 지점인데, 원본에는 없는 지점이다 — GraGOD의 SlidingWindowDataset이 (W, N)으로 내놓기 때문에 생겼다.

---

## 2부 — 최소 실행 확인

- 환경 고지: 이 환경은 고정 환경이 아니다. torch 2.10.0+cpu / torch_geometric 2.8.0.post1(스크래치 설치) / pytorch_lightning 2.6.5(스크래치 설치; GraGOD 고정본은 저자 포크·torch 2.2.2). 여기서의 판정은 모양·정렬에 한정된다.
- 우회 2건(GraGOD 코드 무수정, 스크립트에 사유 주석 있음): `datasets/config.py`의 dataclass 기본값이 Python 3.13의 강화된 규칙과 충돌해서 — GraGOD는 3.10 고정이라 원 환경에선 정상이다 — 이 검증에서 호출되지 않는 데이터셋 로더 3개 이름만 스텁으로 대체했고, `datasets/dataset.py`·`datasets/graph.py`는 패키지 `__init__`을 거치지 않고 파일 경로로 직접 로드했다. 검증 대상 코드(GDN, GDN_PLModule, get_data_loader, build_fully_connected_edge_index)는 전부 클론의 실제 파일에서 로드했다(출력의 "임포트 확인" 줄).
- 실험 설계: 교체 가설의 대상인 271·304행의 연산만 스크립트가 대체하고, forward·post_process_predictions·calculate_anomaly_score·get_data_loader·X_true 슬라이스(predict.py:121-139의 문장 재현)는 실제 코드를 그대로 사용.

### 판정 요약

- **e1 (문자 그대로 permute만): 실패** — `models/gdn/model.py:117`의 view가 비연속 텐서를 거부해 RuntimeError. 두 줄 교체만으로는 불충분하다는 직접 증거다.
- **e2 (연속화 부가): 성공** — forward가 돌고 출력 shape (2, 3) = (batch, n_features).
- **f (채널 정렬): 정합** — 채널 2에만 준 오프셋 500이 실제 X_true 경로를 거쳐 `calculate_anomaly_score` 출력의 채널 2 자리에만 정확히 나타났다. 모델 파라미터를 0으로 고정해 예측을 0으로 만든 상태다(사유는 스크립트 주석에).
- **g (y 정렬): 정합** — y는 `squeeze(1)` 하나만 겪고, loss 피연산자 (B, N)이 원시계열의 시점 W+i 행과 원소 단위로 일치한다. 채널·시점 혼합이 없다.

### 실행 출력 (그대로)

```text
[환경] 이 실행 환경은 GraGOD 고정 환경이 아니다. 판정은 모양·정렬에 한정된다.
  torch              : 2.10.0+cpu
  torch_geometric    : 2.8.0.post1 (스크래치 용도 설치)
  pytorch_lightning  : 2.6.5 (스크래치 용도 설치; GraGOD 고정본은 저자 포크)
GraGOD 경로: C:/Users/simon/AppData/Local/Temp/claude/c--Users-simon-time-series-TSAD-project/fc8279e7-be23-4534-83c2-f1ed3140e08b/scratchpad/gragod
임포트 확인: GDN <- C:\Users/simon/AppData/Local/Temp/claude/c--Users-simon-time-series-TSAD-project/fc8279e7-be23-4534-83c2-f1ed3140e08b/scratchpad/gragod\models\gdn\model.py
임포트 확인: get_data_loader <- C:/Users/simon/AppData/Local/Temp/claude/c--Users-simon-time-series-TSAD-project/fc8279e7-be23-4534-83c2-f1ed3140e08b/scratchpad/gragod\datasets\dataset.py

========================================================================
[e] permute 입력으로 GDN.forward 실행
========================================================================
입력 (batch, window, feature) = (2, 4, 3)

--- e1: 문자 그대로 permute(0, 2, 1) 만 적용 (비연속 텐서) ---
permute 후 shape: (2, 3, 4), is_contiguous: False
forward 실패 (RuntimeError):
  view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
  (실패 지점: models/gdn/model.py:117 의 x.view(-1, all_feature) — 비연속 텐서에 view 불가)

--- e2: permute(0, 2, 1).contiguous() 적용 ---
변환 후 shape: (2, 3, 4), is_contiguous: True
forward 성공, 출력 shape: (2, 3)
출력 shape == (batch, n_features) == (2, 3) 인가: True

========================================================================
[f] 채널 정렬: 오프셋이 calculate_anomaly_score의 채널 2 자리에 나타나는가
========================================================================
입력 시계열 shape: (30, 3), 채널2만 500.0, 나머지 0.0
Using all 26 windows
실제 경로로 만든 X_true shape: (25, 3) (= (L - window_size - 1, n_features) = (25, 3))
X_true 첫 행: [0.0, 0.0, 500.0]

calculate_anomaly_score 출력 shape: (25, 3)
점수 첫 5행:
tensor([[  0.,   0., 500.],
        [  0.,   0., 500.],
        [  0.,   0., 500.],
        [  0.,   0., 500.],
        [  0.,   0., 500.]])
채널별 (min, max):
  채널 0: (0.0, 0.0)
  채널 1: (0.0, 0.0)
  채널 2: (500.0, 500.0)

오프셋 500의 흔적이 채널 2 자리에만 정확히 나타나는가: True

========================================================================
[g] y 정렬: __getitem__ 의 y 가 loss 피연산자가 되기까지
========================================================================
Using all 8 windows
__getitem__ 배치: x shape (4, 4, 3) (batch, window, feature), y shape (4, 1, 3) (batch, horizon=1, feature)

y.squeeze(1) 후 shape: (4, 3)
y 피연산자 값:
tensor([[40., 41., 42.],
        [50., 51., 52.],
        [60., 61., 62.],
        [70., 71., 72.]])
기대값 (원시계열의 시점 W+i 행):
tensor([[40., 41., 42.],
        [50., 51., 52.],
        [60., 61., 62.],
        [70., 71., 72.]])
y 피연산자 == 기대값 (채널·시점 혼합 없음): True

loss 피연산자 shape: out (4, 3) vs y (4, 3)
criterion(MSELoss) 계산 성공, loss = 3261.6667
```

---

## 불충분 판정의 추가 수정 지점 (패치 내용은 결정하지 않음)

- `models/gdn/model.py:271` — 교체 대상 1 (shared_step).
- `models/gdn/model.py:304` — 교체 대상 2 (predict_step).
- `models/gdn/model.py:117` — `x.view(-1, all_feature)`가 비연속 입력을 거부한다. permute 결과의 연속성이 271·304 쪽에서 보장되지 않으면 여기서 죽는다(2부 e1). 연속성을 두 교체 지점에서 보장할지, 117행 쪽을 손볼지는 패치 결정 사항이다.

그 외 경로(272행 y, 307-311행 post_process, 313-316행 점수, predict.py:121-139 X_true)는 추가 수정이 필요 없다 — 1부 표와 2부 f·g로 확인했다.
