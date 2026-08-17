# exp00 — GraGOD 정찰 보고서

- 조사일: 2026-08-14
- 조사 대상:
  - **GraGOD**: `github.com/GraGODs/GraGOD`, `develop` 브랜치, HEAD = `ec8cd452a410ba903a31beb097a010ba0448c095` (2025-06-12)
  - **d-ailin/GDN**: `github.com/d-ailin/GDN`, `main` 브랜치, HEAD = `9853899da860682669a134e4af315d036aab4eca` (2021-11-21)
- 조사 방법: 두 레포를 로컬에 클론해 실제 파일을 열어 확인했다. 아래 줄 번호는 전부 위 커밋 기준이다.
- GDN 구현이 d-ailin/GDN 기반이라는 사실은 GraGOD README에 적혀 있다 (GraGOD `README.md:112`).
- 논문 값은 arXiv 2106.06947 (AAAI 2021)의 ar5iv HTML 렌더 §4.4(Experimental Setup)·§3.6(Graph Deviation Scoring)에서 확인했다. PDF 직접 파싱이 이 환경에서 안 되어 HTML 렌더를 썼고, ar5iv가 수식 숫자를 중복 표기하는 문제는 원문 대조로 보정했다.

---

## [A] 커밋 이력

### develop 최근 커밋 25개

표기: ● = `models/gdn/**` 직접 변경, ○ = GDN 실행·점수 경로가 공유하는 코드(`datasets/`, `gragod/predictions/`, `gragod/training/`, `models/predict.py`) 변경, − = 무관(README·연구노트 등). 변경 파일은 `git diff-tree --no-commit-id --name-only -r <hash>`로 확인.

| # | 해시 | 날짜 | 요약 | GDN | 변경 파일(요지) |
|---|---|---|---|---|---|
| 1 | ec8cd45 | 2025-06-12 | ENTREGAMOS | − | README.md |
| 2 | c8a443e | 2025-06-10 | Update README.md | − | README.md |
| 3 | 084c171 | 2025-06-10 | Update README.md | − | README.md |
| 4 | 9192afa | 2025-06-10 | Update README.md | − | README.md |
| 5 | 11209bc | 2025-06-09 | Merge PR#92 chore/purge-unused-code | ○(머지) | 아래 16~22 병합 |
| 6 | bb673f6 | 2025-06-09 | Merge PR#91 fix/visualization | ○(머지) | 아래 8~11 병합 |
| 7 | 9ca1701 | 2025-05-16 | Update README.md | − | README.md |
| 8 | 5c20f0a | 2025-04-10 | chore: remove unused param | − | gragod/metrics/visualization.py |
| 9 | 7082234 | 2025-04-10 | fix: predict to match visualization | ○ | gragod/metrics/visualization.py, **models/predict.py** |
| 10 | 3182353 | 2025-04-10 | chore: different aggreagtion modes | − | gragod/metrics/visualization.py |
| 11 | 3a49069 | 2025-04-10 | fix: visualization scripts | − | gragod/metrics/visualization.py |
| 12 | 8f0c5e5 | 2025-04-12 | Merge PR#94 fix/f1-optimized-threshold | ○(머지) | 아래 13 병합 |
| 13 | 01bb9f2 | 2025-04-12 | fix: range based f1 thresholding | ○ | gragod/predictions/per_class_threshold_calculator.py, gragod/predictions/system_threshold_calculator.py |
| 14 | 76d9e42 | 2025-04-11 | chore: poetry lock | − | poetry.lock |
| 15 | 1b952bb | 2025-04-10 | chore: update poetry lock | − | poetry.lock |
| 16 | 54d619b | 2025-04-10 | chore: remove 5g3e related code | − | research/datasets/20240518_5g3e.py |
| 17 | 10f940a | 2025-04-10 | chore: remove code related to mihaela dataset | ○ | datasets/__init__.py, gragod/training/main.py |
| 18 | a19c002 | 2025-04-10 | chore: remove temporian and related code | ○ | datasets/telco.py |
| 19 | d898ee5 | 2025-04-10 | chore: remove gdn evaluation file | ● | models/gdn/evaluate.py 삭제 |
| 20 | 7fffbba | 2025-04-10 | chore: remove mihaela dataset logic | ○ | datasets/mihaela.py 삭제 |
| 21 | e08bb25 | 2025-04-10 | chore: purge SPOT prediction file | ○ | gragod/predictions/spot.py 삭제 |
| 22 | 755446e | 2025-04-10 | chore: purge model predict files | ● | models/gdn/predict.py 삭제 |
| 23 | 476d562 | 2025-04-11 | Merge PR#93 analysis-notebooks | − | (노트북) |
| 24 | 5ac17b2 | 2025-04-11 | Merge PR#89 feat/thresholds | ○(머지) | (threshold 계열) |
| 25 | bedbe0d | 2025-04-11 | add ute dataset | ○ | predict.sh, train.sh |

`models/gdn/**`을 건드린 커밋은 develop 전체에서 84개다 (`git log --oneline -- models/gdn/ | wc -l`). 최초 커밋은 `edfb674` 2024-10-16 "feat: basic gdn model code", learn_graph 스위치 도입은 `00096c9` 2025-02-16 "feat: optionally learn graph in gdn".

### models/gdn/ 파일별 마지막 변경 (git log -1 -- <file>)

| 파일 | 마지막 변경 커밋 |
|---|---|
| models/gdn/model.py | `e0c8846` 2025-04-03 "fix: X_true shape" |
| models/gdn/modules.py | `710f12a` 2024-12-01 "fix: typing errors" |
| models/gdn/params.yaml | `6077f0a` 2025-04-05 |
| models/gdn/params_swat.yaml | `6077f0a` 2025-04-05 |
| models/gdn/params_telco.yaml | `6077f0a` 2025-04-05 |
| models/gdn/params_ute.yaml | `4c04e71` 2025-04-11 "update ute params" |
| models/gdn/tune_params.py | `6825ef4` 2024-12-28 |

### GDN 포팅 완결 시점 후보 (해시 확정은 보류 — 선택은 사용자 몫)

1. `ec8cd452a410ba903a31beb097a010ba0448c095` (HEAD, 2025-06-12 "ENTREGAMOS")
   근거: 논문 제출 시점 스냅숏이다("entregamos"=제출했다). `11209bc`(purge 머지, 2025-06-09) 이후 develop의 변경은 전부 README.md뿐임을 diff-tree로 확인했다(위 표 1~4, 7행). 코드 상태는 아래 2번 후보와 같으면서 최종 문서까지 담고 있다.
2. `11209bc` (2025-06-09, Merge PR#92 purge-unused-code)
   근거: 죽은 GDN 파일(models/gdn/evaluate.py, models/gdn/predict.py — 위 표 19·22행)이 제거되어 "실제로 쓰이는 GDN 코드"만 남은 첫 시점이다. 이후 코드 변경이 없다.
3. `e0c8846` (2025-04-03, "fix: X_true shape")
   근거: GDN 모델 본체 `models/gdn/model.py`의 마지막 기능 수정이다. 이 시점 이후 GDN 코어(model.py, modules.py)는 불변인데, 이후 삭제될 죽은 파일들이 아직 남아 있고 threshold 픽스(`01bb9f2`)가 빠져 있다.

---

## [B] 데이터 입력 인터페이스

### 진입점

- 학습 진입점: **`models/train.py`** — CLI `__main__`(234-258행) → `main()`(215-231행, 함수) → `train()`(25-212행, 함수).
  실행 형태: `python models/train.py --model gdn --dataset <swat|telco|ute> --params_file models/gdn/params_<ds>.yaml` (GraGOD `README.md:64-73`, `train.sh:21-27`).
- 예측 진입점: **`models/predict.py`** — `main()`(415-455행) → `predict()`(260-412행).

### 받는 자료형·형태

- `train()`은 배열도 파일 경로도 직접 받지 않는다. `Datasets` enum(`gragod/types.py:7-10`)을 받아 내부에서 `load_training_data()`를 호출한다 (`models/train.py:100-111` → `gragod/training/main.py`, `load_training_data`, 52-114행). 실제 파일 경로는 데이터셋별 로더와 config에 하드코딩 (`datasets/swat.py`, `load_swat_df_split`/`load_swat_training_data`, 11-171행; `datasets/config.py:46-101`).
- `load_training_data` 반환: 6-튜플 `(X_train, X_val, X_test, y_train, y_val, y_test)` (`datasets/swat.py:171`). X는 `torch.float32` 텐서, shape **`(n_samples, n_features)`** (`datasets/data_processing.py`, `preprocess_df`, 178-181행에서 텐서 변환). 라벨 y는 같은 길이의 별도 텐서 — X에 라벨 열이 포함되지 않음.
- 열 순서: CSV 열 순서에서 타임스탬프·드롭 대상 열 제거 후, train 기준으로 val/test 열을 재정렬 (`datasets/swat.py:36-44`, `71-77`).
- 라벨의 용도: 학습 손실에는 쓰이지 않고(`models/gdn/model.py`, `GDN_PLModule.shared_step`, 265-278행 — 라벨 무시), `clean: "drop"`일 때 이상 구간 윈도 제거에만 사용 (`datasets/dataset.py`, `SlidingWindowDataset._get_valid_indices`, 64-94행).
- 윈도잉: `SlidingWindowDataset.__getitem__`(`datasets/dataset.py:44-62`)이 x=`(window_size, n_features)`, y=`(horizon, n_features)`를 반환.
- ⚠ 관찰: `GDN_PLModule.shared_step`/`predict_step`은 x를 `x.reshape(-1, x.size(2), x.size(1))`로 `(batch, n_features, window)`로 바꾼다 (`models/gdn/model.py:271`, `304`). permute(전치)가 아니라 reshape(메모리 재해석)다. 원본 d-ailin은 처음부터 `(node_num, slide_win)` 모양으로 윈도를 만든다 (`gdn/datasets/TimeDataset.py`, `TimeDataset.process`, 48행 `ft = data[:, i-slide_win:i]`). window_size ≠ n_features일 때 reshape는 전치와 결과가 다르므로 정합성 의혹이 있음 — 판단 보류, "내가 결정해야 할 사항" 참조.

### 앞자르기 축소 배열 주입 지점 후보 2개

후보 (i): `datasets/`에 우리 데이터셋 로더를 추가하고 `Datasets` enum 확장
- 수정 지점: `gragod/types.py:7-10`(enum), `datasets/config.py:95-101`(`get_dataset_config` 딕셔너리), `gragod/training/main.py:81-114`(`load_training_data` 분기), + 새 로더 파일 1개.
- 장점: `train()`/`predict()` 파이프라인 전체(콜백·로깅·predict 흐름 포함)를 무변형으로 재사용. GraGOD의 관습과 일치.
- 단점: GraGOD 포크에 최소 3개 파일 수정 → 커밋 고정·패치 관리 필요. 로더가 "파일에서 읽는" 구조라서 축소 비율(r005~r100)마다 파일을 미리 만들어 두거나 로더에 비율 파라미터를 뚫어야 함(추가 침습).

후보 (ii): `train()`을 우회하는 우리 러너 — 텐서를 `get_data_loader()`에 직접 주입
- 주입 지점: `datasets/dataset.py`, `get_data_loader`, 97-145행 — `X: torch.Tensor`를 그대로 받으므로 앞자르기한 텐서를 바로 넣을 수 있음. 이후 `models/train.py:115-204`의 조립 절차(`get_model_and_module` → `TrainerPL.fit`)를 우리 러너에서 복제.
- 장점: GraGOD를 한 줄도 수정하지 않음(임포트만) → 커밋 해시 고정만으로 재현성 확보. 축소 비율·시드를 우리 쪽에서 자유롭게 제어.
- 단점: `models/train.py:115-204`의 오케스트레이션(~90줄)을 복제하므로 GraGOD 쪽이 바뀌면 수동 동기화 필요. edge_index·콜백 구성 실수 여지.

---

## [C] 점수 산출 경로

실제 코드 순서 (예측 시, `models/predict.py` 기준):

`predict()`(260-412행) → 스플릿별 `process_dataset()`(73-257행) → `run_model()`(26-70행):

1. **예측**: `pl.Trainer.predict` → `GDN_PLModule.predict_step` (`models/gdn/model.py:292-305`) — 배치별 1-스텝 예측값 `(batch, n_features)`.
2. **예측 후처리**: `GDN_PLModule.post_process_predictions` (`models/gdn/model.py:307-311`) — 배치 concat 후 마지막 1개 샘플 drop.
3. **오차 → 채널별 점수**: `GDN_PLModule.calculate_anomaly_score` (`models/gdn/model.py:313-316`) — `torch.abs(predictions - X_true)`. 절대 오차가 하드코딩돼 있어 predictor_params의 `score_type`은 GDN에서 무시된다 — `generate_scores`(`gragod/predictions/prediction.py:33-61`)는 GDN 경로에서 호출되지 않는다.
4. **정규화 + smoothing**: `predictor_params["post_process_scores"]`가 true면 `post_process_scores` (`models/predict.py:66-68`에서 호출 → `gragod/predictions/prediction.py:64-82`):
   - 4a. `standarize_error_scores` (85-105행): 채널별 median·IQR 정규화 (상세는 [D]).
   - 4b. `smooth_scores` (108-124행): 이동평균, `window_size_smooth`(기본 5), 왼쪽 replicate 패딩.
5. **채널 집계**: `get_system_scores` (`gragod/predictions/prediction.py:6-30`) — `system_output_mode`에 따라 max/mean/sum. 단, 라벨이 시스템 단위(1열)일 때만 수행 (`models/predict.py:191-199`).
6. **임계값**: `get_thresholds` (`models/predict.py:180-189` → `gragod/predictions/threshold_calculator.py`).

⚠ 우리 예상(오차→정규화→집계→smoothing)과 다른 점: smoothing(4b)이 채널 집계(5)보다 먼저다. 채널별 점수에 smoothing을 하고 나서 집계한다는 뜻이다.

⚠ 관찰(치명 가능성): `smooth_scores`는 `(n_samples, n_features)` 2D 텐서에 `F.pad(scores, (pad,0))` + `avg_pool1d`를 적용한다 (`gragod/predictions/prediction.py:119-124`). 2D 입력에서 이 연산들은 마지막 축, 곧 feature 축에 작용하는 것으로 보인다 — 시간 축이 아니다. 원본 d-ailin은 명확히 시간 축 후행 창으로 smoothing한다 (`gdn/evaluate.py:62-65`). 코드 실행 없이 단정하지 않는다 — 검증 필요, "내가 결정해야 할 사항" 참조.

### raw(smoothing 전) 점수를 꺼내는 지점

- **방법 a (무수정)**: predictor_params에 `post_process_scores: false`를 주면 `run_model`이 4단계를 건너뛰고 3단계 산출물(정규화도 안 된 절대 오차)을 그대로 반환 (`models/predict.py:66-68` 분기; 실제 전례 `models/gdn/params_ute.yaml:40`).
- **방법 b (래퍼)**: `run_model` 반환 직후의 `scores`(`models/predict.py:68-70`)를 우리 코드에서 받아 저장. `run_model`은 공개 함수라 우리 러너에서 직접 호출 가능.
- 주의: "raw"의 정의가 두 가지 가능 — (3단계) 정규화 전 절대 오차 vs (4a 후) 정규화 후·smoothing 전. GraGOD 코드에는 4a와 4b 사이를 자연스럽게 끊는 스위치가 없다(둘은 `post_process_scores` 안에서 연쇄, `prediction.py:80-82`). 정의 확정은 보류.

---

## [D] 정규화 통계 (median·IQR)

### 코드 위치

- **GraGOD**: `gragod/predictions/prediction.py`, `standarize_error_scores`, 85-105행.
  - median: `torch.median(scores, dim=0)` (97행), IQR: `quantile(0.75) - quantile(0.25)` (98-100행), 적용: `(scores - medians) / iqr` (103행). **분모에 epsilon 없음** — IQR=0인 채널이면 0-나눗셈.
- **d-ailin 원본**: `gdn/evaluate.py`, `get_err_scores`, 48-68행; 통계 계산은 `gdn/util/data.py`, `get_err_median_and_iqr`, 75-82행. 분모는 `|iqr| + 1e-2` (epsilon 있음, `evaluate.py:58-60`).

### 어느 구간의 데이터로 계산하는가 (코드 확인 결과)

- GraGOD: `process_dataset()`이 스플릿(train/val/test)마다 독립 호출되고 (`models/predict.py:390-406`), 각 호출 안에서 그 스플릿의 scores 텐서에 `standarize_error_scores`가 적용된다. 함수 자체가 입력 텐서의 dim=0으로 통계를 내므로 (`prediction.py:96-100`), test 스플릿 점수는 test 구간 통계로 정규화된다.
- d-ailin 원본도 같다. `get_err_scores(test_res, val_res)`는 val을 인자로 받지만 통계는 `test_predict, test_gt`에서 계산한다 (`gdn/evaluate.py:52`). 원저자 구현도 테스트셋 자체에서 median·IQR을 추정한다는 뜻이다. (val 인자는 이 함수에서 통계에 쓰이지 않는다. `get_full_err_scores`가 val 분포용으로 `get_err_scores(val_re_list, val_re_list)`를 따로 호출할 뿐이다 — 21행.)
- 논문 §3.6은 "median and inter-quartile range across time ticks of the Err_i(t) values"라고만 쓰고 구간을 밝히지 않는다.
- 결론: **두 레포 모두 우리 CLAUDE.md 규약(학습/검증 구간에서만 추정)과 다르다.**

### 학습·검증 구간 추정으로 바꾸는 두 방식의 가능성 판정 (선택은 보류)

- (i) GraGOD 내부 수정 — 가능하다. 필요한 변경 범위: `standarize_error_scores`에 사전 계산된 (medians, iqr) 인자 추가 (`prediction.py:85-105`), `post_process_scores` 시그니처 통과 (64-82행), `run_model`(`models/predict.py:26-70`)과 `process_dataset`(73-257행)에서 스플릿 간 통계 전달 로직 추가. 함수 3~4개의 시그니처가 바뀌는 포크 수정이다.
- (ii) 래퍼에서 우회 — 역시 가능하다. `post_process_scores: false`로 각 스플릿의 raw 채널별 점수를 받은 뒤(위 [C] 방법 a), 학습/검증 raw 점수에서 우리 코드로 median·IQR을 추정해 test 점수에 적용한다. GraGOD는 손대지 않는다. smoothing까지 재현하려면 smooth 함수도 우리 쪽에 필요하다(smoothed 구현 확정은 어차피 별도 승인 사항).

---

## [E] learn_graph 스위치

### 정의 위치

- 생성자 파라미터 `learn_graph: bool = True` — `models/gdn/model.py:56` (`GDN.__init__`), 저장 83행.
- yaml: `models/gdn/params_swat.yaml:8`, `models/gdn/params_ute.yaml:10` (둘 다 `true`). `params.yaml`·`params_telco.yaml`에는 키 자체가 없음 → 기본값 True 적용.
- 도입 커밋: `00096c9` 2025-02-16 "feat: optionally learn graph in gdn".

### 켜고 끌 때 달라지는 코드 경로 (`GDN.forward`, `models/gdn/model.py:136-159`)

- **True** (136-156행): embedding 가중치로 cos-sim 행렬 계산(139-143) → `torch.topk`(144) → gated_edge_index 구성(146-155) → `self.learned_edge_index`에 저장(156).
- **False** (157-159행): `gated_edge_index = self.edge_index_sets[i]` — 생성 시 주입된 a priori 그래프를 그대로 사용. `learned_edge_index`에도 그 고정 그래프가 저장됨(159행).
- 이후 경로(배치 확장 161-163, GNN 165-170, 출력 174-198)는 두 분기 공통.

### 끈 상태 = "그래프를 학습하지 않음"과 동등한가

- 구조적으로는 동등하다. False면 그래프가 embedding과 무관하게 학습 내내 고정이고, 그래프를 만드는 코드(139-156행)가 아예 실행되지 않는다. 고정 그래프의 출처는 `get_edge_index` (`datasets/graph.py:5-43`) — `edge_index_path`가 있으면 파일 로드(19-33행), 없으면 fully-connected 생성(42-43행 → `build_fully_connected_edge_index`, 46-66행).
- 단서 3가지 (코드 근거):
  1. embedding 자체는 계속 학습된다 — attention 계산(`models/gdn/modules.py`, `GraphLayer.message`, 233-285행)과 출력 곱(`model.py:177-190`)에 쓰이므로. "그래프 구조를 embedding에서 유도하지 않는다"는 의미의 스위치이지, embedding 학습을 멈추는 스위치가 아니다.
  2. `build_fully_connected_edge_index`는 `i == j` 자기쌍을 포함하지만 (`graph.py:59` — `i != j` 필터 없음), `GraphLayer.forward`가 self-loop을 제거 후 재추가하므로 (`modules.py:213-214`) 실효 차이 없음.
  3. d-ailin 원본에는 이 스위치가 없다 — 항상 topk 그래프를 학습한다 (`gdn/models/GDN.py:148-163`, 무조건 실행).

---

## [F] 인접행렬

### TopK 선택 코드 위치

- GraGOD: `models/gdn/model.py:144` — `topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]` (`GDN.forward` 내부).
- d-ailin: `gdn/models/GDN.py:157` — 동일 로직 (`GDN.forward`).

### 갱신 시점

- 매 forward 호출마다 재계산된다. TopK 블록(`model.py:136-156`)이 `forward` 본문 안에 있고 캐시가 없어, 학습 step·validation step·predict step 모두에서 실행된다 (d-ailin도 같다, `GDN.py:150-163`). epoch 단위도, 학습 후 고정도 아니다.
- embedding이 optimizer step마다 갱신되므로 그래프는 사실상 매 학습 step 후 다음 forward에서 바뀔 수 있다. 다만 그래프는 embedding에만 의존하고 입력 x에는 의존하지 않아(139-144행의 입력은 `self.embedding` 뿐), 같은 가중치 상태라면 어떤 배치든 같은 그래프가 나온다.

### 최종 인접행렬 추출 지점 후보

1. `model.learned_edge_index` 속성 (`models/gdn/model.py:81`에서 초기화, 156/159행에서 대입; shape `[2, n_features × topk]`).
   의미: 마지막으로 forward가 돈 시점의 그래프. 학습 직후 읽으면 마지막 학습(또는 마지막 validation) 배치 시점이고, predict 후 읽으면 마지막 predict 배치 시점이다. "best 모델의 그래프"가 아니라 "마지막 가중치의 그래프"라는 점에 주의 — predict를 best.ckpt로 로드해 돌린 직후라면 둘이 일치한다.
2. best.ckpt의 embedding에서 재계산. `ModelCheckpoint`가 `Loss/val` 최소 시점을 `best.ckpt`로 저장한다 (`gragod/training/callbacks.py:44-50`; 로드는 `models/predict.py:353-366`). 체크포인트의 `embedding.weight`로 `model.py:137-144`와 같은 cos-sim→topk 계산을 하면 best validation 시점의 그래프를 forward 없이 결정적으로 얻는다 — 그래프가 embedding에만 의존하기 때문이다. GraGOD에 이 재계산 함수는 없어 우리 코드로 재현해야 하고, 로직 출처 주석이 필요하다.
3. **(참고) d-ailin 스타일 `learned_graph`**: 원본은 topk 인덱스 행렬 `[n, k]` 자체를 `self.learned_graph`에 저장한다 (`gdn/models/GDN.py:159`). GraGOD은 엣지 리스트 형태만 저장하므로, 인덱스 행렬이 필요하면 `learned_edge_index`를 reshape하거나 후보 2의 재계산에서 `topk_indices_ji`를 직접 취해야 한다.

### TopK의 자기 자신(대각선) 포함 여부 — 제외하지 않는다

- cos-sim 행렬의 대각선을 topk 전에 제거하는 코드는 양쪽 레포 어디에도 없다(GraGOD `model.py:139-144`; d-ailin `GDN.py:150-157`). 비퇴화 embedding에서는 자기 유사도 1이 보통 선택돼 다른 이웃은 k−1개가 된다. 다만 다른 embedding과 cosine이 정확히 같은 동률에서는 `torch.topk`가 self를 반드시 고른다고 보장하지 않으므로, 추출 결과에서 실제 self-edge를 세고 제거해야 한다.
- 이후 `GraphLayer.forward`가 self-loop을 일괄 제거하고 다시 추가한다 (GraGOD `modules.py:213-214`; d-ailin `gdn/models/graph_layer.py:61-62`). 즉 topk가 소모한 자기-엣지는 제거되고, 모든 노드에 균일한 self-loop이 재부여된다.

---

## [G] 하이퍼파라미터 대조표

출처: GraGOD `models/gdn/params.yaml`(이하 G-기본)·`params_swat.yaml`(G-swat)·`params_telco.yaml`(G-telco)·`params_ute.yaml`(G-ute); d-ailin `main.py:201-217`(argparse 기본값)·`run.sh:4-19`(권장 실행값)·`train.py`(하드코딩); 논문 §4.4. **굵게** = 세 출처(GraGOD/d-ailin/논문)가 서로 다른 항목.

| 항목 | G-기본 | G-swat | G-telco | G-ute | d-ailin 기본 (main.py) | d-ailin run.sh | 논문 §4.4 |
|---|---|---|---|---|---|---|---|
| **embed_dim** | 64 (:26) | 64 (:3) | 80 (:3) | 96 (:3) | 64 (:204) | 64 | WADI 128 / SWaT 64 |
| **topk k** | 2 (:29) | 15 (:6) | 3 (:6) | 10 (:6) | 20 (:215) | 5 | WADI 30 / SWaT 15 |
| **window_size** | 50 (:25) | 5 (:2) | 155 (:2) | 55 (:2) | slide_win 15 (:203) | 5 | 5 |
| **out_layer_inter_dim** | 128 (:28) | 64 (:5) | 448 (:5) | 896 (:5) | 256 (:212) | 128 | hidden 128/64 |
| out_layer_num | 1 (:27) | 1 (:4) | 7 (:4) | 1 (:4) | 1 (:211) | 1 | (미기재) |
| heads | 1 (:30) | 1 (:7) | 4 (:7) | 2 (:7) | (1 고정, GDN.py:101) | — | (미기재) |
| **dropout** | 0.0 (:31) | 0.2 (:11) | 0.1 (:8) | 0.1 (:8) | 0.2 하드코딩 (GDN.py:114) | — | (미기재) |
| 학습률 | 0.001 (:7) | 0.001 (:18) | 0.001 (:16) | 0.001 (:19) | 0.001 하드코딩 (train.py:31) | — | 1e-3 |
| **betas** | [0.9, 0.99] (:16) | [0.9, 0.999] (:27) | [0.9, 0.999] (:25) | [0.9, 0.99] (:28) | 미지정→torch 기본 (0.9, 0.999) (train.py:31) | — | (0.9, 0.99) |
| **batch_size** | 32 (:4) | 512 (:17) | 512 (:15) | 32 (:16) | 128 (:201) | 32 | (미기재) |
| **n_epochs** | 2 (:5) | 200 (:16) | 200 (:14) | 200 (:17) | 100 (:202) | 30 | 50 |
| **early stop patience** | 20 (:19) | 10 (:35) | 10 (:31) | 10 (:31) | early_stop_win 15 하드코딩 (train.py:49) | — | 10 |
| early stop delta | 0.0001 (:20) | 0.001 (:36) | 0.001 (:32) | 0.0001 (:32) | (없음 — loss 개선만 비교, train.py:93-103) | — | (미기재) |
| weight_decay | 0 (:14) | 1e-5 (:25) | 1e-5 (:23) | 0 (:26) | decay 0 (:213) | 0 | (미기재) |
| val 비율 | val_size 0.1 (:6) | (별도 val 파일, swat.py:64-66) | (별도 분할) | 0.1 (:18) | val_ratio 0.1 (:214) | 0.2 | (미기재) |
| seed | 42 하드코딩 (models/train.py:22) | 동일 | 동일 | 동일 | 0 (:209) | 5 | (미기재) |
| **smoothing** | window_size_smooth 5 (:45) | 5 (:41) | 5 (:37) | 5 (:39, 단 post_process false :40) | 후행 4-창(before_num=3), 처음 3개는 0 (evaluate.py:62-65) | — | "SMA" (크기 미기재) |
| **채널 집계** | (system_output_mode 키 없음) | mean (:45) | (키 없음) | (키 없음) | topk=1 sum ≒ max (evaluate.py:102-107, main.py:160-161 topk=1) | — | max |
| **오차 정규화 epsilon** | 없음 (prediction.py:103) | — | — | — | 1e-2 (evaluate.py:58-60) | — | (미기재) |
| learn_graph | (키 없음→True) | true (:8) | (키 없음→True) | true (:10) | (스위치 없음 — 항상 학습) | — | (항상 학습) |
| LR 스케줄러 | ReduceLROnPlateau f=0.5 p=8 (trainer.py:127-129) | 동일 | 동일 | 동일 | 없음 (train.py:31) | — | (미기재) |
| gradient clip | 1.0 (trainer.py:218) | 동일 | 동일 | 동일 | 없음 | — | (미기재) |
| 손실 | MSE (models/train.py:162-169) | 동일 | 동일 | 동일 | MSE (train.py:20-23) | — | MSE |

주: GraGOD의 `predictor_params.score_type`(`params_swat.yaml:46` "mse" 등)은 GDN 경로에서 **무시**된다 — `GDN_PLModule.calculate_anomaly_score`가 절대 오차를 하드코딩 (`models/gdn/model.py:313-316`, [C] 참조).

---

## [H] 실행 환경

### GraGOD (develop @ ec8cd45)

- 파이썬: `>=3.10,<=3.11` (`pyproject.toml:16`); 개발 고정 버전 3.10.13 (`.python-version:1`).
- 패키지 관리: poetry 1.8 (2.0 미만) 요구 (`README.md:39`). 주요 poetry 의존성: numpy ^1.26.3, pandas >=1.3.0, scikit-learn ^1.4.2, optuna ^4.1.0, timeeval ^1.4.2, prts ^1.0.0.3 등 (`pyproject.toml:15-36`).
- poetry 밖 수동 설치 (`requirements.txt:1-3`): `torch==2.2.2`, pytorch-lightning은 저자 개인 포크 `git+https://github.com/gonzachiar/pytorch-lightning.git@feature/best-k-metrics`, `tensorboardX==2.6.2.2`. poetry.lock에도 torch 2.2.2 (`poetry.lock:4338-4339`).
- ⚠ torch-geometric은 어디에도 선언돼 있지 않다. `models/gdn/modules.py:5-7`이 `torch_geometric`를 임포트하는데 `pyproject.toml`·`requirements.txt`·`poetry.lock` 전부 0건이다(grep 확인). 필요 버전은 확인 불가 — 레포에 선언 자체가 없어서다. 별도 수동 설치가 필요하다는 사실만 확정된다.
- GPU: 필수 아님 — `set_device`가 cuda → mps → cpu 순 자동 선택 (`gragod/utils.py:83-92`). 테스트는 cpu로 돌게 되어 있음 (`tests/test_training.py:47`). GDN 학습·예측 자체의 CUDA 강제 없음.

### d-ailin/GDN (main @ 9853899)

- Python >= 3.6, cuda == 10.2, PyTorch == 1.5.1, torch-geometric == 1.5.0 (`README.md:8-11`); torch-scatter/sparse/cluster/spline-conv를 torch-1.5.0+cu102 휠에서 설치 (`install.sh:1-5`).
- cpu 실행도 공식 지원 (`run.sh:22-40`, `-device 'cpu'`).
- GraGOD은 torch 2.2.2 기반이므로 d-ailin의 환경 스펙은 참고용일 뿐 호환되지 않음.

---

## 내가 결정해야 할 사항 (판단 보류 목록)

1. **[A] GraGOD 고정 커밋 해시** — 후보 `ec8cd45` / `11209bc` / `e0c8846` 중 선택 (CLAUDE.md의 TBD 채움).
2. **[B] 주입 지점** — 후보 (i) 포크에 데이터셋 로더 추가 vs (ii) `get_data_loader` 직접 주입 러너.
3. **[B] reshape 의혹** — `models/gdn/model.py:271, 304`의 `reshape(-1, N, W)`가 전치 의도의 버그인지(원본은 `(N, W)` 원생성) 실행 검증 후, 원본 재현이냐 수정이냐 결정.
4. **[C] "raw" 점수의 정의** — 정규화 전 절대 오차(3단계)인지, 정규화 후·smoothing 전(4a 후)인지.
5. **[C]·[G] smoothed 구현 승인** — d-ailin 후행 4-창(경계: 처음 3개 0) vs GraGOD 5-창 replicate 패딩 vs 논문 "SMA"(크기 미기재). 아울러 GraGOD `smooth_scores`의 pooling 축 의혹(`prediction.py:119-124`, feature 축에 걸리는 것으로 보임) 실행 검증 필요. 승인 전 smoothed 파일 미생산(규약).
6. **[D] 정규화 통계 변경 방식** — (i) GraGOD 내부 수정 vs (ii) 래퍼 우회. 둘 다 가능함은 확인 완료.
7. **[D]·[G] IQR epsilon** — d-ailin 1e-2 vs GraGOD 없음(0-나눗셈 위험). 값은 configs/로.
8. **[F] 인접행렬 추출 시점** — 후보 1(마지막 forward) vs 후보 2(best.ckpt 재계산). exp03의 adjacency/ 산출물 정의.
9. **[G] 채널 집계 모드** — 우리 규약 max vs GraGOD swat 설정 mean vs d-ailin topk=1(≒max) vs 논문 max. 값은 configs/로, 차이는 DECISIONS.md에.
10. **[H] torch-geometric 버전** — GraGOD에 미선언. torch 2.2.2 호환 버전 선정 필요.
11. **시드 주입 방식** — GraGOD은 42 하드코딩(`models/train.py:22`, `models/predict.py:23`) + `set_seeds`(`gragod/training/main.py:18-28`). exp03의 시드 10개 실험과 충돌 — `train()`을 직접 호출하면 우회 가능하나 방식 확정 필요.
