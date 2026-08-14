# exp00 — D-03 패치 적용 + D-10 버전 확정 + D-12 재검증 결과

**종합: 재검증 3항목 전부 통과 — 3b 개방 조건 충족 (D-12).**

- 수행일: 2026-08-14
- 검증 스크립트: `experiments/exp00_gragod_recon/reverify_patched.py`
- 패치 포크: `../gragod-fork/` (D-14 위치) — 베이스 `ec8cd452a410ba903a31beb097a010ba0448c095` + 패치 커밋 `485e26b0c6b1d63f4f3531c8d05597db82e9db29` (브랜치 `fix/gdn-input-transform`)
- 패치 커밋 메시지: `fix: input transform reshape->permute+contiguous (D-03, SHAPEFLOW)`
- diff 사본: `patches/gdn_input_transform.diff` (아래 전문과 동일; 두 줄 교체 외 변경 없음 — `git diff --stat` = 1 file changed, 2 insertions(+), 2 deletions(-))

---

## [1] 고정 환경 (D-13) — 확정 버전과 선정 근거

| 패키지 | 확정 버전 | 근거 |
|---|---|---|
| python | 3.10.20 (conda env `tsad_fixed`) | GraGOD 허용 범위 >=3.10,<=3.11 (pyproject.toml:16), .python-version 3.10.x 계열 |
| torch | 2.2.2+cpu | GraGOD requirements.txt:1 고정값 |
| numpy | 1.26.4 (`numpy<2` 핀) | torch 2.2.2 휠은 numpy 1.x C-API 기준 빌드 — numpy 2.x와 비호환이라 설치 시 `<2` 핀 |
| torch-geometric | 2.5.3 (D-10 확정) | 근거 셋. 배포 메타데이터에 torch 핀이 없다(순수 파이썬; `pip show` Requires: aiohttp, fsspec, jinja2, numpy, psutil, pyparsing, requests, scikit-learn, scipy, tqdm — torch 없음). PyG 공식 휠 아카이브(data.pyg.org/whl)에 `torch-2.2.2+cpu` 인덱스가 있어 torch 2.2.2가 공식 지원 대상임을 확인했고, 2.5.3은 torch 2.2 시기(2024-04)의 릴리스다. 설치 후 임포트 + `add_self_loops` 기본 연산도 돌았다: `torch_geometric 2.5.3 | torch 2.2.2+cpu | self_loops ok: True` |
| pytorch-lightning | 저자 포크 설치 성공 — `git+https://github.com/gonzachiar/pytorch-lightning.git@feature/best-k-metrics` (커밋 `834dbf3039ee82a2ac5e65eed25f9989222283c6`, 버전 표기 2.6.2) | GraGOD requirements.txt:2 고정 소스 그대로다. 대체가 없으므로 DECISIONS 질문 목록 기입도 필요 없다. 설치 주의: 포크는 Lightning 모노레포라 기본 빌드명이 `lightning`이 되어 직접 설치가 실패한다 — `PACKAGE_NAME=pytorch` 환경변수를 주고 `pip install git+...`로 빌드해야 `pytorch_lightning` 패키지가 깔린다. `pip freeze` 확인: `pytorch-lightning @ git+https://github.com/gonzachiar/pytorch-lightning.git@834dbf3039ee82a2ac5e65eed25f9989222283c6` |
| tensorboardX | 2.6.2.2 | GraGOD requirements.txt:3 고정값 |
| pandas / scikit-learn / pyyaml / networkx / colorama | 설치(버전 자유) | GraGOD 임포트 체인 요구(datasets·gragod.utils·training) — 값 계산에 관여하지 않음 |

전 패키지 임포트 확인: `all imports ok | numpy 1.26.4`.

이 고정 환경에서는 SHAPEFLOW 때 필요했던 우회(datasets 스텁·파일 경로 로드 — Python 3.13 dataclass 충돌)가 필요 없어져, GraGOD 패키지를 정식 경로로 임포트했다(스크립트의 "임포트 확인" 줄).

---

## [2] 패치 diff 전문 (patches/gdn_input_transform.diff)

```diff
diff --git a/models/gdn/model.py b/models/gdn/model.py
index 7ba3eeb..0c9cee1 100644
--- a/models/gdn/model.py
+++ b/models/gdn/model.py
@@ -268,7 +268,7 @@ class GDN_PLModule(PLBaseModule):
         x, y, edge_index = [
             item.float().to(self.device)
             for item in [
-                x.reshape(-1, x.size(2), x.size(1)),
+                x.permute(0, 2, 1).contiguous(),
                 y.squeeze(1),
                 edge_index,
             ]
@@ -301,7 +301,7 @@ class GDN_PLModule(PLBaseModule):
             predictions: Predictions for the input batch
         """
         x = batch[0] if isinstance(batch, (list, tuple)) else batch
-        predictions = self(x.reshape(-1, x.size(2), x.size(1)))
+        predictions = self(x.permute(0, 2, 1).contiguous())
         return predictions
 
     def post_process_predictions(self, predictions):
```

117행(`x.view(-1, all_feature)`)은 건드리지 않았다 (D-03).

---

## [3] D-12 재검증 — 항목별 판정

- **(1) e2 상당: 통과** — 패치된 predict_step 경로로 forward가 에러 없이 돌고, 출력 shape가 (2, 3) = (batch, n_features)다.
- **(2) f 상당: 통과** — 채널 2 오프셋 500이 실제 X_true 경로를 거쳐 점수의 채널 2 자리에만 정확히 나타났다.
- **(3) g 상당: 통과** — y는 squeeze(1)만 겪고, loss 피연산자가 원시계열 시점 W+i 행과 원소 단위로 일치한다. 패치된 shared_step 반환 loss = mean(y²) = 3261.6667로 내부 피연산자 동일성까지 확인했다.

수치는 SHAPEFLOW 2부의 실측 출력과 전 항목 같다(점수 500/채널 2, X_true 첫 행 [0, 0, 500], y 행 40~72, loss 3261.6667). 달라진 곳이 없으니 나란히 붙일 차이 항목도 없다. 유일한 설계 차이는 검증 방식 자체인데, SHAPEFLOW는 스크립트가 교체 대상 변환을 대신 수행했고 이번에는 패치된 shared_step/predict_step 메서드를 직접 통과시켰다.

### 실행 출력 (그대로)

```text
[환경] 고정 환경 (D-13).
  python             : 3.10.20
  torch              : 2.2.2+cpu
  torch_geometric    : 2.5.3
  pytorch_lightning  : 2.6.2 (저자 포크 git+gonzachiar@feature/best-k-metrics)
포크 경로: c:/Users/simon/time_series/gragod-fork
임포트 확인: GDN_PLModule <- c:\Users/simon/time_series/gragod-fork\models\gdn\model.py

========================================================================
[재검증 1 — e2 상당] 패치된 predict_step 경로로 forward 실행
========================================================================
입력 (batch, window, feature) = (2, 4, 3)
패치된 predict_step(models/gdn/model.py:292-305, 304행 = permute+contiguous) 호출:
forward 성공, 출력 shape: (2, 3)
출력 shape == (batch, n_features) == (2, 3) 인가: True

========================================================================
[재검증 2 — f 상당] 채널 오프셋이 점수의 채널 2 자리에만 나타나는가
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
[재검증 3 — g 상당] 패치된 shared_step 경로에서 y 정렬 유지
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

패치된 shared_step(models/gdn/model.py:265-278, 271행 = permute+contiguous) 반환 loss: 3261.6667
기대 loss = mean(기대 y²) = 3261.6667
shared_step loss == 기대 loss (y 피연산자가 내부에서도 동일함의 증거): True

========================================================================
[종합] (1) forward/shape: 통과 / (2) 채널 정렬: 통과 / (3) y 정렬: 통과
========================================================================
```
