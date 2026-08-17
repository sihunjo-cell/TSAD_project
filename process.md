# 1~3단계 수행 과정과 판단 근거

- 작성일: 2026-08-17
- 범위: 1단계, 2단계, 3a, 3b
- 제외: 4단계 GHL 실데이터 스모크와 전체 모델 학습
- 결정 원본: `DECISIONS.md`

이 문서는 4단계 전에 무엇을 했고, 어떤 데이터·코드·실행 결과를 보고 결정을 내렸는지 한 흐름으로 정리한다. 수치와 결정은 새로 계산하지 않았다. Manifest, preflight 로그, 고정 외부 저장소, 단위 테스트, D-01~D-46에 이미 남은 근거만 연결했다.

작업 규약의 단계명은 논문의 주차별 실험 단계와 다르다. 여기서 1단계는 구현 의혹의 실행 검증, 2단계는 GDN shape 흐름과 패치 완전성 검증, 3a는 데이터 정찰·Manifest·EDA, 3b는 GDN 러너·배치·실행 봉인이다. 0단계 정찰은 1단계의 입력이므로 필요한 부분만 앞에 적는다.

## 전체 흐름

| 단계 | 핵심 질문 | 사용한 근거 | 끝난 결과 |
| --- | --- | --- | --- |
| 1 | GraGOD의 입력 변환과 smoothing을 그대로 믿어도 되는가 | 실제 코드, 표식이 든 장난감 텐서, d-ailin 원본 대조 | 두 의혹이 모두 참임을 확인하고 수정 방향을 좁힘 |
| 2 | 두 줄 패치가 shape·채널·라벨 정렬을 끝까지 보존하는가 | forward 전 경로 정적 추적, 최소 실행 e1·e2·f·g, 고정 환경 재검증 | `permute(...).contiguous()` 두 줄과 고정 포크를 확정하고 3b 개방 |
| 3a | 강혁님 Manifest 전에도 실제 데이터로 경계·비율·전처리를 잠글 수 있는가 | GHL 25개, HAI 23.05 실물, 공식 목록·README·기술문서, 학습 구간 EDA | 데이터 계약, GHL 5개 비율, HAI 10·100%, 전처리와 `topk=5·22` 근거 확보 |
| 3b | 실험 당일에는 설정만 읽어 안전하게 실행할 수 있는가 | 원 논문·두 구현, loader·runner·batch 테스트, 합성 실행 기록, 7회 감사 | 러너·배치·완전성 검사·재개·snapshot·대조 팔 구현 완료. 새 실행본 봉인만 대기 |

## 0단계에서 1단계로 넘긴 문제

`experiments/exp00_gragod_recon/RECON.md`에서 GraGOD develop `ec8cd452`와 d-ailin/GDN main `9853899d`를 대조했다. 이때는 코드를 고치지 않고 아래 문제를 후보로 남겼다.

- GraGOD `models/gdn/model.py:271,304`는 `(batch, window, feature)`를 `(batch, feature, window)`로 바꿀 때 `reshape`를 쓴다. 원본 d-ailin은 애초에 `(node, window)`를 만들기 때문에 같은 변환 지점이 없다.
- GraGOD의 `smooth_scores`가 시간축이 아니라 feature 축에 작용할 가능성이 있었다.
- GraGOD와 d-ailin 모두 점수 정규화 통계를 테스트 오차에서 추정한다. 이 프로젝트의 train·validation 추정 규칙과 충돌했다.
- raw 점수, smoothing, 채널 집계, `learn_graph=false`, 인접행렬 추출, 시드 주입, 미선언 `torch-geometric` 버전을 별도로 결정해야 했다.

정찰만으로 결론을 내리지 않았다. 값의 배열을 사람이 추적할 수 있는 입력을 만들어 1단계에서 의혹을 검증했다.

## 1단계 — 구현 의혹 실행 검증

### 1-1. `reshape`와 전치가 같은지 확인

검증 스크립트는 `experiments/exp00_gragod_recon/verify_suspicions.py`, 결과 원문은 `VERIFICATION.md:14-78`에 있다. 입력값을 `배치×1000 + 시점×10 + 채널`로 만들었다. 어느 원소가 어느 시점·채널에서 왔는지 숫자만 보고 추적하려는 설계다.

비정방 입력 `(B=2,W=4,N=3)`에서 두 연산의 출력 shape는 모두 `(2,3,4)`였지만 값은 달랐다. 첫 배치의 채널 0 자리는 다음처럼 갈렸다.

| 변환 | 채널 0의 윈도로 읽히는 값 |
| --- | --- |
| GraGOD `reshape` | `[0, 1, 2, 10]` |
| 올바른 `permute` | `[0, 10, 20, 30]` |

`reshape` 결과에는 채널 표지와 시점 표지가 섞였다. `W=N=4`인 정방 입력에서도 shape만 같고 원소 정렬은 달랐다. 따라서 오류가 특정 채널 수나 window에서만 생기는 문제가 아니라고 판정했다.

결정은 D-03이다. `shared_step`과 `predict_step`의 두 입력 변환을 전치로 바꾸되, downstream shape가 맞는지는 2단계에서 따로 확인하기로 했다.

### 1-2. smoothing 축 확인

두 번째 입력은 10시점×3채널이었다. 채널 0은 0, 채널 1은 100, 채널 2는 200으로 고정하고 시점 5의 채널 2만 1000으로 올렸다. 시간축 smoothing이라면 상수 채널 1은 100을 유지하고 시점 5의 뾰족값은 이웃 시점으로 퍼져야 한다.

실제 GraGOD 출력에서 채널 1은 20으로 바뀌었고 기본 행은 `[0,20,60]`이었다. 반면 시점 5의 뾰족값은 시점 6~9로 퍼지지 않았다. `F.pad`와 `avg_pool1d`가 마지막 축인 feature 축에 적용된다는 직접 증거다.

d-ailin `evaluate.py:62-65`의 후행 4-창을 같은 입력에 재현했다. 처음 3개 시점은 0, 상수 채널은 이후 100을 유지했고 뾰족값은 시점 5~8에 반영됐다. 비교 결과에 따라 다음을 고정했다.

- GraGOD `smooth_scores`는 사용하지 않는다(D-05).
- 자체 후행 4-창을 쓴다. 처음 3점은 0이며 채널별 시간축에 적용한다(D-04).
- 순서는 정규화 → smoothing → 채널 max 집계다.
- raw와 smoothed를 모두 저장해 후처리 선택 때문에 재학습하지 않게 한다.

### 1-3. 점수와 정규화 경계 결정

정찰 코드와 1단계 결과를 합쳐 다음 경계를 정했다.

- 모델 오차는 절대값, 채널 집계는 `max`다(D-09).
- 본 결과인 `trainnorm`은 train·validation 예측 오차에서 채널별 median·IQR을 추정한다(D-06).
- 저자 방식인 테스트 오차 추정은 `testnorm`으로 따로 저장하고 부록에서만 비교한다(D-15).
- IQR 분모에는 d-ailin 값인 `epsilon=0.01`을 쓴다(D-07).
- 집계 전 채널별 점수는 `__channels` 보조 산출물로 남긴다. 회수 분석과 재집계에는 쓰되 1급 채점 배열과 섞지 않는다(D-16).

1단계에서 오류의 존재는 확인했지만 패치를 바로 적용하지 않았다. `permute`가 만든 비연속 텐서를 이후 코드가 받아들이는지 확인해야 했기 때문이다.

## 2단계 — shape 흐름과 패치 완전성 검증

### 2-1. 입력부터 loss·점수까지 축 추적

`experiments/exp00_gragod_recon/SHAPEFLOW.md`에서 GraGOD `models/gdn/model.py:100-200,265-316`, `models/predict.py:121-152`와 d-ailin `models/GDN.py:122-187`을 줄 단위로 추적했다.

확인한 shape는 다음과 같다.

```text
SlidingWindowDataset x       : (B, W, N)
전치 뒤 GDN 입력             : (B, N, W)
forward 내부 GNN 입력        : (B*N, W)
모델 출력과 y                : (B, N)
시계열 전체 forecast 오차    : (L-W, N)
```

forward 본문은 `(B,N,W)`가 값까지 올바르게 들어온다는 전제에서 노드·채널 정렬을 유지했다. 원본 d-ailin도 같은 shape를 받는다. 문제는 GraGOD가 dataset의 `(B,W,N)`을 바꾸는 두 줄에 한정됐다.

### 2-2. 단순 `permute`만으로 부족한 이유 확인

최소 실행 e1에서 `permute(0,2,1)`만 적용하자 `model.py:117`의 `view(-1, all_feature)`가 비연속 텐서를 거부했다. 패치 후보가 의미상 맞더라도 실제 forward는 시작하지 못했다.

e2에서 `permute(0,2,1).contiguous()`를 사용하자 forward가 정상 실행되고 출력 shape `(B,N)`이 나왔다. 이어서 두 가지를 더 확인했다.

- 채널 2에만 500 오프셋을 준 입력에서 절대 오차 500이 채널 2에만 나타났다.
- `y.squeeze(1)` 뒤의 target이 원시계열 `W+i` 행과 원소 단위로 같았고, loss도 기대값 `3261.6667`과 일치했다.

이에 따라 D-03 패치는 271·304행의 두 줄만 `permute(...).contiguous()`로 바꾸는 것으로 확정했다. 117행의 `view`는 원본 d-ailin과 같은 코드이고 연속 입력에서는 맞으므로 수정하지 않았다. 패치 사본은 `patches/gdn_input_transform.diff`에 있다.

### 2-3. 고정 환경 재검증과 포크 봉인

1·2단계의 첫 검증은 torch 2.10 스크래치 환경이어서 shape·정렬 판정에만 사용했다(D-13). 실제 실행 환경은 다음처럼 별도로 고정했다.

| 항목 | 고정값 |
| --- | --- |
| Python | 3.10.20 |
| torch | 2.2.2 |
| NumPy | 1.26.4 |
| torch-geometric | 2.5.3 |
| pytorch-lightning | 저자 포크 `834dbf3039ee82a2ac5e65eed25f9989222283c6` |
| GraGOD | 베이스 `ec8cd452` + 패치 포크 `485e26b0c6b1d63f4f3531c8d05597db82e9db29` |

`experiments/exp00_gragod_recon/PATCH_REVERIFY.md`에서 forward shape, 채널 오프셋, y·loss 정렬 세 항목을 고정 환경으로 다시 실행해 전부 통과시켰다(D-12). GraGOD 수정은 이 두 줄뿐이며 포크는 `../gragod-fork/`에 둔다(D-14).

오케스트레이션도 `ORCHESTRATION.md`에서 확인했다. GraGOD의 데이터 로드·post-process·시드 42 경로를 그대로 부르지 않고, 분할이 끝난 배열을 `get_data_loader`에 주입하는 자체 러너를 쓰기로 했다(D-02·D-11). 이 판정으로 3b 구현을 시작할 조건이 열렸다.

## 3a — 데이터 정찰·Manifest·비율 EDA

### 3a-1. GHL 실물과 학습 경계 확정

공식 목록 `TheDatumOrg/TSB-AD/Datasets/File_List/TSB-AD-M.csv:33-57`과 GHL CSV 25개를 대조했다. 결과는 `docs/manifest_draft.md:5-41`과 `experiments/exp01b_ghl_preflight/logs/inventory.csv`에 있다.

- 시계열은 25개이며 센서 19열과 `Label` 1열이다.
- 파일명의 `tr_`가 테스트 시작 0-based 인덱스다. `1st_`는 전체 파일의 첫 이상 인덱스이지 학습 경계가 아니다.
- 학습 길이는 39,938, 43,750, 50,000 세 값이다. 계획 범위 39,938~50,000 밖의 시계열은 0개다.
- 학습·테스트 결측값은 0개, 학습 라벨 이상은 0개다.
- `1st_`와 실제 첫 `Label=1`의 불일치는 0개다.

이 확인으로 강혁님 표를 기다리지 않고도 GHL loader의 파일 경계와 feature 수를 고정할 수 있었다. 파일명·행 수·크기·SHA-256은 `configs/input_manifest.yaml`에 봉인했다.

### 3a-2. HAI 버전과 시계열 단위 확정

HAI는 Git LFS 용량 초과 때문에 공식 저장소에서 객체를 직접 받지 못했다. 공식 README가 안내한 Kaggle 미러에서 HAI 23.05를 받고, 저장소의 LFS 포인터에 적힌 SHA-256과 바이트 크기를 10개 모두 대조했다. 실행에 쓰는 8개 CSV의 첫 줄도 열어 포인터 문구가 아니라 실제 `timestamp,...` 헤더임을 확인했다.

Manifest의 실측 결과는 다음과 같다.

| 구분 | 파일 수 | 행 수 합계 | 모델 입력 채널 |
| --- | ---: | ---: | ---: |
| train | 4 | 896,400 | 86 |
| test | 2 | 284,400 | 86 |
| test label | 2 | test와 각각 동일 | label 1 |

버전은 HAI 23.05로 정했다. “시계열 하나”는 연속성을 보장하는 CSV 한 파일이다. train 4세션은 한 모델의 훈련 corpus로 쓰되 raw window는 파일 경계를 넘지 않는다. test 2세션도 각 label 파일과 짝지어 따로 채점한다(D-17).

timestamp와 label을 빼면 `N=86`이다. 사전 규칙 `k=max(5,ceil(0.25N))`을 적용해 HAI `topk=22`를 확정했다. GHL은 `N=19`라 하한 규칙에 따라 `topk=5`다.

### 3a-3. GHL 비율 5·10·20·50·100%의 역할 결정

비율은 처음부터 최적값이라고 가정하지 않았다. 테스트 구간과 모델 점수를 보지 않고 앞쪽 학습 구간만 분석했다. `experiments/exp01b_ghl_preflight/ANALYSIS.md:28-46`에서 다음 지표를 함께 사용했다.

- 보존 길이와 train·validation window 수
- 활성 채널 수: `unique_value_count>=2`
- IQR이 변하는 채널 수
- 축소 구간의 범위가 전체 학습 Q1~Q3을 포함하는 비율
- 전체 학습 대비 표준화 평균 차이
- 채널 쌍 Pearson 상관 차이로 만든 관계 유사도 정찰치

`std>0`은 긴 정수 상수열에서 약 `1.11e-16`의 오차가 나와 활성 판정에서 제외했다. Pearson 지표도 GDN 성능의 대용치로 쓰지 않고 관계 구조가 어느 구간에서 살아나는지 보는 정찰치로만 사용했다.

| 비율 | 보존 길이 중앙값 | 활성 채널 최소/중앙값 | 관계 유사도 중앙값 | 사전 역할 |
| ---: | ---: | ---: | ---: | --- |
| 5% | 2,500 | 9/13 | 0.351 | 스트레스 하한 |
| 10% | 5,000 | 13/13 | 0.383 | 저데이터 기준점 |
| 20% | 10,000 | 13/19 | 0.814 | 구조 회복 지점 |
| 50% | 25,000 | 19/19 | 0.928 | 고데이터 대조 |
| 100% | 50,000 | 19/19 | 1.000 | 전체 기준 |

5%와 10%는 대표성이 충분해서 남긴 것이 아니다. 채널 활동과 관계 구조가 빠지는 저데이터 영역을 의도적으로 측정하려고 남겼다. 20%에서 관계 유사도가 0.814로 뛰고 활성 채널 중앙값이 처음 19/19가 돼 구조 회복 지점으로 삼았다. 50%와 100%는 고데이터 대조와 전체 기준이다.

25시계열×5비율×W 후보 5개, 총 625조건은 모두 실행 가능했다. 가장 짧은 series 12도 5%·W=155에서 train window 1,642개, validation window 45개, 정규화 표본 1,842개가 남았다. 5%에서 완전 고정된 채널 행은 154/475, IQR=0은 279/475였지만 학습 전체에서 완전 고정된 행은 0/475였다. 저비율에서 멈춘 채널을 임의로 삭제하지 않고 비율마다 scaler와 점수 정규화 통계를 다시 추정하기로 했다.

### 3a-4. HAI 10% 조건과 전처리 결정

HAI EDA는 train 4세션만 썼다. 테스트 파일, 테스트 라벨, 모델 점수는 쓰지 않았다. 각 세션의 timestamp는 1초 간격이고 중복·역행·누락 간격은 0건이었다. 센서 결측값과 비유한값도 0개였다.

각 세션 앞 10%는 28,080, 29,160, 12,600, 19,800행이다. 네 조각을 합치면 전체 훈련에서 움직이는 66채널 중 59채널을 관측했다. 세션별 관계 유사도는 0.545~0.701이었다. 따라서 HAI 10%도 충분 데이터가 아니라 저데이터 대표 조건으로 해석하고, 100%를 전체 기준으로 뒀다(D-21).

4세션×2비율×W 후보 5개, 총 40조건에서 학습 불가능 조건은 0개였다. 이 결과와 공식 README·기술문서를 근거로 전처리를 다음처럼 고정했다(D-20).

- 다운샘플 배율 1: 원본이 1초 간격이고 몇 초짜리 단기 동작이 있어 시간 해상도를 임의로 낮추지 않는다.
- 초기 절단 0포인트: 파일 시작이 설비 기동 시점이라는 직접 근거가 없다.
- timestamp와 label은 입력에서 제외한다.
- 비율과 validation은 세션별로 자른다. validation은 축소 구간의 뒤 10%다.
- scaler는 validation 분할 뒤 train에만 fit한다. GHL은 시계열별 scaler, HAI는 네 train 부분을 합친 scaler 하나를 쓴다.
- validation·test에는 같은 scaler의 transform만 적용하며 HAI 파일 사이에는 window를 만들지 않는다.

강혁님 산출물이 나중에 데이터 버전·경계·채널 수를 다르게 제시하거나 훈련 데이터에서 초기 안정화 구간 제거 근거를 제시하면 preflight를 다시 실행한다. 결과를 본 뒤 비율을 바꾸는 것은 허용하지 않는다.

## 3b — 하이퍼파라미터·러너·배치·봉인

### 3b-0A. 분할 수식과 실행 가능성 고정

앞자르기는 길이 `T`의 앞 `ceil(T×ratio)`개를 보존한다. validation은 이 축소 구간의 뒤 10%다. 장난감 배열 `T=19,20,100`에서 앞·뒤 인덱스와 ceil 경계를 직접 대조했고, 100%에서는 앞자르기와 뒷자르기 결과가 같음을 확인했다. 실제 GHL 625조건과 HAI 40조건은 모두 최소 train 길이를 넘었다.

GHL 뒷자르기는 앞자르기의 위치 편향을 확인하는 통제군이다. 계획서에 정한 5·20·100%만 쓰며 100%는 주 실행을 재사용한다.

### 3b-0B. GDN 하이퍼파라미터 봉인

원 논문, d-ailin/GDN, 고정 GraGOD 포크의 값을 항목별로 대조했다. 데이터셋별 튜닝값을 새 데이터에 옮기지 않고, 논문·실행 스크립트·공통 기본값이 겹치는 값을 우선했다. 모든 비율에 같은 설정을 써서 데이터 양 효과와 튜닝 효과가 섞이지 않게 했다(D-22).

| 항목 | 확정값 | 판단 근거 |
| --- | --- | --- |
| window | 5 | 원 논문 두 데이터셋과 d-ailin 실행값 일치 |
| embedding·output hidden | 64·128 | d-ailin 실행값과 GraGOD 기본값 |
| heads·dropout | 1·0.2 | d-ailin 구현과 GraGOD 설정 |
| batch·epoch | 32·최대 50 | d-ailin 실행값과 논문 최대 epoch |
| Adam | `lr=0.001`, `weight_decay=0`, `eps=1e-8`, `betas=(0.9,0.99)` | 논문·d-ailin·GraGOD 공통값과 포크 기본값 |
| early stopping | patience 10, delta 0 | 논문 값과 d-ailin 개선 판정 |
| validation | 0.1 | 계획서와 두 구현의 공통값 |
| topk | GHL 5, HAI 22 | `max(5,ceil(0.25N))`, N=19·86 |
| seed | GHL 1~3, HAI 1~10 | 계획서 반복 규격 |

0.25는 결과에서 최적화한 값이 아니다. 논문의 WADI `30/127=23.6%`와 SWaT `15/51=29.4%` 사이에 둔 사전 기준이며 하한 5는 d-ailin 실행값이다. GHL `k={2,5,10}` 민감도를 따로 두어 이 선택에 대한 의존성을 확인하게 했다(D-40).

### 3b-1. 단일·다중 세션 러너 작성

러너는 비율 적용과 입력 scaling이 끝난 배열만 받는다. GraGOD의 전체 train/predict 오케스트레이션 대신 검증한 loader와 GDN module을 직접 호출한다. 핵심 계약은 다음과 같다.

- 한 세션과 HAI 다중 세션을 같은 본체로 처리한다. 세션마다 `SlidingWindowDataset`을 만들고 dataset만 `ConcatDataset`으로 합쳐 경계 window를 막는다(D-25).
- best checkpoint는 fit당 한 번 고르고, train-reference와 test 추론에 같은 checkpoint를 쓴다.
- trainnorm 통계는 같은 세션의 train·validation 오차를 합친 뒤 세션별 오차 배열을 행축으로 합쳐 추정한다. test 값은 본표 정규화에 들어가지 않는다.
- 점수 길이는 `L-W`, 라벨은 `[W:]`다. GraGOD 공통 predict 경로의 마지막 1점 제거는 reconstruction 설명에서 온 동작이라 GDN에 적용하지 않는다(D-34).
- test 세션마다 raw/smoothed × trainnorm/testnorm × 집계/채널별 조합의 점수 배열 8개와 metadata를 저장한다.
- 비유한 forecast 오차는 저장하지 않는다. HAI embedding이 비유하거나 norm 0이면 graph를 만들지 않는다(D-39).

처음 구현은 GraGOD 경로를 따라 `L-W-1`을 썼으나 1-step target 수를 다시 추적한 감사에서 오류로 확인됐다. `datasets/dataset.py:47-50,64-77`은 마지막 행까지 포함한 `L-W`개의 target을 만든다. D-34 이후 코드·preflight·metadata를 모두 `L-W`로 통일했으며, 옛 합성 결과 191점은 현행 봉인 근거에서 제외했다.

### 3b-2. GHL·HAI loader 작성

GHL loader는 파일명 `tr_`로 train/test를 자르고 5·10·20·50·100%와 앞·뒤 방향을 적용한다. HAI loader는 train 4세션을 각각 자른 뒤 네 train 부분에 `MinMaxScaler.partial_fit`을 적용한다. 두 loader 모두 다음 실패를 모델 호출 전에 막는다.

- timestamp·label의 feature 혼입
- label 길이 불일치, 비유한 값, 0·1 이외 값
- scaler 반환 배열을 버리고 원본을 사용하는 오류
- HAI test와 label timestamp 불일치
- HAI 세션 경계를 넘는 raw tensor 연결
- Manifest에 없는 파일, 크기·SHA-256 불일치, 검증 전후 파일 변경

입력 SHA-256은 배치 시작 시 한 번 계산한다. 장시간 배치에서 검증 뒤 파일이 바뀌는 문제는 로드 직전·직후 크기와 `mtime_ns`를 비교해 막는다. 매 조합마다 대형 파일을 다시 해시하지는 않는다(D-36·D-42).

### 3b-3. 배치·대조 팔·그래프 산출물 작성

주 실행 조합은 GHL `25×5×3=375`, HAI `2×10=20`이다. 조합마다 독립된 `runs/` 폴더를 써 checkpoint·snapshot·로그 덮어쓰기를 막았다(D-26).

GHL 대조 팔은 다음과 같다.

| 대조 | 논리 조합 | 추가 fit | 이유 |
| --- | ---: | ---: | --- |
| back-trim 5·20·100% | 225 | 150 | 100%의 앞·뒤 입력은 같아 주 실행 재사용 |
| `GDN_NOTOPK` 10·100% | 150 | 150 | 학습 graph 기여 분리 |
| `GDN_K2/K5/K10` 10·100% | 450 | 300 | k=5 주 실행 150개 재사용 |

HAI는 best checkpoint의 embedding으로 TopK edge를 다시 계산한다. self-edge 포함본과 실제 제거본을 모두 저장한다. TopK가 self-edge를 반드시 고른다고 가정하지 않는다. seed 10개가 만드는 45쌍 전체의 Jaccard를 비율별로 계산해 중앙값·사분위·범위를 낸다(D-08·D-18·D-40).

### 3b-4. 합성 드라이런의 역할과 한계

당시 구현은 train `(300,5)`, test `(200,5)`, W=8, 2 epoch 합성 실행에서 점수 8개, checkpoint, early stopping 로그, TopK 복원을 확인했다(D-27). 외부 포크 Git hash가 `unknown`으로 기록된 첫 실패도 재현해, 명시된 포크 경로만 해당 Git 호출의 `safe.directory`로 넘기도록 고쳤다(D-32).

다만 이 실행의 점수 길이 191과 `label_slice=[8,-1]`은 D-34 이전 계약이다. 현행 기대값은 192와 `[8,null]`이다. 따라서 과거 드라이런은 실행 경로가 한 번 연결됐다는 역사 기록일 뿐 현재 코드의 봉인 증거가 아니다.

### 3b-5. snapshot·완료·재개 계약

실험 결과가 코드와 분리되지 않도록 다음 조건을 구현했다.

- 학습 전에 config 전체, 입력 지문, package 버전, TSAD·GraGOD commit을 snapshot으로 저장한다.
- TSAD와 GraGOD 작업 트리가 모두 clean이고 GraGOD HEAD가 고정 hash일 때만 시작한다. 종료 때도 두 저장소 상태를 다시 검사한다(D-35).
- `timing.json`에는 accelerator, 학습, train-reference 추론, test 추론 시간을 나눠 적는다.
- 필수 점수·metadata·checkpoint·early stopping·snapshot이 존재하고 0바이트가 아니어야 한다.
- 모든 저장과 종료 검증이 끝난 뒤에만 `COMPLETE`를 쓴다. snapshot의 두 hash가 현재 두 저장소와 같을 때만 재개 과정에서 완료로 센다.
- 이전 commit 산출물, 깨진 snapshot, HAI graph가 덜 저장된 실행은 다시 실행할 대상으로 돌린다.
- Jaccard는 현재 commit의 HAI 20조합이 전부 완료된 뒤에만 계산하고 결과 행마다 두 commit을 남긴다.

### 3b-6. 4단계 전 7회 감사에서 보강한 부분

기본 구현을 끝낸 뒤 같은 파라미터를 다시 고르는 대신 입력→분할→fit→점수→저장→재개 경계를 역순으로 점검했다.

| 감사 | 확인·수정한 핵심 | 검증 기록 |
| ---: | --- | --- |
| 1차, D-33~D-38 | `L-W` 정렬, clean Git, 입력·환경 계약, label·scaler, timing, 대조 팔 | 전체 단위 테스트 91건 |
| 2차, D-39 | EDA 봉인값 재계산, 후처리 설정 검사, 비유한 산출물 차단, `COMPLETE` 최종 기록 | 전체 96건 |
| 3차, D-40 | snapshot과 현재 commit 결합, 미완료 HAI graph 혼입 차단, self-edge 설명 정정 | 전체 99건 |
| 4차, D-41 | preflight의 `L-W-1` 잔존 제거, timing accelerator와 Jaccard commit 기록 | 전체 100건 |
| 5차, D-42 | Manifest 검증 뒤 실제 로드까지 파일 변경 차단, 순환 수정 여부 확인 | 전체 104건 |
| 6차, D-43 | 설정부터 재개까지 재추적, 실행 코드 결함 없음, HAI 채널 문구를 86으로 통일 | 5차의 전체 104건 유지 |
| 7차, D-44 | 패치 근거 경로, Jaccard Git 제외, preflight W 상태, 통계 일반화 범위 수정 | 관련 테스트 7건 |

2차 재대조에서는 GHL 25개·19채널·학습 39,938~50,000·feasibility 실패 0, HAI train 4세션·896,400행·86채널·feasibility 실패 0을 다시 확인했다. 4차에서는 실제 길이 23종의 validation ceil 계산과 GHL 625행·HAI 40행의 표본 수 수식을 대조했다. 5차에서는 D-34 이후 비율·seed·validation·window·topk가 한 번도 번복되지 않았음을 확인했다.

7차에서는 GHL 25개가 같은 simulator family에서 나온 task라는 한계를 통계 문서에 반영했다. Wilcoxon·TOST·bootstrap은 GHL 내부 민감도 요약이며 제조 공정 모집단 추론으로 확대하지 않는다.

## 3단계 종료 시점의 고정 계약

| 항목 | 현재 값 |
| --- | --- |
| 데이터 | GHL 25시계열×19채널, HAI 23.05 train 4·test 2세션×86채널 |
| 비율 | GHL 5·10·20·50·100%, HAI 10·100%, GHL back-trim 5·20·100% |
| GDN | W=5, topk GHL 5·HAI 22, batch 32, 최대 50 epoch, validation 0.1 |
| 반복 | GHL seed 1~3, HAI seed 1~10 |
| 점수 | 절대 오차, median·IQR, epsilon 0.01, 후행 4-창, 채널 max |
| 정렬 | 점수 `L-W`, 라벨 `[W:]` |
| 본표·부록 | trainnorm 본표, testnorm 부록, raw·smoothed와 채널별 배열 모두 보존 |
| 실행 신원 | 입력 SHA-256·환경·config·두 commit snapshot, 시작·종료 clean 확인 |
| 완료 기준 | 필수 산출물·graph·종료 검증 뒤 `COMPLETE`, 현재 commit과 snapshot 일치 |

## 4단계 진입 결과

3a와 3b의 설계·코드·정적 검증은 commit `597d03310201773f337e2256edfb417f088b0cfe`로 봉인했다. 고정 환경 전체 테스트 104건과 합성 드라이런의 8개 판정이 통과했다. 이어서 `GHL series 01·10%·seed 1`을 실행했고 점수 8개, metadata, timing, checkpoint, early stopping 로그, 현재 두 commit을 담은 snapshot과 `COMPLETE`를 확인했다.

GHL 주 배치는 현재 HEAD의 같은 스모크가 완료 상태일 때만 연다. 실행 중에는 tracked 파일을 바꾸지 않으며, 완료 기준은 `missing=0/375`다. GHL 주 배치가 끝나기 전에는 HAI 배치나 GHL 대조 팔로 넘어가지 않는다.

## 근거 파일 안내

- 결정 원본: `DECISIONS.md` D-01~D-46
- 코드 정찰: `experiments/exp00_gragod_recon/RECON.md`
- 1단계 실행 검증: `experiments/exp00_gragod_recon/VERIFICATION.md`
- 2단계 shape 검증: `experiments/exp00_gragod_recon/SHAPEFLOW.md`
- 패치·고정 환경 재검증: `experiments/exp00_gragod_recon/PATCH_REVERIFY.md`
- 데이터 계약: `docs/manifest_draft.md`, `configs/input_manifest.yaml`
- GHL·HAI EDA: `experiments/exp01b_ghl_preflight/`, `experiments/exp01c_hai_preflight/`
- 하이퍼파라미터: `docs/gdn_hyperparameter_decisions.md`, `configs/gdn_hyperparams.yaml`
- 점수 계약: `docs/score_interface.md`, `configs/scoring_pipeline.yaml`
- 4단계 진입 조건: `docs/pre_run_checklist.md`, `docs/NEXT_SESSION_PLAN.md`
