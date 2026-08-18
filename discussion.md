## 회의 필요: GDN

### 1. 무엇을 재현 기준으로 삼을지

현재 설정은 한 구현의 기본값이 아니다.

| 항목 | 현재 설정 | d-ailin 공식 구현 기본값 | GraGOD 일반 설정 | GraGOD SWaT 설정 |
| --- | ---: | ---: | ---: | ---: |
| window | 5 | 15 | 50 | 5 |
| embedding | 64 | 64 | 64 | 64 |
| output hidden | 128 | 256 | 128 | 64 |
| dropout | 0.2 | 0.2 | 0.0 | 0.2 |
| batch | 32 | 128 | 32 | 512 |
| 최대 epoch | 50 | 100 | 2 | 200 |
| early stopping patience | 10 | 15 | 20 | 10 |
| Adam betas | 0.9, 0.99 | PyTorch 기본값 | 0.9, 0.99 | 0.9, 0.999 |

현재 조합은 window와 dropout은 GraGOD SWaT, hidden과 batch·betas는 GraGOD 일반 설정,
epoch는 위 세 기준 어디에도 없는 값이다. GraGOD의 학습률 scheduler와 gradient clipping도
d-ailin 공식 학습 코드에는 없다. 이 상태를 `저자 기본값`이라고 부르면 안 된다.

권장안은 d-ailin 공식 구현의 공개 기본값을 기준선으로 삼고, 이 연구에 필요한 시간순
validation·trainnorm·데이터셋별 topk만 명시적 변경으로 적는 것이다. 계산 예산 때문에 최대
50 epoch를 쓸 경우에는 `저자 기본값`이 아니라 사전 고정한 예산 상한이라고 밝혀야 한다.
GraGOD SWaT 설정을 택한다면 GHL·HAI에 옮기는 이유와 batch 512를 그대로 쓸지 따로 합의해야
한다.

### 2. 학습 window 간격

d-ailin 공식 실행 기본값은 학습 window 시작점을 5칸씩 이동시키고 테스트는 1칸씩 이동시킨다.
현재 GraGOD loader는 학습·validation·test 모두 1칸 간격이다. 같은 원시 데이터라도 현재
러너는 공식 구현보다 겹치는 학습 window를 약 5배 많이 만든다. 저데이터 성능을 비교하는
연구에서는 이 차이가 특히 크다.

`train stride=5, validation·test stride=1`을 쓸지, 전 구간 stride 1을 쓸지 실험 전에
고정해야 한다. 공식 구현 재현을 우선하면 train stride 5가 권장안이다. 현재 코드에는 stride
설정 자체가 없으므로 하이퍼파라미터 회의와 함께 처리해야 한다.

### 3. topk 5와 22의 지위

`k=max(5, ceil(0.25N))`은 논문 표준이나 EDA 결과가 아니다. WADI와 SWaT의 두 설정 사이에서
만든 사전 휴리스틱이다. GHL 5와 HAI 22를 유지하려면 `데이터에서 고른 최적값`이 아니라
`채널 수에 비례시킨 고정 규칙`이라고 적어야 한다.

규칙을 유지할지, 공식 구현의 고정 topk를 쓸지 회의에서 결정한다. 유지한다면 k/2의 반올림,
2k가 채널 수를 넘을 때의 처리, 민감도 분석 범위도 함께 정한다. 현재 구현처럼 self-edge를
가리기 전에 TopK를 뽑는 방식은 d-ailin과 GraGOD의 동작과 맞으므로 그대로 둘 수 있다.

### 4. raw와 smoothed 중 본 결과

코드는 두 점수를 모두 저장하지만 어느 쪽이 주 분석인지 정하지 않았다. 결과를 본 뒤 더 좋은
쪽을 고르면 비교가 오염된다. VUS-PR 자체가 시간 오차에 buffer를 주므로 smoothing까지 본
결과로 쓰면 시간 허용이 두 번 들어갈 수 있다.

권장안은 `raw + trainnorm`을 본 결과로, `smoothed + trainnorm`을 민감도 분석으로 두는 것이다.
`testnorm`은 현재 계획대로 부록에만 둔다. 다른 선택을 하더라도 실험 전에 한 가지를 본 결과로
고정해야 한다.

## 회의 필요: 나머지 모델과 공통 점수

현재 `src/models/tier1/`과 `src/models/tier3/`은 비어 있고, 계층 2도 GDN 외 모델이 없다.
따라서 아래 항목은 구현 전에 먼저 한 문장씩 고정해야 한다.

| 모델 | 아직 정하지 않은 핵심값 |
| --- | --- |
| z-score·L2 | 채널 점수와 집계 점수의 정확한 정의, 표준편차 0 처리 |
| PCA | 주성분 10·30의 근거 또는 설명된 고정 규칙, 채널별 재구성 오차 정의 |
| KNN | window, k, 거리, 학습 참조 집합 크기와 메모리 처리 |
| IForest | 입력 단위, 트리 수, subsample, random seed |
| MatrixProfile | subsequence 길이와 train-reference/test-query 계산 방식 |
| 채널독립 AE | 채널별 개별 모델인지 공유 모델인지, window와 latent 크기 |
| LSTMAD·USAD | 공식 구현 기준, window, architecture, epoch, early stopping |
| MOMENT_ZS | checkpoint, 입력 길이·patch, 채널 처리, 이상 점수 정의 |
| Chronos | 실제 포함 여부, checkpoint, forecast 문맥과 점수 정의 |

현재 점수 저장 함수는 채널별 2차원 오차를 전제로 한다. KNN·IForest·MatrixProfile처럼 처음부터
1차원 점수를 내는 모델에 `__channels`를 억지로 만들면 안 된다. 모델 공통 필수 산출물이
1차원 raw 점수인지, 채널 점수는 만들 수 있는 모델만 보조로 둘지 먼저 정해야 한다. 결정적
계층 1 모델의 파일명에 어떤 seed를 쓸지도 함께 고정한다.

전처리도 같은 문제가 있다. 공통 MinMax를 모든 모델 앞에 둘지, 모델 공식 구현의 내부
전처리를 인정할지 아직 명시되지 않았다. 권장안은 train에만 fit한 공통 입력 변환을 우선하고,
모델 내부에서 중복 scaling이 일어나지 않게 모델별 계약을 적는 것이다. PCA처럼 scaling에
민감한 모델은 이 결정이 성능을 바꾸므로 구현 뒤에 정하면 안 된다.

## 담당자 산출물 대기

### 지우: VUS-PR과 ℓ_max

threshold 250개는 운영 threshold가 아니라 VUS 곡선을 수치 적분할 때 쓰는 격자 해상도다.
`VUS-PR은 threshold와 무관하다`는 문장과 `250개 threshold를 쓴다`는 문장은 이 구분 없이
읽으면 모순처럼 보인다. TSB-AD의 `thre=250`을 그대로 쓸지, 정확한 구현 버전과 함께 지우가
고정해야 한다.

ℓ_max도 `채널별 자기상관 국소 최대의 중앙값`만으로는 실행할 수 없다. 아래가 빠져 있다.

- lag 탐색 하한·상한과 국소 최대가 없는 채널 처리
- GHL 한 시계열 안의 채널 중앙값을 다시 25개 시계열에서 합칠지 여부
- HAI 4개 훈련 세션의 결과를 합치는 순서와 세션 경계를 지키는 방법
- ℓ/2와 2ℓ의 정수화 규칙

테스트 라벨의 이상 구간 길이는 선택 근거에서 제외한다는 원칙은 유지한다. 지우가 위 세부와
무작위 점수·point-adjust 대조를 끝내기 전에는 채점 결과를 만들지 않는다.

### 주혜: 난이도와 통계

난이도 규칙의 q99, window 길이, 구간을 덮는 여러 window의 집계 방식은 아직 제안값이다.
학습 표준편차가 0인 채널도 GHL 저비율과 HAI에 실제로 있으므로 0으로 나누는 경우를 먼저
정해야 한다. 이 규칙은 threshold를 학습 데이터에서 정해 모델 점수에는 독립적이지만, 이상
구간별 이름표를 붙일 때 테스트 라벨의 구간 경계는 사용한다. `라벨을 전혀 쓰지 않는다`보다
`난이도 threshold를 테스트 성능으로 고르지 않는다`가 정확한 설명이다.

통계 계획에는 두 가지 공백이 있다.

1. GDN-PCA와 GDN-MOMENT_ZS를 모두 주 비교로 두면 비율 5개씩 총 10개 검정이다. 현재
   Bonferroni `0.05/5=0.01`은 두 비교를 별도 family로 볼 때만 맞다. 하나의 family인지 두
   family인지 사전 등록해야 한다.
2. 계층 1 모델의 시계열 간 성능 산포를 등가성 경계 δ로 쓰는 논리는 약하다. 시계열 간 차이는
   측정 오차가 아니라 task 난이도 차이일 수 있다. 실무적으로 무시할 VUS-PR 차이를 먼저
   정의하거나 외부 근거를 제시하는 편이 낫다.

GHL 25개가 같은 simulator family라 일반 모집단 표본이 아니라는 제한, 교차점을 관측한 다섯
비율 안의 이산 위치로만 보고한다는 원칙은 유지해도 된다.

## 회의 순서

1. GDN 재현 기준, train stride, topk를 한 번에 확정한다.
2. raw·smoothed의 본 결과와 모델 공통 점수 계약을 확정한다.
3. 각 모델의 공식 구현과 고정값을 모델별로 하나씩 봉인한 뒤 구현한다.
4. 지우가 VUS-PR·ℓ_max를, 주혜가 난이도·다중검정·δ를 봉인한다.
5. 서버 데이터 경로를 확인한 뒤 합성 dry-run을 한 번만 실행한다.
6. 위 항목이 모두 끝났을 때만 GHL·HAI 실데이터 학습을 연다.

## 확인한 범위

저장소 안의 Python 파일 47개를 모두 읽고 계획·결정·역할 문서, YAML 설정, 현재 감사 결과와
실행 흐름을 대조했다. 외부 기준은 d-ailin/GDN의
`main.py`, `train.py`, `datasets/TimeDataset.py`, `models/GDN.py`와 고정 GraGOD 포크의
`models/gdn/params.yaml`, `models/gdn/params_swat.yaml`, `datasets/dataset.py`,
`gragod/training/trainer.py`를 확인했다.
