# 점수 파일 인터페이스

모든 모델은 지우의 채점기와 주혜의 통계 코드에 같은 형식으로 점수를 넘긴다. 문서와 실제
파일이 다르면 임의로 보정하지 말고 실행을 멈춘다.

## 시점과 라벨

원 wrapper가 앞뒤 점수를 복제해 길이를 `L`로 맞추는 방식은 쓰지 않는다. 실제로 계산된 core
점수만 저장하고 대응하는 원시 시점 범위를 metadata에 적는다.

| 모델 | 입력 길이 | 점수 길이 | 첫 점수의 원시 index | 대응 범위 |
| --- | ---: | --- | ---: | --- |
| CI-AE | 100 | `L-99` | 50 | `source[50:L-49]` |
| LSTM-AD | 100 | `L-100` | 100 | `source[100:L]` |
| USAD | 10 | `L-9` | 5 | `source[5:L-4]` |
| GDN | 5 | `L-5` | 5 | `source[5:L]` |

CI-AE와 USAD의 재구성 점수는 window 중앙 시점에 둔다. LSTM-AD와 GDN은 1-step 예측
대상이므로 `labels[W:]`와 맞춘다. 채점기는 모델 이름으로 offset을 추측하지 않고 metadata의
범위를 그대로 쓴다.

## 파일명과 채점 대상

```
{dataset}__{series:02d}__{model}__{tier}__r{ratio:03d}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}(__channels).npy
```

series는 두 자리, ratio는 `005`, `010`, `020`, `040`, `060`, `080`, `100`처럼 세 자리로 쓴다. seed는
패딩하지 않는다. 예시는 `GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy`다.

| 구분 | 형태 | 용도 |
| --- | --- | --- |
| 본 채점 | `__channels`가 없는 1차원 집계본 | `trainnorm`은 본 결과, `testnorm`은 부록 |
| 보조 배열 | `__channels`가 붙은 2차원 채널별 점수 | 회수 분석과 재집계 |

raw와 smoothed는 항상 함께 만든다.

## 정규화와 smoothing

trainnorm의 median과 IQR은 누적 학습·validation 구간의 채널별 절대 오차에서 추정한다.
HAI처럼 세션이 여러 개면 세션 사이에 윈도를 만들지 않고 오차 배열만 행 방향으로 합친다.
이 통계를 테스트 오차에 적용하며 테스트 통계는 trainnorm에 쓰지 않는다.

학습 손실은 모델별 구현을 따르지만 저장 점수는 절대오차로 통일한다. CI-AE와 USAD는
window 안의 절대오차를 채널별 평균한 뒤 중앙 시점에 놓는다. LSTM-AD와 GDN은 1-step
대상의 채널별 절대오차를 그대로 쓴다.

testnorm은 테스트 오차 자체에서 median과 IQR을 추정한 부록 대조본이다. epsilon은 0.01이다.
오차에 비유한 값이 있으면 점수 파일을 만들지 않는다.

smoothed는 정규화한 채널별 점수에 시간축 후행 4칸 평균을 적용한 뒤 max로 집계한다. 처음
3개 시점은 0이다. 처리 순서는 정규화 → smoothing → 채널 집계이며 raw는 smoothing만
건너뛴다.

## metadata

집계본과 같은 이름의 `.meta.json`에는 아래 필드를 둔다.

- `window_size`
- `test_length`
- `score_length`
- `label_slice`
- `source_start`
- `source_end_exclusive`
- `alignment`

`label_slice`는 `[source_start,source_end_exclusive]`다. metadata는 raw·trainnorm 집계본을
기준으로 저장한다.

무작위 대조군은 `model=RANDOM`, 합성 검증은 `tier=t0`을 쓴다. zero-shot 모델은 학습 비율과
무관하므로 `r100` 한 벌만 저장하고 결과표에서 필요한 비율 위치에 표시한다.
