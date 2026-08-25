# 점수 파일 인터페이스

모든 모델은 지우의 채점기와 주혜의 통계 코드에 같은 형식으로 점수를 넘긴다. 문서와 실제
파일이 다르면 임의로 보정하지 말고 실행을 멈춘다.

## 시점과 라벨

GDN의 점수 길이는 `L-window_size`다. 점수 행 `i`는 원시계열 시점
`window_size+i`의 1-step forecast 오차이며 라벨은 `labels[window_size:]`와 대응한다.
`window_size`는 metadata에 기록한다. 현재 합성 검증값은 5이며 본 실험값은 재현 기준 회의에서
확정한다.

다른 모델도 실제 점수 길이와 라벨 범위를 metadata에 적어야 한다. 채점기는 모델 이름으로
offset을 추측하지 않는다.

## 파일명과 채점 대상

```
{dataset}__{series:02d}__{model}__{tier}__r{ratio:03d}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}(__channels).npy
```

series는 두 자리, ratio는 `005`, `010`, `020`, `050`, `100`처럼 세 자리로 쓴다. seed는
패딩하지 않는다. 예시는 `GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy`다.
HAI 시간 조건은 series `01`=train1→test1, `02`=train1+train2→test2로 기록한다. 두 조건은
각각 사용 가능한 train 전체를 쓰므로 ratio는 `100`이다.

| 구분 | 형태 | 용도 |
| --- | --- | --- |
| 본 채점 | `__channels`가 없는 1차원 집계본 | `trainnorm`은 본 결과, `testnorm`은 부록 |
| 보조 배열 | `__channels`가 붙은 2차원 채널별 점수 | 회수 분석과 재집계 |

raw와 smoothed는 항상 함께 만든다.

## 정규화와 smoothing

trainnorm의 median과 IQR은 누적 학습·validation 구간의 채널별 절대 오차에서 추정한다.
HAI처럼 세션이 여러 개면 세션 사이에 윈도를 만들지 않고 오차 배열만 행 방향으로 합친다.
이 통계를 테스트 오차에 적용하며 테스트 통계는 trainnorm에 쓰지 않는다.

testnorm은 테스트 오차 자체에서 median과 IQR을 추정한 부록 대조본이다. epsilon은 0.01이다.
오차에 비유한 값이 있으면 점수 파일을 만들지 않는다.

smoothed는 정규화한 채널별 점수에 시간축 후행 4칸 평균을 적용한 뒤 max로 집계한다. 처음
3개 시점은 0이다. 처리 순서는 정규화 → smoothing → 채널 집계이며 raw는 smoothing만
건너뛴다.

## metadata

집계본과 같은 이름의 `.meta.json`에는 기본 필수 4개와, 본 채점 F1용 추가 4개를 둔다.

- `window_size`
- `test_length`
- `score_length`
- `label_slice`
- `validation_threshold` (point-adjust 없는 보조 F1용)
- `validation_threshold_quantile` (`0.99`으로 고정)
- `validation_score_count`
- `validation_score_source` (`raw_trainnorm_aggregated_validation`으로 고정)

GDN의 `label_slice`는 `[W,null]`이다. metadata는 raw·trainnorm 집계본을 기준으로 저장한다.
`validation_threshold`는 모델별 정상 validation의 **raw·trainnorm 집계 점수**에서 계산한 99% 분위수다.
테스트 점수·라벨로 최적화하지 않으며 point-adjust도 적용하지 않는다. 채점기는 F1용 추가 4개 필드가 없거나
분위수·점수 출처가 이 계약과 다르면 본 결과 원표 생성을 중단한다.
validation score가 1,000개 미만인 조건의 F1은 `auxiliary_low_validation_sample`로 표시하며,
VUS-PR·AUPRC 해석에는 영향을 주지 않는다.
F1은 raw/trainnorm 본 결과에서만 계산한다. smoothed 또는 testnorm 부록 원표에는 VUS-PR와
AUPRC만 기록하며 F1은 `not_applicable_appendix_score`로 표시한다.

무작위 대조군은 `model=RANDOM`, 합성 검증은 `tier=t0`을 쓴다. zero-shot 모델은 학습 비율과
무관하므로 `r100` 한 벌만 저장하고 결과표에서 필요한 비율 위치에 표시한다.
