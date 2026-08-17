# 점수 파일 인터페이스 명세 — B(채점기)·C(통계) 수신용

작성일은 2026-08-14, 마지막 개정일은 2026-08-17이다. GDN 러너의 `.npy` 점수를
받아 쓸 때 필요한 계약을 적었다. 코드와 문서가 다르면 어느 한쪽을 임의로 따르지 말고
실행을 멈춘다.

## a. 점수 배열의 시점 대응

- 점수 배열 길이는 `L-window_size`다. L은 전처리 후 테스트 시계열 길이다.
  GraGOD `datasets/dataset.py:47-50,71-77`은 각 window 뒤의 1개 시점을 예측해
  `L-W`개 forecast를 만든다. `models/gdn/model.py:307-316`의 마지막 1점 제거는
  reconstruction용 설명을 GDN에 잘못 적용한 부분이라 우리 러너에서 쓰지 않는다(D-34).
- 점수 행 i는 원시계열 시점 window_size + i의 이상 점수다. 윈도 `[i, i+W)`가 시점 `W+i`를 1-스텝 예측한 오차에서 나온다.
- 라벨 대응은 `labels[window_size:]`다. GDN의 1-step forecast는 길이 `L-W`이고
  마지막 예측은 마지막 입력 행 `X[L-1]`과 정상적으로 대응한다.
- window_size는 사이드카 metadata로 전달한다. GHL·HAI 본 실험값은 5이고 합성
  드라이런값은 8이다.

## b. 파일명 규약과 채점 대상의 격

규약 전문 (D-15 확장형 + D-16 보조 산출물, AGENTS.md 파일명 규약):

```
{dataset}__{series:02d}__{model}__{tier}__r{ratio:03d}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}(__channels).npy
```

- series 두 자리 제로 패딩, ratio 세 자리(005/010/020/050/100). seed는 패딩 없이 쓴다(s1, s10).
- 생성·역파싱 코드는 `src/common/naming.py` — 순수 stdlib라 B가 그대로 복사해 써도 된다(환경 이원화와 무충돌). `parse_score_filename`은 규약 위반 이름을 ValueError로 거부한다.
- 주 실행 model은 `GDN`, 대조 팔은 `GDN_NOTOPK`, `GDN_K2`, `GDN_K10`이다.
  back-trim은 파일명을 바꾸지 않고 `back_trim_runs/` 상대경로로 구분하므로 전달할 때
  폴더 구조를 평탄화하면 안 된다.

실물 예시 — 합성 드라이런이 실제로 만든 8개 파일명 그대로:

```
SYNTH__00__GDN__t0__r100__s1__raw__testnorm.npy
SYNTH__00__GDN__t0__r100__s1__raw__testnorm__channels.npy
SYNTH__00__GDN__t0__r100__s1__raw__trainnorm.npy
SYNTH__00__GDN__t0__r100__s1__raw__trainnorm__channels.npy
SYNTH__00__GDN__t0__r100__s1__smoothed__testnorm.npy
SYNTH__00__GDN__t0__r100__s1__smoothed__testnorm__channels.npy
SYNTH__00__GDN__t0__r100__s1__smoothed__trainnorm.npy
SYNTH__00__GDN__t0__r100__s1__smoothed__trainnorm__channels.npy
```

격 구분 (D-15·D-16):

| 격 | 파일 | 용도 |
|---|---|---|
| 채점 대상 | `__channels` 없는 집계본 (1차원, 채널 max 집계 — D-09) | **trainnorm이 본 결과표**, testnorm은 부록 별도 열 |
| 채점 제외 | `__channels` 붙은 채널별 배열 (2차원) | 회수 분석 전용 (D-16) |

## c. 정규화 통계의 추정 구간 (D-06) — 코드 기준 서술

median·IQR은 축소 학습 배열의 train과 validation을 원래 세션별로 다시 붙여 얻은
채널별 절대 오차에서 추정한다. 여러 HAI 세션은 window를 서로 가로질러 만들지 않고
오차 배열만 행축으로 합친다. 러너의 핵심 흐름은 아래와 같다.

```python
train_val_errors = numpy.concatenate([
    compute_absolute_errors(best_module, torch.cat((train, validation)), **arguments)
    for train, validation in zip(train_tensors, validation_tensors)
])
median, iqr = estimate_median_iqr(train_val_errors)
```

- 세션별 통계 표본 길이는 축소 길이 `- W`다.
- 추정한 (median, iqr)를 테스트 오차에 적용한 것이 trainnorm이다. 테스트 구간 통계는 trainnorm 어디에도 쓰이지 않는다.
- testnorm은 저자 원본 방식 재현 — 테스트 오차 자체에서 통계를 추정한다(177행, RECON [D]: d-ailin evaluate.py:52도 같은 방식).
- epsilon은 1e-2, 값의 소유는 `configs/scoring_pipeline.yaml` (D-07).
- 모델 예측에서 계산한 채널별 절대 오차가 하나라도 비유한 값이면 점수 파일을 만들지 않는다(D-39).

## d. smoothing 명세 (D-04·D-05)

- 시간축 후행 4-창 평균(현재 시점 포함), 처음 3개 시점은 0, 채널별 독립 적용. d-ailin evaluate.py:62-65 방식의 자체 구현(`src/common/smoothing.py`)이고, GraGOD의 smooth_scores는 feature 축에 작용하는 버그가 있어 어느 경로에서도 쓰지 않는다(VERIFICATION 의혹 2).
- 순서는 정규화 → smoothing → 집계로 고정한다 (D-04). smoothed 집계본은 채널별 정규화 점수를 smoothing한 뒤 max를 취한 것이다.
- B가 계층1·3 점수에도 같은 smoothing을 적용할지는 이 GDN 인터페이스 범위 밖이다.
  GDN은 raw·smoothed를 모두 저장하므로 어느 쪽을 택해도 재학습은 필요 없다.

## e. metadata 확정과 남은 팀 안건

- 길이·offset은 집계본과 같은 이름의 `.meta.json` 사이드카로 확정했다. 필드는
  `window_size`, `test_length`, `score_length`, `label_slice` 네 개이며
  `label_slice=[W,null]`이다. 파일명 앵커는 raw·trainnorm 집계본이다.
- (ii) **예약 토큰.** 제안: 무작위 대조군은 `model=RANDOM`, `tier=t0`은 합성·검증 전용으로 예약. 드라이런이 이미 `SYNTH__…__t0__…`을 쓰고 있어 자연스럽다. naming.py는 임의 문자열을 허용하므로 예약은 문서 규약으로만 강제된다.
- (iii) **zero-shot의 ratio 표기.** 제안: 학습 데이터를 안 보므로 5개 비율 값이 같다 — `r100` 한 벌만 저장하고 수평선 전개는 C가 결과표에서 한다.

## 문의 경로

파서 호환 여부 회신(B)과 안건 결정은 팀 회의로. 명세 자체의 오류 발견 시 이 문서가 아니라 코드·DECISIONS.md를 먼저 확인해 달라 — 문서는 코드의 그림자다.
