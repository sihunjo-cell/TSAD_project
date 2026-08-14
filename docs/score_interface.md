# 점수 파일 인터페이스 명세 — B(채점기)·C(통계) 수신용

작성일 2026-08-14. 저의 GDN 러너가 내놓는 `.npy` 점수 파일을 받아 쓸 때 필요한 전부를 여기에 담았다. 각 항목 끝의 괄호가 근거다. 코드와 이 문서가 어긋나면 코드가 맞다 — 그 경우 회의 안건으로 올려 달라.

## a. 점수 배열의 시점 대응

- 점수 배열 길이 = L − window_size − 1. L은 전처리 후 테스트 시계열 길이다.
  (근거: RECON [C] predict 경로 표 — 예측을 배치 concat 후 마지막 1행을 버리고(models/gdn/model.py:307-311), X_true를 `[window_size:-1]`로 자르는(models/predict.py:139) 구조를 러너가 그대로 복제. ORCHESTRATION.md 예측 경로 보강 표, src/gdn_runner/run_gdn_single.py:54-60)
- 점수 행 i는 원시계열 시점 window_size + i의 이상 점수다. 윈도 `[i, i+W)`가 시점 `W+i`를 1-스텝 예측한 오차에서 나온다.
- 라벨 대응은 `labels[window_size:-1]` — 이렇게 자르면 점수와 길이·시점이 원소 단위로 맞는다.
- window_size 값 자체는 실행마다 다를 수 있으므로 사이드카 메타(회의 안건 (i))로 전달한다. 드라이런 기준은 8이었다.

## b. 파일명 규약과 채점 대상의 격

규약 전문 (D-15 확장형 + D-16 보조 산출물, CLAUDE.md 파일명 규약):

```
{dataset}__{series:02d}__{model}__{tier}__r{ratio:03d}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}(__channels).npy
```

- series 두 자리 제로 패딩, ratio 세 자리(005/010/020/050/100). seed는 패딩 없이 쓴다(s1, s10).
- 생성·역파싱 코드는 `src/common/naming.py` — 순수 stdlib라 B가 그대로 복사해 써도 된다(환경 이원화와 무충돌). `parse_score_filename`은 규약 위반 이름을 ValueError로 거부한다.

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

median·IQR은 validation 분할 전의 축소 학습 배열 전체(train_part + val_part)가 낸 채널별 절대 오차에서 추정한다. `src/gdn_runner/run_gdn_single.py`의 해당 줄 인용:

```python
reduced_tensor = torch.tensor(numpy.asarray(train_array), dtype=torch.float32)   # 103행
train_val_errors = compute_absolute_errors(best_module, reduced_tensor, **error_arguments)  # 157행
median, iqr = estimate_median_iqr(train_val_errors)                              # 171행
```

- 오차 계산 시 이 배열도 `[window_size:-1]`로 잘리므로(54-60행) 통계 표본 길이는 축소 길이 − W − 1이다.
- 추정한 (median, iqr)를 테스트 오차에 적용한 것이 trainnorm이다. 테스트 구간 통계는 trainnorm 어디에도 쓰이지 않는다.
- testnorm은 저자 원본 방식 재현 — 테스트 오차 자체에서 통계를 추정한다(177행, RECON [D]: d-ailin evaluate.py:52도 같은 방식).
- epsilon은 1e-2, 값의 소유는 `configs/scoring_pipeline.yaml` (D-07).

## d. smoothing 명세 (D-04·D-05)

- 시간축 후행 4-창 평균(현재 시점 포함), 처음 3개 시점은 0, 채널별 독립 적용. d-ailin evaluate.py:62-65 방식의 자체 구현(`src/common/smoothing.py`)이고, GraGOD의 smooth_scores는 feature 축에 작용하는 버그가 있어 어느 경로에서도 쓰지 않는다(VERIFICATION 의혹 2).
- 순서는 정규화 → smoothing → 집계로 고정한다 (D-04). smoothed 집계본은 채널별 정규화 점수를 smoothing한 뒤 max를 취한 것이다.
- **회의 안건**: B가 계층1·3 점수에도 같은 smoothing을 적용해 smoothed 벌을 만들 것인가. 계획서 5-4의 "GDN만 이중 완화" 논리는 GDN에만 smoothing이 있던 원 구현 기준이라, 계층1·3의 smoothed 벌 생산 여부는 팀 결정 사항이다.

## e. 회의 안건 (미결 3건)

- (i) **길이·offset 메타 전달 방식.** 제안: 집계본과 같은 이름의 `.meta.json` 사이드카 — 필드 4개(window_size, test_length, score_length, label_slice). 러너가 이미 저장하도록 붙여 뒀고(`src/common/save_scores.py`의 `save_score_metadata`), 파일명 앵커는 본표 채점 대상인 raw·trainnorm 집계본이다. 값은 그 실행의 8개 파일 전부에 공통이다. 채택 여부와 필드 확정은 회의에서.
- (ii) **예약 토큰.** 제안: 무작위 대조군은 `model=RANDOM`, `tier=t0`은 합성·검증 전용으로 예약. 드라이런이 이미 `SYNTH__…__t0__…`을 쓰고 있어 자연스럽다. naming.py는 임의 문자열을 허용하므로 예약은 문서 규약으로만 강제된다.
- (iii) **zero-shot의 ratio 표기.** 제안: 학습 데이터를 안 보므로 5개 비율 값이 같다 — `r100` 한 벌만 저장하고 수평선 전개는 C가 결과표에서 한다.

## 문의 경로

파서 호환 여부 회신(B)과 안건 결정은 팀 회의로. 명세 자체의 오류 발견 시 이 문서가 아니라 코드·DECISIONS.md를 먼저 확인해 달라 — 문서는 코드의 그림자다.
