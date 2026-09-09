# 점수·선택 인터페이스

모든 모델은 지우님의 채점기와 주혜님의 통계 코드에 같은 형식으로 threshold 전 연속 score를
넘긴다. 문서와 배열이 다르면 임의로 padding하거나 자르지 않고 실행을 중단한다.

## score shape와 시점

본 채점 배열은 원시 시점에 정렬된 1차원 `(score_length,)`다. 채널 score를 만들 수 있는 모델은
`(score_length, channel_count)` 보조 배열도 저장한다. 채점기는 모델 이름으로 offset을 추측하지
않고 metadata의 `source_start`, `source_end_exclusive`, `label_slice`, `alignment`를 읽는다.

| 모델 | 본 score 계약 |
| --- | --- |
| MWVAR | 입력과 같은 길이 `L`, 등록된 중앙 window의 채널 score를 집계 |
| SQDIFF_LAST1·LAST3·CENTERED5 | 입력과 같은 길이 `L`, 각 공식 경계 규칙으로 채움 |
| 두 MWVAR96 앙상블 | 전체 평가 component min-max 후 component·채널 max, 길이 `L` |
| PCA_LEGACY | 전체 평가 입력 fit·zero pruning과 weighted component distance, 길이 `L` |
| PaAno | test 세션 길이 `L`, 세션 경계를 넘기지 않은 point score |
| GDN | native 교정·후처리를 마친 길이 `L-W`, `source_start=W`, `labels[W:]` |
| TimeRCD | scalar `(L,)` |
| TSPulse | 공식 time·fft·pred·ensemble의 native 후처리 점수, 길이 `L` |

GDN은 공식 `d-ailin/GDN` 적응 구현 하나만 이 표의 대상이다. 과거 HAI 전용 GDN output과
edge 배열은 입력으로 받지 않는다.

## 파일명과 저장 위치

```text
{dataset}__{series:02d}__{model}__{tier}__r{ratio:03d}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}(__channels).npy
```

`ratio`는 `005`, `010`, `020`, `040`, `060`, `080`, `100`이다. HPO 설정은 파일명이 아니라
`config_id` 실행 디렉터리로 구분한다.

```text
experiments/01_ghl_main/scores/{dev18|ghl25}/{tier}/{model}/{config_id}/r{ratio}/s{seed}/<score filename>
```

경로의 `dev18`은 TSB 비-GHL 튜닝 패널 18개의 기존 내부 식별자다. 최종 데이터셋 이름이 아니다.
HAI는 `experiments/02_hai_extension/` 아래에서 같은 파일명 규칙을 쓴다.

`config_id`는 model, source commit, source checkpoint SHA-256, hyperparameters와 고정 전처리
recipe의 canonical JSON SHA-256 앞 12자리에 `c`를 붙여 만든다. dataset, series, `q`, seed와
실행 시각은 넣지 않는다. 같은 score를 여러 분석이 쓰면 복사하지 않고 manifest가 같은
`score_sha256`을 참조한다.

## 정규화와 smoothing

주점수는 `raw__trainnorm`이다. `paper_tuning_v4`에서는 raw와 smoothed에 같은 native 점수를
저장한다. 이름은 기존 인터페이스를 유지하며 실제 교정 출처는 metadata로 판정한다.

현재 q-prefix 전부를 모델에 전달한다. PaAno는 native RevIN과 학습 loss checkpoint,
GDN은 prefix의 window 내부 검증과 최저 검증 loss checkpoint를 쓴다. GDN 입력 scaler는
prefix에만 맞추고 점수는 전체 평가 채널 오차의 median·IQR과 native trailing 4·max를 적용한다.
PCA·One-Liner 앙상블·Tier 3도 계획서의 공식 전체 평가 통계 범위를 따른다.

native 점수에 공통 교정이나 smoothing을 덧붙이지 않는다. 채널 score가 남은 모델만 채널
max로 집계하고 공식 scalar score에는 가짜 채널 축을 만들지 않는다.

q-prefix를 사용하지 않는 Tier 1·Tier 3는 동일한 물리 score를
일곱 `q`에 재사용할 수 있으며 저장 표시는 `physical_ratio=100`이다. 이는 100% target 데이터로
recipe를 골랐다는 뜻이 아니다.

## metadata

집계 score의 `.meta.json`에는 적어도 아래 필드를 둔다.

```text
window_size, test_length, score_length, label_slice,
source_start, source_end_exclusive, alignment, score_primitive,
channel_count, dataset_role, split_role, config_id,
config_registry_sha256, input_manifest_sha256,
source_commit, source_checkpoint_sha256,
run_checkpoint_manifest_sha256, training_prefix_ratio,
hyperparameters, normalization_scope, calibration_reference,
execution_evidence, uses_test_labels_for_scoring
```

`uses_test_labels_for_scoring`은 항상 `false`다. scratch 모델의
`source_checkpoint_sha256`은 문자열 `none`이고, target 학습으로 생긴 checkpoint는
`run_checkpoint_manifest_sha256`으로 연결한다.

`execution_evidence`에는 다음 값을 둔다.

```text
measurement_protocol_id, execution_phase, status, retry_count,
training_sessions, test_sessions, timing, runtime_seconds,
peak_memory_mb, model_artifact_bytes
```

Dev18의 `measurement_protocol_id`는 `dev18_registered_runner.full_prefix_v3`다. `timing`은 등록 executor 안에서
`split_preprocess_seconds`, `model_setup_seconds`, 학습, calibration 추론, test 추론을 나눠 기록한다.
다섯 값의 합은 `runtime_seconds`와 맞아야 한다. 세션에는 관측 수와 검증 가능한 지속시간·근거를 둔다.
지속시간을 입증하지 못하면 값을 만들지 않고 `null`과 `duration_basis=unavailable`을 쓴다.

## TSB 튜닝 채점 원표

`dev18_trial_score_ledger.csv`는 아래 정보를 score manifest에 연결한다.

```text
series, family, tier, model, config_id, ratio, seed,
score_variant, normalization, vus_pr,
score_file, score_sha256, evaluator_sha256, ell_max_id,
status, status_reason
```

공식20에서 GHL 09·18을 뺀 18개만 이 원표에 들어간다. q별 가능 후보 집합이 같은 파일을
조건 집단으로 묶고, 같은 파일의 seed 평균 → family 안 파일 평균 → 포함된 family 동일 가중을 적용한다.

```text
J(model,config,q)
  = mean_family mean_{series in family}(mean_seed VUSPR)
```

같은 조건 집단에서 파일을 동일 가중한 series macro는 보조값이다. 후보 전체를 비교하는
`full_prefix_per_ratio`에서 모델·q별 설정을 독립 선택하며 family leave-one-out도 같은 q에서 수행한다.


## 필수 선택표

`model_ratio_policy.csv`는 모델·q·조건 집단별 설정을, `tier_adaptive.csv`는 같은 조건의
모델과 설정을 함께 선택한 결과를 담는다. family-LOFO로 선택 절차를 평가하고 실제 전달
설정은 해당 조건 집단 전체에서 고른다. q60과 q80의 config가 달라도 된다.
과거 model_fixed·tier_fixed와 294행 고정 membership은 현행 계약이 아니다.

## 최종 실행 요청

`final_policy_membership.csv`의 조건부 schema는 아래와 같다.

```text
analysis_kind, split_role, tier, model, evaluation_ratio,
physical_ratio, config_id, score_variant, status, status_reason,
series, group_id, support_status
```

analysis_kind는 model_ratio와 tier_adaptive다. runnable 행은 현재 registry에 등록된
config와 유효한 physical_ratio를 갖는다. 불가능한 조합은 unavailable과 이유를 남긴다.
target-free는 같은 물리 점수를 여러 q에서 참조하되 q별 설정 선택은 독립이다.
모델별 선택과 Tier 공동 선택에서 필요한 series/config/q/seed의 합집합을 실행한다.

runner는 membership과 registry·config·입력 신원 및 조건 집단을 대조한다.
선택식을 다시 계산하거나 membership 밖의 final 실행을 추가하지 않는다.

## 최종 채점 원표

`ghl25_score_ledger.csv`와 `hai_score_ledger.csv`는 final score manifest를 참조하고 VUS-PR,
evaluator·`ℓ_max` 신원, `analysis_kind`, 평가 비율을 연결한다. 같은 물리 score가 `model_fixed`와
`tier_fixed`와 `tier_adaptive`에 함께 쓰이면 ledger 행은 분석별로 두되 배열은 하나만 보존한다.

GHL25 결과로 선택표나 membership을 바꾸지 않는다. HAI test1·test2도 서로 다른 split으로
채점한다. 두 HAI 결과를 한 숫자로 요약해야 할 때만 산술평균하며 행 수로 가중하지 않는다.
