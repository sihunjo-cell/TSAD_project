# 점수·선택 인터페이스

모든 모델은 지우님의 채점기와 주혜님의 통계 코드에 같은 형식으로 threshold 전 연속 score를
넘긴다. 문서와 배열이 다르면 임의로 padding하거나 자르지 않고 실행을 중단한다.

## score shape와 시점

본 채점 배열은 원시 시점에 정렬된 1차원 `(score_length,)`다. 채널 score를 만들 수 있는 모델은
`(score_length, channel_count)` 보조 배열도 저장한다. 채점기는 모델 이름으로 offset을 추측하지
않고 metadata의 `source_start`, `source_end_exclusive`, `label_slice`, `alignment`를 읽는다.

| 모델 | 본 score 계약 |
| --- | --- |
| MWVAR | 입력과 같은 길이 `L`, 중앙 window 96의 채널 score를 집계 |
| SQDIFF_LAST3 | 입력과 같은 길이 `L`, 첫 3점을 공식 경계 규칙으로 채움 |
| PCA_LEGACY | 입력과 같은 길이 `L`, fit-only PCA의 재구성 score |
| PaAno | test 세션 길이 `L`, 세션 경계를 넘기지 않은 point score |
| ALoRa | 공식 stitching으로 test 길이 `L` 복원 |
| GDN | 1-step forecast error `L-W`, `source_start=W`, `labels[W:]` |
| TimeRCD | scalar `(L,)` |
| TSPulse | 공식 time·FFT·prediction raw score를 길이 `L`로 정렬 |

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

주점수는 `raw__trainnorm`이다. `smoothed`와 `testnorm`은 민감도다. raw와 smoothed는 같은
실행에서 함께 저장한다.

학습형 모델은 현재 `q` prefix 안에서 `fit=floor(0.8×available)`과 validation을 시간순으로
나눈다. 입력 scaler는 fit에만 맞춘다. score의 median·IQR은 같은 prefix의 validation raw
score로만 추정한다. calibration에 쓴 score 배열과 SHA-256을 metadata에 연결한다.

채널 score가 있으면 정규화 → smoothing → 채널 `max` 순서로 집계한다. smoothing은 후행 4칸
평균이고 처음 3점은 0이다. 공식 scalar score에는 가짜 채널 축이나 추가 `max`를 만들지 않는다.
어떤 모델도 test 전체에 scaler·score MinMax를 맞추지 않는다.

training-free Tier 1과 strict zero-shot은 target calibration을 쓰지 않는다. 동일한 물리 score를
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

Dev18의 `measurement_protocol_id`는 `dev18_registered_runner.v2`다. `timing`은 in-memory 등록 executor 안에서
`split_preprocess_seconds`, `model_setup_seconds`, 학습, validation 추론, test 추론을 나눠 기록한다.
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

공식20에서 GHL 09·18을 뺀 18개만 이 원표에 들어간다. seed 평균 뒤 10개 family를 동일
가중한다.

```text
J(model,config,q)
  = (1/10) × sum_family mean_{series in family}(mean_seed VUSPR)
```

`1/18` series macro는 민감도 열이다. 모델 선택은 봉인된 `equal_trial` panel에서 family
leave-one-out으로 수행한다. 선택 모델의 recipe는 같은 panel의 18개 전체에서 고정한다.

## 필수 선택표

`model_fixed_policy.csv`는 모든 활성 모델을 보존한다.

```text
tier, model, q_support, config_id, hyperparameters,
j_fixed, score_variant, hpo_regime, budget_id,
source_commit, source_checkpoint_sha256, selection_status, selection_reason
```

`tier_fixed_policy.csv`는 계층별 주분석을 고정한다.

```text
tier, selected_model, config_id, hyperparameters,
q_floor, selection_q_common, evaluation_q_support,
score_variant, selection_score, j_tier,
hpo_regime, budget_id, evaluator_sha256,
source_commit, source_checkpoint_sha256, selection_status, selection_reason
```

`selection_score`는 family leave-one-out 모델 점수, `j_tier`는 선택 모델의 full tuning-panel 고정
recipe 점수다. `evaluation_q_support`는 `ghl25_final`, `train1_to_test1`,
`train1_train2_to_test2`를 구분한다.

비율별 model-adaptive, tier-adaptive와 native-feasible 표는 선택적 민감도다. 이 표들을 필수
bundle에 넣거나 주실험 runner의 시작 조건으로 삼지 않는다. Dev18 exact panel의 자원·동등성 gate가
막혔을 때도 adaptive 표나 수동 모델 실행으로 대체하지 않는다.

## 최종 실행 요청

`final_policy_membership.csv`의 최소 schema는 아래와 같다.

```text
analysis_kind, split_role, tier, model, evaluation_ratio,
physical_ratio, config_id, score_variant, status, status_reason
```

필수 `analysis_kind`는 `model_fixed`와 `tier_fixed`다. runnable 행은 유효한 `physical_ratio`와
빈 `status_reason`을 갖는다. 실행 불가 행은 `status=unavailable`, 빈 `physical_ratio`와 구체적인
이유를 남긴다. target-free 모델의 모든 평가 비율은 같은 `physical_ratio=100` score를 참조한다.

모델 runner는 membership, registry, config와 입력 SHA-256만 확인한다. 튜닝 원표나 선택식을
읽어 다시 검증하지 않는다. 모델 담당자는 membership 밖의 final 실행을 추가하지 않는다.

## 최종 채점 원표

`ghl25_score_ledger.csv`와 `hai_score_ledger.csv`는 final score manifest를 참조하고 VUS-PR,
evaluator·`ℓ_max` 신원, `analysis_kind`, 평가 비율을 연결한다. 같은 물리 score가 `model_fixed`와
`tier_fixed`에 모두 쓰이면 ledger 행은 둘로 두되 배열은 하나만 보존한다.

GHL25 결과로 선택표나 membership을 바꾸지 않는다. HAI test1·test2도 서로 다른 split으로
채점한다. 두 HAI 결과를 한 숫자로 요약해야 할 때만 산술평균하며 행 수로 가중하지 않는다.
