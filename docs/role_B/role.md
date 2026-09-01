# B(지우) — VUS-PR·선택표

지우님은 모델 담당자가 만든 threshold 전 연속 점수를 같은 규칙으로 채점하고, TSB 튜닝 패널에서
모델과 recipe를 고정합니다. 모델 본체와 runner는 수정하지 않습니다.

## 입력

입력은 `dev18_score_manifest.csv`와 각 score의 metadata입니다. 이름에 남은 `dev18`은
TSB-AD-M 비-GHL 튜닝 패널 18개의 내부 식별자이며 별도 본실험 데이터셋이 아닙니다.

`config_id`, score SHA-256, source 범위, label slice와 실제 배열 길이가 맞지 않으면 채점을
멈춥니다. 모델별 offset을 채점기에서 추측하지 않고 metadata를 읽습니다. GDN은 test 길이 `L`,
window `W`일 때 `L-W` 점수와 `labels[W:]`를 받습니다.

## 평가기

주지표는 `raw__trainnorm` VUS-PR입니다. threshold 250개는 VUS 적분 격자이며 현장 경보
threshold가 아닙니다. AUPRC와 point adjustment 없는 F1은 보조 지표로 둘 수 있지만 주 선택
규칙을 바꾸지 않습니다. test-optimal threshold와 point adjustment는 본 결과에 쓰지 않습니다.

`ℓ_max`는 강혁님이 넘긴 학습 구간 주기성 근거를 받고 GHL·HAI 성능을 보기 전에 고정합니다.
이상 구간 길이는 선택값을 최적화하는 근거로 쓰지 않습니다. `ℓ/2`, `ℓ`, `2ℓ`은 본 규칙을
바꾸지 않는 민감도입니다.

평가기의 최소 검증은 parser, score-label 정렬, 상수·무작위 점수와 point adjustment 대조입니다.
검증한 evaluator와 `ℓ_max` 명세의 SHA-256을 모든 채점 원표에 연결합니다.

## TSB 튜닝과 선택

공식 `TSB-AD-M-Tuning.csv` 20개 중 GHL 09·18을 뺀 18개만 모델·recipe 선택에 씁니다. 먼저
시계열별 seed 평균을 내고 10개 family를 각각 `1/10`로 평균합니다. `1/18` 시계열 macro는
민감도로만 남깁니다.

모델 선택은 봉인된 `equal_trial` panel에서 family leave-one-out 바깥 검증으로 합니다. 각
holdout family마다 나머지 9개 family에서 recipe를 고른 뒤 holdout 점수를 모아 `S(m)`을
계산합니다. 모델을 고른 뒤 같은 panel의 18개 전체에서 고정 recipe 한 벌을 정합니다. 점수 차이가
`1e-6` 이내면 `(model, config_id, score_variant)` 사전순으로 고릅니다.

필수 출력은 아래 파일입니다.

1. `dev18_trial_score_ledger.csv`: 시계열·seed·model·config·`q`별 채점 원표
2. `model_fixed_policy.csv`: 활성 모델별 고정 recipe와 지원 `q`
3. `tier_fixed_policy.csv`: Tier 대표를 고정한 통제 비교
4. `ratio_adaptive_selection.csv`: 비율별 Tier 대표를 고른 운영 주분석
5. `tier_ratio_candidate_audit.csv`, `tier_policy_transitions.csv`: 후보 배제와 모델 전환 감사
6. `family_lofo.csv`: 10개 holdout family별 선택 config와 바깥 점수
7. `final_policy_membership.csv`: GHL25·HAI가 소비할 294행 단일 실행 요청

모든 파일과 검토용 그림은 `experiments/01_ghl_main/results/dev18_tuning/` 한 폴더에 둡니다.
모델별 파일은 `PaAno.csv`, `PaAno.png`처럼 짧게 이름을 붙입니다. 최종 `selection.png`에는
Tier 선 세 개, PCA_LEGACY q100 참고 점선과 작은 모델명만 둡니다. 후보 점수, 배제 사유와 전환
내역은 그림에 넣지 않고 위 상세 CSV에 남깁니다.

`tier_adaptive`는 모델별 recipe를 고정한 상태에서 비율마다 대표 모델을 고르는 운영 주분석입니다.
`tier_fixed`는 데이터 양의 효과를 분리하는 통제 비교입니다. adaptive 선택은 같은 완료 ledger를
재사용하며 HPO나 VUS-PR을 다시 실행하지 않습니다. PCA_LEGACY는 대표 후보나 성능 gate가 아니라
그림의 참고선입니다.

## GHL·HAI 실행 요청과 채점

선택표가 봉인되면 `final_policy_membership.csv`를 만들어 모델 담당자에게 넘깁니다. 이 파일에는
split, model, `config_id`, 평가 비율, 실제 점수 비율, score variant와 runnable 상태만 둡니다.
runner는 이 실행 요청만 소비합니다. 지우님의 선택식이나 VUS-PR을 runner 안에서 다시 계산하지
않습니다.

GHL25에서는 모든 활성 모델의 `model_fixed` 지원점을 빠뜨리지 않습니다. 그래야 마지막 비용
목적함수가 Tier 대표 외의 더 싼 후보도 비교할 수 있습니다. `tier_adaptive`는 운영 주분석이고
`tier_fixed`는 통제 비교입니다. adaptive membership은 `model_fixed`에 없는 물리 실행을 추가하지
않습니다. GHL 결과로 TSB 선택표나 membership을 다시 만들지 않습니다.

HAI는 `train1 → test1`, `train1+train2 → test2`를 별도 행으로 채점합니다. 모델·recipe는 TSB
튜닝에서 고정한 값을 그대로 씁니다. 두 결과를 먼저 따로 넘기고, 요약이 필요할 때만 같은
가중치로 산술평균합니다.

최종 출력은 `ghl25_score_ledger.csv`와 `hai_score_ledger.csv`입니다. 성능 해석, 통계와 교차점은
주혜님에게 넘깁니다. 현장 operating threshold는 비용 단가와 제약이 정해지는 마지막 단계에서
별도로 선택합니다.

## 완료 신호

evaluator·`ℓ_max` 신원, TSB 튜닝 원표, 세 정책표와 294행 final membership이 서로 맞고 GHL·HAI
채점 원표가 봉인되면 완료입니다. 모델 구현, runner 수정, 난이도·통계·현장 비용값 결정은 완료
범위에 넣지 않습니다.
