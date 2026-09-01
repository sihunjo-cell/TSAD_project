# C(주혜) — 난이도·통계·교차점

주혜님은 모델과 recipe가 봉인된 뒤 GHL·HAI 결과를 해석합니다. 모델 담당자에게 score metadata와
실행비 원자료를 받고, 지우님에게 최종 채점 원표와 고정 정책표를 받습니다. 모델·recipe·VUS-PR·
`ℓ_max`를 다시 고르지 않습니다.

## 실험 전에 고정할 것

난이도 규칙, hit 판정, 비교 family, 다중검정 보정, TOST의 등가성 경계와 bootstrap 단위를 GHL
결과를 보기 전에 정합니다. 난이도 threshold는 정상 학습 통계로 정하고 테스트 성능으로 조정하지
않습니다. 테스트 라벨은 이상 구간의 경계와 hit 판정에만 씁니다.

TOST는 `±δ` 단측 검정 두 번으로 구현합니다. bootstrap은 10,000회입니다. 지속 교차 판정기는
`D=(+,−,+,+,+,+,+)`에서 `c*=20%`를 반환해야 합니다. 전 비율이 양수면 5%를 관측 하한으로,
교차가 없으면 `no_crossover`로 기록합니다.

별도 GDN 그래프 edge·Jaccard 분석은 하지 않습니다. GDN은 Tier 2의 다른 모델과 같은 성능·비용
입력으로만 다룹니다.

## 받는 입력

- `model_fixed_policy.csv`: 모든 활성 모델의 고정 recipe 곡선
- `ratio_adaptive_selection.csv`: 비율마다 고른 Tier 대표 모델의 주분석 경로
- `tier_fixed_policy.csv`: Tier 대표 모델을 전 비율에 고정한 통제 비교
- `final_policy_membership.csv`: `model_fixed·tier_fixed·tier_adaptive` 실행 요청
- `ghl25_score_ledger.csv`, `hai_score_ledger.csv`: 지우님의 최종 채점 원표
- 모델별 실행 시간, peak memory, artifact 크기와 관측량 근거
- 강혁님의 이상 구간·채널·주기성 EDA 요약

`tier_adaptive`는 모델별 recipe를 고정한 채 비율마다 Tier 대표를 바꾸는 운영 주분석입니다.
`tier_fixed`는 대표 모델까지 고정해 데이터 양의 효과를 분리하는 통제 비교입니다. 비율별로 recipe를
다시 고르거나 재튜닝하지 않습니다. TSB 튜닝 점수는 최종 성능이나 교차점 계산에 넣지 않습니다.

## GHL 분석

주분석은 같은 `q`와 같은 GHL 시계열에서 그 비율에 봉인된 Tier 대표와 고정 recipe를 짝지어
비교합니다. GHL 시계열 macro는 `1/25`입니다. 일곱 `q`를 독립 표본으로 세지 않고 한 bootstrap
replicate에서 같은 시계열을 모델·Tier·`q` 전반에 함께 적용합니다.

같은 simulator에서 나온 시계열의 의존성이 확인되면 provenance cluster 단위 bootstrap을
주결과로 올립니다. GHL25를 제조 공정 전체의 독립 표본이라고 일반화하지 않습니다.

교차점은 두 정책의 지원 비율 교집합에서만 판정합니다. 한 지점의 일시적 역전과 뒤의 모든
관측점에서 유지되는 지속적 역전을 구분합니다. 관측 비율 사이를 보간하지 않고 구간으로
보고합니다. `unavailable`은 성능 패배와 분리합니다.

난이도별 결과는 `구간 × 모델 × q` hit 원표와 회수율로 만듭니다. 모델 score를 본 뒤 난이도
threshold, window나 집계 순서를 바꾸지 않습니다.

## HAI 분석

`train1 → test1`과 `train1+train2 → test2` 결과를 먼저 따로 보고합니다. 둘 중 하나가
`unavailable`이면 이유를 그대로 남깁니다. 두 결과가 모두 있을 때 한 요약값이 필요하면 같은
가중치로 산술평균하며 행 수와 학습 데이터량을 가중치로 쓰지 않습니다.

HAI는 외부 확인이므로 HAI 성능으로 모델·recipe·threshold를 다시 고르지 않습니다. GHL과 HAI를
하나의 표본으로 합쳐 Wilcoxon, TOST나 bootstrap을 계산하지 않습니다.

## 비용 최적화로 넘길 원표

각 평가 split의 `(tier, model, config_id, q)`에 아래 값을 연결합니다. 최종 행동에서는 split을
선택 변수로 쓰지 않고 현장 시나리오의 근거로만 사용합니다.

- VUS-PR과 threshold별 오탐·미탐 수
- 난이도별 hit와 최소 성능 제약에 쓸 지표
- 관측량과 검증 가능한 관측 지속시간
- 현장 재학습·validation·test 추론 시간
- peak memory와 artifact 크기
- 실패·timeout·unavailable 상태

TSB HPO 비용은 개발비로 따로 합산하고 현장 반복비와 섞지 않습니다. 비용 가중치, 경보 단가와
latency·memory 제약은 마지막 의사결정 단계에서 받습니다. 주혜님은 원표와 통계 불확실성을
제공하며 최종 단가를 임의로 정하지 않습니다.

## 완료 신호

난이도·hit 판정, GHL paired 통계와 지속 교차점, HAI 두 실행의 외부 확인, 비용 목적함수용 성능·
불확실성 원표가 준비되면 완료입니다. TSB 선택표를 다시 만들거나 모델 runner를 수정하지 않습니다.
