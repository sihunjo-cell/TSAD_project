# Dev18 비율별 Tier 대표 선택 설계

이 문서는 폐기한 과거 점수와 고정 예산을 사용하던 당시 설계 기록이다. 현재 실행에는
적용하지 않으며 후보·q별 선택·새 예산 계약은 [계획서 v5](plan_v5.md)를 따른다.

## 목적

완료된 Dev18 점수 원표를 다시 채점하지 않고, 정상 데이터 비율마다 각 Tier의 대표 모델을 고른다. 최종 그림은 선택 정책의 성능과 모델 전환을 한 장에서 읽을 수 있어야 한다. 현장 비용값은 웹사이트가 나중에 받으며, 현재 코드나 산출물에 임의 숫자를 넣지 않는다.

## 입력과 재사용 경계

- 입력은 완료된 `dev18_trial_score_ledger.csv` 1,602행과 기존 feasibility·budget·registry 봉인본이다.
- 모델 실행, HPO, VUS-PR 계산과 원본 CSV 재처리는 하지 않는다.
- 기존 `dev18_budget_manifest.json`과 score manifest는 과거 실행 근거이므로 수정하지 않는다.
- 새 선택 산출물은 입력 ledger SHA-256, budget ID와 별도 정책 규칙 ID로 봉인한다.

## 비율별 대표 선택

- 최종 정책은 기존 `model_fixed`가 고른 모델별 config와 score variant를 그대로 쓴다. 비율마다 배포 config를 다시 고르지 않는다.
- 각 `(tier, ratio)`에서 해당 비율 점수가 있고 Dev18 18개 시계열 전체에 적용 가능한 모델만 후보로 둔다.
- 후보마다 10개 family를 한 번씩 제외한다. 각 fold에서는 남은 9개 family와 그 모델의 전체 지원 비율로 config를 한 번만 고정하고, 제외한 family의 비율별 VUS-PR을 기록한다. 비율마다 fold config를 다시 고르지 않는다.
- 모델별 10개 holdout 점수 평균을 해당 비율의 선택 점수로 쓴다. 대표 모델을 고른 뒤에는 18개 전체에서 봉인한 `model_fixed` config를 연결한다.
- 기존 tolerance와 결정적 동률 규칙을 그대로 적용해 `tier_adaptive` 대표 모델 하나를 고른다.
- `tier_adaptive`를 주분석으로 사용한다. `model_fixed`와 기존 `tier_fixed`는 비교·감사용으로 보존한다.
- PCA_LEGACY는 대표 후보에서 빼고 별도 기준선으로만 쓴다. 성능 하한이나 비용 gate도 적용하지 않는다.

완료된 ledger로 확인한 대표 경로는 아래와 같다.

| Tier | 비율별 대표 |
| --- | --- |
| Tier 1 | 모든 비율 `MWVAR` |
| Tier 2 | 5% `unavailable`, 10·20% `GDN`, 40·60·80·100% `PaAno` |
| Tier 3 | 모든 비율 `TSPulse` |

Tier 2의 5%는 현재 봉인본에 실행 가능한 모델이 없으므로 다른 비율의 점수를 복사하지 않는다.

## 후보와 배제 감사

`tier_ratio_candidate_audit.csv`는 모든 Tier·비율·모델 조합을 한 행씩 기록한다. 각 행에는 후보 여부, 지원 상태, config, score variant, family-LOFO VUS-PR, 선택 여부와 구체적인 제외 사유를 넣는다.

- PCA_LEGACY는 100% 점수를 기준선으로 보존하되 대표 후보에는 넣지 않고 `reference_only`로 기록한다.
- PaAno는 40% 이상에서 후보로 인정한다.
- GDN은 10% 이상에서 후보로 인정한다.
- ALoRa는 일부 model·config·series 조합이 실행 가능해도 어느 비율에서도 하나의 config가 18개 전체를 덮지 못하므로 제외한다. `pair_count < heads 8`을 포함한 차단 사유와 시계열을 감사표에 남긴다.

## 최종 그림

최종 `selection.png`는 한 축만 쓴다.

- x축은 정상 데이터 비율 `5, 10, 20, 40, 60, 80, 100%`다.
- y축은 선택에 사용한 `Family-LOFO VUS-PR`이다.
- Tier 1·2·3은 각각 하나의 선으로 잇고 모든 사용 가능한 비율에 marker를 찍는다.
- 각 marker 옆에는 그 비율에서 선택된 모델명만 작은 글씨로 표시한다.
- PCA_LEGACY의 100% VUS-PR `0.283669`는 ledger에서 읽어 전 구간 수평 점선으로 표시하고 범례에 `q100 reference only`라고 쓴다. 이 선은 gate가 아니다.
- Tier 2의 5%는 선을 만들 값이 없으므로 회색 `unavailable` 표지만 둔다.
- 후보별 세부 점수와 제외 근거는 CSV에 남기고 최종 그림 안에 긴 설명을 넣지 않는다.

Windows에서는 Malgun Gothic을 우선 등록하고 `matplotlib.rcParams["font.family"]`에 적용한다. Lightning에서는 환경 변수로 지정한 글꼴 파일, 설치된 Malgun Gothic, 설치된 NanumGothic 순서로 찾는다. 최종 그림의 축·범례·모델명은 ASCII로 써서 한글 글꼴이 없어도 문자가 깨지지 않게 한다.

## 현장 비용 연결 계약

`tier_policy_transitions.csv`는 각 비율의 이전·현재 모델과 config, `initial·keep·switch·unavailable`, 안정적인 `transition_key`를 기록한다. 별도 JSON Schema는 웹사이트가 받을 관측·학습·추론·저장·오경보·미탐·모델 전환·검증·배포 비용 필드를 정의한다.

- 비용값과 통화에는 기본 숫자를 두지 않는다.
- 값이 없는 비용을 0으로 해석하지 않는다.
- 현재 선택·그림 재생성은 비용 입력 없이 실행한다.
- 비용 목적함수는 웹사이트가 유효한 현장 scenario를 제출한 뒤 별도 단계에서 계산한다.

## 재생성과 오류 처리

Lightning 완료 경로에 selection-only 옵션을 추가한다. 이 옵션은 ledger의 schema, 1,602개 complete 행, 중복 key, budget ID와 evaluator·`ell_max` 봉인 신원을 검증한 뒤 선택표와 그림만 다시 만든다. ledger가 불완전하면 기존 산출물을 덮기 전에 중단한다.

## 검증

- 작은 합성 원표로 비율별 후보 진입, 대표 전환, Tier 2 5% unavailable과 결정적 동률 처리를 단위 검증한다.
- 최종 plot에 Tier 선 세 개, PCA 수평 점선, 비율별 모델 annotation과 축 의미가 전달되는지 figure 객체 수준에서 검증한다. 로컬에서는 PNG를 렌더링하지 않는다.
- membership loader가 `tier_adaptive`의 완전한 Tier·비율·split grid와 unavailable 행을 받아들이는지 확인한다.
- selection-only 경로가 VUS 평가 함수를 호출하지 않고 기존 ledger를 읽는지 검증한다.
- 관련 단위 테스트, Python compile, `git diff --check`만 로컬에서 실행한다. 모델·checkpoint·실데이터·전체 그림 렌더링은 Lightning에서 수행한다.
