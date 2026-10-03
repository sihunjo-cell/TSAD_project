# DP 운영 경로

ML이 만든 구간별 성능·비용표를 받아, 남은 예산 안에서 운영 기간 전체의 예측 VUS-PR이 가장 높은 경로를 고른다.
경로의 성능은 구간마다 쓰는 모델의 예측 VUS-PR을 구간 길이로 가중 평균한 값이다.

화면의 `운영 경로`는 첫 화면(기업 의사결정 창)에서 제출한 ML 추정표(`st.session_state["dp_input"]`)를 읽는다.
없으면 같은 형식의 JSON을 올려도 된다. 이 모듈은 추천만 하며 모델 학습이나 DB 기록은 하지 않는다.

## 방법

상태는 `(구간, 후보, 마지막 학습 구간)`이다. 같은 후보도 학습한 구간이 다르면 유지할 때의 성능이 달라 다른 상태로 둔다.
예산이 있으면 상태마다 가장 싼 경로 하나만 남기면 틀린다. 싸고 낮은 경로와 비싸고 높은 경로 중 무엇이 나은지는 남은 예산에 달렸다.
그래서 상태마다 `(누적 비용, 누적 가중 성능)` 라벨을 두고, 더 싼 라벨보다 성능이 높지 않은 라벨은 버린다.

1. 직전 구간 라벨마다 그 checkpoint를 유지하는 행동을 잇는다.
2. 새 학습은 직전 구간 전체에서 밀리지 않는 라벨들에서만 잇는다. 새 checkpoint의 비용·성능은 직전 경로와 무관하다.
3. 추론 행이 없는 행동과, 하한을 넣었다면 하한에 못 미치는 행동을 뺀다.
4. 마지막 구간에서 예산 안 최고 성능을 찾고, 그와 ε 이내인 라벨 중 가장 싼 라벨을 고른다. 비용도 같으면 학습 횟수가 적은 쪽이다.

구간 VUS-PR을 그냥 더하면 짧은 구간과 긴 구간이 같은 무게를 갖는다. 그래서 `구간 길이 / 전체 기간`을 곱해 더한다.

라벨은 경로를 통째로 들고 다니지 않고 이전 라벨만 가리킨다. 경로 표는 마지막에 고른 라벨에서 한 번만 만든다.
라벨 하나는 약 0.15KB다. 라벨이 300만 개(약 0.5GB)를 넘으면 답을 바꾸지 않고 `too_large`로 멈춘다.
이 상한은 실행하는 기계와 관계없이 같다.

## 입력 계약

`optimize_plan(payload)`가 받는 JSON 객체다. 비용과 예산은 모두 `cost_unit` 단위다.

| 필드 | 내용 |
| --- | --- |
| `metric` | `VUS-PR`만 받는다 |
| `cost_unit` | `KRW`, `seconds` 등 |
| `budget` | 남은 예산. `null`이면 제한 없음 |
| `tolerance` | 선택. 성능 차이가 이 값 이내면 싼 경로를 고른다. 기본 0 (Dev18 백테스트에서 0보다 크면 실제 성능이 낮아졌다) |
| `candidates` | `candidate_id`, `model`, `config_id`, `head` |
| `stages` | `stage`(0부터 연속), `weight`(구간 길이), 선택적으로 `minimum_performance` |
| `training_options` | `stage`, `candidate_id`, `predicted_performance`, `training_cost` |
| `inference_options` | `stage`, `candidate_id`, `trained_stage`, `inference_cost`. `trained_stage: -1`은 기존 checkpoint |
| `current_checkpoint` | 없으면 `null`. 있으면 `candidate_id`, `checkpoint_id`, `predicted_performance` |

`stages`에 `ratio_percent`를 넣으면 경로 표의 `데이터 비율 (%)`로 보여 준다. 경로는 "데이터가 몇 %일 때 무엇을 하라"로 읽는다.
유지 성능은 마지막 학습 때의 예측으로 고정한다. 학습 없는 모델의 도입비는 0이다. `null`, NaN, 무한대, 음수 비용은 오류로 본다.

## 결과

| `status` | 뜻 |
| --- | --- |
| `ok` | 예산 안 경로를 찾았다 |
| `over_budget` | 끝까지 가는 경로가 모두 예산을 넘는다. 가장 싼 경로를 돌려준다 |
| `infeasible` | `blocked_stage` 구간을 이어 갈 후보나 추론 행이 없다 |
| `too_large` | 라벨이 상한을 넘어 `blocked_stage` 구간에서 계산을 멈췄다 |

`ok`와 `over_budget`은 `total_cost`, `timeline_performance`, `path`, `current_action`, `frontier`를 함께 돌려준다.
`frontier`는 비용이 늘 때 성능이 오르는 경로를 최대 6개 보여 준다.

예측 VUS-PR의 절대값은 새 현장에서 ±0.3 정도 틀린다. 결과는 가장 싼 경로와의 차이로 읽는다.
화면과 보고서는 예산이 가장 싼 경로 비용의 100배 이하이면 가장 싼 경로를 추천하고, DP 경로는 참고로 둔다(`recommend.py`).
Dev18 백테스트에서 이 범위의 예산은 어떤 방식도 가장 싼 계획을 통계적으로 넘지 못했다.
현재 행동만 적용하고, 데이터가 다음 비율에 닿으면 새 누적 prefix와 남은 예산으로 다시 계산한다.

## 검증

```bash
python -m unittest streamlit_website.DP.test_planner
```

손계산 예제, 예산과 ε, 구간 길이 가중치, 비싼 직전 경로 뒤의 새 학습, checkpoint 유지·재학습, 하한, head 교체,
실행 불가 구간, 잘못된 입력을 검사하고 작은 문제에서는 모든 경로를 열거한 정답과 비교한다.
