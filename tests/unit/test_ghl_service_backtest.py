"""GHL 서비스 점검의 구간 실행·비용 누적·사후 최적을 손계산과 대조한다."""

import unittest

from streamlit_website.DP.planner import optimize_plan
from tests.ghl_main.evaluate_ghl_service import simulate, substitute_predictions
from tests.ghl_main.run_ghl_candidates import SERVICE_RATIOS


STARTS = [0.0, 5.0, 15.0, 35.0, 55.0, 75.0]
SEGMENTS = list(zip(SERVICE_RATIOS, STARTS, STARTS[1:] + [100.0]))
# a는 모든 비율에서, b는 20%부터 실측 VUS-PR이 있다.
ACTUAL = {**{("a", ratio): 0.4 for ratio in SERVICE_RATIOS},
          **{("b", ratio): 0.8 for ratio in SERVICE_RATIOS if ratio >= 20}}
TRAINING_COST = {"a": 10.0, "b": 100.0}
INFERENCE_COST_PER_DAY = {"a": 1.0, "b": 2.0}


def fake_plan_input(ratio, current):
    start, end = next((start, end) for q, start, end in SEGMENTS if q == ratio)
    stages = [{"ratio_percent": ratio, "elapsed_days": 0.0}]
    return {"stages": stages + ([{"elapsed_days": end - start}] if ratio != SERVICE_RATIOS[-1] else [])}


def fake_choose(payload):
    ratio = payload["stages"][0]["ratio_percent"]
    action, candidate = {5: ("adopt", "a"), 20: ("switch", "b")}.get(ratio, ("keep", "a" if ratio < 20 else "b"))
    training_cost = TRAINING_COST[candidate] if action != "keep" else 0.0
    return {"status": "ok", "path": [{"action": action, "candidate_id": candidate,
                                      "training_cost": training_cost, "inference_cost": 1.0}]}


class TestGhlServiceBacktest(unittest.TestCase):
    def test_simulation_follows_first_stage_with_measured_performance(self):
        result = simulate(fake_plan_input, fake_choose, None, ACTUAL, segments=SEGMENTS)
        # a를 15일, b를 85일 쓴다. 비용은 경로의 학습비 10·100과 구간마다 추론비 1이다.
        self.assertAlmostEqual(result["vus_pr"], 0.4 * 0.15 + 0.8 * 0.85)
        self.assertAlmostEqual(result["spent_seconds"], 116.0)
        self.assertEqual((result["trainings"], result["missing_actual"]), (2, 0))

    def test_oracle_replaces_predictions_with_measured_values(self):
        stages = [{"stage": index, "weight": end - start, "ratio_percent": ratio}
                  for index, (ratio, start, end) in enumerate(SEGMENTS)]
        training = [{"stage": stage, "candidate_id": name, "predicted_performance": 0.5,
                     "training_cost": TRAINING_COST[name]} for stage in range(6) for name in ("a", "b")]
        inference = [{"stage": later, "candidate_id": option["candidate_id"], "trained_stage": option["stage"],
                      "inference_cost": INFERENCE_COST_PER_DAY[option["candidate_id"]] * stages[later]["weight"]}
                     for option in training for later in range(option["stage"], 6)]
        base = {"metric": "VUS-PR", "cost_unit": "seconds", "budget": None, "stages": stages,
                "candidates": [{"candidate_id": name, "model": name, "config_id": name, "head": ""}
                               for name in ("a", "b")],
                "training_options": training, "inference_options": inference, "current_checkpoint": None}
        # a 25초(15일) 뒤 20%에서 b로 바꾼다: 10 + 15 + 100 + 170.
        measured = substitute_predictions(base, ACTUAL)
        unlimited = optimize_plan(measured)
        self.assertAlmostEqual(unlimited["timeline_performance"], 0.74)
        self.assertAlmostEqual(unlimited["total_cost"], 295.0)
        # b로 바꾸는 가장 싼 경로(80%에서 교체)도 235초라 200초 예산에서는 a만 남는다.
        limited = optimize_plan({**measured, "budget": 200.0})
        self.assertAlmostEqual(limited["timeline_performance"], 0.4)
        self.assertAlmostEqual(limited["total_cost"], 110.0)


    def test_substitution_replaces_or_drops_kept_checkpoint(self):
        payload = {"stages": [{"stage": 0, "ratio_percent": 20}],
                   "training_options": [{"stage": 0, "candidate_id": "b", "predicted_performance": 0.5}],
                   "inference_options": [{"stage": 0, "candidate_id": "a", "trained_stage": -1},
                                         {"stage": 0, "candidate_id": "b", "trained_stage": 0}],
                   "current_checkpoint": {"candidate_id": "a", "predicted_performance": 0.5}}
        replaced = substitute_predictions(payload, ACTUAL, ("a", 10))
        self.assertEqual(replaced["current_checkpoint"]["predicted_performance"], 0.4)
        self.assertEqual(replaced["training_options"][0]["predicted_performance"], 0.8)
        self.assertEqual(len(replaced["inference_options"]), 2)
        # b를 5%에서 학습한 checkpoint는 실측이 없어 유지 선택지가 빠진다.
        dropped = substitute_predictions({**payload, "current_checkpoint": {"candidate_id": "b"}}, ACTUAL, ("b", 5))
        self.assertIsNone(dropped["current_checkpoint"])
        self.assertEqual([row["trained_stage"] for row in dropped["inference_options"]], [0])


if __name__ == "__main__":
    unittest.main()
