"""예산 안 운영 경로 DP의 계약을 작은 합성 입력으로 확인한다."""

import copy
import itertools
import math
import unittest
from unittest.mock import patch

from streamlit_website.DP import planner
from streamlit_website.DP.planner import optimize_plan
from streamlit_website.DP.recommend import recommend


def plan_input(weights=(1.0,), identities=("a",), budget=None):
    return {
        "metric": "VUS-PR", "cost_unit": "synthetic_units", "budget": budget,
        "candidates": [{"candidate_id": name, "model": "GDN",
                        "config_id": f"config-{name}", "head": ""}
                       for name in identities],
        "stages": [{"stage": stage, "weight": weight} for stage, weight in enumerate(weights)],
        "training_options": [], "inference_options": [], "current_checkpoint": None,
    }


def train(stage, candidate, performance, cost):
    return {"stage": stage, "candidate_id": candidate,
            "predicted_performance": performance, "training_cost": cost}


def infer(stage, candidate, origin, cost):
    return {"stage": stage, "candidate_id": candidate,
            "trained_stage": origin, "inference_cost": cost}


def build_example(*, with_checkpoint=False):
    """손으로 계산한 4구간 예제. MWVAR는 처음부터, PaAno는 구간 3부터 학습할 수 있다."""
    request = plan_input((0.1, 0.2, 0.3, 0.4), ("mwvar", "gdn", "paano"), budget=13.0)
    request["tolerance"] = 0.0
    options = [train(0, "mwvar", 0.28, 0.0)]
    for stage in range(4):
        options.append(train(stage, "gdn", 0.34 + stage * 0.01, 8.0 + stage * 2))
        if stage >= 2:
            options.append(train(stage, "paano", 0.38 + (stage - 2) * 0.01, 2.0 + (stage - 2) * 2))
    interval_cost = {"mwvar": 4.0, "gdn": 1.0, "paano": 0.5}
    request["training_options"] = options
    request["inference_options"] = [infer(stage, option["candidate_id"], option["stage"],
                                          interval_cost[option["candidate_id"]])
                                    for option in options for stage in range(option["stage"], 4)]
    if with_checkpoint:
        request["current_checkpoint"] = {"candidate_id": "gdn", "checkpoint_id": "existing-gdn",
                                         "predicted_performance": 0.34}
        request["training_options"] = [option for option in options if option["candidate_id"] != "gdn"]
        request["inference_options"] = [row for row in request["inference_options"] if row["candidate_id"] != "gdn"]
        request["inference_options"] += [infer(stage, "gdn", -1, 1.0) for stage in range(4)]
    return request


class TestPlanner(unittest.TestCase):
    def test_bundled_examples_match_hand_calculation(self):
        result = optimize_plan(build_example())
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["total_cost"], 13)
        self.assertAlmostEqual(result["timeline_performance"], 0.368)
        self.assertEqual([row["candidate_id"] for row in result["path"]], ["gdn", "gdn", "paano", "paano"])
        self.assertEqual([row["action"] for row in result["path"]], ["adopt", "keep", "switch", "keep"])
        self.assertTrue(any(point["chosen"] for point in result["frontier"]))
        result = optimize_plan(build_example(with_checkpoint=True))
        self.assertAlmostEqual(result["total_cost"], 9)
        self.assertAlmostEqual(result["timeline_performance"], 0.372)
        self.assertEqual([row["trained_stage"] for row in result["path"]], [-1, -1, 2, 3])
        self.assertEqual([row["action"] for row in result["path"]], ["keep", "keep", "switch", "retrain"])

    def test_budget_and_tolerance_trade_value_for_cost(self):
        request = build_example()
        request["budget"] = 12
        result = optimize_plan(request)
        self.assertAlmostEqual(result["total_cost"], 11)
        self.assertAlmostEqual(result["timeline_performance"], 0.35)
        self.assertEqual([row["candidate_id"] for row in result["path"]], ["mwvar", "mwvar", "paano", "paano"])
        request["budget"] = None
        result = optimize_plan(request)
        self.assertAlmostEqual(result["timeline_performance"], 0.374)
        self.assertAlmostEqual(result["total_cost"], 27)
        request["tolerance"] = 0.02
        self.assertAlmostEqual(optimize_plan(request)["total_cost"], 13)
        request["budget"] = 5
        result = optimize_plan(request)
        self.assertEqual(result["status"], "over_budget")
        self.assertAlmostEqual(result["total_cost"], 11)

    def test_duration_weights_decide_between_now_and_later(self):
        # 예산 5로는 a 유지(5)나 c→b(4)만 가능하고 a→b(7)는 안 된다.
        request = plan_input((0.9, 0.1), ("a", "b", "c"), budget=5)
        request["training_options"] = [train(0, "a", 0.6, 3), train(0, "c", 0.3, 0), train(1, "b", 0.9, 2)]
        request["inference_options"] = [infer(0, "a", 0, 1), infer(1, "a", 0, 1),
                                        infer(0, "c", 0, 1), infer(1, "b", 1, 1)]
        self.assertEqual([row["candidate_id"] for row in optimize_plan(request)["path"]], ["a", "a"])
        request["stages"][0]["weight"], request["stages"][1]["weight"] = 0.1, 0.9
        self.assertEqual([row["candidate_id"] for row in optimize_plan(request)["path"]], ["c", "b"])

    def test_fresh_training_can_follow_a_costlier_predecessor(self):
        request = plan_input((0.5, 0.5), ("x", "y", "z"), budget=10)
        request["training_options"] = [train(0, "x", 0.2, 0), train(0, "y", 0.9, 4), train(1, "z", 0.9, 0)]
        request["inference_options"] = [infer(0, "x", 0, 1), infer(0, "y", 0, 1), infer(1, "z", 1, 1)]
        result = optimize_plan(request)
        self.assertEqual([row["candidate_id"] for row in result["path"]], ["y", "z"])
        self.assertAlmostEqual(result["timeline_performance"], 0.9)

    def test_current_checkpoint_can_stay_outside_training_options(self):
        request = plan_input((0.5, 0.5))
        request["current_checkpoint"] = {
            "candidate_id": "a", "checkpoint_id": "actual.pt",
            "last_trained_at": "2026-09-22", "predicted_performance": 0.6}
        request["inference_options"] = [infer(stage, "a", -1, 1) for stage in range(2)]
        original = copy.deepcopy(request)
        result = optimize_plan(request)
        self.assertEqual(request, original)
        self.assertEqual(result["total_cost"], 2)
        self.assertEqual([row["action"] for row in result["path"]], ["keep", "keep"])
        self.assertEqual([row["checkpoint_id"] for row in result["path"]], ["actual.pt", "actual.pt"])

    def test_retrain_when_more_data_pays_and_budget_allows(self):
        request = plan_input((0.5, 0.5))
        request["current_checkpoint"] = {"candidate_id": "a", "checkpoint_id": "old.pt",
                                         "predicted_performance": 0.55}
        request["training_options"] = [train(1, "a", 0.9, 2)]
        request["inference_options"] = [infer(0, "a", -1, 0.5), infer(1, "a", -1, 0.1), infer(1, "a", 1, 0.5)]
        result = optimize_plan(request)
        self.assertEqual([row["action"] for row in result["path"]], ["keep", "retrain"])
        self.assertEqual([row["cumulative_cost"] for row in result["path"]], [0.5, 3.0])
        request["budget"] = 1
        self.assertEqual([row["action"] for row in optimize_plan(request)["path"]], ["keep", "keep"])

    def test_optional_floor_still_filters_states(self):
        request = plan_input()
        request["training_options"] = [train(0, "a", 0.5, 0)]
        request["inference_options"] = [infer(0, "a", 0, 0)]
        request["stages"][0]["minimum_performance"] = 0.5
        self.assertEqual(optimize_plan(request)["total_cost"], 0)
        request["stages"][0]["minimum_performance"] = 0.6
        self.assertEqual(optimize_plan(request)["status"], "infeasible")

    def test_heads_with_shared_config_are_distinct_candidates(self):
        request = plan_input((0.5, 0.5), ("time", "fft"))
        request["candidates"][0].update(model="TSPulse", config_id="shared", head="time")
        request["candidates"][1].update(model="TSPulse", config_id="shared", head="fft")
        request["training_options"] = [train(0, "time", 0.8, 1), train(1, "fft", 0.8, 1)]
        request["inference_options"] = [infer(0, "time", 0, 1), infer(1, "time", 0, 10), infer(1, "fft", 1, 1)]
        result = optimize_plan(request)
        self.assertEqual([row["action"] for row in result["path"]], ["adopt", "switch"])
        self.assertEqual([row["candidate_id"] for row in result["path"]], ["time", "fft"])

    def test_missing_inference_returns_empty_infeasible_plan(self):
        request = plan_input((0.5, 0.5))
        request["training_options"] = [train(0, "a", 0.8, 1)]
        request["inference_options"] = [infer(0, "a", 0, 1)]
        result = optimize_plan(request)
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["blocked_stage"], 1)
        self.assertEqual(result["path"], [])
        self.assertIsNone(result["current_action"])

    def test_tight_budget_recommends_cheapest_path(self):
        request = build_example()
        result = optimize_plan(request)
        plan, reference, ratio = recommend(request, result)
        self.assertAlmostEqual(ratio, 13 / 11)
        self.assertIs(reference, result)
        self.assertAlmostEqual(plan["total_cost"], 11)
        self.assertEqual([row["candidate_id"] for row in plan["path"]], ["mwvar", "mwvar", "paano", "paano"])
        self.assertEqual(plan["budget"], 13.0)
        self.assertTrue(plan["frontier"][0]["chosen"])
        self.assertEqual([point["reference"] for point in plan["frontier"]], [point["chosen"] for point in result["frontier"]])

    def test_roomy_budget_or_same_path_keeps_dp_result(self):
        for budget in (None, 2000, 11):
            with self.subTest(budget=budget):
                request = build_example()
                request["budget"] = budget
                result = optimize_plan(request)
                self.assertIs(recommend(request, result)[0], result)

    def test_label_limit_stops_without_changing_answer(self):
        request = build_example()
        expected = optimize_plan(request)
        self.assertGreater(planner.MAX_LABELS, sum(row["labels"] for row in expected["stage_summaries"]))
        with patch.object(planner, "MAX_LABELS", 5):
            result = optimize_plan(request)
        self.assertEqual(result["status"], "too_large")
        self.assertEqual(result["path"], [])
        self.assertIsNotNone(result["blocked_stage"])

    def test_horizon_end_cost_is_counted(self):
        request = plan_input((1, 1, 1))
        request["training_options"] = [train(0, "a", 0.8, 1)]
        request["inference_options"] = [infer(stage, "a", 0, stage + 1) for stage in range(3)]
        result = optimize_plan(request)
        self.assertEqual([row["stage"] for row in result["path"]], [0, 1, 2])
        self.assertEqual(result["total_cost"], 7)
        self.assertAlmostEqual(result["timeline_performance"], 0.8)

    def test_invalid_values_and_duplicate_or_orphan_options(self):
        valid = plan_input()
        valid["training_options"] = [train(0, "a", 0.8, 1)]
        valid["inference_options"] = [infer(0, "a", 0, 1)]
        changes = {
            "wrong metric": lambda item: item.update(metric="F1"),
            "empty unit": lambda item: item.update(cost_unit=""),
            "negative budget": lambda item: item.update(budget=-1),
            "tolerance above one": lambda item: item.update(tolerance=2),
            "negative weight": lambda item: item["stages"][0].update(weight=-1),
            "zero weights": lambda item: item["stages"][0].update(weight=0),
            "NaN score": lambda item: item["training_options"][0].update(predicted_performance=math.nan),
            "floor above one": lambda item: item["stages"][0].update(minimum_performance=1.1),
            "negative cost": lambda item: item["inference_options"][0].update(inference_cost=-1),
            "infinite cost": lambda item: item["training_options"][0].update(training_cost=math.inf),
            "overflow cost": lambda item: item["training_options"][0].update(training_cost=10 ** 1000),
            "wrong current id type": lambda item: item.update(current_checkpoint={"candidate_id": []}),
            "unknown id": lambda item: item["training_options"][0].update(candidate_id="missing"),
            "duplicate train": lambda item: item["training_options"].append(copy.deepcopy(item["training_options"][0])),
            "duplicate infer": lambda item: item["inference_options"].append(copy.deepcopy(item["inference_options"][0])),
            "future origin": lambda item: item["inference_options"][0].update(trained_stage=1),
            "orphan origin": lambda item: item["inference_options"][0].update(trained_stage=-1),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                request = copy.deepcopy(valid)
                change(request)
                with self.assertRaises(ValueError):
                    optimize_plan(request)

    def test_exhaustive_three_stage_oracle_under_budgets(self):
        weights = (0.2, 0.3, 0.5)
        scores = {"a": (0.6, 0.8, 0.95), "b": (0.7, 0.9, 0.95)}
        costs = {"a": (1, 4, 3), "b": (4, 2, 5)}
        request = plan_input(weights, ("a", "b"))
        for stage in range(3):
            for candidate in ("a", "b"):
                request["training_options"].append(train(stage, candidate, scores[candidate][stage], costs[candidate][stage]))
                request["inference_options"].extend(infer(stage, candidate, origin, 1 + stage + (candidate == "b"))
                                                    for origin in range(stage + 1))
        plans = []
        origins = [(candidate, stage) for candidate in ("a", "b") for stage in range(3)]
        for sequence in itertools.product(origins, repeat=3):
            cost = value = 0
            for stage, (candidate, origin) in enumerate(sequence):
                if (origin > stage or (stage == 0 and origin != 0) or
                        (stage > 0 and sequence[stage - 1] != (candidate, origin) and origin != stage)):
                    break
                cost += 1 + stage + (candidate == "b") + (costs[candidate][stage] if origin == stage else 0)
                value += weights[stage] * scores[candidate][origin]
            else:
                plans.append((cost, value))
        for budget in (8, 10, 13, None):
            with self.subTest(budget=budget):
                request["budget"] = budget
                affordable = [plan for plan in plans if budget is None or plan[0] <= budget]
                best = max(value for _, value in affordable)
                cheapest = min(cost for cost, value in affordable if math.isclose(value, best))
                result = optimize_plan(request)
                self.assertAlmostEqual(result["timeline_performance"], best)
                self.assertAlmostEqual(result["total_cost"], cheapest)
                self.assertEqual(result["path"][-1]["cumulative_cost"], result["total_cost"])


if __name__ == "__main__":
    unittest.main()
