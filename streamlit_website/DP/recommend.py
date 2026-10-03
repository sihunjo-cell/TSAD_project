"""빠듯한 예산에서는 가장 싼 경로를 추천으로 올리고, DP 경로는 참고로 둔다."""

from streamlit_website.DP.planner import optimize_plan

# Dev18 백테스트에서 가장 싼 경로 비용의 100배 이하 예산은 어떤 방식도 가장 싼 계획을 통계적으로 넘지 못했다.
TIGHT_BUDGET_RATIO = 100


def cheapest_plan(payload, result):
    """가장 싼 frontier 비용을 예산으로 두고 다시 푼 경로."""
    if not result.get("frontier"):
        return None
    plan = optimize_plan({**payload, "budget": result["frontier"][0]["total_cost"]})
    return plan if plan["status"] == "ok" else None


def tight_ratio(budget, cheapest):
    """예산이 가장 싼 경로 비용의 몇 배인지. 빠듯하지 않으면 None."""
    if not budget or not cheapest or cheapest["total_cost"] <= 0:
        return None
    ratio = budget / cheapest["total_cost"]
    return ratio if ratio <= TIGHT_BUDGET_RATIO else None


def steps(plan):
    return [(step["candidate_id"], step["trained_stage"]) for step in plan["path"]]


def recommend(payload, result):
    """(추천 결과, 비교 경로, 빠듯한 예산 배율).

    빠듯하고 DP 경로가 가장 싼 경로와 다르면 가장 싼 경로를 추천하고 DP 결과를 비교 경로로 돌려준다.
    그 밖에는 DP 결과를 그대로 추천하고 가장 싼 경로를 비교 경로로 돌려준다.
    """
    cheapest = cheapest_plan(payload, result)
    ratio = tight_ratio(result.get("budget"), cheapest)
    same = cheapest and steps(cheapest) == steps(result)
    if ratio is None or same or result["status"] != "ok":
        return result, cheapest, ratio
    keys = ("path", "current_action", "total_cost", "timeline_performance", "trainings")
    frontier = [{**point, "chosen": index == 0, "reference": point["chosen"]}
                for index, point in enumerate(result["frontier"])]
    return {**result, **{key: cheapest[key] for key in keys}, "frontier": frontier}, result, ratio
