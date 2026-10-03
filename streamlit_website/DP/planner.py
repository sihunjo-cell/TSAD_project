"""예산 안에서 구간 길이로 가중한 예측 VUS-PR이 가장 높은 운영 경로를 고른다.

라벨은 (누적 비용, 누적 가중 VUS-PR, 학습 횟수, 이전 라벨, 후보, 학습 구간, 행동, 성능, 학습비, 추론비)다.
경로를 통째로 들고 다니지 않고 이전 라벨만 가리키며, 경로 표는 고른 라벨에서 한 번만 만든다.
"""

from collections import defaultdict
from math import isfinite

# 최고 경로와 이만큼 차이 나면 동률로 보고 싼 경로를 고른다.
# Dev18 백테스트에서 0.01~0.05로 두면 실제 VUS-PR이 0.07~0.10 낮아져 0으로 둔다.
DEFAULT_TOLERANCE = 0.0
FRONTIER_POINTS = 6
# 라벨이 이만큼 쌓이면 답을 바꾸지 않고 계산을 멈춘다. 라벨 하나가 약 0.15KB라 약 0.5GB다.
MAX_LABELS = 3_000_000
START_LABEL = (0.0, 0.0, 0, None, None, None, None, None, None, None)


class TooManyLabels(Exception):
    pass


def check_text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name}: 문자열이 필요합니다.")
    return value


def check_number(value, name, *, maximum=None):
    try:
        valid = type(value) in (int, float) and isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name}: 0 이상의 유한한 수가 필요합니다.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name}: {maximum} 이하여야 합니다.")
    return value


def read_records(payload, name):
    records = payload.get(name)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"{name}: 객체 목록이 필요합니다.")
    return records


def read_candidates(payload):
    candidates, identities = {}, set()
    for row in read_records(payload, "candidates"):
        candidate_id = check_text(row.get("candidate_id"), "candidate_id")
        identity = (check_text(row.get("model"), "model"), check_text(row.get("config_id"), "config_id"),
                    check_text(row.get("head"), "head", empty=True))
        if identity[0].casefold() == "alora":
            raise ValueError("ALoRa는 운영 후보에서 제외합니다.")
        if candidate_id in candidates or identity in identities:
            raise ValueError("후보 ID 또는 모델·설정·head가 중복됐습니다.")
        candidates[candidate_id] = row
        identities.add(identity)
    return candidates


def read_stages(payload):
    stages = read_records(payload, "stages")
    if not stages:
        raise ValueError("운영 단계가 하나 이상 필요합니다.")
    for index, row in enumerate(stages):
        if type(row.get("stage")) is not int or row["stage"] != index:
            raise ValueError("stage는 0부터 연속으로 나열해야 합니다.")
        check_number(row.get("weight"), "weight")
        if row.get("minimum_performance") is not None:
            check_number(row["minimum_performance"], "minimum_performance", maximum=1)
    if not sum(row["weight"] for row in stages) > 0:
        raise ValueError("weight 합이 0보다 커야 합니다.")
    return stages


def read_current(payload, candidates):
    current = payload.get("current_checkpoint")
    if current is None:
        return None
    if (not isinstance(current, dict) or not isinstance(current.get("candidate_id"), str)
            or current["candidate_id"] not in candidates):
        raise ValueError("current_checkpoint의 후보가 등록되지 않았습니다.")
    check_text(current.get("checkpoint_id"), "checkpoint_id")
    if current.get("last_trained_at") is not None:
        check_text(current["last_trained_at"], "last_trained_at")
    check_number(current.get("predicted_performance"), "checkpoint performance", maximum=1)
    return current


def read_options(payload, candidates, stage_count, current):
    """학습 행은 (구간, 후보), 추론 행은 (구간, 후보, 학습 구간)을 키로 둔다. 학습 구간 −1은 기존 checkpoint다."""
    training, inference = {}, {}
    for table, destination in (("training_options", training), ("inference_options", inference)):
        for row in read_records(payload, table):
            stage, candidate_id = row.get("stage"), row.get("candidate_id")
            if type(stage) is not int or not 0 <= stage < stage_count:
                raise ValueError(f"{table}: stage가 운영 범위를 벗어났습니다.")
            if not isinstance(candidate_id, str) or candidate_id not in candidates:
                raise ValueError(f"{table}: 등록되지 않은 후보입니다.")
            if "feasible" in row and row["feasible"] is not True:
                raise ValueError(f"{table}: 실행 불가·미검증 행은 ML에서 제외해 주세요.")
            if table == "training_options":
                key = (stage, candidate_id)
                check_number(row.get("predicted_performance"), "predicted_performance", maximum=1)
                check_number(row.get("training_cost"), "training_cost")
            else:
                origin = row.get("trained_stage")
                if type(origin) is not int or not -1 <= origin <= stage:
                    raise ValueError("trained_stage는 -1(기존 checkpoint)부터 현재 stage 사이여야 합니다.")
                if origin == -1 and (current is None or candidate_id != current["candidate_id"]):
                    raise ValueError("기존 checkpoint가 없는 유지 추론 행입니다.")
                if origin >= 0 and (origin, candidate_id) not in training:
                    raise ValueError("학습 원점이 없는 추론 행입니다.")
                key = (stage, candidate_id, origin)
                check_number(row.get("inference_cost"), "inference_cost")
            if key in destination:
                raise ValueError(f"{table}: 중복된 행입니다: {key}")
            destination[key] = row
    return training, inference


def pareto(labels):
    """더 싼 라벨보다 성능이 높은 라벨만 남긴다. 비용·성능이 같으면 학습 횟수가 적은 쪽을 남긴다."""
    kept = []
    for label in sorted(labels, key=lambda item: (item[0], -item[1], item[2])):
        if not kept or label[1] > kept[-1][1]:
            kept.append(label)
    return kept


def next_labels(labels, stage, weight, floor, plan, room):
    """직전 구간 라벨에 유지와 새 학습을 이어 이번 구간 라벨을 만든다. room개를 넘게 만들면 멈춘다."""
    current, training, inference, options = plan
    new = defaultdict(list)
    created = 0

    def extend(label, state, performance, training_cost, action):
        nonlocal created
        candidate_id, origin = state
        forecast = inference.get((stage, candidate_id, origin))
        if forecast is None or (floor is not None and performance < floor):
            return
        stage_cost = training_cost + forecast["inference_cost"]
        if not isfinite(label[0] + stage_cost):
            raise ValueError("누적 비용이 수치 범위를 넘었습니다.")
        created += 1
        if created > room:
            raise TooManyLabels
        new[state].append((label[0] + stage_cost, label[1] + weight * performance, label[2] + (action != "keep"),
                           label, candidate_id, origin, action, performance, training_cost,
                           forecast["inference_cost"]))

    for state, items in labels.items():
        if state is None:
            continue
        candidate_id, origin = state
        performance = (current["predicted_performance"] if origin == -1
                       else training[(origin, candidate_id)]["predicted_performance"])
        for label in items:
            extend(label, state, performance, 0.0, "keep")
    # 새 checkpoint는 어느 직전 경로 뒤에 와도 같은 비용·성능을 더하므로 밀리지 않는 라벨에서만 잇는다.
    for label in pareto([label for items in labels.values() for label in items]):
        previous = label[4] if label[3] is not None else (current["candidate_id"] if current else None)
        for candidate_id, option in options[stage]:
            action = "adopt" if previous is None else "retrain" if previous == candidate_id else "switch"
            extend(label, (candidate_id, stage), option["predicted_performance"], option["training_cost"], action)
    return {state: pareto(items) for state, items in new.items()}


def build_path(label, stages, total_weight, candidates, current):
    """이전 라벨을 거슬러 올라가 구간별 경로 표를 만든다."""
    chain = []
    while label[3] is not None:
        chain.append(label)
        label = label[3]
    path = []
    for stage, item in enumerate(reversed(chain)):
        cost, _, _, _, candidate_id, origin, action, performance, training_cost, inference_cost = item
        candidate, kept = candidates[candidate_id], origin == -1
        path.append({
            "stage": stage, "candidate_id": candidate_id, "model": candidate["model"],
            "config_id": candidate["config_id"], "head": candidate["head"], "trained_stage": origin,
            "action": action, "checkpoint_id": current["checkpoint_id"] if kept else None,
            "last_trained_at": current.get("last_trained_at") if kept else None,
            "predicted_performance": performance, "weight": stages[stage]["weight"] / total_weight,
            "minimum_performance": stages[stage].get("minimum_performance"),
            "training_cost": training_cost, "inference_cost": inference_cost,
            "stage_cost": training_cost + inference_cost, "cumulative_cost": cost,
        })
    return path


def choose_label(finals, budget, tolerance):
    affordable = [label for label in finals if budget is None or label[0] <= budget]
    if not affordable:
        return min(finals, key=lambda label: (label[0], -label[1])), "over_budget"
    best = max(label[1] for label in affordable)
    return min((label for label in affordable if label[1] >= best - tolerance),
               key=lambda label: (label[0], label[2])), "ok"


def build_frontier(labels, chosen, budget):
    """가장 싼 경로, 고른 경로, 최고 경로와 그 사이 몇 개로 비용·성능 곡선을 만든다."""
    curve, best = [], -1.0
    for label in sorted(labels, key=lambda item: (item[0], -item[1])):
        if label[1] > best or label is chosen:
            curve.append(label)
            best = max(best, label[1])
    if len(curve) > FRONTIER_POINTS:
        step = (len(curve) - 1) / (FRONTIER_POINTS - 1)
        picked = {round(index * step) for index in range(FRONTIER_POINTS)}
        curve = [label for index, label in enumerate(curve) if index in picked or label is chosen]
    return [{"total_cost": label[0], "timeline_performance": label[1], "trainings": label[2],
             "within_budget": budget is None or label[0] <= budget, "chosen": label is chosen}
            for label in curve]


def optimize_plan(payload):
    """추천 경로만 돌려준다. 모델 학습이나 배포 상태는 바꾸지 않는다."""
    if not isinstance(payload, dict):
        raise ValueError("DP 입력은 JSON 객체여야 합니다.")
    if payload.get("metric") != "VUS-PR":
        raise ValueError("metric: VUS-PR만 지원합니다.")
    check_text(payload.get("cost_unit"), "cost_unit")
    budget = payload.get("budget")
    if budget is not None:
        check_number(budget, "budget")
    tolerance = check_number(payload.get("tolerance", DEFAULT_TOLERANCE), "tolerance", maximum=1)
    candidates = read_candidates(payload)
    stages = read_stages(payload)
    current = read_current(payload, candidates)
    training, inference = read_options(payload, candidates, len(stages), current)

    options = [[] for _ in stages]
    for (stage, candidate_id), option in training.items():
        options[stage].append((candidate_id, option))
    plan = (current, training, inference, options)
    total_weight = sum(row["weight"] for row in stages)
    labels = {(current["candidate_id"], -1) if current else None: [START_LABEL]}
    common = {"budget": budget, "metric": payload["metric"], "cost_unit": payload["cost_unit"]}
    summaries = []

    def stop(stage, status, reason):
        return {**common, "status": status, "total_cost": None, "timeline_performance": None, "path": [],
                "current_action": None, "blocked_stage": stage, "stage_summaries": summaries, "frontier": [],
                "reason": reason}

    kept = 0
    for stage, row in enumerate(stages):
        try:
            labels = next_labels(labels, stage, row["weight"] / total_weight, row.get("minimum_performance"),
                                 plan, MAX_LABELS - kept)
        except TooManyLabels:
            return stop(stage, "too_large", f"라벨이 {MAX_LABELS:,}개를 넘어 계산을 멈췄습니다.")
        count = sum(len(items) for items in labels.values())
        kept += count
        summaries.append({"stage": stage, "reachable_states": len(labels), "labels": count})
        if not labels:
            return stop(stage, "infeasible", "이 구간을 이어 갈 후보·추론 행이 없습니다.")

    finals = pareto([label for items in labels.values() for label in items])
    chosen, status = choose_label(finals, budget, tolerance)
    path = build_path(chosen, stages, total_weight, candidates, current)
    return {**common, "status": status, "total_cost": chosen[0], "timeline_performance": chosen[1],
            "trainings": chosen[2], "path": path, "current_action": path[0], "blocked_stage": None,
            "tolerance": tolerance, "stage_summaries": summaries, "frontier": build_frontier(finals, chosen, budget)}
