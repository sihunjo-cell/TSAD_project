"""ML 추정표를 DP 입력 계약(`streamlit_website/DP/README.md`)으로 옮긴다."""

SECONDS_PER_HOUR = 3600


def build_dp_input(ml_output, ml_input):
    """실행 가능한 후보 전부를 학습 선택지로, 구간 추론비를 학습 원점별 추론 행으로 넘긴다.

    구간 가중치는 구간 길이 / 남은 운영 기간이다. CPU·GPU 시간당 단가가 있으면 초를 금액으로
    바꾸고 예산도 그 단위로 보며, 단가가 없으면 계산 초를 그대로 쓴다. 추론비는 학습 시점과
    관계없이 그 구간의 후보 추정값이다. 기존 checkpoint는 현재 후보와 checkpoint ID,
    성능 추정이 모두 있을 때만 넘긴다.
    """
    conditions = ml_input["operating_conditions"]
    rates = {"cpu": conditions.get("cpu_hourly_cost") or 0.0, "cuda": conditions.get("gpu_hourly_cost") or 0.0}
    priced = any(rates.values())
    backend = {f"{c['config_id']}::{c['head']}": c.get("backend") for c in ml_input["candidates"]}

    def cost(candidate_id, seconds):
        # 측정 장치를 모르면 공짜로 두지 않고 비싼 쪽 단가를 쓴다.
        return seconds * rates.get(backend.get(candidate_id), max(rates.values())) / SECONDS_PER_HOUR if priced else seconds

    stages = ml_output.future_stages
    days = conditions.get("operating_days") or 0
    ends = [stage.elapsed_days for stage in stages[1:]] + [days]
    weights = [end - stage.elapsed_days for stage, end in zip(stages, ends)] if days else [1.0] * len(stages)

    estimates = [estimate for stage_estimates in (*ml_output.stage_candidates.values(),
                                                  *ml_output.stage_candidates_excluded.values())
                 for estimate in stage_estimates]
    inference_cost = {(estimate.stage, estimate.candidate_id): estimate.estimated_inference_cost_seconds
                      for estimate in estimates}
    training = [{"stage": estimate.stage, "candidate_id": estimate.candidate_id,
                 "predicted_performance": estimate.predicted_performance,
                 "training_cost": cost(estimate.candidate_id, estimate.estimated_training_cost_seconds or 0.0)}
                for stage_estimates in ml_output.stage_candidates.values() for estimate in stage_estimates]
    inference = [{"stage": stage, "candidate_id": option["candidate_id"], "trained_stage": option["stage"],
                  "inference_cost": cost(option["candidate_id"], inference_cost[stage, option["candidate_id"]])}
                 for option in training for stage in range(option["stage"], len(stages))]
    keep = ml_output.checkpoint_maintenance_options
    current = None
    if keep and keep[0].checkpoint_reference and keep[0].predicted_performance is not None:
        current = {"candidate_id": keep[0].candidate_id, "checkpoint_id": keep[0].checkpoint_reference,
                   "last_trained_at": ml_input["current_model"].get("last_trained_at") or None,
                   "predicted_performance": keep[0].predicted_performance}
        inference += [{"stage": option.stage, "candidate_id": option.candidate_id, "trained_stage": -1,
                       "inference_cost": cost(option.candidate_id, option.estimated_inference_cost_seconds)}
                      for option in keep if option.estimated_inference_cost_seconds is not None]
    payload = {
        "metric": "VUS-PR", "cost_unit": "KRW" if priced else "seconds",
        "budget": conditions.get("budget") or None,
        "candidates": list({estimate.candidate_id: {
            "candidate_id": estimate.candidate_id, "model": estimate.model,
            "config_id": estimate.configuration, "head": estimate.head,
        } for estimate in estimates}.values()),
        "stages": [{"stage": stage.stage, "weight": weight, "elapsed_days": stage.elapsed_days,
                    "ratio_percent": stage.ratio_percent, "available_rows": stage.future_rows,
                    "inference_rows": stage.inference_volume}
                   for stage, weight in zip(stages, weights)],
        "training_options": training, "inference_options": inference, "current_checkpoint": current,
    }
    if conditions.get("tolerance") is not None:
        payload["tolerance"] = conditions["tolerance"]
    return payload
