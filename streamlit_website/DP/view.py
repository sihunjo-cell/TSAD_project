"""ML 추정표로 예산 안 운영 경로를 계산해 보여 준다."""

import json

import pandas as pd
import streamlit as st

from streamlit_website.DP.planner import optimize_plan
from streamlit_website.DP.recommend import TIGHT_BUDGET_RATIO, recommend

ACTION_LABELS = {"adopt": "최초 도입", "keep": "유지", "retrain": "재학습", "switch": "교체"}
PATH_COLUMNS = {
    "stage": "구간", "ratio_percent": "데이터 비율 (%)", "model": "모델", "config_id": "설정 ID", "head": "head",
    "trained_stage": "학습 구간", "action": "행동", "weight": "구간 비중", "predicted_performance": "예측 VUS-PR",
    "training_cost": "학습비", "inference_cost": "추론비", "cumulative_cost": "누적 비용",
}
FRONTIER_COLUMNS = {"total_cost": "전체 비용", "timeline_performance": "타임라인 예측 VUS-PR",
                    "trainings": "학습 횟수", "within_budget": "예산 안", "chosen": "선택", "reference": "참고"}


def read_payload():
    """입력 화면에서 넘어온 ML 추정표, 없으면 업로드한 JSON."""
    if st.session_state.get("dp_input") is not None:
        st.caption("입력 화면에서 제출한 ML 추정표를 씁니다.")
        return st.session_state["dp_input"]
    upload = st.file_uploader("ML 추정표 JSON (입력 화면에서 제출하면 자동으로 들어옵니다)", type=["json"])
    return None if upload is None else json.loads(upload.getvalue().decode("utf-8-sig"))


def show_inputs(payload):
    with st.expander("구간별 입력표와 전달 형식"):
        st.write("운영 구간 (weight는 구간 길이)")
        st.dataframe(pd.DataFrame(payload["stages"]), hide_index=True, use_container_width=True)
        st.write("신규 도입·재학습 후보")
        st.dataframe(pd.DataFrame(payload["training_options"]), hide_index=True, use_container_width=True)
        st.write("학습 구간별 추론비")
        st.dataframe(pd.DataFrame(payload["inference_options"]), hide_index=True, use_container_width=True)
    st.download_button("입력 JSON 내려받기", json.dumps(payload, ensure_ascii=False, indent=2),
                       file_name="dp_input.json", mime="application/json")


def path_table(result, payload):
    frame = pd.DataFrame(result["path"])
    frame["ratio_percent"] = frame["stage"].map({row["stage"]: row.get("ratio_percent") for row in payload["stages"]})
    frame["stage"] += 1
    frame["trained_stage"] = frame["trained_stage"].map(lambda stage: "기존" if stage == -1 else stage + 1)
    frame["action"] = frame["action"].map(ACTION_LABELS)
    frame["predicted_performance"] = frame["predicted_performance"].round(2)
    return frame[list(PATH_COLUMNS)].rename(columns=PATH_COLUMNS)


def render_dp_planner():
    st.title("운영 경로")
    st.caption("예산 안에서 구간 길이로 가중한 예측 VUS-PR이 가장 높은 경로를 고릅니다.")
    try:
        payload = read_payload()
        if payload is None:
            return
        result = optimize_plan(payload)
    except (ValueError, UnicodeError) as error:
        st.error(f"DP 입력을 확인해 주세요: {error}")
        return
    show_inputs(payload)
    if result["status"] in ("infeasible", "too_large"):
        st.warning(f"구간 {result['blocked_stage'] + 1}: {result['reason']}")
        return
    if result["status"] == "over_budget":
        st.warning("예산으로는 어떤 경로도 끝까지 갈 수 없습니다. 가장 싼 경로와 그 비용을 보여 줍니다.")
    plan, _, tight = recommend(payload, result)
    switched = plan is not result
    action = plan["current_action"]
    st.subheader(f"현재 제안: {action['model']} {ACTION_LABELS[action['action']]}")
    if switched:
        st.info(f"빠듯한 예산이라 가장 싼 경로를 제안합니다(예산이 가장 싼 경로 비용의 {tight:,.1f}배). "
                f"Dev18 백테스트에서 {TIGHT_BUDGET_RATIO}배 이하 예산은 어떤 방식도 가장 싼 계획을 통계적으로 넘지 "
                "못했습니다. 예측이 더 높은 경로는 아래에 참고로 둡니다.")
    performance, cost = st.columns(2)
    performance.metric("타임라인 예측 VUS-PR (구간 길이 가중 평균)", f"{plan['timeline_performance']:.2f}")
    budget = "제한 없음" if plan["budget"] is None else f"{plan['budget']:,.3f}"
    cost.metric(f"전체 학습·추론비 ({plan['cost_unit']})", f"{plan['total_cost']:,.3f}", help=f"예산: {budget}")
    st.dataframe(path_table(plan, payload), hide_index=True, use_container_width=True)
    st.caption("예측 VUS-PR의 절대값은 새 현장에서 ±0.3 정도 틀리므로 가장 싼 경로와의 차이로 읽어 주세요. "
               "현재 행동은 데이터가 다음 비율에 닿을 때까지 유효하고, 그때 누적 prefix와 남은 예산으로 다시 계산합니다. "
               "데이터 비율과 재학습 시점은 남은 운영 기간·수집 속도 입력에 따라 달라집니다.")
    if switched:
        with st.expander("참고 : 예측상 고른 경로 (빠듯한 예산이라 차이에 근거가 없음)"):
            st.dataframe(path_table(result, payload), hide_index=True, use_container_width=True)
    st.write("예산을 바꿀 때의 선택지 (비용이 늘 때 성능이 오르는 경로만)")
    frontier = pd.DataFrame(plan["frontier"]).round({"timeline_performance": 2})
    st.dataframe(frontier.rename(columns=FRONTIER_COLUMNS), hide_index=True, use_container_width=True)
    with st.expander("DP가 구간마다 남긴 상태"):
        st.dataframe(pd.DataFrame(result["stage_summaries"]), hide_index=True, use_container_width=True)
    st.download_button("경로 JSON 내려받기", json.dumps(plan, ensure_ascii=False, indent=2),
                       file_name="dp_plan.json", mime="application/json")
