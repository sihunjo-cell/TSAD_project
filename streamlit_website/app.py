"""TSAD Decision Studio: 기업 의사결정 창 → 제안 요약 → 스튜디오(운영 경로·후보 추정·보고서)."""

import sys
from html import escape
from pathlib import Path

import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from streamlit_website.db_connection.view import render_candidate_results, render_intake_form
from streamlit_website.DP.planner import optimize_plan
from streamlit_website.DP.recommend import TIGHT_BUDGET_RATIO, recommend
from streamlit_website.DP.view import render_dp_planner
from streamlit_website.report.view import render_report

CSS = Path(__file__).with_name("app.css").read_text(encoding="utf-8")
INTRO = ("<div class='app-intro'><div class='app-intro__halo'></div><div class='app-intro__mark'>◈ TSAD</div>"
         "<div class='app-intro__copy'>DECISION STUDIO</div><div class='app-intro__progress'></div></div>")
START_STAGE = (
    "<section class='start-stage'><div class='start-stage__grid'></div><div class='signal-logo'>"
    "<svg class='signal-logo__aperture' viewBox='0 0 300 252' role='img' aria-label='Signal Aperture logo'>"
    "<defs><linearGradient id='aperture-gradient' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#f5efff'/><stop offset='.52' stop-color='#ae7aff'/><stop offset='1' stop-color='#6e37d2'/></linearGradient></defs>"
    "<circle class='aperture-ring' cx='150' cy='126' r='102' stroke-dasharray='8 12'/><circle class='aperture-ring' cx='150' cy='126' r='68' stroke-dasharray='3 10'/>"
    "<path class='aperture-trace' d='M32 202 C83 202 78 143 124 130 C145 124 151 103 150 72 C149 44 174 35 233 35'/><path class='aperture-trace aperture-trace--inner' d='M32 127 C88 127 95 101 128 111 C146 116 150 130 172 137 C195 145 203 188 268 188'/><path class='aperture-trace aperture-trace--inner' d='M66 228 C113 228 116 179 140 153 C152 140 161 135 177 129 C209 116 204 65 267 65'/><circle class='aperture-core' cx='150' cy='126' r='13'/></svg>"
    "<div class='signal-logo__name'>SIGNAL APERTURE</div><div class='signal-logo__sub'>TIME SERIES ANOMALY DECISION</div></div>"
    "<p class='start-stage__statement'>데이터가 부족한 지금부터,<br>다음 모델 전환의 순간까지.</p>"
    "<div class='start-stage__cue'>↓ START A COMPANY DECISION</div></section>"
    "<section class='decision-window'><div class='decision-window__eyebrow'>COMPANY DECISION WINDOW</div>"
    "<h1>우리 환경에서,<br>무엇을 먼저 검토해야 할까요?</h1>"
    "<p class='decision-window__copy'>누적 정상 데이터와 운영 조건·예산을 입력하면 구간별로 실행할 수 있는 모델과 다음 재평가 시점, "
    "예산 안 운영 경로를 함께 정리합니다.</p></section>"
)
BRAND = ("<div class='top-brand'><span class='top-brand__symbol'>◈</span>"
         "<div><strong>TSAD</strong><small>DECISION STUDIO</small></div></div>")
FAMILIES = {"training_free": "경량·통계 기반", "fit_full_prefix": "target 학습형 다변량",
            "strict_zero_shot": "zero-shot / foundation"}
ACTIONS = {"adopt": "도입", "keep": "유지", "retrain": "재학습", "switch": "교체"}
PAGES = {"운영 경로": render_dp_planner, "후보·추정": render_candidate_results, "보고서": render_report}


def show_intro():
    """브라우저 세션마다 한 번 시작 화면을 띄운다."""
    if "intro_seen" not in st.session_state:
        st.markdown(INTRO, unsafe_allow_html=True)
        st.session_state.intro_seen = True


def back_to_window():
    st.session_state.app_view = "intake"
    st.session_state.profile_ready = False
    st.rerun()


def proposal_text(plan, payload, ml_input):
    """제안 요약의 (머리말, 제목, 본문)."""
    info, conditions, stages = ml_input["input"], ml_input["operating_conditions"], payload["stages"]
    unit = "원" if payload["cost_unit"] == "KRW" else "초"
    budget = "제한 없음" if payload.get("budget") is None else f"{payload['budget']:,g}{unit}"
    label = f"YOUR PROPOSAL · {stages[0]['ratio_percent']}% REFERENCE BAND"
    if not plan.get("path"):
        return label, "계획을 만들지 못함", f"구간 {plan['blocked_stage'] + 1}: {plan['reason']}"
    step = plan["path"][0]
    uses = {f"{row['config_id']}::{row['head']}": row.get("target_use") for row in ml_input["candidates"]}
    title = f"{FAMILIES.get(uses.get(step['candidate_id']), '모델')} · {step['model']} {ACTIONS[step['action']]}"
    upcoming = (f"다음 검토 지점은 데이터 {stages[1]['ratio_percent']}%(약 {stages[1]['elapsed_days']:,.1f}일 뒤)입니다."
                if len(stages) > 1 else "남은 기간에는 다음 검토 지점이 없습니다.")
    body = (f"{info['row_count']:,} 정상 rows · {info['channel_count']} channels · "
            f"{conditions.get('gpu_memory_gib') or 0:g} GiB VRAM · 예산 {budget} 조건을 기준으로 제안했습니다. "
            f"예측 VUS-PR {plan['timeline_performance']:.2f}, 전체 비용 {plan['total_cost']:,.3g}{unit}. {upcoming} "
            f"{info.get('source_name') or ''}")
    return label, title, body


def render_proposal():
    payload, ml_input = st.session_state["dp_input"], st.session_state["ml_input"]
    result = optimize_plan(payload)
    plan, _, tight = recommend(payload, result)
    label, title, body = (escape(text) for text in proposal_text(plan, payload, ml_input))
    st.markdown(f"<section class='decision-summary'><div class='decision-summary__label'>{label}</div>"
                f"<h2>{title}</h2><p>{body}</p></section>", unsafe_allow_html=True)
    st.caption("예측 VUS-PR의 절대값은 새 현장에서 ±0.3 정도 틀립니다. 경로 전체와 예산별 대안은 스튜디오의 '운영 경로'와 '보고서'에 있습니다.")
    notes = []
    if plan is not result:
        notes.append(f"빠듯한 예산(가장 싼 경로 비용의 {tight:,.1f}배)이라 가장 싼 경로를 제안합니다. "
                     f"Dev18 백테스트에서 {TIGHT_BUDGET_RATIO}배 이하 예산은 어떤 방식도 가장 싼 계획을 통계적으로 넘지 못했습니다.")
    if result["status"] == "over_budget":
        notes.append("예산으로는 어떤 경로도 끝까지 갈 수 없어 가장 싼 경로를 보여 줍니다.")
    if notes:
        st.markdown("<div class='warn'><b>읽을 때 주의</b><br>" + "<br>".join(f"• {escape(note)}" for note in notes)
                    + "</div>", unsafe_allow_html=True)
    st.markdown("<div class='recommendation-actions'>", unsafe_allow_html=True)
    open_studio, revise = st.columns(2)
    if open_studio.button("맞춤 Decision Studio 열기", type="primary", use_container_width=True):
        st.session_state.app_view = "studio"
        st.rerun()
    if revise.button("입력값 다시 보기", use_container_width=True):
        back_to_window()
    st.markdown("</div>", unsafe_allow_html=True)


def decision_window():
    if st.session_state.get("profile_ready") and st.session_state.get("dp_input") is not None:
        render_proposal()
        return
    st.markdown(START_STAGE, unsafe_allow_html=True)
    if render_intake_form():
        st.session_state.profile_ready = True
        st.rerun()


def show_profile():
    info, conditions = st.session_state["ml_input"]["input"], st.session_state["ml_input"]["operating_conditions"]
    st.markdown("<div class='control-kicker'>COMPANY PROFILE</div><p class='control-copy'>"
                "첫 화면에서 입력한 조건으로 현재 분석 화면을 계산하고 있습니다.</p>", unsafe_allow_html=True)
    st.write(f"**{info['row_count']:,} 정상 rows** · {info['channel_count']} channels · {info.get('source_name') or ''}")
    st.write(f"남은 운영 {conditions['operating_days']}일 · 수집 {conditions['collection_rows_per_second']:g}행/초 · "
             f"추론 {conditions['inference_rows_per_day']:,}행/일")
    budget = f"{conditions['budget']:,g}" if conditions.get("budget") else "제한 없음"
    st.write(f"VRAM {conditions.get('gpu_memory_gib') or 0:g} GiB · CPU {conditions['cpu_hourly_cost']:,g}원/시간 · "
             f"GPU {conditions['gpu_hourly_cost']:,g}원/시간 · 예산 {budget}")
    if st.button("기업 의사결정 창으로 돌아가기", use_container_width=True):
        back_to_window()


def render_studio():
    brand, navigation, controls = st.columns([1.05, 2.45, 0.82], gap="small")
    brand.markdown(BRAND, unsafe_allow_html=True)
    with navigation:
        page = st.radio("Workspace navigation", list(PAGES), horizontal=True, label_visibility="collapsed")
    with controls:
        if st.button("← 처음으로", key="back-to-company-window", use_container_width=True):
            back_to_window()
        with st.popover("기업 프로필"):
            show_profile()
    st.markdown("<div class='workspace-rule'></div>", unsafe_allow_html=True)
    PAGES[page]()


st.set_page_config(page_title="TSAD Decision Studio", page_icon="◈", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
show_intro()
if st.session_state.get("app_view") == "studio" and st.session_state.get("dp_input") is not None:
    render_studio()
else:
    decision_window()
