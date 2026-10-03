"""기업 의사결정 창에서 제출한 ML 추정과 DP 경로를 인쇄용 운영 계획 보고서로 보여 준다."""

from datetime import datetime, timedelta, timezone

import streamlit as st
import streamlit.components.v1 as components

from streamlit_website.DP.planner import optimize_plan
from streamlit_website.report.build_report import build_report_html

KST = timezone(timedelta(hours=9))


def render_report():
    payload, ml_input, ml_output = (st.session_state.get(key) for key in ("dp_input", "ml_input", "ml_output"))
    if payload is None or ml_input is None or ml_output is None:
        st.title("운영 계획 보고서")
        st.info("첫 화면(기업 의사결정 창)에서 CSV와 운영 조건을 제출하면 보고서가 만들어집니다.")
        return
    try:
        result = optimize_plan(payload)
    except ValueError as error:
        st.error(f"운영 경로를 계산하지 못했습니다: {error}")
        return
    # 같은 제출의 보고서는 다시 그려도 계획 번호가 바뀌지 않게 처음 만든 시각을 쓴다.
    generated_at = st.session_state.setdefault("report_time", datetime.now(KST))
    report = build_report_html(payload=payload, result=result, ml_input=ml_input, ml_output=ml_output,
                               generated_at=generated_at)
    st.download_button("보고서 HTML 내려받기", report, file_name=f"TSAD_운영계획_{generated_at:%Y%m%d_%H%M}.html",
                       mime="text/html", help="브라우저에서 열어 인쇄하면 A4 PDF로 저장됩니다.")
    components.html(report, height=1500, scrolling=True)
