"""TSAD 운영 계획 화면: 데이터 입력 → 후보·ML 추정 → 예산 안 운영 경로 → 인쇄용 보고서."""

import sys
from pathlib import Path

import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from streamlit_website.db_connection.view import render_candidate_intake
from streamlit_website.DP.view import render_dp_planner
from streamlit_website.report.view import render_report

STYLE = """
<style>
  .stApp { background:#fff; color:#241c35; font-family:Inter, Pretendard, "Noto Sans KR", sans-serif; }
  #MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"],
  [data-testid="stSidebar"], [data-testid="stExpandSidebarButton"] { display:none !important; }
  [data-testid="stHeader"] { background:transparent; height:0; }
  .block-container { max-width:1380px; padding:24px 42px 84px; }
  div[data-testid="stMetric"] { border-bottom:1px solid #17121e; padding:13px 0; }
  [data-testid="stDataFrame"] { border:1px solid #e6e0f0; border-radius:12px; overflow:hidden; }
  .stButton > button { background:#7c3aed; border:1px solid #7c3aed; border-radius:8px; color:#fff; font-weight:650; }
  [data-testid="stMain"] [data-testid="stRadioGroup"] { display:flex; flex-direction:row; gap:5px; justify-content:center; }
  [data-testid="stMain"] label[data-testid="stRadioOption"] { border-radius:999px; padding:0 13px; min-height:38px; }
  [data-testid="stMain"] label[data-testid="stRadioOption"][data-selected="true"] { background:#17121e; }
  [data-testid="stMain"] label[data-testid="stRadioOption"][data-selected="true"] p { color:#fff; }
  @media (max-width: 768px) { .block-container { padding:20px 16px 44px; } }
</style>
"""

st.set_page_config(page_title="TSAD Decision Studio", page_icon="◈", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(STYLE, unsafe_allow_html=True)
page = st.radio("작업", ["데이터 입력", "운영 경로", "보고서"], horizontal=True, label_visibility="collapsed")
if page == "운영 경로":
    render_dp_planner()
elif page == "보고서":
    render_report()
else:
    render_candidate_intake()
