"""TSAD Decision Studio — a decision-support prototype based on Dev18 evidence."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from streamlit_website.db_connection.view import render_candidate_intake


st.set_page_config(
    page_title="TSAD Decision Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# The figures marked observed come from the supplied Dev18 brief.  Values marked
# candidate are selection milestones, not claims of statistically significant wins.
CHECKPOINTS = [5, 10, 20, 40, 60, 80, 100]
OBSERVED_VUS = {
    "MWVAR": {q: 0.274663 for q in CHECKPOINTS},
    "GDN": {10: 0.276017, 20: 0.306231},
    "TSPulse": {q: 0.281481 for q in CHECKPOINTS},
    "PaAno": {40: 0.331},
}

MODEL_DATA = {
    "MWVAR": {
        "tier": "Tier 1", "kind": "경량·비학습 기준선", "min_q": 5,
        "gpu_gib": 0.0, "cost": 1, "latency": "낮음", "training": "불필요",
        "desc": "Cold start에서도 비교 가능한 경량 기준선입니다.",
        "status": "Observed (Dev18)",
    },
    "GDN": {
        "tier": "Tier 2", "kind": "target 정상 데이터 학습", "min_q": 10,
        "gpu_gib": 18.35, "cost": 4, "latency": "중간", "training": "필요",
        "desc": "10%부터 공통 실행 가능; 20%에서 Dev18 성능 우세 후보입니다.",
        "status": "Observed (Dev18)",
    },
    "PaAno": {
        "tier": "Tier 2", "kind": "target 정상 데이터 학습", "min_q": 40,
        "gpu_gib": 0.53, "cost": 3, "latency": "중간", "training": "필요",
        "desc": "40%부터 전환 재평가 대상인 학습형 후보입니다.",
        "status": "Candidate milestone",
    },
    "TSPulse": {
        "tier": "Tier 3", "kind": "target 학습 불필요 zero-shot", "min_q": 5,
        "gpu_gib": 8.23, "cost": 4, "latency": "높음", "training": "불필요",
        "desc": "Cold start의 zero-shot 선택지이며, 비용 우위는 아직 검증되지 않았습니다.",
        "status": "Observed (Dev18)",
    },
    "SQDIFF_LAST3": {
        "tier": "Baseline", "kind": "경량 기준선", "min_q": 5,
        "gpu_gib": 0.0, "cost": 1, "latency": "낮음", "training": "불필요",
        "desc": "최근 값 차분 기반의 설명 가능한 기준선입니다.", "status": "Portfolio",
    },
    "PCA": {
        "tier": "Baseline", "kind": "전통적 비지도", "min_q": 5,
        "gpu_gib": 0.0, "cost": 2, "latency": "낮음", "training": "필요",
        "desc": "낮은 운영 부담의 전통적 비교 모델입니다.", "status": "Portfolio",
    },
    "TimeRCD": {
        "tier": "Portfolio", "kind": "구조 기반 모델", "min_q": 10,
        "gpu_gib": 3.17, "cost": 3, "latency": "중간", "training": "필요",
        "desc": "현재 recipe에서 peak GPU memory가 3.17 GiB로 기록되었습니다.",
        "status": "Resource observed",
    },
    "ALoRa": {
        "tier": "Excluded", "kind": "recipe 제약", "min_q": math.inf,
        "gpu_gib": None, "cost": None, "latency": "—", "training": "—",
        "desc": "현재 채널 구조에서는 공통 후보로 비교할 수 없습니다.", "status": "Constraint",
    },
}

PALETTE = {"MWVAR": "#8da2b6", "GDN": "#45d5b0", "PaAno": "#7b8cff", "TSPulse": "#eeae57"}

# These are research references, not a claim that a paper's method is identical
# to the selected implementation.  They are surfaced beside a recommendation so
# a decision maker can immediately inspect the modelling rationale.
PAPER_LIBRARY = {
    "baseline": {
        "title": "Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?",
        "venue": "ICML 2024",
        "summary": "복잡한 모델 전환 전에 단순·해석 가능한 기준선과 공정한 벤치마크를 먼저 검증해야 한다는 관점을 제시합니다.",
        "pdf": "https://proceedings.mlr.press/v235/sarfraz24a/sarfraz24a.pdf",
        "paper": "https://proceedings.mlr.press/v235/sarfraz24a.html",
        "code": "",
    },
    "catch": {
        "title": "CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching",
        "venue": "ICLR 2025",
        "summary": "주파수 대역별 패치와 채널 상관관계를 함께 학습해 다변량 이상 구간을 재구성 기반으로 포착합니다.",
        "pdf": "https://proceedings.iclr.cc/paper_files/paper/2025/file/2b25c39788e5cf11d3541de433ebf4c0-Paper-Conference.pdf",
        "paper": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b25c39788e5cf11d3541de433ebf4c0-Abstract-Conference.html",
        "code": "https://github.com/decisionintelligence/CATCH",
    },
    "sarad": {
        "title": "SARAD: Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series",
        "venue": "NeurIPS 2024",
        "summary": "센서 간 공간적 연관성의 변화까지 모델링해 다변량 공정 데이터의 탐지와 진단을 함께 다룹니다.",
        "pdf": "https://proceedings.neurips.cc/paper_files/paper/2024/file/56ad264ac7448239145606cf4106042f-Paper-Conference.pdf",
        "paper": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/56ad264ac7448239145606cf4106042f-Abstract-Conference.html",
        "code": "",
    },
    "timeinf": {
        "title": "TimeInf: Time Series Data Contribution via Influence Functions",
        "venue": "ICLR 2025",
        "summary": "각 시점이 예측에 미치는 영향을 추적해 이상 탐지 결과를 더 설명 가능하게 만드는 model-agnostic 접근입니다.",
        "pdf": "https://arxiv.org/pdf/2407.15247",
        "paper": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/214382ea2931ca1637ebd7d15ef4b454-Abstract-Conference.html",
        "code": "https://github.com/yzhang511/TimeInf",
    },
    "units": {
        "title": "UniTS: A Unified Multi-Task Time Series Model",
        "venue": "NeurIPS 2024",
        "summary": "이질적인 시계열을 공통 표현으로 옮겨 few-shot·prompt 방식의 다양한 downstream task에 전이합니다.",
        "pdf": "https://proceedings.neurips.cc/paper_files/paper/2024/file/fe248e22b241ae5a9adf11493c8c12bc-Paper-Conference.pdf",
        "paper": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html",
        "code": "https://github.com/mims-harvard/UniTS",
    },
}


def inject_css() -> None:
    st.markdown("""
    <style>
      :root {
        --bg: #f8f6fc; --surface: #ffffff; --surface-subtle: #f2eff9;
        --ink: #241c35; --muted: #716a82; --quiet: #9b94a9; --line: #e6e0f0;
        --navy: #1b1430; --navy-soft: #30234e; --primary: #7c3aed;
        --primary-strong: #6528d1; --primary-soft: #f0eafe; --success: #16756a; --success-soft: #e8f7f3;
        --warning: #9a5a18; --warning-soft: #fff6e9; --radius: 12px;
      }
      * { box-sizing: border-box; }
      .stApp { background:radial-gradient(circle at 84% -18%, #ede5ff 0, transparent 28%), var(--bg); color:var(--ink); font-family:Inter, Pretendard, "Noto Sans KR", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      #MainMenu, footer { visibility:hidden; }
      [data-testid="stHeader"] { background:transparent; height:0; }
      [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"], [data-testid="stMainMenu"] { display:none !important; }
      /* The workspace has no persistent rail: controls live in the top control room. */
      [data-testid="stSidebar"], [data-testid="stExpandSidebarButton"] { display:none !important; }
      .block-container { max-width:1440px; padding:40px 48px 72px; }
      [data-testid="stSidebar"] { background:var(--navy); border-right:1px solid #372755; }
      [data-testid="stSidebar"] > div:first-child { background:var(--navy); }
      [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:28px 16px 32px; }
      [data-testid="stSidebar"] * { color:#edf2f8 !important; }
      [data-testid="stSidebar"] hr { border-color:#3c2a5c !important; margin:24px 0 !important; }
      [data-testid="stSidebar"] [data-testid="stRadio"] > div { gap:4px; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { min-height:38px; border:1px solid transparent; border-radius:8px; padding:7px 9px; transition:background .16s ease, border-color .16s ease; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#291d45; transform:translateX(2px); }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background:var(--navy-soft); border-color:#5b4386; box-shadow:inset 3px 0 0 #a78bfa; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color:#fff !important; font-weight:650; }
      [data-testid="stSidebar"] [data-testid="stRadio"] input { accent-color:#7da4ff; }
      [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#c9d4e4 !important; font-size:12px; font-weight:600; letter-spacing:-.01em; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > p > strong { font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#9eb0c8 !important; }
      .sidebar-logo { color:#fff; font-size:19px; font-weight:760; letter-spacing:-.045em; line-height:1.18; }
      .sidebar-caption { color:#c4b5fd !important; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:600; letter-spacing:.1em; margin:7px 0 28px; }
      .hero { background:linear-gradient(126deg, #1a122e 0%, #2c1c4d 58%, #171124 100%); border:1px solid #4a3371; border-radius:14px; color:#fff; margin-bottom:42px; overflow:hidden; padding:34px 38px 36px; position:relative; animation:hero-enter .72s cubic-bezier(.2,.8,.2,1) both; }
      .hero:before { background:radial-gradient(circle, rgba(169,132,255,.48) 0%, rgba(169,132,255,0) 68%); content:""; height:330px; pointer-events:none; position:absolute; right:-76px; top:-180px; width:330px; }
      .hero:after { border:1px solid rgba(215,198,255,.18); border-radius:50%; content:""; height:250px; pointer-events:none; position:absolute; right:52px; top:-132px; width:250px; }
      .eyebrow { color:#d1c4ff; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:700; letter-spacing:.14em; position:relative; text-transform:uppercase; margin-bottom:14px; }
      .hero h1 { max-width:840px; font-size:clamp(30px, 3vw, 40px); font-weight:760; letter-spacing:-.055em; line-height:1.16; margin:0 0 14px; text-wrap:balance; }
      .hero h1, .hero p { position:relative; }
      .hero p { max-width:690px; color:#d9d1e8; font-size:14px; line-height:1.75; margin:0; }
      .section-title { color:var(--ink); font-size:22px; font-weight:730; letter-spacing:-.045em; line-height:1.3; margin:0 0 6px; }
      .section-sub { color:var(--muted); font-size:13px; line-height:1.65; margin:0 0 20px; }
      .metric-box { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); min-height:112px; padding:18px; transition:border-color .2s ease, box-shadow .2s ease, transform .2s ease; animation:card-enter .55s cubic-bezier(.2,.8,.2,1) both; }
      .metric-box:hover { border-color:#cbbcf2; box-shadow:0 10px 26px rgba(71,44,123,.08); transform:translateY(-3px); }
      .metric-kicker { color:var(--muted); font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:650; letter-spacing:.08em; text-transform:uppercase; }
      .metric-value { color:var(--ink); font-size:25px; font-weight:740; letter-spacing:-.055em; line-height:1.2; margin-top:10px; }
      .metric-note { color:var(--muted); font-size:11px; line-height:1.45; margin-top:9px; }
      .recommendation { background:linear-gradient(135deg, #24163f 0%, #40216c 100%); border:1px solid #68469a; border-radius:var(--radius); box-shadow:0 16px 36px rgba(60,33,103,.16); overflow:hidden; padding:26px; position:relative; animation:card-enter .7s .08s cubic-bezier(.2,.8,.2,1) both; }
      .recommendation:after { background:rgba(193,169,255,.12); border-radius:50%; content:""; height:180px; position:absolute; right:-90px; top:-105px; width:180px; }
      .recommendation h2 { color:#fff; font-size:28px; font-weight:730; letter-spacing:-.055em; line-height:1.15; margin:10px 0 10px; }
      .recommendation h2, .recommendation p, .recommendation .badge { position:relative; z-index:1; }
      .recommendation p { color:#e4d9f7; font-size:13px; line-height:1.65; margin:0; }
      .badge { background:var(--surface-subtle); border:1px solid #dfe5ee; border-radius:999px; color:#475467; display:inline-flex; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:650; letter-spacing:.02em; padding:4px 8px; }
      .recommendation .badge { background:rgba(202,181,255,.16); border-color:rgba(202,181,255,.34); color:#eee8ff; }
      .evidence { background:transparent; border-left:2px solid #c4b5fd; border-radius:0; color:#443957; margin:0; padding:10px 0 10px 14px; transition:border-color .18s ease, padding-left .18s ease; }
      .evidence:hover { border-left-color:var(--primary); padding-left:18px; }
      .evidence + .evidence { border-top:1px solid var(--line); }
      .evidence strong { color:#344054; font-size:13px; font-weight:600; line-height:1.55; }
      .evidence span { color:var(--muted); font-size:12px; line-height:1.6; }
      .warn { background:var(--warning-soft); border:1px solid #f4ddb1; border-left:3px solid #d89a31; border-radius:8px; color:#765116; font-size:12px; line-height:1.65; padding:13px 15px; }
      div[data-testid="stMetric"] { background:transparent; border:0; border-bottom:1px solid var(--line); border-radius:0; padding:13px 0; }
      div[data-testid="stMetric"]:first-of-type { border-top:1px solid var(--line); }
      div[data-testid="stMetricLabel"] { color:var(--muted); font-size:12px; font-weight:560; }
      div[data-testid="stMetricValue"] { color:var(--ink); font-size:25px; font-weight:720; letter-spacing:-.045em; }
      [data-baseweb="input"] { background:var(--surface) !important; border-color:#d0d5dd !important; border-radius:8px !important; box-shadow:none !important; min-height:42px; transition:border-color .15s ease, box-shadow .15s ease; }
      [data-baseweb="input"]:focus-within { border-color:var(--primary) !important; box-shadow:0 0 0 3px rgba(124,58,237,.14) !important; }
      [data-baseweb="input"] input { color:var(--ink) !important; font-size:13px !important; }
      [data-testid="stSidebar"] [data-baseweb="input"] { background:#271b43 !important; border-color:#4b376d !important; }
      [data-testid="stSidebar"] [data-baseweb="input"] input { color:#fff !important; }
      [data-testid="stSlider"] [role="slider"] { border-color:var(--primary) !important; background:var(--primary) !important; box-shadow:0 0 0 3px rgba(124,58,237,.14); }
      [data-testid="stSlider"] div[data-testid="stTickBar"] { opacity:.65; }
      [data-testid="stToggle"] [data-checked="true"] { background-color:var(--primary) !important; }
      .stButton > button { align-items:center; background:var(--primary); border:1px solid var(--primary); border-radius:8px; color:#fff; font-size:13px; font-weight:650; min-height:40px; padding:0 15px; transition:background .15s ease, border-color .15s ease, transform .15s ease; }
      .stButton > button:hover { background:var(--primary-strong); border-color:var(--primary-strong); box-shadow:0 8px 18px rgba(95,47,188,.22); transform:translateY(-2px); }
      .stButton > button:focus-visible { box-shadow:0 0 0 3px rgba(124,58,237,.22); }
      [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; }
      [data-testid="stDataFrame"] [role="columnheader"] { background:#f8fafc !important; color:#475467 !important; font-size:11px !important; font-weight:650 !important; }
      [data-testid="stPlotlyChart"] { border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; background:var(--surface); padding:6px; }
      [data-testid="stCaptionContainer"] { color:var(--muted); font-size:11px; line-height:1.6; }
      .roadmap-card { border:1px solid var(--line); border-top:3px solid #bda9ee; border-radius:var(--radius); background:var(--surface); min-height:278px; padding:20px; transition:box-shadow .2s ease, transform .2s ease; }
      .roadmap-card:hover { box-shadow:0 12px 28px rgba(72,42,124,.08); transform:translateY(-3px); }
      .roadmap-card--available { border-top-color:#4d9e8c; }
      .roadmap-card--measure { border-top-color:#8c6ed3; }
      .roadmap-card--decide { border-top-color:#d3a05d; }
      .roadmap-card strong { color:var(--ink); font-size:14px; font-weight:700; letter-spacing:-.02em; }
      .roadmap-card div { border-bottom:1px solid var(--line); color:#667085; font-size:12px; line-height:1.45; padding:10px 0; }
      .roadmap-card div:last-child { border-bottom:0; }
      .model-card { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); min-height:178px; padding:20px; transition:border-color .16s ease, transform .16s ease; }
      .model-card:hover { border-color:#cbd5e1; transform:translateY(-1px); }
      .model-card strong { color:var(--ink); display:inline-block; font-size:18px; font-weight:720; letter-spacing:-.04em; margin:12px 0 6px; }
      .model-card span:last-child { color:var(--muted); display:block; font-size:12px; line-height:1.65; }
      .architecture-card { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); min-height:138px; padding:20px; }
      .architecture-card strong { color:var(--ink); font-size:14px; }
      .architecture-card span { color:var(--muted); font-size:12px; line-height:1.65; }
      .architecture-join { color:var(--primary); font-size:24px; font-weight:400; padding-top:35px; text-align:center; }
      .top-brand { align-items:center; display:flex; gap:10px; min-height:46px; }
      .top-brand__symbol { align-items:center; background:linear-gradient(145deg, #8b5cf6, #4c1d95); border:1px solid rgba(255,255,255,.25); border-radius:12px; box-shadow:0 9px 22px rgba(74,35,147,.22); color:#fff; display:inline-flex; font-size:17px; height:38px; justify-content:center; transition:transform .22s cubic-bezier(.2,.9,.25,1), box-shadow .22s ease; width:38px; }
      .top-brand:hover .top-brand__symbol { box-shadow:0 13px 26px rgba(74,35,147,.3); transform:rotate(12deg) scale(1.08); }
      .top-brand strong { color:var(--ink); display:block; font-size:16px; font-weight:780; letter-spacing:-.06em; line-height:1; }
      .top-brand small { color:var(--muted); display:block; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:8px; font-weight:700; letter-spacing:.12em; margin-top:5px; }
      .workspace-rule { background:linear-gradient(90deg, #7c3aed 0%, #b99cf8 28%, rgba(185,156,248,.18) 72%, transparent 100%); height:1px; margin:14px 0 31px; }
      [data-testid="stMain"] [data-testid="stRadioGroup"] { align-items:center; display:flex; flex-direction:row; flex-wrap:wrap; gap:5px; justify-content:center; }
      [data-testid="stMain"] label[data-testid="stRadioOption"] { align-items:center; background:transparent; border:1px solid transparent; border-radius:999px; cursor:pointer; display:inline-flex; min-height:38px; padding:0 13px; transition:background .2s ease, border-color .2s ease, box-shadow .24s ease, transform .24s cubic-bezier(.2,.9,.25,1); }
      [data-testid="stMain"] label[data-testid="stRadioOption"]:hover { background:#efe9fc; border-color:#ded0fa; transform:translateY(-1px) scale(1.025); }
      [data-testid="stMain"] label[data-testid="stRadioOption"][data-selected="true"] { background:linear-gradient(135deg, #7c3aed, #5b21b6); border-color:#6d35d8; box-shadow:0 8px 18px rgba(103,56,192,.25); transform:translateY(-2px) scale(1.055); }
      [data-testid="stMain"] label[data-testid="stRadioOption"] p { color:#655b75; font-size:12px; font-weight:670; letter-spacing:-.018em; }
      [data-testid="stMain"] label[data-testid="stRadioOption"][data-selected="true"] p { color:#fff; }
      [data-testid="stMain"] label[data-testid="stRadioOption"] > div > div > div:first-child { display:none; }
      [data-testid="stPopover"] > button { align-items:center; background:#fff; border:1px solid #d9cff0; border-radius:10px; color:#4c3572; display:flex; font-size:12px; font-weight:720; justify-content:center; min-height:40px; padding:0 13px; transition:border-color .2s ease, box-shadow .2s ease, transform .2s cubic-bezier(.2,.9,.25,1); width:100%; }
      [data-testid="stPopover"] > button:hover, [data-testid="stPopover"] > button[aria-expanded="true"] { border-color:#9f7aea; box-shadow:0 9px 20px rgba(94,56,170,.15); transform:translateY(-2px) scale(1.045); }
      [data-testid="stPopoverBody"] { background:#fff !important; border:1px solid #ded3f3 !important; border-radius:14px !important; box-shadow:0 20px 48px rgba(48,29,83,.2) !important; padding:7px 8px 13px !important; }
      .control-kicker { color:#7652ba; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:9px; font-weight:750; letter-spacing:.12em; margin:7px 4px 3px; }
      .control-copy { color:var(--muted); font-size:11px; line-height:1.5; margin:0 4px 13px; }
      [data-testid="stPopoverBody"] [data-testid="stWidgetLabel"] p { color:#61576f; font-size:11px; font-weight:680; }
      [data-testid="stPopoverBody"] hr { border-color:#eee8f7; margin:15px 0; }
      [data-testid="stPopoverBody"] [data-testid="stSlider"] { padding:0 4px; }
      .stButton > button:active, [data-testid="stPopover"] > button:active { transform:translateY(1px) scale(.97); }
      /* Editorial product-site direction: generous white space, strong type,
         and a single feature canvas rather than a dashboard full of boxes. */
      .stApp { background:#fff; }
      .block-container { max-width:1380px; padding:24px 42px 84px; }
      .workspace-rule { background:#16121d; margin:16px 0 0; opacity:.92; }
      .top-brand__symbol { background:#17121e; border-radius:8px; box-shadow:none; }
      .top-brand:hover .top-brand__symbol { box-shadow:0 9px 18px rgba(44,24,78,.18); }
      .top-brand strong { font-size:17px; }
      .top-brand small { color:#736b7f; }
      [data-testid="stMain"] label[data-testid="stRadioOption"][data-selected="true"] { background:#17121e; border-color:#17121e; box-shadow:none; }
      [data-testid="stPopover"] > button { border-color:#1b1622; border-radius:0; color:#17121e; }
      [data-testid="stPopover"] > button:hover, [data-testid="stPopover"] > button[aria-expanded="true"] { border-color:#6d36ca; box-shadow:0 7px 0 rgba(109,54,202,.15); }
      .hero { background:transparent; border:0; border-radius:0; color:#17121e; margin:0; overflow:visible; padding:82px 18px 64px; text-align:center; }
      .hero:before, .hero:after { display:none; }
      .eyebrow { color:#6937c8; font-size:11px; letter-spacing:.11em; margin-bottom:18px; }
      .hero h1 { color:#17121e; font-size:clamp(38px, 5vw, 68px); font-weight:790; letter-spacing:-.078em; line-height:1.05; margin:0 auto 20px; max-width:940px; }
      .hero p { color:#756e7b; font-size:15px; line-height:1.72; margin:0 auto; max-width:620px; }
      .decision-canvas { background:#191225; color:#fff; display:grid; grid-template-columns:1.08fr .92fr; margin:0 0 64px; min-height:358px; overflow:hidden; position:relative; }
      .decision-canvas:before { background:radial-gradient(circle, rgba(148,99,255,.74) 0%, rgba(148,99,255,0) 68%); content:""; height:470px; position:absolute; right:-105px; top:-188px; width:470px; }
      .decision-canvas__copy { align-items:flex-start; display:flex; flex-direction:column; justify-content:space-between; padding:36px 38px; position:relative; z-index:1; }
      .decision-canvas__label { color:#c7adff; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:750; letter-spacing:.12em; }
      .decision-canvas h2 { font-size:clamp(33px, 4vw, 54px); font-weight:760; letter-spacing:-.07em; line-height:.98; margin:24px 0 15px; }
      .decision-canvas p { color:#d8cee8; font-size:14px; line-height:1.65; margin:0; max-width:470px; }
      .decision-canvas__tag { border:1px solid rgba(255,255,255,.26); color:#f5efff; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; letter-spacing:.07em; padding:8px 10px; }
      .decision-canvas__signal { align-items:center; display:flex; justify-content:center; padding:28px; position:relative; z-index:1; }
      .signal-orbit { align-items:center; border:1px solid rgba(224,209,255,.32); border-radius:50%; display:flex; height:238px; justify-content:center; position:relative; width:238px; }
      .signal-orbit:before, .signal-orbit:after { border:1px solid rgba(224,209,255,.22); border-radius:50%; content:""; position:absolute; }
      .signal-orbit:before { height:174px; width:174px; }
      .signal-orbit:after { height:304px; width:304px; }
      .signal-core { align-items:center; background:#fff; border-radius:50%; box-shadow:0 0 0 15px rgba(190,155,255,.16), 0 24px 48px rgba(0,0,0,.24); color:#1b1226; display:flex; flex-direction:column; height:106px; justify-content:center; position:relative; width:106px; }
      .signal-core strong { font-size:24px; font-weight:780; letter-spacing:-.07em; }
      .signal-core span { color:#705b86; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:8px; font-weight:700; letter-spacing:.08em; margin-top:4px; }
      .signal-dot { background:#b993ff; border-radius:50%; box-shadow:0 0 0 6px rgba(185,147,255,.12); height:10px; position:absolute; right:24px; top:52px; width:10px; }
      .section-title { font-size:28px; font-weight:760; letter-spacing:-.06em; margin-bottom:8px; }
      .section-sub { color:#7a7380; font-size:14px; margin-bottom:26px; }
      .metric-box { background:#f7f6f8; border:0; border-radius:0; min-height:136px; padding:20px; }
      .metric-box:hover { box-shadow:none; transform:translateY(-4px); }
      .metric-kicker { color:#7153a9; }
      .metric-value { font-size:27px; }
      .recommendation { background:#f0eaff; border:0; border-left:4px solid #6430c7; border-radius:0; box-shadow:none; padding:29px; }
      .recommendation:after { background:rgba(106,52,200,.12); }
      .recommendation h2 { color:#1b102c; font-size:32px; }
      .recommendation p { color:#5c4d70; }
      .recommendation .badge { background:#fff; border-color:#d6c3f8; color:#5422ad; }
      .evidence { border-left-color:#221631; padding:13px 0 13px 16px; }
      .evidence:hover { border-left-color:#6834cb; }
      .evidence strong { font-size:14px; }
      div[data-testid="stMetric"] { border-bottom-color:#17121e; }
      div[data-testid="stMetric"]:first-of-type { border-top-color:#17121e; }
      [data-testid="stPlotlyChart"] { border-color:#e5e1e9; border-radius:0; padding:8px; }
      .model-card { border-radius:0; }
      .roadmap-card, .architecture-card { border-radius:0; }
      .decision-window { margin:0 auto; max-width:1040px; padding:64px 0 48px; }
      .decision-window__eyebrow { color:#6937c8; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; font-weight:750; letter-spacing:.13em; text-align:center; }
      .decision-window h1 { color:#17121e; font-size:clamp(40px, 5.4vw, 70px); font-weight:790; letter-spacing:-.08em; line-height:1.04; margin:18px auto; max-width:900px; text-align:center; }
      .decision-window__copy { color:#756e7b; font-size:15px; line-height:1.7; margin:0 auto 42px; max-width:620px; text-align:center; }
      .intake-panel { background:#f7f6f8; border-top:2px solid #17121e; padding:29px; }
      .intake-step { color:#6d36ca; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:750; letter-spacing:.12em; margin-bottom:8px; }
      .intake-heading { color:#1c1624; font-size:21px; font-weight:760; letter-spacing:-.05em; margin-bottom:22px; }
      .intake-rule { border:0; border-top:1px solid #ded9e4; margin:25px 0; }
      .intake-note { color:#776d83; font-size:12px; line-height:1.6; margin-top:10px; }
      .decision-summary { background:#1b1328; color:#fff; margin:0 auto; max-width:1120px; padding:30px; }
      .decision-summary__label { color:#c9b3ff; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:700; letter-spacing:.12em; }
      .decision-summary h2 { font-size:clamp(30px, 4vw, 49px); font-weight:770; letter-spacing:-.07em; line-height:1.05; margin:13px 0 10px; }
      .decision-summary p { color:#ddd3ea; font-size:14px; line-height:1.65; margin:0; }
      .paper-section { border-top:1px solid #1d1625; margin-top:30px; padding-top:18px; }
      .paper-section__heading { align-items:baseline; display:flex; gap:12px; justify-content:space-between; margin-bottom:13px; }
      .paper-section__heading strong { color:#1b1328; font-size:20px; font-weight:760; letter-spacing:-.05em; }
      .paper-section__heading span { color:#776d83; font-size:12px; }
      .paper-grid { display:grid; gap:1px; grid-template-columns:repeat(2, minmax(0,1fr)); background:#ded9e4; border:1px solid #ded9e4; }
      .paper-card { background:#fff; min-height:180px; padding:19px; transition:background .2s ease, transform .2s ease; }
      .paper-card:hover { background:#faf7ff; transform:translateY(-3px); }
      .paper-venue { color:#6b32cc; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:750; letter-spacing:.08em; }
      .paper-title { color:#1d1625; font-size:15px; font-weight:730; letter-spacing:-.03em; line-height:1.35; margin:10px 0 9px; }
      .paper-summary { color:#716877; font-size:12px; line-height:1.55; margin:0; }
      .paper-links { display:flex; flex-wrap:wrap; gap:12px; margin-top:15px; }
      .paper-links a { border-bottom:1px solid #1d1625; color:#1d1625; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:750; letter-spacing:.05em; padding-bottom:2px; text-decoration:none; }
      .paper-links a:hover { color:#6d36ca; border-color:#6d36ca; }
      .recommendation-actions { display:flex; gap:10px; margin:24px auto 0; max-width:1120px; }
      .recommendation-actions [data-testid="stButton"] { flex:1; }
      .recommendation-actions .stButton > button { border-radius:0; width:100%; }
      .start-stage { align-items:center; background:#100a18; color:#fff; display:flex; flex-direction:column; justify-content:center; margin:0 calc(50% - 50vw) 0; min-height:calc(100vh - 8px); overflow:hidden; padding:70px 24px; position:relative; text-align:center; }
      .start-stage:before { background:radial-gradient(circle at 50% 42%, rgba(155,93,255,.8), rgba(155,93,255,.12) 22%, transparent 52%); content:""; height:1200px; left:50%; position:absolute; top:50%; transform:translate(-50%,-50%); width:1200px; }
      .start-stage:after { animation:signal-scan 4.2s linear infinite; background:linear-gradient(90deg, transparent, rgba(215,182,255,.7), transparent); content:""; height:1px; left:-20%; position:absolute; top:0; width:140%; }
      .start-stage__grid { background-image:linear-gradient(rgba(217,191,255,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(217,191,255,.08) 1px, transparent 1px); background-size:64px 64px; inset:0; mask-image:radial-gradient(ellipse at center, black, transparent 71%); opacity:.36; position:absolute; }
      .signal-logo { align-items:center; animation:brand-arrival 1.1s cubic-bezier(.16,1,.3,1) both; display:flex; flex-direction:column; position:relative; z-index:1; }
      .signal-logo__aperture { filter:drop-shadow(0 0 28px rgba(175,126,255,.34)); height:auto; margin-bottom:8px; overflow:visible; width:clamp(210px, 28vw, 360px); }
      .signal-logo__aperture .aperture-trace { animation:trace-draw 2.3s cubic-bezier(.25,.85,.3,1) infinite alternate; fill:none; stroke:url(#aperture-gradient); stroke-dasharray:360; stroke-dashoffset:0; stroke-linecap:round; stroke-width:6; }
      .signal-logo__aperture .aperture-trace--inner { animation-delay:.28s; opacity:.76; stroke-width:4; }
      .signal-logo__aperture .aperture-core { animation:core-breathe 2.1s ease-in-out infinite; fill:#f6f0ff; }
      .signal-logo__aperture .aperture-ring { animation:ring-turn 6s linear infinite; fill:none; stroke:rgba(225,207,255,.55); stroke-width:2; transform-origin:150px 126px; }
      .signal-logo__name { color:#fff; font-size:clamp(34px, 5vw, 52px); font-weight:800; letter-spacing:-.08em; line-height:1; margin-top:14px; }
      .signal-logo__sub { color:#d9c7ff; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:clamp(10px, 1.4vw, 14px); font-weight:700; letter-spacing:.42em; margin:16px 0 15px; padding-left:.42em; }
      .start-stage__statement { color:#eee7f8; font-size:clamp(16px, 2vw, 22px); letter-spacing:-.035em; line-height:1.5; margin:0; max-width:540px; position:relative; z-index:1; }
      .start-stage__cue { animation:cue-pulse 1.8s ease-in-out infinite; bottom:32px; color:#bda3ef; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:700; letter-spacing:.13em; position:absolute; }
      .unknown-choice { background:#f0ebf8; border-left:3px solid #6b32cc; color:#4c4059; font-size:12px; line-height:1.55; margin:0 0 17px; padding:11px 13px; }
      .unknown-choice [data-testid="stCheckbox"] { margin-bottom:0; }
      .unknown-choice [data-testid="stCheckbox"] label p { color:#2b2037; font-size:12px; font-weight:720; }
      @keyframes brand-arrival { from { opacity:0; transform:scale(.82) translateY(20px); } to { opacity:1; transform:scale(1) translateY(0); } }
      @keyframes trace-draw { from { opacity:.3; stroke-dashoffset:215; } to { opacity:1; stroke-dashoffset:0; } }
      @keyframes core-breathe { 0%,100% { filter:drop-shadow(0 0 0 rgba(220,190,255,0)); transform:scale(.92); transform-origin:150px 126px; } 50% { filter:drop-shadow(0 0 12px rgba(226,200,255,.9)); transform:scale(1.08); transform-origin:150px 126px; } }
      @keyframes ring-turn { to { transform:rotate(360deg); } }
      @keyframes signal-scan { 0% { top:-4%; opacity:0; } 15% { opacity:1; } 86% { opacity:1; } 100% { top:104%; opacity:0; } }
      @keyframes cue-pulse { 0%,100% { opacity:.45; transform:translateY(0); } 50% { opacity:1; transform:translateY(-5px); } }
      .app-intro { align-items:center; animation:intro-out .7s ease 1.6s forwards; background:radial-gradient(circle at 68% 28%, #5d3693 0%, #241437 42%, #120d20 100%); color:#fff; display:flex; flex-direction:column; inset:0; justify-content:center; padding:24px; pointer-events:none; position:fixed; text-align:center; z-index:999999; }
      .app-intro__halo { animation:halo-pulse 1.6s ease-in-out infinite; border:1px solid rgba(222,208,255,.38); border-radius:50%; height:146px; position:absolute; width:146px; }
      .app-intro__mark { animation:intro-rise .55s cubic-bezier(.2,.8,.2,1) both; background:rgba(237,229,255,.12); border:1px solid rgba(237,229,255,.28); border-radius:18px; box-shadow:0 20px 48px rgba(0,0,0,.22); color:#fff; font-size:24px; font-weight:760; letter-spacing:-.06em; padding:18px 20px; }
      .app-intro__copy { animation:intro-rise .55s .14s cubic-bezier(.2,.8,.2,1) both; color:#ded3f4; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:10px; font-weight:650; letter-spacing:.16em; margin-top:18px; }
      .app-intro__progress { animation:progress-grow 1.18s .24s cubic-bezier(.3,.9,.3,1) both; background:#c5b3ff; border-radius:99px; height:2px; margin-top:17px; transform-origin:left; width:124px; }
      @keyframes intro-rise { from { opacity:0; transform:translateY(12px) scale(.98); } to { opacity:1; transform:translateY(0) scale(1); } }
      @keyframes intro-out { to { opacity:0; visibility:hidden; } }
      @keyframes halo-pulse { 0%,100% { opacity:.48; transform:scale(.82); } 50% { opacity:.12; transform:scale(1.48); } }
      @keyframes progress-grow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
      @keyframes hero-enter { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
      @keyframes card-enter { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
      @media (prefers-reduced-motion: reduce) { *, *:before, *:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; } }
      @media (max-width: 900px) {
        .block-container { padding:24px 28px 64px; }
        .hero { margin-bottom:34px; padding:30px; }
        .decision-canvas { grid-template-columns:1fr; }
        .decision-canvas__copy { min-height:255px; }
        .decision-canvas__signal { min-height:210px; padding:0 28px 34px; }
      }
      @media (max-width: 768px) {
        .block-container { padding:20px 16px 44px; }
        [data-testid="stSidebar"] { min-width:min(304px, 84vw) !important; max-width:84vw !important; }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:24px 14px 32px; }
        .hero { border-radius:12px; margin-bottom:30px; padding:25px 20px; }
        .hero h1 { font-size:39px; line-height:1.08; }
        .hero p { font-size:13px; line-height:1.65; }
        .section-title { font-size:20px; }
        .section-sub { margin-bottom:16px; }
        .metric-box { min-height:auto; padding:16px; }
        .recommendation { padding:21px 19px; }
        .recommendation h2 { font-size:25px; }
        .main [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:12px !important; }
        .main [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex:1 1 100% !important; min-width:100% !important; width:100% !important; }
        [data-testid="stPlotlyChart"] { padding:2px; }
        .roadmap-card { min-height:auto; }
        .architecture-join { padding:0; text-align:left; }
        .top-brand { margin-top:2px; }
        [data-testid="stMain"] [data-testid="stRadioGroup"] { justify-content:flex-start; overflow-x:auto; padding:2px 0 7px; scrollbar-width:none; flex-wrap:nowrap; }
        [data-testid="stMain"] label[data-testid="stRadioOption"] { flex:none; padding:0 12px; }
        [data-testid="stPopover"] > button { justify-content:flex-start; }
        .decision-canvas { margin-bottom:42px; }
        .decision-canvas__copy { min-height:240px; padding:27px 24px; }
        .decision-canvas h2 { font-size:40px; }
        .signal-orbit { height:188px; width:188px; }
        .signal-orbit:before { height:136px; width:136px; }
        .signal-orbit:after { height:244px; width:244px; }
        .decision-window { padding:38px 0 34px; }
        .decision-window h1 { font-size:43px; }
        .intake-panel { padding:21px 17px; }
        .paper-grid { grid-template-columns:1fr; }
        .paper-section__heading { align-items:flex-start; flex-direction:column; gap:4px; }
        .recommendation-actions { flex-direction:column; }
        .start-stage { min-height:90vh; padding:54px 18px; }
        .signal-logo__sub { letter-spacing:.25em; padding-left:.25em; }
      }
    </style>
    """, unsafe_allow_html=True)


def checkpoint_for(percent: int) -> int:
    return max(q for q in CHECKPOINTS if q <= percent)


def next_checkpoint(q: int) -> int | None:
    return next((v for v in CHECKPOINTS if v > q), None)


def availability(percent: int, vram: float, channels: int | None = None) -> tuple[list[str], list[str]]:
    viable, blocked = [], []
    for name in ("MWVAR", "GDN", "PaAno", "TSPulse"):
        item = MODEL_DATA[name]
        if percent < item["min_q"]:
            blocked.append(f"{name}: 검증된 데이터 단계({item['min_q']}%) 미도달")
        elif item["gpu_gib"] > vram:
            blocked.append(f"{name}: 현재 recipe의 peak GPU memory {item['gpu_gib']:.2f} GiB 필요")
        else:
            viable.append(name)
    if channels is not None and channels < 8:
        blocked.append("ALoRa: 현재 recipe는 channel 8개 미만에서 공통 후보로 비교할 수 없음")
    return viable, blocked


def choose_model(viable: list[str], perf_weight: int, realtime: bool) -> str:
    if not viable:
        return "MWVAR"
    # Normalize only across the candidates currently under consideration.  This
    # prevents the 1–5 operational cost index from numerically overpowering VUS-PR.
    raw_performance = {
        name: OBSERVED_VUS.get(name, {}).get(max(OBSERVED_VUS.get(name, {0: 0})), 0.20)
        for name in viable
    }
    low, high = min(raw_performance.values()), max(raw_performance.values())
    score = {}
    for name in viable:
        item = MODEL_DATA[name]
        performance = 1.0 if high == low else (raw_performance[name] - low) / (high - low)
        cost_score = 1 - ((item["cost"] - 1) / 4)
        latency_score = 0.25 if realtime and item["latency"] == "높음" else 1
        score[name] = (performance * perf_weight / 100 + cost_score * (100 - perf_weight) / 100) * latency_score
    return max(score, key=score.get)


def stage_label(q: int) -> str:
    if q <= 5:
        return "Cold Start"
    if q < 40:
        return "Early Learning"
    return "Scale & Re-evaluate"


def hero(title: str, copy: str, eyebrow: str = "ENTERPRISE ANOMALY DECISIONING") -> None:
    st.markdown(f"<div class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{copy}</p></div>", unsafe_allow_html=True)


def show_intro() -> None:
    """Render the brand entrance once per browser session without touching app logic."""
    if "intro_seen" not in st.session_state:
        st.markdown(
            "<div class='app-intro'><div class='app-intro__halo'></div>"
            "<div class='app-intro__mark'>◈ TSAD</div>"
            "<div class='app-intro__copy'>DECISION STUDIO</div>"
            "<div class='app-intro__progress'></div></div>",
            unsafe_allow_html=True,
        )
        st.session_state.intro_seen = True


def paper_keys_for(model: str) -> list[str]:
    """Return modelling references that explain, rather than replace, the model choice."""
    if model == "MWVAR":
        return ["baseline", "timeinf"]
    if model in {"GDN", "PaAno"}:
        return ["catch", "sarad"]
    return ["units", "timeinf"]


def research_panel(model: str, label: str = "추천 모델군을 이해하는 연구 근거") -> None:
    cards = []
    for key in paper_keys_for(model):
        paper = PAPER_LIBRARY[key]
        links = f"<a href='{paper['pdf']}' target='_blank' rel='noopener'>PDF 열기 ↗</a><a href='{paper['paper']}' target='_blank' rel='noopener'>공식 페이지 ↗</a>"
        if paper["code"]:
            links += f"<a href='{paper['code']}' target='_blank' rel='noopener'>코드 ↗</a>"
        cards.append(
            f"<article class='paper-card'><div class='paper-venue'>{paper['venue']}</div>"
            f"<div class='paper-title'>{paper['title']}</div><p class='paper-summary'>{paper['summary']}</p>"
            f"<div class='paper-links'>{links}</div></article>"
        )
    st.markdown(
        f"<section class='paper-section'><div class='paper-section__heading'><strong>{label}</strong>"
        f"<span>논문은 추천 모델과 같은 계열의 참고 근거이며, 해당 구현 자체를 뜻하지 않습니다.</span></div>"
        f"<div class='paper-grid'>{''.join(cards)}</div></section>",
        unsafe_allow_html=True,
    )


def default_profile() -> dict[str, object]:
    return {
        "normal_rows": 27_000, "target_rows": 100_000, "channels": 18,
        "frequency": "1 min", "retention": "12 months", "realtime": True,
        "latency": "1 min 이내", "retraining": "월 1회", "cpu": 4, "ram": 16,
        "vram": 16.0, "daily_rows": 100_000, "perf_weight": 70,
        "acquisition_per_10k": 300_000, "gpu_hourly": 4_000,
        "upload_note": "파일 미업로드 · 입력값 기반 프로파일", "data_known": True, "infra_known": True,
    }


def decision_window() -> None:
    """First screen: collect a company profile, then expose the model and its research evidence."""
    profile = st.session_state.setdefault("company_profile", default_profile())
    ready = st.session_state.get("profile_ready", False)
    if not ready:
        st.markdown(
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
            "<p class='decision-window__copy'>데이터 규모·채널·인프라를 입력하면 현재 가능한 모델군과 다음 재평가 시점, 그리고 그 추천을 읽는 데 필요한 최신 논문 근거를 함께 정리합니다.</p></section>",
            unsafe_allow_html=True,
        )

    if not ready:
        with st.form("company-decision-form", border=False):
            st.markdown("<section class='intake-panel'><div class='intake-step'>01 · DATA PROFILE</div><div class='intake-heading'>기업 데이터와 운영 조건</div>", unsafe_allow_html=True)
            st.markdown("<div class='unknown-choice'>정확한 데이터 규모를 아직 모르더라도 시작할 수 있습니다. 체크하면 실제 값으로 단정하지 않고, 보수적인 Cold Start 조건으로만 후보를 제안합니다.</div>", unsafe_allow_html=True)
            data_unknown = st.checkbox("기업 데이터·운영 조건을 아직 모르겠어요", value=not bool(profile.get("data_known", True)))
            upload = st.file_uploader("데이터셋 파일 업로드 (CSV, 선택)", type=["csv"], help="파일을 올리면 row·column 수를 참고합니다. 원본 파일은 이 prototype에서 저장하지 않습니다.")
            data_a, data_b, data_c = st.columns(3)
            normal_rows = data_a.number_input("현재 정상 데이터 rows", min_value=1_000, max_value=100_000_000, value=int(profile["normal_rows"]), step=1_000)
            target_rows = data_b.number_input("목표 정상 데이터 rows", min_value=1_000, max_value=100_000_000, value=int(profile["target_rows"]), step=1_000)
            channels = data_c.number_input("channel 수", min_value=1, max_value=10_000, value=int(profile["channels"]), step=1)
            frequency, retention, realtime = st.columns(3)
            frequency_value = frequency.text_input("데이터 frequency", value=str(profile["frequency"]), placeholder="예: 1 min")
            retention_value = retention.text_input("데이터 저장 기간", value=str(profile["retention"]), placeholder="예: 12 months")
            realtime_value = realtime.toggle("Real-time 탐지 필요", value=bool(profile["realtime"]))
            st.markdown("<hr class='intake-rule'><div class='intake-step'>02 · OPERATING CONSTRAINTS</div><div class='intake-heading'>현재 운영 인프라</div>", unsafe_allow_html=True)
            st.markdown("<div class='unknown-choice'>GPU·CPU·메모리 사양을 모르겠다면 체크하세요. GPU 없는 환경으로 가정해 실행 가능성이 확실한 모델군부터 제안합니다.</div>", unsafe_allow_html=True)
            infra_unknown = st.checkbox("현재 운영 인프라를 아직 모르겠어요", value=not bool(profile.get("infra_known", True)))
            infra_a, infra_b, infra_c, infra_d = st.columns(4)
            cpu = infra_a.number_input("CPU core", min_value=1, max_value=512, value=int(profile["cpu"]), step=1)
            ram = infra_b.number_input("RAM (GiB)", min_value=1, max_value=4096, value=int(profile["ram"]), step=1)
            vram = infra_c.number_input("GPU VRAM (GiB)", min_value=0.0, max_value=256.0, value=float(profile["vram"]), step=1.0)
            daily_rows = infra_d.number_input("하루 처리 rows", min_value=1_000, max_value=100_000_000, value=int(profile["daily_rows"]), step=1_000)
            latency, retraining, weight = st.columns(3)
            latency_value = latency.selectbox("허용 latency", ["1 sec 이내", "1 min 이내", "1 hour 이내", "Batch"], index=["1 sec 이내", "1 min 이내", "1 hour 이내", "Batch"].index(str(profile["latency"])))
            retraining_value = retraining.selectbox("재학습 주기", ["매일", "주 1회", "월 1회", "필요 시"], index=["매일", "주 1회", "월 1회", "필요 시"].index(str(profile["retraining"])))
            weight_value = weight.slider("성능 중요도", min_value=0, max_value=100, value=int(profile["perf_weight"]), help="높을수록 현재 Dev18 성능값에 더 큰 가중치를 둡니다.")
            data_cost, gpu_cost = st.columns(2)
            acquisition_value = data_cost.number_input("정상 데이터 1만 rows 확보비용 (₩)", min_value=0, max_value=100_000_000, value=int(profile["acquisition_per_10k"]), step=10_000)
            gpu_hourly_value = gpu_cost.number_input("GPU 시간당 비용 가정 (₩)", min_value=0, max_value=100_000, value=int(profile["gpu_hourly"]), step=500)
            st.markdown("<p class='intake-note'>비용과 재학습 시간은 현재 사용자가 입력한 가정과 Dev18 관측치만 사용합니다. GHL·HAI 외부 검증 및 실제 GPU-hour 결과는 연결 전까지 ‘추가 측정 필요’로 표시합니다.</p></section>", unsafe_allow_html=True)
            submitted = st.form_submit_button("기업 맞춤 제안 만들기", use_container_width=True)

        if submitted:
            upload_note = "파일 미업로드 · 입력값 기반 프로파일"
            if upload is not None:
                try:
                    uploaded_frame = pd.read_csv(upload)
                    channels = int(uploaded_frame.shape[1])
                    upload_note = f"{upload.name} · {uploaded_frame.shape[0]:,} rows · {uploaded_frame.shape[1]:,} columns"
                except Exception:
                    upload_note = f"{upload.name} · 구조를 읽지 못해 수동 입력값을 사용"
            if data_unknown:
                normal_rows, target_rows, channels = 5_000, 100_000, 8
                frequency_value, retention_value, realtime_value = "미확인", "미확인", False
                upload_note = "데이터 정보 미확인 · Cold Start 보수 가정(실제 기업 값 아님)"
            if infra_unknown:
                cpu, ram, vram, daily_rows = 1, 1, 0.0, 1_000
                latency_value, retraining_value = "Batch", "필요 시"
                upload_note += " · 인프라 정보 미확인(GPU 없음 가정)"
            st.session_state.company_profile = {
                "normal_rows": int(normal_rows), "target_rows": int(max(target_rows, normal_rows)), "channels": int(channels),
                "frequency": frequency_value, "retention": retention_value, "realtime": realtime_value,
                "latency": latency_value, "retraining": retraining_value, "cpu": int(cpu), "ram": int(ram),
                "vram": float(vram), "daily_rows": int(daily_rows), "perf_weight": int(weight_value),
                "acquisition_per_10k": int(acquisition_value), "gpu_hourly": int(gpu_hourly_value),
                "upload_note": upload_note, "data_known": not data_unknown, "infra_known": not infra_unknown,
            }
            st.session_state.profile_ready = True
            st.rerun()
        return

    percent = min(100, round(int(profile["normal_rows"]) / int(profile["target_rows"]) * 100))
    q = checkpoint_for(max(5, percent))
    viable, blocked = availability(q, float(profile["vram"]), int(profile["channels"]))
    recommended = choose_model(viable, int(profile["perf_weight"]), bool(profile["realtime"]))
    family = "경량·통계 기반" if recommended == "MWVAR" else ("target 학습형 다변량" if recommended in {"GDN", "PaAno"} else "zero-shot / foundation")
    next_q = next_checkpoint(q)
    st.markdown(
        f"<section class='decision-summary'><div class='decision-summary__label'>YOUR PROPOSAL · {q}% REFERENCE BAND</div>"
        f"<h2>{family} · {recommended}</h2><p>{profile['normal_rows']:,} 정상 rows · {profile['channels']} channels · {profile['vram']:g} GiB VRAM · {profile['latency']} 조건을 기준으로 제안했습니다. "
        f"다음 검토 지점은 {next_q if next_q else '최종'}%입니다. {profile['upload_note']}</p></section>",
        unsafe_allow_html=True,
    )
    st.caption(f"유사 실험 구간: 현재 입력은 Dev18의 {q}% checkpoint와 가장 가깝습니다. GHL·HAI의 row·channel·frequency별 외부 검증 결과는 연결 후 같은 방식으로 함께 매칭합니다.")
    if blocked:
        st.markdown("<div class='warn'><b>현재 제외 또는 검토 필요</b><br>" + "<br>".join(f"• {reason}" for reason in blocked) + "</div>", unsafe_allow_html=True)
    research_panel(recommended, f"{recommended} 제안을 읽는 대표 논문")
    st.markdown("<div class='recommendation-actions'>", unsafe_allow_html=True)
    open_studio, revise = st.columns(2)
    with open_studio:
        if st.button("맞춤 Decision Studio 열기", type="primary", use_container_width=True):
            st.session_state.app_view = "studio"
            st.rerun()
    with revise:
        if st.button("입력값 다시 보기", use_container_width=True):
            st.session_state.profile_ready = False
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def dashboard(percent: int, target_rows: int, vram: float, perf_weight: int, realtime: bool, daily_rows: int, acquisition_per_10k: float, gpu_hourly: float, channels: int | None = None) -> None:
    q = checkpoint_for(percent)
    upcoming = next_checkpoint(q)
    viable, blocked = availability(q, vram, channels)
    recommendation = choose_model(viable, perf_weight, realtime)
    current_rows = round(target_rows * percent / 100)
    to_next_rows = 0 if upcoming is None else max(0, round(target_rows * (upcoming - percent) / 100))
    acq_cost = to_next_rows / 10000 * acquisition_per_10k
    rec_item = MODEL_DATA[recommendation]
    performance = OBSERVED_VUS.get(recommendation, {}).get(q)

    hero("운영 조건을 선택하고, 다음 의사결정 지점을 찾으세요.", "Dev18 결과를 기업 환경 입력과 함께 해석합니다. 추천은 자동 교체 지시가 아니라, 다음 검증·전환 논의를 위한 설명 가능한 후보입니다.")
    st.markdown(
        f"<section class='decision-canvas'>"
        f"<div class='decision-canvas__copy'><div><div class='decision-canvas__label'>YOUR DECISION CANVAS</div>"
        f"<h2>{recommendation}<br>at {q}%.</h2><p>현재 데이터 단계와 인프라 조건이 교차하는 지점을 읽어, 다음 검증에서 우선 볼 모델 후보를 제안합니다.</p></div>"
        f"<span class='decision-canvas__tag'>NEXT REVIEW · {upcoming if upcoming else 'FINAL'}%</span></div>"
        f"<div class='decision-canvas__signal'><div class='signal-orbit'><span class='signal-dot'></span><div class='signal-core'><strong>{q}%</strong><span>CHECKPOINT</span></div></div></div>"
        f"</section>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-title'>현재 운영 상태</div><p class='section-sub'>입력된 정상 데이터와 인프라 조건을 실험의 공통 비교 지점에 매핑했습니다.</p>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    for col, kicker, value, note in [
        (a, "normal data", f"{current_rows:,} rows", f"목표 {target_rows:,} rows · {percent}%"),
        (b, "decision stage", f"{q}% · {stage_label(q)}", "가장 가까운 검증 완료 checkpoint"),
        (c, "next review", f"{upcoming if upcoming else '—'}%", "자동 교체가 아닌 재평가 시점"),
        (d, "environment", f"{vram:g} GiB VRAM", "현재 recipe 기준 실행 가능성"),
    ]:
        col.markdown(f"<div class='metric-box'><div class='metric-kicker'>{kicker}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)

    left, right = st.columns([1.25, .75], gap="large")
    with left:
        score_text = f"Dev18 VUS-PR {performance:.3f}" if performance is not None else "해당 checkpoint의 수치 입력 예정"
        st.markdown(f"<div class='recommendation'><span class='badge'>{rec_item['tier']} · {rec_item['status']}</span><h2>Recommended: {recommendation}</h2><p>{score_text} · {rec_item['desc']}</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Why this candidate</div>", unsafe_allow_html=True)
        for text in [
            f"데이터 조건: {recommendation}은 현재 실험 구조에서 {rec_item['min_q']}%부터 공통 비교 가능합니다.",
            f"자원 조건: peak GPU memory {rec_item['gpu_gib']:.2f} GiB (현재 {vram:g} GiB).",
            "운영 조건: " + ("실시간 탐지로 설정되어 높은 latency 후보에는 감점을 적용했습니다." if realtime else "배치 탐지로 설정되어 latency 제약을 완화했습니다."),
        ]:
            st.markdown(f"<div class='evidence'><strong>{text}</strong></div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-title'>Investment preview</div><p class='section-sub'>기업 입력값에 따른 시나리오 추정치입니다.</p>", unsafe_allow_html=True)
        if upcoming:
            st.metric(f"{upcoming}%까지 추가 정상 데이터", f"{to_next_rows:,} rows")
            st.metric("데이터 확보비용 (입력 기반)", f"₩{acq_cost:,.0f}")
        else:
            st.metric("다음 checkpoint", "최종 구간")
        gpu_note = "실측 GPU-hour 입력 후 산출" if rec_item["gpu_gib"] else "GPU 사용 없음"
        st.metric("컴퓨팅 비용", gpu_note)
        st.caption(f"참고 GPU 시간당 요금: ₩{gpu_hourly:,.0f} (사용자 가정)")

    research_panel(recommendation, f"{recommendation} 추천과 함께 보는 대표 논문")

    st.markdown("<div class='section-title'>Data investment vs. model change</div><p class='section-sub'>같은 성능 변화라도 ‘데이터를 더 확보하는 경로’와 ‘모델을 바꾸는 경로’의 근거와 비용은 다르게 관리합니다.</p>", unsafe_allow_html=True)
    data_path, model_path = st.columns(2)
    with data_path:
        next_label = f"{upcoming}% checkpoint" if upcoming else "최종 checkpoint"
        st.markdown(f"<div class='architecture-card'><strong>A · 데이터 추가 확보</strong><br><span>{next_label}까지 약 {to_next_rows:,} 정상 rows를 추가 확보합니다.<br><br><b>입력 기반 투자 추정:</b> ₩{acq_cost:,.0f}<br><b>성능 근거:</b> 해당 구간의 관측 VUS-PR이 있는 모델만 비교합니다.</span></div>", unsafe_allow_html=True)
    with model_path:
        gain = None
        if performance is not None and q <= 20 and OBSERVED_VUS["PaAno"].get(40) is not None:
            gain = OBSERVED_VUS["PaAno"][40] - performance
        gain_line = f"후보 비교 ΔVUS-PR: +{gain:.3f} (Dev18 구간 간 참고치)" if gain is not None else "현재 수치만으로는 신뢰할 수 있는 ΔVUS-PR 산정이 어렵습니다."
        st.markdown(f"<div class='architecture-card'><strong>B · 모델 계열 전환</strong><br><span>재학습·검증·배포 비용은 별도 기록이 필요합니다.<br><br><b>{gain_line}</b><br>GPU-hour·train seconds·artifact size를 확보한 뒤 재학습 주기별 연간 비용으로 환산합니다.</span></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Model switching timeline</div><p class='section-sub'>각 점은 모델을 자동으로 바꾸라는 의미가 아니라, 새 후보를 검증할 가장 이른 시점입니다.</p>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=CHECKPOINTS, y=[1] * len(CHECKPOINTS), mode="lines+markers", line=dict(color="#bfd0df", width=2), marker=dict(color="#ffffff", size=13, line=dict(color="#607892", width=2),), hoverinfo="skip"))
    annotations = [(5, "TSPulse / MWVAR", "Cold start"), (10, "GDN 도입 검토", "Tier 2 available"), (40, "PaAno 전환 검토", "Re-evaluate")]
    for x, label, sub in annotations:
        fig.add_annotation(x=x, y=1, text=f"<b>{label}</b><br><span style='font-size:10px'>{sub}</span>", showarrow=False, yshift=43 if x != 10 else -44, font=dict(color="#17324d", size=12), align="center")
    fig.add_vline(x=q, line_color="#45d5b0", line_width=3)
    fig.add_annotation(x=q, y=1, text="현재", showarrow=False, yshift=-78, font=dict(color="#177f68", size=11))
    fig.update_layout(height=270, margin=dict(l=10, r=10, t=78, b=72), plot_bgcolor="#ffffff", paper_bgcolor="#f5f8fc", xaxis=dict(tickvals=CHECKPOINTS, ticksuffix="%", showgrid=False, zeroline=False), yaxis=dict(visible=False, range=[.3,1.7]), showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='section-title'>Unavailable / review required</div>", unsafe_allow_html=True)
    if blocked:
        st.markdown("<div class='warn'>" + "<br>".join(f"• {item}" for item in blocked) + "</div>", unsafe_allow_html=True)
    else:
        st.success("현재 4개 핵심 후보가 데이터·GPU 조건을 충족합니다. 실측 비용과 외부 검증 결과로 최종 선택을 확인하세요.")

    st.markdown("<div class='section-title'>Data availability → performance</div><p class='section-sub'>Zero-shot TSPulse는 데이터 확보율에 따른 성능 향상으로 해석하지 않도록 동일 수치를 반복 표시합니다.</p>", unsafe_allow_html=True)
    chart = go.Figure()
    for name, values in OBSERVED_VUS.items():
        xs, ys = zip(*sorted(values.items()))
        chart.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name, line=dict(color=PALETTE.get(name, "#52677b"), width=3), marker=dict(size=8)))
    chart.add_vline(x=q, line_dash="dot", line_color="#45d5b0")
    chart.update_layout(height=350, margin=dict(l=5, r=5, t=25, b=5), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", legend=dict(orientation="h", y=1.12), xaxis=dict(title="정상 데이터 확보율", ticksuffix="%", tickvals=CHECKPOINTS, gridcolor="#edf1f5"), yaxis=dict(title="VUS-PR", range=[.24,.35], gridcolor="#edf1f5"))
    st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    st.caption("PaAno 40% 값은 제공된 Dev18 brief의 전환 예시(0.331)를 사용한 후보 시각화입니다. 전체 checkpoint, seed variation, Family-LOFO는 결과 파일 연결 후 갱신해야 합니다.")


def portfolio() -> None:
    hero("모델 카드는 성능 주장과 운영 제약을 분리합니다.", "Tier는 비용 등급이 아니라 실험 구조입니다. Tier 1은 경량·비학습 기준선, Tier 2는 target 정상 데이터 학습, Tier 3는 target 학습 불필요 zero-shot입니다.", "MODEL PORTFOLIO")
    st.markdown("<div class='section-title'>Portfolio at a glance</div><p class='section-sub'>현재 측정된 자원 수치와 실험 recipe 제약을 함께 봅니다.</p>", unsafe_allow_html=True)
    rows = []
    for name, item in MODEL_DATA.items():
        rows.append({"Model": name, "Tier / role": item["tier"], "Common from": "—" if math.isinf(item["min_q"]) else f"{item['min_q']}%", "Peak GPU memory": "—" if item["gpu_gib"] is None else f"{item['gpu_gib']:.2f} GiB", "Training": item["training"], "Latency": item["latency"], "Evidence": item["status"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={"Model": st.column_config.TextColumn(width="medium"), "Tier / role": st.column_config.TextColumn(width="medium")})
    st.markdown("<div class='section-title'>Model cards</div>", unsafe_allow_html=True)
    cards = ["MWVAR", "GDN", "PaAno", "TSPulse", "TimeRCD", "ALoRa"]
    for first, second in zip(cards[::2], cards[1::2]):
        c1, c2 = st.columns(2)
        for col, name in [(c1, first), (c2, second)]:
            item = MODEL_DATA[name]
            gpu = "공통 후보 불가" if item["gpu_gib"] is None else ("GPU 불필요" if item["gpu_gib"] == 0 else f"peak {item['gpu_gib']:.2f} GiB")
            col.markdown(f"<div class='model-card'><span class='badge'>{item['tier']} · {item['status']}</span><br><strong>{name}</strong><span>{item['kind']} · {item['desc']}<br><br>학습: {item['training']} &nbsp; | &nbsp; GPU: {gpu} &nbsp; | &nbsp; Latency: {item['latency']}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Reproduction guidance</div><p class='section-sub'>아래는 현재 recipe를 다시 실행하기 위한 권장 환경이며, 제품 운영의 최소 사양으로 단정하지 않습니다.</p>", unsafe_allow_html=True)
    guidance = pd.DataFrame([
        {"Model": "MWVAR", "GPU": "불필요", "RAM": "8 GiB+", "CPU": "2 vCPU+", "학습": "불필요"},
        {"Model": "GDN", "GPU": "24 GiB VRAM급 권장", "RAM": "16 GiB+", "CPU": "4 vCPU+", "학습": "필요"},
        {"Model": "PaAno", "GPU": "peak 0.53 GiB 관측", "RAM": "16 GiB+", "CPU": "4 vCPU+", "학습": "필요"},
        {"Model": "TSPulse", "GPU": "peak 8.23 GiB 관측", "RAM": "추가 측정", "CPU": "추가 측정", "학습": "target 학습 불필요"},
    ])
    st.dataframe(guidance, hide_index=True, use_container_width=True)
    st.markdown("<div class='warn'><b>해석 가드레일</b><br>‘Tier 3 = 저비용’, ‘40% = PaAno의 실제 최소 데이터량’, ‘Dev18 우세 = 현장 일반화’는 현재 근거로 주장하지 않습니다. 각 항목은 본 실험의 비용·외부 검증으로 확인해야 합니다.</div>", unsafe_allow_html=True)


def evidence() -> None:
    hero("근거의 경계를 보여주는 것이 기업용 신뢰의 시작입니다.", "대시보드는 관측값, 사용자 입력, 그리고 추가 측정이 필요한 항목을 의도적으로 분리합니다.", "EVIDENCE & ROADMAP")
    cols = st.columns(3)
    sections = [
        ("01 · 지금 사용 가능", "roadmap-card--available", ["데이터 확보율별 선택 가능 모델", "Dev18 VUS-PR (제공된 수치)", "GPU peak memory", "모델 recipe · 채널 제약", "Cold start · switching timeline"]),
        ("02 · 본 실험 측정", "roadmap-card--measure", ["1K / 10K / 100K inference time", "checkpoint별 재학습 시간", "GPU-hour · CPU-hour · peak RAM", "checkpoint / artifact storage", "GHL25 · HAI external validation"]),
        ("03 · 방법론 확정", "roadmap-card--decide", ["GPU 시간당 비용 benchmark", "Data acquisition cost 입력 방식", "Cost index 정규화", "FP / FN threshold cost function", "전환 가치 및 latency 기준"]),
    ]
    for col, (title, tone, items) in zip(cols, sections):
        body = "".join(f"<div>{x}</div>" for x in items)
        col.markdown(f"<div class='roadmap-card {tone}'><strong>{title}</strong><div style='margin-top:12px'>{body}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Cost architecture</div><p class='section-sub'>서로 성격이 다른 비용을 하나의 근거 없는 숫자로 합치지 않습니다.</p>", unsafe_allow_html=True)
    a, arrow, b = st.columns([1,.15,1])
    with a:
        st.markdown("<div class='architecture-card'><strong>Computational cost</strong><br><span>연구에서 직접 측정: GPU / CPU 시간, 메모리, 추론 시간, storage.<br>→ cloud GPU benchmark로 비용 추정</span></div>", unsafe_allow_html=True)
    with arrow:
        st.markdown("<div class='architecture-join'>+</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='architecture-card'><strong>Data investment cost</strong><br><span>기업별 입력: 정상 운전 비용, 센서 운영, 저장, 기회비용.<br>→ 현재 데이터에서 다음 checkpoint까지의 투자 추정</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Service coverage</div><p class='section-sub'>웹서비스에서 바로 쓰는 판단과, 실험 결과가 연결되어야 확정되는 판단을 분리합니다.</p>", unsafe_allow_html=True)
    coverage = pd.DataFrame([
        {"기능": "CSV row · column 프로파일", "상태": "현재 사용", "처리": "업로드 메타데이터와 수동 정상 구간 입력"},
        {"기능": "GHL · HAI 유사 구간 매칭", "상태": "결과 연결 예정", "처리": "row · channel · frequency 기준의 가장 가까운 실험 조건"},
        {"기능": "VUS-PR · Tier · 전환 시점", "상태": "Dev18 사용", "처리": "관측값과 candidate milestone을 분리 표기"},
        {"기능": "Training / inference / 재학습 비용", "상태": "추가 측정", "처리": "GPU-hour · CPU-hour · peak RAM · artifact size"},
        {"기능": "FP / FN 사업 비용", "상태": "방법론 확정 필요", "처리": "threshold·오탐·미탐 비용을 기업별 입력으로 설계"},
        {"기능": "추천 논문 · PDF · 코드", "상태": "현재 사용", "처리": "추천 모델군과 함께 공식 링크를 제공"},
    ])
    st.dataframe(coverage, hide_index=True, use_container_width=True)
    st.markdown("<div class='section-title'>Recommended experimental log</div>", unsafe_allow_html=True)
    st.code("checkpoint, model, seed, normal_rows, train_seconds, inference_rows, inference_seconds, gpu_hours, cpu_hours, peak_vram_gib, peak_ram_gib, artifact_mb, vus_pr, family, run_id", language="text")
    st.caption("위 스키마의 CSV/JSONL을 연결하면, 현재 prototype의 정적 evidence 영역을 재현 가능한 실험 데이터로 교체할 수 있습니다.")


inject_css()
show_intro()

render_candidate_intake()
