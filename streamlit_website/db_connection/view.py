"""데이터 입력부터 후보 1차 축소, ML 추정까지 보여 준다."""

import io
import json
import sqlite3

import pandas as pd
import streamlit as st

from streamlit_website.db_connection.filter_candidates import (
    FEATURE_LABELS, STATUS_LABELS, build_ml_input, filter_candidates, load_candidates, profile_normal_data,
)
from streamlit_website.ml.dp_input import build_dp_input
from streamlit_website.ml.pipeline import run_ml_pipeline

LABEL_COLUMNS = {"label", "labels", "anomaly", "라벨"}
TIME_COLUMNS = {"time", "timestamp", "date", "datetime"}


def clear_handoff():
    for key in ("ml_input", "ml_output", "dp_input", "report_time"):
        st.session_state.pop(key, None)


def read_upload():
    upload = st.file_uploader("누적 정상 prefix CSV", type=["csv"], on_change=clear_handoff,
                              help="올린 전체 구간을 씁니다. 파일은 저장하지 않고 세션 메모리에만 둡니다.")
    if upload is None:
        return None, None
    try:
        return pd.read_csv(io.BytesIO(upload.getvalue())), upload.name
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"CSV를 읽을 수 없습니다: {error}")
        return None, None


def read_form(frame, candidates):
    """폼 값을 (센서 열, 현재 모델, 운영 조건)으로 돌려준다. 제출 전이면 None."""
    columns = [column for column in frame.columns if column.casefold() not in LABEL_COLUMNS]
    defaults = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])
                and column.casefold() not in TIME_COLUMNS]
    choices = {"": "없음"} | {f"{row['config_id']}::{row['head']}":
                               " · ".join(filter(None, (row["model"], row["config_id"], row["head"])))
                               for row in candidates}
    with st.form("candidate-intake"):
        sensors = st.multiselect("운영 기간에 사용할 센서 열", columns, default=defaults,
                                 help="시간·식별자 열은 빼고 센서만 고르세요. 상수 센서도 지우지 않습니다.")
        st.subheader("현재 모델 (있을 때만)")
        left, right = st.columns(2)
        candidate_id = left.selectbox("현재 사용 중인 후보", list(choices), format_func=choices.get)
        checkpoint = right.text_input("현재 checkpoint 경로·ID")
        trained_rows = left.number_input("현재 checkpoint 학습에 쓴 정상 행 수", min_value=0, value=0, step=1,
                                         help="0이면 지금 올린 행 수로 학습했다고 봅니다.")
        last_trained_at = right.text_input("마지막 학습 시점 (표시용)", placeholder="2026-09-22 14:00 KST")
        st.subheader("남은 운영 조건과 예산")
        first, second, third = st.columns(3)
        operating_days = first.number_input("남은 운영 기간 (일)", min_value=1, value=365, step=1)
        collection_rate = second.number_input("예상 수집 속도 (행/초)", min_value=0.0, value=1.0)
        inference_rows = third.number_input("예상 추론량 (행/일)", min_value=0, value=86400, step=1)
        budget = first.number_input("남은 예산 (0이면 제한 없음)", min_value=0.0, value=0.0,
                                    help="시간당 비용을 넣으면 원, 아니면 계산 시간(초) 단위입니다. 이미 쓴 비용은 빼고 넣으세요.")
        cpu_hourly_cost = second.number_input("CPU 1시간 비용 (원)", min_value=0.0, value=0.0)
        gpu_hourly_cost = third.number_input("GPU 1시간 비용 (원)", min_value=0.0, value=0.0)
        gpu_memory = first.number_input("GPU 메모리 (GiB, 없으면 0)", min_value=0.0, value=24.0,
                                        help="0이면 GPU에서만 잰 후보를 뺍니다. 비용은 L4 GPU·CPU 8개 기준 실행 시간입니다.")
        batch_length = second.number_input("한 번에 모델이 보는 연속 추론 데이터 길이 (선택)", min_value=0, value=0,
                                           step=1, help="0이면 예상 추론량 (행/일)으로 대신합니다.")
        if not st.form_submit_button("전체 계획 후보 풀 확인", use_container_width=True):
            return None
    current_model = {"candidate_id": candidate_id or None, "checkpoint": checkpoint,
                     "trained_rows": trained_rows or None, "last_trained_at": last_trained_at}
    conditions = {"operating_days": operating_days, "collection_rows_per_second": collection_rate,
                  "inference_rows_per_day": inference_rows, "inference_batch_length": batch_length or None,
                  "budget": budget or None, "cpu_hourly_cost": cpu_hourly_cost,
                  "gpu_hourly_cost": gpu_hourly_cost, "gpu_memory_gib": gpu_memory}
    return sensors, current_model, conditions


def show_pool(rows):
    st.subheader("전체 계획에서 검토할 후보 풀")
    for column, (status, label) in zip(st.columns(3), STATUS_LABELS.items()):
        column.metric(label, sum(row["status"] == status for row in rows))
    st.caption("지금 데이터 길이·통계·결측으로는 후보를 빼지 않습니다. 실행 가능 여부와 비용은 구간별로 다시 봅니다.")
    for tab, eligible in zip(st.tabs(["계획 후보", "고정 제약 제외·미검증"]), (True, False)):
        selected = [row for row in rows if (row["status"] == "eligible") == eligible]
        with tab:
            if not selected:
                st.info("해당하는 후보가 없습니다.")
                continue
            st.dataframe(pd.DataFrame([{
                "모델": row["model"], "설정 ID": row["config_id"], "head": row["head"] or "기본",
                "상태": STATUS_LABELS[row["status"]], "window / patch / context": row["window"],
                "stride / block step": row["stride"], "측정 장치": row.get("backend") or "-",
                "GPU 메모리 최대 (GiB)": row.get("gpu_peak_gib"), "판정 사유": row["reason"],
                "고정 설정": json.dumps(row["parameters"], ensure_ascii=False),
            } for row in selected]), hide_index=True, use_container_width=True)


def show_features(summary, channels, ml_input):
    st.subheader("현재 데이터의 feature")
    st.caption("데이터가 쌓이면 달라지는 값이며 후보 1차 축소에는 쓰지 않습니다.")
    st.dataframe(pd.DataFrame([{"특징": label, "값": summary[key]} for key, label in FEATURE_LABELS.items()]),
                 hide_index=True, use_container_width=True)
    with st.expander("입력 조건과 feature 상세"):
        st.json({"입력": ml_input["input"], "현재 모델": ml_input["current_model"],
                 "운영 조건": ml_input["operating_conditions"]})
        st.dataframe(pd.DataFrame(channels), hide_index=True, use_container_width=True)
        st.json(summary)


def show_estimates(ml_output):
    st.subheader("구간별 신규 도입·재학습 후보 (ML 추정)")
    st.caption("상단 '운영 경로'에서 이 추정표로 예산 안 경로를 계산하고, '보고서'에서 인쇄용 계획서를 봅니다. "
               "예상 VUS-PR의 절대값은 새 현장에서 ±0.3 정도 틀리므로 가장 싼 계획보다 얼마나 나은지로 읽어 주세요.")
    st.dataframe(pd.DataFrame([{
        "구간": estimate.stage + 1, "데이터 비율 (%)": ml_output.future_stages[estimate.stage].ratio_percent,
        "모델": estimate.model, "설정 ID": estimate.configuration, "head": estimate.head or "기본",
        "예상 VUS-PR": round(estimate.predicted_performance, 2),
        "학습비 (초)": estimate.estimated_training_cost_seconds,
        "구간 추론비 (초)": estimate.estimated_inference_cost_seconds,
    } for estimates in ml_output.stage_candidates.values() for estimate in estimates]),
        hide_index=True, use_container_width=True)
    with st.expander("ML 추정의 가정과 경고"):
        for warning in ml_output.metadata["warnings"] or ["따로 남길 경고가 없습니다."]:
            st.caption(warning)


def render_candidate_intake():
    st.title("운영 계획 후보 확인")
    try:
        candidates = load_candidates()
    except sqlite3.Error as error:
        clear_handoff()
        st.error(f"후보 DB를 읽을 수 없습니다: {error}")
        return
    st.caption(f"등록 후보 {len(candidates)}개 (ALoRa 제외). 운영 기간에 같은 센서 구성을 쓴다고 보고, "
               "갱신할 때는 기존 데이터를 포함한 누적 정상 prefix 전체를 올립니다.")
    frame, source_name = read_upload()
    if frame is None:
        return
    submitted = read_form(frame, candidates)
    if submitted is None:
        return
    sensors, current_model, conditions = submitted
    clear_handoff()
    if frame.empty or not sensors:
        st.error("정상 데이터와 센서 열을 선택해 주세요.")
        return
    rows_metric, sensors_metric = st.columns(2)
    rows_metric.metric("현재 누적 행 수", len(frame))
    sensors_metric.metric("센서 수", len(sensors))
    rows = filter_candidates(candidates, channel_count=len(sensors), gpu_available=conditions["gpu_memory_gib"] > 0)
    ml_input = build_ml_input(frame, sensors, rows, current_model=current_model, operating_conditions=conditions)
    ml_input["input"]["source_name"] = source_name
    st.session_state["ml_input"] = ml_input
    show_pool(rows)
    try:
        summary, channels = profile_normal_data(frame, sensors)
    except (ValueError, TypeError) as error:
        st.warning(f"현재 feature를 계산하지 못했습니다. 후보 풀은 유지합니다: {error}")
        return
    ml_input["current_features"] = {"summary": summary, "channels": channels}
    show_features(summary, channels, ml_input)
    with st.spinner("비슷한 과거 prefix로 구간별 성능·비용을 추정하고 있습니다."):
        try:
            ml_output = run_ml_pipeline(ml_input)
        except (OSError, ValueError, sqlite3.Error) as error:
            st.warning(f"구간별 성능·비용을 추정하지 못했습니다. 후보 풀은 유지합니다: {error}")
            return
    st.session_state["ml_output"] = ml_output
    st.session_state["dp_input"] = build_dp_input(ml_output, ml_input)
    show_estimates(ml_output)
