"""정상 데이터 입력부터 구조적 후보 축소까지 표시한다."""

import io
import json
import sqlite3

import pandas as pd
import streamlit as st

from streamlit_website.db_connection.filter_candidates import (
    STATUS_LABELS, build_ml_input, filter_candidates, load_candidates, profile_normal_data,
)


def render_candidate_intake():
    st.title("운영 계획 후보 확인")
    st.caption("운영 기간 동안 같은 센서 구성을 쓴다고 가정하고, 고정된 입력 조건으로 전체 계획의 후보 풀을 만듭니다.")
    try:
        candidates = load_candidates()
    except sqlite3.Error as error:
        st.session_state.pop("ml_input", None)
        st.error(f"후보 DB를 읽을 수 없습니다: {error}")
        return
    st.caption(f"등록 후보 {len(candidates)}개 · ALoRa 제외 · 실행 이력이나 성능으로 후보를 미리 줄이지 않습니다.")
    st.caption("갱신할 때는 기존 데이터를 포함한 누적 정상 prefix 전체를 올려 주세요. 새 입력이 이전 입력을 대체합니다.")
    upload = st.file_uploader("누적 정상 prefix CSV", type=["csv"],
                              on_change=lambda: st.session_state.pop("ml_input", None),
                              help="업로드한 전체 구간을 사용합니다. 데이터는 파일로 저장하지 않고 현재 세션 메모리에만 둡니다.")
    if upload is None:
        return
    try:
        frame = pd.read_csv(io.BytesIO(upload.getvalue()))
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"CSV를 읽을 수 없습니다: {error}")
        return
    columns = [column for column in frame.columns if column.casefold() not in {"label", "labels", "anomaly", "라벨"}]
    defaults = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])
                and column.casefold() not in {"time", "timestamp", "date", "datetime"}]
    with st.form("structural-candidate-intake"):
        sensors = st.multiselect("운영 기간에 사용할 센서 열", columns, default=defaults,
                                 help="시간·식별자 열은 제외하고 센서만 선택하세요. 상수 센서는 자동으로 삭제하지 않습니다.")
        current_model = st.text_input("현재 사용 중인 모델 (선택)")
        current_settings = st.text_input("현재 모델 설정·config ID (선택)")
        current_checkpoint = st.text_input("현재 checkpoint 경로·ID (선택)")
        last_trained_at = st.text_input("마지막 학습 시점 (선택)", placeholder="2026-09-22 14:00 KST")
        st.subheader("운영 조건")
        first, second = st.columns(2)
        cpu = first.text_input("CPU·장비 이름 (선택)")
        gpu = second.text_input("GPU 이름 (없으면 비워 두세요)")
        cores, memory, gpu_memory = st.columns(3)
        cpu_cores = cores.number_input("CPU 코어 수", min_value=1, value=8, step=1)
        ram = memory.number_input("RAM (GiB)", min_value=0.0, value=16.0)
        vram = gpu_memory.number_input("VRAM (GiB)", min_value=0.0, value=0.0)
        metric, minimum, duration = st.columns(3)
        performance_metric = metric.selectbox("성능 하한 지표", ["VUS-PR", "Precision", "Recall", "F1"])
        performance_floor = minimum.number_input("성능 하한 (0~1)", min_value=0.0, max_value=1.0, value=0.8)
        operating_days = duration.number_input("운영 기간 (일)", min_value=1, value=365, step=1)
        collection, inference = st.columns(2)
        collection_rate = collection.number_input("예상 수집 속도 (행/초)", min_value=0.0, value=1.0)
        inference_rows = inference.number_input("예상 추론량 (행/일)", min_value=0, value=86400, step=1)
        inference_batch_length = st.number_input(
            "한 번에 모델이 보는 연속 추론 데이터 길이 (선택)", min_value=0, value=0, step=1,
            help="비워두면(0) ML 단계가 '예상 추론량 (행/일)'로 근사한다 — 이 근사치는 검증되지 않았다. "
                 "실제로 한 번에 모델에 들어가는 연속 구간 길이를 알고 있다면 여기에 입력하면 더 정확하다.",
        )
        submitted = st.form_submit_button("전체 계획 후보 풀 확인", use_container_width=True)
    if not submitted:
        return
    st.session_state.pop("ml_input", None)
    if frame.empty or not sensors:
        st.error("정상 데이터와 센서 열을 선택해 주세요.")
        return
    row_metric, sensor_metric = st.columns(2)
    row_metric.metric("현재 누적 행 수", len(frame))
    sensor_metric.metric("센서 수", len(sensors))
    rows = filter_candidates(candidates, channel_count=len(sensors))
    ml_input = build_ml_input(
        frame, sensors, rows,
        current_model={"model": current_model, "settings": current_settings,
                       "checkpoint": current_checkpoint, "last_trained_at": last_trained_at},
        operating_conditions={"cpu": cpu, "gpu": gpu, "cpu_cores": cpu_cores,
                              "ram_gib": ram, "vram_gib": vram, "performance_metric": performance_metric,
                              "performance_floor": performance_floor, "operating_days": operating_days,
                              "collection_rows_per_second": collection_rate, "inference_rows_per_day": inference_rows,
                              "inference_batch_length": inference_batch_length or None},
    )
    st.session_state["ml_input"] = ml_input
    st.subheader("전체 계획에서 검토할 후보 풀")
    for column, (status, label) in zip(st.columns(3), STATUS_LABELS.items()):
        column.metric(label, sum(row["status"] == status for row in rows))
    st.caption("현재 학습 길이·유효 창 수·상수 여부·분산·상관·결측으로 미래 후보를 제외하지 않습니다. "
               "후보 포함은 지금 실행 가능하다는 뜻이 아닙니다.")
    st.caption("실행 시점의 데이터 조건과 장비별 자원·처리량·성능 하한은 이후 단계에서 확인합니다.")
    possible, excluded = st.tabs(["계획 후보", "고정 제약 제외·미검증"])
    for tab, eligible in ((possible, True), (excluded, False)):
        selected = [row for row in rows if (row["status"] == "eligible") == eligible]
        with tab:
            if not selected:
                st.info("해당하는 후보가 없습니다.")
                continue
            st.dataframe(pd.DataFrame([{
                "모델": row["model"], "설정 ID": row["config_id"], "head": row["head"] or "기본",
                "상태": STATUS_LABELS[row["status"]], "window / patch / context": row["window"],
                "stride / block step": row["stride"],
                "판정 사유": row["reason"], "고정 설정": json.dumps(row["parameters"], ensure_ascii=False),
            } for row in selected]), hide_index=True, use_container_width=True)
    summary, channels = None, []
    with st.spinner("참고용으로 현재 데이터의 feature를 계산하고 있습니다."):
        try:
            summary, channels = profile_normal_data(frame, sensors)
        except (ValueError, TypeError) as error:
            st.warning(f"현재 feature를 계산하지 못했습니다. 계획 후보 풀은 유지합니다: {error}")
    ml_input["current_features"] = {"summary": summary, "channels": channels}

    # ML 단계 최소 연결점: build_ml_input() 다음 단계로 st.session_state["ml_input"]을
    # 그대로 run_ml_pipeline()에 넘긴다. UI는 의도적으로 최소화한다 (버튼 하나).
    # DP(경로 최적화)는 이 단계에서 하지 않는다 — run_ml_pipeline()의 반환값(MLOutput)까지만 만든다.
    st.divider()
    if st.button("ML 단계 실행 (similarity · 성능/비용 추정)"):
        with st.spinner("과거 유사 사례를 찾고 단계별 성능·비용을 추정하고 있습니다."):
            try:
                from streamlit_website.ml.pipeline import run_ml_pipeline
                ml_output = run_ml_pipeline(st.session_state["ml_input"])
            except FileNotFoundError as error:
                st.error(f"ML 단계용 DB를 읽을 수 없습니다: {error}")
            else:
                st.session_state["ml_output"] = ml_output
                for warning in ml_output.metadata.get("warnings", []):
                    st.warning(warning)
                st.success(
                    f"ML 단계 완료: similarity match {len(ml_output.similarity_matches)}건, "
                    f"미래 단계 {len(ml_output.future_stages)}개, "
                    f"checkpoint 유지 선택지 {len(ml_output.checkpoint_maintenance_options)}건. "
                    "결과는 st.session_state['ml_output']에 저장했습니다 (DP 단계가 여기서 이어받습니다)."
                )

    if summary is not None:
        st.subheader("현재 데이터의 feature (참고)")
        st.caption("데이터가 쌓이면 달라질 수 있는 현재 관측값이며, 1차 후보 축소에는 사용하지 않습니다.")
        labels = {
            "observed_row": "확보한 행 수", "input_column": "센서 수",
            "channel_std_median": "채널 표준편차 중앙값", "channel_interquartile_range_median": "채널 IQR 중앙값",
            "channel_acf_lag1_median": "자기상관 중앙값 (lag 1)",
            "absolute_correlation_median": "채널 간 절대 상관 중앙값",
            "channel_difference_q90_iqr_ratio_median": "변화량 Q90 / IQR 중앙값",
            "channel_median_shift_iqr_ratio_median": "앞·뒤 구간 중앙값 변화 / IQR 중앙값",
            "channel_spectral_entropy_median": "Spectral entropy 중앙값",
        }
        st.dataframe(pd.DataFrame([{"특징": label, "값": summary[key]} for key, label in labels.items()]),
                     hide_index=True, use_container_width=True)
    with st.expander("입력 조건과 feature 상세"):
        st.json({"입력": ml_input["input"], "현재 모델": ml_input["current_model"],
                 "운영 조건": ml_input["operating_conditions"]})
        if summary is not None:
            st.dataframe(pd.DataFrame(channels), hide_index=True, use_container_width=True)
            st.json(summary)
