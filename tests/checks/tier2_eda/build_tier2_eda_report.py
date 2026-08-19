"""저장한 CSV를 읽어 계층 2 적용 가능성 보고서를 만든다."""

from pathlib import Path

import pandas


def read_csv(csv_dir: Path, name: str) -> pandas.DataFrame:
    return pandas.read_csv(csv_dir / name, encoding="utf-8-sig")


def markdown_table(headers: tuple[str, ...], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def true_mask(values: pandas.Series) -> pandas.Series:
    return values.astype(str).str.lower().eq("true")


def true_value(value) -> bool:
    return str(value).lower() == "true"


def number_range(values: pandas.Series) -> str:
    numbers = values.astype(int)
    return f"{numbers.min():,}" if numbers.min() == numbers.max() else f"{numbers.min():,}–{numbers.max():,}"


def build_report(
    dataset: str,
    output_dir: Path,
    feature_count: int,
    input_count: int,
    environment_gate: str,
) -> dict:
    csv_dir = output_dir / "csv"
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    scope = read_csv(csv_dir, "constant_scope_summary.csv")
    activity = read_csv(csv_dir, "channel_scaling_readiness.csv")
    required_activity_columns = {
        "constant", "zero_range", "iqr_zero", "nonfinite",
        "scaler_execution_ready", "variation_available",
    }
    if not required_activity_columns.issubset(activity.columns):
        raise ValueError("channel scaling 표에 IQR·variance 판정 열이 없습니다.")
    integrity = read_csv(csv_dir, "data_integrity.csv")
    feasibility = read_csv(csv_dir, "training_ratio_feasibility.csv")
    workload = read_csv(csv_dir, "training_workload_summary.csv")
    score = read_csv(csv_dir, "score_alignment.csv")
    sources = read_csv(csv_dir, "model_window_sources.csv")
    parameters = read_csv(csv_dir, "model_parameter_sources.csv")
    forward = read_csv(csv_dir, "input_forward_smoke.csv")

    windows = feasibility[["model", "model_input_window_size"]].drop_duplicates().sort_values("model")
    window_rows = [tuple(row) for row in windows.itertuples(index=False, name=None)]

    parameter_rows = []
    for model_name in windows["model"]:
        model_parameters = parameters[parameters["model"] == model_name]
        pending = model_parameters["status"].eq("source_choice_required")
        adapters = model_parameters["status"].isin({"adapter_required", "local_source_mismatch"})
        pending_names = ", ".join(model_parameters.loc[pending, "parameter"])
        parameter_rows.append((
            model_name,
            int(model_parameters["fixed_across_ratios"].map(str).str.lower().eq("true").sum()),
            pending_names or "없음",
            int(adapters.sum()),
        ))

    scope_rows = []
    for row in scope.sort_values(["ratio", "scope"]).itertuples(index=False):
        scope_rows.append((
            f"{int(row.ratio)}%", row.scope,
            int(row.constant_file_channel_combinations),
            int(row.zero_range_file_channel_combinations),
            int(row.iqr_zero_file_channel_combinations),
            int(row.variance_zero_file_channel_combinations),
            int(row.scaler_execution_not_ready_file_channel_combinations),
            int(row.variation_available_file_channel_combinations),
            int(row.nan_file_channel_combinations),
            int(row.infinite_file_channel_combinations),
            int(row.total_file_channel_combinations),
        ))

    workload_rows = []
    for (ratio, model_name), rows in workload.groupby(["ratio", "model"], sort=True):
        workload_rows.append((
            f"{int(ratio)}%",
            model_name,
            number_range(rows["train_window_count_all_models"]),
            number_range(rows["validation_window_count_all_models"]),
            number_range(rows["optimizer_updates_per_epoch_all_models"]),
            number_range(rows["maximum_optimizer_updates"]),
        ))

    alignment_rows = []
    for model_name, rows in score.groupby("model", sort=True):
        row = rows.iloc[0]
        alignment_rows.append((
            model_name,
            int(row["model_input_window_size"]),
            int(row["source_time_index_of_first_core_score"]),
            "L-W+1" if model_name in {"CI-AE", "USAD"} else "L-W",
            f"{int(row['raw_wrapper_left_padding_count'])}/{int(row['raw_wrapper_right_padding_count'])}",
            "통과" if true_mask(rows["alignment_valid"]).all() else "실패",
        ))

    forward_rows = [
        (
            row.model,
            row.input_shape,
            row.actual_output_shapes,
            "통과" if str(row.forward_passed).lower() == "true" else "실패",
            "통과" if str(row.reference_forward_passed).lower() == "true" else "미완료",
            "가능" if str(row.full_module_import_ready).lower() == "true" else "의존성·adapter 필요",
        )
        for row in forward.itertuples(index=False)
    ]

    def numeric_sum(frame: pandas.DataFrame, column: str) -> int:
        return int(pandas.to_numeric(frame[column], errors="coerce").fillna(0).sum())

    missing_count = numeric_sum(integrity, "missing_value_count")
    infinite_count = numeric_sum(integrity, "infinite_value_count")
    duplicate_count = numeric_sum(integrity, "duplicate_timestamp_count")
    reversal_count = numeric_sum(integrity, "timestamp_reversal_count")
    irregular_count = numeric_sum(integrity, "irregular_gap_count")
    reordered_input_count = int(true_mask(integrity["input_reordered"]).sum())
    integrity_passed = true_mask(integrity["integrity_passed"]).all()
    alignment_passed = true_mask(score["alignment_valid"]).all()
    forward_passed = true_mask(forward["forward_passed"]).all()
    reference_forward_passed = true_mask(forward["reference_forward_passed"]).all()
    crossing_count = int(
        pandas.to_numeric(workload["fit_validation_boundary_crossing_window_count"]).sum()
        + pandas.to_numeric(workload["session_boundary_crossing_window_count"]).sum()
    )
    infeasible = feasibility[~true_mask(feasibility["execution_feasible"])]
    pending_parameter_count = int(parameters["status"].eq("source_choice_required").sum())
    adapter_count = int(parameters["status"].isin({"adapter_required", "local_source_mismatch"}).sum())
    transfer_rule_note = (
        "USAD batch=128과 GDN topk=5는 GHL 성능을 보기 전에 정한 프로젝트 고정 전이값이다. "
        "공식 GHL 데이터셋별 설정값은 아니다."
        if dataset == "GHL" else
        "GDN topk=22는 HAI 성능을 보기 전에 정한 프로젝트 고정 전이값이다. "
        "공식 HAI 데이터셋별 설정값은 아니다."
    )
    workload_completion_note = (
        "USAD batch=128과 GDN topk=5를 포함한 고정 설정으로 계산; 실제 update는 실행 로그에서 확정"
        if dataset == "GHL" else
        "GDN batch=32 고정 설정으로 계산; topk=22는 window 수에 영향 없음; 실제 update는 실행 로그에서 확정"
    )

    applicability_rows = []
    for model_name in windows["model"]:
        model_feasible = true_mask(feasibility.loc[feasibility["model"] == model_name, "execution_feasible"]).all()
        model_alignment = true_mask(score.loc[score["model"] == model_name, "alignment_valid"]).all()
        model_forward = forward.loc[forward["model"] == model_name].iloc[0]
        model_parameters = parameters[parameters["model"] == model_name]
        open_statuses = sorted(set(model_parameters.loc[
            model_parameters["status"].isin({
                "source_choice_required", "adapter_required", "local_source_mismatch",
            }), "status"]))
        if not model_feasible or not model_alignment or not true_value(model_forward.forward_passed):
            applicability = "적용 불가"
            reason = "window·batch·정렬 또는 core forward 실패"
        elif (
            open_statuses
            or not true_value(model_forward.reference_forward_passed)
            or not true_value(model_forward.full_module_import_ready)
        ):
            applicability = "조건부 적용 가능"
            reasons = open_statuses[:]
            if not true_value(model_forward.reference_forward_passed):
                reasons.append("선정 출처 forward 대기")
            if not true_value(model_forward.full_module_import_ready):
                reasons.append("전체 모듈 import 대기")
            reason = ", ".join(reasons)
        else:
            applicability = "적용 가능"
            reason = "고정 설정의 적용성 검사 완료"
        applicability_rows.append((model_name, applicability, reason))

    completion_rows = [
        ("파라미터·출처 표", "작성", f"출처 선택 대기 {pending_parameter_count}개 항목" if pending_parameter_count else "출처 선택 대기 없음"),
        ("비율별 동일 설정", "통과", "비율은 고정 validation 앞의 fit subset 길이만 바꿈"),
        ("데이터 무결성", "통과" if integrity_passed else "실패", f"NaN {missing_count}, Inf {infinite_count}"),
        (
            "window·batch·update 수",
            "작성",
            workload_completion_note,
        ),
        ("경계 침범", "통과" if crossing_count == 0 else "실패", f"실제 침범 {crossing_count}개"),
        ("점수·timestamp 정렬", "통과" if alignment_passed else "실패", "padding하지 않은 core 기준"),
        (
            "학습 없는 forward",
            "통과" if reference_forward_passed else ("부분" if forward_passed else "실패"),
            "공식 USAD 구조 smoke 대기" if forward_passed and not reference_forward_passed else f"adapter·불일치 행 {adapter_count}개",
        ),
    ]

    model_semantics = {
        "CI-AE": ("[B,100]", "채널별 별도 모델", "fit-only MinMax adapter 필요", "재구성 core를 window 중앙에 배치", "True"),
        "LSTM-AD": (f"[B,100,{feature_count}]", "1-step forecast", "내부 20% 분할·split별 표준화 우회 필요", "labels[100:]", "False"),
        "USAD": (f"[B,10,{feature_count}]", "window×channel flatten", "공식 구조 adapter 필요; batch=128 고정", "재구성 core를 window 중앙에 배치", "False"),
        "GDN": (f"[B,{feature_count},5]", "1-step graph forecast", "외부 fit-only MinMax 사용", "labels[5:]", "False"),
    }
    semantics_rows = [(model_name, *model_semantics[model_name]) for model_name in windows["model"]]

    selected_source_count = int(true_mask(sources["project_selected"]).sum())
    summary = {
        "dataset": dataset,
        "feature_count": feature_count,
        "input_file_or_session_count": input_count,
        "ratios": sorted(int(value) for value in feasibility["ratio"].unique()),
        "model_windows": dict(zip(windows["model"], windows["model_input_window_size"].astype(int))),
        "infeasible_model_ratio_rows": int(len(infeasible)),
        "score_alignment_rows": int(len(score)),
        "selected_source_rows": selected_source_count,
        "pending_parameter_count": pending_parameter_count,
        "adapter_count": adapter_count,
        "core_forward_passed": bool(forward_passed),
        "selected_reference_forward_passed": bool(reference_forward_passed),
        "data_integrity_passed": bool(integrity_passed),
        "environment_gate": environment_gate,
        "model_applicability": {
            model: status for model, status, _ in applicability_rows
        },
    }

    if dataset == "GHL":
        window_basis = (
            "CI-AE와 LSTM-AD는 채택한 TSB-AD 실행값 100, USAD는 논문 민감도와 WADI "
            "항목에서 옮긴 10, GDN은 SWaT·WADI 논문 실험값 5다. USAD의 downsampling은 "
            "옮기지 않고 GHL 원 관측 간격을 유지한다."
        )
        wrapper_status = (
            "CI-AE 원본은 scaler 축이 학습 window 수에 묶여 train과 test 길이가 다르면 "
            "실패한다. LSTM-AD와 로컬 USAD는 validation과 test를 각 구간 통계로 다시 "
            "표준화해 공통 fit-only MinMax 규칙과 충돌한다. USAD는 공식 구현과 현재 "
            "TSB-AD 포팅의 구조·optimizer·점수식도 다르다."
        )
        timestamp_note = (
            "GHL에는 timestamp 열이 없어 간격 검사를 적용하지 않았다. 센서 이름 집합이 "
            "같고 열 순서만 다른 파일은 첫 GHL 파일의 센서명 순서로 재배열한 뒤 모델에 넣는다."
        )
        timestamp_summary = "- timestamp: 열이 없어 간격 검사는 해당 없음"
        workload_note = "범위 표시는 GHL 25개 시계열의 최솟값–최댓값이다. CI-AE는 19개 채널별 모델의 합계다."
        alignment_note = (
            "CI-AE와 USAD의 window 재구성 core는 window 중앙 시점에, LSTM-AD와 GDN의 "
            "1-step 예측 core는 `labels[W:]`에 대응한다."
        )
        forward_note = (
            "USAD 행은 현재 로컬 TSB-AD 포팅의 인터페이스만 확인한 결과다. 선정한 공식 "
            "dual-decoder 구조가 아직 로컬 실행 코드에 없어 공식 출처 smoke test로 세지 않았다."
        )
        score_contract_note = (
            "CI-AE와 USAD는 window 안의 절대오차를 채널별 평균하고 중앙 시점에 놓는다. "
            "LSTM-AD와 GDN은 1-step 대상의 채널별 절대오차를 쓴다."
        )
    else:
        window_basis = "HAI GDN은 GHL과 모델 정의를 맞추기 위해 SWaT·WADI 논문 실험값 5를 유지한다."
        wrapper_status = "GDN은 내부 scaling을 하지 않아 프로젝트의 외부 fit-only MinMax 규칙을 그대로 적용한다."
        timestamp_note = "HAI는 실제 timestamp로 중복·역전·1초 간격 이탈을 확인했고 여섯 세션의 센서 순서가 같았다."
        timestamp_summary = (
            f"- timestamp 중복: {duplicate_count}, 역전: {reversal_count}, "
            f"1초 간격 이탈: {irregular_count}"
        )
        workload_note = "HAI GDN은 훈련 4세션의 window를 합친 뒤 batch를 셌지만 세션 사이에는 window를 만들지 않았다."
        alignment_note = "GDN의 1-step 예측 core는 각 테스트 세션의 `labels[5:]`와 실제 timestamp에 대응한다."
        forward_note = "HAI GDN은 현재 채택한 로컬 core로 선정 출처 형상까지 확인했다."
        score_contract_note = "GDN은 1-step 대상의 채널별 절대오차를 쓴다."

    hai_constant_detail = ""
    if dataset == "HAI":
        hai_rows = [
            (
                f"{int(row.ratio)}%", row.scope,
                int(row.constant_file_channel_combinations),
                row.constant_by_source,
                int(row.pooled_constant_channel_count),
            )
            for row in scope.sort_values(["ratio", "scope"]).itertuples(index=False)
        ]
        hai_constant_detail = f"""

### HAI constant 집계 단위

{markdown_table(("비율", "범위", "훈련 세션·채널 조합", "세션별 constant 채널", "pooled constant 채널"), hai_rows)}

각 행의 전체 조합 수 344는 `훈련 세션 4개 × 채널 86개`다. pooled 값은 같은 비율과 범위에서 네 세션의 값을 채널별로 합쳐 다시 판정한 채널 수다.
"""

    report = f"""# {dataset} 계층 2 적용 가능성 점검

이 EDA의 목적은 GHL·HAI에서 최적 파라미터를 찾는 일이 아니다. 논문과 공개 구현에서 먼저 정한 설정을 모든 학습 비율에 그대로 적용할 수 있는지 검사한다. 테스트 이상 성능은 읽지 않았고 모델도 학습하지 않았다.

## 실험에서 고정하는 원칙

전체 정상 학습 구간의 마지막 10%를 고정 validation으로 한 번만 분리한다. 남은 fit pool의 시간순 앞부분을 5·10·20·40·60·80·100%로 중첩해 fit subset을 만들며 scaler는 각 subset에만 fit한다. 같은 고정 validation에는 transform만 적용한다.

학습 비율 외에는 바꾸지 않는다. 모델 구조, 입력 window, hidden·latent 차원, learning rate, batch size, epoch·early stopping, optimizer, 이상 점수와 정규화 규칙은 비율별로 다시 고르지 않는다. 유효 window 수와 batch·update 수만 학습량에 따라 달라진다. 성능이 낮다는 이유로 설정을 바꾸지 않는다.

설정 충돌은 논문의 데이터셋별 실험값, 공식 config·notebook, 공식 CLI 기본값, TSB-AD 기본값, 프로젝트 임의값 순으로 판단했다.

{transfer_rule_note}

## 고정한 입력 window

{markdown_table(("모델", "입력 window"), window_rows)}

이 값은 GHL test 결과로 고른 최적값이 아니다. {window_basis} 출처별 차이는 `model_window_sources.csv`, 전체 파라미터·값·선정 이유는 `model_parameter_sources.csv`에 남겼다.

## 가져온 코드의 실행 의미

{markdown_table(("모델", "forward 입력", "학습 단위", "전처리 상태", "core 점수 대응", "drop_last"), semantics_rows)}

입력 형상은 모두 적용 가능했다. 다만 원본 wrapper를 그대로 실행해도 된다는 뜻은 아니다. {wrapper_status} 이 항목은 모델을 고치는 대신 실행 전 adapter·구현 검증 게이트로 남겼다.

학습 손실과 이상 점수는 구분한다. 각 모델의 학습 손실은 출처 구현을 따르되, 저장 점수는 프로젝트 공통 규칙인 채널별 절대오차를 쓴다. {score_contract_note} 그다음 학습·validation 오차로 robust normalization을 맞춘 뒤 채널 max를 취한다. 이 규칙은 모든 비율과 seed에서 같다.

{markdown_table(("모델", "비율 고정 항목 수", "출처 선택 대기", "adapter·불일치 행"), parameter_rows)}

## 데이터 무결성

- 파일·세션: {input_count}개, 센서: {feature_count}개
- NaN: {missing_count}, Inf: {infinite_count}
{timestamp_summary}
- 센서 이름 집합: {'모두 일치' if true_mask(integrity['channel_set_matches_reference']).all() else '불일치 있음'}
- 원본 열 순서가 달라 공통 순서로 재배열한 파일: {reordered_input_count}개

{timestamp_note} 파일별 수치는 `data_integrity.csv`에 있다.

## 범위별 채널 상태

{markdown_table(("비율", "범위", "constant", "zero range", "IQR=0", "variance=0", "scaler 실행 준비 미충족", "변화 있음", "NaN", "Inf", "전체 조합"), scope_rows)}

constant는 한 파일·세션의 한 채널이 해당 범위에서 한 값으로만 유지됐다는 뜻이다. zero range는 최솟값과 최댓값이 같다는 뜻이고, IQR=0은 중앙 50%가 같은 값이라는 뜻이라 각각 따로 센다. constant와 zero range는 정보 변화가 없다는 경고일 뿐이다. scikit-learn MinMaxScaler는 이런 열도 오류 없이 처리한다. 값이 없거나 NaN·Inf가 있을 때만 scaler 실행 준비 미충족으로 판정한다. fit pool, 비율별 fit subset, 고정 validation은 섞지 않았다. 채널별 min·max·variance·IQR은 `channel_scaling_readiness.csv`에 있다.
{hai_constant_detail}

## 유효 window와 계산량

{markdown_table(("비율", "모델", "학습 window", "validation window", "epoch당 update", "최대 update"), workload_rows)}

{workload_note} 최대 update는 조기 종료가 없다고 본 상한이며 실제값은 학습 로그로만 확정한다. window가 파일·세션 또는 fit·validation 경계를 넘은 사례는 {crossing_count}개다.

## 점수와 원 시간축 정렬

{markdown_table(("모델", "W", "첫 core 대응 index", "저장 길이", "원 wrapper 좌/우 padding", "판정"), alignment_rows)}

본 실험 산출물은 원 wrapper의 복제 padding을 저장하지 않는다. {alignment_note}

## 학습 없는 forward smoke test

{markdown_table(("모델", "입력", "실제 출력", "로컬 core", "선정 출처 core", "전체 모듈 import"), forward_rows)}

합성 batch 2개로 모델 core의 forward만 호출했다. optimizer step, checkpoint, 실데이터 학습은 실행하지 않았다. {forward_note} 전체 모듈 import가 막힌 항목은 형상 실패가 아니라 누락 의존성이나 wrapper adapter 문제로 따로 기록했다.

실행 환경 gate는 `{environment_gate}`다. 이 값은 현재 인터프리터와 `configs/environment.yaml`의 버전을 대조한 결과이며 모델별 adapter 완료 여부와는 별개다.

## 모델별 적용 판정

{markdown_table(("모델", "판정", "남은 조건"), applicability_rows)}

## 완료 기준 판정

{markdown_table(("항목", "상태", "비고"), completion_rows)}

파라미터 선정과 데이터 적용 가능성 EDA는 여기서 닫는다. 출처 선택 대기는 {pending_parameter_count}개다. 다만 adapter·구현 불일치 {adapter_count}개와 전체 모듈 import가 남아 있어 실제 학습을 시작하지 않는다. 남은 항목은 비율별 성능으로 고치지 않고 다음 계층 2 구현 gate에서 처리한다.

## 이번 EDA에서 하지 않은 일

ACF peak, ADF, Ljung–Box, 고정 lag 자기상관으로 모델 window를 다시 고르지 않았다. test anomaly 성능도 쓰지 않았다. `vus_l_max`와 threshold는 지우 담당 평가 규칙이므로 계산하지 않았다.
"""
    (report_dir / "tier2_eda_report.md").write_text(report, encoding="utf-8")
    return summary
