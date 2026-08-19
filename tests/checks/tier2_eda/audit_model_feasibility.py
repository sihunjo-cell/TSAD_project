"""고정한 입력 길이로 모델별 계산 가능성을 요약한다."""


def build_model_feasibility_rows(ratio_rows: list[dict], feature_count: int) -> list[dict]:
    rows = []
    for ratio_row in ratio_rows:
        model = ratio_row["model"]
        window_size = ratio_row["model_input_window_size"]
        model_features = 1 if model == "CI-AE" else feature_count
        instance_count = feature_count if model == "CI-AE" else 1
        if model == "CI-AE":
            input_shape = f"[batch, {window_size}]"
            output_shape = f"[batch, {window_size}]"
            flattened_dimension = window_size
            score_rule = "채널별 window 재구성 오차; 중앙 시점 정렬"
        elif model == "LSTM-AD":
            input_shape = f"[batch, {window_size}, {feature_count}]"
            output_shape = f"[1, batch, {feature_count}]"
            flattened_dimension = "not_applicable"
            score_rule = "1-step 예측 오차; source[W:] 정렬"
        elif model == "USAD":
            input_shape = f"[batch, {window_size}, {feature_count}]"
            output_shape = f"3 x [batch, {window_size * feature_count}]"
            flattened_dimension = window_size * feature_count
            score_rule = "window×channel 재구성 오차; 중앙 시점 정렬"
        else:
            input_shape = f"[batch, {feature_count}, {window_size}]"
            output_shape = f"[batch, {feature_count}]"
            flattened_dimension = "not_applicable"
            score_rule = "1-step 채널별 절대 예측 오차; source[W:] 정렬"
        rows.append({
            **ratio_row,
            "input_tensor_shape": f"({ratio_row['train_window_count']}, {window_size}, {model_features})",
            "model_forward_input_shape": input_shape,
            "model_forward_output_shape": output_shape,
            "flattened_input_dimension": flattened_dimension,
            "graph_node_count": feature_count if model == "GDN" else "not_applicable",
            "node_window_length": window_size if model == "GDN" else "not_applicable",
            "channel_order_rule": "manifest order preserved",
            "score_alignment_rule": score_rule,
            "channel_independent": model == "CI-AE",
            "implementation_status": "data shape applicable; raw wrapper contract checked separately",
        })
    return rows
