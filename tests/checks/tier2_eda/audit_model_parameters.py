"""계층 2 파라미터의 선택값과 출처 충돌을 한 표로 만든다."""


def build_model_parameter_rows(dataset: str) -> list[dict]:
    rows = []

    def add(
        model: str,
        parameter: str,
        value: object,
        priority: int,
        source: str,
        reason: str,
        decision_status: str = "fixed",
        implementation_status: str = "matches",
    ) -> None:
        if decision_status == "source_choice_required":
            status = decision_status
        elif implementation_status != "matches":
            status = implementation_status
        else:
            status = "verified" if priority <= 4 else "fixed"
        rows.append({
            "dataset": dataset,
            "model": model,
            "parameter": parameter,
            "reference_value": value,
            "source_priority": priority,
            "source": source,
            "selection_reason": reason,
            "fixed_across_ratios": True,
            "status": status,
        })

    if dataset == "GHL":
        add("CI-AE", "model_structure", "19 separate univariate autoencoders", 5,
            "project control definition", "채널 간 정보 공유를 원리적으로 차단한다.",
            implementation_status="adapter_required")
        add("CI-AE", "input_window", 100, 4, "TSB-AD run_AutoEncoder",
            "채택 wrapper의 실제 입력 길이다.")
        add("CI-AE", "hidden_and_latent", "64,32; latent=32", 4, "TSB-AD run_AutoEncoder",
            "wrapper가 넘기는 hidden_neurons를 그대로 쓴다.")
        add("CI-AE", "activation_and_regularization", "ReLU; BatchNorm; dropout=0.2", 4,
            "TSB-AD AutoEncoder", "채택 구현의 구조를 그대로 쓴다.")
        add("CI-AE", "batch_size", 128, 4, "TSB-AD run_AutoEncoder",
            "wrapper가 클래스 기본값 32를 128로 덮어쓴다.")
        add("CI-AE", "max_epochs", 50, 4, "TSB-AD run_AutoEncoder",
            "wrapper가 클래스 기본값 100을 50으로 덮어쓴다.")
        add("CI-AE", "optimizer", "Adam(lr=0.001, weight_decay=1e-5)", 4,
            "TSB-AD AutoEncoder", "채택 코드의 학습 규칙이다.")
        add("CI-AE", "early_stopping", "none; use final epoch state", 4,
            "TSB-AD AutoEncoder", "원본은 state_dict를 복사하지 않아 최저 손실 시점이 아니라 마지막 epoch 상태가 남는다.")
        add("CI-AE", "loss", "MSE reconstruction", 4, "TSB-AD AutoEncoder",
            "채택 코드의 학습 손실이다.")
        add("CI-AE", "anomaly_score", "mean absolute reconstruction error over window; robust normalization; channel max", 5,
            "project score contract", "학습 손실은 MSE로 두고 출력 점수만 공통 절대오차 계약에 맞춘다.",
            implementation_status="adapter_required")
        add("CI-AE", "input_normalization", "fit-only MinMax; reuse on validation and test", 5,
            "project preprocessing contract", "테스트 통계 사용과 이중 정규화를 막는다.",
            implementation_status="adapter_required")
        add("CI-AE", "validation_ratio", 0.1, 5, "project preprocessing contract",
            "모든 모델에 같은 시간순 10% validation을 적용한다.", implementation_status="adapter_required")
        add("CI-AE", "drop_last", True, 4, "TSB-AD AutoEncoder DataLoader",
            "채택 코드의 실제 batch 동작이다.")

        add("LSTM-AD", "model_structure", "encoder-decoder LSTM; hidden=20; layers=2; horizon=1", 4,
            "TSB-AD LSTMAD", "채택 구현의 예측 구조를 그대로 쓴다.")
        add("LSTM-AD", "input_window", 100, 4, "TSB-AD run_LSTMAD",
            "채택 wrapper의 실제 입력 길이다.")
        add("LSTM-AD", "batch_size", 128, 4, "TSB-AD run_LSTMAD",
            "wrapper가 명시한 batch다.")
        add("LSTM-AD", "max_epochs", 50, 4, "TSB-AD LSTMAD",
            "채택 구현의 상한이다. 원본의 49 epoch 종료 조건은 바로잡아야 한다.",
            implementation_status="adapter_required")
        add("LSTM-AD", "optimizer", "Adam(lr=0.0008); StepLR(step=5,gamma=0.75)", 4,
            "TSB-AD LSTMAD", "채택 구현의 학습 규칙이다.")
        add("LSTM-AD", "early_stopping", "patience=3; delta=0.0001", 4,
            "TSB-AD LSTMAD", "채택 구현의 조기 종료 규칙이다.", implementation_status="adapter_required")
        add("LSTM-AD", "loss", "MSE one-step forecast", 4, "TSB-AD LSTMAD",
            "채택 구현의 학습 손실이다.")
        add("LSTM-AD", "activation", "GELU decoder output projection", 4, "TSB-AD LSTMAD",
            "채택 구현의 decoder 구조다.")
        add("LSTM-AD", "anomaly_score", "per-channel absolute forecast error; robust normalization; channel max", 5,
            "project score contract", "학습 손실은 MSE로 두고 출력 점수는 공통 절대오차 계약을 따른다.",
            implementation_status="adapter_required")
        add("LSTM-AD", "input_normalization", "fit-only MinMax; internal split normalization disabled", 5,
            "project preprocessing contract", "validation과 test 자체 통계 사용을 막는다.",
            implementation_status="adapter_required")
        add("LSTM-AD", "validation_ratio", 0.1, 5, "project preprocessing contract",
            "내부 20% 재분할 대신 공통 경계를 쓴다.", implementation_status="adapter_required")
        add("LSTM-AD", "drop_last", False, 4, "TSB-AD LSTMAD DataLoader",
            "마지막 작은 batch도 학습에 쓴다.")

        add("USAD", "model_structure", "official dual-decoder USAD with two optimizers", 2,
            "manigalati/usad usad.py", "공식 구현을 TSB-AD 포팅보다 우선한다.",
            implementation_status="local_source_mismatch")
        add("USAD", "input_window", 10, 1, "USAD paper sensitivity and WADI setting",
            "GHL 결과를 보기 전에 정한 transfer 기준이다.")
        add("USAD", "hidden_and_latent", "D=10N; D/2,D/4; latent=100", 1,
            "USAD paper WADI setting and official usad.py", "WADI에 적힌 latent 100만 옮기고 데이터 전처리 전체를 복제하지 않는다.",
            implementation_status="local_source_mismatch")
        add("USAD", "batch_size", 128, 5,
            "project fixed transfer rule", "SWaT 규모에 묶인 7,919를 옮기지 않고 모든 GHL 비율에 128을 고정한다.")
        add("USAD", "max_epochs", 70, 1, "USAD paper SWaT/WADI setting",
            "논문 데이터셋별 실험값을 우선한다.", implementation_status="local_source_mismatch")
        add("USAD", "optimizer", "two Adam(lr=0.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0)", 2,
            "manigalati/usad usad.py", "encoder+decoder1과 encoder+decoder2를 따로 갱신한다.",
            implementation_status="local_source_mismatch")
        add("USAD", "optimizer_updates_per_batch", 2, 2, "manigalati/usad usad.py",
            "공식 학습은 batch마다 두 optimizer를 한 번씩 갱신한다.",
            implementation_status="local_source_mismatch")
        add("USAD", "early_stopping", "none", 1, "USAD paper and official code",
            "70 epoch를 고정하고 조기 종료하지 않는다.", implementation_status="local_source_mismatch")
        add("USAD", "loss", "official alternating loss1/loss2", 2, "manigalati/usad usad.py",
            "논문의 두 autoencoder 학습 의미를 보존한다.", implementation_status="local_source_mismatch")
        add("USAD", "activation", "ReLU encoder; Sigmoid decoders", 2,
            "manigalati/usad usad.py", "공식 구현의 출력 범위와 맞춘다.",
            implementation_status="local_source_mismatch")
        add("USAD", "anomaly_score", "alpha=0.5,beta=0.5; window-mean absolute error per channel; robust normalization; channel max", 5,
            "official USAD weights + project score contract", "공식 가중치는 유지하되 출력은 공통 절대오차·채널 max 계약에 맞춘다.",
            implementation_status="adapter_required")
        add("USAD", "sampling_policy", "no downsampling; preserve GHL observation interval", 5,
            "project data protocol", "WADI의 일부 파라미터만 옮기며 GHL 시간축은 바꾸지 않는다.")
        add("USAD", "input_normalization", "fit-only MinMax; reuse on validation and test", 5,
            "project preprocessing contract", "split별 자체 표준화를 금지한다.", implementation_status="adapter_required")
        add("USAD", "validation_ratio", 0.1, 5, "project preprocessing contract",
            "공통 시간순 validation 경계를 쓴다.", implementation_status="adapter_required")
        add("USAD", "drop_last", False, 2, "official USAD DataLoader",
            "마지막 작은 batch도 사용한다.")

    add("GDN", "model_structure", "learned sensor graph; one-step forecast", 1, "GDN paper",
        "센서 관계를 학습하는 주 모델 정의다.")
    add("GDN", "input_window", 5, 1, "GDN paper SWaT/WADI experiments",
        "논문 실험값을 CLI 기본 15보다 우선한다.")
    add("GDN", "embedding_dimension", 64, 2, "d-ailin/GDN run.sh",
        "공식 실행 스크립트 값이다.")
    add("GDN", "output_hidden_dimension", 128, 2, "d-ailin/GDN run.sh",
        "공식 실행 스크립트가 CLI 기본 256을 덮어쓴다.")
    add("GDN", "output_layer_count", 1, 2, "d-ailin/GDN run.sh",
        "공식 실행 스크립트 값이다.")
    add("GDN", "heads", 1, 4, "pinned GraGOD implementation",
        "현재 채택 구현의 단일 head 구조다.")
    add("GDN", "dropout", 0.2, 2, "d-ailin/GDN model.py",
        "원 저자 모델의 출력 dropout 값이다. GraGOD는 같은 인자를 attention에도 적용한다.",
        implementation_status="local_source_mismatch")
    add("GDN", "topk", 5 if dataset == "GHL" else 22, 5,
        "project fixed transfer rule",
        "GHL·HAI 성능을 보기 전에 채널 수에 맞춰 고정한 프로젝트 값이다.")
    add("GDN", "batch_size", 32, 2, "d-ailin/GDN run.sh",
        "공식 실행 스크립트 값이다.")
    add("GDN", "train_and_evaluation_stride", "train=1; evaluation=1", 2,
        "d-ailin/GDN run.sh and TimeDataset.py",
        "run.sh의 SLIDE_STRIDE=1과 평가 구간의 고정 stride 1을 따른다.")
    add("GDN", "max_epochs", 50, 1, "GDN paper SWaT/WADI experiments",
        "논문에 적힌 최대 학습 횟수를 우선한다.")
    add("GDN", "optimizer", "Adam(lr=0.001, weight_decay=0)", 2, "d-ailin/GDN train.py/run.sh",
        "공식 학습 코드와 일치한다.")
    add("GDN", "adam_betas", "(0.9,0.99)", 1, "GDN paper SWaT/WADI experiments",
        "논문에 적힌 Adam 설정을 따른다.")
    add("GDN", "scheduler_and_gradient_clip", "none", 2,
        "d-ailin/GDN train.py", "원 저자 학습 경로에는 scheduler와 gradient clipping이 없다.",
        implementation_status="adapter_required")
    add("GDN", "early_stopping", "patience=10", 1, "GDN paper SWaT/WADI experiments",
        "논문에 적힌 조기 종료 규칙을 따른다.")
    add("GDN", "loss", "MSE one-step forecast", 2, "d-ailin/GDN train.py",
        "공식 학습 손실이다.")
    add("GDN", "anomaly_score", "per-channel absolute forecast error; channel max after normalization", 5,
        "project score contract", "공통 점수 인터페이스와 현재 GDN 계산이 맞는다.")
    add("GDN", "input_normalization", "fit-only MinMax; reuse on validation and test", 5,
        "project preprocessing contract", "GDN 내부에는 별도 scaling이 없어 그대로 적용된다.")
    add("GDN", "validation_ratio", 0.1, 5, "project preprocessing contract",
        "전체 정상 구간에서 마지막 10%를 먼저 고정하고 남은 fit pool에 비율을 적용한다.")
    add("GDN", "drop_last", False, 4, "pinned GraGOD DataLoader",
        "마지막 작은 batch도 사용한다.")
    return rows
