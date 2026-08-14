"""exp00 — reshape→permute 두 줄 교체의 완전성 실행 확인 (SHAPEFLOW.md 2부).

검증 가설: GraGODs/GraGOD(develop, ec8cd452) models/gdn/model.py:271, 304 의
x.reshape(-1, x.size(2), x.size(1)) 를 x.permute(0, 2, 1) 로 바꾸면
그 이후 전체 경로가 (batch, n_features, window) 전제로 정합한가.

실험 설계 원칙: 교체 대상인 두 줄의 연산만 이 스크립트에서 대체하고,
나머지는 전부 GraGOD 클론의 실제 코드를 임포트해 그대로 사용한다.
GraGOD 코드는 수정하지 않는다. GDN 학습·데이터셋 로드 없음(합성 텐서만).

  [e] 작은 GDN 인스턴스에 permute 입력으로 forward가 도는지, 출력 shape 확인.
      e1 = 문자 그대로 permute(0,2,1)만 적용(비연속 텐서),
      e2 = permute(0,2,1).contiguous() 적용.
  [f] 채널 정렬: 채널 2에만 상수 오프셋을 준 입력이 실제 X_true 구성 경로
      (models/predict.py:121-139의 문장들을 그대로 재현)를 거쳐
      calculate_anomaly_score의 채널 2 자리에 나타나는지.
  [g] y 정렬: __getitem__의 y가 shared_step의 loss 피연산자가 되기까지
      겪는 변형(squeeze(1) 하나)에 채널·시점 혼합이 없는지.

실행 예:
    python trace_shape_flow.py --gragod-repository-path <GraGOD 클론 경로>
"""

import argparse
import importlib.util
import os
import sys
import types

import torch

# Windows 콘솔(cp949)에서도 출력이 깨지지 않게 UTF-8로 고정한다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def print_environment_versions() -> None:
    import pytorch_lightning
    import torch_geometric

    print("[환경] 이 실행 환경은 GraGOD 고정 환경이 아니다. 판정은 모양·정렬에 한정된다.")
    print(f"  torch              : {torch.__version__}")
    print(f"  torch_geometric    : {torch_geometric.__version__} (스크래치 용도 설치)")
    print(f"  pytorch_lightning  : {pytorch_lightning.__version__} (스크래치 용도 설치; GraGOD 고정본은 저자 포크)")


def install_datasets_package_stub() -> None:
    """`datasets` 패키지 자리에 스텁을 등록해 임포트 그래프의 죽은 가지를 끊는다.

    사유: models/gdn/model.py:9 → gragod/training/__init__.py:1 →
    gragod/training/main.py:7 이 `from datasets import load_*_training_data` 를
    당기고, datasets/__init__.py → datasets/config.py 의 dataclass 기본값이
    Python 3.13 의 강화된 규칙(unhashable 기본값 금지)과 충돌해 임포트가 깨진다.
    GraGOD 는 Python 3.10 고정(.python-version)이라 원 환경에서는 문제가 없다.

    스텁이 대체하는 것은 이 검증에서 한 번도 호출되지 않는 데이터셋 로더
    3개의 이름뿐이다(데이터셋 로드 금지 조건과도 일치). 검증 대상 코드
    (GDN, GDN_PLModule, get_data_loader, build_fully_connected_edge_index)는
    전부 GraGOD 클론의 실제 파일에서 로드된다. GraGOD 코드 수정 없음.
    """
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_swat_training_data = None
    datasets_stub.load_telco_training_data = None
    datasets_stub.load_ute_training_data = None
    sys.modules["datasets"] = datasets_stub


def load_gragod_module_from_file(module_name: str, file_path: str):
    """datasets 패키지의 개별 모듈을 __init__.py 를 거치지 않고 로드한다.

    사유: datasets/__init__.py 가 당기는 datasets/config.py 의 dataclass 기본값이
    Python 3.13 의 강화된 규칙(unhashable 기본값 금지)과 충돌해 임포트가 깨진다.
    GraGOD 는 Python 3.10 고정(.python-version)이라 원 환경에서는 문제가 없다.
    파일 경로 직접 로드는 GraGOD 코드를 한 줄도 수정하지 않는 우회다.
    """
    module_spec = importlib.util.spec_from_file_location(module_name, file_path)
    loaded_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(loaded_module)
    return loaded_module


def build_small_gdn_and_module(gragod_repository_path: str):
    """n_features=3, window=4, topk=2, embed 8 의 작은 GDN과 PL 모듈을 만든다."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    from models.gdn.model import GDN, GDN_PLModule

    graph_module = load_gragod_module_from_file(  # datasets/graph.py:46-66
        "gragod_graph_module",
        os.path.join(gragod_repository_path, "datasets", "graph.py"),
    )
    placeholder_series = torch.zeros(1, 3)
    edge_index = graph_module.build_fully_connected_edge_index(
        placeholder_series, device="cpu"
    )

    gdn_model = GDN(
        edge_index=[edge_index],
        n_features=3,
        embed_dim=8,
        out_layer_inter_dim=16,
        window_size=4,
        out_layer_num=1,
        topk=2,
        heads=1,
        dropout=0.0,
    )
    gdn_model.eval()

    lightning_module = GDN_PLModule(
        model=gdn_model,
        model_params={},
        checkpoint_cb=ModelCheckpoint(),
        init_lr=0.001,
        criterion=torch.nn.MSELoss(),
    )
    lightning_module.eval()
    return gdn_model, lightning_module


def build_time_channel_labeled_windows(
    batch_count: int, window_size: int, feature_count: int
) -> torch.Tensor:
    """값 = 배치*1000 + 시점*10 + 채널 인 (batch, window, feature) 텐서."""
    values = torch.zeros(batch_count, window_size, feature_count)
    for batch_index in range(batch_count):
        for time_index in range(window_size):
            for channel_index in range(feature_count):
                values[batch_index, time_index, channel_index] = (
                    batch_index * 1000 + time_index * 10 + channel_index
                )
    return values


def report_forward_with_permuted_input(gdn_model) -> None:
    print()
    print("=" * 72)
    print("[e] permute 입력으로 GDN.forward 실행")
    print("=" * 72)

    labeled_windows = build_time_channel_labeled_windows(
        batch_count=2, window_size=4, feature_count=3
    )
    print(f"입력 (batch, window, feature) = {tuple(labeled_windows.shape)}")

    print("\n--- e1: 문자 그대로 permute(0, 2, 1) 만 적용 (비연속 텐서) ---")
    permuted_only = labeled_windows.permute(0, 2, 1)
    print(f"permute 후 shape: {tuple(permuted_only.shape)}, is_contiguous: {permuted_only.is_contiguous()}")
    try:
        with torch.no_grad():
            output = gdn_model(permuted_only)
        print(f"forward 성공, 출력 shape: {tuple(output.shape)}")
    except RuntimeError as runtime_error:
        print("forward 실패 (RuntimeError):")
        print(f"  {runtime_error}")
        print("  (실패 지점: models/gdn/model.py:117 의 x.view(-1, all_feature) — 비연속 텐서에 view 불가)")

    print("\n--- e2: permute(0, 2, 1).contiguous() 적용 ---")
    permuted_contiguous = labeled_windows.permute(0, 2, 1).contiguous()
    print(f"변환 후 shape: {tuple(permuted_contiguous.shape)}, is_contiguous: {permuted_contiguous.is_contiguous()}")
    with torch.no_grad():
        output = gdn_model(permuted_contiguous)
    print(f"forward 성공, 출력 shape: {tuple(output.shape)}")
    expected_shape = (2, 3)
    print(f"출력 shape == (batch, n_features) == {expected_shape} 인가: {tuple(output.shape) == expected_shape}")


def zero_model_parameters_keep_embedding(gdn_model) -> None:
    """모델 파라미터를 전부 0으로 만들어 예측을 항등적으로 0으로 고정한다.

    [f]의 목적은 '오프셋의 흔적이 채널 2 자리에 나타나는가'라는 정렬 판정이다.
    미학습 난수 가중치로는 그래프 어텐션이 설계상 채널을 섞어 판정이 흐려지므로,
    예측을 정확히 0으로 만들어 scores = |0 - X_true| = X_true 가 되게 한다.
    embedding 만은 0이 아닌 값으로 되돌린다 — cos-sim(model.py:139-143)의
    0/0(nan)을 피하기 위해서이며, 다른 가중치가 전부 0이라 출력에는 영향 없다.
    """
    with torch.no_grad():
        for parameter in gdn_model.parameters():
            parameter.zero_()
        embedding_row_count, embedding_dim = gdn_model.embedding.weight.shape
        for row_index in range(embedding_row_count):
            for column_index in range(embedding_dim):
                gdn_model.embedding.weight[row_index, column_index] = (
                    1.0 + row_index + 0.1 * column_index
                )


def apply_fixed_predict_step_transform(window_batch: torch.Tensor) -> torch.Tensor:
    """교체 가설의 대상인 models/gdn/model.py:304 한 줄만 대체한다.

    원본: predictions = self(x.reshape(-1, x.size(2), x.size(1)))
    가설: predictions = self(x.permute(0, 2, 1))  (+ e1 결과에 따라 contiguous)
    """
    return window_batch.permute(0, 2, 1).contiguous()


def report_channel_alignment_through_score_path(
    gdn_model, lightning_module, dataset_module
) -> None:
    print()
    print("=" * 72)
    print("[f] 채널 정렬: 오프셋이 calculate_anomaly_score의 채널 2 자리에 나타나는가")
    print("=" * 72)

    from gragod import CleanMethods

    get_data_loader = dataset_module.get_data_loader  # datasets/dataset.py:97-145

    zero_model_parameters_keep_embedding(gdn_model)

    window_size = 4
    sample_count = 30
    # 채널 0, 1 = 전부 0 / 채널 2 = 전부 상수 오프셋 500.
    offset_series = torch.zeros(sample_count, 3)
    offset_series[:, 2] = 500.0
    labels = torch.zeros(sample_count)
    print(f"입력 시계열 shape: {tuple(offset_series.shape)}, 채널2만 500.0, 나머지 0.0")

    # 이하 X_true 구성은 models/predict.py process_dataset 의 실제 문장을 그대로 재현.
    # start_index 는 window_size 이상이어야 하며(models/predict.py:441-444),
    # 최솟값 start_index = window_size 를 사용한다.
    start_index = window_size
    X_true = offset_series[start_index - window_size :]  # models/predict.py:123
    labels = labels[start_index - window_size :]  # models/predict.py:124

    loader = get_data_loader(  # models/predict.py:127-136 과 동일 인자 구성
        X=X_true,
        edge_index=gdn_model.edge_index_sets[0],
        y=labels,
        window_size=window_size,
        clean=CleanMethods.NONE,
        batch_size=8,
        n_workers=0,
        shuffle=False,
    )

    X_true = X_true[window_size:-1, :]  # models/predict.py:139
    print(f"실제 경로로 만든 X_true shape: {tuple(X_true.shape)} (= (L - window_size - 1, n_features) = (25, 3))")
    print(f"X_true 첫 행: {X_true[0].tolist()}")

    # predict_step(models/gdn/model.py:292-305)의 교체 대상 한 줄만 대체하고 forward 는 실제 코드.
    prediction_batches = []
    with torch.no_grad():
        for window_batch, _, _, _ in loader:
            fixed_input = apply_fixed_predict_step_transform(window_batch.float())
            prediction_batches.append(gdn_model(fixed_input))

    # 이하는 GraGOD 실제 메서드 그대로: post_process_predictions + calculate_anomaly_score
    # (models/gdn/model.py:307-316).
    scores = lightning_module.calculate_anomaly_score(
        predict_output=prediction_batches, X_true=X_true
    )
    print(f"\ncalculate_anomaly_score 출력 shape: {tuple(scores.shape)}")
    print("점수 첫 5행:")
    print(scores[:5])
    print("채널별 (min, max):")
    for channel_index in range(3):
        channel_column = scores[:, channel_index]
        print(f"  채널 {channel_index}: ({float(channel_column.min()):.1f}, {float(channel_column.max()):.1f})")

    offset_only_at_channel_2 = bool(
        torch.equal(scores[:, 2], torch.full((scores.shape[0],), 500.0))
        and torch.equal(scores[:, 0], torch.zeros(scores.shape[0]))
        and torch.equal(scores[:, 1], torch.zeros(scores.shape[0]))
    )
    print(f"\n오프셋 500의 흔적이 채널 2 자리에만 정확히 나타나는가: {offset_only_at_channel_2}")


def report_label_alignment_through_shared_step(
    gdn_model, lightning_module, dataset_module
) -> None:
    print()
    print("=" * 72)
    print("[g] y 정렬: __getitem__ 의 y 가 loss 피연산자가 되기까지")
    print("=" * 72)

    from gragod import CleanMethods

    get_data_loader = dataset_module.get_data_loader  # datasets/dataset.py:97-145

    window_size = 4
    sample_count = 12
    # 값 = 시점*10 + 채널 (datasets/dataset.py:44-62 의 __getitem__ 검증용).
    labeled_series = torch.zeros(sample_count, 3)
    for time_index in range(sample_count):
        for channel_index in range(3):
            labeled_series[time_index, channel_index] = time_index * 10 + channel_index

    loader = get_data_loader(
        X=labeled_series,
        edge_index=gdn_model.edge_index_sets[0],
        y=torch.zeros(sample_count),
        window_size=window_size,
        clean=CleanMethods.NONE,
        batch_size=4,
        n_workers=0,
        shuffle=False,
    )

    window_batch, target_batch, _, _ = next(iter(loader))
    print(f"__getitem__ 배치: x shape {tuple(window_batch.shape)} (batch, window, feature),"
          f" y shape {tuple(target_batch.shape)} (batch, horizon=1, feature)")

    # shared_step(models/gdn/model.py:265-278) 재현: x 쪽은 교체 가설(permute) 대체,
    # y 쪽은 실제 문장 그대로 — y.squeeze(1) (models/gdn/model.py:272).
    fixed_input = apply_fixed_predict_step_transform(window_batch.float())
    target_after_squeeze = target_batch.float().squeeze(1)
    print(f"\ny.squeeze(1) 후 shape: {tuple(target_after_squeeze.shape)}")
    print("y 피연산자 값:")
    print(target_after_squeeze)

    # 기대값: 배치 i 의 y = 원시계열의 시점 (window_size + i) 행 = [(W+i)*10 + 0, +1, +2].
    expected_targets = torch.stack(
        [labeled_series[window_size + batch_index] for batch_index in range(target_after_squeeze.shape[0])]
    )
    print("기대값 (원시계열의 시점 W+i 행):")
    print(expected_targets)
    targets_match = bool(torch.equal(target_after_squeeze, expected_targets))
    print(f"y 피연산자 == 기대값 (채널·시점 혼합 없음): {targets_match}")

    with torch.no_grad():
        model_output = gdn_model(fixed_input)
    loss_value = lightning_module.criterion(model_output, target_after_squeeze)
    print(f"\nloss 피연산자 shape: out {tuple(model_output.shape)} vs y {tuple(target_after_squeeze.shape)}")
    print(f"criterion(MSELoss) 계산 성공, loss = {float(loss_value):.4f}")


def run_shape_flow_trace() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gragod-repository-path",
        required=True,
        help="GraGODs/GraGOD 클론의 루트 경로 (develop, ec8cd452 기준)",
    )
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.gragod_repository_path)

    print_environment_versions()
    print(f"GraGOD 경로: {arguments.gragod_repository_path}")

    install_datasets_package_stub()
    gdn_model, lightning_module = build_small_gdn_and_module(
        arguments.gragod_repository_path
    )
    from models.gdn.model import GDN  # 임포트 출처 증빙용
    print(f"임포트 확인: GDN <- {sys.modules['models.gdn.model'].__file__}")

    dataset_module = load_gragod_module_from_file(
        "gragod_dataset_module",
        os.path.join(arguments.gragod_repository_path, "datasets", "dataset.py"),
    )
    print(f"임포트 확인: get_data_loader <- {dataset_module.__file__}")

    report_forward_with_permuted_input(gdn_model)
    report_channel_alignment_through_score_path(gdn_model, lightning_module, dataset_module)
    report_label_alignment_through_shared_step(gdn_model, lightning_module, dataset_module)


if __name__ == "__main__":
    run_shape_flow_trace()
