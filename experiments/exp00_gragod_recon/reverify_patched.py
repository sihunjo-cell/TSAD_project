"""exp00 — D-12 재검증: 패치된 포크(D-03)의 shared_step/predict_step 경로 검증.

검증 대상: gragod-fork (베이스 ec8cd452 + 패치 커밋, models/gdn/model.py 271·304행이
reshape(-1, x.size(2), x.size(1)) → permute(0, 2, 1).contiguous() 로 교체된 상태).

SHAPEFLOW의 trace_shape_flow.py와 동일한 입력 설계를 쓰되, 이번에는 스크립트가
변환을 대신하지 않고 패치된 shared_step / predict_step 경로 자체를 통과시킨다.
기대값의 원본은 SHAPEFLOW.md 2부의 실행 출력이다 (D-12).

고정 환경(python 3.10 + torch 2.2.2)에서는 datasets/config.py의 dataclass가
정상 임포트되므로, SHAPEFLOW 때 필요했던 스텁·파일 경로 우회 없이
GraGOD 패키지를 정식 경로로 임포트한다.

실행 예:
    python reverify_patched.py --fork-repository-path <gragod-fork 경로>
"""

import argparse
import sys

import torch

# Windows 콘솔(cp949)에서도 출력이 깨지지 않게 UTF-8로 고정한다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def print_environment_versions() -> None:
    import pytorch_lightning
    import torch_geometric

    print("[환경] 고정 환경 (D-13).")
    print(f"  python             : {sys.version.split()[0]}")
    print(f"  torch              : {torch.__version__}")
    print(f"  torch_geometric    : {torch_geometric.__version__}")
    print(f"  pytorch_lightning  : {pytorch_lightning.__version__} (저자 포크 git+gonzachiar@feature/best-k-metrics)")


def build_small_gdn_and_module():
    """SHAPEFLOW 2부와 동일: n_features=3, window=4, topk=2, embed 8."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    from datasets.graph import build_fully_connected_edge_index  # datasets/graph.py:46-66
    from models.gdn.model import GDN, GDN_PLModule

    placeholder_series = torch.zeros(1, 3)
    edge_index = build_fully_connected_edge_index(placeholder_series, device="cpu")

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


def report_forward_through_patched_predict_step(lightning_module) -> bool:
    print()
    print("=" * 72)
    print("[재검증 1 — e2 상당] 패치된 predict_step 경로로 forward 실행")
    print("=" * 72)

    labeled_windows = build_time_channel_labeled_windows(
        batch_count=2, window_size=4, feature_count=3
    )
    print(f"입력 (batch, window, feature) = {tuple(labeled_windows.shape)}")
    print("패치된 predict_step(models/gdn/model.py:292-305, 304행 = permute+contiguous) 호출:")
    with torch.no_grad():
        output = lightning_module.predict_step(labeled_windows, 0)
    print(f"forward 성공, 출력 shape: {tuple(output.shape)}")
    expected_shape = (2, 3)
    shape_ok = tuple(output.shape) == expected_shape
    print(f"출력 shape == (batch, n_features) == {expected_shape} 인가: {shape_ok}")
    return shape_ok


def zero_model_parameters_keep_embedding(gdn_model) -> None:
    """SHAPEFLOW f와 동일: 예측을 항등적으로 0으로 고정해 정렬 판정을 분리한다.

    embedding 만 0이 아닌 값으로 되돌린다 — cos-sim(model.py:139-143)의
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


def report_channel_alignment_through_patched_path(gdn_model, lightning_module) -> bool:
    print()
    print("=" * 72)
    print("[재검증 2 — f 상당] 채널 오프셋이 점수의 채널 2 자리에만 나타나는가")
    print("=" * 72)

    from gragod import CleanMethods
    from datasets.dataset import get_data_loader  # datasets/dataset.py:97-145

    zero_model_parameters_keep_embedding(gdn_model)

    window_size = 4
    sample_count = 30
    offset_series = torch.zeros(sample_count, 3)
    offset_series[:, 2] = 500.0
    labels = torch.zeros(sample_count)
    print(f"입력 시계열 shape: {tuple(offset_series.shape)}, 채널2만 500.0, 나머지 0.0")

    # X_true 구성은 models/predict.py process_dataset 의 실제 문장을 그대로 재현.
    # start_index 최솟값 = window_size (models/predict.py:441-444).
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

    # 예측은 패치된 predict_step 경로 자체를 통과 (스크립트 대체 변환 없음).
    prediction_batches = []
    with torch.no_grad():
        for batch in loader:
            prediction_batches.append(lightning_module.predict_step(batch, 0))

    # post_process_predictions + calculate_anomaly_score 는 실제 메서드 그대로
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
    return offset_only_at_channel_2


def report_label_alignment_through_patched_shared_step(
    gdn_model, lightning_module
) -> bool:
    print()
    print("=" * 72)
    print("[재검증 3 — g 상당] 패치된 shared_step 경로에서 y 정렬 유지")
    print("=" * 72)

    from gragod import CleanMethods
    from datasets.dataset import get_data_loader

    window_size = 4
    sample_count = 12
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

    batch = next(iter(loader))
    window_batch, target_batch, _, _ = batch
    print(f"__getitem__ 배치: x shape {tuple(window_batch.shape)} (batch, window, feature),"
          f" y shape {tuple(target_batch.shape)} (batch, horizon=1, feature)")

    # y 피연산자 검사: shared_step 의 실제 문장(y.squeeze(1), model.py:272)과 동일.
    target_after_squeeze = target_batch.float().squeeze(1)
    print(f"\ny.squeeze(1) 후 shape: {tuple(target_after_squeeze.shape)}")
    print("y 피연산자 값:")
    print(target_after_squeeze)
    expected_targets = torch.stack(
        [labeled_series[window_size + batch_index] for batch_index in range(target_after_squeeze.shape[0])]
    )
    print("기대값 (원시계열의 시점 W+i 행):")
    print(expected_targets)
    targets_match = bool(torch.equal(target_after_squeeze, expected_targets))
    print(f"y 피연산자 == 기대값 (채널·시점 혼합 없음): {targets_match}")

    # 패치된 shared_step 자체를 통과: 모델이 0으로 고정돼 있으므로 out ≡ 0,
    # 따라서 shared_step 이 반환하는 loss 는 mean(y²)와 정확히 같아야 한다.
    with torch.no_grad():
        loss_from_patched_shared_step = lightning_module.shared_step(batch, 0)
    expected_loss = (expected_targets ** 2).mean()
    print(f"\n패치된 shared_step(models/gdn/model.py:265-278, 271행 = permute+contiguous) 반환 loss: "
          f"{float(loss_from_patched_shared_step):.4f}")
    print(f"기대 loss = mean(기대 y²) = {float(expected_loss):.4f}")
    loss_match = bool(torch.allclose(loss_from_patched_shared_step, expected_loss))
    print(f"shared_step loss == 기대 loss (y 피연산자가 내부에서도 동일함의 증거): {loss_match}")
    return targets_match and loss_match


def run_reverification() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fork-repository-path",
        required=True,
        help="패치된 gragod-fork 루트 경로 (베이스 ec8cd452 + D-03 패치)",
    )
    arguments = parser.parse_args()
    sys.path.insert(0, arguments.fork_repository_path)

    print_environment_versions()
    print(f"포크 경로: {arguments.fork_repository_path}")

    gdn_model, lightning_module = build_small_gdn_and_module()
    from models.gdn import model as patched_model_module
    print(f"임포트 확인: GDN_PLModule <- {patched_model_module.__file__}")

    forward_ok = report_forward_through_patched_predict_step(lightning_module)
    channel_ok = report_channel_alignment_through_patched_path(gdn_model, lightning_module)
    label_ok = report_label_alignment_through_patched_shared_step(gdn_model, lightning_module)

    print()
    print("=" * 72)
    print(f"[종합] (1) forward/shape: {'통과' if forward_ok else '실패'}"
          f" / (2) 채널 정렬: {'통과' if channel_ok else '실패'}"
          f" / (3) y 정렬: {'통과' if label_ok else '실패'}")
    print("=" * 72)
    if not (forward_ok and channel_ok and label_ok):
        sys.exit(1)


if __name__ == "__main__":
    run_reverification()
