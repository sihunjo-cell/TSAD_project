"""exp00 의혹 2건 실행 검증 스크립트.

RECON.md에 기록된 의혹을 실행으로 확정한다. GDN 학습·데이터셋 로드·GraGOD 수정 없음.

의혹 1: GraGODs/GraGOD(develop, ec8cd452) models/gdn/model.py:271,304 의
        x.reshape(-1, x.size(2), x.size(1)) 가 의도된 전치 permute(0, 2, 1)와 다른가.
의혹 2: 같은 레포 gragod/predictions/prediction.py:108-124 의 smooth_scores 가
        시간 축이 아니라 feature 축에 작용하는가.
대조  : d-ailin/GDN(main, 9853899d) evaluate.py:62-65 의 후행 4-창 smoothing 재현.

실행 예:
    python verify_suspicions.py --gragod-repository-path <GraGOD 클론 경로>
"""

import argparse
import sys

import numpy
import torch

# Windows 콘솔(cp949)에서도 출력이 깨지지 않게 UTF-8로 고정한다.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def build_time_channel_labeled_windows(
    batch_count: int, window_size: int, feature_count: int
) -> torch.Tensor:
    """값만 보고 (배치, 시점, 채널)을 식별할 수 있는 윈도 텐서를 만든다.

    값 = 배치*1000 + 시점*10 + 채널.
    shape (batch_count, window_size, feature_count) —
    GraGOD SlidingWindowDataset(datasets/dataset.py:44-62)이 내놓는 배치 모양과 동일.
    """
    values = torch.zeros(batch_count, window_size, feature_count)
    for batch_index in range(batch_count):
        for time_index in range(window_size):
            for channel_index in range(feature_count):
                values[batch_index, time_index, channel_index] = (
                    batch_index * 1000 + time_index * 10 + channel_index
                )
    return values


def apply_gragod_reshape(windows: torch.Tensor) -> torch.Tensor:
    """GraGOD models/gdn/model.py:271 (및 304) 의 연산을 그대로 재현한다."""
    return windows.reshape(-1, windows.size(2), windows.size(1))


def apply_intended_permute(windows: torch.Tensor) -> torch.Tensor:
    """의도된 전치: (batch, window, feature) -> (batch, feature, window)."""
    return windows.permute(0, 2, 1)


def report_reshape_suspicion() -> None:
    print("=" * 72)
    print("[의혹 1] reshape(-1, size(2), size(1)) vs permute(0, 2, 1)")
    print("=" * 72)

    for window_size, feature_count, case_name in [
        (4, 3, "비정방 (window_size=4, n_features=3)"),
        (4, 4, "정방 (window_size=4, n_features=4)"),
    ]:
        windows = build_time_channel_labeled_windows(
            batch_count=2, window_size=window_size, feature_count=feature_count
        )
        reshaped = apply_gragod_reshape(windows)
        permuted = apply_intended_permute(windows)

        print(f"\n--- {case_name}, batch=2, 값 = 배치*1000 + 시점*10 + 채널 ---")
        print(f"원본 windows shape: {tuple(windows.shape)}  (batch, window, feature)")
        print("원본 windows[0] (행=시점, 열=채널):")
        print(windows[0].to(torch.int64))
        print(f"\n(a) GraGOD reshape 결과 shape: {tuple(reshaped.shape)}")
        print("reshape 결과[0] (모델이 '행=채널, 열=시점'으로 읽는 행렬):")
        print(reshaped[0].to(torch.int64))
        print(f"\n(b) 의도된 permute 결과 shape: {tuple(permuted.shape)}")
        print("permute 결과[0] (행=채널, 열=시점):")
        print(permuted[0].to(torch.int64))

        same_shape = reshaped.shape == permuted.shape
        elementwise_equal = same_shape and bool(torch.equal(reshaped, permuted))
        print(f"\nshape 동일 여부: {same_shape}")
        print(f"원소 단위 동일 여부: {elementwise_equal}")

        if elementwise_equal:
            print("판정 문장: 채널 j의 시계열로 읽히는 값이 실제로 채널 j의 값들이다.")
        else:
            first_channel_row = reshaped[0][0].to(torch.int64).tolist()
            print(
                "판정 문장: 채널 j의 시계열로 읽히는 값이 실제로 채널 j의 값들이 아니다."
                f" (예: reshape 결과에서 '채널 0의 시계열'로 읽히는 행 = {first_channel_row}"
                " -- 채널 표지(1의 자리)와 시점 표지(10의 자리)가 뒤섞여 있음)"
            )


def build_spiked_score_tensor() -> torch.Tensor:
    """축을 구분할 수 있는 점수 텐서 (n_samples=10, n_features=3).

    채널 0 = 전부 0, 채널 1 = 전부 100, 채널 2 = 전부 200에
    시간축 t=5 한 지점만 1000 (뾰족값).
    """
    scores = torch.zeros(10, 3)
    scores[:, 1] = 100.0
    scores[:, 2] = 200.0
    scores[5, 2] = 1000.0
    return scores


def replicate_dailin_trailing_smoothing(
    channel_scores: numpy.ndarray, trailing_window_length: int = 3
) -> numpy.ndarray:
    """d-ailin/GDN(main, 9853899d) evaluate.py:62-65 를 그대로 재현한 독립 함수.

    원본은 1차원(한 채널의 시간축) 배열에 작용한다:
        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3
        for i in range(before_num, len(err_scores)):
            smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])
    (원본에서 채널별 호출은 evaluate.py:16-20 의 feature 루프가 담당)
    """
    smoothed_scores = numpy.zeros(channel_scores.shape)
    for i in range(trailing_window_length, len(channel_scores)):
        smoothed_scores[i] = numpy.mean(
            channel_scores[i - trailing_window_length : i + 1]
        )
    return smoothed_scores


def apply_dailin_smoothing_per_channel(scores: torch.Tensor) -> torch.Tensor:
    """위 1차원 재현 함수를 채널마다 적용한다 (원본의 feature 루프에 해당)."""
    smoothed_columns = [
        replicate_dailin_trailing_smoothing(scores[:, channel_index].numpy())
        for channel_index in range(scores.shape[1])
    ]
    return torch.tensor(numpy.stack(smoothed_columns, axis=1), dtype=torch.float32)


def print_channel_table(title: str, tensors_by_label: dict[str, torch.Tensor]) -> None:
    """시점별 x 채널별 값을 표로 출력한다."""
    print(f"\n{title}")
    labels = list(tensors_by_label.keys())
    header_cells = ["t"] + [
        f"{label}·ch{channel_index}"
        for label in labels
        for channel_index in range(3)
    ]
    print(" | ".join(f"{cell:>14}" for cell in header_cells))
    sample_count = next(iter(tensors_by_label.values())).shape[0]
    for time_index in range(sample_count):
        row_cells = [f"{time_index:>14}"]
        for label in labels:
            for channel_index in range(3):
                value = float(tensors_by_label[label][time_index, channel_index])
                row_cells.append(f"{value:>14.1f}")
        print(" | ".join(row_cells))


def report_smoothing_suspicion(gragod_repository_path: str) -> None:
    print()
    print("=" * 72)
    print("[의혹 2] smooth_scores 가 어느 축에 작용하는가 (+ d-ailin 대조)")
    print("=" * 72)

    sys.path.insert(0, gragod_repository_path)
    # GraGOD 함수를 실제로 임포트한다 (재구현 대체 금지 조건).
    from gragod.predictions.prediction import smooth_scores

    print(f"임포트 확인: smooth_scores <- {sys.modules['gragod.predictions.prediction'].__file__}")

    spiked_scores = build_spiked_score_tensor()
    print("\n입력 (n_samples=10, n_features=3): 채널0=0, 채널1=100, 채널2=200, 단 t=5의 채널2=1000")
    print(spiked_scores)

    smoothing_window_size = 5  # GraGOD 설정값 window_size_smooth: 5 (models/gdn/params_swat.yaml:41)
    gragod_smoothed = smooth_scores(spiked_scores, window_size=smoothing_window_size)
    print(f"\nGraGOD smooth_scores(입력, window_size={smoothing_window_size}) 출력:")
    print(gragod_smoothed)

    # 판정 1: 시간 축 작용이라면 상수 채널은 어떤 평균 창에서도 원값 그대로여야 한다.
    constant_channel_preserved = bool(
        torch.equal(gragod_smoothed[:, 1], spiked_scores[:, 1])
    )
    # 판정 2: feature 축 작용이라면 출력 행은 replicate 패딩 후 feature 축 5-창
    # 평균의 손계산 값과 일치해야 한다. 스파이크 없는 행 기준:
    # ch0 = mean(0,0,0,0,0)=0, ch1 = mean(0,0,0,0,100)=20, ch2 = mean(0,0,0,100,200)=60.
    expected_feature_axis_row = [0.0, 20.0, 60.0]
    matches_feature_axis_hand_calc = (
        gragod_smoothed[0].tolist() == expected_feature_axis_row
    )
    # 판정 3: 시간 축 작용이라면 t=5의 뾰족값이 이웃 시점(t=6..9)으로 퍼져야 한다.
    spike_spread_over_time = bool(
        torch.any(gragod_smoothed[6:, 2] != gragod_smoothed[0, 2])
    )
    print(f"\n상수 채널1(전부 100)이 원값 그대로 유지되는가 (시간 축 작용이라면 True여야 함): {constant_channel_preserved}")
    print(
        f"출력 행이 feature 축 5-창 평균 손계산 값 {expected_feature_axis_row} 과 일치하는가"
        f" (feature 축 작용 증거): {matches_feature_axis_hand_calc}"
    )
    print(f"t=5 뾰족값이 시간축 이웃 t=6..9로 퍼졌는가 (시간 축 작용이라면 True여야 함): {spike_spread_over_time}")

    dailin_smoothed = apply_dailin_smoothing_per_channel(spiked_scores)
    print("\nd-ailin 재현(후행 4-창, 처음 3개 시점은 0) 출력:")
    print(dailin_smoothed)

    print_channel_table(
        "같은 입력에 대한 두 구현 비교 표 (GraGOD=smooth_scores, dailin=후행 4-창 재현):",
        {"입력": spiked_scores, "GraGOD": gragod_smoothed, "dailin": dailin_smoothed},
    )

    # 시간축 평활을 의도할 경우의 올바른 호출 형태(전치 후 통과, 전치 복원) 수치 확인.
    # GraGOD 코드는 수정하지 않고 호출 형태만 바꾼 것이다.
    transposed_call_result = smooth_scores(
        spiked_scores.T.contiguous(), window_size=smoothing_window_size
    ).T
    print("\n[참고] 시간축 평활 의도 시 호출 형태 smooth_scores(scores.T, w).T 의 출력:")
    print(transposed_call_result)
    channels_independent = bool(
        torch.equal(transposed_call_result[:, 0], spiked_scores[:, 0])
        and torch.equal(transposed_call_result[:, 1], spiked_scores[:, 1])
    )
    print(f"이 호출 형태에서 채널 0·1이 원값 그대로 유지되는가: {channels_independent}")


def run_verification() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gragod-repository-path",
        required=True,
        help="GraGODs/GraGOD 클론의 루트 경로 (develop, ec8cd452 기준)",
    )
    arguments = parser.parse_args()

    print(f"torch 버전: {torch.__version__}")
    print(f"GraGOD 경로: {arguments.gragod_repository_path}")

    report_reshape_suspicion()
    report_smoothing_suspicion(arguments.gragod_repository_path)


if __name__ == "__main__":
    run_verification()
