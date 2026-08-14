"""exp01 — 분할 모듈 검증표 생성 (장난감 배열 전용, 실제 데이터셋 로드 없음).

근거: docs/plan_v4.md 7-1(216행)·7-2(218행), DECISIONS D-11(스냅숏은 CLAUDE.md 재현성).
front/back × 5비율 전부와 validation_split 결과를 표로 만들어
logs/split_check_table.txt 에 저장하고 화면에도 출력한다.
"""

import subprocess
import sys
from pathlib import Path

import numpy
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.save_scores import snapshot_config
from src.data_split.back_trim_split import back_trim_split
from src.data_split.front_trim_split import ALLOWED_RATIO_FRACTIONS, front_trim_split
from src.data_split.validation_split import InsufficientTrainLengthError, validation_split

EXPERIMENT_DIR = Path(__file__).resolve().parent
RATIOS = sorted(ALLOWED_RATIO_FRACTIONS)


def build_labeled_series(total_length: int, channel_count: int = 3) -> numpy.ndarray:
    """값 = 시점*10 + 채널."""
    time_index, channel_index = numpy.meshgrid(
        range(total_length), range(channel_count), indexing="ij"
    )
    return time_index * 10 + channel_index


def build_trim_table_lines() -> list[str]:
    lines = ["[표 1] front/back trim — 장난감 배열 3종 (N=3, 값 = 시점*10 + 채널)",
             f"{'T':>4} | {'direction':>9} | {'ratio':>5} | {'kept_length':>11} | {'kept_index_range':>16} | {'첫 원소':>8} | {'마지막 원소':>10}"]
    for total_length in (20, 19, 100):
        series = build_labeled_series(total_length)
        for direction, split_function in (("front", front_trim_split), ("back", back_trim_split)):
            for ratio in RATIOS:
                kept, split_info = split_function(series, ratio)
                lines.append(
                    f"{split_info['T']:>4} | {direction:>9} | {ratio:>5} | "
                    f"{split_info['kept_length']:>11} | {str(split_info['kept_index_range']):>16} | "
                    f"{kept[0, 0]:>8} | {kept[-1, -1]:>10}"
                )
    return lines


def build_validation_table_lines() -> list[str]:
    with open(REPOSITORY_ROOT / "configs" / "gdn_hyperparams.yaml", encoding="utf-8") as config_file:
        val_fraction = yaml.safe_load(config_file)["train_params"]["val_size"]
    min_train_length = 10  # 예시값 — 실제 하한은 EDA(E1) 후 결정 (validation_split 주석 참조)

    lines = ["", f"[표 2] validation_split — T=100, val_fraction={val_fraction}(configs val_size), min_train_length={min_train_length}(예시)",
             f"{'ratio':>5} | {'kept_length':>11} | {'train 길이':>9} | {'val 길이':>7} | {'val 첫 원소':>9} | {'val 마지막 원소':>11} | 결과"]
    series = build_labeled_series(100)
    for ratio in RATIOS:
        reduced, split_info = front_trim_split(series, ratio)
        try:
            train_part, val_part = validation_split(
                reduced, val_fraction=val_fraction,
                min_train_length=min_train_length, split_info=split_info,
            )
            lines.append(
                f"{ratio:>5} | {split_info['kept_length']:>11} | {train_part.shape[0]:>9} | "
                f"{val_part.shape[0]:>7} | {val_part[0, 0]:>9} | {val_part[-1, -1]:>11} | 정상"
            )
        except InsufficientTrainLengthError as error:
            lines.append(f"{ratio:>5} | {split_info['kept_length']:>11} | {'—':>9} | {'—':>7} | {'—':>9} | {'—':>11} | 예외")
            lines.append(f"      예외 메시지 전문: {error}")
    return lines


def read_git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=REPOSITORY_ROOT, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown(커밋 없음)"


def run_split_check() -> None:
    table_lines = build_trim_table_lines() + build_validation_table_lines()
    table_text = "\n".join(table_lines)

    logs_dir = EXPERIMENT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "split_check_table.txt").write_text(table_text + "\n", encoding="utf-8")

    snapshot_config(  # CLAUDE.md 재현성: 실행 config + git hash 를 snapshots/ 에 저장
        {"ratios": RATIOS, "toy_lengths": [20, 19, 100], "min_train_length_example": 10},
        read_git_hash(),
        str(EXPERIMENT_DIR / "snapshots"),
    )
    print(table_text)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_split_check()
