"""점수 배열 파일명 생성·역파싱.

근거: AGENTS.md 파일명 규약 — D-15 확장형 + D-16 __channels 보조 산출물.
{dataset}__{series}__{model}__{tier}__r{ratio}__s{seed}__{raw|smoothed}__{trainnorm|testnorm}(__channels).npy
series 두 자리, ratio 세 자리(005/010/020/050/100) 제로 패딩.
"""

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS as ALLOWED_RATIO_PERCENTS


ALLOWED_SMOOTHING_KINDS = ("raw", "smoothed")
ALLOWED_NORM_KINDS = ("trainnorm", "testnorm")


def build_score_filename(
    dataset: str,
    series: int,
    model: str,
    tier: str,
    ratio: int,
    seed: int,
    smoothing_kind: str,
    norm_kind: str,
    channels: bool = False,
) -> str:
    for field_name, field_value in (("dataset", dataset), ("model", model), ("tier", tier)):
        if "__" in str(field_value):
            raise ValueError(f"{field_name}에 구분자 '__'를 넣을 수 없다: {field_value!r}")
    if int(ratio) not in ALLOWED_RATIO_PERCENTS:
        raise ValueError(f"ratio는 {ALLOWED_RATIO_PERCENTS} 중 하나여야 한다: {ratio!r}")
    if smoothing_kind not in ALLOWED_SMOOTHING_KINDS:
        raise ValueError(f"smoothing_kind는 {ALLOWED_SMOOTHING_KINDS} 중 하나여야 한다: {smoothing_kind!r}")
    if norm_kind not in ALLOWED_NORM_KINDS:
        raise ValueError(f"norm_kind는 {ALLOWED_NORM_KINDS} 중 하나여야 한다: {norm_kind!r}")
    if not 0 <= int(series) <= 99:
        raise ValueError(f"series는 0~99(두 자리 제로 패딩)여야 한다: {series!r}")

    parts = [
        str(dataset),
        f"{int(series):02d}",
        str(model),
        str(tier),
        f"r{int(ratio):03d}",
        f"s{int(seed)}",
        smoothing_kind,
        norm_kind,
    ]
    if channels:
        parts.append("channels")
    return "__".join(parts) + ".npy"


def parse_score_filename(filename: str) -> dict:
    """build_score_filename 의 역함수. build(**parse(name)) == name 이 성립한다."""
    if not filename.endswith(".npy"):
        raise ValueError(f".npy 파일명이 아니다: {filename!r}")
    parts = filename[: -len(".npy")].split("__")

    channels = parts[-1] == "channels"
    if channels:
        parts = parts[:-1]
    if len(parts) != 8:
        raise ValueError(f"필드 수가 규약(8개)과 다르다: {filename!r}")

    dataset, series_text, model, tier, ratio_text, seed_text, smoothing_kind, norm_kind = parts
    if not (ratio_text.startswith("r") and ratio_text[1:].isdigit() and len(ratio_text) == 4):
        raise ValueError(f"ratio 필드 형식(r###)이 아니다: {ratio_text!r}")
    if not (seed_text.startswith("s") and seed_text[1:].isdigit()):
        raise ValueError(f"seed 필드 형식(s#)이 아니다: {seed_text!r}")
    if not (series_text.isdigit() and len(series_text) == 2):
        raise ValueError(f"series 필드는 두 자리 숫자여야 한다: {series_text!r}")

    parsed = {
        "dataset": dataset,
        "series": int(series_text),
        "model": model,
        "tier": tier,
        "ratio": int(ratio_text[1:]),
        "seed": int(seed_text[1:]),
        "smoothing_kind": smoothing_kind,
        "norm_kind": norm_kind,
        "channels": channels,
    }
    # 값 검증은 build 쪽 규칙을 재사용한다 (생성 불가능한 이름은 파싱도 거부).
    build_score_filename(**parsed)
    return parsed
