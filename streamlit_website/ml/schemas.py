"""ML input/output을 명시적으로 정의하는 schema.

레포지토리 다른 곳(`streamlit_website/db_connection/`, `src/common/`)은 dataclass나
pydantic을 쓰지 않고 plain dict + 명시적 검증(`raise ValueError`) 관례를 따른다.
하지만 DP 팀이 이 ML 모듈의 내부 구현을 몰라도 되도록 ML → DP 경계만큼은
`dataclass`로 명시한다 (표준 라이브러리만 사용, 새 의존성 없음). DP는 이
dataclass들이 감싸는 값만 읽으면 되고, 그 값을 만드는 방법은 몰라도 된다.

`MLInput`은 `streamlit_website/db_connection/filter_candidates.py::build_ml_input()`이
실제로 만드는 dict 구조를 그대로 감싼다 — 새 필드를 요구하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# ML Input — 기존 build_ml_input()의 산출물을 감싸는 얇은 wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLInput:
    """`streamlit_website/db_connection/filter_candidates.py::build_ml_input()`의 산출물.

    이 dataclass는 그 dict의 필드를 그대로 옮긴 것이며, 새로운 데이터를 요구하지
    않는다. `MLInput.from_ml_input_dict()`로 기존 dict에서 바로 만들 수 있다.
    """

    row_count: int
    channel_count: int
    sensor_columns: list[str]
    normal_prefix: pd.DataFrame
    current_model: dict[str, Any]
    operating_conditions: dict[str, Any]
    candidates: list[dict[str, Any]]
    current_features: dict[str, Any] | None = None

    @staticmethod
    def from_ml_input_dict(ml_input: dict) -> "MLInput":
        """`st.session_state["ml_input"]`(=`build_ml_input()` 반환값)에서 만든다."""
        if not isinstance(ml_input, dict):
            raise ValueError("ml_input은 mapping이어야 한다")
        try:
            input_section = ml_input["input"]
            normal_prefix = ml_input["normal_prefix"]
            current_model = ml_input["current_model"]
            operating_conditions = ml_input["operating_conditions"]
            candidates = ml_input["candidates"]
        except KeyError as error:
            raise ValueError(f"ml_input에 필수 필드가 없다: {error.args[0]}") from error
        return MLInput(
            row_count=int(input_section["row_count"]),
            channel_count=int(input_section["channel_count"]),
            sensor_columns=list(input_section["sensor_columns"]),
            normal_prefix=normal_prefix,
            current_model=dict(current_model),
            operating_conditions=dict(operating_conditions),
            candidates=list(candidates),
            current_features=ml_input.get("current_features"),
        )


# ---------------------------------------------------------------------------
# 중간 산출물
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarityMatch:
    """현재 prefix와 유사한 과거 Dev18 prefix 한 건."""

    prefix_feature_id: str
    series: str
    csv_file: str
    family: str
    q_percent: int
    observed_row: int
    distance: float
    weight: float
    features_used: tuple[str, ...]
    """이 거리 계산에 실제로 쓰인 feature 컬럼 (양쪽 모두 not-null인 것만)."""


@dataclass(frozen=True)
class FutureStagePlan:
    """운영 한 구간의 시작 행 수와 구간 추론량.

    stage는 elapsed_days에 시작하며 stage=0은 지금 시작한다. future_rows는 구간이
    시작할 때 확보한 행 수, inference_volume은 그 구간 동안 처리할 추론 행 수다.
    ratio_percent는 운영 기간 끝까지 모을 행 수를 100으로 본 DB 등록 비율(q)이다.
    """

    stage: int
    elapsed_days: float
    future_rows: int
    inference_volume: float
    ratio_percent: int = 100
    scenario: str = "data_generation_characteristics_unchanged"


# ---------------------------------------------------------------------------
# ML Output — DP가 소비하는 최종 스키마
# ---------------------------------------------------------------------------


@dataclass
class CandidateEstimate:
    """DP가 받는 (stage, candidate) 한 행. 신규 도입/재학습 후보용."""

    stage: int
    candidate_id: str
    model: str
    configuration: str
    head: str

    feasible: bool

    predicted_performance: float | None
    performance_lower_bound: float | None

    estimated_training_cost_seconds: float | None
    estimated_inference_cost_seconds: float | None
    estimated_total_cost_seconds: float | None

    data_rows: int
    inference_volume: float

    prediction_source: str | None
    cost_estimation_source: str | None

    exclusion_reason: str | None = None


@dataclass
class CheckpointMaintenanceOption:
    """기존 checkpoint를 유지하는 선택지. top-k 신규 후보와 별도로 보존한다."""

    stage: int
    candidate_id: str
    model: str
    configuration: str
    head: str
    checkpoint_reference: str | None

    feasible: bool

    predicted_performance: float | None
    performance_lower_bound: float | None

    estimated_inference_cost_seconds: float | None
    estimated_total_cost_seconds: float | None

    data_rows: int
    inference_volume: float

    prediction_source: str
    cost_estimation_source: str | None

    exclusion_reason: str | None = None


@dataclass
class MLOutput:
    """`run_ml_pipeline`의 최종 output. `dp_input.build_dp_input`이 DP 입력으로 옮긴다."""

    current_features: dict | None
    current_feature_reasons: dict | None
    similarity_matches: list[SimilarityMatch]
    future_stages: list[FutureStagePlan]

    # stage -> 실행할 수 있고 성능·비용을 추정한 신규 도입/재학습 후보 전부
    stage_candidates: dict[int, list[CandidateEstimate]]

    # stage -> 실행할 수 없거나 성능·비용을 추정하지 못한 후보 (exclusion_reason 보존)
    stage_candidates_excluded: dict[int, list[CandidateEstimate]]

    # 기존 checkpoint 유지 선택지. 신규 후보와 별개의 트랙이다.
    checkpoint_maintenance_options: list[CheckpointMaintenanceOption]

    metadata: dict = field(default_factory=dict)
