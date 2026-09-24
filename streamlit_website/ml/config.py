"""ML 단계의 magic number를 한 곳에 모은 설정.

k, feature 목록, weighting 방법, 미래 단계 grid 등은 전부 여기서 조정한다.
값을 바꿀 때는 이 파일만 고치면 되고, 다른 모듈은 이 상수를 import해서 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS

# `src/common/compute_prefix_features.py`가 채우는 summary 컬럼 중 similarity에 쓸 목록.
# DB `prefix_features` 테이블의 컬럼명과 동일하다 (현재/과거 feature가 같은 산식으로 계산되므로).
# 행 수(observed_row)는 의도적으로 제외한다 — 규모는 별도의 scale 정보로 다룬다 (원칙).
SIMILARITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "channel_std_median",
    "channel_acf_lag1_median",
    "absolute_correlation_median",
    "channel_interquartile_range_median",
    "channel_difference_q90_iqr_ratio_median",
    "channel_median_shift_iqr_ratio_median",
    "channel_spectral_entropy_median",
)


@dataclass(frozen=True)
class MLConfig:
    """ML pipeline 전체에서 쓰는 조정 가능한 값들."""

    # distance-weighted kNN에서 사용할 이웃 수. 과거 사례가 이보다 적으면 있는 만큼만 쓴다.
    similarity_knn_k: int = 10

    # 0 거리(완전히 동일한 feature) 이웃의 weight 처리: 아주 작은 값으로 나누기 오류를 피한다.
    zero_distance_epsilon: float = 1e-9

    # 미래 단계를 정의하는 grid. 운영 기간(operating_days)에 대한 경과 비율(%)이며,
    # 기존 prefix q_percent(5,10,20,40,60,80,100) 관례를 "시간 경과율"로 재사용한다.
    # stage 0은 "지금"(elapsed_days=0)을 뜻하며 이 grid와 별도로 항상 포함된다.
    future_stage_elapsed_day_fractions: tuple[int, ...] = SUPPORTED_RATIO_PERCENTS

    # 단계별 신규 도입/재학습 후보를 이 개수로 줄인다. 후보 구성은 단계마다 달라질 수 있다.
    top_k_candidates: int = 10

    # 후보별 학습/추론 비용을 회귀할 때 필요한 최소 관측 개수(서로 다른 rows 값 기준).
    # 이 미만이면 estimable=False로 표시하고 임의의 증가율을 적용하지 않는다.
    minimum_cost_observations_for_regression: int = 2

    def __post_init__(self):
        if self.similarity_knn_k < 1:
            raise ValueError("similarity_knn_k는 1 이상이어야 한다")
        if self.top_k_candidates < 1:
            raise ValueError("top_k_candidates는 1 이상이어야 한다")
        if self.minimum_cost_observations_for_regression < 1:
            raise ValueError("minimum_cost_observations_for_regression은 1 이상이어야 한다")


DEFAULT_CONFIG = MLConfig()

# prediction provenance 상수
PREDICTION_SOURCE_OBSERVED = "observed"
PREDICTION_SOURCE_SIMILARITY = "estimated_from_similarity"
PREDICTION_SOURCE_EXTRAPOLATED = "extrapolated_held_constant"
PREDICTION_SOURCE_CHECKPOINT_MAINTAINED = "checkpoint_maintained"

# cost estimation source 상수
COST_SOURCE_REGRESSION = "model_execution_history_regression"
COST_SOURCE_SINGLE_OBSERVATION = "single_observation_flat_estimate"
COST_SOURCE_INSUFFICIENT = "insufficient_data"
