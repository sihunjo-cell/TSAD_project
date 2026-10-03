"""ML 단계의 magic number를 한 곳에 모은 설정.

k, feature 목록, weighting 방법 등은 전부 여기서 조정한다.
값을 바꿀 때는 이 파일만 고치면 되고, 다른 모듈은 이 상수를 import해서 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

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


# similarity.py가 지원하는 거리/유사도 방법. "euclidean"이 V1 기본값(README 근거:
# "특징 표준화와 Euclidean 거리, 거리 가중 kNN을 첫 비교안으로 둔다"). "cosine"은
# 비교 실험용으로 추가한 대안이며, 어느 쪽이 더 나은지는 문서 근거가 없으므로
# 기본값을 바꾸지 않는다 — sensitivity 실험에서 둘을 나란히 비교하기 위한 옵션이다.
SIMILARITY_METRIC_EUCLIDEAN = "euclidean"
SIMILARITY_METRIC_COSINE = "cosine"
SUPPORTED_SIMILARITY_METRICS = (SIMILARITY_METRIC_EUCLIDEAN, SIMILARITY_METRIC_COSINE)


@dataclass(frozen=True)
class MLConfig:
    """ML pipeline 전체에서 쓰는 조정 가능한 값들."""

    # distance-weighted kNN에서 사용할 이웃 수. 과거 사례가 이보다 적으면 있는 만큼만 쓴다.
    # 30은 held-out(파일·family 제외) 검증에서 10보다 절대 오차와 선택 손실이 작았던 값이다 (README).
    similarity_knn_k: int = 30

    # similarity 거리 방법. SUPPORTED_SIMILARITY_METRICS 중 하나. 기본값(euclidean)은
    # 바꾸지 않는다 — cosine은 실험(k/N sensitivity와 같은 성격의 비교 실험)용이다.
    similarity_metric: str = SIMILARITY_METRIC_EUCLIDEAN

    # 0 거리(완전히 동일한 feature) 이웃의 weight 처리: 아주 작은 값으로 나누기 오류를 피한다.
    zero_distance_epsilon: float = 1e-9

    # PaAno 행 수 선형회귀에서 기울기를 추정할 최소 관측 개수(서로 다른 rows 값 기준).
    # 이 미만이면 관측 평균의 수평선으로 둔다.
    minimum_cost_observations_for_regression: int = 2

    def __post_init__(self):
        if self.similarity_knn_k < 1:
            raise ValueError("similarity_knn_k는 1 이상이어야 한다")
        if self.similarity_metric not in SUPPORTED_SIMILARITY_METRICS:
            raise ValueError(f"similarity_metric은 {SUPPORTED_SIMILARITY_METRICS} 중 하나여야 한다")
        if self.minimum_cost_observations_for_regression < 1:
            raise ValueError("minimum_cost_observations_for_regression은 1 이상이어야 한다")


DEFAULT_CONFIG = MLConfig()

# prediction provenance 상수
PREDICTION_SOURCE_OBSERVED = "observed"
PREDICTION_SOURCE_SIMILARITY = "estimated_from_similarity"
PREDICTION_SOURCE_EXTRAPOLATED = "extrapolated_held_constant"
PREDICTION_SOURCE_CHECKPOINT_MAINTAINED = "checkpoint_maintained"

# cost estimation source 상수
COST_SOURCE_CELL_RATE = "median_seconds_per_cell"
COST_SOURCE_REGRESSION = "model_execution_history_regression"
COST_SOURCE_SINGLE_OBSERVATION = "single_observation_flat_estimate"
COST_SOURCE_INSUFFICIENT = "insufficient_data"
