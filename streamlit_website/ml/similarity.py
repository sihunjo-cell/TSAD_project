"""feature standardization + Euclidean distance + distance-weighted kNN.

설계 원칙 (docstring에 명시적으로 기록):

- **표준화 통계는 과거(historical) prefix 전체를 기준으로 계산한다.** 현재 prefix 1개만
  으로는 분산을 추정할 수 없기 때문이다. 현재 prefix의 feature 값도 이 통계로
  표준화해서 과거 값과 같은 척도에서 비교한다.
- **표준화 통계는 population mean/std(ddof=0)**를 쓴다. `compute_prefix_features.py`가
  채널 std를 계산할 때도 ddof=0을 쓰므로(EXTRACTOR_CONTRACT) 관례를 맞췄다.
- 어떤 feature 컬럼이 과거 전체에서 전부 NULL이면 그 컬럼은 표준화할 수 없다
  (mean/std를 정의할 수 없음) — 이런 컬럼은 거리 계산에서 아예 제외한다.
- 어떤 feature 컬럼의 과거 분산이 0이면(전부 같은 값) 그 컬럼은 어떤 두 prefix도
  구분하지 못하므로 표준화값을 0으로 두고 거리에 포함시키되 실질적인 기여는 없다.
- 거리 계산은 **두 prefix 모두에서 값이 있는 feature만** 사용한다 (NULL을 0으로
  대체하지 않는다). 공통으로 쓸 수 있는 feature가 하나도 없으면 그 historical
  prefix는 비교 불가로 제외한다 (거리를 억지로 만들지 않는다).
- 거리 0(완전히 동일한 표준화 벡터)은 `zero_distance_epsilon`을 더해 나누기
  오류를 피한다.
- k보다 사용 가능한 historical prefix가 적으면 있는 만큼만 반환한다 (실패시키지 않는다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from streamlit_website.ml.config import DEFAULT_CONFIG, SIMILARITY_FEATURE_COLUMNS, MLConfig
from streamlit_website.ml.features import extract_historical_features
from streamlit_website.ml.schemas import SimilarityMatch


@dataclass(frozen=True)
class FeatureStandardizer:
    """과거 prefix 전체에서 계산한 컬럼별 (mean, std)."""

    stats: dict[str, tuple[float, float]]  # column -> (mean, std>0)
    zero_variance_columns: frozenset[str]
    insufficient_data_columns: frozenset[str]  # 과거 전체에서 값이 하나도 없는 컬럼

    def standardize(self, features: dict[str, float | None]) -> dict[str, float | None]:
        """원본 feature 벡터를 이 통계 기준으로 표준화한다. None은 그대로 None."""
        result: dict[str, float | None] = {}
        for column in SIMILARITY_FEATURE_COLUMNS:
            value = features.get(column)
            if value is None or column in self.insufficient_data_columns:
                result[column] = None
            elif column in self.zero_variance_columns:
                result[column] = 0.0
            else:
                mean, std = self.stats[column]
                result[column] = (value - mean) / std
        return result


def build_standardizer(historical_features_list: list[dict[str, float | None]]) -> FeatureStandardizer:
    """과거 prefix 전체(모든 series/q)에서 컬럼별 population mean/std를 계산한다."""
    stats: dict[str, tuple[float, float]] = {}
    zero_variance: set[str] = set()
    insufficient: set[str] = set()
    for column in SIMILARITY_FEATURE_COLUMNS:
        values = [row[column] for row in historical_features_list if row.get(column) is not None]
        if not values:
            insufficient.add(column)
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std == 0:
            zero_variance.add(column)
        else:
            stats[column] = (mean, std)
    return FeatureStandardizer(
        stats=stats, zero_variance_columns=frozenset(zero_variance),
        insufficient_data_columns=frozenset(insufficient),
    )


def euclidean_distance(
    current_standardized: dict[str, float | None], historical_standardized: dict[str, float | None],
) -> tuple[float | None, tuple[str, ...]]:
    """두 표준화 벡터 사이의 Euclidean distance. 공통으로 값이 있는 feature만 쓴다.

    Returns (distance, features_used). 공통 feature가 하나도 없으면 (None, ()).
    """
    used = []
    total = 0.0
    for column in SIMILARITY_FEATURE_COLUMNS:
        a, b = current_standardized.get(column), historical_standardized.get(column)
        if a is None or b is None:
            continue
        used.append(column)
        total += (a - b) ** 2
    if not used:
        return None, ()
    return math.sqrt(total), tuple(used)


def find_similar_historical_prefixes(
    current_features: dict[str, float | None],
    historical_prefixes: list[dict],
    *,
    config: MLConfig = DEFAULT_CONFIG,
) -> tuple[list[SimilarityMatch], FeatureStandardizer]:
    """현재 prefix와 가장 유사한 과거 prefix들을 distance-weighted kNN으로 찾는다.

    `historical_prefixes`는 `db.load_historical_prefixes()`의 반환값이다.
    반환된 리스트는 거리 오름차순으로, 최대 `config.similarity_knn_k`개다.
    """
    if not historical_prefixes:
        raise ValueError("과거 prefix가 없다 — DB가 비어있는지 확인하라")
    historical_feature_rows = [(row, extract_historical_features(row)) for row in historical_prefixes]
    standardizer = build_standardizer([features for _, features in historical_feature_rows])
    current_standardized = standardizer.standardize(current_features)

    scored = []
    for row, hist_features in historical_feature_rows:
        hist_standardized = standardizer.standardize(hist_features)
        distance, features_used = euclidean_distance(current_standardized, hist_standardized)
        if distance is None:
            continue  # 공통으로 비교할 feature가 없음 — 거리를 억지로 만들지 않는다
        scored.append((distance, features_used, row))

    scored.sort(key=lambda item: item[0])
    neighbors = scored[: config.similarity_knn_k]
    matches = []
    for distance, features_used, row in neighbors:
        weight = 1.0 / (distance + config.zero_distance_epsilon)
        matches.append(SimilarityMatch(
            prefix_feature_id=row["prefix_feature_id"], series=row["series"], csv_file=row["csv_file"],
            family=row["family"], q_percent=row["q_percent"], observed_row=row["observed_row"],
            distance=distance, weight=weight, features_used=features_used,
        ))
    return matches, standardizer
