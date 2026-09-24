"""현재 누적 prefix의 similarity feature 벡터를 정리한다.

실제 feature 산식(mean/std/ACF/상관/spectral entropy 등)은 이미
`src/common/compute_prefix_features.py`에 구현돼 있고, Streamlit
`db_connection/filter_candidates.profile_normal_data()`가 이를 호출해
`ml_input["current_features"]["summary"]`에 채워 넣는다. 이 모듈은 그 결과를
similarity 계산이 바로 쓸 수 있는 벡터 형태로만 다시 정리한다 (중복 계산하지
않는다). 후보/모델에 독립적이므로 한 번 계산해 모든 historical case 비교에
재사용할 수 있다.

DB의 `prefix_features` 테이블도 같은 함수로 계산된 같은 컬럼명을 쓰므로
(`tests/tuning/store_recommendation_evidence.py::prepare_features()` 참고),
현재/과거 feature는 값을 그대로 나란히 비교할 수 있다.
"""

from __future__ import annotations

from streamlit_website.ml.config import SIMILARITY_FEATURE_COLUMNS


def extract_similarity_features(current_feature_summary: dict) -> tuple[dict, dict]:
    """`compute_prefix_features`의 summary에서 similarity feature 벡터를 뽑는다.

    행 수(observed_row)는 넣지 않는다 — summary에 있더라도 이 함수가 반환하는
    벡터에는 포함시키지 않는다 (원칙: 행 수는 similarity feature와 분리, 규모는
    별도로 다룬다).

    Returns:
        (features, reasons)
        features: {column: 값(float) or None}
        reasons: 값이 None인 column에 대해서만 {column: 사유 문자열}.
                 summary에 `f"{column}_reason"`이 있으면 그 값을, 없으면
                 "not_computed"를 사유로 쓴다. 이 함수는 None을 0으로 바꾸지 않는다.
    """
    if not isinstance(current_feature_summary, dict):
        raise ValueError("current_feature_summary는 mapping이어야 한다")
    features: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    for column in SIMILARITY_FEATURE_COLUMNS:
        if column not in current_feature_summary:
            raise ValueError(f"현재 prefix feature에 {column}이 없다 — extractor 계약이 달라졌는지 확인하라")
        value = current_feature_summary[column]
        if value is not None:
            features[column] = float(value)
        else:
            features[column] = None
            reasons[column] = current_feature_summary.get(f"{column}_reason") or "not_computed"
    return features, reasons


def extract_historical_features(prefix_row: dict) -> dict[str, float | None]:
    """`db.load_historical_prefixes()`가 반환한 한 행에서 같은 벡터를 뽑는다.

    DB 컬럼은 SQLite REAL이 NULL이면 파이썬 None으로 그대로 들어오므로 별도
    변환이 필요 없다. 여기서도 None을 대체하지 않는다.
    """
    features: dict[str, float | None] = {}
    for column in SIMILARITY_FEATURE_COLUMNS:
        if column not in prefix_row:
            raise ValueError(f"historical prefix row에 {column}이 없다")
        value = prefix_row[column]
        features[column] = None if value is None else float(value)
    return features
