"""ML part: similarity + 성능·비용 추정.

DB/Streamlit → (이 패키지) → DP/optimization 사이의 중간 인터페이스를 구현한다.
이 패키지는 similarity 계산, 과거 사례 검색, 성능·비용 추정, feasibility 판정,
단계별 후보 축소까지만 담당하며 경로 최적화(DP)는 하지 않는다.
진입점은 :func:`pipeline.run_ml_pipeline`.
"""

from streamlit_website.ml.pipeline import run_ml_pipeline

__all__ = ["run_ml_pipeline"]
