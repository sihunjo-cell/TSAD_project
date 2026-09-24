"""ML 단계의 실험/검증 harness (held-out 검증, N/k sensitivity 등).

이 서브패키지는 `run_ml_pipeline()`이 만드는 프로덕션 경로가 아니다 — 팀 피드백
문서가 요구한 검증/실험 구조를 담는다. `pipeline.py`의 정식 흐름(feasibility ->
top-k -> checkpoint)은 건드리지 않고, 이미 검증된 similarity/trajectory/performance
모듈만 재사용한다.
"""
