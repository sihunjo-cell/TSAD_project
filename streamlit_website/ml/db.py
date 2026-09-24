"""`recommendation.sqlite3`에서 similarity·성능·비용 추정에 필요한 값만 읽는다.

`streamlit_website/db_connection/filter_candidates.py`와 같은 관례를 따른다:
읽기 전용 연결만 열고, 실행 이력이나 성능으로 후보를 거르지 않는다 (그건 이 ML
단계의 다른 함수들이 명시적으로 한다). 이 모듈은 DB access만 담당하고 계산은
하지 않는다.

실제 DB(`dev18_results_final/recommendation.sqlite3`) 확인 결과, `results` 테이블의
`metadata_file`이 가리키는 개별 실행 JSON은 로컬에 존재하지 않지만, 학습/추론
시간은 `cost_executions` 테이블에 이미 컬럼으로 들어있고 `results` → `cost_result_links`
→ `cost_executions` join으로 7,811/7,811건 전부 연결된다. 따라서 이 모듈은 metadata
JSON을 열지 않고 이 join만 사용한다.

DB 경로는 코드에 하드코딩하지 않는다 — 우선순위: 명시적 인자 > 환경변수
`TSAD_RECOMMENDATION_DB` > 리포지토리 관례상 기본 경로
(`experiments/tuning/results/recommendation.sqlite3`, git에 커밋되지 않는 실행 산출물).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from streamlit_website.ml.config import SIMILARITY_FEATURE_COLUMNS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH_ENV_VAR = "TSAD_RECOMMENDATION_DB"
DEFAULT_DATABASE_PATH = REPOSITORY_ROOT / "experiments/tuning/results/recommendation.sqlite3"


def resolve_database_path(database: str | Path | None = None) -> Path:
    """DB 경로를 인자 > 환경변수 > 기본 경로 순으로 정한다."""
    if database is not None:
        return Path(database)
    from_env = os.environ.get(DATABASE_PATH_ENV_VAR)
    if from_env:
        return Path(from_env)
    return DEFAULT_DATABASE_PATH


def _connect(database: str | Path | None):
    path = resolve_database_path(database)
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"recommendation DB를 찾을 수 없다: {path} "
            f"(환경변수 {DATABASE_PATH_ENV_VAR}로 다른 경로를 지정할 수 있다)"
        )
    return closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True))


def load_historical_prefixes(database: str | Path | None = None) -> list[dict]:
    """과거 Dev18 prefix마다 similarity feature와 규모(행 수)를 읽는다.

    행 수(observed_row/training_boundary)는 함께 반환하지만, 이 값을 similarity
    feature로 쓰는 것은 이 함수를 호출하는 쪽의 책임이 아니다 — `similarity.py`가
    명시적으로 행 수를 제외하고 벡터를 만든다 (원칙: 행 수는 similarity feature와 분리).
    """
    columns = ("prefix_feature_id", "csv_id", "csv_file", "family", "series", "q_percent",
               "training_boundary", "observed_row", *SIMILARITY_FEATURE_COLUMNS)
    with _connect(database) as connection:
        rows = connection.execute(
            f"SELECT {','.join(columns)} FROM prefix_features ORDER BY series, q_percent",
        ).fetchall()
    return [dict(zip(columns, row)) for row in rows]


def load_results_for_candidate(config_id: str, head: str, *, database: str | Path | None = None) -> list[dict]:
    """한 (config_id, head) 후보의 과거 실행 결과를, 성능과 비용을 함께 담아 읽는다.

    `head`는 DB의 `score_variant`에 대응한다 (TSPulse의 여러 head, 그 외 모델의
    단일 score_head는 빈 문자열). `results`는 이 DB에서 전부 `status='complete'`,
    `primary_score=1`이고 `vus_pr`이 채워져 있다고 확인했지만, 그 사실에 의존하지
    않고 상태를 함께 반환한다 — 성능/비용 계산 쪽에서 명시적으로 필터링한다.

    비용(`training_seconds`, `test_inference_seconds` 등)은 `cost_result_links`를
    거쳐 `cost_executions`에서 가져온다. `training-free` 모델처럼 여러 논리 q가
    같은 물리 실행(run_id)을 재사용하는 경우를 구분할 수 있도록 `run_id`와
    물리 실행의 `q_percent`(physical_q_percent)도 함께 반환한다.
    """
    query = """
        SELECT r.prefix_feature_id, p.series, p.csv_file, p.family, p.q_percent, p.observed_row,
               p.training_boundary, r.status, r.vus_pr,
               l.run_id, ce.status AS cost_status,
               ce.training_seconds, ce.test_inference_seconds, ce.split_preprocess_seconds,
               ce.model_setup_seconds, ce.calibration_inference_seconds,
               ce.actual_runtime_seconds, ce.available_training_rows, ce.test_observations,
               physical.q_percent AS physical_q_percent
        FROM results r
        JOIN prefix_features p USING(prefix_feature_id)
        LEFT JOIN cost_result_links l
            ON l.prefix_feature_id = r.prefix_feature_id AND l.config_id = r.config_id
            AND l.seed = r.seed AND l.score_variant = r.score_variant
        LEFT JOIN cost_executions ce ON ce.run_id = l.run_id
        LEFT JOIN prefix_features physical ON physical.prefix_feature_id = ce.physical_prefix_feature_id
        WHERE r.config_id = ? AND r.score_variant = ? AND r.primary_score = 1
        ORDER BY p.series, p.q_percent
    """
    columns = ("prefix_feature_id", "series", "csv_file", "family", "q_percent", "observed_row",
               "training_boundary", "status", "vus_pr", "run_id", "cost_status",
               "training_seconds", "test_inference_seconds", "split_preprocess_seconds",
               "model_setup_seconds", "calibration_inference_seconds", "actual_runtime_seconds",
               "available_training_rows", "test_observations", "physical_q_percent")
    with _connect(database) as connection:
        rows = connection.execute(query, (config_id, head)).fetchall()
    return [dict(zip(columns, row)) for row in rows]


def load_all_results_grouped_by_candidate(
    database: str | Path | None = None,
) -> dict[tuple[str, str], list[dict]]:
    """`load_results_for_candidate()`와 같은 join을, 후보 전체에 대해 **한 번의 쿼리**로 읽는다.

    후보(config_id, head)마다 커넥션을 새로 열고 쿼리를 날리면 후보 수에 선형으로
    비례해 느려진다 — 지금 DB(45개)에서는 후보당 약 15ms(전체 약 0.65초)로 눈에
    띄지 않지만, 팀 피드백 문서가 목표로 하는 미래/프로덕션 규모(~3,000개 후보)에서는
    그대로면 40초 이상 걸릴 수 있다. 이 함수는 WHERE 절 없이 같은 join을 한 번만
    실행하고 파이썬에서 (config_id, score_variant)별로 묶어, 결과는
    `load_results_for_candidate()`와 완전히 동일하되 후보 수와 무관하게 쿼리 1회로
    끝난다. N-sensitivity처럼 "후보 전체를 순회"해야 하는 호출부는 이 함수를 쓴다.
    """
    query = """
        SELECT r.config_id, r.score_variant,
               r.prefix_feature_id, p.series, p.csv_file, p.family, p.q_percent, p.observed_row,
               p.training_boundary, r.status, r.vus_pr,
               l.run_id, ce.status AS cost_status,
               ce.training_seconds, ce.test_inference_seconds, ce.split_preprocess_seconds,
               ce.model_setup_seconds, ce.calibration_inference_seconds,
               ce.actual_runtime_seconds, ce.available_training_rows, ce.test_observations,
               physical.q_percent AS physical_q_percent
        FROM results r
        JOIN prefix_features p USING(prefix_feature_id)
        LEFT JOIN cost_result_links l
            ON l.prefix_feature_id = r.prefix_feature_id AND l.config_id = r.config_id
            AND l.seed = r.seed AND l.score_variant = r.score_variant
        LEFT JOIN cost_executions ce ON ce.run_id = l.run_id
        LEFT JOIN prefix_features physical ON physical.prefix_feature_id = ce.physical_prefix_feature_id
        WHERE r.primary_score = 1
        ORDER BY r.config_id, r.score_variant, p.series, p.q_percent
    """
    columns = ("config_id", "score_variant", "prefix_feature_id", "series", "csv_file", "family",
               "q_percent", "observed_row", "training_boundary", "status", "vus_pr", "run_id",
               "cost_status", "training_seconds", "test_inference_seconds", "split_preprocess_seconds",
               "model_setup_seconds", "calibration_inference_seconds", "actual_runtime_seconds",
               "available_training_rows", "test_observations", "physical_q_percent")
    grouped: dict[tuple[str, str], list[dict]] = {}
    with _connect(database) as connection:
        for row in connection.execute(query):
            record = dict(zip(columns, row))
            key = (record.pop("config_id"), record.pop("score_variant"))
            grouped.setdefault(key, []).append(record)
    return grouped


def load_candidate_definitions(database: str | Path | None = None) -> dict[str, dict]:
    """`model_configs.settings_json`에서 `config_id`별 feasibility 판정용 정의를 읽는다.

    `streamlit_website/db_connection/filter_candidates.py::load_candidates()`와 같은
    JSON 파싱 관례를 따르되(파싱 실패 시 빈 값으로, 억지로 채우지 않음), head는
    펼치지 않는다 — `model`/`parameters`/`target_use`/`recipe`는 head가 아니라
    config_id 단위 속성이다. 반환값은 `feasibility.py::assess_future_feasibility()`가
    바로 받을 수 있는 `candidate` dict 모양(`model`, `parameters`, `target_use`, `recipe`)이다.
    """
    import json

    with _connect(database) as connection:
        rows = connection.execute("SELECT config_id, model, settings_json FROM model_configs").fetchall()
    definitions: dict[str, dict] = {}
    for config_id, model, serialized in rows:
        try:
            settings = json.loads(serialized)
        except (TypeError, json.JSONDecodeError):
            settings = {}
        if not isinstance(settings, dict):
            settings = {}
        parameters = settings.get("hyperparameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        definitions[config_id] = {
            "model": model, "config_id": config_id, "parameters": parameters,
            "target_use": settings.get("target_use", ""), "recipe": settings.get("common_recipe", {}),
        }
    return definitions


def load_candidate_target_use(database: str | Path | None = None) -> dict[str, str]:
    """`load_candidate_definitions()`에서 `target_use`만 뽑는다 (하위 호환용 얇은 wrapper)."""
    return {config_id: definition["target_use"] for config_id, definition in load_candidate_definitions(database).items()}


def load_channel_count_by_prefix(database: str | Path | None = None) -> dict[str, int]:
    """`channel_features`에서 `prefix_feature_id`별 채널(센서) 수를 센다.

    센서 수는 그 series의 등록 센서 구성이라 q_percent와 무관하게 일정하다(실제 DB로
    확인함) — 이 함수는 그 사실에 의존하지 않고 prefix_feature_id별로 그대로 센다.
    """
    with _connect(database) as connection:
        rows = connection.execute(
            "SELECT prefix_feature_id, COUNT(*) FROM channel_features GROUP BY prefix_feature_id",
        ).fetchall()
    return {prefix_feature_id: count for prefix_feature_id, count in rows}


def list_candidate_ids(database: str | Path | None = None) -> list[tuple[str, str]]:
    """DB에 결과가 존재하는 (config_id, head) 쌍 전체를 반환한다.

    `head`는 `score_variant` 컬럼이다 (TSPulse의 여러 head, 그 외 모델은 빈 문자열).
    후보 정의 자체(파라미터/실행 가능성)는 `model_configs`/`filter_candidates.py`가
    담당하므로, 여기서는 "실제로 평가 결과가 있는" 쌍만 나열한다 — held-out 검증처럼
    과거 결과를 근거로 candidate를 순회해야 하는 쪽에서 쓴다.
    """
    with _connect(database) as connection:
        rows = connection.execute(
            "SELECT DISTINCT config_id, score_variant FROM results ORDER BY config_id, score_variant",
        ).fetchall()
    return [(row[0], row[1]) for row in rows]
