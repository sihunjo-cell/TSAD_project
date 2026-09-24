"""held-out validation harness에 대한 테스트.

검증 대상:
1. leave-series-out: held-out series 자신의 데이터는 similarity match/예측에 전혀
   쓰이지 않는다 (ground truth로만 쓰인다).
2. ground truth 정의: held-out series의 실제 q=100 vus_pr 상위 top_n개.
3. Recall@N이 예측/ground truth가 일치/불일치하는 합성 시나리오에서 정확히 계산됨.
4. N-sensitivity sweep이 N마다 올바른 top_n/reduction_ratio를 낸다.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from streamlit_website.ml.config import SIMILARITY_METRIC_COSINE, SIMILARITY_METRIC_EUCLIDEAN, MLConfig
from streamlit_website.ml.experiments.holdout_validation import (
    compare_similarity_metrics, evaluate_holdout, sweep_k_sensitivity, sweep_n_sensitivity,
)
from streamlit_website.ml.experiments.stress_test_scale import _build_synthetic_database
from streamlit_website.ml.test_ml_pipeline import _create_schema


def _insert_prefix(connection, prefix_feature_id, series, q_percent, observed_row):
    connection.execute(
        "INSERT INTO prefix_features VALUES (?,?,?,?,?,?,?,?,1.0,0.5,0.3,2.0,0.4,0.2,0.6)",
        (prefix_feature_id, f"csv-{prefix_feature_id}", f"{prefix_feature_id}.csv", "famA", series,
         q_percent, observed_row, observed_row),
    )


def _insert_result(connection, prefix_feature_id, config_id, vus_pr):
    connection.execute(
        "INSERT INTO results VALUES (?,?,1,'',1,'complete',?)", (prefix_feature_id, config_id, vus_pr),
    )


def _insert_cost(connection, prefix_feature_id, config_id, run_id, *, test_observations, test_inference_seconds):
    """`_insert_result`가 쓰는 seed=1, score_variant=''과 맞춰 cost를 연결한다."""
    connection.execute(
        "INSERT INTO cost_result_links VALUES (?,?,1,'',?)", (prefix_feature_id, config_id, run_id),
    )
    connection.execute(
        "INSERT INTO cost_executions VALUES (?,?,'complete',NULL,?,NULL,NULL,NULL,NULL,NULL,?)",
        (run_id, prefix_feature_id, test_inference_seconds, test_observations),
    )


def _insert_model_config(connection, config_id, model, target_use):
    import json
    connection.execute(
        "INSERT INTO model_configs VALUES (?,?,?)",
        (config_id, model, json.dumps({"target_use": target_use})),
    )


class TestHoldoutEvaluation(unittest.TestCase):
    """s1을 held-out으로, s2/s3를 historical pool로 쓰는 합성 DB.

    c1: s2/s3 @100 vus_pr=0.8 (예측 근거) / s1 @100(ground truth)=0.9 -> 예측 0.8.
    c2: s2/s3 @100 vus_pr=0.3 (예측 근거) / s1 @100(ground truth)=0.5 -> 예측 0.3.
    ground truth 순위: c1(0.9) > c2(0.5). 예측 순위: c1(0.8) > c2(0.3) -> 방향 일치.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "holdout.sqlite3"
        with sqlite3.connect(self.database) as connection:
            _create_schema(connection)
            _insert_prefix(connection, "s1-20", "s1", 20, 20)
            _insert_prefix(connection, "s1-100", "s1", 100, 100)
            _insert_prefix(connection, "s2-20", "s2", 20, 20)
            _insert_prefix(connection, "s2-100", "s2", 100, 100)
            _insert_prefix(connection, "s3-20", "s3", 20, 20)
            _insert_prefix(connection, "s3-100", "s3", 100, 100)
            _insert_result(connection, "s1-100", "c1", 0.9)
            _insert_result(connection, "s1-100", "c2", 0.5)
            _insert_result(connection, "s2-100", "c1", 0.8)
            _insert_result(connection, "s2-100", "c2", 0.3)
            _insert_result(connection, "s3-100", "c1", 0.8)
            _insert_result(connection, "s3-100", "c2", 0.3)
            # cost 데이터가 없으면 candidate_selection.py와 동일한 usability 기준(feasible+
            # performance+cost 모두 있어야 함) 때문에 top-N 경쟁에서 전부 빠진다 — 이 클래스는
            # cost 자체가 아니라 성능 기반 순위/leave-series-out을 테스트하므로, 순위에 영향을
            # 주지 않는 동일한 비용을 c1/c2 모두에 붙여 usable하게만 만든다.
            _insert_cost(connection, "s2-100", "c1", "run-c1", test_observations=100, test_inference_seconds=5.0)
            _insert_cost(connection, "s2-100", "c2", "run-c2", test_observations=100, test_inference_seconds=5.0)

    def test_ground_truth_uses_held_out_series_actual_q100(self):
        result = evaluate_holdout("s1", 20, database=self.database, top_n=1)
        self.assertEqual(result.ground_truth_top_n, ("c1::",))  # s1의 실제 q=100: c1(0.9) > c2(0.5)

    def test_prediction_never_uses_held_out_series_own_data(self):
        # 예측은 s2/s3(둘 다 0.8)의 평균이어야 한다 — s1 자신의 실제값(0.9)과 다르다는 것이
        # leave-series-out이 실제로 지켜지고 있다는 증거다.
        result = evaluate_holdout("s1", 20, database=self.database, top_n=2)
        # historical pool은 s2/s3의 q=20,100 행 전부(4개)를 포함한다 — s1 자신의 행(2개)은
        # leave-series-out으로 전부 제외된다 (전체 6개 중 4개만 매치).
        self.assertEqual(result.similarity_matches_found, 4)
        # predicted_top_n의 순서(c1이 c2보다 예측 성능이 높음)로 간접 확인:
        self.assertEqual(result.predicted_top_n[0], "c1::")

    def test_recall_at_n_is_one_when_predicted_and_ground_truth_agree(self):
        result = evaluate_holdout("s1", 20, database=self.database, top_n=1)
        self.assertEqual(result.predicted_top_n, ("c1::",))
        self.assertEqual(result.recall_at_n, 1.0)

    def test_recall_at_n_is_zero_when_predicted_and_ground_truth_disagree(self):
        # c2가 예측에서는 1등이 되도록 새 DB를 만든다 (s2/s3에서 c2가 c1보다 높게),
        # 하지만 실제(s1) ground truth는 여전히 c1이 1등이다.
        database = Path(self.temporary.name) / "holdout_disagree.sqlite3"
        with sqlite3.connect(database) as connection:
            _create_schema(connection)
            _insert_prefix(connection, "s1-20", "s1", 20, 20)
            _insert_prefix(connection, "s1-100", "s1", 100, 100)
            _insert_prefix(connection, "s2-20", "s2", 20, 20)
            _insert_prefix(connection, "s2-100", "s2", 100, 100)
            _insert_result(connection, "s1-100", "c1", 0.9)  # ground truth: c1이 1등
            _insert_result(connection, "s1-100", "c2", 0.5)
            _insert_result(connection, "s2-100", "c1", 0.2)  # 예측 근거: c2가 1등
            _insert_result(connection, "s2-100", "c2", 0.7)
            _insert_cost(connection, "s2-100", "c1", "run-c1", test_observations=100, test_inference_seconds=5.0)
            _insert_cost(connection, "s2-100", "c2", "run-c2", test_observations=100, test_inference_seconds=5.0)
        result = evaluate_holdout("s1", 20, database=database, top_n=1)
        self.assertEqual(result.ground_truth_top_n, ("c1::",))
        self.assertEqual(result.predicted_top_n, ("c2::",))
        self.assertEqual(result.recall_at_n, 0.0)

    def test_candidate_reduction_ratio_reflects_top_n_over_total(self):
        result = evaluate_holdout("s1", 20, database=self.database, top_n=1)
        self.assertEqual(result.total_candidates, 2)
        self.assertAlmostEqual(result.candidate_reduction_ratio, 1 - 1 / 2)

    def test_unknown_series_or_q_percent_raises_instead_of_silently_returning_empty(self):
        with self.assertRaises(ValueError):
            evaluate_holdout("does-not-exist", 20, database=self.database, top_n=1)
        with self.assertRaises(ValueError):
            evaluate_holdout("s1", 999, database=self.database, top_n=1)

    def test_sweep_n_sensitivity_runs_each_n_independently(self):
        results = sweep_n_sensitivity("s1", 20, [1, 2, 5], database=self.database)
        self.assertEqual([r.top_n for r in results], [1, 2, 5])
        # N이 total_candidates(2)를 넘어도 실패하지 않고 top_n을 그대로 요청값으로 남긴다
        # (reduction_ratio가 음수가 될 수 있음 — 임의로 자르지 않는다).
        self.assertEqual(results[2].top_n, 5)
        self.assertEqual(len(results[2].predicted_top_n), 2)  # 실제로는 후보가 2개뿐

    def test_sweep_k_sensitivity_applies_different_knn_k_via_config(self):
        # k=1이면 s2/s3 중 하나만 써서 예측하지만(distance가 동률이면 정렬 안정성에 따라
        # 하나만 선택), k=2/5면 s2/s3 둘 다 쓴다 — 값 자체보다 "config가 실제로 전달됐는지"를
        # similarity_matches_found로 확인한다.
        results = sweep_k_sensitivity("s1", 20, [1, 2, 5], database=self.database, top_n=1)
        self.assertEqual([r.similarity_matches_found for r in results], [1, 2, 4])

    def test_compare_similarity_metrics_returns_both_and_does_not_change_default(self):
        results = compare_similarity_metrics("s1", 20, database=self.database, top_n=1)
        self.assertEqual(set(results), {SIMILARITY_METRIC_EUCLIDEAN, SIMILARITY_METRIC_COSINE})
        self.assertEqual(results[SIMILARITY_METRIC_EUCLIDEAN].predicted_top_n, ("c1::",))
        # 이 합성 데이터는 모든 series/q에서 feature 값이 완전히 같다(zero-variance) ->
        # 표준화하면 전부 0벡터가 되고, cosine은 0벡터끼리 방향을 정의할 수 없어 억지로
        # 거리를 만들지 않는다(`cosine_distance`의 "no arbitrary distance" 원칙) -> 매치 0건.
        # 이는 버그가 아니라 이 metric의 설계된 동작이며, 실제 DB(분산이 있는 feature)에서는
        # 이런 퇴화 현상이 생기지 않는다.
        self.assertEqual(results[SIMILARITY_METRIC_COSINE].similarity_matches_found, 0)
        self.assertEqual(results[SIMILARITY_METRIC_COSINE].predicted_top_n, ())
        # base_config의 기본값(euclidean)은 이 비교 함수 호출로 바뀌지 않는다.
        self.assertEqual(MLConfig().similarity_metric, SIMILARITY_METRIC_EUCLIDEAN)


class TestCostTieBreak(unittest.TestCase):
    """predicted_performance가 동률일 때 cost.py 기반 총비용으로 tie-break하는지 확인한다
    (candidate_selection.py와 동일한 "performance desc, cost asc" 정책, 2026-09-25 추가).
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "cost.sqlite3"
        with sqlite3.connect(self.database) as connection:
            _create_schema(connection)
            _insert_prefix(connection, "s1-20", "s1", 20, 20)
            _insert_prefix(connection, "s1-100", "s1", 100, 100)
            _insert_prefix(connection, "s2-100", "s2", 100, 100)
            _insert_model_config(connection, "c1", "MWVAR", "training_free")
            _insert_model_config(connection, "c2", "MWVAR", "training_free")
            # c1/c2 모두 s2에서 vus_pr=0.8로 동률 -> 예측 성능이 같아진다.
            _insert_result(connection, "s2-100", "c1", 0.8)
            _insert_result(connection, "s2-100", "c2", 0.8)
            _insert_result(connection, "s1-100", "c1", 0.9)  # ground truth (사용 안 함, 참고용)
            _insert_result(connection, "s1-100", "c2", 0.9)
            # c1은 저비용, c2는 고비용.
            _insert_cost(connection, "s2-100", "c1", "run-c1", test_observations=100, test_inference_seconds=5.0)
            _insert_cost(connection, "s2-100", "c2", "run-c2", test_observations=100, test_inference_seconds=50.0)

    def test_lower_cost_candidate_ranked_first_when_performance_ties(self):
        result = evaluate_holdout("s1", 20, database=self.database, top_n=2)
        self.assertEqual(result.predicted_top_n, ("c1::", "c2::"))  # c1(저비용)이 c2보다 먼저
        self.assertEqual(result.candidates_with_cost_estimate, 2)

    def test_mean_cost_of_top_n_is_reported(self):
        result = evaluate_holdout("s1", 20, database=self.database, top_n=1)
        self.assertAlmostEqual(result.mean_estimated_total_cost_seconds_of_top_n, 5.0)  # top_n=1 -> c1만

    def test_cost_regression_excludes_held_out_series_own_execution_data(self):
        # c1에 held-out series(s1) 자신의 실행 기록을 하나 더 얹는다 — 성능처럼 leave-series-out이
        # 지켜진다면 이 값은 cost 추정에 전혀 영향을 주면 안 된다. 값을 극단적으로(500초) 잡아서,
        # 새어 들어가면 flat 평균이 (5+500)/2=252.5가 되어 c2(50초)보다 훨씬 비싸져 순위가
        # 뒤집히는 것으로 누수 여부를 명확히 드러낸다.
        with sqlite3.connect(self.database) as connection:
            _insert_cost(
                connection, "s1-100", "c1", "run-c1-heldout",
                test_observations=100, test_inference_seconds=500.0,
            )
        result = evaluate_holdout("s1", 20, database=self.database, top_n=2)
        # 순위가 그대로 c1(저비용) 먼저여야 한다 — 누수가 있었다면 c2가 먼저 온다.
        self.assertEqual(result.predicted_top_n, ("c1::", "c2::"))
        # c1의 비용 추정치 자체도 held-out의 500초에 영향받지 않고 5.0 그대로여야 한다.
        top_n_1 = evaluate_holdout("s1", 20, database=self.database, top_n=1)
        self.assertAlmostEqual(top_n_1.mean_estimated_total_cost_seconds_of_top_n, 5.0)


def _insert_channel_features(connection, prefix_feature_id, channel_count):
    connection.executemany(
        "INSERT INTO channel_features VALUES (?,?)",
        [(prefix_feature_id, i) for i in range(channel_count)],
    )


class TestFeasibilityIntegration(unittest.TestCase):
    """feasibility.py 연결: infeasible 후보는 top-N 경쟁에서 제외되고, held-out series의
    실제 test_observations를 지어내지 않고 그대로 쓰는지 확인한다 (2026-09-25 추가).

    c1 = PCA_LEGACY(window=5): held-out series의 실제 q=100 규모(observed_row=10)에서는
    구조적으로 infeasible (`test_infeasible_now_becomes_feasible_at_a_later_stage_with_more_rows`
    와 같은 값으로 재확인함).
    c2 = MWVAR(window=3): 같은 규모에서 feasible.
    성능은 c1이 더 높게(0.9) 설계해서, feasibility가 없었다면 c1이 1등이었을 것임을 보장한다.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "feasibility.sqlite3"
        with sqlite3.connect(self.database) as connection:
            _create_schema(connection)
            _insert_prefix(connection, "s1-20", "s1", 20, 5)
            _insert_prefix(connection, "s1-100", "s1", 100, 10)  # 작은 규모 -> c1(PCA_LEGACY) infeasible
            _insert_prefix(connection, "s2-100", "s2", 100, 100)
            _insert_channel_features(connection, "s1-20", 3)
            _insert_model_config(connection, "c1", "PCA_LEGACY", "training_free")
            _insert_model_config(connection, "c2", "MWVAR", "training_free")
            connection.execute(
                "UPDATE model_configs SET settings_json=? WHERE config_id='c1'",
                ('{"target_use": "training_free", "hyperparameters": {"window": 5}}',),
            )
            connection.execute(
                "UPDATE model_configs SET settings_json=? WHERE config_id='c2'",
                ('{"target_use": "training_free", "hyperparameters": {"window": 3}}',),
            )
            _insert_result(connection, "s2-100", "c1", 0.9)  # c1이 성능은 더 높다
            _insert_result(connection, "s2-100", "c2", 0.5)
            _insert_cost(connection, "s2-100", "c1", "run-c1", test_observations=100, test_inference_seconds=5.0)
            _insert_cost(connection, "s2-100", "c2", "run-c2", test_observations=100, test_inference_seconds=5.0)
            # held-out(s1) 자신의 q=100 실제 결과도 있어야 한다 (ground truth이자, 아래
            # test_observations를 join으로 끌어오려면 results 행이 있어야 cost가 노출됨).
            _insert_result(connection, "s1-100", "c1", 0.7)
            # held-out series 자신의 실제 q=100 test_observations (feasibility의
            # stage_inference_rows로 그대로 재사용됨 — 모듈 docstring 참고).
            _insert_cost(connection, "s1-100", "c1", "run-s1-c1", test_observations=20, test_inference_seconds=1.0)

    def test_infeasible_candidate_excluded_even_with_higher_performance(self):
        result = evaluate_holdout("s1", 20, database=self.database, top_n=2)
        self.assertTrue(result.feasibility_evaluated)
        self.assertEqual(result.stage_inference_rows_used, 20)
        self.assertEqual(result.candidates_feasible, 1)  # c2만 feasible
        self.assertEqual(result.predicted_top_n, ("c2::",))  # c1이 성능은 높지만 infeasible이라 빠진다
        self.assertEqual(result.candidates_usable, 1)

    def test_feasibility_not_evaluated_when_held_out_test_observations_missing(self):
        # cost_executions에서 s1(held-out)의 test_observations를 지우면(=관측 없음),
        # feasibility를 계산하지 않고("모른다"), infeasible로 임의 판정해 걸러내지도 않는다.
        database = Path(self.temporary.name) / "no_test_observations.sqlite3"
        with sqlite3.connect(database) as connection:
            _create_schema(connection)
            _insert_prefix(connection, "s1-20", "s1", 20, 5)
            _insert_prefix(connection, "s1-100", "s1", 100, 10)
            _insert_prefix(connection, "s2-100", "s2", 100, 100)
            _insert_channel_features(connection, "s1-20", 3)
            _insert_model_config(connection, "c1", "PCA_LEGACY", "training_free")
            connection.execute(
                "UPDATE model_configs SET settings_json=? WHERE config_id='c1'",
                ('{"target_use": "training_free", "hyperparameters": {"window": 5}}',),
            )
            _insert_result(connection, "s2-100", "c1", 0.9)
            _insert_cost(connection, "s2-100", "c1", "run-c1", test_observations=100, test_inference_seconds=5.0)
            # s1(held-out) 자신의 test_observations는 넣지 않는다.
        result = evaluate_holdout("s1", 20, database=database, top_n=1)
        self.assertFalse(result.feasibility_evaluated)
        self.assertIsNone(result.stage_inference_rows_used)
        self.assertEqual(result.predicted_top_n, ("c1::",))  # feasibility 미평가 -> 걸러내지 않는다


class TestCandidateCountIsNotHardcoded(unittest.TestCase):
    """N/후보 수를 하드코딩하지 않았는지, 45개보다 훨씬 큰 규모(합성 데이터)에서도
    구조적으로 동작하는지 확인한다. (전체 3,000-후보 스트레스 테스트는
    `stress_test_scale.py`를 직접 실행 — 이 테스트는 그보다 작은 규모로 빠르게
    같은 것을 회귀 검증한다.)
    """

    CANDIDATE_COUNT = 300

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "scale.sqlite3"
        _build_synthetic_database(self.database, series_count=6, candidate_count=self.CANDIDATE_COUNT, seed=42)

    def test_total_candidates_reflects_actual_db_size_not_a_constant(self):
        result = evaluate_holdout("s000", 20, database=self.database, top_n=50)
        self.assertEqual(result.total_candidates, self.CANDIDATE_COUNT)  # 45가 아니라 실제 규모 그대로

    def test_n_larger_than_45_does_not_raise_or_truncate_silently(self):
        # 팀 피드백 문서가 예시로 든 N=50/100/200을 하드코딩된 45-후보 가정 없이 그대로 돌릴 수 있어야 한다.
        for n in (50, 100, 200):
            result = evaluate_holdout("s000", 20, database=self.database, top_n=n)
            self.assertEqual(result.top_n, n)
            self.assertLessEqual(len(result.predicted_top_n), n)

    def test_sweep_n_sensitivity_reuses_loaded_data_across_all_n_values(self):
        results = sweep_n_sensitivity("s000", 20, [50, 100, 200], database=self.database)
        self.assertEqual([r.total_candidates for r in results], [self.CANDIDATE_COUNT] * 3)


if __name__ == "__main__":
    unittest.main()
