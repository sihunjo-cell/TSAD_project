"""streamlit_website/ml/ 전체에 대한 테스트.

사용자 요구사항(Phase 3)에 따라 최소 11개 시나리오를 커버한다:
 1. similarity에서 row_count가 feature로 사용되지 않음
 2. NULL feature가 0으로 대체되지 않음
 3. historical q trajectory가 올바르게 연결됨
 4. training-free physical run 재사용이 올바르게 처리됨
 5. future stage feasibility가 stage마다 다시 판정됨
 6. performance held-last extrapolation
 7. checkpoint maintenance가 top-k와 독립적으로 유지됨
 8. cost estimation에 model별 데이터가 사용됨 (동일 증가율 아님)
 9. insufficient data 후보가 exclusion_reason과 함께 처리됨
 10. k보다 후보가 적으면 모두 유지됨
 11. pipeline 전체가 MLOutput schema를 만족함

대부분의 모듈(trajectory/performance/cost/similarity/candidate_selection)은
DB 없이 dict/list만으로 테스트할 수 있다 (db.py가 반환하는 row shape을 직접
흉내낸다). DB 접근 자체(join, NULL 보존, 경로 해석)와 전체 pipeline은 임시
sqlite DB로 검증하고, 마지막으로 실제 dev18 DB에 대해서도 smoke 테스트한다.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from streamlit_website.DP.planner import optimize_plan
from streamlit_website.ml import db
from streamlit_website.ml.candidate_selection import select_stage_candidates
from streamlit_website.ml.config import (
    COST_SOURCE_INSUFFICIENT,
    DEFAULT_CONFIG,
    PREDICTION_SOURCE_EXTRAPOLATED,
    PREDICTION_SOURCE_SIMILARITY,
    SIMILARITY_FEATURE_COLUMNS,
    SIMILARITY_METRIC_COSINE,
    MLConfig,
)
from streamlit_website.ml.cost import build_candidate_cost_model, build_cost_curve, estimate_stage_cost
from streamlit_website.ml.dp_input import build_dp_input
from streamlit_website.ml.feasibility import assess_future_feasibility, compute_stage_training_lengths
from streamlit_website.ml.features import extract_historical_features, extract_similarity_features
from streamlit_website.ml.performance import estimate_checkpoint_maintenance_performance, estimate_stage_performance
from streamlit_website.ml.pipeline import _steady_state_inference_rows, build_future_stages, run_ml_pipeline
from streamlit_website.ml.schemas import CandidateEstimate, CheckpointMaintenanceOption, MLOutput, SimilarityMatch
from streamlit_website.ml.similarity import (
    build_standardizer,
    cosine_distance,
    euclidean_distance,
    find_similar_historical_prefixes,
)
from streamlit_website.ml.trajectory import get_series_trajectory, observed_point_up_to

REAL_DB_PATH = Path(__file__).resolve().parents[2] / "experiments/tuning/results/recommendation.sqlite3"


def _summary(**overrides) -> dict:
    base = {column: 1.0 for column in SIMILARITY_FEATURE_COLUMNS}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. row_count는 similarity feature로 쓰이지 않는다
# ---------------------------------------------------------------------------


class TestRowCountExcludedFromSimilarity(unittest.TestCase):
    def test_similarity_columns_do_not_include_row_count(self):
        self.assertNotIn("observed_row", SIMILARITY_FEATURE_COLUMNS)
        self.assertNotIn("row_count", SIMILARITY_FEATURE_COLUMNS)

    def test_extract_similarity_features_drops_row_count_even_if_present_in_summary(self):
        summary = _summary(observed_row=999999, row_count=999999)
        features, _reasons = extract_similarity_features(summary)
        self.assertNotIn("observed_row", features)
        self.assertNotIn("row_count", features)
        self.assertEqual(set(features), set(SIMILARITY_FEATURE_COLUMNS))

    def test_distance_is_unaffected_by_row_count_value(self):
        historical = [
            {"prefix_feature_id": "p1", "series": "s1", "csv_file": "s1.csv", "family": "f",
             "q_percent": 100, "observed_row": 10, **_summary()},
            {"prefix_feature_id": "p2", "series": "s2", "csv_file": "s2.csv", "family": "f",
             "q_percent": 100, "observed_row": 20, **_summary(channel_std_median=9.0)},
        ]
        current_small, _ = extract_similarity_features(_summary(observed_row=1))
        current_huge, _ = extract_similarity_features(_summary(observed_row=10 ** 9))
        matches_small, _ = find_similar_historical_prefixes(current_small, historical)
        matches_huge, _ = find_similar_historical_prefixes(current_huge, historical)
        self.assertEqual([m.distance for m in matches_small], [m.distance for m in matches_huge])
        self.assertEqual([m.prefix_feature_id for m in matches_small], [m.prefix_feature_id for m in matches_huge])


# ---------------------------------------------------------------------------
# 2. NULL feature는 0으로 대체되지 않는다
# ---------------------------------------------------------------------------


class TestNullFeaturesNeverZeroFilled(unittest.TestCase):
    def test_none_stays_none_with_reason_recorded(self):
        summary = _summary(channel_spectral_entropy_median=None,
                            channel_spectral_entropy_median_reason="insufficient_window_count")
        features, reasons = extract_similarity_features(summary)
        self.assertIsNone(features["channel_spectral_entropy_median"])
        self.assertEqual(reasons["channel_spectral_entropy_median"], "insufficient_window_count")
        # None인 다른 컬럼에 대해서만 reason이 채워진다.
        self.assertNotIn("channel_std_median", reasons)

    def test_missing_reason_falls_back_to_not_computed(self):
        summary = _summary(channel_acf_lag1_median=None)
        _features, reasons = extract_similarity_features(summary)
        self.assertEqual(reasons["channel_acf_lag1_median"], "not_computed")

    def test_historical_extraction_preserves_none(self):
        row = {**_summary(), "channel_median_shift_iqr_ratio_median": None}
        features = extract_historical_features(row)
        self.assertIsNone(features["channel_median_shift_iqr_ratio_median"])

    def test_column_entirely_null_in_history_is_excluded_not_zero_filled(self):
        historical_features = [
            {**_summary(), "channel_acf_lag1_median": None},
            {**_summary(channel_std_median=2.0), "channel_acf_lag1_median": None},
        ]
        standardizer = build_standardizer(historical_features)
        self.assertIn("channel_acf_lag1_median", standardizer.insufficient_data_columns)
        standardized = standardizer.standardize(_summary(channel_acf_lag1_median=5.0))
        # 과거 전체가 NULL이라 표준화 통계가 없으므로, 현재 값이 있어도 비교 불가 -> None.
        # 0.0으로 대체되지 않는다 (0.0은 "값이 있고 표준화 결과가 0"인 zero-variance 케이스와 다르다).
        self.assertIsNone(standardized["channel_acf_lag1_median"])

    def test_distance_only_uses_mutually_present_features_never_zero_fills(self):
        current = {**{c: 1.0 for c in SIMILARITY_FEATURE_COLUMNS}, "channel_std_median": None}
        historical = {c: 1.0 for c in SIMILARITY_FEATURE_COLUMNS}
        distance, used = euclidean_distance(current, historical)
        self.assertNotIn("channel_std_median", used)
        self.assertEqual(len(used), len(SIMILARITY_FEATURE_COLUMNS) - 1)
        self.assertEqual(distance, 0.0)  # 나머지 feature는 전부 동일하므로 거리 0

    def test_no_common_features_excludes_historical_prefix_entirely(self):
        current = {c: None for c in SIMILARITY_FEATURE_COLUMNS}
        historical_prefixes = [
            {"prefix_feature_id": "p1", "series": "s1", "csv_file": "s1.csv", "family": "f",
             "q_percent": 100, "observed_row": 10, **_summary()},
        ]
        matches, _standardizer = find_similar_historical_prefixes(current, historical_prefixes)
        self.assertEqual(matches, [])


# ---------------------------------------------------------------------------
# Cosine similarity (V1 기본은 Euclidean, cosine은 비교 실험용 대안)
# ---------------------------------------------------------------------------


class TestCosineSimilarityOption(unittest.TestCase):
    def test_identical_direction_gives_zero_distance_regardless_of_magnitude(self):
        # 방향이 같으면(스케일만 다르면) cosine distance는 0에 가깝다 — Euclidean과
        # 다른 특성(크기 차이에 둔감함)을 보여주는 핵심 테스트.
        current = {"channel_std_median": 1.0, "channel_acf_lag1_median": 2.0,
                   "absolute_correlation_median": None, "channel_interquartile_range_median": None,
                   "channel_difference_q90_iqr_ratio_median": None,
                   "channel_median_shift_iqr_ratio_median": None, "channel_spectral_entropy_median": None}
        historical = {"channel_std_median": 10.0, "channel_acf_lag1_median": 20.0,
                      "absolute_correlation_median": None, "channel_interquartile_range_median": None,
                      "channel_difference_q90_iqr_ratio_median": None,
                      "channel_median_shift_iqr_ratio_median": None, "channel_spectral_entropy_median": None}
        distance, used = cosine_distance(current, historical)
        self.assertAlmostEqual(distance, 0.0, places=9)
        self.assertEqual(len(used), 2)
        # 같은 벡터쌍이라도 Euclidean은 스케일 차이 때문에 거리가 크다.
        euclidean, _ = euclidean_distance(current, historical)
        self.assertGreater(euclidean, distance)

    def test_zero_vector_is_incomparable_not_forced_to_a_distance(self):
        zero = {c: 0.0 for c in SIMILARITY_FEATURE_COLUMNS}
        nonzero = {c: 1.0 for c in SIMILARITY_FEATURE_COLUMNS}
        distance, used = cosine_distance(zero, nonzero)
        self.assertIsNone(distance)
        self.assertEqual(used, ())

    def test_no_common_features_is_incomparable(self):
        current = {c: None for c in SIMILARITY_FEATURE_COLUMNS}
        historical = {c: 1.0 for c in SIMILARITY_FEATURE_COLUMNS}
        distance, used = cosine_distance(current, historical)
        self.assertIsNone(distance)
        self.assertEqual(used, ())

    def test_find_similar_historical_prefixes_can_switch_to_cosine_via_config(self):
        historical_prefixes = [
            {"prefix_feature_id": "p1", "series": "s1", "csv_file": "s1.csv", "family": "f",
             "q_percent": 100, "observed_row": 10, **_summary(channel_std_median=1.0)},
            {"prefix_feature_id": "p2", "series": "s2", "csv_file": "s2.csv", "family": "f",
             "q_percent": 100, "observed_row": 10, **_summary(channel_std_median=10.0)},
        ]
        current, _ = extract_similarity_features(_summary(channel_std_median=1.0))
        cosine_matches, _ = find_similar_historical_prefixes(
            current, historical_prefixes, config=MLConfig(similarity_metric=SIMILARITY_METRIC_COSINE),
        )
        euclidean_matches, _ = find_similar_historical_prefixes(current, historical_prefixes)  # 기본값
        self.assertEqual(len(cosine_matches), 2)
        self.assertEqual(len(euclidean_matches), 2)
        # 두 방법이 항상 같은 순위를 낼 필요는 없다 — 여기서는 그냥 둘 다 정상 동작하고
        # 결과 타입이 SimilarityMatch임을 확인한다 (교체 가능성 확인이 목적).
        for match in cosine_matches + euclidean_matches:
            self.assertIsInstance(match, SimilarityMatch)

    def test_unknown_similarity_metric_raises(self):
        with self.assertRaises(ValueError):
            MLConfig(similarity_metric="manhattan")


# ---------------------------------------------------------------------------
# 3. historical q trajectory가 올바르게 연결된다 / 4. training-free physical run 재사용
# ---------------------------------------------------------------------------


class TestTrajectoryLinking(unittest.TestCase):
    def _row(self, *, q, vus_pr, status="complete", run_id=None, seed=1, **cost):
        return {"series": "s1", "q_percent": q, "observed_row": q * 10, "status": status,
                "vus_pr": vus_pr, "run_id": run_id, "seed": seed, **cost}

    def test_trajectory_links_correct_series_and_averages_seeds(self):
        rows = [
            self._row(q=20, vus_pr=0.4, seed=1), self._row(q=20, vus_pr=0.6, seed=2),
            self._row(q=100, vus_pr=0.9, seed=1),
            {"series": "other", "q_percent": 20, "observed_row": 1, "status": "complete", "vus_pr": 0.1,
             "run_id": None, "seed": 1},
        ]
        points = get_series_trajectory(rows, "s1")
        self.assertEqual([p.q_percent for p in points], [20, 100])
        self.assertAlmostEqual(points[0].vus_pr, 0.5)  # seed 평균
        self.assertEqual(points[0].complete_result_count, 2)
        self.assertEqual(points[1].vus_pr, 0.9)

    def test_incomplete_status_excluded_from_average(self):
        rows = [self._row(q=20, vus_pr=0.4, status="complete"), self._row(q=20, vus_pr=None, status="failed")]
        points = get_series_trajectory(rows, "s1")
        self.assertEqual(points[0].complete_result_count, 1)
        self.assertEqual(points[0].vus_pr, 0.4)

    def test_training_free_physical_run_reuse_flagged_correctly(self):
        # training-free 모델: q=20과 q=60이 같은 물리 실행(run_id="run-A")을 재사용하고,
        # q=100에서 새 물리 실행(run_id="run-B")이 일어난 경우.
        rows = [
            self._row(q=20, vus_pr=0.5, run_id="run-A"),
            self._row(q=60, vus_pr=0.5, run_id="run-A"),
            self._row(q=100, vus_pr=0.5, run_id="run-B"),
        ]
        points = get_series_trajectory(rows, "s1")
        self.assertTrue(points[0].is_new_physical_execution)   # q=20: run-A 최초 관측
        self.assertFalse(points[1].is_new_physical_execution)  # q=60: run-A 재사용
        self.assertTrue(points[2].is_new_physical_execution)   # q=100: run-B 최초 관측
        self.assertEqual(points[1].run_ids, frozenset({"run-A"}))

    def test_observed_point_holds_last_lower_ratio_and_never_looks_above(self):
        points = get_series_trajectory([self._row(q=20, vus_pr=0.5), self._row(q=60, vus_pr=None, status="failed")], "s1")
        self.assertIsNone(observed_point_up_to(points, 10))  # 더 적은 데이터의 결과가 없으면 추정하지 않는다
        self.assertEqual(observed_point_up_to(points, 20).q_percent, 20)
        self.assertEqual(observed_point_up_to(points, 80).q_percent, 20)  # 실패한 q=60은 건너뛴다


# ---------------------------------------------------------------------------
# 5. future stage feasibility가 stage마다 다시 판정된다
# ---------------------------------------------------------------------------


class TestFutureStageFeasibility(unittest.TestCase):
    def test_infeasible_now_becomes_feasible_at_a_later_stage_with_more_rows(self):
        candidate = {"model": "PCA_LEGACY", "parameters": {"window": 5},
                     "target_use": "training_free", "recipe": {}}
        now = assess_future_feasibility(candidate, stage_training_rows=10, stage_inference_rows=20, channel_count=3)
        self.assertEqual(now["status"], "structurally_infeasible")
        later = assess_future_feasibility(candidate, stage_training_rows=100, stage_inference_rows=20, channel_count=3)
        self.assertEqual(later["status"], "feasible")

    def test_future_candidate_not_permanently_removed_for_current_data_shortage(self):
        # "미래 후보를 현재 데이터 부족만으로 영구 제거하지 않는다" 원칙:
        # 같은 candidate가 stage별로 독립적으로 재판정되므로, 한 stage의 infeasible이
        # 다른(미래) stage의 판정에 전혀 영향을 주지 않는다.
        candidate = {"model": "PCA_LEGACY", "parameters": {"window": 5},
                     "target_use": "training_free", "recipe": {}}
        results = [
            assess_future_feasibility(candidate, stage_training_rows=rows, stage_inference_rows=20, channel_count=3)
            for rows in (5, 500)
        ]
        self.assertEqual(results[0]["status"], "structurally_infeasible")
        self.assertEqual(results[1]["status"], "feasible")

    def test_compute_stage_training_lengths_full_prefix_vs_split(self):
        fit_split, validation_split = compute_stage_training_lengths("training_free", 100)
        self.assertEqual((fit_split, validation_split), (80, 20))
        fit_full, validation_full = compute_stage_training_lengths("fit_full_prefix", 100)
        self.assertEqual((fit_full, validation_full), (100, 0))

    def test_stages_start_when_data_reaches_each_db_ratio(self):
        # 하루 43,200행씩 95일 모으면 끝에 4,320,000행 = 100%. 지금 216,000행은 5%다.
        stages = build_future_stages(216_000, {"operating_days": 95, "collection_rows_per_second": 0.5,
                                               "inference_rows_per_day": 10})
        self.assertEqual([stage.ratio_percent for stage in stages], [5, 10, 20, 40, 60, 80])
        self.assertEqual([stage.elapsed_days for stage in stages], [0, 5, 15, 35, 55, 75])
        self.assertEqual([stage.future_rows for stage in stages], [216_000, 432_000, 864_000, 1_728_000, 2_592_000, 3_456_000])
        self.assertEqual([stage.inference_volume for stage in stages], [50, 100, 200, 200, 200, 200])

    def test_between_ratios_and_without_collection_stay_on_one_measured_ratio(self):
        between = build_future_stages(300_000, {"operating_days": 95, "collection_rows_per_second": 0.5})
        self.assertEqual(between[0].ratio_percent, 5)  # 끝 행 수 4,404,000의 6.8%는 DB가 잰 5%로 내린다
        self.assertEqual((between[1].ratio_percent, between[1].future_rows), (10, 440_400))
        idle = build_future_stages(1000, {"operating_days": 30, "inference_rows_per_day": 10})
        self.assertEqual([(stage.ratio_percent, stage.inference_volume) for stage in idle], [(100, 300)])


class TestInferenceBatchLengthResolution(unittest.TestCase):
    """test_length에 넣을 값 결정: inference_batch_length가 있으면 그대로 쓰고
    (기존 model_feasibility.assess_candidate 계약과 의미 일치), 없으면 검증되지
    않은 inference_rows_per_day proxy로 fallback한다."""

    def test_explicit_batch_length_takes_priority_and_is_flagged_validated(self):
        rows, used_proxy = _steady_state_inference_rows(
            {"inference_batch_length": 10, "inference_rows_per_day": 99999},
        )
        self.assertEqual(rows, 10)
        self.assertFalse(used_proxy)

    def test_falls_back_to_inference_rows_per_day_when_batch_length_absent(self):
        rows, used_proxy = _steady_state_inference_rows({"inference_rows_per_day": 50})
        self.assertEqual(rows, 50)
        self.assertTrue(used_proxy)

    def test_falls_back_when_batch_length_is_none_or_zero(self):
        for value in (None, 0):
            rows, used_proxy = _steady_state_inference_rows(
                {"inference_batch_length": value, "inference_rows_per_day": 20},
            )
            self.assertEqual(rows, 20)
            self.assertTrue(used_proxy)

    def test_both_missing_returns_zero_and_flags_proxy(self):
        rows, used_proxy = _steady_state_inference_rows({})
        self.assertEqual(rows, 0)
        self.assertTrue(used_proxy)

    def test_feasibility_outcome_actually_changes_with_batch_length(self):
        # PCA_LEGACY window=5: test_length>=5이어야 feasible. inference_rows_per_day
        # 근사치(3)로는 infeasible이지만, 실제 배치 길이(inference_batch_length=10)를
        # 쓰면 feasible로 뒤집힌다 — resolution이 실제로 결과에 영향을 준다는 증거.
        candidate = {"model": "PCA_LEGACY", "parameters": {"window": 5},
                     "target_use": "training_free", "recipe": {}}
        proxy_rows, _ = _steady_state_inference_rows({"inference_rows_per_day": 3})
        batch_rows, _ = _steady_state_inference_rows(
            {"inference_rows_per_day": 3, "inference_batch_length": 10},
        )
        with_proxy = assess_future_feasibility(
            candidate, stage_training_rows=1000, stage_inference_rows=proxy_rows, channel_count=3,
        )
        with_batch_length = assess_future_feasibility(
            candidate, stage_training_rows=1000, stage_inference_rows=batch_rows, channel_count=3,
        )
        self.assertEqual(with_proxy["status"], "structurally_infeasible")
        self.assertEqual(with_batch_length["status"], "feasible")


# ---------------------------------------------------------------------------
# 6. performance held-last extrapolation
# ---------------------------------------------------------------------------


class TestPerformanceExtrapolation(unittest.TestCase):
    def _results(self, series="s1", values=((20, 0.5), (100, 0.75))):
        return [{"series": series, "q_percent": q, "observed_row": q, "status": "complete", "vus_pr": value,
                 "run_id": f"{series}-{q}"} for q, value in values]

    def _match(self, q=100, series="s1"):
        return SimilarityMatch(prefix_feature_id=f"{series}-{q}", series=series, csv_file=f"{series}.csv",
                                family="f", q_percent=q, observed_row=q, distance=0.0, weight=1.0,
                                features_used=SIMILARITY_FEATURE_COLUMNS)

    def test_stage_reads_the_same_ratio_whatever_prefix_matched(self):
        # q=100 prefix와 비슷하다고 나와도 20% 구간은 그 series의 20% 결과를 읽는다.
        estimate = estimate_stage_performance(self._results(), [self._match(q=100)], ratio_percent=20)
        self.assertEqual((estimate.predicted_performance, estimate.prediction_source), (0.5, PREDICTION_SOURCE_SIMILARITY))

    def test_missing_ratio_holds_last_lower_value_constant(self):
        estimate = estimate_stage_performance(self._results(), [self._match()], ratio_percent=60)
        self.assertEqual(estimate.predicted_performance, 0.5)
        self.assertEqual(estimate.prediction_source, PREDICTION_SOURCE_EXTRAPOLATED)
        self.assertEqual(estimate.performance_lower_bound, estimate.predicted_performance)

    def test_missing_candidate_data_returns_none_not_zero(self):
        for results, ratio in (([], 20), (self._results(), 10)):  # 결과 없음, 더 적은 데이터의 결과 없음
            estimate = estimate_stage_performance(results, [self._match()], ratio_percent=ratio)
            self.assertIsNone(estimate.predicted_performance)
            self.assertIsNone(estimate.prediction_source)
            self.assertEqual(estimate.used_match_count, 0)

    def test_partial_extrapolation_still_reports_similarity_source(self):
        results = self._results() + self._results("s2", ((40, 0.55),))
        estimate = estimate_stage_performance(results, [self._match(), self._match(q=20, series="s2")],
                                               ratio_percent=40)
        self.assertEqual(estimate.prediction_source, PREDICTION_SOURCE_SIMILARITY)
        self.assertEqual(estimate.used_match_count, 2)
        self.assertAlmostEqual(estimate.predicted_performance, (0.5 + 0.55) / 2)


# ---------------------------------------------------------------------------
# 7. checkpoint maintenance는 top-k와 독립적으로 유지된다
# ---------------------------------------------------------------------------


class TestCheckpointMaintenanceIndependence(unittest.TestCase):
    def test_checkpoint_maintenance_performance_uses_current_stage_as_last_trained_approximation(self):
        results = [{"series": "s1", "q_percent": 100, "observed_row": 100, "status": "complete",
                    "vus_pr": 0.7, "run_id": "r1"}]
        match = SimilarityMatch(prefix_feature_id="p1", series="s1", csv_file="s1.csv", family="f",
                                 q_percent=100, observed_row=100, distance=0.0, weight=1.0,
                                 features_used=SIMILARITY_FEATURE_COLUMNS)
        estimate = estimate_checkpoint_maintenance_performance(results, [match], ratio_percent=100)
        self.assertEqual(estimate.predicted_performance, 0.7)
        self.assertEqual(estimate.prediction_source, "checkpoint_maintained")

    def test_checkpoint_trained_on_fewer_rows_uses_that_scale(self):
        results = [{"series": "s1", "q_percent": q, "observed_row": q, "status": "complete",
                    "vus_pr": value, "run_id": f"r{q}"} for q, value in ((20, 0.4), (100, 0.7))]
        match = SimilarityMatch(prefix_feature_id="p1", series="s1", csv_file="s1.csv", family="f",
                                 q_percent=100, observed_row=100, distance=0.0, weight=1.0,
                                 features_used=SIMILARITY_FEATURE_COLUMNS)
        estimate = estimate_checkpoint_maintenance_performance(results, [match], ratio_percent=20)
        self.assertEqual(estimate.predicted_performance, 0.4)

    def test_select_stage_candidates_never_sees_checkpoint_options(self):
        # candidate_selection.select_stage_candidates는 CandidateEstimate만 다룬다 —
        # CheckpointMaintenanceOption은 애초에 다른 타입이라 섞일 수 없다.
        candidate = CandidateEstimate(
            stage=0, candidate_id="c1::", model="MWVAR", configuration="c1", head="", feasible=True,
            predicted_performance=0.5, performance_lower_bound=0.5,
            estimated_training_cost_seconds=None, estimated_inference_cost_seconds=1.0,
            estimated_total_cost_seconds=1.0, data_rows=100, inference_volume=10.0,
            prediction_source=PREDICTION_SOURCE_SIMILARITY, cost_estimation_source="x",
        )
        usable, excluded = select_stage_candidates([candidate])
        self.assertEqual(usable, [candidate])
        self.assertEqual(excluded, [])
        for group in (usable, excluded):
            for item in group:
                self.assertNotIsInstance(item, CheckpointMaintenanceOption)


# ---------------------------------------------------------------------------
# 8. cost estimation은 model마다 자기 자신의 데이터로 독립적으로 추정된다
# ---------------------------------------------------------------------------


class TestCostEstimationIsPerModel(unittest.TestCase):
    def test_different_candidates_get_independent_regression_slopes(self):
        slow_growth = [
            {"run_id": "a1", "available_training_rows": 100, "training_seconds": 10.0},
            {"run_id": "a2", "available_training_rows": 200, "training_seconds": 20.0},
        ]
        fast_growth = [
            {"run_id": "b1", "available_training_rows": 100, "training_seconds": 50.0},
            {"run_id": "b2", "available_training_rows": 300, "training_seconds": 130.0},
        ]
        curve_a = build_cost_curve(slow_growth, rows_field="available_training_rows", seconds_field="training_seconds",
                                   per_cell=False)
        curve_b = build_cost_curve(fast_growth, rows_field="available_training_rows", seconds_field="training_seconds",
                                   per_cell=False)
        self.assertNotAlmostEqual(curve_a.slope, curve_b.slope)
        self.assertAlmostEqual(curve_a.slope, 0.1)
        self.assertAlmostEqual(curve_b.slope, 0.4)
        # 같은 미래 행 수에서 서로 다른 비용을 예측한다 — "모델마다 동일 증가율"이 아님을 증명.
        self.assertNotAlmostEqual(curve_a.predict(1000), curve_b.predict(1000))

    def test_run_id_deduplication_prevents_double_counting_physical_reuse(self):
        rows = [
            {"run_id": "shared", "available_training_rows": 100, "training_seconds": 10.0},
            {"run_id": "shared", "available_training_rows": 100, "training_seconds": 10.0},  # 같은 물리 실행 재사용
            {"run_id": "new", "available_training_rows": 200, "training_seconds": 20.0},
        ]
        curve = build_cost_curve(rows, rows_field="available_training_rows", seconds_field="training_seconds",
                                 per_cell=False)
        self.assertEqual(curve.observation_count, 2)  # 3개 행이지만 distinct run_id는 2개

    def test_cell_rate_stays_positive_when_rows_and_channels_move_opposite(self):
        # Dev18처럼 행 수가 늘수록 채널 수가 줄면 행 수 선형회귀는 음수 기울기가 되어 0초로 잘렸다.
        rows = [
            {"run_id": "wide", "test_observations": 1000, "input_column": 200, "test_inference_seconds": 200.0},
            {"run_id": "long", "test_observations": 40000, "input_column": 2, "test_inference_seconds": 80.0},
        ]
        row_only = build_cost_curve(rows, rows_field="test_observations", seconds_field="test_inference_seconds",
                                    per_cell=False)
        self.assertEqual(row_only.predict(1_000_000), 0.0)
        per_cell = build_cost_curve(rows, rows_field="test_observations", seconds_field="test_inference_seconds")
        self.assertAlmostEqual(per_cell.predict(1_000_000, channel_count=10), 0.001 * 1_000_000 * 10)

    def test_training_rows_are_not_capped_at_the_largest_observed_run(self):
        rows = [{"run_id": "r1", "available_training_rows": 100, "input_column": 1, "training_seconds": 10.0,
                 "test_observations": 100, "test_inference_seconds": 1.0}]
        cost_model = build_candidate_cost_model(rows, model="GDN")
        training, _inference, _total, _source, _reason = estimate_stage_cost(
            cost_model, training_rows=1_000_000, inference_volume=100.0, needs_training=True,
        )
        self.assertAlmostEqual(training, 0.1 * 1_000_000)


# ---------------------------------------------------------------------------
# 9. insufficient data 후보는 exclusion_reason과 함께 처리된다 (0으로 대체하지 않는다)
# ---------------------------------------------------------------------------


class TestInsufficientDataHandling(unittest.TestCase):
    def test_zero_observations_marks_not_estimable_with_reason(self):
        curve = build_cost_curve([], rows_field="available_training_rows", seconds_field="training_seconds")
        self.assertFalse(curve.estimable)
        self.assertEqual(curve.source, COST_SOURCE_INSUFFICIENT)
        self.assertIsNone(curve.predict(100))

    def test_estimate_stage_cost_surfaces_exclusion_reason_for_training(self):
        cost_model = build_candidate_cost_model([])
        training, inference, total, source, reason = estimate_stage_cost(
            cost_model, training_rows=100, inference_volume=10.0, needs_training=True,
        )
        self.assertIsNone(training)
        self.assertIsNone(inference)
        self.assertIsNone(total)
        self.assertIsNone(source)
        self.assertEqual(reason, "training_cost_insufficient_historical_execution_data")

    def test_needs_training_false_makes_training_cost_not_applicable_not_zero(self):
        results = [{"run_id": "r1", "test_observations": 100, "input_column": 1, "test_inference_seconds": 5.0}]
        cost_model = build_candidate_cost_model(results)
        training, inference, total, _source, reason = estimate_stage_cost(
            cost_model, training_rows=None, inference_volume=100.0, needs_training=False,
        )
        self.assertIsNone(training)  # "해당 없음" — 0으로 채우지 않는다
        self.assertIsNone(reason)
        self.assertEqual(inference, 5.0)
        self.assertEqual(total, 5.0)


# ---------------------------------------------------------------------------
# 10. k보다 후보가 적으면 전부 유지된다
# ---------------------------------------------------------------------------


class TestStageCandidateSplit(unittest.TestCase):
    def _candidate(self, candidate_id, performance, cost, feasible=True, exclusion_reason=None):
        return CandidateEstimate(
            stage=0, candidate_id=candidate_id, model="MWVAR", configuration=candidate_id, head="",
            feasible=feasible, predicted_performance=performance, performance_lower_bound=performance,
            estimated_training_cost_seconds=None, estimated_inference_cost_seconds=cost,
            estimated_total_cost_seconds=cost, data_rows=100, inference_volume=10.0,
            prediction_source=PREDICTION_SOURCE_SIMILARITY if performance is not None else None,
            cost_estimation_source="x" if cost is not None else None, exclusion_reason=exclusion_reason,
        )

    def test_every_usable_candidate_reaches_dp_without_a_rank_cut(self):
        candidates = [self._candidate(str(i), 1.0 - i * 0.01, 1.0) for i in range(15)]
        usable, excluded = select_stage_candidates(candidates)
        self.assertEqual(len(usable), 15)
        self.assertEqual(excluded, [])

    def test_only_infeasible_or_unestimable_candidates_are_excluded(self):
        candidates = [
            self._candidate("good", 0.9, 1.0),
            self._candidate("infeasible", None, None, feasible=False, exclusion_reason="structurally_infeasible"),
            self._candidate("no_perf", None, None, feasible=True, exclusion_reason="no_historical_performance_estimate_for_candidate"),
        ]
        usable, excluded = select_stage_candidates(candidates)
        self.assertEqual([c.candidate_id for c in usable], ["good"])
        self.assertEqual({c.candidate_id for c in excluded}, {"infeasible", "no_perf"})

    def test_display_order_is_performance_then_cost(self):
        candidates = [self._candidate("cheap_worse", 0.5, 1.0), self._candidate("expensive_better", 0.9, 100.0)]
        usable, _excluded = select_stage_candidates(candidates)
        self.assertEqual([c.candidate_id for c in usable], ["expensive_better", "cheap_worse"])


# ---------------------------------------------------------------------------
# DB 접근 자체 (join, 경로 해석, NULL 보존) — 임시 sqlite DB
# ---------------------------------------------------------------------------


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE prefix_features (
            prefix_feature_id TEXT, csv_id TEXT, csv_file TEXT, family TEXT, series TEXT,
            q_percent INTEGER, training_boundary INTEGER, observed_row INTEGER,
            channel_std_median REAL, channel_acf_lag1_median REAL, absolute_correlation_median REAL,
            channel_interquartile_range_median REAL, channel_difference_q90_iqr_ratio_median REAL,
            channel_median_shift_iqr_ratio_median REAL, channel_spectral_entropy_median REAL,
            input_column INTEGER
        );
        CREATE TABLE results (
            prefix_feature_id TEXT, config_id TEXT, seed INTEGER, score_variant TEXT,
            primary_score INTEGER, status TEXT, vus_pr REAL
        );
        CREATE TABLE cost_result_links (
            prefix_feature_id TEXT, config_id TEXT, seed INTEGER, score_variant TEXT, run_id TEXT
        );
        CREATE TABLE cost_executions (
            run_id TEXT, physical_prefix_feature_id TEXT, status TEXT,
            training_seconds REAL, test_inference_seconds REAL,
            split_preprocess_seconds REAL, model_setup_seconds REAL,
            calibration_inference_seconds REAL, actual_runtime_seconds REAL,
            available_training_rows INTEGER, test_observations INTEGER
        );
        CREATE TABLE model_configs (
            config_id TEXT, model TEXT, settings_json TEXT
        );
        CREATE TABLE channel_features (
            prefix_feature_id TEXT, channel_index INTEGER
        );
    """)


class TestDatabaseLoaders(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "test.sqlite3"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            _create_schema(connection)
            connection.execute(
                "INSERT INTO prefix_features VALUES ('p1','csv1','s1.csv','famA','s1',100,1000,1000,"
                "1.0,0.5,0.3,2.0,0.4,NULL,0.6,3)"  # channel_median_shift_iqr_ratio_median = NULL 의도적
            )
            connection.execute("INSERT INTO results VALUES ('p1','c1',1,'',1,'complete',0.8)")
            connection.execute("INSERT INTO cost_result_links VALUES ('p1','c1',1,'','run-1')")
            connection.execute(
                "INSERT INTO cost_executions VALUES "
                "('run-1','p1','complete',50.0,5.0,NULL,NULL,NULL,NULL,800,100)"
            )

    def test_load_historical_prefixes_preserves_null(self):
        rows = db.load_historical_prefixes(self.database)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["channel_median_shift_iqr_ratio_median"])
        self.assertEqual(rows[0]["channel_std_median"], 1.0)

    def test_load_results_for_candidate_joins_cost_fields(self):
        rows = db.load_results_for_candidate("c1", "", database=self.database)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["run_id"], "run-1")
        self.assertEqual(row["training_seconds"], 50.0)
        self.assertEqual(row["available_training_rows"], 800)

    def test_load_results_for_unknown_candidate_returns_empty_not_error(self):
        self.assertEqual(db.load_results_for_candidate("does-not-exist", "", database=self.database), [])

    def test_resolve_database_path_priority_arg_over_env_over_default(self):
        with_arg = db.resolve_database_path("/explicit/path.sqlite3")
        self.assertEqual(with_arg, Path("/explicit/path.sqlite3"))
        old = os.environ.get(db.DATABASE_PATH_ENV_VAR)
        try:
            os.environ[db.DATABASE_PATH_ENV_VAR] = "/from/env.sqlite3"
            self.assertEqual(db.resolve_database_path(None), Path("/from/env.sqlite3"))
        finally:
            if old is None:
                os.environ.pop(db.DATABASE_PATH_ENV_VAR, None)
            else:
                os.environ[db.DATABASE_PATH_ENV_VAR] = old

    def test_missing_database_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            db.load_historical_prefixes(Path(self.temporary.name) / "missing.sqlite3")


# ---------------------------------------------------------------------------
# 11. pipeline 전체가 MLOutput schema를 만족한다 (+ 여러 시나리오의 통합 확인)
# ---------------------------------------------------------------------------


class TestFullPipelineSyntheticDatabase(unittest.TestCase):
    """임시 DB로 `run_ml_pipeline` 전체를 실행해 MLOutput 스키마를 확인한다.

    시나리오 조합:
    - c1 (MWVAR, training_free): 과거 결과 1건(q=100) 존재 -> similarity/extrapolation을 stage별로 검증
    - c2 (PCA_LEGACY, fit_full_prefix): 과거 결과 1건 존재, needs_training=True 경로 검증
    - c3 (MWVAR, training_free): 과거 결과 없음 -> exclusion_reason으로 처리되는지 검증 (#9)
    - current_model이 c1과 매칭 -> checkpoint_maintenance_options이 신규 후보와 별도로 생성 (#7)
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "pipeline.sqlite3"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            _create_schema(connection)
            connection.execute(
                "INSERT INTO prefix_features VALUES ('p1','csv1','s1.csv','famA','s1',100,1000,1000,"
                "1.0,0.5,0.3,2.0,0.4,0.2,0.6,3)"
            )
            connection.execute(
                "INSERT INTO prefix_features VALUES ('p0','csv1','s1.csv','famA','s1',5,1000,50,"
                "1.0,0.5,0.3,2.0,0.4,0.2,0.6,3)"
            )
            for prefix in ("p0", "p1"):  # 5%·100%에서 잰 결과, 같은 물리 실행을 재사용한다
                connection.execute(f"INSERT INTO results VALUES ('{prefix}','c1',1,'',1,'complete',0.8)")
                connection.execute(f"INSERT INTO results VALUES ('{prefix}','c2',1,'',1,'complete',0.6)")
                connection.execute(f"INSERT INTO cost_result_links VALUES ('{prefix}','c1',1,'','run-c1')")
                connection.execute(f"INSERT INTO cost_result_links VALUES ('{prefix}','c2',1,'','run-c2')")
            connection.execute(
                "INSERT INTO cost_executions VALUES "
                "('run-c1','p1','complete',NULL,5.0,NULL,NULL,NULL,NULL,NULL,100)"
            )
            connection.execute(
                "INSERT INTO cost_executions VALUES "
                "('run-c2','p1','complete',50.0,8.0,NULL,NULL,NULL,NULL,800,100)"
            )

        self.ml_input = {
            "input": {"row_count": 10, "channel_count": 3, "sensor_columns": ["a", "b", "c"]},
            "normal_prefix": pd.DataFrame({"a": [1.0], "b": [1.0], "c": [1.0]}),
            "current_model": {"candidate_id": "c1::"},
            "operating_conditions": {"operating_days": 100, "collection_rows_per_second": 0.0001,
                                      "inference_rows_per_day": 50},
            "candidates": [
                {"config_id": "c1", "head": "", "model": "MWVAR", "target_use": "training_free",
                 "parameters": {"window": 3}, "recipe": {}},
                {"config_id": "c2", "head": "", "model": "PCA_LEGACY", "target_use": "fit_full_prefix",
                 "parameters": {"window": 5}, "recipe": {}},
                {"config_id": "c3", "head": "", "model": "MWVAR", "target_use": "training_free",
                 "parameters": {"window": 3}, "recipe": {}},
            ],
            "current_features": {"summary": {
                "channel_std_median": 1.0, "channel_acf_lag1_median": 0.5, "absolute_correlation_median": 0.3,
                "channel_interquartile_range_median": 2.0, "channel_difference_q90_iqr_ratio_median": 0.4,
                "channel_median_shift_iqr_ratio_median": 0.2,
                "channel_spectral_entropy_median": None,
                "channel_spectral_entropy_median_reason": "insufficient_window_count",
            }},
        }

    def test_output_is_well_typed_ml_output(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        self.assertIsInstance(output, MLOutput)
        self.assertIsInstance(output.similarity_matches, list)
        self.assertTrue(all(isinstance(m, SimilarityMatch) for m in output.similarity_matches))
        self.assertTrue(all(isinstance(stage_list, list) for stage_list in output.stage_candidates.values()))
        for stage_list in output.stage_candidates.values():
            self.assertTrue(all(isinstance(c, CandidateEstimate) for c in stage_list))
        for stage_list in output.stage_candidates_excluded.values():
            self.assertTrue(all(isinstance(c, CandidateEstimate) for c in stage_list))
        self.assertTrue(all(isinstance(o, CheckpointMaintenanceOption) for o in output.checkpoint_maintenance_options))

    def test_null_current_feature_preserved_and_reason_recorded(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        self.assertIsNone(output.current_features["channel_spectral_entropy_median"])
        self.assertEqual(output.current_feature_reasons["channel_spectral_entropy_median"],
                          "insufficient_window_count")

    def test_candidate_without_historical_results_gets_exclusion_reason_not_dropped_silently(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        stage0_excluded_ids = {c.candidate_id for c in output.stage_candidates_excluded[0]}
        self.assertIn("c3::", stage0_excluded_ids)
        c3 = next(c for c in output.stage_candidates_excluded[0] if c.candidate_id == "c3::")
        self.assertEqual(c3.exclusion_reason, "no_historical_performance_estimate_for_candidate")

    def test_every_usable_candidate_reaches_dp(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        # c1, c2 둘 다 실행 가능하고 성능·비용을 추정했으므로 순위와 무관하게 전부 DP로 간다.
        self.assertEqual({c.candidate_id for c in output.stage_candidates[0]}, {"c1::", "c2::"})

    def test_stage0_uses_similarity_source_and_later_stage_extrapolates_held_last(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        c1_stage0 = next(c for c in output.stage_candidates[0] if c.candidate_id == "c1::")
        self.assertEqual(c1_stage0.prediction_source, PREDICTION_SOURCE_SIMILARITY)
        self.assertAlmostEqual(c1_stage0.predicted_performance, 0.8)

        later_stage = max(output.stage_candidates)
        c1_later = next(c for c in output.stage_candidates[later_stage] if c.candidate_id == "c1::")
        self.assertEqual(c1_later.prediction_source, PREDICTION_SOURCE_EXTRAPOLATED)
        self.assertAlmostEqual(c1_later.predicted_performance, 0.8)  # 마지막 관측 유지 (값이 그대로)

    def test_needs_training_candidate_gets_training_cost_and_training_free_does_not(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        c1_stage0 = next(c for c in output.stage_candidates[0] if c.candidate_id == "c1::")
        c2_stage0 = next(c for c in output.stage_candidates[0] if c.candidate_id == "c2::")
        self.assertIsNone(c1_stage0.estimated_training_cost_seconds)  # training_free -> 해당 없음
        self.assertIsNotNone(c2_stage0.estimated_training_cost_seconds)  # fit_full_prefix -> 학습비 추정

    def test_checkpoint_maintenance_kept_separate_from_new_candidates(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        self.assertTrue(len(output.checkpoint_maintenance_options) > 0)
        checkpoint_ids = {o.candidate_id for o in output.checkpoint_maintenance_options}
        self.assertEqual(checkpoint_ids, {"c1::"})
        for option in output.checkpoint_maintenance_options:
            self.assertEqual(option.prediction_source, "checkpoint_maintained")
            self.assertIsNone(getattr(option, "estimated_training_cost_seconds", None))

    def test_test_length_semantics_are_surfaced_in_metadata(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        self.assertTrue(any("test_length" in w for w in output.metadata["warnings"]))

    def test_dp_input_keeps_current_checkpoint_and_is_accepted_by_planner(self):
        # c2(window 20)는 지금 10행으로 새로 학습할 수 없지만 기존 checkpoint 유지는 막지 않는다.
        candidates = [dict(candidate) for candidate in self.ml_input["candidates"]]
        candidates[1]["parameters"] = {"window": 20}
        ml_input = {**self.ml_input, "candidates": candidates,
                    "current_model": {"candidate_id": "c2::", "checkpoint": "ckpt-c2"}}
        output = run_ml_pipeline(ml_input, db_path=self.database)
        self.assertNotIn("c2::", {c.candidate_id for c in output.stage_candidates[0]})
        self.assertTrue(all(option.feasible for option in output.checkpoint_maintenance_options))
        payload = build_dp_input(output, ml_input)
        self.assertAlmostEqual(sum(stage["weight"] for stage in payload["stages"]), 100)  # 구간 길이(일)
        plan = optimize_plan(payload)
        self.assertEqual(plan["status"], "ok")
        self.assertEqual((plan["current_action"]["candidate_id"], plan["current_action"]["action"]), ("c1::", "switch"))
        self.assertAlmostEqual(plan["timeline_performance"], 0.8)
        # c1 셀당 추론 단가 5초/(100행×3채널) × 전체 추론 5000행 × 3채널 = 250초
        self.assertAlmostEqual(plan["total_cost"], 250.0)

    def test_dp_input_prices_seconds_by_measured_device(self):
        candidates = [dict(candidate) for candidate in self.ml_input["candidates"]]
        candidates[0]["backend"] = "cpu"  # c2는 측정 장치를 모른다
        ml_input = {**self.ml_input, "candidates": candidates, "operating_conditions": {
            **self.ml_input["operating_conditions"], "cpu_hourly_cost": 3600, "gpu_hourly_cost": 7200, "budget": 100}}
        output = run_ml_pipeline(ml_input, db_path=self.database)
        seconds, priced = build_dp_input(output, self.ml_input), build_dp_input(output, ml_input)
        self.assertEqual((seconds["cost_unit"], priced["cost_unit"], priced["budget"]), ("seconds", "KRW", 100))
        self.assertTrue(any(row["training_cost"] > 0 for row in seconds["training_options"]))
        for key, cost in (("training_options", "training_cost"), ("inference_options", "inference_cost")):
            for plain, money in zip(seconds[key], priced[key]):
                rate = 1 if plain["candidate_id"] == "c1::" else 2  # 원/초: CPU 3600원/시, 모르면 비싼 7200원/시
                self.assertAlmostEqual(money[cost], plain[cost] * rate)


class TestSimilaritySummaryMetadata(unittest.TestCase):
    """metadata["similarity_summary"] 및 "부족한 매치" warning을 검증한다.

    임의의 저유사도 threshold 숫자를 도입하지 않는다는 원칙에 따라, 이 테스트는
    특정 distance 컷오프가 아니라 (1) 실제 관측된 distance 분포가 그대로
    노출되는지, (2) 요청한 k보다 매치가 적을 때만 warning이 붙는지만 확인한다.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "pipeline.sqlite3"
        with closing(sqlite3.connect(self.database)) as connection, connection:
            _create_schema(connection)
            connection.execute(
                "INSERT INTO prefix_features VALUES ('p1','csv1','s1.csv','famA','s1',100,1000,1000,"
                "1.0,0.5,0.3,2.0,0.4,0.2,0.6,3)"
            )
            connection.execute("INSERT INTO results VALUES ('p1','c1',1,'',1,'complete',0.8)")
            connection.execute("INSERT INTO cost_result_links VALUES ('p1','c1',1,'','run-c1')")
            connection.execute(
                "INSERT INTO cost_executions VALUES "
                "('run-c1','p1','complete',NULL,5.0,NULL,NULL,NULL,NULL,NULL,100)"
            )

        self.ml_input = {
            "input": {"row_count": 10, "channel_count": 3, "sensor_columns": ["a", "b", "c"]},
            "normal_prefix": pd.DataFrame({"a": [1.0], "b": [1.0], "c": [1.0]}),
            "current_model": {"candidate_id": "c1::"},
            "operating_conditions": {"operating_days": 100, "collection_rows_per_second": 0.0001,
                                      "inference_rows_per_day": 50},
            "candidates": [
                {"config_id": "c1", "head": "", "model": "MWVAR", "target_use": "training_free",
                 "parameters": {"window": 3}, "recipe": {}},
            ],
            "current_features": {"summary": _summary()},
        }

    def test_similarity_summary_present_with_matching_distance_stats(self):
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        summary = output.metadata["similarity_summary"]
        self.assertEqual(summary["similarity_metric"], DEFAULT_CONFIG.similarity_metric)
        self.assertEqual(summary["requested_k"], DEFAULT_CONFIG.similarity_knn_k)
        self.assertEqual(summary["matches_found"], len(output.similarity_matches))
        self.assertEqual(summary["matches_found"], 1)  # DB에 historical prefix가 1개뿐
        distances = [m.distance for m in output.similarity_matches]
        self.assertAlmostEqual(summary["min_distance"], min(distances))
        self.assertAlmostEqual(summary["max_distance"], max(distances))
        self.assertAlmostEqual(summary["mean_distance"], sum(distances) / len(distances))

    def test_fewer_matches_than_requested_k_produces_warning(self):
        # 기본 similarity_knn_k(30) > historical prefix 1개 -> warning이 붙는다.
        output = run_ml_pipeline(self.ml_input, db_path=self.database)
        self.assertTrue(any("similarity kNN" in w and "k=" in w for w in output.metadata["warnings"]))

    def test_enough_matches_does_not_produce_fewer_than_k_warning(self):
        # similarity_knn_k=1로 낮추면 historical prefix 1개로 충분하므로 warning이 없어야 한다.
        config = MLConfig(similarity_knn_k=1)
        output = run_ml_pipeline(self.ml_input, db_path=self.database, config=config)
        self.assertEqual(output.metadata["similarity_summary"]["matches_found"], 1)
        self.assertFalse(any("similarity kNN" in w and "k=" in w for w in output.metadata["warnings"]))

    def test_similarity_summary_reports_none_when_no_matches_found(self):
        # current_features를 historical과 공통 feature가 전혀 없도록 만들면 (전부 None)
        # matches_found=0이 되고 min/max/mean은 None이어야 한다 (억지로 값을 만들지 않는다).
        ml_input = {**self.ml_input, "current_features": {
            "summary": {column: None for column in SIMILARITY_FEATURE_COLUMNS}
        }}
        output = run_ml_pipeline(ml_input, db_path=self.database)
        summary = output.metadata["similarity_summary"]
        self.assertEqual(summary["matches_found"], 0)
        self.assertIsNone(summary["min_distance"])
        self.assertIsNone(summary["max_distance"])
        self.assertIsNone(summary["mean_distance"])


# ---------------------------------------------------------------------------
# 실제 dev18 DB에 대한 통합(smoke) 테스트
# ---------------------------------------------------------------------------


@unittest.skipUnless(REAL_DB_PATH.is_file(), "실제 dev18 DB가 없으면 통합 테스트를 건너뛴다")
class TestRealDatabaseIntegration(unittest.TestCase):
    def test_load_historical_prefixes_from_real_db(self):
        rows = db.load_historical_prefixes(REAL_DB_PATH)
        self.assertGreater(len(rows), 0)
        sample = rows[0]
        self.assertEqual(set(sample), {
            "prefix_feature_id", "csv_id", "csv_file", "family", "series", "q_percent",
            "training_boundary", "observed_row", *SIMILARITY_FEATURE_COLUMNS,
        })

    def test_load_results_for_a_real_candidate_has_cost_fields_joined(self):
        with closing(sqlite3.connect(REAL_DB_PATH)) as connection:
            config_id, head = connection.execute(
                "SELECT config_id, score_variant FROM results WHERE primary_score=1 AND status='complete' "
                "GROUP BY config_id, score_variant ORDER BY COUNT(*) DESC LIMIT 1"
            ).fetchone()
        rows = db.load_results_for_candidate(config_id, head, database=REAL_DB_PATH)
        self.assertGreater(len(rows), 0)
        with_cost = [row for row in rows if row["run_id"] is not None]
        self.assertGreater(len(with_cost), 0, "실제 DB에서 cost_result_links join이 비어있으면 안 된다")

    def test_similarity_and_trajectory_run_end_to_end_on_real_data(self):
        historical = db.load_historical_prefixes(REAL_DB_PATH)
        current, _reasons = extract_similarity_features({
            column: historical[0][column] for column in SIMILARITY_FEATURE_COLUMNS
        })
        matches, standardizer = find_similar_historical_prefixes(current, historical)
        self.assertGreater(len(matches), 0)
        self.assertLessEqual(len(matches), DEFAULT_CONFIG.similarity_knn_k)
        # 실제 데이터에도 population 전체가 NULL인 컬럼이 있을 수 있다 — 있어도 죽지 않아야 한다.
        self.assertIsInstance(standardizer.insufficient_data_columns, frozenset)

    def test_run_ml_pipeline_end_to_end_on_real_db_does_not_crash(self):
        import json as _json

        with closing(sqlite3.connect(REAL_DB_PATH)) as connection:
            config_id, head = connection.execute(
                "SELECT config_id, score_variant FROM results WHERE primary_score=1 AND status='complete' "
                "GROUP BY config_id, score_variant ORDER BY COUNT(*) DESC LIMIT 1"
            ).fetchone()
            model, settings_json = connection.execute(
                "SELECT model, settings_json FROM model_configs WHERE config_id=?", (config_id,),
            ).fetchone()
        settings = _json.loads(settings_json)
        candidate = {
            "config_id": config_id, "head": head, "model": model,
            "target_use": settings.get("target_use", ""),
            "parameters": settings.get("hyperparameters", {}),
            "recipe": settings.get("common_recipe", {}),
        }
        historical = db.load_historical_prefixes(REAL_DB_PATH)
        anchor = historical[0]
        summary = {column: anchor[column] for column in SIMILARITY_FEATURE_COLUMNS}
        ml_input = {
            "input": {"row_count": 500, "channel_count": 19, "sensor_columns": [f"s{i}" for i in range(19)]},
            "normal_prefix": pd.DataFrame({"s0": [1.0]}),
            "current_model": {"model": "", "settings": ""},
            "operating_conditions": {"operating_days": 30, "collection_rows_per_second": 0.01,
                                      "inference_rows_per_day": 200},
            "candidates": [candidate],
            "current_features": {"summary": summary},
        }
        output = run_ml_pipeline(ml_input, db_path=REAL_DB_PATH)
        self.assertIsInstance(output, MLOutput)
        self.assertGreater(len(output.future_stages), 1)
        self.assertIn(0, output.stage_candidates)


if __name__ == "__main__":
    unittest.main()
