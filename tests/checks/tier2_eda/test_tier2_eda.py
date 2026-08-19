import inspect
import math
import tempfile
from pathlib import Path
import unittest

import numpy

from src.data_split.take_training_prefix import compute_kept_length
from tests.checks.tier2_eda.audit_channel_activity import (
    build_scope_summary_rows,
    is_constant_series,
    summarize_channel_activity,
    validate_nested_constant_sets,
)
from tests.checks.tier2_eda.audit_label_alignment import build_score_alignment_rows, restore_score_alignment
from tests.checks.tier2_eda.audit_model_feasibility import build_model_feasibility_rows
from tests.checks.tier2_eda.audit_model_forward import expected_forward_contracts
from tests.checks.tier2_eda.audit_model_parameters import build_model_parameter_rows
from tests.checks.tier2_eda.audit_training_ratios import (
    MODEL_CANDIDATES,
    build_ratio_rows,
    build_training_workload_summary_rows,
    compute_split_lengths,
    compute_window_count,
    compute_window_counts_by_segment,
)
from tests.checks.tier2_eda import run_tier2_eda


sha256_path = run_tier2_eda.sha256_path


class Tier2EdaTest(unittest.TestCase):
    def test_manifest_hash_uses_the_generated_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "result.csv"
            path.write_bytes(b"ratio,value\n5,1\n")
            self.assertEqual(
                sha256_path(path),
                "6fe1a177016cb660f7bb3ab1d4a55410a24c776966766aa55c25b740b68acc81",
            )

    def test_project_split_and_validation_rules(self):
        self.assertEqual(compute_kept_length(101, 0.05), 6)
        self.assertEqual(compute_split_lengths(101, 5), (90, 5, 11))
        self.assertEqual(compute_split_lengths(101, 100), (90, 90, 11))

    def test_window_count_uses_model_target(self):
        self.assertEqual(compute_window_count(10, 4, 2, 0), 4)
        self.assertEqual(compute_window_count(10, 4, 2, 1), 3)
        self.assertEqual(compute_window_count(4, 4, 1, 1), 0)

    def test_windows_do_not_cross_file_or_session_boundaries(self):
        per_segment = compute_window_counts_by_segment([6, 6], 4, 1, 1)
        self.assertEqual(per_segment, [2, 2])
        self.assertNotEqual(sum(per_segment), compute_window_count(12, 4, 1, 1))

    def test_constant_and_later_active_channels_are_reported(self):
        for values in ([1, 1, 1, 1], [0, 0, 0, 0], [3.7, 3.7, 3.7]):
            self.assertTrue(is_constant_series(numpy.array(values)))

    def test_equal_channels_are_not_constant_when_they_change_over_time(self):
        rows = summarize_channel_activity(
            numpy.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            ["left", "right"], 5, scope="fit_subset",
        )
        self.assertFalse(rows[0]["constant"])
        self.assertFalse(rows[1]["constant"])

    def test_constant_flags_agree_for_finite_inputs(self):
        row = summarize_channel_activity(
            numpy.array([[3.7], [3.7], [3.7]]), ["sensor"], 5, scope="fit_subset"
        )[0]
        self.assertEqual((row["unique_value_count"], row["range"], row["constant"]), (1, 0.0, True))
        self.assertEqual((row["minimum"], row["maximum"], row["variance"]), (3.7, 3.7, 0.0))
        self.assertEqual((row["iqr"], row["iqr_zero"]), (0.0, True))
        self.assertTrue(row["zero_range"])
        self.assertFalse(row["nonfinite"])
        self.assertTrue(row["scaler_execution_ready"])
        self.assertFalse(row["variation_available"])

    def test_nonfinite_values_are_the_scaler_execution_failure(self):
        row = summarize_channel_activity(
            numpy.array([[1.0], [numpy.nan]]), ["sensor"], 5, scope="fit_subset"
        )[0]
        self.assertTrue(row["nonfinite"])
        self.assertFalse(row["scaler_execution_ready"])

    def test_nested_prefix_constant_set_cannot_grow(self):
        rows = [
            {"dataset": "GHL", "file_or_session": "series_01", "channel": "sensor", "scope": "fit_subset", "ratio": 5, "constant": False},
            {"dataset": "GHL", "file_or_session": "series_01", "channel": "sensor", "scope": "fit_subset", "ratio": 10, "constant": True},
        ]
        with self.assertRaisesRegex(ValueError, "active.*constant"):
            validate_nested_constant_sets(rows)

    def test_scope_summary_keeps_fit_pool_subset_and_fixed_validation_separate(self):
        fit_pool = numpy.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [1.0, 3.0]])
        rows = []
        rows += summarize_channel_activity(fit_pool, ["a", "b"], 5, scope="fit_pool")
        rows += summarize_channel_activity(fit_pool[:3], ["a", "b"], 5, scope="fit_subset")
        rows += summarize_channel_activity(fit_pool[3:], ["a", "b"], 5, scope="fixed_validation")
        summaries = {row["scope"]: row for row in build_scope_summary_rows(rows)}
        self.assertEqual(summaries["fit_pool"]["constant_file_channel_combinations"], 0)
        self.assertEqual(summaries["fit_subset"]["constant_file_channel_combinations"], 1)
        self.assertEqual(summaries["fixed_validation"]["constant_file_channel_combinations"], 2)
        self.assertEqual(summaries["fixed_validation"]["iqr_zero_file_channel_combinations"], 2)
        self.assertEqual(summaries["fixed_validation"]["zero_range_file_channel_combinations"], 2)
        self.assertEqual(
            summaries["fixed_validation"]["scaler_execution_not_ready_file_channel_combinations"],
            0,
        )
        self.assertEqual(summaries["fixed_validation"]["variation_available_file_channel_combinations"], 0)

    def test_score_offset_restores_original_axis(self):
        restored = restore_score_alignment(numpy.array([2.0, 3.0]), 5, 3)
        numpy.testing.assert_array_equal(restored[:3], numpy.zeros(3))
        numpy.testing.assert_array_equal(restored[3:], numpy.array([2.0, 3.0]))

    def test_required_rows_include_trace_columns(self):
        rows = summarize_channel_activity(numpy.array([[0.0], [1.0]]), ["sensor"], 10)
        self.assertTrue({"dataset", "file_or_session", "ratio", "stage", "channel"} <= rows[0].keys())

    def test_feasibility_rows_preserve_window_definition(self):
        row = build_ratio_rows("GHL", "series_01", 1_000, 5, feature_count=19)[0]
        self.assertTrue({"model_input_window_size", "target_horizon", "train_stride", "drop_last"} <= row.keys())
        self.assertEqual(
            row["train_window_count"],
            compute_window_count(
                row["fit_subset_length"], row["model_input_window_size"], row["train_stride"], row["target_horizon"]
            ),
        )

    def test_model_feasibility_does_not_estimate_memory(self):
        ratio_row = build_ratio_rows("GHL", "series_01", 1_000, 5, feature_count=19)[0]
        row = build_model_feasibility_rows([ratio_row], feature_count=19)[0]
        self.assertFalse({
            "materialized_input_bytes_lower_bound",
            "materialized_input_gib_lower_bound",
            "memory_estimate_scope",
        } & row.keys())

    def test_manifest_statuses_separate_eda_closure_from_execution_readiness(self):
        self.assertTrue(hasattr(run_tier2_eda, "build_manifest_statuses"))
        statuses = run_tier2_eda.build_manifest_statuses(
            artifacts_generated=True,
            row_counts_match=True,
            checksums_match=True,
            environment_verified=True,
            source_choice_count=0,
            ratios_fixed=True,
            fixed_validation_invariant=True,
            fit_subsets_nested=True,
            infeasible_count=0,
            boundary_crossing_count=0,
            alignment_passed=True,
            parameters_fixed=True,
            full_imports_passed=False,
            adapter_count=2,
            reference_forward_passed=False,
        )
        self.assertEqual(statuses, {
            "eda_generation_success": True,
            "parameter_eda_closed": True,
            "tier2_execution_ready": False,
        })

    def test_manifest_hashes_the_split_and_loader_code(self):
        self.assertTrue({
            "src/data_split/validation_split.py",
            "src/data_split/load_ghl_series.py",
            "src/data_split/load_hai_sessions.py",
        } <= set(run_tier2_eda.GENERATION_CODE_FILES))

    def test_model_windows_are_fixed_across_ratios_and_datasets(self):
        expected = {"CI-AE": 100, "LSTM-AD": 100, "USAD": 10, "GDN": 5}
        for ratio in (5, 10, 20, 40, 60, 80, 100):
            rows = build_ratio_rows("GHL", "series_01", 50_000, ratio, feature_count=19)
            self.assertEqual({row["model"]: row["model_input_window_size"] for row in rows}, expected)
        for ratio in (5, 10, 20, 40, 60, 80, 100):
            hai = build_ratio_rows("HAI", "train_1", 280_800, ratio, feature_count=86)
            self.assertEqual(
                [(row["model"], row["model_input_window_size"]) for row in hai],
                [("GDN", 5)],
            )

    def test_partial_last_batch_is_feasible_when_not_dropped(self):
        row = next(
            row for row in build_ratio_rows("GHL", "series_01", 301, 100, feature_count=19)
            if row["model"] == "GDN"
        )
        self.assertNotEqual(row["last_train_batch_size"], 0)
        self.assertTrue(row["execution_feasible"])

    def test_window_count_and_score_offset_use_the_same_model_window(self):
        candidate = next(row for row in MODEL_CANDIDATES["GHL"] if row["model"] == "GDN")
        ratio_row = build_ratio_rows("GHL", "series_01", 1_000, 5, feature_count=19)[-1]
        score_row = build_score_alignment_rows("GHL", "series_01", 200, (candidate,))[0]
        self.assertEqual(ratio_row["model_input_window_size"], score_row["model_input_window_size"])
        self.assertEqual(score_row["first_valid_score_index"], 5)
        self.assertEqual(score_row["expected_saved_score_length"], 195)

    def test_reconstruction_and_forecast_scores_use_explicit_source_slices(self):
        rows = {
            row["model"]: row
            for row in build_score_alignment_rows("GHL", "series_01", 1_000, MODEL_CANDIDATES["GHL"])
        }
        self.assertEqual(
            (rows["CI-AE"]["source_time_index_of_first_core_score"], rows["CI-AE"]["expected_saved_score_length"]),
            (50, 901),
        )
        self.assertEqual(
            (rows["USAD"]["source_time_index_of_first_core_score"], rows["USAD"]["expected_saved_score_length"]),
            (5, 991),
        )
        self.assertEqual(
            (rows["LSTM-AD"]["source_time_index_of_first_core_score"], rows["LSTM-AD"]["expected_saved_score_length"]),
            (100, 900),
        )
        self.assertTrue(all(row["alignment_valid"] for row in rows.values()))

        ratio_rows = {
            row["model"]: row
            for row in build_ratio_rows("GHL", "series_01", 1_000, 100, feature_count=19)
        }
        self.assertEqual(ratio_rows["CI-AE"]["first_valid_score_index"], 50)
        self.assertEqual(ratio_rows["USAD"]["first_valid_score_index"], 5)
        self.assertEqual(
            ratio_rows["CI-AE"]["expected_saved_score_length"],
            "test_length-model_input_window_size+1",
        )

    def test_channel_independent_ae_counts_all_separate_models(self):
        ratio_rows = build_ratio_rows("GHL", "series_01", 50_000, 5, feature_count=19)
        summary = build_training_workload_summary_rows(ratio_rows, "GHL", feature_count=19)
        row = next(row for row in summary if row["model"] == "CI-AE")
        self.assertEqual(row["model_instance_count"], 19)
        self.assertEqual(row["train_window_count_all_models"], row["train_window_count_per_model"] * 19)
        self.assertEqual(
            row["optimizer_updates_per_epoch_all_models"],
            row["optimizer_updates_per_epoch_per_model"] * 19,
        )

    def test_hai_batches_are_counted_after_session_concatenation(self):
        ratio_rows = []
        for source, length in (("train_1", 100), ("train_2", 101), ("train_3", 102), ("train_4", 103)):
            ratio_rows.extend(build_ratio_rows("HAI", source, length, 100, feature_count=86))
        row = build_training_workload_summary_rows(ratio_rows, "HAI", feature_count=86)[0]
        self.assertEqual(row["training_scope"], "all_train_sessions")
        self.assertEqual(
            row["optimizer_updates_per_epoch_all_models"],
            math.ceil(row["train_window_count_all_models"] / row["batch_size"]),
        )

    def test_usad_counts_two_optimizer_steps_for_seventy_epochs(self):
        row = next(
            row for row in build_ratio_rows("GHL", "series_01", 50_000, 5, feature_count=19)
            if row["model"] == "USAD"
        )
        self.assertEqual(row["max_epochs"], 70)
        self.assertEqual(row["optimizer_steps_per_batch"], 2)
        self.assertEqual(row["optimizer_updates_per_epoch"], 2 * row["train_batch_count_per_epoch"])
        self.assertEqual(row["maximum_optimizer_updates"], 70 * row["optimizer_updates_per_epoch"])
        self.assertEqual(row["workload_status"], "fixed_project_transfer_rule")

    def test_parameter_table_separates_fixed_values_from_adapter_gaps(self):
        rows = build_model_parameter_rows("GHL")
        windows = {
            row["model"]: int(row["reference_value"])
            for row in rows
            if row["parameter"] == "input_window"
        }
        self.assertEqual(windows, {"CI-AE": 100, "LSTM-AD": 100, "USAD": 10, "GDN": 5})
        self.assertTrue(all(row["fixed_across_ratios"] for row in rows))
        allowed_statuses = {
            "fixed", "source_choice_required", "adapter_required",
            "local_source_mismatch", "verified",
        }
        self.assertEqual({row["status"] for row in rows} - allowed_statuses, set())
        self.assertTrue(any(row["status"] == "adapter_required" for row in rows))
        choices = {
            (row["model"], row["parameter"])
            for row in rows
            if row["status"] == "source_choice_required"
        }
        self.assertEqual(choices, set())
        fixed_project_values = {
            (row["model"], row["parameter"], row["reference_value"])
            for row in rows
            if row["status"] == "fixed"
        }
        self.assertIn(("USAD", "batch_size", 128), fixed_project_values)
        self.assertIn(("GDN", "topk", 5), fixed_project_values)
        gdn_stride = next(
            row["reference_value"]
            for row in rows
            if row["model"] == "GDN" and row["parameter"] == "train_and_evaluation_stride"
        )
        self.assertEqual(gdn_stride, "train=1; evaluation=1")

    def test_forward_contracts_cover_each_planned_model(self):
        contracts = expected_forward_contracts("GHL", feature_count=19)
        self.assertEqual(set(contracts), {"CI-AE", "LSTM-AD", "USAD", "GDN"})
        self.assertEqual(contracts["USAD"]["input_shape"], (2, 10, 19))
        self.assertEqual(contracts["GDN"]["output_shapes"], ((2, 19),))

    def test_normal_parameter_calculation_has_no_label_input(self):
        self.assertNotIn("label", " ".join(inspect.signature(build_ratio_rows).parameters))


if __name__ == "__main__":
    unittest.main()
