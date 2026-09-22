"""후속 비용 최적화에 넘길 실행 원자료 계약을 검증한다."""

import math
import unittest
from copy import deepcopy

import numpy

from src.common.execution_evidence import (
    build_execution_evidence,
    derive_execution_phase,
    validate_execution_evidence,
    validate_execution_evidence_for_run,
    validate_training_evidence_bindings,
)


VALID_EVIDENCE = {
    "measurement_protocol_id": "label_blind_timing.v1",
    "execution_phase": "development_hpo",
    "status": "complete",
    "retry_count": 0,
    "training_sessions": [{
        "available_count": 8,
        "fit_count": 6,
        "validation_count": 2,
        "observation_count": 10,
        "observed_duration_seconds": 9.0,
        "duration_basis": "timestamp",
    }],
    "test_sessions": [{
        "observation_count": 4,
        "observed_duration_seconds": None,
        "duration_basis": "unavailable",
    }],
    "timing": {
        "split_preprocess_seconds": 0.1,
        "model_setup_seconds": 0.0,
        "training_seconds": 0.2,
        "validation_inference_seconds": 0.3,
        "test_inference_seconds": 0.4,
    },
    "runtime_seconds": 1.0,
    "peak_memory_mb": 12.5,
    "model_artifact_bytes": 1024,
}


class TestValidateExecutionEvidence(unittest.TestCase):
    def test_returns_plain_json_safe_values(self):
        normalized = validate_execution_evidence(VALID_EVIDENCE)

        self.assertEqual(normalized, VALID_EVIDENCE)
        self.assertIsNot(normalized, VALID_EVIDENCE)
        self.assertIsNot(normalized["training_sessions"], VALID_EVIDENCE["training_sessions"])
        self.assertIs(type(normalized["timing"]), dict)
        self.assertIs(type(normalized["runtime_seconds"]), float)

    def test_rejects_non_mapping_and_top_level_field_drift(self):
        cases = []
        missing = deepcopy(VALID_EVIDENCE)
        missing.pop("retry_count")
        cases.append(missing)
        extra = {**VALID_EVIDENCE, "device": "unexpected"}
        cases.extend(([], extra))

        for index, values in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    validate_execution_evidence(values)

    def test_rejects_non_list_sessions_and_non_mapping_rows(self):
        cases = []
        tuple_sessions = deepcopy(VALID_EVIDENCE)
        tuple_sessions["test_sessions"] = tuple(tuple_sessions["test_sessions"])
        cases.append(tuple_sessions)
        object_row = deepcopy(VALID_EVIDENCE)
        object_row["training_sessions"] = [object()]
        cases.append(object_row)
        extra_row_field = deepcopy(VALID_EVIDENCE)
        extra_row_field["test_sessions"][0]["series"] = "01"
        cases.append(extra_row_field)
        missing_row_field = deepcopy(VALID_EVIDENCE)
        missing_row_field["training_sessions"][0].pop("duration_basis")
        cases.append(missing_row_field)

        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_execution_evidence(values)

    def test_rejects_invalid_counts_and_count_relationship(self):
        cases = []
        for field, value in (
            ("retry_count", True),
            ("retry_count", -1),
            ("model_artifact_bytes", 1.5),
        ):
            changed = deepcopy(VALID_EVIDENCE)
            changed[field] = value
            cases.append(changed)
        bool_count = deepcopy(VALID_EVIDENCE)
        bool_count["test_sessions"][0]["observation_count"] = True
        cases.append(bool_count)
        empty_test_session = deepcopy(VALID_EVIDENCE)
        empty_test_session["test_sessions"][0]["observation_count"] = 0
        cases.append(empty_test_session)
        inconsistent = deepcopy(VALID_EVIDENCE)
        inconsistent["training_sessions"][0]["available_count"] = 7
        cases.append(inconsistent)
        too_many_available = deepcopy(VALID_EVIDENCE)
        too_many_available["training_sessions"][0]["observation_count"] = 7
        cases.append(too_many_available)

        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_execution_evidence(values)

    def test_requires_explicit_duration_semantics(self):
        cases = []
        null_with_timestamp = deepcopy(VALID_EVIDENCE)
        null_with_timestamp["test_sessions"][0]["duration_basis"] = "timestamp"
        cases.append(null_with_timestamp)
        value_with_unavailable = deepcopy(VALID_EVIDENCE)
        value_with_unavailable["training_sessions"][0]["duration_basis"] = "unavailable"
        cases.append(value_with_unavailable)
        nonfinite = deepcopy(VALID_EVIDENCE)
        nonfinite["training_sessions"][0]["observed_duration_seconds"] = math.inf
        cases.append(nonfinite)
        unstable_basis = deepcopy(VALID_EVIDENCE)
        unstable_basis["training_sessions"][0]["duration_basis"] = "has spaces"
        cases.append(unstable_basis)

        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_execution_evidence(values)

    def test_rejects_timing_field_drift_nonfinite_values_and_wrong_sum(self):
        cases = []
        missing_setup = deepcopy(VALID_EVIDENCE)
        missing_setup["timing"].pop("model_setup_seconds")
        cases.append(missing_setup)
        missing = deepcopy(VALID_EVIDENCE)
        missing["timing"].pop("training_seconds")
        cases.append(missing)
        extra = deepcopy(VALID_EVIDENCE)
        extra["timing"]["total"] = 1.0
        cases.append(extra)
        boolean = deepcopy(VALID_EVIDENCE)
        boolean["timing"]["training_seconds"] = True
        cases.append(boolean)
        nonfinite = deepcopy(VALID_EVIDENCE)
        nonfinite["peak_memory_mb"] = math.nan
        cases.append(nonfinite)
        overflowing = deepcopy(VALID_EVIDENCE)
        overflowing["peak_memory_mb"] = 10 ** 10_000
        cases.append(overflowing)
        wrong_sum = deepcopy(VALID_EVIDENCE)
        wrong_sum["runtime_seconds"] += 1e-8
        cases.append(wrong_sum)

        for index, values in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    validate_execution_evidence(values)

    def test_runtime_includes_model_setup(self):
        with_setup = deepcopy(VALID_EVIDENCE)
        with_setup["timing"]["model_setup_seconds"] = 0.5
        with_setup["runtime_seconds"] = 1.5
        self.assertEqual(
            validate_execution_evidence(with_setup)["runtime_seconds"],
            1.5,
        )

        old_four_field_sum = deepcopy(with_setup)
        old_four_field_sum["runtime_seconds"] = 1.0
        with self.assertRaises(ValueError):
            validate_execution_evidence(old_four_field_sum)

    def test_returns_canonical_runtime_and_normalizes_fsum_overflow(self):
        within_tolerance = deepcopy(VALID_EVIDENCE)
        canonical = math.fsum(within_tolerance["timing"].values())
        within_tolerance["runtime_seconds"] = canonical + 5e-13
        self.assertEqual(
            validate_execution_evidence(within_tolerance)["runtime_seconds"],
            canonical,
        )

        overflowing = deepcopy(VALID_EVIDENCE)
        overflowing["timing"] = {
            field: 1e308 for field in overflowing["timing"]
        }
        overflowing["runtime_seconds"] = 0.0
        with self.assertRaises(ValueError):
            validate_execution_evidence(overflowing)

    def test_rejects_unstable_ids_noncomplete_status_and_wrong_phase(self):
        cases = []
        for field, value in (
            ("measurement_protocol_id", ""),
            ("measurement_protocol_id", "protocol with spaces"),
            ("execution_phase", "deployment"),
            ("execution_phase", []),
            ("status", "timeout"),
        ):
            changed = deepcopy(VALID_EVIDENCE)
            changed[field] = value
            cases.append(changed)

        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_execution_evidence(values)

    def test_target_free_has_no_training_session(self):
        accepted = deepcopy(VALID_EVIDENCE)
        accepted["execution_phase"] = "target_free_inference"
        accepted["training_sessions"] = []
        self.assertEqual(
            validate_execution_evidence(accepted)["training_sessions"], []
        )

        rejected = deepcopy(VALID_EVIDENCE)
        rejected["execution_phase"] = "target_free_inference"
        with self.assertRaises(ValueError):
            validate_execution_evidence(rejected)

        final_retraining_without_training = deepcopy(VALID_EVIDENCE)
        final_retraining_without_training["execution_phase"] = "final_retraining"
        final_retraining_without_training["training_sessions"] = []
        with self.assertRaises(ValueError):
            validate_execution_evidence(final_retraining_without_training)

        development_target_free = deepcopy(VALID_EVIDENCE)
        development_target_free["training_sessions"] = []
        self.assertEqual(
            validate_execution_evidence(development_target_free)["training_sessions"],
            [],
        )

        without_test = deepcopy(VALID_EVIDENCE)
        without_test["test_sessions"] = []
        with self.assertRaises(ValueError):
            validate_execution_evidence(without_test)


class TestExecutionPhase(unittest.TestCase):
    def test_derives_phase_from_dataset_role_and_target_use(self):
        cases = (
            ("development", "fit_validation", "development_hpo"),
            ("development", "training_free", "development_hpo"),
            ("development", "strict_zero_shot", "development_hpo"),
            ("final", "fit_validation", "final_retraining"),
            ("final", "training_free", "target_free_inference"),
            ("final", "strict_zero_shot", "target_free_inference"),
        )
        for dataset_role, target_use, expected in cases:
            with self.subTest(dataset_role=dataset_role, target_use=target_use):
                self.assertEqual(
                    derive_execution_phase(dataset_role, target_use), expected,
                )

        for dataset_role, target_use in (("unknown", "training_free"), ("final", "other")):
            with self.subTest(dataset_role=dataset_role, target_use=target_use):
                with self.assertRaises(ValueError):
                    derive_execution_phase(dataset_role, target_use)

    def test_validation_score_alignment_offset_is_explicit(self):
        validate_training_evidence_bindings(
            VALID_EVIDENCE, ratio=80,
            validation_session_lengths=(1,), validation_source_starts=(1,),
        )
        with self.assertRaises(ValueError):
            validate_training_evidence_bindings(
                VALID_EVIDENCE, ratio=80,
                validation_session_lengths=(1,), validation_source_starts=(0,),
            )

    def test_cross_checks_phase_and_training_sessions_before_use(self):
        development_target_free = deepcopy(VALID_EVIDENCE)
        development_target_free["training_sessions"] = []
        self.assertEqual(
            validate_execution_evidence_for_run(
                development_target_free,
                dataset_role="development",
                target_use="strict_zero_shot",
            )["execution_phase"],
            "development_hpo",
        )

        forged_phase = deepcopy(development_target_free)
        forged_phase["execution_phase"] = "target_free_inference"
        with self.assertRaises(ValueError):
            validate_execution_evidence_for_run(
                forged_phase,
                dataset_role="development",
                target_use="strict_zero_shot",
            )

        with self.assertRaises(ValueError):
            validate_execution_evidence_for_run(
                VALID_EVIDENCE,
                dataset_role="development",
                target_use="training_free",
            )


class TestBuildExecutionEvidence(unittest.TestCase):
    def test_builds_counts_from_runner_split_and_exact_timing(self):
        split = {
            "normal_range": (0, 10),
            "available_range": (0, 8),
            "fit_range": (0, 6),
            "validation_range": (6, 8),
            "unseen_range": (8, 10),
            "ratio_percent": 80,
        }
        runner_timing = {
            **VALID_EVIDENCE["timing"],
            "accelerator": "ignored_runner_detail",
        }

        evidence = build_execution_evidence(
            split,
            runner_timing,
            spec={
                "dataset_role": "development", "target_use": "fit_validation",
                "ratio": 80,
            },
            measurement_protocol_id="label_blind_timing.v1",
            retry_count=0,
            training_session_durations=[{
                "observed_duration_seconds": 9.0,
                "duration_basis": "timestamp",
            }],
            test_input_sessions=[numpy.zeros((4, 2))],
            test_session_durations=[{
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            peak_memory_mb=12.5,
            model_artifact_bytes=1024,
        )

        self.assertEqual(evidence, VALID_EVIDENCE)
        self.assertNotIn("accelerator", evidence["timing"])

    def test_rejects_split_range_drift_and_duration_count_mismatch(self):
        split = {
            "normal_range": (0, 10),
            "available_range": (0, 8),
            "fit_range": (0, 6),
            "validation_range": (6, 8),
        }
        arguments = {
            "spec": {
                "dataset_role": "development", "target_use": "fit_validation",
                "ratio": 80,
            },
            "measurement_protocol_id": "label_blind_timing.v1",
            "retry_count": 0,
            "training_session_durations": [],
            "test_input_sessions": [numpy.zeros((4, 2))],
            "test_session_durations": [{
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "peak_memory_mb": 12.5,
            "model_artifact_bytes": 1024,
        }
        with self.assertRaises(ValueError):
            build_execution_evidence(split, VALID_EVIDENCE["timing"], **arguments)

        malformed = deepcopy(split)
        malformed["validation_range"] = (5, 8)
        arguments["training_session_durations"] = [{
            "observed_duration_seconds": None,
            "duration_basis": "unavailable",
        }]
        with self.assertRaises(ValueError):
            build_execution_evidence(malformed, VALID_EVIDENCE["timing"], **arguments)

    def test_recomputes_every_prefix_range_from_ratio(self):
        split = {
            "normal_range": (0, 10),
            "available_range": (0, 8),
            "fit_range": (0, 6),
            "validation_range": (6, 8),
            "unseen_range": (8, 10),
            "ratio_percent": 80,
        }
        arguments = {
            "spec": {
                "dataset_role": "development", "target_use": "fit_validation",
                "ratio": 80,
            },
            "measurement_protocol_id": "label_blind_timing.v1",
            "retry_count": 0,
            "training_session_durations": [{
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "test_input_sessions": [numpy.zeros((4, 2))],
            "test_session_durations": [{
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "peak_memory_mb": 12.5,
            "model_artifact_bytes": 1024,
        }
        mutations = (
            {**split, "ratio_percent": 60},
            {**split, "available_range": (0, 7)},
            {**split, "fit_range": (0, 7), "validation_range": (7, 8)},
            {**split, "unseen_range": (7, 10)},
            {key: value for key, value in split.items() if key != "unseen_range"},
        )
        for index, malformed in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                build_execution_evidence(
                    malformed, VALID_EVIDENCE["timing"], **arguments,
                )

    def test_requires_spec_ratio_and_derives_test_observation_count(self):
        ratio_60_split = {
            "normal_range": (0, 10),
            "available_range": (0, 6),
            "fit_range": (0, 4),
            "validation_range": (4, 6),
            "unseen_range": (6, 10),
            "ratio_percent": 60,
        }
        arguments = {
            "measurement_protocol_id": "label_blind_timing.v1",
            "retry_count": 0,
            "training_session_durations": [{
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "test_input_sessions": [numpy.zeros((7, 2))],
            "test_session_durations": [{
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "peak_memory_mb": 12.5,
            "model_artifact_bytes": 1024,
        }
        with self.assertRaisesRegex(ValueError, "ratio"):
            build_execution_evidence(
                ratio_60_split,
                VALID_EVIDENCE["timing"],
                spec={
                    "dataset_role": "development",
                    "target_use": "fit_validation",
                    "ratio": 80,
                },
                **arguments,
            )

        target_free_arguments = {
            **arguments,
            "training_session_durations": [],
        }
        with self.assertRaisesRegex(ValueError, "ratio"):
            build_execution_evidence(
                None,
                VALID_EVIDENCE["timing"],
                spec={
                    "dataset_role": "development",
                    "target_use": "strict_zero_shot",
                    "ratio": 80,
                },
                **target_free_arguments,
            )

        evidence = build_execution_evidence(
            None,
            VALID_EVIDENCE["timing"],
            spec={
                "dataset_role": "development",
                "target_use": "strict_zero_shot",
                "ratio": 100,
            },
            **target_free_arguments,
        )
        self.assertEqual(evidence["test_sessions"][0]["observation_count"], 7)

    def test_rejects_empty_or_non_matrix_test_input_session(self):
        arguments = {
            "spec": {
                "dataset_role": "development",
                "target_use": "strict_zero_shot",
                "ratio": 100,
            },
            "measurement_protocol_id": "label_blind_timing.v1",
            "retry_count": 0,
            "training_session_durations": [],
            "test_session_durations": [{
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            }],
            "peak_memory_mb": 12.5,
            "model_artifact_bytes": 1024,
        }
        for session in (numpy.zeros((0, 2)), numpy.zeros(4), object()):
            with self.subTest(session=session), self.assertRaises(ValueError):
                build_execution_evidence(
                    None,
                    VALID_EVIDENCE["timing"],
                    test_input_sessions=[session],
                    **arguments,
                )


if __name__ == "__main__":
    unittest.main()
