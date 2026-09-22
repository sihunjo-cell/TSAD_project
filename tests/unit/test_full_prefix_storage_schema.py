"""전체 prefix 저장의 단일 관측 행 수와 과거 증거 호환을 확인한다."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy

from src.common.execution_evidence import (
    FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
    FULL_PREFIX_STORAGE_SCHEMA_VERSION,
    LEGACY_FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
    build_execution_evidence,
    validate_execution_evidence,
    validate_training_evidence_bindings,
)
from src.data_split.split_ratio_prefix import split_ratio_prefix
from src.common.save_scores import save_score_arrays
from src.common.run_registered_model import _finish_execution_result, prepare_session_inputs


class FullPrefixStorageSchemaTests(TestCase):
    def evidence(self, protocol=FULL_PREFIX_MEASUREMENT_PROTOCOL_ID):
        _, _, split = split_ratio_prefix(numpy.ones((500, 2)), 5, full_prefix=True)
        duration = {"observed_duration_seconds": None, "duration_basis": "unavailable"}
        return build_execution_evidence(
            split,
            {"split_preprocess_seconds": 1.0, "model_setup_seconds": 2.0,
             "training_seconds": 3.0, "validation_inference_seconds": 0.0,
             "calibration_inference_seconds": 4.0, "test_inference_seconds": 5.0},
            spec={"dataset_role": "development", "target_use": "fit_full_prefix", "ratio": 5},
            measurement_protocol_id=protocol, retry_count=0,
            training_session_durations=[duration], test_input_sessions=[numpy.ones((10, 2))],
            test_session_durations=[duration], peak_memory_mb=None if protocol == FULL_PREFIX_MEASUREMENT_PROTOCOL_ID else 0.0,
            model_artifact_bytes=12,
        )

    def test_current_storage_keeps_one_prefix_count_and_no_validation_timing(self):
        evidence = self.evidence()
        self.assertEqual(evidence["storage_schema_version"], FULL_PREFIX_STORAGE_SCHEMA_VERSION)
        self.assertEqual(evidence["training_sessions"], [{
            "training_boundary": 500, "observed_row": 25,
            "observed_duration_seconds": None, "duration_basis": "unavailable",
        }])
        self.assertNotIn("validation_inference_seconds", evidence["timing"])
        self.assertEqual(evidence["runtime_seconds"], 15.0)
        self.assertIsNone(evidence["peak_memory_mb"])
        self.assertEqual(evidence["resource_usage"]["status"], "unavailable")
        validate_training_evidence_bindings(
            evidence, ratio=5, target_use="fit_full_prefix", calibration_source="none",
            validation_session_lengths=(),
        )
        for forbidden in ("fit_rows", "fit_count", "validation_rows", "validation_count",
                          "val_rows", "val_count", "available_count", "observed_prefix_count"):
            malformed = deepcopy(evidence)
            malformed["training_sessions"][0][forbidden] = 0
            with self.assertRaises(ValueError):
                validate_execution_evidence(malformed)

    def test_legacy_full_prefix_evidence_stays_readable_without_migration(self):
        evidence = self.evidence(LEGACY_FULL_PREFIX_MEASUREMENT_PROTOCOL_ID)
        original = deepcopy(evidence)
        self.assertEqual(validate_execution_evidence(evidence), original)
        self.assertNotIn("storage_schema_version", evidence)
        self.assertEqual(evidence["training_sessions"][0]["fit_count"], 25)

    def test_ratio_drift_and_fit_calibration_outside_prefix_are_rejected(self):
        evidence = self.evidence()
        with self.assertRaises(ValueError):
            validate_training_evidence_bindings(
                evidence, ratio=10, target_use="fit_full_prefix", calibration_source="none",
                validation_session_lengths=(),
            )
        with self.assertRaises(ValueError):
            validate_training_evidence_bindings(
                evidence, ratio=5, target_use="fit_full_prefix", calibration_source="fit",
                validation_session_lengths=(26,), validation_source_starts=(0,),
            )

    def test_full_prefix_guard_rejects_a_same_length_tail_view(self):
        values = numpy.arange(1000).reshape(500, 2)
        _, validation, split = split_ratio_prefix(values, 5, full_prefix=True)
        with patch("src.common.run_registered_model.split_ratio_prefix",
                   return_value=(values[1:26], validation, split)):
            with self.assertRaises(ValueError):
                prepare_session_inputs(normal_training=values, test_sessions=(values[:3],),
                                       ratio_percent=5, scale=False, full_prefix=True)

    def test_cpu_backend_is_preserved_when_the_host_requested_cuda(self):
        result = {"timing": {"validation_inference_seconds": 0.0, "accelerator": "cuda"},
                  "validation_outputs": (), "input_column": 3}
        spec = {"model": "PCA_LEGACY", "tier": "t1", "target_use": "fit_full_prefix"}
        observed = _finish_execution_result(result, spec, {"n_components": .5}, "cuda")
        self.assertEqual(observed["effective_execution"]["backend"], "cpu")
        self.assertEqual(observed["effective_execution"]["input_column"], 3)
        self.assertNotIn("validation_inference_seconds", observed["timing"])

    def test_failed_array_write_keeps_the_existing_complete_file(self):
        with TemporaryDirectory() as directory:
            arguments = (directory, "DEV18", 1, "PCA_LEGACY", "t1", 5, 0, "trainnorm")
            paths = save_score_arrays(numpy.arange(8), *arguments)
            original = {path: Path(path).read_bytes() for path in paths}

            def fail_write(destination, values):
                destination.write(b"incomplete")
                raise OSError("storage failure")

            with patch("src.common.save_scores.numpy.save", side_effect=fail_write):
                with self.assertRaises(OSError):
                    save_score_arrays(numpy.arange(8), *arguments)
            self.assertEqual({path: Path(path).read_bytes() for path in paths}, original)
            self.assertEqual(sorted(Path(directory).iterdir()), sorted(Path(path) for path in paths))
