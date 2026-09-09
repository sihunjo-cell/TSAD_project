"""Check native saved scores without constructing a model."""

import json
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy

from src.common.execution_evidence import FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, FULL_PREFIX_STORAGE_SCHEMA_VERSION
from src.common.naming import build_score_filename
from src.common.model_registry import load_model_registry
from tests.ghl_main.check_registered_outputs import check_registered_output


class TestPaperTuningOutputContract(unittest.TestCase):
    def tspulse_calibration(self):
        return {
            "schema_version": 1, "source": "full_evaluation", "source_start": 0,
            "source_end_exclusive": 4, "score_head": "time",
            "input_standard_scaler": {"mean": [1.0, 7.0], "variance": [4.0, 0.0],
                                      "scale": [2.0, 1.0], "sample_count": 4},
            "head_minmax": {"time": {"data_min": 0.0, "data_max": 3.0, "data_range": 3.0,
                                      "transform": "sklearn.preprocessing.MinMaxScaler()",
                                      "sample_count": 4, "score_exponent": 1.0,
                                      "least_significant_scale": 0.0, "least_significant_score": 1.0}},
            "output_maximum": 0.0, "output_maximum_epsilon": 1e-5, "output_maximum_divisor": 1e-5,
            "output_minmax_scaler": {"data_min": [0.0], "data_max": [0.0], "data_range": [0.0],
                                     "scale": [1.0], "offset": [0.0], "sample_count": 4},
        }

    def check_fixture(self, model_name, *, changed_metadata=None, changed_smoothed=False,
                      channel_fault=None):
        registered_model = load_model_registry()["models"][model_name]
        target_use = registered_model["target_use"]
        calibration = "official_full_evaluation" if model_name in {"GDN", "TSPulse"} else "none"
        identity = {"dataset_role": "development", "split_role": "dev18_selection",
                    "input_manifest_sha256": "a" * 64, "final_policy_membership_sha256": None}
        spec = {**identity, "model": model_name, "tier": registered_model["tier"],
                "target_use": target_use, "config_id": "c" + "b" * 12,
                "config_registry_sha256": "c" * 64, "ratio": 100, "seed": 0,
                "score_variants": ("time",) if model_name == "TSPulse" else ("",),
                "common_recipe": {"training_split": "full_prefix_v2", "methodology_revision": "paper_tuning_v4"}}
        registry = {"models": {model_name: {"tier": spec["tier"], "target_use": target_use,
                    "candidates": [{"config_id": spec["config_id"]}],
                    "preprocess_recipe": registered_model["preprocess_recipe"]}}}
        evidence = {"measurement_protocol_id": FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
                    "test_sessions": [{"observation_count": 4}]}
        metadata = {**spec, "dataset": "DEV18", "series": 1, "score_length": 4,
                    "score_shape": [4], "source_start": 0, "source_end_exclusive": 4,
                    "score_variant": "time" if model_name == "TSPulse" else None,
                    "execution_evidence": evidence, "storage_schema_version": FULL_PREFIX_STORAGE_SCHEMA_VERSION,
                    "calibration_mode": calibration, "calibration_reference": None,
                    "native_postprocessing": True, "official_protocol": "paper_tuning_v4",
                    "smoothing": {"kind": "model_native", "window": 0, "boundary": "model_native"}}
        if model_name == "GDN":
            metadata.update(native_channel_score_shape=[4, 2], native_calibration={
                "source": "full_evaluation", "source_start": 0, "source_end_exclusive": 4,
                "score_space": "absolute_error", "median": [0.5, 0.0], "iqr": [0.2, 0.0],
                "epsilon": 0.01,
            })
        if model_name == "TSPulse":
            metadata.update(native_calibration=self.tspulse_calibration(), effective_execution={"input_column": 2})
        metadata.update(changed_metadata or {})
        with TemporaryDirectory() as directory, ExitStack() as stack:
            for name, returned in (
                ("validate_execution_identity", identity),
                ("validate_input_manifest_role", identity["input_manifest_sha256"]),
                ("load_model_registry_with_sha", (registry, spec["config_registry_sha256"])),
                ("validate_execution_evidence_for_run", evidence),
                ("validate_training_evidence_bindings", None),
            ):
                stack.enter_context(patch(f"tests.ghl_main.check_registered_outputs.{name}", return_value=returned))
            for kind in ("raw", "smoothed"):
                name = build_score_filename(
                    dataset="DEV18", series=1, model=model_name, tier=spec["tier"],
                    ratio=100, seed=0, smoothing_kind=kind, norm_kind="trainnorm",
                )
                scores = numpy.arange(4, dtype=float)
                if kind == "smoothed" and changed_smoothed:
                    scores[0] += 1
                numpy.save(Path(directory) / name, scores)
                if model_name == "GDN" and channel_fault != "missing":
                    channels = numpy.column_stack((numpy.arange(4, dtype=float), numpy.zeros(4)))
                    if channel_fault == "shape":
                        channels = channels[:, :1]
                    elif channel_fault == "aggregation":
                        channels[-1, 0] += 1
                    elif channel_fault == "smoothed" and kind == "smoothed":
                        channels[0, 1] -= 1
                    numpy.save(Path(directory) / name.replace(".npy", "__channels.npy"), channels)
                if kind == "raw":
                    (Path(directory) / name.replace(".npy", ".meta.json")).write_text(
                        json.dumps(metadata), encoding="utf-8",
                    )
            return check_registered_output(
                directory, spec, dataset="DEV18", series=1, input_manifest_path="unused",
                score_variant="time" if model_name == "TSPulse" else None,
            )

    def test_native_full_evaluation_and_uncalibrated_outputs_are_accepted(self):
        for name in load_model_registry()["models"]:
            with self.subTest(model=name):
                self.assertEqual(self.check_fixture(name)["status"], "complete")

    def test_tspulse_requires_registry_full_evaluation_calibration(self):
        with self.assertRaisesRegex(ValueError, "calibration_mode"):
            self.check_fixture("TSPulse", changed_metadata={"calibration_mode": "none"})

    def test_tspulse_requires_numeric_calibration_bound_to_head_extent_and_channels(self):
        invalid = [None]
        for keys, value in (
            (("score_head",), "fft"), (("source_end_exclusive",), 5),
            (("input_standard_scaler", "mean"), [1.0]),
            (("input_standard_scaler", "variance"), [float("nan"), 0.0]),
            (("input_standard_scaler", "scale"), [0.0, 1.0]),
            (("head_minmax",), {}), (("head_minmax", "time", "data_range"), 2.0),
            (("output_maximum_divisor",), 0.1),
            (("output_minmax_scaler", "sample_count"), 3),
            (("output_minmax_scaler", "scale"), [float("inf")]),
        ):
            calibration = deepcopy(self.tspulse_calibration())
            target = calibration
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            invalid.append(calibration)
        for calibration in invalid:
            with self.subTest(calibration=calibration), self.assertRaisesRegex(ValueError, "TSPulse"):
                self.check_fixture("TSPulse", changed_metadata={"native_calibration": calibration})

    def test_official_proof_and_smoothing_are_required(self):
        for change in (
            {"official_protocol": "source_faithful_v3"},
            {"native_postprocessing": False},
            {"smoothing": {"kind": "trailing_average", "window": 4, "boundary": "zero"}},
            {"calibration_reference": {"source": "fit"}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.check_fixture("GDN", changed_metadata=change)

    def test_modified_smoothed_array_is_incomplete(self):
        with self.assertRaisesRegex(ValueError, "raw·smoothed"):
            self.check_fixture("GDN", changed_smoothed=True)

    def test_gdn_requires_matching_native_channels(self):
        for fault in ("missing", "shape", "aggregation", "smoothed"):
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.check_fixture("GDN", channel_fault=fault)
        with self.assertRaises(ValueError):
            self.check_fixture("GDN", changed_metadata={"native_channel_score_shape": None})

    def test_gdn_requires_valid_native_calibration(self):
        calibration = {
            "source": "full_evaluation", "source_start": 0, "source_end_exclusive": 4,
            "score_space": "absolute_error", "median": [0.5, 0.0], "iqr": [0.2, 0.0],
            "epsilon": 0.01,
        }
        invalid = [None, *({**calibration, **change} for change in (
            {"median": [0.5]}, {"median": [float("nan"), 0.0]}, {"iqr": [-0.2, 0.0]},
            {"epsilon": 0.02}, {"source": "current_prefix_fit"}, {"source_start": 1},
            {"source_start": 0.0},
        ))]
        for values in invalid:
            with self.subTest(calibration=values), self.assertRaises(ValueError):
                self.check_fixture("GDN", changed_metadata={"native_calibration": values})
