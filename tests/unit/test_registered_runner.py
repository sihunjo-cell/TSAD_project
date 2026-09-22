"""새 registry runner의 탐색 기회와 실데이터 gate를 검증한다."""

import inspect
import json
import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import numpy

from src.common.build_config_id import build_common_recipe_id
from src.common.save_model_artifacts import save_model_score as _save_model_score
from src.common.execution_evidence import (
    DEV18_MEASUREMENT_PROTOCOL_ID, FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
    FULL_PREFIX_STORAGE_SCHEMA_VERSION,
)
from src.common.model_registry import load_model_registry, model_registry_sha256
from src.common.run_registered_model import (
    build_entrypoint_arguments,
    execute_registered_model,
    load_model_entrypoint,
)
from src.data_split.split_ratio_prefix import compute_prefix_counts
from tests.ghl_main.check_registered_outputs import (
    check_registered_output as _check_registered_output,
)
from tests.ghl_main.run_registered_models import (
    RealDataExecutionBlocked,
    build_output_directory,
    build_specs,
    load_registered_inputs,
    run_batch,
)


INPUT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "input_manifest.yaml"
EXECUTION_IDENTITY_FIELDS = (
    "dataset_role", "split_role", "input_manifest_sha256",
    "final_policy_membership_sha256",
)
LEGACY_OUTPUT_RECIPE = {
    "methodology_revision": "source_faithful_v3", "training_split": "full_prefix_v2",
    "score_calibration": {
        "fit_validation": "validation_median_iqr", "target_free": "none",
        "full_prefix_scalar": "none", "full_prefix_channels": "fit_median_iqr",
        "epsilon": 0.01,
    },
    "smoothing": {"kind": "trailing_mean", "window": 4, "boundary": "first_three_timesteps_zero"},
    "order": {"raw": "normalize_then_aggregate", "smoothed": "normalize_then_smooth_per_channel_then_aggregate"},
    "aggregation": {"channel_scores": "max", "scalar_scores": "model_native"},
}


def execution_identity(spec: dict) -> dict:
    return {field: spec[field] for field in EXECUTION_IDENTITY_FIELDS}


def save_model_score(output, output_directory, **arguments):
    """합성 산출물은 과거 fit 교정·trailing-4 저장 계약으로 고정한다."""
    arguments["common_recipe"] = deepcopy(LEGACY_OUTPUT_RECIPE)
    arguments["common_recipe_id"] = build_common_recipe_id(arguments["common_recipe"])
    arguments["dataset"] = "DEV18"
    target_free = arguments["target_use"] in {"training_free", "strict_zero_shot"}
    full_prefix = arguments["target_use"] == "fit_full_prefix"
    full_prefix_protocol = arguments["common_recipe"].get("training_split") == "full_prefix_v2"
    training_sessions = []
    if not target_free:
        supplied = arguments.get("calibration_scores" if full_prefix else "validation_scores", ())
        references = (
            (supplied,)
            if isinstance(supplied, numpy.ndarray) and len(supplied)
            else tuple(supplied)
        )
        for index, reference in enumerate(references):
            validation_count = len(reference)
            ratio = arguments["ratio"]
            if full_prefix:
                starts = arguments.get("calibration_source_starts", (0,) * len(references))
                observation_count = (100 * (len(reference) + starts[index]) + ratio - 1) // ratio
            else:
                observation_count = (500 * validation_count + ratio - 1) // ratio
            available_count, fit_count, expected_validation = compute_prefix_counts(
                observation_count, ratio, full_prefix=full_prefix,
            )
            training_sessions.append({
                **({"training_boundary": observation_count, "observed_row": available_count}
                   if full_prefix_protocol else {
                       "available_count": available_count, "fit_count": fit_count,
                       "validation_count": expected_validation, "observation_count": observation_count,
                   }),
                "observed_duration_seconds": None,
                "duration_basis": "unavailable",
            })
    arguments.setdefault("execution_evidence", {
        **({"storage_schema_version": FULL_PREFIX_STORAGE_SCHEMA_VERSION,
            "resource_usage": {"status": "unavailable", "reason": "mock"}}
           if full_prefix_protocol else {}),
        "measurement_protocol_id": FULL_PREFIX_MEASUREMENT_PROTOCOL_ID if full_prefix_protocol else DEV18_MEASUREMENT_PROTOCOL_ID,
        "execution_phase": "development_hpo",
        "status": "complete", "retry_count": 0,
        "training_sessions": training_sessions,
        "test_sessions": [{
            "observation_count": output["source_end_exclusive"],
            "observed_duration_seconds": None, "duration_basis": "unavailable",
        }],
        "timing": {
            "split_preprocess_seconds": 0.0, "model_setup_seconds": 0.0,
            "training_seconds": 0.0,
            **({} if full_prefix_protocol else {"validation_inference_seconds": 0.0}),
            "test_inference_seconds": 0.0,
            **({"calibration_inference_seconds": 0.0} if full_prefix_protocol else {}),
        },
        "runtime_seconds": 0.0, "peak_memory_mb": 0.0,
        "model_artifact_bytes": 0,
    })
    return _save_model_score(output, output_directory, **arguments)


def check_registered_output(output_directory, spec, **arguments):
    arguments["dataset"] = "DEV18"
    arguments["input_manifest_path"] = INPUT_MANIFEST_PATH
    spec = {**spec, "common_recipe": deepcopy(LEGACY_OUTPUT_RECIPE),
            "common_recipe_id": build_common_recipe_id(LEGACY_OUTPUT_RECIPE)}
    registry = load_model_registry()
    registry["models"]["GDN"]["preprocess_recipe"]["calibration"] = "fit_median_iqr"
    with patch("tests.ghl_main.check_registered_outputs.load_model_registry_with_sha",
               return_value=(registry, model_registry_sha256())):
        return _check_registered_output(output_directory, spec, **arguments)


class TestSessionRunnerTiming(unittest.TestCase):
    def test_session_runner_rejects_each_missing_timing_phase(self):
        spec = next(
            spec for spec in build_specs("development")
            if spec["model"] == "GDN"
        )
        normal = numpy.arange(500 * 2, dtype=float).reshape(500, 2)
        test = numpy.arange(40, dtype=float).reshape(20, 2)
        complete_timing = {
            "model_setup_seconds": 0.4,
            "training_seconds": 0.1,
            "validation_inference_seconds": 0.2,
            "test_inference_seconds": 0.3,
            "calibration_inference_seconds": 0.2,
        }

        for missing in complete_timing:
            def session_runner(
                fit_sessions, validation_sessions, test_sessions, **arguments,
            ):
                return {
                    "checkpoint": None,
                    "validation_outputs": ({"scores": numpy.zeros(1)},),
                    "test_outputs": ({"scores": numpy.zeros(1)},),
                    "training_log": {},
                    "timing": {
                        field: value for field, value in complete_timing.items()
                        if field != missing
                    },
                }

            with self.subTest(missing=missing), patch(
                "src.common.run_registered_model.set_reproducible_seed",
                return_value={"seed": spec["seed"]},
            ), self.assertRaisesRegex(ValueError, "timing.*빠졌다"):
                execute_registered_model(
                    spec, normal_training=normal, test_sessions=(test,),
                    device="unused", entrypoint=session_runner,
                )

        for field in complete_timing:
            for invalid in (True, numpy.nan, -0.1):
                def session_runner(
                    fit_sessions, validation_sessions, test_sessions, **arguments,
                ):
                    return {
                        "checkpoint": None,
                        "validation_outputs": ({"scores": numpy.zeros(1)},),
                        "test_outputs": ({"scores": numpy.zeros(1)},),
                        "training_log": {},
                        "timing": {**complete_timing, field: invalid},
                    }

                with self.subTest(field=field, invalid=invalid), patch(
                    "src.common.run_registered_model.set_reproducible_seed",
                    return_value={"seed": spec["seed"]},
                ), self.assertRaisesRegex(ValueError, "timing.*유한한 0 이상의 수"):
                    execute_registered_model(
                        spec, normal_training=normal, test_sessions=(test,),
                        device="unused", entrypoint=session_runner,
                    )


class TestRegisteredSpecs(unittest.TestCase):
    def test_development_specs_follow_model_seed_and_ratio_contract(self):
        specs = build_specs("development")
        self.assertEqual(len(specs), 276)
        self.assertEqual({spec["model"] for spec in specs}, {
            "MWVAR", "SQDIFF_LAST1", "SQDIFF_LAST3", "SQDIFF_CENTERED5",
            "MWVAR96_SQDIFF_LAST3", "MWVAR96_SQDIFF_CENTERED5",
            "PCA_LEGACY", "PaAno", "GDN", "TimeRCD", "TSPulse",
        })

        mwvar = [spec for spec in specs if spec["model"] == "MWVAR"]
        self.assertEqual(len(mwvar), 11)
        self.assertEqual({(spec["ratio"], spec["seed"]) for spec in mwvar}, {(100, 0)})
        paano = [spec for spec in specs if spec["model"] == "PaAno"]
        self.assertEqual(len(paano), 9 * 7 * 3)
        self.assertEqual({spec["seed"] for spec in paano}, {0, 1, 2})

    def test_output_path_contains_tier_model_and_config_id(self):
        spec = next(spec for spec in build_specs("development") if spec["model"] == "PaAno")
        path = build_output_directory(Path("experiment"), spec)
        self.assertEqual(path.parts[:5], (
            "experiment", "scores", "dev18", "tier2", "PaAno",
        ))
        self.assertEqual(path.parts[5], spec["config_id"])

    def test_each_spec_owns_its_hyperparameter_mapping(self):
        paano = [
            spec for spec in build_specs("development")
            if spec["model"] == "PaAno"
        ]
        self.assertIsNot(paano[0]["hyperparameters"], paano[1]["hyperparameters"])

    def test_each_spec_owns_complete_executor_identity(self):
        specs = build_specs("development")
        first, second = [spec for spec in specs if spec["model"] == "PaAno"][:2]
        required = {
            "target_use", "source_commit", "source_checkpoint_sha256",
            "checkpoint_config_sha256", "checkpoint_revision",
            "preprocess_recipe", "common_recipe", "common_recipe_id",
        }
        self.assertTrue(required <= first.keys())
        self.assertIsNot(first["preprocess_recipe"], second["preprocess_recipe"])
        self.assertIsNot(first["common_recipe"], second["common_recipe"])
        before = deepcopy(second["common_recipe"])
        first["common_recipe"]["smoothing"]["window"] = 99
        self.assertEqual(second["common_recipe"], before)
        self.assertEqual(load_model_registry()["common_recipe"], before)

    def test_each_spec_captures_the_registry_sha(self):
        self.assertTrue(all(
            spec["config_registry_sha256"] == model_registry_sha256()
            for spec in build_specs("development")
        ))

    def test_specs_use_sha_returned_with_the_registry_snapshot(self):
        registry = load_model_registry()
        with patch(
            "tests.ghl_main.run_registered_models.load_model_registry_with_sha",
            return_value=(registry, "a" * 64),
        ):
            specs = build_specs("development")
        self.assertTrue(all(
            spec["config_registry_sha256"] == "a" * 64 for spec in specs
        ))

    def test_tspulse_heads_have_distinct_output_directories(self):
        spec = next(
            spec for spec in build_specs("development")
            if spec["model"] == "TSPulse"
        )
        paths = {
            build_output_directory(Path("experiment"), spec, score_variant=head)
            for head in ("time", "fft", "pred", "raw_max")
        }
        self.assertEqual(len(paths), 4)
        self.assertEqual({path.name for path in paths}, {"time", "fft", "pred", "raw_max"})

    def test_real_data_execution_requires_explicit_call_permission(self):
        key = ("MWVAR", "c43024819503c", 100)
        with self.assertRaises(RealDataExecutionBlocked):
            run_batch(
                dataset_role="development", allow_real_data=False,
                feasible_keys={key}, executor=lambda spec: spec,
            )

        executed = run_batch(
            dataset_role="development", allow_real_data=True,
            feasible_keys={key}, executor=lambda spec: spec,
        )
        self.assertEqual(
            [(spec["model"], spec["config_id"], spec["ratio"]) for spec in executed],
            [key],
        )

    def test_runner_input_dispatch_has_no_unsealed_role_parameters(self):
        parameters = inspect.signature(load_registered_inputs).parameters
        self.assertIn("spec", parameters)
        self.assertNotIn("dataset_role", parameters)
        self.assertNotIn("split_role", parameters)

    def test_real_batch_omits_a_model_changed_back_to_pending(self):
        self.assertNotIn("include_pending", inspect.signature(run_batch).parameters)
        registry = deepcopy(load_model_registry())
        registry["models"]["TimeRCD"]["execution_status"] = "pending_checkpoint_smoke"
        registry["models"]["TimeRCD"]["status_reason"] = "unit test"
        with patch(
            "tests.ghl_main.run_registered_models.load_model_registry_with_sha",
            return_value=(registry, model_registry_sha256()),
        ):
            specs = build_specs("development")
            feasible_keys = frozenset(
                (spec["model"], spec["config_id"], spec["ratio"])
                for spec in specs
            )
            executed = run_batch(
                dataset_role="development", allow_real_data=True,
                feasible_keys=feasible_keys, executor=lambda spec: spec,
            )
        self.assertFalse(any(spec["model"] == "TimeRCD" for spec in executed))
        self.assertTrue(any(spec["model"] == "TSPulse" for spec in executed))

    def test_feasibility_allowlist_filters_seed_expansion(self):
        self.assertIn("feasible_keys", inspect.signature(build_specs).parameters)
        candidate = next(
            spec for spec in build_specs("development")
            if (spec["model"], spec["ratio"]) == ("GDN", 20)
        )
        key = (candidate["model"], candidate["config_id"], candidate["ratio"])

        specs = build_specs("development", feasible_keys=frozenset({key}))

        self.assertEqual(len(specs), 3)
        self.assertEqual(
            {(spec["model"], spec["config_id"], spec["ratio"]) for spec in specs},
            {key},
        )

    def test_feasibility_allowlist_rejects_invalid_keys(self):
        development = build_specs("development")
        ready = next(spec for spec in development if spec["model"] == "GDN")
        time_rcd = next(spec for spec in development if spec["model"] == "TimeRCD")
        invalid_keys = (
            (("GDN", ready["config_id"]), "키"),
            (("Unknown", ready["config_id"], 20), "ready"),
            (("GDN", "c000000000000", 20), "config_id"),
            (("GDN", ready["config_id"], 30), "ratio"),
            (("TimeRCD", time_rcd["config_id"], 5), "ratio"),
        )
        for key, message in invalid_keys:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, message):
                    build_specs("development", feasible_keys=frozenset({key}))

    def test_unknown_execution_status_is_fail_closed(self):
        registry = deepcopy(load_model_registry())
        registry["models"]["MWVAR"]["execution_status"] = "misspelled_ready"
        with patch(
            "tests.ghl_main.run_registered_models.load_model_registry_with_sha",
            return_value=(registry, model_registry_sha256()),
        ):
            with self.assertRaisesRegex(ValueError, "execution_status"):
                build_specs("development")

    def test_include_pending_never_includes_unavailable(self):
        registry = deepcopy(load_model_registry())
        registry["models"]["MWVAR"]["execution_status"] = "unavailable"
        registry["models"]["MWVAR"]["status_reason"] = "synthetic exclusion"
        with patch(
            "tests.ghl_main.run_registered_models.load_model_registry_with_sha",
            return_value=(registry, model_registry_sha256()),
        ):
            specs = build_specs("development", include_pending=True)
        self.assertFalse(any(spec["model"] == "MWVAR" for spec in specs))

    def test_every_registered_model_has_a_lazy_entrypoint(self):
        for model, details in load_model_registry()["models"].items():
            if details["execution_status"] == "unavailable":
                continue
            self.assertTrue(callable(load_model_entrypoint(model)))

    def test_every_registry_mapping_binds_to_its_entrypoint_signature(self):
        first_specs = {}
        for spec in build_specs("development"):
            first_specs.setdefault(spec["model"], spec)
        for model, spec in first_specs.items():
            arguments = build_entrypoint_arguments(
                spec, device="cpu", channel_count=19,
            )
            inspect.signature(load_model_entrypoint(model)).bind_partial(**arguments)

    def test_registry_keys_translate_to_model_arguments_without_runtime_identity(self):
        specs = build_specs("development")
        paano = next(spec for spec in specs if spec["model"] == "PaAno")
        paano_arguments = build_entrypoint_arguments(
            paano, device="cpu", channel_count=19,
        )
        self.assertEqual(paano_arguments["neighbors"], 3)
        self.assertNotIn("ratio", paano_arguments)
        self.assertNotIn("seed", paano_arguments)

        gdn = next(spec for spec in specs if spec["model"] == "GDN")
        gdn_arguments = build_entrypoint_arguments(
            gdn, device="cpu", channel_count=19,
        )
        self.assertEqual(gdn_arguments["embedding_dimension"], 64)
        self.assertEqual(gdn_arguments["hidden_dimension"], 128)
        self.assertEqual(gdn_arguments["learning_rate"], 0.001)

        tspulse = next(spec for spec in specs if spec["model"] == "TSPulse")
        tspulse_arguments = build_entrypoint_arguments(
            tspulse, device="cpu", channel_count=19,
        )
        self.assertEqual(tspulse_arguments["device"], "cpu")
        self.assertEqual(tspulse_arguments["batch_size"], 32)
        self.assertEqual(
            load_model_entrypoint("TimeRCD").__name__, "score_time_rcd_official",
        )

    def test_hardcoded_entrypoints_reject_registry_drift(self):
        first_specs = {}
        for spec in build_specs("development"):
            first_specs.setdefault(spec["model"], spec)
        mutations = {
            "MWVAR": ("ddof", 0),
            "SQDIFF_LAST3": ("lag", 2),
            "PCA_LEGACY": ("zero_pruning", False),
            "TimeRCD": ("score_head", "logit"),
            "TSPulse": ("patch_size", 16),
        }
        for model, (name, value) in mutations.items():
            with self.subTest(model=model):
                spec = {**first_specs[model]}
                spec["hyperparameters"] = {
                    **spec["hyperparameters"], name: value,
                }
                with self.assertRaisesRegex(ValueError, "고정값"):
                    build_entrypoint_arguments(
                        spec, device="cpu", channel_count=19,
                    )

        extra = {**first_specs["MWVAR"]}
        extra["hyperparameters"] = {
            **extra["hyperparameters"], "unconsumed": 1,
        }
        with self.assertRaisesRegex(ValueError, "파라미터 키"):
            build_entrypoint_arguments(extra, device="cpu", channel_count=19)

    def test_output_checker_matches_metadata_to_registry_spec(self):
        spec = next(spec for spec in build_specs("development") if spec["model"] == "MWVAR")
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            save_model_score(
                {
                    "scores": numpy.arange(8, dtype=float),
                    "source_start": 0,
                    "source_end_exclusive": 8,
                    "alignment": "same_timestep",
                    "primitive": "sample_variance",
                    "calibration_mode": "none",
                    "evaluation_mode": "offline_noncausal",
                    "lookahead": 47,
                    "maximum_effective_lookahead": 95,
                    "normalization_scope": "none",
                },
                output_directory,
                dataset="GHL",
                series=1,
                model=spec["model"],
                target_use=spec["target_use"],
                tier=spec["tier"],
                ratio=spec["ratio"],
                seed=spec["seed"],
                config_id=spec["config_id"],
                common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="none",
                execution_identity=execution_identity(spec),
            )
            save_model_score(
                {
                    "scores": numpy.arange(6, dtype=float),
                    "source_start": 0,
                    "source_end_exclusive": 6,
                    "alignment": "same_timestep",
                    "primitive": "sample_variance",
                    "calibration_mode": "none",
                    "evaluation_mode": "offline_noncausal",
                    "lookahead": 47,
                    "maximum_effective_lookahead": 95,
                    "normalization_scope": "none",
                },
                output_directory, dataset="GHL", series=2,
                model=spec["model"], target_use=spec["target_use"],
                tier=spec["tier"], ratio=spec["ratio"],
                seed=spec["seed"], config_id=spec["config_id"],
                common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="none",
                execution_identity=execution_identity(spec),
            )

            result = check_registered_output(
                output_directory, spec, dataset="GHL", series=1,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["score_file_count"], 2)

    def test_dev18_output_checker_rejects_wrong_measurement_protocol_id(self):
        spec = next(spec for spec in build_specs("development") if spec["model"] == "MWVAR")
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            saved = save_model_score(
                {
                    "scores": numpy.arange(8, dtype=float),
                    "source_start": 0,
                    "source_end_exclusive": 8,
                    "alignment": "same_timestep",
                    "primitive": "sample_variance",
                    "calibration_mode": "none",
                    "evaluation_mode": "offline_noncausal",
                    "lookahead": 47,
                    "maximum_effective_lookahead": 95,
                    "normalization_scope": "none",
                },
                output_directory,
                series=1,
                model=spec["model"], target_use=spec["target_use"],
                tier=spec["tier"], ratio=spec["ratio"], seed=spec["seed"],
                config_id=spec["config_id"], common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="none", execution_identity=execution_identity(spec),
            )
            metadata_path = Path(saved["metadata_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["execution_evidence"]["measurement_protocol_id"] = "wrong.v2"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "measurement_protocol_id"):
                check_registered_output(output_directory, spec, series=1)

    def test_output_checker_cross_checks_evidence_count_and_calibration_mode(self):
        spec = next(spec for spec in build_specs("development") if spec["model"] == "MWVAR")
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            saved = save_model_score(
                {
                    "scores": numpy.arange(8, dtype=float),
                    "source_start": 0,
                    "source_end_exclusive": 8,
                    "alignment": "same_timestep",
                    "primitive": "sample_variance",
                    "calibration_mode": "none",
                    "evaluation_mode": "offline_noncausal",
                    "lookahead": 47,
                    "maximum_effective_lookahead": 95,
                    "normalization_scope": "none",
                },
                output_directory,
                dataset="GHL",
                series=1,
                model=spec["model"],
                target_use=spec["target_use"],
                tier=spec["tier"],
                ratio=spec["ratio"],
                seed=spec["seed"],
                config_id=spec["config_id"],
                common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="none",
                execution_identity=execution_identity(spec),
            )
            metadata_path = Path(saved["metadata_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            metadata["execution_evidence"]["test_sessions"][0][
                "observation_count"
            ] = 9
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "observation_count"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

            metadata["execution_evidence"]["test_sessions"][0][
                "observation_count"
            ] = 8
            metadata["calibration_mode"] = "validation_median_iqr"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target_use.*calibration_mode"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

    def test_output_checker_rebinds_learned_evidence_to_ratio_and_reference_shapes(self):
        spec = next(
            spec for spec in build_specs("development")
            if (spec["model"], spec["ratio"]) == ("GDN", 20)
        )
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            saved = save_model_score(
                {
                    "scores": numpy.array([[2.0, 20.0], [4.0, 40.0]]),
                    "source_start": 5,
                    "source_end_exclusive": 7,
                    "alignment": "next_step",
                    "primitive": "absolute_error",
                    "calibration_mode": "fit_median_iqr",
                    "evaluation_mode": "causal",
                    "lookahead": 0,
                    "maximum_effective_lookahead": 0,
                    "normalization_scope": "current_prefix_fit",
                },
                output_directory,
                dataset="GHL",
                series=1,
                model=spec["model"],
                target_use=spec["target_use"],
                tier=spec["tier"],
                ratio=spec["ratio"],
                seed=spec["seed"],
                config_id=spec["config_id"],
                common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="current_prefix_fit",
                calibration_scores=(numpy.array([[1.0, 10.0], [3.0, 30.0]]),),
                calibration_source_starts=(5,),
                execution_identity=execution_identity(spec),
            )
            self.assertEqual(
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )["status"],
                "complete",
            )
            metadata_path = Path(saved["metadata_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            training = metadata["execution_evidence"]["training_sessions"][0]

            training["observed_row"] = 8
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "spec ratio"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

            training.update({
                "observed_row": 6,
                "training_boundary": 30,
            })
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

    def test_output_checker_rejects_wrong_array_shape_even_when_length_matches(self):
        spec = next(spec for spec in build_specs("development") if spec["model"] == "MWVAR")
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            saved = save_model_score(
                {
                    "scores": numpy.arange(8, dtype=float),
                    "source_start": 0,
                    "source_end_exclusive": 8,
                    "alignment": "same_timestep",
                    "primitive": "sample_variance",
                    "calibration_mode": "none",
                    "evaluation_mode": "offline_noncausal",
                    "lookahead": 47,
                    "maximum_effective_lookahead": 95,
                    "normalization_scope": "none",
                },
                output_directory, dataset="GHL", series=1,
                model=spec["model"], target_use=spec["target_use"],
                tier=spec["tier"], ratio=spec["ratio"],
                seed=spec["seed"], config_id=spec["config_id"],
                common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="none",
                execution_identity=execution_identity(spec),
            )
            smoothed = next(
                Path(path) for path in saved["score_paths"] if "__smoothed__" in path
            )
            numpy.save(smoothed, numpy.arange(8, dtype=float).reshape(8, 1))
            with self.assertRaisesRegex(ValueError, "shape"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

    def test_output_checker_requires_declared_calibration_reference(self):
        spec = next(
            spec for spec in build_specs("development")
            if (spec["model"], spec["ratio"]) == ("GDN", 20)
        )
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            saved = save_model_score(
                {
                    "scores": numpy.array([[2.0, 20.0], [4.0, 40.0]]),
                    "source_start": 5,
                    "source_end_exclusive": 7,
                    "alignment": "next_step",
                    "primitive": "absolute_error",
                    "calibration_mode": "fit_median_iqr",
                    "evaluation_mode": "causal",
                    "lookahead": 0,
                    "maximum_effective_lookahead": 0,
                    "normalization_scope": "current_prefix_fit",
                },
                output_directory, dataset="GHL", series=1,
                model=spec["model"], target_use=spec["target_use"],
                tier=spec["tier"], ratio=spec["ratio"],
                seed=spec["seed"], config_id=spec["config_id"],
                common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="current_prefix_fit",
                calibration_scores=(numpy.array([[1.0, 10.0], [3.0, 30.0]]),),
                calibration_source_starts=(5,),
                execution_identity=execution_identity(spec),
            )
            self.assertEqual(
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )["status"],
                "complete",
            )
            reference_path = Path(saved["calibration_reference_path"])
            metadata_path = Path(saved["metadata_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            original_median = metadata["calibration_reference"]["median"]
            original_iqr = metadata["calibration_reference"]["iqr"]

            metadata["calibration_reference"]["median"] = [original_median]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "median"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

            metadata["calibration_reference"]["median"] = original_median
            metadata["calibration_reference"]["iqr"] = [
                original_iqr[0] + 1e-12, *original_iqr[1:],
            ]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "IQR"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

            metadata["calibration_reference"]["iqr"] = original_iqr
            alternate_path = output_directory / "alternate__calibration_reference.npz"
            alternate_path.write_bytes(reference_path.read_bytes())
            metadata["calibration_reference"]["file"] = alternate_path.name
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "파일명"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

            metadata["calibration_reference"]["file"] = reference_path.name
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            reference_path.write_bytes(reference_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )
            reference_path.unlink()

            with self.assertRaisesRegex(ValueError, "교정 기준"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

    def test_output_checker_rejects_stale_registry_sha(self):
        spec = next(spec for spec in build_specs("development") if spec["model"] == "MWVAR")
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            saved = save_model_score(
                {
                    "scores": numpy.arange(8, dtype=float),
                    "source_start": 0,
                    "source_end_exclusive": 8,
                    "alignment": "same_timestep",
                    "primitive": "sample_variance",
                    "calibration_mode": "none",
                    "evaluation_mode": "offline_noncausal",
                    "lookahead": 47,
                    "maximum_effective_lookahead": 95,
                    "normalization_scope": "none",
                },
                output_directory, dataset="GHL", series=1,
                model=spec["model"], target_use=spec["target_use"],
                tier=spec["tier"], ratio=spec["ratio"],
                seed=spec["seed"], config_id=spec["config_id"],
                common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="none",
                execution_identity=execution_identity(spec),
            )
            metadata_path = Path(saved["metadata_path"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["config_registry_sha256"] = "0" * 64
            metadata_path.write_text(
                json.dumps(metadata), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "registry SHA"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )

    def test_output_checker_requires_exact_tspulse_score_variant(self):
        spec = next(
            spec for spec in build_specs("development")
            if spec["model"] == "TSPulse"
        )
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            save_model_score(
                {
                    "scores": numpy.arange(8, dtype=float),
                    "source_start": 0,
                    "source_end_exclusive": 8,
                    "alignment": "same_timestep",
                    "primitive": "raw_fft_head_score",
                    "calibration_mode": "none",
                    "evaluation_mode": "offline_noncausal",
                    "lookahead": 32,
                    "maximum_effective_lookahead": 511,
                    "normalization_scope": "none",
                    "input_normalization": "inference_context_zscore",
                    "input_normalization_scope": "past_context",
                    "persistent_target_fit": False,
                },
                output_directory, dataset="GHL", series=1,
                model=spec["model"], target_use=spec["target_use"],
                tier=spec["tier"], ratio=spec["ratio"],
                seed=spec["seed"], config_id=spec["config_id"],
                common_recipe=spec["common_recipe"],
                common_recipe_id=spec["common_recipe_id"],
                normalization_scope="none", score_variant="fft",
                execution_identity=execution_identity(spec),
            )
            with self.assertRaisesRegex(ValueError, "score_variant"):
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                )
            self.assertEqual(
                check_registered_output(
                    output_directory, spec, dataset="GHL", series=1,
                    score_variant="fft",
                )["status"],
                "complete",
            )


if __name__ == "__main__":
    unittest.main()
