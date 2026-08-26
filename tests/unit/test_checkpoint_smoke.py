"""실제 checkpoint smoke runner의 label-free 감사 계약을 검증한다."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy

from tests.checks.run_checkpoint_smoke import (
    CheckpointProbeError,
    SYNTHETIC_SEED,
    TSPULSE_SMOKE_LENGTH,
    TIME_RCD_SMOKE_LENGTH,
    _VerifiedFileRecorder,
    forbid_project_scaler_fit,
    forbid_time_rcd_window_dataset,
    make_synthetic_wave,
    require_checkpoint_smoke_success,
    run_checkpoint_smoke,
    run_dev18_checkpoint_smoke,
    run_time_rcd_checkpoint_smoke,
    run_tspulse_checkpoint_smoke,
    validate_model_report_identity,
    validate_score_pair,
)
from src.models.tier3.time_rcd import (
    TIME_RCD_CHECKPOINT_SHA256,
    TIME_RCD_CONFIG_SHA256,
)
from src.models.tier3.tspulse import (
    TSPULSE_CHECKPOINT_SHA256,
    TSPULSE_CONFIG_SHA256,
)


def _environment():
    return {
        "python": "3.11.14",
        "packages": {"torch": "2.10.0"},
        "torch": {
            "version": "2.10.0+cpu",
            "build_suffix": "cpu",
            "cuda_available": False,
            "cuda_version": None,
            "cudnn_version": None,
            "deterministic_algorithms": True,
            "cudnn_deterministic": False,
            "cudnn_benchmark": False,
        },
    }


def _files(model):
    if model == "TimeRCD":
        checkpoint_sha = TIME_RCD_CHECKPOINT_SHA256
        config_sha = TIME_RCD_CONFIG_SHA256
    else:
        checkpoint_sha = TSPULSE_CHECKPOINT_SHA256
        config_sha = TSPULSE_CONFIG_SHA256
    return {
        "checkpoint": {
            "sha256": checkpoint_sha,
            "bytes": 100,
            "path": "C:/cache/checkpoint",
        },
        "config": {
            "sha256": config_sha,
            "bytes": 20,
            "path": "C:/cache/config.json",
        },
    }


def _score_pairs(heads, length):
    base = numpy.linspace(0.0, 1.0, length, dtype=numpy.float64)
    return {head: (base + index, base + index) for index, head in enumerate(heads)}


class _FakeStandardScaler:
    def fit(self, values):
        return values


class _FakeMinMaxScaler:
    def fit(self, values):
        return values


class TestCheckpointSmoke(unittest.TestCase):
    def test_dev18_quick_smoke_uses_the_sealed_two_channel_prefix_and_budget_configs(self):
        observed = []

        def load_input(*, spec, data_root, series):
            observed.append(("input", spec["model"], str(data_root), series))
            return {
                "normal_training": numpy.arange(4000, dtype=numpy.float32).reshape(2000, 2),
                "test_sessions": (numpy.zeros((5, 2), dtype=numpy.float32),),
                "series": "03",
                "input_identity": {
                    "name": "023_MITDB_id_5_Medical_tr_25000_1st_36913.csv",
                    "sha256": "a" * 64,
                    "input_manifest_sha256": spec["input_manifest_sha256"],
                },
            }

        def time_probe(*, channel_count, session, context_length):
            observed.append(("TimeRCD", channel_count, session.shape, context_length))
            return {
                "score_pairs": _score_pairs(("score",), len(session)),
                "files": _files("TimeRCD"),
                "forbidden_calls": {
                    "standard_scaler_fit": False,
                    "minmax_scaler_fit": False,
                    "time_rcd_window_dataset": False,
                },
            }

        def tspulse_probe(*, channel_count, session, context_length, aggregation_window):
            observed.append((
                "TSPulse", channel_count, session.shape, context_length,
                aggregation_window,
            ))
            return {
                "score_pairs": _score_pairs(
                    ("time", "fft", "pred", "raw_max"), len(session),
                ),
                "files": _files("TSPulse"),
                "forbidden_calls": {
                    "standard_scaler_fit": False,
                    "minmax_scaler_fit": False,
                },
                "loader": {
                    "from_pretrained_source": "verified_local_snapshot",
                    "snapshot_directory": "C:/cache/snapshot",
                },
            }

        def set_seed(seed):
            observed.append(("seed", seed))
            return {"seed": seed}

        def collect_environment():
            observed.append(("environment",))
            return _environment()

        with tempfile.TemporaryDirectory() as directory:
            report = run_dev18_checkpoint_smoke(
                data_root="C:/sealed-data",
                output_root=directory,
                input_loader=load_input,
                probes={"TimeRCD": time_probe, "TSPulse": tspulse_probe},
                environment_collector=collect_environment,
                seed_setter=set_seed,
            )
            saved = {
                model: json.loads((
                    Path(directory)
                    / ("time_rcd" if model == "TimeRCD" else "tspulse")
                    / "dev18_checkpoint_smoke.json"
                ).read_text(encoding="utf-8"))
                for model in ("TimeRCD", "TSPulse")
            }

        self.assertEqual(report["status"], "passed")
        self.assertEqual(observed[0][0], "input")
        self.assertEqual([event[0] for event in observed[1:3]], ["seed", "environment"])
        self.assertIn(("TimeRCD", 2, (1536, 2), 5000), observed)
        self.assertIn(("TSPulse", 2, (1536, 2), 512, 64), observed)
        self.assertEqual(saved["TimeRCD"]["config_id"], "c1c5aeea6f7d3")
        self.assertEqual(saved["TSPulse"]["config_id"], "c12c5e6196ea5")
        for model_report in saved.values():
            self.assertEqual(model_report["input"]["kind"], "dev18_normal_prefix")
            self.assertEqual(model_report["input"]["slice"], [0, 1536])
            self.assertEqual(model_report["input"]["shape"], [1536, 2])
            self.assertFalse(model_report["input"]["uses_labels"])
            self.assertFalse(model_report["input"]["uses_test"])
            self.assertNotIn("values", json.dumps(model_report))

    def test_verified_file_recorder_preserves_snapshot_paths_without_resolving_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory).absolute() / "snapshots" / "sealed-revision"
            snapshot.mkdir(parents=True)
            paths = {
                "model.safetensors": snapshot / "model.safetensors",
                "config.json": snapshot / "config.json",
            }
            for path in paths.values():
                path.write_bytes(path.name.encode())

            recorder = _VerifiedFileRecorder(
                lambda *, repo_id, filename, revision: str(paths[filename]),
                lambda path, *, expected_sha256: expected_sha256,
            )
            with patch.object(Path, "resolve", side_effect=AssertionError("must not resolve")):
                for filename, expected_sha256 in (
                    ("model.safetensors", "checkpoint-sha"),
                    ("config.json", "config-sha"),
                ):
                    downloaded = recorder.download(
                        repo_id="sealed/repository",
                        filename=filename,
                        revision="sealed-revision",
                    )
                    recorder.verify(downloaded, expected_sha256=expected_sha256)

            identities = recorder.identities(
                checkpoint_file="model.safetensors", config_file="config.json",
            )
            self.assertEqual(
                Path(identities["checkpoint"]["path"]).parent,
                snapshot,
            )
            self.assertEqual(
                Path(identities["config"]["path"]).parent,
                snapshot,
            )

    def test_public_runner_api_has_no_label_or_normal_training_input(self):
        forbidden = {"label", "labels", "normal_training", "training_data"}
        for function in (
            run_checkpoint_smoke,
            run_time_rcd_checkpoint_smoke,
            run_tspulse_checkpoint_smoke,
        ):
            self.assertTrue(
                forbidden.isdisjoint(inspect.signature(function).parameters),
                function.__name__,
            )

    def test_score_pair_rejects_invalid_or_nondeterministic_outputs(self):
        valid = numpy.array([0.0, 1.0, 2.0])
        summary = validate_score_pair(valid, valid.copy(), expected_length=3)
        self.assertEqual(summary["shape"], [3])
        self.assertEqual(summary["peak_to_peak"], 2.0)
        self.assertTrue(summary["finite"])
        self.assertTrue(summary["cpu_deterministic"])

        invalid_pairs = (
            (numpy.ones(3), numpy.ones(3), "constant"),
            (numpy.array([0.0, 5e-10, 0.0]), numpy.zeros(3), "constant"),
            (numpy.array([0.0, numpy.nan, 2.0]), valid, "finite"),
            (valid[:2], valid[:2], "length"),
            (valid, valid + 0.1, "deterministic"),
        )
        for first, second, message in invalid_pairs:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_score_pair(first, second, expected_length=3)

    def test_forbidden_preprocessing_guards_raise_on_calls(self):
        inference_module = SimpleNamespace(_WindowDataset=lambda: None)
        with forbid_project_scaler_fit(
            standard_scaler_class=_FakeStandardScaler,
            minmax_scaler_class=_FakeMinMaxScaler,
        ):
            with self.assertRaisesRegex(RuntimeError, "StandardScaler.fit"):
                _FakeStandardScaler().fit(numpy.ones((2, 1)))
            with self.assertRaisesRegex(RuntimeError, "MinMaxScaler.fit"):
                _FakeMinMaxScaler().fit(numpy.ones((2, 1)))

        with forbid_time_rcd_window_dataset(inference_module=inference_module):
            with self.assertRaisesRegex(RuntimeError, "_WindowDataset"):
                inference_module._WindowDataset()

    def test_time_rcd_runs_both_channel_counts_after_one_fails(self):
        observed = []

        def probe(*, channel_count, session):
            observed.append((channel_count, session.shape, numpy.isfinite(session).all()))
            if channel_count == 19:
                raise RuntimeError("synthetic 19-channel failure")
            return {
                "score_pairs": _score_pairs(("score",), len(session)),
                "files": _files("TimeRCD"),
                "forbidden_calls": {
                    "standard_scaler_fit": False,
                    "minmax_scaler_fit": False,
                    "time_rcd_window_dataset": False,
                },
            }

        report = run_time_rcd_checkpoint_smoke(
            environment=_environment(), seed_state={"seed": 25}, probe=probe,
        )

        self.assertEqual(
            observed,
            [(19, (TIME_RCD_SMOKE_LENGTH, 19), True),
             (86, (TIME_RCD_SMOKE_LENGTH, 86), True)],
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["channels"]["19"]["status"], "failed")
        self.assertIn("synthetic 19-channel failure", report["channels"]["19"]["error"]["message"])
        self.assertEqual(report["channels"]["86"]["status"], "passed")

    def test_verified_file_identity_survives_a_forward_failure(self):
        def probe(*, channel_count, session):
            del channel_count, session
            raise CheckpointProbeError(
                RuntimeError("forward failed after verification"),
                files=_files("TimeRCD"),
            )

        report = run_time_rcd_checkpoint_smoke(
            environment=_environment(), seed_state={"seed": 25}, probe=probe,
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["checkpoint"]["sha256"], TIME_RCD_CHECKPOINT_SHA256)
        self.assertEqual(report["checkpoint"]["bytes"], 100)
        self.assertEqual(report["config"]["sha256"], TIME_RCD_CONFIG_SHA256)
        self.assertIn(
            "forward failed after verification",
            report["channels"]["19"]["error"]["message"],
        )

    def test_verified_file_identity_survives_score_validation_failure(self):
        def probe(*, channel_count, session):
            del channel_count
            constant = numpy.ones(len(session), dtype=numpy.float64)
            return {
                "score_pairs": {"score": (constant, constant.copy())},
                "files": _files("TimeRCD"),
                "forbidden_calls": {
                    "standard_scaler_fit": False,
                    "minmax_scaler_fit": False,
                    "time_rcd_window_dataset": False,
                },
            }

        report = run_time_rcd_checkpoint_smoke(
            environment=_environment(), seed_state={"seed": 25}, probe=probe,
        )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["checkpoint"]["sha256"], TIME_RCD_CHECKPOINT_SHA256)
        self.assertEqual(report["checkpoint"]["bytes"], 100)
        self.assertEqual(report["config"]["sha256"], TIME_RCD_CONFIG_SHA256)
        self.assertRegex(report["channels"]["19"]["error"]["message"], "constant")

    def test_tspulse_checks_every_raw_head_without_storing_arrays(self):
        observed = []

        def probe(*, channel_count, session):
            observed.append((channel_count, session.shape))
            return {
                "score_pairs": _score_pairs(
                    ("time", "fft", "pred", "raw_max"), len(session),
                ),
                "files": _files("TSPulse"),
                "forbidden_calls": {
                    "standard_scaler_fit": False,
                    "minmax_scaler_fit": False,
                },
                "loader": {
                    "from_pretrained_source": "verified_local_snapshot",
                    "snapshot_directory": "C:/cache/snapshot",
                },
            }

        report = run_tspulse_checkpoint_smoke(
            environment=_environment(), seed_state={"seed": 25}, probe=probe,
        )

        self.assertEqual(
            observed,
            [(19, (TSPULSE_SMOKE_LENGTH, 19)),
             (86, (TSPULSE_SMOKE_LENGTH, 86))],
        )
        self.assertEqual(report["status"], "passed")
        for channel in ("19", "86"):
            self.assertEqual(
                set(report["channels"][channel]["scores"]),
                {"time", "fft", "pred", "raw_max"},
            )
            for summary in report["channels"][channel]["scores"].values():
                self.assertEqual(summary["shape"], [TSPULSE_SMOKE_LENGTH])
                self.assertNotIn("values", summary)

    def test_missing_source_checkpoint_config_or_environment_identity_fails(self):
        report = run_tspulse_checkpoint_smoke(
            environment=_environment(),
            seed_state={"seed": 25},
            probe=lambda *, channel_count, session: {
                "score_pairs": _score_pairs(
                    ("time", "fft", "pred", "raw_max"), len(session),
                ),
                "files": _files("TSPulse"),
                "forbidden_calls": {
                    "standard_scaler_fit": False,
                    "minmax_scaler_fit": False,
                },
                "loader": {
                    "from_pretrained_source": "verified_local_snapshot",
                    "snapshot_directory": "C:/cache/snapshot",
                },
            },
        )
        validate_model_report_identity(report)

        for missing in ("source", "checkpoint", "config", "environment"):
            broken = dict(report)
            broken.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, missing):
                    validate_model_report_identity(broken)

    def test_orchestrator_writes_both_reports_then_exposes_overall_failure(self):
        calls = []

        def failed_runner(*, environment, seed_state):
            calls.append(("TimeRCD", environment, seed_state))
            raise RuntimeError("checkpoint load failed")

        def passed_runner(*, environment, seed_state):
            calls.append(("TSPulse", environment, seed_state))
            return {"model": "TSPulse", "status": "passed"}

        with tempfile.TemporaryDirectory() as directory:
            report = run_checkpoint_smoke(
                output_root=directory,
                model_runners={
                    "TimeRCD": failed_runner,
                    "TSPulse": passed_runner,
                },
                environment_collector=_environment,
                seed_setter=lambda seed: {"seed": seed},
            )
            time_report = json.loads(
                (Path(directory) / "time_rcd" / "synthetic_checkpoint_smoke.json").read_text(
                    encoding="utf-8",
                ),
            )
            tspulse_report = json.loads(
                (Path(directory) / "tspulse" / "synthetic_checkpoint_smoke.json").read_text(
                    encoding="utf-8",
                ),
            )

        self.assertEqual([call[0] for call in calls], ["TimeRCD", "TSPulse"])
        self.assertEqual(time_report["status"], "failed")
        self.assertEqual(time_report["error"]["message"], "checkpoint load failed")
        self.assertFalse(time_report["input"]["uses_labels"])
        self.assertFalse(time_report["input"]["uses_normal_training"])
        self.assertTrue(time_report["source"]["commit"])
        self.assertEqual(
            time_report["checkpoint"]["expected_sha256"],
            TIME_RCD_CHECKPOINT_SHA256,
        )
        self.assertEqual(time_report["config"]["expected_sha256"], TIME_RCD_CONFIG_SHA256)
        self.assertEqual(tspulse_report["status"], "passed")
        with self.assertRaisesRegex(RuntimeError, "TimeRCD"):
            require_checkpoint_smoke_success(report)

    def test_environment_failure_writes_complete_base_reports_without_running_models(self):
        def fail_environment():
            raise RuntimeError("environment unavailable")

        def must_not_run(**_arguments):
            self.fail("model runner must not run after environment failure")

        with tempfile.TemporaryDirectory() as directory:
            report = run_checkpoint_smoke(
                output_root=directory,
                model_runners={"TimeRCD": must_not_run, "TSPulse": must_not_run},
                environment_collector=fail_environment,
                seed_setter=lambda seed: {"seed": seed},
            )
            reports = {
                model: json.loads(
                    (
                        Path(directory)
                        / ("time_rcd" if model == "TimeRCD" else "tspulse")
                        / "synthetic_checkpoint_smoke.json"
                    ).read_text(encoding="utf-8"),
                )
                for model in ("TimeRCD", "TSPulse")
            }

        self.assertEqual(report["status"], "failed")
        for model, model_report in reports.items():
            self.assertEqual(model_report["status"], "failed", model)
            self.assertEqual(model_report["stage"], "environment")
            self.assertEqual(model_report["error"]["message"], "environment unavailable")
            self.assertFalse(model_report["input"]["uses_labels"])
            self.assertFalse(model_report["input"]["uses_normal_training"])
            self.assertTrue(model_report["source"]["commit"])
            self.assertIn("expected_sha256", model_report["checkpoint"])
            self.assertIn("expected_sha256", model_report["config"])
            self.assertEqual(model_report["seed_state"], {"seed": SYNTHETIC_SEED})
            self.assertEqual(model_report["environment"]["status"], "unavailable")

    def test_synthetic_wave_is_repeatable_finite_and_not_constant(self):
        first = make_synthetic_wave(length=160, channel_count=19)
        second = make_synthetic_wave(length=160, channel_count=19)
        numpy.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (160, 19))
        self.assertTrue(numpy.isfinite(first).all())
        self.assertGreater(float(numpy.ptp(first)), 0.0)


if __name__ == "__main__":
    unittest.main()
