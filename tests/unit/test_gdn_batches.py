"""Tier 2 배치 조합·재개·완료 판정을 임시 산출물로 검증한다."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy

from src.common.naming import build_score_filename
from tests.ghl_main import check_tier2_outputs as ghl_checker
from tests.ghl_main.run_tier2 import build_specs as build_ghl_specs, run_batch as run_ghl_batch
from tests.hai_extension import check_gdn_outputs as hai_checker
from tests.hai_extension.run_gdn import build_specs as build_hai_specs, run_batch as run_hai_batch


SOURCE_IDENTITY = {
    "project_commit": "a" * 40,
    "local_model_sha256": {"model.py": "b" * 64},
    "upstream_source_commits": {"source": "c" * 40},
}


def fake_context(_project):
    return SOURCE_IDENTITY


def fake_inputs(*_arguments, **_keywords):
    values = numpy.zeros((120, 19), dtype=numpy.float32)
    return {
        "feature_names": tuple(f"sensor_{index}" for index in range(19)),
        "train_sessions": (values,),
        "validation_sessions": (values,),
        "test_sessions": (values,),
        "session_splits": ({"source": "fake.csv"},),
    }


def fake_input_verifier(_dataset, data_dir):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return ({"name": "fake.csv", "size_bytes": 1, "sha256": "0" * 64},)


def write_fake_artifacts(output_dir, config, source_identity, **_arguments):
    output_dir = Path(output_dir)
    naming = config["naming"]
    series_values = naming["series"] if isinstance(naming["series"], tuple) else (naming["series"],)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for series in series_values:
        arguments = {
            "dataset": naming["dataset"], "series": series,
            "model": naming["model"], "tier": "t2",
            "ratio": naming["ratio"], "seed": config["seed"],
        }
        for smoothing in ("raw", "smoothed"):
            for norm in ("trainnorm", "testnorm"):
                for channels in (False, True):
                    (scores_dir / build_score_filename(
                        smoothing_kind=smoothing, norm_kind=norm,
                        channels=channels, **arguments,
                    )).write_bytes(b"x")
        metadata = build_score_filename(
            smoothing_kind="raw", norm_kind="trainnorm", channels=False,
            **arguments,
        ).replace(".npy", ".meta.json")
        (scores_dir / metadata).write_text("{}", encoding="utf-8")
    checkpoint = output_dir / "training" / naming["model"].lower().replace("-", "_") / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"x")
    (output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots" / "config_snapshot.json").write_text(json.dumps({
        "git_commit_hash": source_identity["project_commit"],
        "config": {"source_identity": source_identity},
    }), encoding="utf-8")
    (output_dir / "early_stopping_log.json").write_text("{}", encoding="utf-8")
    (output_dir / "timing.json").write_text("{}", encoding="utf-8")
    return {"checkpoint_path": str(checkpoint)}


class TestGhlBatch(unittest.TestCase):
    def test_specs_cover_four_models_25_series_7_ratios_3_seeds(self):
        specs = build_ghl_specs()
        self.assertEqual(len(specs), 4 * 25 * 7 * 3)
        self.assertEqual({spec[0] for spec in specs}, {"CI-AE", "LSTM-AD", "USAD", "GDN"})
        self.assertEqual(len(build_ghl_specs("GDN")), 25 * 7 * 3)

    def test_one_combination_completes_and_resumes(self):
        spec = ("GDN", 1, 5, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = run_ghl_batch(
                experiment_dir=temporary_dir, data_dir=Path(temporary_dir) / "data",
                specs=(spec,), input_loader=fake_inputs,
                artifact_writer=write_fake_artifacts,
                input_verifier=fake_input_verifier, context_verifier=fake_context,
                identity_verifier=lambda *_arguments: None,
                model_executor=lambda **_arguments: {},
            )
            self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 0})
            self.assertTrue(ghl_checker.is_run_complete(
                temporary_dir, spec, expected_identity=SOURCE_IDENTITY,
            ))
            resumed = run_ghl_batch(
                experiment_dir=temporary_dir, data_dir=Path(temporary_dir) / "data",
                specs=(spec,), input_loader=lambda *_args: self.fail("입력을 다시 읽었다"),
                artifact_writer=lambda **_kwargs: self.fail("산출물을 다시 썼다"),
                input_verifier=lambda *_args: self.fail("입력을 다시 검증했다"),
                context_verifier=fake_context,
                identity_verifier=lambda *_arguments: None,
                model_executor=lambda **_arguments: self.fail("모델을 다시 실행했다"),
            )
            self.assertEqual(resumed, {"completed": 0, "skipped": 1, "failed": 0})

    def test_failure_has_no_completion_marker(self):
        spec = ("USAD", 1, 5, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = run_ghl_batch(
                experiment_dir=temporary_dir, data_dir=Path(temporary_dir) / "data",
                specs=(spec,), input_loader=fake_inputs,
                artifact_writer=write_fake_artifacts,
                input_verifier=fake_input_verifier, context_verifier=fake_context,
                identity_verifier=lambda *_arguments: None,
                model_executor=lambda **_arguments: (_ for _ in ()).throw(RuntimeError("의도한 실패")),
            )
            self.assertEqual(result["failed"], 1)
            self.assertFalse((ghl_checker.run_directory(temporary_dir, spec) / "COMPLETE").exists())


class TestHaiBatch(unittest.TestCase):
    def test_specs_cover_7_ratios_10_seeds(self):
        self.assertEqual(len(build_hai_specs()), 70)

    def test_one_model_produces_two_test_series_and_two_adjacencies(self):
        spec = (5, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            def two_session_inputs(*_arguments, **_keywords):
                inputs = fake_inputs()
                inputs["test_sessions"] = (inputs["test_sessions"][0], inputs["test_sessions"][0])
                return inputs

            result = run_hai_batch(
                experiment_dir=temporary_dir, data_dir=Path(temporary_dir) / "data",
                specs=(spec,), input_loader=two_session_inputs,
                artifact_writer=write_fake_artifacts,
                adjacency_extractor=lambda *_arguments: ({(0, 0), (1, 0)}, {(1, 0)}),
                input_verifier=fake_input_verifier, context_verifier=fake_context,
                identity_verifier=lambda *_arguments: None,
                model_executor=lambda **_arguments: {},
            )
            self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 0})
            self.assertTrue(hai_checker.is_run_complete(
                temporary_dir, spec, expected_identity=SOURCE_IDENTITY,
            ))
            self.assertTrue(hai_checker.adjacency_path(temporary_dir, 5, 1, True).is_file())
            self.assertTrue(hai_checker.adjacency_path(temporary_dir, 5, 1, False).is_file())


if __name__ == "__main__":
    unittest.main()
