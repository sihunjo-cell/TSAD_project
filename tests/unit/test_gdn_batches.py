"""GDN 배치·완전성 계약을 임시 산출물로 검증한다."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy

from tests.ghl_main import check_gdn_outputs as ghl_completeness
from tests.ghl_main.check_gdn_outputs import (
    find_missing_runs as find_missing_ghl_runs,
)
from tests.ghl_main.run_gdn import (
    build_specs as build_ghl_specs,
    run_batch as run_ghl_batch,
)
from tests.hai_extension.check_gdn_outputs import (
    adjacency_path,
    find_missing_runs as find_missing_hai_runs,
)
from tests.hai_extension.run_gdn import (
    build_specs as build_hai_specs,
    run_batch as run_hai_batch,
)
from src.common.naming import build_score_filename


VERIFIED_FILES = ({
    "name": "fake.csv", "size_bytes": 3, "sha256": "abc",
},)
VERIFIED_GIT_HASHES = {
    "tsad_project": "current-tsad-commit",
    "gragod_fork": "current-gragod-commit",
}


def fake_input_verifier(_dataset, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    input_path = data_dir / "fake.csv"
    input_path.write_bytes(b"abc")
    return VERIFIED_FILES


def fake_context_verifier(_project_dir, _fork_dir):
    return VERIFIED_GIT_HASHES


def write_fake_run(output_dir: str, config: dict, seed: int, **_inputs) -> dict:
    """모델을 돌리지 않고 완전성 검사에 필요한 파일만 만든다."""
    output_dir = Path(output_dir)
    naming = config["naming"]
    series_values = naming["series"]
    if isinstance(series_values, int):
        series_values = (series_values,)

    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for series in series_values:
        arguments = {
            "dataset": naming["dataset"],
            "series": series,
            "model": naming.get("model", "GDN"),
            "tier": naming["tier"],
            "ratio": naming["ratio"],
            "seed": seed,
        }
        for smoothing_kind in ("raw", "smoothed"):
            for norm_kind in ("trainnorm", "testnorm"):
                for channels in (False, True):
                    (scores_dir / build_score_filename(
                        smoothing_kind=smoothing_kind,
                        norm_kind=norm_kind,
                        channels=channels,
                        **arguments,
                    )).write_bytes(b"x")
        metadata_name = build_score_filename(
            smoothing_kind="raw", norm_kind="trainnorm", channels=False, **arguments,
        ).replace(".npy", ".meta.json")
        (scores_dir / metadata_name).write_text("{}", encoding="utf-8")

    checkpoint_path = output_dir / "training" / "gdn" / "version_0" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"x")
    (output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots" / "config_snapshot.json").write_text(
        json.dumps({
            "git_commit_hash": VERIFIED_GIT_HASHES["tsad_project"],
            "config": {"git_hashes": VERIFIED_GIT_HASHES},
        }),
        encoding="utf-8",
    )
    (output_dir / "early_stopping_log.json").write_text("{}", encoding="utf-8")
    (output_dir / "timing.json").write_text("{}", encoding="utf-8")
    return {
        "best_checkpoint_path": str(checkpoint_path),
        "timing_path": str(output_dir / "timing.json"),
    }


def write_fake_completion_marker(run_dir: Path) -> None:
    (Path(run_dir) / "COMPLETE").write_text("complete\n", encoding="ascii")


def fake_inputs(*_arguments, **_keywords) -> dict:
    placeholder = numpy.zeros((6, 2))
    return {
        "feature_names": ("sensor_00", "sensor_01"),
        "train_sessions": (placeholder,),
        "validation_sessions": (placeholder,),
        "test_sessions": (placeholder,),
        "session_splits": ({"source": "fake.csv", "train_range": (0, 6)},),
    }


class TestGhlBatch(unittest.TestCase):
    def test_specs_cover_25_series_5_ratios_3_seeds(self):
        specs = build_ghl_specs()
        self.assertEqual(len(specs), 25 * 5 * 3)
        self.assertEqual(len(set(specs)), len(specs))
        self.assertEqual(specs[0], (1, 5, 1))
        self.assertEqual(specs[-1], (25, 100, 3))

    def test_batch_loads_only_the_training_prefix(self):
        spec = (1, 5, 1)
        loader_calls = []

        def load_prefix(data_dir, series, ratio, config):
            loader_calls.append((data_dir, series, ratio, config["naming"]["ratio"]))
            return fake_inputs()

        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            data_dir = experiment_dir / "data"
            result = run_ghl_batch(
                experiment_dir=experiment_dir,
                data_dir=data_dir,
                specs=(spec,),
                input_loader=load_prefix,
                model_runner=write_fake_run,
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

        self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 0})
        self.assertEqual(loader_calls, [(data_dir, 1, 5, 5)])

    def test_empty_folder_missing_count_resume_and_failure_log(self):
        first_spec = (1, 5, 1)
        failing_spec = (1, 5, 2)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            input_metadata_seen = []
            verification_calls = []
            self.assertEqual(len(find_missing_ghl_runs(experiment_dir)), 25 * 5 * 3)

            def model_runner(config, output_dir, seed, input_metadata, **_inputs):
                input_metadata_seen.append(input_metadata)
                if seed == 2:
                    raise RuntimeError("의도한 실패")
                return write_fake_run(output_dir, config, seed)

            result = run_ghl_batch(
                experiment_dir=experiment_dir,
                data_dir=experiment_dir / "data",
                specs=(first_spec, failing_spec),
                input_loader=fake_inputs,
                model_runner=model_runner,
                input_verifier=lambda dataset, data_dir: (
                    verification_calls.append((dataset, data_dir))
                    or fake_input_verifier(dataset, data_dir)
                ),
                context_verifier=fake_context_verifier,
            )

            self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 1})
            self.assertEqual(input_metadata_seen[0], {
                "path": str((experiment_dir / "data").resolve()),
                "feature_names": ("sensor_00", "sensor_01"),
                "session_splits": ({"source": "fake.csv", "train_range": (0, 6)},),
                "files": VERIFIED_FILES,
            })
            self.assertEqual(verification_calls, [("GHL", experiment_dir / "data")])
            self.assertEqual(find_missing_ghl_runs(experiment_dir, (first_spec,)), [])
            with (experiment_dir / "logs" / "failures.csv").open(encoding="utf-8") as failure_file:
                failure_rows = list(csv.DictReader(failure_file))
            self.assertEqual(failure_rows[0]["series"], "1")
            self.assertEqual(failure_rows[0]["seed"], "2")
            self.assertIn("의도한 실패", failure_rows[0]["error"])

            def should_not_run(*_arguments, **_keywords):
                self.fail("완료된 조합을 다시 실행했다")

            resumed = run_ghl_batch(
                experiment_dir=experiment_dir,
                data_dir=experiment_dir / "data",
                specs=(first_spec,),
                input_loader=should_not_run,
                model_runner=should_not_run,
                input_verifier=should_not_run,
                context_verifier=fake_context_verifier,
            )
            self.assertEqual(resumed, {"completed": 0, "skipped": 1, "failed": 0})

    def test_zero_byte_required_artifact_is_incomplete(self):
        spec = (1, 10, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            run_dir = ghl_completeness.run_directory(experiment_dir, spec)
            config = {
                "naming": {"dataset": "GHL", "series": 1, "tier": "t2", "ratio": 10},
            }
            write_fake_run(str(run_dir), config, seed=1)
            write_fake_completion_marker(run_dir)
            (run_dir / "timing.json").write_bytes(b"")

            self.assertFalse(ghl_completeness.is_run_complete(experiment_dir, spec))

    def test_files_without_final_success_marker_are_incomplete(self):
        spec = (1, 10, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            run_dir = ghl_completeness.run_directory(experiment_dir, spec)
            config = {
                "naming": {"dataset": "GHL", "series": 1, "tier": "t2", "ratio": 10},
            }
            write_fake_run(str(run_dir), config, seed=1)

            self.assertFalse(ghl_completeness.is_run_complete(experiment_dir, spec))

    def test_completion_from_another_commit_is_not_current(self):
        spec = (1, 10, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            run_dir = ghl_completeness.run_directory(experiment_dir, spec)
            config = {
                "naming": {"dataset": "GHL", "series": 1, "tier": "t2", "ratio": 10},
            }
            write_fake_run(str(run_dir), config, seed=1)
            write_fake_completion_marker(run_dir)
            snapshot_path = run_dir / "snapshots" / "config_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            snapshot["git_commit_hash"] = "old-tsad-commit"
            snapshot["config"]["git_hashes"]["tsad_project"] = "old-tsad-commit"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

            self.assertFalse(ghl_completeness.is_run_complete(
                experiment_dir, spec, expected_git_hashes=VERIFIED_GIT_HASHES,
            ))

            model_calls = []

            def model_runner(config, output_dir, seed, **_inputs):
                model_calls.append(spec)
                return write_fake_run(output_dir, config, seed)

            result = run_ghl_batch(
                experiment_dir=experiment_dir,
                data_dir=experiment_dir / "data",
                specs=(spec,),
                input_loader=fake_inputs,
                model_runner=model_runner,
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

            self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 0})
            self.assertEqual(model_calls, [spec])

    def test_malformed_snapshot_identity_is_incomplete(self):
        spec = (1, 10, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            run_dir = ghl_completeness.run_directory(experiment_dir, spec)
            config = {
                "naming": {"dataset": "GHL", "series": 1, "tier": "t2", "ratio": 10},
            }
            write_fake_run(str(run_dir), config, seed=1)
            write_fake_completion_marker(run_dir)
            (run_dir / "snapshots" / "config_snapshot.json").write_text(
                json.dumps({"config": {"git_hashes": VERIFIED_GIT_HASHES}}),
                encoding="utf-8",
            )

            self.assertFalse(ghl_completeness.is_run_complete(
                experiment_dir, spec, expected_git_hashes=VERIFIED_GIT_HASHES,
            ))


class TestHaiBatch(unittest.TestCase):
    def test_specs_cover_2_temporal_conditions_10_seeds(self):
        specs = build_hai_specs()
        self.assertEqual(len(specs), 2 * 10)
        self.assertEqual(len(set(specs)), len(specs))
        self.assertEqual(specs[0], (1, 1))
        self.assertEqual(specs[-1], (2, 10))

    def test_batch_saves_both_edge_sets_and_becomes_complete(self):
        spec = (1, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            input_metadata_seen = []
            self.assertEqual(len(find_missing_hai_runs(experiment_dir)), 2 * 10)

            def model_runner(input_metadata, **arguments):
                input_metadata_seen.append(input_metadata)
                return write_fake_run(**arguments)

            result = run_hai_batch(
                experiment_dir=experiment_dir,
                data_dir=experiment_dir / "data",
                specs=(spec,),
                input_loader=fake_inputs,
                model_runner=model_runner,
                adjacency_extractor=lambda _path, _topk: (
                    {(0, 0), (1, 0)}, {(1, 0)},
                ),
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

            self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 0})
            self.assertEqual(input_metadata_seen, [{
                "path": str((experiment_dir / "data").resolve()),
                "feature_names": ("sensor_00", "sensor_01"),
                "session_splits": ({"source": "fake.csv", "train_range": (0, 6)},),
                "files": VERIFIED_FILES,
            }])
            self.assertEqual(find_missing_hai_runs(experiment_dir, (spec,)), [])
            self.assertEqual(
                (experiment_dir / "scores" / "tier2" / "gdn"
                 / "condition1" / "s1" / "COMPLETE").read_text(
                    encoding="ascii",
                ),
                "complete\n",
            )
            without_self = numpy.load(adjacency_path(
                experiment_dir, condition=1, seed=1, self_edges=False,
            ))
            self.assertEqual({tuple(edge) for edge in without_self.T}, {(1, 0)})

            def should_not_run(*_arguments, **_keywords):
                self.fail("완료된 HAI 조합이 입력을 다시 검증했다")

            resumed = run_hai_batch(
                experiment_dir=experiment_dir,
                data_dir=experiment_dir / "data",
                specs=(spec,),
                input_loader=should_not_run,
                model_runner=should_not_run,
                adjacency_extractor=should_not_run,
                input_verifier=should_not_run,
                context_verifier=fake_context_verifier,
            )
            self.assertEqual(resumed, {"completed": 0, "skipped": 1, "failed": 0})

    def test_zero_byte_checkpoint_is_incomplete(self):
        spec = (1, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            run_dir = Path(temporary_dir) / "runs" / "condition1" / "s1"
            config = {
                "naming": {"dataset": "HAI", "series": 1, "tier": "t2", "ratio": 100},
            }
            result = write_fake_run(str(run_dir), config, seed=1)
            Path(result["best_checkpoint_path"]).write_bytes(b"")
            for self_edges in (True, False):
                path = adjacency_path(experiment_dir, condition=1, seed=1, self_edges=self_edges)
                path.parent.mkdir(parents=True, exist_ok=True)
                numpy.save(path, numpy.empty((2, 0), dtype=int))
            write_fake_completion_marker(run_dir)

            self.assertEqual(find_missing_hai_runs(experiment_dir, (spec,)), [spec])

if __name__ == "__main__":
    unittest.main()
