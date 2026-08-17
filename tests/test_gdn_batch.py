"""3b-3 배치·완전성·HAI Jaccard 계약을 임시 산출물로 검증한다."""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy
from unittest.mock import patch

from experiments.exp02_gdn_ghl import check_completeness as ghl_completeness
from experiments.exp02_gdn_ghl import run_controls as ghl_control_runner
from experiments.exp02_gdn_ghl.check_completeness import (
    find_missing_runs as find_missing_ghl_runs,
)
from experiments.exp02_gdn_ghl.run_batch import (
    build_specs as build_ghl_specs,
    run_batch as run_ghl_batch,
)
from experiments.exp02_gdn_ghl.check_controls import (
    build_notopk_specs,
    build_topk_sensitivity_specs,
    notopk_run_directory,
    sensitivity_run_directory,
)
from experiments.exp02_gdn_ghl.run_controls import run_controls
from experiments.exp03_gdn_hai_seed10.check_completeness import (
    adjacency_path,
    find_missing_runs as find_missing_hai_runs,
)
from experiments.exp03_gdn_hai_seed10.compute_jaccard_agreement import (
    calculate_jaccard,
    compute_pairwise_rows,
)
from experiments.exp03_gdn_hai_seed10 import compute_jaccard_agreement
from experiments.exp03_gdn_hai_seed10.run_batch import (
    build_specs as build_hai_specs,
    run_batch as run_hai_batch,
)
from src.common.naming import build_score_filename


VERIFIED_MTIME_NS = 1_700_000_000_000_000_000
VERIFIED_FILES = ({
    "name": "fake.csv", "size_bytes": 3, "sha256": "abc",
    "mtime_ns": VERIFIED_MTIME_NS,
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
    os.utime(input_path, ns=(VERIFIED_MTIME_NS, VERIFIED_MTIME_NS))
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

    def test_input_changed_during_load_stops_before_model(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            data_dir = experiment_dir / "data"
            model_calls = []

            def mutating_loader(*_arguments, **_keywords):
                inputs = fake_inputs()
                (data_dir / "fake.csv").write_bytes(b"changed")
                return inputs

            result = run_ghl_batch(
                experiment_dir=experiment_dir,
                data_dir=data_dir,
                specs=((1, 10, 1),),
                input_loader=mutating_loader,
                model_runner=lambda **_arguments: model_calls.append("model"),
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

            self.assertEqual(result, {"completed": 0, "skipped": 0, "failed": 1})
            self.assertEqual(model_calls, [])
            failure_text = (experiment_dir / "logs" / "failures.csv").read_text(encoding="utf-8")
            self.assertIn("검증 뒤 바뀌었다", failure_text)

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


class TestGhlBackTrimCompleteness(unittest.TestCase):
    def test_specs_and_paths_cover_control_without_duplicating_full_ratio(self):
        required_names = (
            "build_back_trim_specs",
            "back_trim_run_directory",
            "find_missing_back_trim_runs",
        )
        self.assertTrue(
            all(hasattr(ghl_completeness, name) for name in required_names),
            "GHL back trim 완전성 계약이 없다",
        )

        specs = ghl_completeness.build_back_trim_specs()
        self.assertEqual(len(specs), 25 * 3 * 3)
        self.assertEqual(len(set(specs)), len(specs))
        self.assertEqual(specs[0], (1, 5, 1))
        self.assertEqual(specs[-1], (25, 100, 3))

        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            self.assertEqual(
                ghl_completeness.back_trim_run_directory(experiment_dir, (1, 5, 1)),
                experiment_dir / "back_trim_runs" / "series_01" / "r005" / "s1",
            )
            self.assertEqual(
                ghl_completeness.back_trim_run_directory(experiment_dir, (1, 100, 1)),
                ghl_completeness.run_directory(experiment_dir, (1, 100, 1)),
            )
            self.assertEqual(
                len(ghl_completeness.find_missing_back_trim_runs(experiment_dir)),
                25 * 3 * 3,
            )

    def test_batch_separates_outputs_reuses_full_ratio_and_resumes(self):
        success_spec = (1, 5, 1)
        failure_spec = (1, 20, 2)
        identity_spec = (1, 100, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            data_dir = experiment_dir / "data"
            identity_config = {
                "naming": {"dataset": "GHL", "series": 1, "tier": "t2", "ratio": 100},
            }
            write_fake_run(
                str(ghl_completeness.run_directory(experiment_dir, identity_spec)),
                identity_config,
                seed=1,
            )
            write_fake_completion_marker(
                ghl_completeness.run_directory(experiment_dir, identity_spec),
            )
            loader_calls = []
            trim_directions_seen = []

            def input_loader(data_dir, series, ratio, config, trim_direction):
                loader_calls.append((data_dir, series, ratio, trim_direction))
                return fake_inputs()

            def model_runner(config, output_dir, seed, **_inputs):
                trim_directions_seen.append(config["trim_direction"])
                if seed == 2:
                    raise RuntimeError("의도한 back trim 실패")
                return write_fake_run(output_dir, config, seed)

            try:
                result = run_ghl_batch(
                    experiment_dir=experiment_dir,
                    data_dir=data_dir,
                    specs=(success_spec, failure_spec, identity_spec),
                    input_loader=input_loader,
                    model_runner=model_runner,
                    trim_direction="back",
                    input_verifier=fake_input_verifier,
                    context_verifier=fake_context_verifier,
                )
            except TypeError as error:
                self.fail(f"GHL back trim batch 계약이 없다: {error}")

            self.assertEqual(result, {"completed": 1, "skipped": 1, "failed": 1})
            self.assertEqual(loader_calls, [
                (data_dir, 1, 5, "back"),
                (data_dir, 1, 20, "back"),
            ])
            self.assertEqual(trim_directions_seen, ["back", "back"])
            self.assertTrue(ghl_completeness.is_back_trim_run_complete(
                experiment_dir, success_spec,
            ))
            self.assertTrue(ghl_completeness.is_back_trim_run_complete(
                experiment_dir, identity_spec,
            ))
            self.assertFalse(
                experiment_dir.joinpath(
                    "back_trim_runs", "series_01", "r100", "s1",
                ).exists()
            )
            failure_path = experiment_dir / "back_trim_logs" / "failures.csv"
            with failure_path.open(encoding="utf-8") as failure_file:
                failure_rows = list(csv.DictReader(failure_file))
            self.assertIn("의도한 back trim 실패", failure_rows[0]["error"])

            def should_not_run(*_arguments, **_keywords):
                self.fail("완료된 back trim 조합을 다시 실행했다")

            resumed = run_ghl_batch(
                experiment_dir=experiment_dir,
                data_dir=data_dir,
                specs=(success_spec, identity_spec),
                input_loader=should_not_run,
                model_runner=should_not_run,
                trim_direction="back",
                input_verifier=should_not_run,
                context_verifier=fake_context_verifier,
            )
            self.assertEqual(resumed, {"completed": 0, "skipped": 2, "failed": 0})


class TestGhlControlArms(unittest.TestCase):
    def test_specs_cover_all_series_and_reuse_main_k5_paths(self):
        self.assertEqual(len(build_notopk_specs()), 25 * 2 * 3)
        self.assertEqual(len(build_topk_sensitivity_specs()), 25 * 2 * 3 * 3)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            self.assertEqual(
                notopk_run_directory(experiment_dir, (1, 10, 1)),
                experiment_dir / "control_runs" / "notopk" / "series_01" / "r010" / "s1",
            )
            self.assertEqual(
                sensitivity_run_directory(experiment_dir, (1, 10, 5, 1)),
                ghl_completeness.run_directory(experiment_dir, (1, 10, 1)),
            )
            self.assertEqual(
                sensitivity_run_directory(experiment_dir, (1, 10, 2, 1)),
                experiment_dir / "control_runs" / "topk_k002" / "series_01" / "r010" / "s1",
            )

    def test_notopk_and_topk_sensitivity_have_distinct_model_contracts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            data_dir = experiment_dir / "data"
            seen = []

            def model_runner(config, output_dir, seed, **_inputs):
                seen.append((
                    config["naming"]["model"],
                    config["model_params"]["learn_graph"],
                    config["model_params"]["topk"],
                ))
                return write_fake_run(output_dir, config, seed)

            notopk_result = run_controls(
                "notopk",
                experiment_dir=experiment_dir,
                data_dir=data_dir,
                specs=((1, 10, 1),),
                input_loader=fake_inputs,
                model_runner=model_runner,
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )
            sensitivity_result = run_controls(
                "topk_sensitivity",
                experiment_dir=experiment_dir,
                data_dir=data_dir,
                specs=((1, 10, 2, 1), (1, 10, 10, 1)),
                input_loader=fake_inputs,
                model_runner=model_runner,
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

        self.assertEqual(notopk_result, {"completed": 1, "skipped": 0, "failed": 0})
        self.assertEqual(sensitivity_result, {"completed": 2, "skipped": 0, "failed": 0})
        self.assertEqual(seen, [
            ("GDN_NOTOPK", False, 5),
            ("GDN_K2", True, 2),
            ("GDN_K10", True, 10),
        ])

    def test_notopk_keeps_the_configured_topk_in_its_snapshot_contract(self):
        original_builder = ghl_control_runner.build_run_config

        def build_with_nondefault_topk(series, ratio):
            config = original_builder(series, ratio)
            config["model_params"]["topk"] = 7
            return config

        seen = []
        with tempfile.TemporaryDirectory() as temporary_dir, patch.object(
            ghl_control_runner, "build_run_config", side_effect=build_with_nondefault_topk,
        ):
            result = run_controls(
                "notopk",
                experiment_dir=temporary_dir,
                data_dir=Path(temporary_dir) / "data",
                specs=((1, 10, 1),),
                input_loader=fake_inputs,
                model_runner=lambda config, **_arguments: (
                    seen.append(config["model_params"]["topk"])
                    or write_fake_run(
                        _arguments["output_dir"], config, _arguments["seed"],
                    )
                ),
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

        self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 0})
        self.assertEqual(seen, [7])

    def test_completed_control_does_not_verify_unused_input(self):
        spec = (1, 10, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            config = {
                "naming": {
                    "dataset": "GHL", "series": 1, "model": "GDN_NOTOPK",
                    "tier": "t2", "ratio": 10,
                },
            }
            run_dir = notopk_run_directory(experiment_dir, spec)
            write_fake_run(str(run_dir), config, seed=1)
            write_fake_completion_marker(run_dir)

            def should_not_run(*_arguments, **_keywords):
                self.fail("완료된 control이 입력을 다시 검증했다")

            result = run_controls(
                "notopk",
                experiment_dir=experiment_dir,
                data_dir=experiment_dir / "data",
                specs=(spec,),
                input_loader=should_not_run,
                model_runner=should_not_run,
                input_verifier=should_not_run,
                context_verifier=fake_context_verifier,
            )

        self.assertEqual(result, {"completed": 0, "skipped": 1, "failed": 0})

    def test_sensitivity_never_replaces_a_missing_main_k5_run(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            calls = []

            def input_loader(*_arguments, **_keywords):
                calls.append("input")
                return fake_inputs()

            def model_runner(**arguments):
                calls.append("model")
                return write_fake_run(**arguments)

            result = run_controls(
                "topk_sensitivity",
                experiment_dir=experiment_dir,
                data_dir=experiment_dir / "data",
                specs=((1, 10, 5, 1),),
                input_loader=input_loader,
                model_runner=model_runner,
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

        self.assertEqual(result, {"completed": 0, "skipped": 0, "failed": 1})
        self.assertEqual(calls, [])

    def test_control_stops_if_input_changes_during_load(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            data_dir = experiment_dir / "data"
            model_calls = []

            def mutating_loader(*_arguments, **_keywords):
                inputs = fake_inputs()
                (data_dir / "fake.csv").write_bytes(b"changed")
                return inputs

            result = run_controls(
                "notopk",
                experiment_dir=experiment_dir,
                data_dir=data_dir,
                specs=((1, 10, 1),),
                input_loader=mutating_loader,
                model_runner=lambda **_arguments: model_calls.append("model"),
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

            self.assertEqual(result, {"completed": 0, "skipped": 0, "failed": 1})
            self.assertEqual(model_calls, [])


class TestHaiBatch(unittest.TestCase):
    def test_specs_cover_2_ratios_10_seeds(self):
        specs = build_hai_specs()
        self.assertEqual(len(specs), 2 * 10)
        self.assertEqual(len(set(specs)), len(specs))
        self.assertEqual(specs[0], (10, 1))
        self.assertEqual(specs[-1], (100, 10))

    def test_batch_saves_both_edge_sets_and_becomes_complete(self):
        spec = (10, 1)
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
                (experiment_dir / "runs" / "r010" / "s1" / "COMPLETE").read_text(
                    encoding="ascii",
                ),
                "complete\n",
            )
            without_self = numpy.load(adjacency_path(
                experiment_dir, ratio=10, seed=1, self_edges=False,
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
        spec = (10, 1)
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            run_dir = Path(temporary_dir) / "runs" / "r010" / "s1"
            config = {
                "naming": {"dataset": "HAI", "series": (1, 2), "tier": "t2", "ratio": 10},
            }
            result = write_fake_run(str(run_dir), config, seed=1)
            Path(result["best_checkpoint_path"]).write_bytes(b"")
            for self_edges in (True, False):
                path = adjacency_path(experiment_dir, ratio=10, seed=1, self_edges=self_edges)
                path.parent.mkdir(parents=True, exist_ok=True)
                numpy.save(path, numpy.empty((2, 0), dtype=int))
            write_fake_completion_marker(run_dir)

            self.assertEqual(find_missing_hai_runs(experiment_dir, (spec,)), [spec])

    def test_input_changed_during_load_stops_hai_before_model(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            data_dir = experiment_dir / "data"
            model_calls = []

            def mutating_loader(*_arguments, **_keywords):
                inputs = fake_inputs()
                (data_dir / "fake.csv").write_bytes(b"changed")
                return inputs

            result = run_hai_batch(
                experiment_dir=experiment_dir,
                data_dir=data_dir,
                specs=((10, 1),),
                input_loader=mutating_loader,
                model_runner=lambda **_arguments: model_calls.append("model"),
                input_verifier=fake_input_verifier,
                context_verifier=fake_context_verifier,
            )

            self.assertEqual(result, {"completed": 0, "skipped": 0, "failed": 1})
            self.assertEqual(model_calls, [])


class TestJaccardAgreement(unittest.TestCase):
    def test_hand_calculation_and_all_45_seed_pairs(self):
        self.assertAlmostEqual(
            calculate_jaccard({(0, 1), (1, 2)}, {(1, 2), (2, 3)}),
            1 / 3,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            for seed in range(1, 11):
                edges = {(0, 1), (1, 2)} if seed == 1 else {(1, 2), (2, 3)}
                path = adjacency_path(
                    experiment_dir, ratio=10, seed=seed, self_edges=False,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                numpy.save(path, numpy.asarray(sorted(edges), dtype=int).T)

            rows = compute_pairwise_rows(experiment_dir, ratio=10)

        self.assertEqual(len(rows), 45)
        self.assertEqual((rows[0]["seed_a"], rows[0]["seed_b"]), (1, 2))
        self.assertAlmostEqual(rows[0]["jaccard"], 1 / 3)
        self.assertEqual(rows[-1]["jaccard"], 1.0)

    def test_main_refuses_to_mix_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temporary_dir, patch.object(
            compute_jaccard_agreement,
            "verify_run_context",
            return_value=VERIFIED_GIT_HASHES,
        ), patch.object(
            sys, "argv", ["compute_jaccard_agreement.py", "--experiment-dir", temporary_dir],
        ):
            with self.assertRaisesRegex(RuntimeError, "20개"):
                compute_jaccard_agreement.main()

            self.assertFalse((Path(temporary_dir) / "analysis").exists())

    def test_main_writes_source_commit_identity_into_both_csv_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            experiment_dir = Path(temporary_dir)
            for seed in (1, 2):
                path = adjacency_path(
                    experiment_dir, ratio=10, seed=seed, self_edges=False,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                numpy.save(path, numpy.asarray([[0], [1]], dtype=int))

            with patch.object(
                compute_jaccard_agreement,
                "verify_run_context",
                return_value=VERIFIED_GIT_HASHES,
            ), patch.object(
                compute_jaccard_agreement,
                "build_specs",
                return_value=[(10, 1), (10, 2)],
            ), patch.object(
                compute_jaccard_agreement,
                "find_missing_runs",
                return_value=[],
            ), patch.object(
                sys,
                "argv",
                ["compute_jaccard_agreement.py", "--experiment-dir", temporary_dir],
            ):
                compute_jaccard_agreement.main()

            for filename in ("jaccard_pairs.csv", "jaccard_summary.csv"):
                with (experiment_dir / "analysis" / filename).open(
                    encoding="utf-8",
                ) as file:
                    row = next(csv.DictReader(file))
                self.assertEqual(row["tsad_commit"], VERIFIED_GIT_HASHES["tsad_project"])
                self.assertEqual(row["gragod_commit"], VERIFIED_GIT_HASHES["gragod_fork"])


if __name__ == "__main__":
    unittest.main()
