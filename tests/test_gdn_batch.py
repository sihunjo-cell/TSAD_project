"""3b-3 배치·완전성·HAI Jaccard 계약을 임시 산출물로 검증한다."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy

from experiments.exp02_gdn_ghl import check_completeness as ghl_completeness
from experiments.exp02_gdn_ghl.check_completeness import (
    find_missing_runs as find_missing_ghl_runs,
)
from experiments.exp02_gdn_ghl.run_batch import (
    build_specs as build_ghl_specs,
    run_batch as run_ghl_batch,
)
from experiments.exp03_gdn_hai_seed10.check_completeness import (
    adjacency_path,
    find_missing_runs as find_missing_hai_runs,
)
from experiments.exp03_gdn_hai_seed10.compute_jaccard_agreement import (
    calculate_jaccard,
    compute_pairwise_rows,
)
from experiments.exp03_gdn_hai_seed10.run_batch import (
    build_specs as build_hai_specs,
    run_batch as run_hai_batch,
)
from src.common.naming import build_score_filename


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
            "model": "GDN",
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
                    )).touch()
        metadata_name = build_score_filename(
            smoothing_kind="raw", norm_kind="trainnorm", channels=False, **arguments,
        ).replace(".npy", ".meta.json")
        (scores_dir / metadata_name).touch()

    checkpoint_path = output_dir / "training" / "gdn" / "version_0" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()
    (output_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (output_dir / "snapshots" / "config_snapshot.json").touch()
    (output_dir / "early_stopping_log.json").touch()
    return {"best_checkpoint_path": str(checkpoint_path)}


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
            )

            self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 1})
            self.assertEqual(input_metadata_seen[0], {
                "path": str((experiment_dir / "data").resolve()),
                "feature_names": ("sensor_00", "sensor_01"),
                "session_splits": ({"source": "fake.csv", "train_range": (0, 6)},),
            })
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
            )
            self.assertEqual(resumed, {"completed": 0, "skipped": 1, "failed": 0})


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
            )
            self.assertEqual(resumed, {"completed": 0, "skipped": 2, "failed": 0})


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
            )

            self.assertEqual(result, {"completed": 1, "skipped": 0, "failed": 0})
            self.assertEqual(input_metadata_seen, [{
                "path": str((experiment_dir / "data").resolve()),
                "feature_names": ("sensor_00", "sensor_01"),
                "session_splits": ({"source": "fake.csv", "train_range": (0, 6)},),
            }])
            self.assertEqual(find_missing_hai_runs(experiment_dir, (spec,)), [])
            without_self = numpy.load(adjacency_path(
                experiment_dir, ratio=10, seed=1, self_edges=False,
            ))
            self.assertEqual({tuple(edge) for edge in without_self.T}, {(1, 0)})


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


if __name__ == "__main__":
    unittest.main()
