"""GDN 단일 러너의 학습 경계와 산출 계약 테스트."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy
import torch
import yaml

from src.models.tier2.gdn import run_gdn_single as runner
from src.common import verify_run_context as context


def make_config(fork_path: str) -> dict:
    return {
        "fork_path": fork_path,
        "model_params": {
            "window_size": 5,
            "embed_dim": 64,
            "out_layer_num": 1,
            "out_layer_inter_dim": 128,
            "heads": 1,
            "dropout": 0.2,
            "negative_slope": 0.2,
            "learn_graph": True,
            "topk": 22,
        },
        "train_params": {
            "batch_size": 4,
            "n_epochs": 2,
            "init_lr": 0.001,
            "weight_decay": 0.0,
            "eps": 1e-8,
            "betas": [0.9, 0.99],
            "early_stop_patience": 10,
            "early_stop_delta": 0.0,
            "val_size": 0.2,
            "min_train_length": 6,
            "shuffle": True,
            "log_every_n_steps": 1,
        },
        "fork_contract": {
            "loss": "mse",
            "forecast_horizon": 1,
            "n_workers": 0,
            "validation_shuffle": False,
            "optimizer": "adam",
            "scheduler": {
                "name": "reduce_lr_on_plateau",
                "factor": 0.5,
                "patience": 8,
                "monitor": "Loss/val",
            },
            "gradient_clip_val": 1.0,
        },
        "naming": {"dataset": "HAI", "series": 1, "tier": "t2", "ratio": 10},
    }


def make_dependencies(state: dict):
    train_errors = torch.tensor([
        [0.0, 0.0], [2.0, 4.0], [4.0, 8.0],
        [6.0, 12.0], [8.0, 16.0], [10.0, 20.0], [12.0, 24.0],
    ])
    test_errors = torch.tensor([
        [5.0, 10.0], [10.0, 20.0], [0.0, 0.0], [15.0, 30.0], [20.0, 40.0],
    ])

    class SlidingWindowDataset:
        def __init__(self, data, window_size, edge_index, labels, drop=False):
            self.data = data
            self.window_size = window_size

        def __len__(self):
            return len(self.data) - self.window_size

    class ConcatDataset:
        def __init__(self, datasets):
            self.datasets = tuple(datasets)

        def __len__(self):
            return sum(len(dataset) for dataset in self.datasets)

    class DataLoader:
        def __init__(self, dataset, **arguments):
            self.dataset = dataset
            self.arguments = arguments

    def get_data_loader(**arguments):
        state.setdefault("loader_calls", []).append(arguments)
        return DataLoader(SlidingWindowDataset(
            data=arguments["X"],
            window_size=arguments["window_size"],
            edge_index=arguments["edge_index"],
            labels=arguments["y"],
        ))

    def model_class(**model_params):
        state["model_params"] = model_params
        return SimpleNamespace(to=lambda device: state.setdefault("model_device", device) or device)

    class BestModule:
        def eval(self):
            state["best_module_evaluated"] = True

        def calculate_anomaly_score(self, predict_output, X_true):
            state.setdefault("score_input_lengths", []).append(X_true.shape[0])
            if X_true.shape[0] == 6:
                return train_errors
            if X_true.shape[0] == 4:
                return test_errors
            values = torch.arange(X_true.shape[0], dtype=torch.float32).reshape(-1, 1)
            return torch.cat((values, values * 2), dim=1)

    class ModuleClass:
        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, map_location):
            state["checkpoint_load_count"] = state.get("checkpoint_load_count", 0) + 1
            state["checkpoint_load"] = (checkpoint_path, map_location)
            return BestModule()

    class TrainerPL:
        def __init__(self, **arguments):
            state["trainer_initialization_count"] = state.get("trainer_initialization_count", 0) + 1
            state["trainer_arguments"] = arguments

        def fit(self, train_loader, val_loader, args_summary):
            state["fit_count"] = state.get("fit_count", 0) + 1
            if "expected_snapshot_path" in state:
                state["snapshot_exists_when_fit_starts"] = Path(
                    state["expected_snapshot_path"]
                ).is_file()
            state["fit"] = (train_loader, val_loader, args_summary)

    class PredictionTrainer:
        def __init__(self, **arguments):
            state.setdefault("prediction_trainer_arguments", []).append(arguments)

        def predict(self, lightning_module, loader):
            dataset = loader.dataset
            truth = dataset.data[dataset.window_size:]
            state.setdefault("score_input_lengths", []).append(truth.shape[0])
            if truth.shape[0] == len(train_errors):
                errors = train_errors
            elif truth.shape[0] == len(test_errors):
                errors = test_errors
            else:
                values = torch.arange(truth.shape[0], dtype=torch.float32).reshape(-1, 1)
                errors = torch.cat((values, values * 2), dim=1)
            return [truth + errors]

    class Logger:
        def __init__(self, save_dir, name, default_hp_metric):
            self.log_dir = str(Path(save_dir) / name / "version_0")
            Path(self.log_dir).mkdir(parents=True)

    checkpoint = SimpleNamespace(best_model_path="fake-best.ckpt")
    early_stop = SimpleNamespace(stopped_epoch=1, best_score=0.125, wait_count=0)

    def get_training_callbacks(**arguments):
        state["callback_arguments"] = arguments
        return {"early_stop": early_stop, "checkpoint": checkpoint, "lr_monitor": object()}

    return SimpleNamespace(
        torch=torch,
        MSELoss=torch.nn.MSELoss,
        TensorBoardLogger=Logger,
        get_data_loader=get_data_loader,
        SlidingWindowDataset=SlidingWindowDataset,
        ConcatDataset=ConcatDataset,
        DataLoader=DataLoader,
        build_fully_connected_edge_index=lambda train_tensor, device: "edge-index",
        CleanMethods=SimpleNamespace(NONE="none"),
        Models=SimpleNamespace(GDN="gdn"),
        get_model_and_module=lambda model: (model_class, ModuleClass),
        set_seeds=lambda seed: state.setdefault("seed", seed),
        get_training_callbacks=get_training_callbacks,
        TrainerPL=TrainerPL,
        set_device=lambda: "cpu",
        PredictionTrainer=PredictionTrainer,
    )


def session_datasets(loader):
    dataset = loader.dataset
    return dataset.datasets if hasattr(dataset, "datasets") else (dataset,)


class TestReadGitHash(unittest.TestCase):
    def test_reads_explicit_external_repository_without_global_safe_directory(self):
        repository_dir = Path("external-owned-repository").resolve()
        expected_command = [
            "git", "-c", f"safe.directory={repository_dir}", "rev-parse", "HEAD",
        ]

        def emulate_git(command, **arguments):
            if command != expected_command:
                raise context.subprocess.CalledProcessError(128, command)
            self.assertEqual(arguments["cwd"], repository_dir)
            return SimpleNamespace(stdout="a" * 40 + "\n")

        with patch.object(context.subprocess, "run", side_effect=emulate_git):
            self.assertEqual(context.read_git_hash(repository_dir), "a" * 40)


class TestComputeAbsoluteErrors(unittest.TestCase):
    def test_rejects_nonfinite_model_output(self):
        class PredictionTrainer:
            def __init__(self, **_arguments):
                pass

            def predict(self, _lightning_module, _loader):
                return [torch.tensor([[float("nan"), 0.0]])]

        dependencies = SimpleNamespace(
            torch=torch,
            CleanMethods=SimpleNamespace(NONE="none"),
            get_data_loader=lambda **_arguments: object(),
            PredictionTrainer=PredictionTrainer,
        )
        with self.assertRaisesRegex(ValueError, "유한"):
            runner.compute_absolute_errors(
                lightning_module=object(),
                series_tensor=torch.zeros((6, 2)),
                edge_index=object(),
                window_size=5,
                batch_size=4,
                device="cpu",
                dependencies=dependencies,
            )


class TestRunGdnSingle(unittest.TestCase):
    def run_once(self, output_dir):
        state = {}
        state["expected_snapshot_path"] = str(
            Path(output_dir) / "snapshots" / "config_snapshot.json"
        )
        dependencies = make_dependencies(state)
        config = make_config(str(Path(output_dir) / "gragod-fork"))
        config["naming"]["dataset"] = "GHL"
        config["model_params"]["topk"] = 5
        config["train_params"]["val_size"] = 0.5
        train_array = numpy.arange(24, dtype=float).reshape(12, 2)
        test_array = numpy.arange(20, dtype=float).reshape(10, 2)
        input_metadata = {
            "path": "synthetic://unit-test",
            "feature_names": ("sensor_00", "sensor_01"),
            "files": ({
                "name": "synthetic.csv",
                "size_bytes": 1,
                "sha256": "a" * 64,
            },),
            "session_splits": ({
                "source": "synthetic",
                "train_range": (0, 6),
                "validation_range": (6, 12),
                "test_range": (0, 10),
            },),
        }

        with (
            patch.object(runner, "load_gdn_dependencies", return_value=dependencies, create=True) as dependency_loader,
            patch.object(runner, "verify_run_context", return_value={
                "tsad_project": "project-hash", "gragod_fork": "fork-hash",
            }),
            patch.object(runner, "verify_git_hashes_unchanged"),
            patch.object(runner, "verify_runtime_versions", return_value={"python": "fixed"}),
            patch.object(runner, "validate_pipeline_contract"),
        ):
            result = runner.run_gdn_single(
                config, train_array, test_array, output_dir, seed=7,
                input_metadata=input_metadata,
            )

        dependency_loader.assert_called_once_with(config["fork_path"])
        with open(result["timing_path"], encoding="utf-8") as timing_file:
            state["timing"] = json.load(timing_file)
        return result, state, config, train_array

    def test_training_contract_splits_tail_and_injects_model_arguments(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result, state, config, train_array = self.run_once(output_dir)

        self.assertEqual(state["seed"], 7)
        self.assertTrue(state["snapshot_exists_when_fit_starts"])
        numpy.testing.assert_array_equal(session_datasets(state["fit"][0])[0].data.numpy(), train_array[:6])
        numpy.testing.assert_array_equal(session_datasets(state["fit"][1])[0].data.numpy(), train_array[6:])
        self.assertEqual(state["fit"][2], {"seed": 7})
        self.assertTrue(state["fit"][0].arguments["shuffle"])
        self.assertFalse(state["fit"][1].arguments["shuffle"])
        self.assertEqual(state["model_params"]["edge_index"], ["edge-index"])
        self.assertEqual(state["model_params"]["n_features"], 2)
        self.assertEqual(state["model_params"]["out_dim"], 2)
        self.assertEqual(state["model_params"]["topk"], 5)
        self.assertNotIn("edge_index", config["model_params"])
        self.assertEqual(result["best_checkpoint_path"], "fake-best.ckpt")
        self.assertEqual(set(state["timing"]), {
            "accelerator",
            "training_seconds",
            "train_reference_inference_seconds",
            "test_inference_seconds",
        })
        self.assertEqual(state["timing"]["accelerator"], "cpu")
        self.assertTrue(all(
            state["timing"][field] >= 0
            for field in (
                "training_seconds",
                "train_reference_inference_seconds",
                "test_inference_seconds",
            )
        ))

    def test_output_contract_uses_train_errors_and_records_alignment_snapshot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result, state, config, _ = self.run_once(output_dir)

            trainnorm_channels = next(
                path for path in result["score_paths"]
                if "raw__trainnorm__channels.npy" in path
            )
            expected = numpy.array([
                [-1.0 / 6.01, -2.0 / 12.01],
                [4.0 / 6.01, 8.0 / 12.01],
                [-6.0 / 6.01, -12.0 / 12.01],
                [9.0 / 6.01, 18.0 / 12.01],
                [14.0 / 6.01, 28.0 / 12.01],
            ])
            numpy.testing.assert_allclose(numpy.load(trainnorm_channels), expected, rtol=1e-6)
            self.assertEqual(state["score_input_lengths"], [7, 5])

            with open(result["metadata_path"], encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(metadata, {
                "window_size": 5,
                "test_length": 10,
                "score_length": 5,
                "label_slice": [5, None],
            })

            with open(result["snapshot_path"], encoding="utf-8") as snapshot_file:
                snapshot = json.load(snapshot_file)
            self.assertEqual(snapshot["git_commit_hash"], "project-hash")
            with open("configs/data_preprocessing.yaml", encoding="utf-8") as config_file:
                data_preprocessing = yaml.safe_load(config_file)
            with open("configs/scoring_pipeline.yaml", encoding="utf-8") as config_file:
                scoring_pipeline = yaml.safe_load(config_file)
            self.assertEqual(snapshot["config"], {
                "run_config": config,
                "data_preprocessing": data_preprocessing,
                "scoring_pipeline": scoring_pipeline,
                "environment": yaml.safe_load(Path("configs/environment.yaml").read_text(encoding="utf-8")),
                "runtime_versions": {"python": "fixed"},
                "input": {
                    "path": "synthetic://unit-test",
                    "feature_names": ["sensor_00", "sensor_01"],
                    "files": [{
                        "name": "synthetic.csv",
                        "size_bytes": 1,
                        "sha256": "a" * 64,
                    }],
                    "session_splits": [{
                        "source": "synthetic",
                        "train_range": [0, 6],
                        "validation_range": [6, 12],
                        "test_range": [0, 10],
                    }],
                },
                "seed": 7,
                "git_hashes": {
                    "tsad_project": "project-hash",
                    "gragod_fork": "fork-hash",
                },
            })


class TestRunGdnSessions(unittest.TestCase):
    def run_once(self, output_dir):
        state = {}
        state["expected_snapshot_path"] = str(
            Path(output_dir) / "snapshots" / "config_snapshot.json"
        )
        dependencies = make_dependencies(state)
        config = make_config(str(Path(output_dir) / "gragod-fork"))
        config["naming"]["series"] = (1, 2)
        train_sessions = (
            numpy.arange(16, dtype=float).reshape(8, 2),
            100 + numpy.arange(18, dtype=float).reshape(9, 2),
        )
        validation_sessions = (
            200 + numpy.arange(14, dtype=float).reshape(7, 2),
            300 + numpy.arange(16, dtype=float).reshape(8, 2),
        )
        test_sessions = (
            400 + numpy.arange(24, dtype=float).reshape(12, 2),
            500 + numpy.arange(26, dtype=float).reshape(13, 2),
        )
        with (
            patch.object(runner, "load_gdn_dependencies", return_value=dependencies) as dependency_loader,
            patch.object(runner, "verify_run_context", return_value={
                "tsad_project": "project-hash", "gragod_fork": "fork-hash",
            }),
            patch.object(runner, "verify_git_hashes_unchanged"),
            patch.object(runner, "verify_runtime_versions", return_value={"python": "fixed"}),
            patch.object(runner, "validate_pipeline_contract"),
        ):
            result = runner.run_gdn_sessions(
                config,
                train_sessions,
                validation_sessions,
                test_sessions,
                output_dir,
                seed=7,
                input_metadata={
                    "path": "synthetic://multi-session",
                    "feature_names": ("sensor_00", "sensor_01"),
                    "files": ({
                        "name": "synthetic.csv",
                        "size_bytes": 1,
                        "sha256": "a" * 64,
                    },),
                    "session_splits": (
                        {"source": "train1"},
                        {"source": "train2"},
                    ),
                },
            )

        dependency_loader.assert_called_once_with(config["fork_path"])
        return result, state

    def test_training_window_count_is_sum_of_session_counts(self):
        with tempfile.TemporaryDirectory() as output_dir:
            _, state = self.run_once(output_dir)

        train_datasets = session_datasets(state["fit"][0])
        validation_datasets = session_datasets(state["fit"][1])
        self.assertEqual([len(dataset) for dataset in train_datasets], [3, 4])
        self.assertEqual([len(dataset) for dataset in validation_datasets], [2, 3])
        self.assertEqual(len(state["fit"][0].dataset), 7)
        self.assertEqual(len(state["fit"][1].dataset), 5)

    def test_rejects_session_without_one_forecast_window(self):
        config = make_config("unused-fork")
        config["naming"]["series"] = 1
        with patch.object(
            runner,
            "load_gdn_dependencies",
            side_effect=AssertionError("길이 검사가 dependency import보다 늦다"),
        ):
            with self.assertRaisesRegex(ValueError, r"window_size\+1"):
                runner.run_gdn_sessions(
                    config,
                    (numpy.zeros((6, 2)),),
                    (numpy.zeros((5, 2)),),
                    (numpy.zeros((6, 2)),),
                    "unused-output",
                    seed=1,
                    input_metadata={},
                )

    def test_rejects_real_dataset_without_verified_input_fingerprint(self):
        config = make_config("unused-fork")
        with patch.object(
            runner,
            "verify_run_context",
            side_effect=AssertionError("입력 지문 검사가 Git 검사보다 늦다"),
        ):
            with self.assertRaisesRegex(ValueError, "files"):
                runner.run_gdn_sessions(
                    config,
                    (numpy.zeros((6, 2)),),
                    (numpy.zeros((6, 2)),),
                    (numpy.zeros((6, 2)),),
                    "unused-output",
                    seed=1,
                    input_metadata={"path": "real-data"},
                )

    def test_rejects_malformed_input_fingerprint(self):
        config = make_config("unused-fork")
        with patch.object(
            runner,
            "verify_run_context",
            side_effect=AssertionError("입력 지문 형식 검사가 Git 검사보다 늦다"),
        ):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                runner.run_gdn_sessions(
                    config,
                    (numpy.zeros((6, 2)),),
                    (numpy.zeros((6, 2)),),
                    (numpy.zeros((6, 2)),),
                    "unused-output",
                    seed=1,
                    input_metadata={
                        "path": "real-data",
                        "files": ({
                            "name": "test.csv", "size_bytes": 10, "sha256": "bad",
                        },),
                    },
                )

    def test_two_test_sessions_share_one_fit_and_write_separate_outputs(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result, state = self.run_once(output_dir)

            self.assertEqual(state["trainer_initialization_count"], 1)
            self.assertEqual(state["fit_count"], 1)
            self.assertEqual(state["checkpoint_load_count"], 1)
            self.assertEqual(state["score_input_lengths"], [10, 12, 7, 8])
            self.assertEqual(len(result["score_paths"]), 16)
            self.assertEqual(
                sum("__01__" in Path(path).name for path in result["score_paths"]), 8,
            )
            self.assertEqual(
                sum("__02__" in Path(path).name for path in result["score_paths"]), 8,
            )

            metadata = []
            for metadata_path in result["metadata_paths"]:
                with open(metadata_path, encoding="utf-8") as metadata_file:
                    metadata.append(json.load(metadata_file))
            self.assertEqual([item["test_length"] for item in metadata], [12, 13])
            self.assertEqual([item["score_length"] for item in metadata], [7, 8])

            with open(result["snapshot_path"], encoding="utf-8") as snapshot_file:
                snapshot = json.load(snapshot_file)
            self.assertEqual(snapshot["config"]["input"], {
                "path": "synthetic://multi-session",
                "feature_names": ["sensor_00", "sensor_01"],
                "files": [{
                    "name": "synthetic.csv",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                }],
                "session_splits": [
                    {"source": "train1"},
                    {"source": "train2"},
                ],
            })


if __name__ == "__main__":
    unittest.main()
