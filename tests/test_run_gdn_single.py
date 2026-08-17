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

from src.gdn_runner import run_gdn_single as runner


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
        "naming": {"dataset": "HAI", "series": 1, "tier": "t2", "ratio": 10},
    }


def make_dependencies(state: dict):
    train_errors = torch.tensor([
        [0.0, 0.0], [2.0, 4.0], [4.0, 8.0],
        [6.0, 12.0], [8.0, 16.0], [10.0, 20.0],
    ])
    test_errors = torch.tensor([
        [5.0, 10.0], [10.0, 20.0], [0.0, 0.0], [15.0, 30.0],
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
            state["fit"] = (train_loader, val_loader, args_summary)

    class PredictionTrainer:
        def __init__(self, **arguments):
            state.setdefault("prediction_trainer_arguments", []).append(arguments)

        def predict(self, lightning_module, loader):
            return None

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
                raise runner.subprocess.CalledProcessError(128, command)
            self.assertEqual(arguments["cwd"], repository_dir)
            return SimpleNamespace(stdout="a" * 40 + "\n")

        with patch.object(runner.subprocess, "run", side_effect=emulate_git):
            self.assertEqual(runner.read_git_hash(repository_dir), "a" * 40)


class TestRunGdnSingle(unittest.TestCase):
    def run_once(self, output_dir):
        state = {}
        dependencies = make_dependencies(state)
        config = make_config(str(Path(output_dir) / "gragod-fork"))
        config["naming"]["dataset"] = "GHL"
        config["model_params"]["topk"] = 5
        train_array = numpy.arange(24, dtype=float).reshape(12, 2)
        test_array = numpy.arange(20, dtype=float).reshape(10, 2)
        input_metadata = {
            "path": "synthetic://unit-test",
            "feature_names": ("sensor_00", "sensor_01"),
            "session_splits": ({
                "source": "synthetic",
                "train_range": (0, 9),
                "validation_range": (9, 12),
                "test_range": (0, 10),
            },),
        }

        with (
            patch.object(runner, "load_gdn_dependencies", return_value=dependencies, create=True) as dependency_loader,
            patch.object(runner, "read_git_hash", side_effect=["project-hash", "fork-hash", "project-hash"]),
        ):
            result = runner.run_gdn_single(
                config, train_array, test_array, output_dir, seed=7,
                input_metadata=input_metadata,
            )

        dependency_loader.assert_called_once_with(config["fork_path"])
        return result, state, config, train_array

    def test_training_contract_splits_tail_and_injects_model_arguments(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result, state, config, train_array = self.run_once(output_dir)

        self.assertEqual(state["seed"], 7)
        numpy.testing.assert_array_equal(session_datasets(state["fit"][0])[0].data.numpy(), train_array[:9])
        numpy.testing.assert_array_equal(session_datasets(state["fit"][1])[0].data.numpy(), train_array[9:])
        self.assertEqual(state["fit"][2], {"seed": 7})
        self.assertEqual(state["model_params"]["edge_index"], ["edge-index"])
        self.assertEqual(state["model_params"]["n_features"], 2)
        self.assertEqual(state["model_params"]["out_dim"], 2)
        self.assertEqual(state["model_params"]["topk"], 5)
        self.assertNotIn("edge_index", config["model_params"])
        self.assertEqual(result["best_checkpoint_path"], "fake-best.ckpt")

    def test_output_contract_uses_train_errors_and_records_alignment_snapshot(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result, state, config, _ = self.run_once(output_dir)

            trainnorm_channels = next(
                path for path in result["score_paths"]
                if "raw__trainnorm__channels.npy" in path
            )
            expected = numpy.array([
                [0.0, 0.0],
                [5.0 / 5.01, 10.0 / 10.01],
                [-5.0 / 5.01, -10.0 / 10.01],
                [10.0 / 5.01, 20.0 / 10.01],
            ])
            numpy.testing.assert_allclose(numpy.load(trainnorm_channels), expected, rtol=1e-6)
            self.assertEqual(state["score_input_lengths"], [6, 4])

            with open(result["metadata_path"], encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(metadata, {
                "window_size": 5,
                "test_length": 10,
                "score_length": 4,
                "label_slice": [5, -1],
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
                "input": {
                    "path": "synthetic://unit-test",
                    "feature_names": ["sensor_00", "sensor_01"],
                    "session_splits": [{
                        "source": "synthetic",
                        "train_range": [0, 9],
                        "validation_range": [9, 12],
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
            patch.object(runner, "read_git_hash", side_effect=["project-hash", "fork-hash", "project-hash"]),
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

    def test_two_test_sessions_share_one_fit_and_write_separate_outputs(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result, state = self.run_once(output_dir)

            self.assertEqual(state["trainer_initialization_count"], 1)
            self.assertEqual(state["fit_count"], 1)
            self.assertEqual(state["checkpoint_load_count"], 1)
            self.assertEqual(state["score_input_lengths"], [9, 11, 6, 7])
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
            self.assertEqual([item["score_length"] for item in metadata], [6, 7])

            with open(result["snapshot_path"], encoding="utf-8") as snapshot_file:
                snapshot = json.load(snapshot_file)
            self.assertEqual(snapshot["config"]["input"], {
                "path": "synthetic://multi-session",
                "feature_names": ["sensor_00", "sensor_01"],
                "session_splits": [
                    {"source": "train1"},
                    {"source": "train2"},
                ],
            })


if __name__ == "__main__":
    unittest.main()
