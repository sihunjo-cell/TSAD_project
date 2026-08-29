"""HAI registry 입력이 라벨과 세션 경계를 건드리지 않는지 검증한다."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy
import pandas

from src.common.run_registered_model import (
    execute_registered_model,
    prepare_session_inputs,
)
from src.data_split.load_hai_sessions import load_hai_registered_inputs


FEATURE_NAMES = tuple(f"sensor_{index:02d}" for index in range(86))


def _write_session(path: Path, values: numpy.ndarray, names=FEATURE_NAMES) -> None:
    frame = pandas.DataFrame(values, columns=names)
    frame.insert(0, "timestamp", [f"t{index}" for index in range(len(frame))])
    frame.to_csv(path, index=False)


def _fake_spec(model: str, *, target_use="fit_validation", tier="t1", ratio=20):
    return {
        "model": model,
        "target_use": target_use,
        "tier": tier,
        "ratio": ratio,
        "seed": 1,
        "hyperparameters": {},
    }


def _patched_executor():
    return patch.multiple(
        "src.common.run_registered_model",
        validate_registered_spec=lambda spec: None,
        set_reproducible_seed=lambda seed: {"seed": seed},
        build_entrypoint_arguments=lambda spec, **arguments: {},
    )


class _FakeAdapter:
    def __init__(self):
        self.fit_inputs = ()
        self.score_inputs = []

    def fit(self, *inputs):
        self.fit_inputs = tuple(numpy.array(values, copy=True) for values in inputs)
        return {"fit_calls": 1}

    def score(self, values):
        self.score_inputs.append(numpy.array(values, copy=True))
        return {"scores": numpy.arange(len(values), dtype=float)}


class _PoisonTraining:
    def __array__(self, *arguments, **keywords):
        raise AssertionError("target-free 경로가 training을 배열로 바꿨다")

    def __iter__(self):
        raise AssertionError("target-free 경로가 training을 순회했다")

    def __len__(self):
        raise AssertionError("target-free 경로가 training 길이를 읽었다")


class TestHaiRegisteredInputs(unittest.TestCase):
    def test_train1_to_test1_reads_only_its_raw_feature_sessions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            train = numpy.arange(3 * 86, dtype=float).reshape(3, 86)
            test = numpy.arange(2 * 86, dtype=float).reshape(2, 86) + 1000
            _write_session(dataset_dir / "hai-train1.csv", train)
            _write_session(dataset_dir / "hai-test1.csv", test)

            inputs = load_hai_registered_inputs(dataset_dir, "train1_to_test1")

        self.assertEqual(inputs["feature_names"], FEATURE_NAMES)
        self.assertEqual(inputs["split_role"], "train1_to_test1")
        self.assertEqual(len(inputs["normal_training_sessions"]), 1)
        self.assertEqual(len(inputs["test_sessions"]), 1)
        numpy.testing.assert_array_equal(inputs["normal_training_sessions"][0], train)
        numpy.testing.assert_array_equal(inputs["test_sessions"][0], test)
        self.assertEqual(inputs["source_ranges"], {
            "normal_training_sessions": ({
                "source": "hai-train1.csv", "range": (0, 3),
            },),
            "test_sessions": ({
                "source": "hai-test1.csv", "range": (0, 2),
            },),
        })

    def test_train1_train2_to_test2_preserves_each_file_as_one_session(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            train1 = numpy.full((3, 86), 1.0)
            train2 = numpy.full((4, 86), 2.0)
            test2 = numpy.full((5, 86), 3.0)
            _write_session(dataset_dir / "hai-train1.csv", train1)
            _write_session(dataset_dir / "hai-train2.csv", train2)
            _write_session(dataset_dir / "hai-test2.csv", test2)

            inputs = load_hai_registered_inputs(
                dataset_dir, "train1_train2_to_test2",
            )

        self.assertEqual(inputs["split_role"], "train1_train2_to_test2")
        self.assertEqual(
            [len(values) for values in inputs["normal_training_sessions"]], [3, 4],
        )
        self.assertEqual([len(values) for values in inputs["test_sessions"]], [5])
        numpy.testing.assert_array_equal(inputs["normal_training_sessions"][0], train1)
        numpy.testing.assert_array_equal(inputs["normal_training_sessions"][1], train2)
        numpy.testing.assert_array_equal(inputs["test_sessions"][0], test2)
        self.assertEqual(
            [entry["source"] for entry in inputs["source_ranges"]["normal_training_sessions"]],
            ["hai-train1.csv", "hai-train2.csv"],
        )

    def test_invalid_role_column_layout_and_nonfinite_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            values = numpy.zeros((3, 86))
            _write_session(dataset_dir / "hai-train1.csv", values)
            _write_session(dataset_dir / "hai-test1.csv", values)

            with self.assertRaisesRegex(ValueError, "split_role"):
                load_hai_registered_inputs(dataset_dir, "train1_to_test2")

            _write_session(
                dataset_dir / "hai-train1.csv", values[:, :-1],
                names=FEATURE_NAMES[:-1],
            )
            with self.assertRaisesRegex(ValueError, "86개"):
                load_hai_registered_inputs(dataset_dir, "train1_to_test1")

            _write_session(dataset_dir / "hai-train1.csv", values)
            _write_session(
                dataset_dir / "hai-test1.csv", values,
                names=tuple(reversed(FEATURE_NAMES)),
            )
            with self.assertRaisesRegex(ValueError, "열 또는 순서"):
                load_hai_registered_inputs(dataset_dir, "train1_to_test1")

            invalid = values.copy()
            invalid[1, 4] = numpy.nan
            _write_session(dataset_dir / "hai-test1.csv", invalid)
            with self.assertRaisesRegex(ValueError, "결측 또는 비유한"):
                load_hai_registered_inputs(dataset_dir, "train1_to_test1")

    def test_empty_training_or_test_session_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            values = numpy.zeros((3, 86))
            empty = numpy.empty((0, 86))
            for empty_filename in ("hai-train1.csv", "hai-test1.csv"):
                _write_session(dataset_dir / "hai-train1.csv", values)
                _write_session(dataset_dir / "hai-test1.csv", values)
                _write_session(dataset_dir / empty_filename, empty)
                with self.subTest(empty_filename=empty_filename):
                    with self.assertRaisesRegex(ValueError, "비어"):
                        load_hai_registered_inputs(dataset_dir, "train1_to_test1")


class TestPrepareSessionInputs(unittest.TestCase):
    def test_splits_each_prefix_independently_and_fits_one_scaler_on_fit_only(self):
        first = numpy.column_stack((numpy.arange(25), numpy.arange(25) + 100.0))
        second = numpy.column_stack((numpy.arange(30) + 10.0, numpy.arange(30) + 200.0))
        tests = (
            numpy.array([[1000.0, 2000.0], [2000.0, 3000.0]]),
            numpy.array([[3000.0, 4000.0]]),
        )

        prepared = prepare_session_inputs(
            normal_training_sessions=(first, second),
            test_sessions=tests,
            ratio_percent=20,
            scale=True,
        )

        self.assertEqual([len(values) for values in prepared["fit_sessions"]], [4, 4])
        self.assertEqual(
            [len(values) for values in prepared["validation_sessions"]], [1, 2],
        )
        self.assertEqual([len(values) for values in prepared["test_sessions"]], [2, 1])
        self.assertEqual(
            [split["available_range"] for split in prepared["session_splits"]],
            [(0, 5), (0, 6)],
        )
        self.assertEqual(
            [split["fit_range"] for split in prepared["session_splits"]],
            [(0, 4), (0, 4)],
        )
        self.assertEqual(prepared["scaler_state"]["data_min"], [0.0, 100.0])
        self.assertEqual(prepared["scaler_state"]["data_max"], [13.0, 203.0])
        self.assertEqual(prepared["scaler_state"]["sample_count"], 8)
        numpy.testing.assert_allclose(prepared["fit_sessions"][0][0], [0.0, 0.0])
        numpy.testing.assert_allclose(prepared["fit_sessions"][1][-1], [1.0, 1.0])
        self.assertGreater(prepared["test_sessions"][0][0, 0], 1.0)
        self.assertGreater(prepared["test_sessions"][0][0, 1], 1.0)

    def test_single_array_style_stays_one_unscaled_session(self):
        normal = numpy.arange(25 * 2, dtype=float).reshape(25, 2)
        test = numpy.arange(6, dtype=float).reshape(3, 2)

        prepared = prepare_session_inputs(
            normal_training=normal,
            test_sessions=(test,),
            ratio_percent=20,
            scale=False,
        )

        self.assertEqual(len(prepared["fit_sessions"]), 1)
        self.assertEqual(len(prepared["validation_sessions"]), 1)
        numpy.testing.assert_array_equal(prepared["fit_sessions"][0], normal[:4])
        numpy.testing.assert_array_equal(prepared["validation_sessions"][0], normal[4:5])
        numpy.testing.assert_array_equal(prepared["test_sessions"][0], test)
        self.assertIsNone(prepared["scaler_state"])

    def test_rejects_ambiguous_mismatched_empty_and_nonfinite_sessions(self):
        valid = numpy.zeros((25, 2))
        cases = (
            {
                "normal_training": valid,
                "normal_training_sessions": (valid,),
                "test_sessions": (valid,),
            },
            {
                "normal_training_sessions": (valid,),
                "test_sessions": (numpy.zeros((4, 3)),),
            },
            {
                "normal_training_sessions": (numpy.empty((0, 2)),),
                "test_sessions": (valid,),
            },
            {
                "normal_training_sessions": (valid,),
                "test_sessions": (numpy.array([[numpy.inf, 0.0]]),),
            },
        )
        for arguments in cases:
            with self.subTest(arguments=tuple(arguments)):
                with self.assertRaises(ValueError):
                    prepare_session_inputs(
                        ratio_percent=20, scale=False, **arguments,
                    )


class TestRegisteredSessionWiring(unittest.TestCase):
    def test_single_array_keeps_legacy_split_and_adapter_inputs(self):
        normal = numpy.arange(25 * 2, dtype=float).reshape(25, 2)
        test = numpy.arange(6, dtype=float).reshape(3, 2)
        adapter = _FakeAdapter()

        with _patched_executor():
            result = execute_registered_model(
                _fake_spec("FakeSingle"),
                normal_training=normal,
                test_sessions=(test,),
                device="unused",
                entrypoint=lambda **arguments: adapter,
            )

        self.assertIsInstance(result["split"], dict)
        self.assertEqual(result["split"]["fit_range"], (0, 4))
        self.assertEqual(result["split"]["validation_range"], (4, 5))
        numpy.testing.assert_array_equal(adapter.fit_inputs[0], normal[:4])
        numpy.testing.assert_array_equal(adapter.score_inputs[0], normal[4:5])
        numpy.testing.assert_array_equal(adapter.score_inputs[1], test)

    def test_session_runner_receives_separate_train_validation_and_test_tuples(self):
        first = numpy.arange(25 * 2, dtype=float).reshape(25, 2)
        second = numpy.arange(30 * 2, dtype=float).reshape(30, 2) + 100
        tests = (numpy.ones((2, 2)), numpy.ones((3, 2)) * 2)
        received = []

        def fake_session_runner(fit_sessions, validation_sessions, test_sessions):
            received.append((fit_sessions, validation_sessions, test_sessions))
            return {
                "checkpoint": None,
                "validation_outputs": (),
                "test_outputs": (),
                "training_log": None,
                "timing": {
                    "model_setup_seconds": 0.0,
                    "training_seconds": 0.0,
                    "validation_inference_seconds": 0.0,
                    "test_inference_seconds": 0.0,
                },
            }

        with _patched_executor(), patch(
            "src.common.run_registered_model.SESSION_RUNNERS", {"FakeSession"},
        ):
            result = execute_registered_model(
                _fake_spec("FakeSession", tier="t2"),
                normal_training_sessions=(first, second),
                test_sessions=tests,
                device="unused",
                entrypoint=fake_session_runner,
            )

        self.assertEqual(len(received), 1)
        fit_sessions, validation_sessions, test_sessions = received[0]
        self.assertEqual([len(values) for values in fit_sessions], [4, 4])
        self.assertEqual([len(values) for values in validation_sessions], [1, 2])
        self.assertEqual([len(values) for values in test_sessions], [2, 3])
        self.assertIsInstance(result["split"], tuple)
        self.assertEqual(len(result["split"]), 2)

    def test_multi_session_single_array_path_fails_before_any_entrypoint(self):
        sessions = (
            numpy.zeros((25, 2)),
            numpy.ones((25, 2)),
        )
        test = (numpy.zeros((3, 2)),)

        def forbidden_entrypoint(*arguments, **keywords):
            raise AssertionError("injected entrypoint가 호출됐다")

        with _patched_executor(), self.assertRaisesRegex(ValueError, "여러 training session"):
            execute_registered_model(
                _fake_spec("FakeSingle"),
                normal_training_sessions=sessions,
                test_sessions=test,
                device="unused",
                entrypoint=forbidden_entrypoint,
            )

        def forbidden_loader(model):
            raise AssertionError("entrypoint loader가 호출됐다")

        with _patched_executor(), patch(
            "src.common.run_registered_model.load_model_entrypoint",
            side_effect=forbidden_loader,
        ), self.assertRaisesRegex(ValueError, "여러 training session"):
            execute_registered_model(
                _fake_spec("FakeSingle"),
                normal_training_sessions=sessions,
                test_sessions=test,
                device="unused",
            )

    def test_target_free_path_never_reads_training(self):
        test = numpy.arange(6, dtype=float).reshape(3, 2)
        received = []

        def fake_scorer(values):
            received.append(numpy.array(values, copy=True))
            return {"scores": numpy.arange(len(values), dtype=float)}

        with _patched_executor():
            result = execute_registered_model(
                _fake_spec(
                    "FakeFree", target_use="strict_zero_shot", tier="t3", ratio=100,
                ),
                normal_training=_PoisonTraining(),
                test_sessions=(test,),
                device="unused",
                entrypoint=fake_scorer,
            )

        self.assertEqual(len(received), 1)
        numpy.testing.assert_array_equal(received[0], test)
        self.assertIsNone(result["split"])
        self.assertIsNone(result["scaler_state"])

    def test_isolated_executor_import_does_not_import_torch(self):
        project_root = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import src.common.run_registered_model; "
                "raise SystemExit('torch' in sys.modules)",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
