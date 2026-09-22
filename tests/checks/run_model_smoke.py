"""실데이터·checkpoint 없이 활성 모델의 점수 정렬 계약을 점검한다."""

import json
import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import numpy
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

OUTPUT_PATH = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code"
    / "active_models" / "source_only_smoke.json"
)
FULL_PREFIX_OUTPUT_PATH = OUTPUT_PATH.parent / "full_prefix_v2/full_prefix_smoke.json"
FULL_PREFIX_SOURCES = (
    "tests/checks/run_model_smoke.py", "src/common/run_registered_model.py",
    "src/common/save_model_artifacts.py", "src/common/execution_evidence.py",
    "src/common/model_feasibility.py", "src/common/normalization.py",
    "src/common/save_scores.py", "src/data_split/split_ratio_prefix.py",
    "src/models/tier2/paano/adapter.py", "src/models/tier2/paano/official.py",
    "src/models/tier2/gdn_official/adapter.py", "src/models/tier2/gdn_official/official.py",
)


class _ConstantTimeRCD(torch.nn.Module):
    def forward(self, *, time_series, mask):
        del mask
        logits = torch.zeros((*time_series.shape, 2), device=time_series.device)
        logits[..., 1] = 1.0
        return logits


def _shape(output):
    return list(numpy.asarray(output["scores"]).shape)


def run_source_only_smoke() -> dict:
    """활성 모델의 순수 점수·stitching 경로를 합성 입력으로 실행한다."""
    from src.models.tier1 import (
        PcaLegacy, score_mwvar, score_mwvar96_sqdiff_centered5,
        score_mwvar96_sqdiff_last3, score_sqdiff_centered5,
        score_sqdiff_last1, score_sqdiff_last3,
    )
    from src.models.tier2.gdn_official.adapter import build_forecast_arrays, topk_from_rho
    from src.models.tier2.paano.adapter import stitch_patch_scores
    from src.models.tier3.time_rcd import score_time_rcd
    from src.models.tier3.tspulse import score_tspulse
    random = numpy.random.default_rng(25)
    matrix = random.normal(size=(140, 4)).astype(numpy.float32)
    models = {}

    def check(name, operation):
        try:
            details = operation()
        except Exception as error:  # pragma: no cover - 보고서에 실패 원인을 보존한다.
            models[name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            models[name] = {"status": "passed", **details}

    check("MWVAR", lambda: {"score_shape": _shape(score_mwvar(matrix))})
    check("SQDIFF_LAST1", lambda: {"score_shape": _shape(score_sqdiff_last1(matrix))})
    check("SQDIFF_LAST3", lambda: {"score_shape": _shape(score_sqdiff_last3(matrix))})
    check("SQDIFF_CENTERED5", lambda: {"score_shape": _shape(score_sqdiff_centered5(matrix))})
    check("MWVAR96_SQDIFF_LAST3", lambda: {"score_shape": _shape(score_mwvar96_sqdiff_last3(matrix))})
    check("MWVAR96_SQDIFF_CENTERED5", lambda: {"score_shape": _shape(score_mwvar96_sqdiff_centered5(matrix))})

    def check_pca():
        model = PcaLegacy(n_components=0.5).fit(matrix)
        return {"score_shape": _shape(model.score(matrix[-120:]))}

    check("PCA_LEGACY", check_pca)
    check("PaAno", lambda: {
        "score_shape": list(stitch_patch_scores(numpy.arange(7), 4, 10).shape),
    })
    def check_gdn():
        arrays = build_forecast_arrays((matrix[:9],), window_size=5)
        return {
            "window_shape": list(arrays[0][0].shape),
            "score_shape": list(arrays[0][1].shape),
            "topk": topk_from_rho(4, 0.3),
        }

    check("GDN", check_gdn)
    check("TimeRCD", lambda: {
        "score_shape": _shape(score_time_rcd(
            _ConstantTimeRCD(), matrix[:9], context_length=5,
        )),
    })

    def check_tspulse():
        values = matrix[:12]

        def raw_heads(_values, *, aggregation_window):
            del aggregation_window
            raw = numpy.arange(4, dtype=numpy.float64)
            return {head: raw for head in ("time", "fft", "pred")}

        outputs = score_tspulse(
            values,
            raw_head_function=raw_heads,
            aggregation_window=4,
            context_length=8,
        )
        return {"score_shapes": {name: _shape(output) for name, output in outputs.items()}}

    check("TSPulse", check_tspulse)
    return {
        "input_kind": "synthetic_only",
        "uses_labels": False,
        "downloads_checkpoints": False,
        "models": models,
    }


def _full_prefix_identity():
    from src.common.execution_identity import file_sha256
    from src.common.model_registry import load_model_registry_with_sha

    registry, registry_sha = load_model_registry_with_sha()
    return {"registry_sha256": registry_sha,
            "methodology_revision": registry["common_recipe"].get("methodology_revision"),
            "source_sha256": {path: file_sha256(REPOSITORY_ROOT / path) for path in FULL_PREFIX_SOURCES}}


def validate_full_prefix_smoke(path=FULL_PREFIX_OUTPUT_PATH):
    """작은 원격 실행의 성공 기록을 현재 코드와 registry에 묶는다."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    identity = _full_prefix_identity()
    official = identity.get("methodology_revision") == "paper_tuning_v4"
    if (report.get("schema_version") != (2 if official else 1)
            or report.get("protocol") != ("paper_tuning_v4" if official else "full_prefix_v2")
            or report.get("status") != "passed" or report.get("input_kind") != "synthetic_only"
            or any(report.get(field) != value for field, value in identity.items())
            or set(report.get("models", {})) != {"PaAno", "GDN"}
            or any(details.get("status") != "passed" for details in report["models"].values())
            or (official and any(
                details.get("registered_arguments") is not True
                or details.get("native_saved_scores") is not True
                for details in report["models"].values()
            ))):
        raise ValueError("full-prefix smoke가 현재 코드·registry의 성공 기록과 다르다")
    return report


def _require_smoke_device(device, remote_execution):
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없어 모델 smoke를 중단한다")
    elif not remote_execution:
        raise RuntimeError("CPU 모델 smoke는 명시한 원격 환경에서만 실행한다")


def _run_official_tuning_smoke(registry, registry_sha, identity, device):
    """Exercise registered argument translation and native saving with reduced caps."""
    from src.common.build_config_id import build_config_id
    from src.common.execution_evidence import FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, build_execution_evidence
    from src.common.run_registered_model import (
        _synchronize_cuda, build_entrypoint_arguments, load_model_entrypoint,
        prepare_session_inputs, select_input_scaler,
    )
    from src.common.save_model_artifacts import save_model_score
    from src.common.set_reproducible_seed import set_reproducible_seed
    from src.models.tier2.paano.adapter import PaAnoAdapter

    random = numpy.random.default_rng(25)
    models = {}
    for name in ("PaAno", "GDN"):
        try:
            set_reproducible_seed(25)
            model = registry["models"][name]
            parameters = dict(model["candidates"][0]["hyperparameters"])
            parameters["iterations" if name == "PaAno" else "epochs"] = 1
            config_id = build_config_id(
                model=name, source_commit=model["source_commit"],
                source_checkpoint_sha256=model["source_checkpoint_sha256"],
                hyperparameters=parameters, preprocess_recipe=model["preprocess_recipe"],
                common_recipe=registry["common_recipe"],
            )
            spec = {
                "model": name, "tier": model["tier"], "hyperparameters": parameters,
                "target_use": model["target_use"],
                "common_recipe": registry["common_recipe"], "preprocess_recipe": model["preprocess_recipe"],
                "seed": 25, "ratio": 100, "dataset_role": "development",
            }
            channels = max(5, parameters.get("topk", 5))
            window = parameters.get("patch_size", parameters.get("window"))
            normal = random.normal(size=(max(2 * window + 4, 50), channels)).astype(numpy.float32)
            test = random.normal(size=(window + 8, channels)).astype(numpy.float32)
            scaler_kind = select_input_scaler(spec)
            preprocessing_started = time.perf_counter()
            prepared = prepare_session_inputs(
                normal_training=normal, test_sessions=(test,), ratio_percent=100,
                scale=scaler_kind != "none", scaler_kind=scaler_kind, full_prefix=True,
            )
            preprocessing_seconds = time.perf_counter() - preprocessing_started
            arguments = build_entrypoint_arguments(spec, device=device, channel_count=channels)
            assert arguments["official_procedure"] is True
            if name == "PaAno":
                setup_started = time.perf_counter()
                adapter = load_model_entrypoint(name)(**arguments)
                _synchronize_cuda(device)
                setup_seconds = time.perf_counter() - setup_started
                training_started = time.perf_counter()
                log = adapter.fit(prepared["fit_sessions"][0])
                _synchronize_cuda(device)
                training_seconds = time.perf_counter() - training_started
                inference_started = time.perf_counter()
                output = adapter.score(prepared["test_sessions"][0])
                _synchronize_cuda(device)
                inference_seconds = time.perf_counter() - inference_started
                restored = PaAnoAdapter.from_checkpoint(adapter.checkpoint(), device=device)
                numpy.testing.assert_allclose(
                    output["scores"], restored.score(prepared["test_sessions"][0])["scores"],
                )
                timing = {"model_setup_seconds": setup_seconds, "training_seconds": training_seconds,
                          "calibration_inference_seconds": 0.0, "test_inference_seconds": inference_seconds}
            else:
                result = load_model_entrypoint(name)(
                    prepared["fit_sessions"], (), prepared["test_sessions"], **arguments,
                )
                output = result["test_outputs"][0]
                log, timing = result["training_log"], result["timing"]
                assert result["validation_outputs"] == () and result["calibration_outputs"] == ()
                assert log["checkpoint_selection"] == "source_prefix_window_validation_mse"
            assert output["native_postprocessing"] is True
            assert output["official_protocol"] == "paper_tuning_v4"
            duration = {"observed_duration_seconds": None, "duration_basis": "unavailable"}
            evidence = build_execution_evidence(
                prepared["session_splits"][0], {**timing, "split_preprocess_seconds": preprocessing_seconds},
                spec=spec, measurement_protocol_id=FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
                retry_count=0, training_session_durations=[duration], test_input_sessions=(test,),
                test_session_durations=[duration], peak_memory_mb=None, model_artifact_bytes=0,
                resource_usage={"status": "unavailable", "reason": "synthetic smoke does not measure resource usage"},
            )
            with tempfile.TemporaryDirectory() as directory:
                saved = save_model_score(
                    output, directory, dataset="DEV18", series=1, model=name, tier="t2",
                    target_use=model["target_use"], ratio=100, seed=25, config_id=config_id,
                    common_recipe=registry["common_recipe"], common_recipe_id=registry["common_recipe_id"],
                    normalization_scope=output["normalization_scope"], config_registry_sha256=registry_sha,
                    execution_identity={"dataset_role": "development", "split_role": "dev18_selection",
                        "input_manifest_sha256": hashlib.sha256(normal.tobytes()).hexdigest(),
                        "final_policy_membership_sha256": None}, execution_evidence=evidence,
                )
                metadata = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))
                arrays = {Path(path).name: numpy.load(path, allow_pickle=False) for path in saved["score_paths"]}
                assert len(arrays) == len(saved["score_paths"]) == (4 if name == "GDN" else 2)
                raw_name = Path(saved["metadata_path"]).name.replace(".meta.json", ".npy")
                for channels in ((False, True) if name == "GDN" else (False,)):
                    filename = raw_name.replace(".npy", "__channels.npy") if channels else raw_name
                    raw = arrays[filename]
                    assert raw.ndim == (2 if channels else 1)
                    assert numpy.array_equal(raw, arrays[filename.replace("__raw__", "__smoothed__")])
                    if channels:
                        assert numpy.array_equal(raw.max(axis=1), arrays[raw_name])
                assert all(numpy.isfinite(values).all() for values in arrays.values())
                assert metadata["calibration_reference"] is None
                assert metadata["smoothing"] == {"kind": "model_native", "window": 0, "boundary": "model_native"}
            models[name] = {"status": "passed", "observed_row": len(normal),
                            "registered_arguments": True, "native_saved_scores": True,
                            "smoke_parameters": parameters, "training_log": log}
        except Exception as error:
            models[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    if _full_prefix_identity() != identity:
        raise RuntimeError("official smoke 실행 중 코드 또는 registry가 바뀌었다")
    return {"schema_version": 2, "protocol": "paper_tuning_v4", "input_kind": "synthetic_only",
            "device": device, "scope": "registered argument translation and native saving; one update or epoch; not HPO",
            **identity, "models": models,
            "status": "passed" if all(details["status"] == "passed" for details in models.values()) else "failed"}


def run_full_prefix_smoke(*, device="cuda", remote_execution=False):
    """원격에서 실제 adapter·checkpoint·fit 정규화를 작은 합성 입력으로 검사한다."""
    _require_smoke_device(device, remote_execution)
    identity = _full_prefix_identity()
    from src.common.build_config_id import build_config_id
    from src.common.execution_evidence import FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, build_execution_evidence
    from src.common.model_registry import load_model_registry_with_sha
    from src.common.run_registered_model import prepare_session_inputs
    from src.common.save_model_artifacts import save_model_score
    from src.common.set_reproducible_seed import set_reproducible_seed
    from src.models.tier2.paano.adapter import PaAnoAdapter
    from src.models.tier2.paano.official import select_memory_count
    from src.models.tier2.gdn_official.adapter import run_gdn_sessions

    set_reproducible_seed(25)
    registry, registry_sha = load_model_registry_with_sha()
    if registry["common_recipe"].get("methodology_revision") == "paper_tuning_v4":
        return _run_official_tuning_smoke(registry, registry_sha, identity, device)
    random = numpy.random.default_rng(25)
    models = {}

    def check(name, operation):
        try:
            models[name] = {"status": "passed", **operation()}
        except Exception as error:
            models[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}

    def check_paano():
        normal = random.normal(size=(200, 5)).astype(numpy.float32)
        test = random.normal(size=(80, 5)).astype(numpy.float32)
        prepared = prepare_session_inputs(normal_training=normal, test_sessions=(test,), ratio_percent=40, scale=False, full_prefix=True)
        parameters = {**registry["models"]["PaAno"]["candidates"][0]["hyperparameters"], "iterations": 1, "batch_size": 4}
        adapter = PaAnoAdapter(**parameters, device=device, full_prefix=True, memory_policy="official_minimum")
        log = adapter.fit(prepared["fit_sessions"][0])
        output = adapter.score(prepared["test_sessions"][0])
        restored = PaAnoAdapter.from_checkpoint(adapter.checkpoint(), device=device)
        numpy.testing.assert_allclose(output["scores"], restored.score(prepared["test_sessions"][0])["scores"])
        expected_memory = select_memory_count(80 - parameters["patch_size"] + 1, memory_policy="official_minimum")
        assert len(adapter.memory_bank) == expected_memory
        assert output["calibration_mode"] == "none" and prepared["validation_sessions"] == ()
        return {"observed_row": 80, "memory_count": expected_memory,
                "checkpoint_roundtrip": True, "optimizer_updates": log["optimizer_updates"], "smoke_parameters": parameters}

    def check_gdn():
        normal = random.normal(size=(40, 5)).astype(numpy.float32)
        test = random.normal(size=(11, 5)).astype(numpy.float32)
        prepared = prepare_session_inputs(normal_training=normal, test_sessions=(test,), ratio_percent=40, scale=True, full_prefix=True)
        result = run_gdn_sessions(prepared["fit_sessions"], (), prepared["test_sessions"], embedding_dimension=8,
                                  hidden_dimension=8, rho=.3, device=device, window_size=5, epochs=2, batch_size=4, full_prefix=True)
        assert result["validation_outputs"] == () and result["training_log"]["epochs_completed"] == 2
        assert result["training_log"]["checkpoint_selection"] == "best_training_loss"
        assert 1 <= result["training_log"]["selected_epoch"] <= 2
        references = result["calibration_outputs"]
        assert len(references[0]["scores"]) == 11
        spec = {"dataset_role": "development", "target_use": "fit_full_prefix", "ratio": 40}
        duration = {"observed_duration_seconds": None, "duration_basis": "unavailable"}
        evidence = build_execution_evidence(prepared["session_splits"][0], {**result["timing"], "split_preprocess_seconds": 0.0},
            spec=spec, measurement_protocol_id=FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, retry_count=0,
            training_session_durations=[duration], test_input_sessions=(test,), test_session_durations=[duration], peak_memory_mb=None, model_artifact_bytes=0)
        model = registry["models"]["GDN"]
        config_id = build_config_id(model="GDN", source_commit=model["source_commit"], source_checkpoint_sha256=model["source_checkpoint_sha256"],
            hyperparameters=result["checkpoint"]["model_config"], preprocess_recipe=model["preprocess_recipe"], common_recipe=registry["common_recipe"])
        with tempfile.TemporaryDirectory() as directory:
            saved = save_model_score(result["test_outputs"][0], directory, dataset="DEV18", series=1, model="GDN", target_use="fit_full_prefix", tier="t2",
                ratio=40, seed=25, config_id=config_id, common_recipe=registry["common_recipe"], common_recipe_id=registry["common_recipe_id"],
                normalization_scope="current_prefix_fit", calibration_scores=tuple(output["scores"] for output in references),
                calibration_source_starts=tuple(output["source_start"] for output in references), config_registry_sha256=registry_sha,
                execution_identity={"dataset_role": "development", "split_role": "dev18_selection", "input_manifest_sha256": hashlib.sha256(normal.tobytes()).hexdigest(), "final_policy_membership_sha256": None}, execution_evidence=evidence)
            metadata = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))
            assert metadata["calibration_reference"]["source"] == "fit"
            assert all(numpy.isfinite(numpy.load(path, allow_pickle=False)).all() for path in saved["score_paths"])
        return {"observed_row": 16, "calibration_count": 11,
                "epochs_completed": 2, "fit_calibration_saved": True}

    check("PaAno", check_paano)
    check("GDN", check_gdn)
    if _full_prefix_identity() != identity:
        raise RuntimeError("full-prefix smoke 실행 중 코드 또는 registry가 바뀌었다")
    return {"schema_version": 1, "protocol": "full_prefix_v2", "input_kind": "synthetic_only", "device": device,
            "scope": "small adapter pipeline; reduced optimization budgets; not HPO performance",
            **identity, "models": models,
            "status": "passed" if all(details["status"] == "passed" for details in models.values()) else "failed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-prefix", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--remote-cpu", action="store_true")
    arguments = parser.parse_args()
    _require_smoke_device(arguments.device, arguments.remote_cpu)
    report = run_full_prefix_smoke(device=arguments.device, remote_execution=arguments.remote_cpu) if arguments.full_prefix else run_source_only_smoke()
    output_path = FULL_PREFIX_OUTPUT_PATH if arguments.full_prefix else OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if any(result["status"] != "passed" for result in report["models"].values()):
        raise RuntimeError(f"model smoke 실패: {output_path}")


if __name__ == "__main__":
    main()
