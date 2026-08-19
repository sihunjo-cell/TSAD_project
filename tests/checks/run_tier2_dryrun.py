"""난수 합성 데이터로 Tier 2 네 모델의 구현 계약만 검증한다."""

import datetime
import json
import sys
import tempfile
from pathlib import Path

import numpy
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.run_completion import write_completion_marker
from src.common.save_tier2_artifacts import save_tier2_artifacts
from src.common.set_seed import set_seed
from src.common.verify_run_context import read_source_identity, verify_runtime_versions
from src.models.tier2.AE.adapter import (
    CI_AE_CONFIG,
    load_ci_ae_checkpoint,
    run_ci_ae_sessions,
    score_ci_ae,
)
from src.models.tier2.GDN.adapter import (
    GDN_CONFIG,
    load_gdn_checkpoint,
    run_gdn_sessions,
    score_gdn,
)
from src.models.tier2.GDN.extract_adjacency import extract_best_adjacency
from src.models.tier2.LSTMAD.adapter import (
    LSTM_AD_CONFIG,
    load_lstm_ad_checkpoint,
    run_lstm_ad_sessions,
    score_lstm_ad,
)
from src.models.tier2.USAD.adapter import (
    USAD_CONFIG,
    load_usad_checkpoint,
    run_usad_sessions,
    score_usad,
)


RESULT_DIR = (
    REPOSITORY_ROOT / "experiments" / "checks" / "tier2_implementation"
)
COMMAND = "conda run -n tsad_fixed python -m tests.checks.run_tier2_dryrun"


def _save_run(
    dryrun_dir, name, result, config, test_sessions, source_identity, series,
    mark_complete=True,
):
    output_dir = dryrun_dir / name
    artifacts = save_tier2_artifacts(
        result, output_dir, "SYNTH", series, config["model"], 100, 1,
        tuple(len(session) for session in test_sessions), config,
        source_identity, {
            "path": "generated://numpy.default_rng(0)",
            "labels_read": False,
        }, 0.01, 4,
    )
    expected_scores = 8 * len(test_sessions)
    if len(artifacts["score_paths"]) != expected_scores:
        raise RuntimeError(f"{name} 점수 파일 수가 다르다")
    if not all(Path(path).is_file() for path in (
        *artifacts["score_paths"], *artifacts["metadata_paths"],
        artifacts["checkpoint_path"], artifacts["snapshot_path"],
        artifacts["early_stopping_log_path"], artifacts["timing_path"],
    )):
        raise RuntimeError(f"{name} 필수 산출물이 빠졌다")
    if mark_complete:
        write_completion_marker(output_dir)
    return artifacts


def _ghl_runs(dryrun_dir, source_identity):
    random_generator = numpy.random.default_rng(0)
    train = (random_generator.random((230, 19), dtype=numpy.float32),)
    validation = (random_generator.random((120, 19), dtype=numpy.float32),)
    test = (random_generator.random((130, 19), dtype=numpy.float32),)
    runs = (
        (
            "ci_ae", "CI-AE", CI_AE_CONFIG,
            lambda: run_ci_ae_sessions(train, validation, test, "cpu", epochs=1),
            load_ci_ae_checkpoint, score_ci_ae,
        ),
        (
            "lstm_ad", "LSTM-AD", LSTM_AD_CONFIG,
            lambda: run_lstm_ad_sessions(train, validation, test, "cpu", epochs=1),
            load_lstm_ad_checkpoint, score_lstm_ad,
        ),
        (
            "usad", "USAD", USAD_CONFIG,
            lambda: run_usad_sessions(train, validation, test, "cpu", epochs=1),
            load_usad_checkpoint, score_usad,
        ),
        (
            "gdn", "GDN", {**GDN_CONFIG, "topk": 5},
            lambda: run_gdn_sessions(train, validation, test, 5, "cpu", epochs=1),
            load_gdn_checkpoint, score_gdn,
        ),
    )
    statuses = {}
    for directory, model_name, production_config, execute, restore, score in runs:
        set_seed(1)
        result = execute()
        config = {
            **production_config,
            "model": model_name,
            "window_size": production_config["window_size"],
            "dryrun_epochs": 1,
        }
        artifacts = _save_run(
            dryrun_dir, directory, result, config, test, source_identity, (1,),
        )
        restored = restore(artifacts["checkpoint_path"], "cpu")
        restored_scores = score(restored, test, "cpu")
        if len(restored_scores) != 1 or restored_scores[0].shape != result["test_errors"][0].shape:
            raise RuntimeError(f"{model_name} checkpoint 복원 후 forward shape가 다르다")
        statuses[model_name] = {
            "passed": True,
            "production_epochs": production_config["n_epochs"],
            "dryrun_epochs": 1,
            "optimizer_updates": result["training_log"]["optimizer_updates"],
            "checkpoint_restored": True,
            "labels_read": False,
        }
    return statuses


def _hai_gdn_run(dryrun_dir, source_identity):
    random_generator = numpy.random.default_rng(1)
    train = tuple(
        random_generator.random((12 + index, 86), dtype=numpy.float32)
        for index in range(4)
    )
    validation = tuple(
        random_generator.random((11 + index, 86), dtype=numpy.float32)
        for index in range(4)
    )
    test = (
        random_generator.random((10, 86), dtype=numpy.float32),
        random_generator.random((14, 86), dtype=numpy.float32),
    )
    set_seed(1)
    result = run_gdn_sessions(train, validation, test, 22, "cpu", epochs=1)
    config = {
        **GDN_CONFIG, "topk": 22, "model": "GDN",
        "window_size": GDN_CONFIG["window_size"], "dryrun_epochs": 1,
    }
    artifacts = _save_run(
        dryrun_dir, "gdn_hai", result, config, test, source_identity, (1, 2),
        mark_complete=False,
    )
    restored = load_gdn_checkpoint(artifacts["checkpoint_path"], "cpu")
    restored_scores = score_gdn(restored, test, "cpu")
    if [score.shape for score in restored_scores] != [(5, 86), (9, 86)]:
        raise RuntimeError("HAI형 GDN checkpoint 복원 후 점수 정렬이 다르다")
    edge_sets = extract_best_adjacency(artifacts["checkpoint_path"], 22)
    adjacency_dir = dryrun_dir / "gdn_hai" / "adjacency"
    adjacency_dir.mkdir(parents=True, exist_ok=True)
    for suffix, edges in zip(("with_self", "without_self"), edge_sets):
        numpy.save(
            adjacency_dir / f"edges_{suffix}.npy",
            numpy.asarray(sorted(edges), dtype=int).T,
        )
    if len(edge_sets[0]) != 86 * 22 or not edge_sets[1]:
        raise RuntimeError("HAI형 GDN 인접행렬이 완전하지 않다")
    write_completion_marker(dryrun_dir / "gdn_hai")
    return {
        "passed": True,
        "production_epochs": GDN_CONFIG["n_epochs"],
        "dryrun_epochs": 1,
        "train_session_count": 4,
        "test_session_count": 2,
        "score_files": 16,
        "adjacency_files": 2,
        "checkpoint_restored": True,
        "labels_read": False,
    }


def run_dryrun() -> None:
    source_identity = read_source_identity(REPOSITORY_ROOT)
    with (REPOSITORY_ROOT / "configs" / "environment.yaml").open(encoding="utf-8") as file:
        environment = verify_runtime_versions(yaml.safe_load(file))
    with tempfile.TemporaryDirectory(prefix="tsad_tier2_dryrun_") as temporary_dir:
        dryrun_dir = Path(temporary_dir)
        statuses = _ghl_runs(dryrun_dir, source_identity)
        statuses["GDN-HAI-multisession"] = _hai_gdn_run(
            dryrun_dir, source_identity,
        )
    manifest = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": COMMAND,
        "synthetic_only": True,
        "real_training_executed": False,
        "labels_read": False,
        "source_identity": source_identity,
        "environment": environment,
        "models": statuses,
        "success": all(status["passed"] for status in statuses.values()),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "implementation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if not manifest["success"]:
        raise RuntimeError("Tier 2 합성 dry-run이 실패했다")
    print(json.dumps({
        "success": True,
        "models": list(statuses),
        "output": str(RESULT_DIR / "implementation_manifest.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    run_dryrun()
