"""난수 합성 시계열로 run_gdn_single 절차를 검증한다(점수 값 자체는 무의미).

실데이터(GHL·HAI) 로드 없음. 출력은 `experiments/checks/reference_code/gdn/dryrun/`에만 쓴다.
확인 항목: 점수 파일 8개와 shape, metadata, config·두 저장소 git hash 스냅숏,
best checkpoint, early stopping 로그, 체크포인트에서 복원한 TopK edge 집합.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.naming import build_score_filename
from src.models.tier2.gdn.extract_adjacency import extract_best_adjacency
from src.models.tier2.gdn.run_gdn_single import run_gdn_single

DRYRUN_DIR = (
    REPOSITORY_ROOT / "experiments" / "checks" / "reference_code" / "gdn" / "dryrun"
)

DRYRUN_CONFIG = {
    "fork_path": str(REPOSITORY_ROOT.parent / "gragod-fork"),
    "naming": {"dataset": "SYNTH", "series": 0, "tier": "t0", "ratio": 100},
    "model_params": {
        "window_size": 8, "embed_dim": 8, "out_layer_num": 1, "out_layer_inter_dim": 16,
        "topk": 2, "heads": 1, "dropout": 0.0, "negative_slope": 0.2, "learn_graph": True,
    },
    "train_params": {
        "batch_size": 32, "n_epochs": 2, "init_lr": 0.001, "weight_decay": 0.0,
        "eps": 1.0e-8, "betas": [0.9, 0.99], "early_stop_patience": 3,
        "early_stop_delta": 0.0001, "val_size": 0.1, "min_train_length": 10,
        "shuffle": True, "log_every_n_steps": 1, "n_workers": 0,
    },
    "fork_contract": {
        "loss": "mse", "forecast_horizon": 1, "n_workers": 0,
        "validation_shuffle": False, "optimizer": "adam",
        "scheduler": {
            "name": "reduce_lr_on_plateau", "factor": 0.5,
            "patience": 8, "monitor": "Loss/val",
        },
        "gradient_clip_val": 1.0,
    },
}
DRYRUN_SEED = 1


def check(label: str, passed: bool) -> None:
    print(f"  ({label}) {'통과' if passed else '실패'}")
    if not passed:
        raise RuntimeError(label)


def run_dryrun() -> None:
    if DRYRUN_DIR.exists():
        shutil.rmtree(DRYRUN_DIR)

    random_generator = numpy.random.default_rng(0)
    train_array = random_generator.normal(size=(300, 5))
    test_array = random_generator.normal(size=(200, 5))

    result = run_gdn_single(
        DRYRUN_CONFIG, train_array, test_array, str(DRYRUN_DIR), seed=DRYRUN_SEED,
        input_metadata={
            "path": "generated://numpy.default_rng(0)",
            "feature_names": [f"feature_{index:02d}" for index in range(train_array.shape[1])],
            "session_splits": [{
                "source": "generated",
                "train_range": [0, 270],
                "validation_range": [270, 300],
                "test_range": [0, 200],
            }],
        },
    )

    print("\n=== 드라이런 판정 ===")
    naming = DRYRUN_CONFIG["naming"]
    expected_filenames = {
        build_score_filename(model="GDN", seed=DRYRUN_SEED, smoothing_kind=smoothing_kind,
                             norm_kind=norm_kind, channels=channels, **naming)
        for smoothing_kind in ("raw", "smoothed")
        for norm_kind in ("trainnorm", "testnorm")
        for channels in (False, True)
    }
    produced_filenames = {Path(path).name for path in result["score_paths"]}
    all_files_exist = expected_filenames == produced_filenames and all(
        Path(path).exists() for path in result["score_paths"])
    check("a. 규약 이름 산출 파일 trainnorm 4벌 + testnorm 4벌", all_files_exist)

    expected_score_length = len(test_array) - DRYRUN_CONFIG["model_params"]["window_size"]
    score_shapes_match = all_files_exist and all(
        (score_array := numpy.load(path)).shape == (
            (expected_score_length, test_array.shape[1])
            if Path(path).stem.endswith("__channels") else (expected_score_length,)
        )
        and numpy.isfinite(score_array).all()
        for path in result["score_paths"]
    )
    check("b. 점수 배열 shape와 유한값", score_shapes_match)

    metadata_paths = [Path(path) for path in result["metadata_paths"]]
    metadata = (
        json.loads(metadata_paths[0].read_text(encoding="utf-8"))
        if len(metadata_paths) == 1 and metadata_paths[0].is_file() else {}
    )
    check("c. metadata 길이·label offset", metadata == {
        "window_size": DRYRUN_CONFIG["model_params"]["window_size"],
        "test_length": len(test_array),
        "score_length": expected_score_length,
        "label_slice": [DRYRUN_CONFIG["model_params"]["window_size"], None],
    })

    with open(result["snapshot_path"], encoding="utf-8") as snapshot_file:
        snapshot = json.load(snapshot_file)
    git_hashes = snapshot["config"]["git_hashes"]
    check("d. config와 두 저장소 git hash 스냅숏",
          snapshot["config"]["run_config"] == DRYRUN_CONFIG
          and snapshot["config"]["seed"] == DRYRUN_SEED
          and len(git_hashes.get("tsad_project", "")) == 40
          and len(git_hashes.get("gragod_fork", "")) == 40)

    check("e. best checkpoint 존재", Path(result["best_checkpoint_path"]).is_file())

    early_log_path = Path(result["early_stopping_log_path"])
    early_log = json.loads(early_log_path.read_text(encoding="utf-8")) if early_log_path.exists() else {}
    check("f. early stopping 스텝 로그 존재", "stopped_epoch" in early_log and "best_score" in early_log)

    timing_path = Path(result["timing_path"])
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}
    timing_fields = (
        "training_seconds", "train_reference_inference_seconds", "test_inference_seconds",
    )
    check("g. 실행 장치와 학습·기준오차·테스트 추론 시간 존재", set(timing) == {
        "accelerator", *timing_fields,
    } and timing["accelerator"] in {"cpu", "cuda"}
        and all(timing[field] >= 0 for field in timing_fields))

    edges_with_self, edges_without_self = extract_best_adjacency(
        result["best_checkpoint_path"], topk=DRYRUN_CONFIG["model_params"]["topk"])
    check("h. extract_best_adjacency가 best.ckpt에서 edge 집합 반환",
          len(edges_with_self) == 5 * 2 and len(edges_without_self) > 0)

    print(f"\n산출 파일 목록 ({len(result['score_paths'])}개):")
    for path in sorted(result["score_paths"]):
        print(f"  {Path(path).name}")
    print(f"best.ckpt: {result['best_checkpoint_path']}")
    print(f"early stopping 로그: {early_log}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_dryrun()
