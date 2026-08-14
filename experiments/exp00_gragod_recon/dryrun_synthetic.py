"""S3b 드라이런 — 난수 합성 시계열로 run_gdn_single 절차 검증 (점수 값 자체는 무의미).

실데이터(GHL·HAI) 로드 없음. 출력은 dryrun_out/ — exp02·exp03 폴더를 쓰지 않는다.
확인 항목: (a) 규약 이름 산출 파일(trainnorm 4벌 + testnorm 4벌),
(b) 스냅샷 JSON 의 두 저장소 git hash, (c) early stopping 스텝 로그,
(d) extract_best_adjacency 가 best.ckpt 에서 edge 집합 반환.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.common.naming import build_score_filename
from src.gdn_runner.extract_adjacency import extract_best_adjacency
from src.gdn_runner.run_gdn_single import run_gdn_single

DRYRUN_DIR = Path(__file__).resolve().parent / "dryrun_out"

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
}
DRYRUN_SEED = 1


def check(label: str, passed: bool) -> bool:
    print(f"  ({label}) {'통과' if passed else '실패'}")
    return passed


def run_dryrun() -> None:
    if DRYRUN_DIR.exists():
        shutil.rmtree(DRYRUN_DIR)

    random_generator = numpy.random.default_rng(0)
    train_array = random_generator.normal(size=(300, 5))
    test_array = random_generator.normal(size=(200, 5))

    result = run_gdn_single(DRYRUN_CONFIG, train_array, test_array, str(DRYRUN_DIR), seed=DRYRUN_SEED)

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

    with open(result["snapshot_path"], encoding="utf-8") as snapshot_file:
        snapshot = json.load(snapshot_file)
    git_hashes = snapshot["config"]["git_hashes"]
    check("b. 스냅샷 JSON 에 두 저장소 git hash",
          bool(git_hashes.get("tsad_project")) and bool(git_hashes.get("gragod_fork"))
          and len(git_hashes["gragod_fork"]) == 40)

    early_log_path = Path(result["early_stopping_log_path"])
    early_log = json.loads(early_log_path.read_text(encoding="utf-8")) if early_log_path.exists() else {}
    check("c. early stopping 스텝 로그 존재", "stopped_epoch" in early_log and "best_score" in early_log)

    edges_with_self, edges_without_self = extract_best_adjacency(
        result["best_checkpoint_path"], topk=DRYRUN_CONFIG["model_params"]["topk"])
    check("d. extract_best_adjacency 가 best.ckpt 에서 edge 집합 반환",
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
