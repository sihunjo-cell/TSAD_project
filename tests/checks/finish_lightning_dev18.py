"""Lightning CPU에서 완료된 Dev18 점수를 재개 가능한 방식으로 채점한다."""

import os

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import datetime
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.ghl_main.run_dev18_tuning import (
    DEFAULT_DATA_ROOT,
    DEFAULT_RESULT_DIRECTORY,
    DEFAULT_SCORE_MANIFEST_PATH,
    DEFAULT_VUS_CHECKPOINT_DIRECTORY,
    _load_score_manifest,
    finish_selection_from_ledger,
    finish_tuning,
)

SELECTION_REQUIRED_FILES = (
    "model_fixed_policy.csv", "tier_fixed_policy.csv",
    "ratio_adaptive_selection.csv", "tier_ratio_candidate_audit.csv",
    "tier_policy_transitions.csv", "final_policy_membership.csv",
    "selection.png",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="기존 Dev18 score를 채점하거나 완료 ledger에서 선택표만 다시 만든다.",
    )
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument(
        "--result-directory", type=Path, default=DEFAULT_RESULT_DIRECTORY,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="0이면 CPU·가용 메모리에 맞춰 1~16개를 자동 선택한다.",
    )
    parser.add_argument(
        "--checkpoint-directory", type=Path,
        default=DEFAULT_VUS_CHECKPOINT_DIRECTORY,
    )
    arguments = parser.parse_args()

    if arguments.selection_only:
        if arguments.ledger is None:
            parser.error("--selection-only에는 --ledger PATH가 필요하다")
        from src.common.execution_identity import file_sha256

        (arguments.result_directory / "selection_complete.json").unlink(missing_ok=True)
        result = finish_selection_from_ledger(
            arguments.ledger, result_directory=arguments.result_directory,
        )
        result_directory = Path(result["result_directory"])
        missing = [
            name for name in SELECTION_REQUIRED_FILES
            if not (result_directory / name).is_file()
        ]
        if missing:
            raise RuntimeError(f"selection-only 필수 산출물이 없다: {missing}")
        receipt = {
            "schema_version": 1,
            "completed_at_utc": datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat(),
            **result,
            "expected_result_files": list(SELECTION_REQUIRED_FILES),
            "result_files_sha256": {
                name: file_sha256(result_directory / name)
                for name in SELECTION_REQUIRED_FILES
            },
        }
        completion_path = result_directory / "selection_complete.json"
        temporary_completion_path = completion_path.with_name(
            f".{completion_path.name}.{os.getpid()}.tmp"
        )
        temporary_completion_path.write_text(
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_completion_path.replace(completion_path)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
        return

    import fcntl

    arguments.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    lock_path = arguments.checkpoint_directory / "finish.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "Dev18 CPU 채점이 이미 실행 중이다. 두 번째 프로세스는 시작하지 않았다."
            ) from error
        lock.seek(0)
        lock.truncate()
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        pid_path = arguments.checkpoint_directory / "finish.pid"
        temporary_pid_path = pid_path.with_name(f".{pid_path.name}.{os.getpid()}.tmp")
        temporary_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        temporary_pid_path.replace(pid_path)
        completion_path = arguments.checkpoint_directory / "finish_complete.json"
        completion_path.unlink(missing_ok=True)
        from src.common.execution_identity import file_sha256

        score_manifest_sha256 = file_sha256(DEFAULT_SCORE_MANIFEST_PATH)
        result = finish_tuning(
            _load_score_manifest(),
            data_root=arguments.data_root,
            workers=arguments.workers,
            checkpoint_directory=arguments.checkpoint_directory,
            require_execution_environment=False,
        )
        if file_sha256(DEFAULT_SCORE_MANIFEST_PATH) != score_manifest_sha256:
            raise RuntimeError("CPU 채점 도중 Dev18 score manifest가 바뀌었다")

        result_directory = Path(result["result_directory"])
        receipt = {
            "schema_version": 1,
            "completed_at_utc": datetime.datetime.now(
                datetime.timezone.utc,
            ).isoformat(),
            **result,
            "score_manifest_sha256": score_manifest_sha256,
            "result_files_sha256": {
                path.relative_to(result_directory).as_posix(): file_sha256(path)
                for path in sorted(result_directory.rglob("*")) if path.is_file()
            },
        }
        temporary_completion_path = completion_path.with_name(
            f".{completion_path.name}.{os.getpid()}.tmp"
        )
        temporary_completion_path.write_text(
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_completion_path.replace(completion_path)
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
