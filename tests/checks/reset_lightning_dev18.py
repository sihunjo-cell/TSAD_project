"""다른 실험과 준비 봉인을 보존하며 이전 Dev18 실행만 초기화한다."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIRMATION = "DELETE_DEV18_RUN"
RUN_PATHS = (
    Path(".runtime/runtime.json"),
    Path(".runtime/.runtime.json.tmp"),
    Path(".runtime/dev18_resource_gate.json"),
    Path(".runtime/.dev18_resource_gate.json.tmp"),
    Path(".runtime/dev18_checkpoint_smoke"),
    Path("experiments/01_ghl_main/scores/dev18"),
    Path("experiments/01_ghl_main/logs/dev18_score_manifest.csv"),
    Path("experiments/01_ghl_main/logs/.dev18_score_manifest.csv.tmp"),
    Path("experiments/01_ghl_main/logs/dev18_oom_recovery.json"),
    Path("experiments/01_ghl_main/logs/.dev18_oom_recovery.json.tmp"),
    Path("experiments/01_ghl_main/results/dev18_tuning"),
)


def _targets(repository_root: Path) -> tuple[Path, ...]:
    root = repository_root.resolve()
    targets = tuple((root / relative).resolve() for relative in RUN_PATHS)
    for target in targets:
        target.relative_to(root)
    return targets


def preview_previous_run(repository_root=REPOSITORY_ROOT) -> dict:
    root = Path(repository_root).resolve()
    found = []
    for target in _targets(root):
        if target.is_dir():
            size = sum(
                path.stat().st_size
                for path in target.rglob("*")
                if path.is_file()
            )
        elif target.is_file():
            size = target.stat().st_size
        else:
            continue
        found.append({"path": target.relative_to(root).as_posix(), "bytes": size})
    return {
        "status": "preview",
        "targets": found,
        "confirmation": CONFIRMATION,
        "preserved": [
            ".runtime/lightning_dev18_input.zip",
            "experiments/01_ghl_main/snapshots/dev18_selection",
            "Hugging Face cache와 설치한 Python package",
        ],
    }


def reset_previous_run(
    repository_root=REPOSITORY_ROOT, *, confirmation: str,
) -> dict:
    if confirmation != CONFIRMATION:
        raise ValueError(f"삭제하려면 confirmation에 {CONFIRMATION}을 정확히 넣는다")
    root = Path(repository_root).resolve()
    deleted = []
    for target in _targets(root):
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
        else:
            continue
        deleted.append(target.relative_to(root).as_posix())
    return {"status": "deleted", "deleted": deleted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm")
    arguments = parser.parse_args()
    result = (
        reset_previous_run(confirmation=arguments.confirm)
        if arguments.confirm is not None
        else preview_previous_run()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
