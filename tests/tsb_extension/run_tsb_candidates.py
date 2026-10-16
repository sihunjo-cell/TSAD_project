"""GHL 밖 TSB-AD-M 파일 하나에서 웹사이트 후보 전체를 실행하고, 저장한 점수를 CPU 병렬로 채점한다.

`run_tsb_candidates.sbatch`가 GPU 한 장으로 파일을 차례대로 넘긴다. 절차는 GHL 실행기
(tests/ghl_main/run_ghl_candidates.py)와 같고 실행·채점·기록 함수도 그 모듈 것을 쓴다. 다른 점은 넷이다.
- 파일은 configs/tsb_extension_manifest.yaml에 지문과 함께 적은 순서로 돌고, series는 그 manifest의 두 자리
  번호다. 파일명의 id는 dataset마다 겹친다(Genesis·SWaT·GECCO·PSM 모두 id_1).
- 센서 수가 파일마다 달라서 후보를 웹사이트의 filter_candidates로 그 파일의 센서 수에서 정한다. GHL은 19개에 맞춰
  같은 규칙(PCA_LEGACY n_components=None과 top-k가 센서 수보다 큰 GDN 제외)을 적어 두었다.
- 점수와 기록은 experiments/03_tsb_extension 아래 TSB 이름으로 남겨 GHL 결과와 섞이지 않는다.
- PaAno·GDN은 STABILIZED_SERIES(Genesis·SWaT id_1·GECCO)만 --seeds를 모두 돌고, 나머지 파일은 첫 seed 하나만 돈다.
학습형 후보는 계획이 학습하는 비율(5~80%)만 돌리고, seed는 registry의 final seed다. 끝난 실행(성공·실패)과 채점은
기록을 보고 건너뛰므로 다시 제출하면 이어서 돈다.
"""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# GHL 실행기를 numpy·pandas보다 먼저 import해야 BLAS를 GHL·Dev18과 같은 순서로 초기화한다.
from tests.ghl_main.run_ghl_candidates import (
    CPU_COUNT, DATA_DIRECTORY, MODEL_ORDER, REPOSITORY_ROOT, RUN_FIELDS, SERVICE_RATIOS,
    _set_labels, append_row, collect_scores, read_rows, require_clean_commit, run_key, run_spec, score_file,
    write_environment,
)

import pandas
import yaml

from src.common.execution_identity import validate_input_manifest_role
from src.common.model_registry import load_model_registry_with_sha
from streamlit_website.db_connection.filter_candidates import filter_candidates, load_candidates
from tests.ghl_main.run_registered_models import build_specs, load_registered_inputs
from tests.tuning.build_dev18_ell_max import summarize_training_periods


MANIFEST_PATH = REPOSITORY_ROOT / "configs" / "tsb_extension_manifest.yaml"
DATA_ROOT = DATA_DIRECTORY.parent
EXPERIMENT_DIRECTORY = REPOSITORY_ROOT / "experiments" / "03_tsb_extension"
RUN_DIRECTORY = EXPERIMENT_DIRECTORY / "results" / "runs"
DATASET, SPLIT_ROLE = "TSB", "tsb_final"
# 준 seed를 모두 돌려 답안지를 안정시키는 파일(Genesis·SWaT id_1·GECCO). 나머지 파일은 첫 seed 하나만 돈다.
STABILIZED_SERIES = ("01", "02", "03")


def tsb_files() -> list[dict]:
    """manifest에 적은 순서의 파일 항목. 실행 순서이자 --file-index의 기준이다."""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest["datasets"][DATASET]["files"]


def run_paths(series: str) -> tuple[Path, Path]:
    return RUN_DIRECTORY / f"TSB_{series}_runs.csv", RUN_DIRECTORY / f"TSB_{series}_vus.csv"


def file_seeds(series: str, seeds):
    """그 파일에서 PaAno·GDN을 돌릴 seed. 실행기와 점검이 같은 규칙을 쓴다."""
    return seeds if series in STABILIZED_SERIES or not seeds else seeds[:1]


def build_tsb_specs(channel_count: int, seeds=None) -> list[dict]:
    """웹사이트가 이 센서 수에서 계획 후보로 두는 설정만 TSB 실행 역할과 registry의 final seed로 옮긴다."""
    registry, _ = load_model_registry_with_sha()
    final_seeds = registry["seeds"]["final"]
    seeds = set(seeds or final_seeds)
    if not seeds <= set(final_seeds):
        raise SystemExit(f"seed는 registry final seed {final_seeds} 중에서 고른다")
    eligible = {row["config_id"] for row in filter_candidates(load_candidates(), channel_count=channel_count)
                if row["status"] == "eligible"}
    physical = [
        spec for spec in build_specs("development")
        if spec["seed"] == registry["seeds"]["development"][0] and spec["config_id"] in eligible
        and (spec["target_use"] != "fit_full_prefix" or spec["ratio"] in SERVICE_RATIOS)
    ]
    plan = [(spec, seed) for spec in physical
            for seed in (final_seeds[:1] if registry["models"][spec["model"]]["deterministic"] else final_seeds)]
    # 선택표 없이 웹사이트 후보 전체를 돌린다. 실행 목록의 SHA를 membership 자리에 둔다.
    plan_sha256 = hashlib.sha256(json.dumps(sorted(
        run_key({**spec, "seed": seed}) for spec, seed in plan)).encode("utf-8")).hexdigest()
    identity = {
        "dataset_role": "final", "split_role": SPLIT_ROLE,
        "input_manifest_sha256": validate_input_manifest_role(MANIFEST_PATH, "final", SPLIT_ROLE),
        "final_policy_membership_sha256": plan_sha256,
    }
    specs = [{**spec, **identity, "seed": seed} for spec, seed in plan
             if seed in seeds or registry["models"][spec["model"]]["deterministic"]]
    return sorted(specs, key=lambda spec: (spec["seed"], MODEL_ORDER.index(spec["model"]),
                                           spec["ratio"], spec["config_id"]))


def run_file(file_index: int, seeds, data_root: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 쓸 수 없다. CPU로 넘어가지 않고 멈춘다")
    commit = require_clean_commit()
    entry = tsb_files()[file_index]
    series = entry["series"]
    specs = build_tsb_specs(entry["feature_count"], file_seeds(series, seeds))
    inputs = load_registered_inputs(spec=specs[0], input_manifest_path=MANIFEST_PATH, series=series,
                                    data_root=data_root)
    boundary = len(inputs["normal_training"])
    labels = pandas.read_csv(Path(data_root) / entry["source_directory"] / entry["name"],
                             usecols=["Label"])["Label"].to_numpy(dtype=int)[boundary:]
    l_max = summarize_training_periods(inputs["normal_training"], inputs["feature_names"])["l_max_samples"]

    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    runs_path, vus_path = run_paths(series)
    write_environment(RUN_DIRECTORY / f"TSB_{series}_environment.json", commit=commit, l_max=l_max)
    context = {"series": series, "csv_file": entry["name"], "project_commit": commit}
    rows = read_rows(runs_path)
    # 예외로 끝난 실행은 같은 GPU에서 다시 돌려도 같은 결과라 끝난 것으로 본다. 다시 돌리려면 그 행을 지운다.
    done = {run_key(row) for row in rows if row["status"] in ("complete", "failed")}
    scored = {row["score_file"] for row in read_rows(vus_path)}
    unscored = [row["score_file"] for row in rows if row["status"] == "complete" and row["score_file"] not in scored]
    pending = [spec for spec in specs if run_key(spec) not in done]
    workers = max(1, CPU_COUNT - 1)
    print(f"TSB {series} {entry['family']} ({entry['name']}): 실행 {len(pending)}/{len(specs)}건, "
          f"미채점 {len(unscored)}건 남음, 센서 {entry['feature_count']}개, l_max={l_max}, 채점 프로세스 {workers}개",
          flush=True)
    # 학습 중인 CUDA 프로세스를 fork하지 않도록 spawn으로 띄운다.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_set_labels, initargs=(labels, l_max)) as executor:
        futures = {executor.submit(score_file, path) for path in unscored}
        for number, spec in enumerate(pending, 1):
            try:
                rows = run_spec(spec, inputs, series=series, dataset=DATASET, experiment_directory=EXPERIMENT_DIRECTORY)
            except ImportError:
                raise
            except Exception as error:
                torch.cuda.empty_cache()
                rows = [{"model": spec["model"], "target_use": spec["target_use"], "config_id": spec["config_id"],
                         "ratio": spec["ratio"], "seed": spec["seed"],
                         "status": "failed", "status_reason": f"{type(error).__name__}: {error}"}]
            finished_at = datetime.now(timezone.utc).isoformat()
            for row in rows:
                append_row(runs_path, RUN_FIELDS, {**context, **row, "finished_at": finished_at})
            futures |= {executor.submit(score_file, row["score_file"]) for row in rows if row["status"] == "complete"}
            collect_scores(futures, vus_path, wait=False)
            print(f"[{number}/{len(pending)}] {spec['model']} {spec['config_id']} r{spec['ratio']} "
                  f"s{spec['seed']}: {rows[0]['status']}, 채점 대기 {len(futures)}건", flush=True)
        print(f"TSB {series}: 실행 끝, 남은 채점 {len(futures)}건을 기다린다", flush=True)
        collect_scores(futures, vus_path, wait=True)
    print(f"TSB {series}: 실행·채점 끝", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=int, required=True, choices=range(len(tsb_files())),
                        help="configs/tsb_extension_manifest.yaml의 파일 순서")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="PaAno·GDN seed. STABILIZED_SERIES 밖 파일은 첫 seed만 쓴다. 기본은 registry final seed "
                             "전부, 결정론 모델은 첫 seed 하나")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="manifest source_directory의 기준 폴더")
    arguments = parser.parse_args()
    run_file(arguments.file_index, arguments.seeds, arguments.data_root)


if __name__ == "__main__":
    main()
