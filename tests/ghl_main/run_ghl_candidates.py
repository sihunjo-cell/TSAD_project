"""GHL 파일 하나에서 웹사이트 후보 전체를 실행하고, 저장한 점수를 CPU 병렬로 채점한다.

`run_ghl_candidates.sbatch`가 GPU 한 장으로 파일을 차례대로 넘긴다. 모델은 GPU에서 하나씩 돌고,
그동안 작업에 할당된 CPU에서 하나를 뺀 수만큼 프로세스가 끝난 점수를 VUS-PR로 채점한다. GHL 검증의
비용은 웹사이트의 추정을 그대로 쓰므로 실행 시간은 재지 않는다. 끝난 실행과 채점은 기록을 보고
건너뛰므로 다시 제출하면 이어서 돈다.

후보와 recipe는 Dev18 DB와 같다. 웹사이트가 GHL(센서 19개)에서 제외하는 PCA_LEGACY
`n_components=None`과 top-k가 센서 수보다 큰 GDN만 뺀다. 학습형 후보는 계획이 학습하는
비율(5~80%)만 돌린다. 100%는 운영 기간 끝이라 학습하지 않는다. seed는 registry의 final seed이며,
모든 후보에서 첫 seed를 먼저 끝낸 뒤 다음 seed로 넘어간다. 채점은 Dev18과 같은 VUS-PR이다
(ℓ_max는 학습 구간 ACF 첫 peak의 중앙값, threshold 250개).
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

# Dev18 실행기와 같은 BLAS 초기화다. OpenBLAS를 1스레드로 초기화한 뒤 늘리면 일부 빌드의 SVD가 충돌한다.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
try:
    CPU_COUNT = len(os.sched_getaffinity(0))
except (AttributeError, OSError):
    CPU_COUNT = os.cpu_count() or 1
os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_COUNT)
import pandas
import scipy.linalg  # noqa: F401
from threadpoolctl import threadpool_limits

_BLAS_SERIAL_LIMIT = threadpool_limits(limits=1, user_api="blas")
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from tests.checks.run_lightning_dev18 import configure_cuda_environment

configure_cuda_environment()

from src.common.execution_evidence import FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, build_execution_evidence
from src.common.execution_identity import validate_input_manifest_role
from src.common.model_registry import load_model_registry_with_sha
from src.common.run_registered_model import execute_registered_model
from src.common.save_model_artifacts import save_execution_result
from src.채점기.parser import load_and_validate_score
from src.채점기.vus_pr import vus_pr
from tests.ghl_main.run_registered_models import (
    DEFAULT_INPUT_MANIFEST_PATH, build_output_directory, build_specs, load_registered_inputs,
)
from tests.tuning.build_dev18_ell_max import summarize_training_periods


DATA_DIRECTORY = REPOSITORY_ROOT.parent / "shared_data" / "TSAD_project" / "TSB-AD-M"
EXPERIMENT_DIRECTORY = REPOSITORY_ROOT / "experiments" / "01_ghl_main"
RUN_DIRECTORY = EXPERIMENT_DIRECTORY / "results" / "runs"
SERVICE_RATIOS = (5, 10, 20, 40, 60, 80)
GHL_SENSOR_COUNT = 19
N_THRESHOLDS = 250
MODEL_ORDER = (
    "MWVAR", "SQDIFF_LAST1", "SQDIFF_LAST3", "SQDIFF_CENTERED5", "MWVAR96_SQDIFF_LAST3",
    "MWVAR96_SQDIFF_CENTERED5", "PCA_LEGACY", "TimeRCD", "PaAno", "GDN", "TSPulse",
)
UNMEASURED_DURATION = {"observed_duration_seconds": None, "duration_basis": "unavailable"}
RUN_FIELDS = (
    "series", "csv_file", "model", "target_use", "config_id", "ratio", "seed", "score_variant",
    "status", "status_reason", "score_file", "project_commit", "finished_at",
)
VUS_FIELDS = ("score_file", "vus_pr", "l_max_samples", "n_thresholds", "vus_seconds")


def run_key(values) -> tuple:
    return values["model"], values["config_id"], int(values["ratio"]), int(values["seed"])


def read_rows(path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def append_row(path, fields, row) -> None:
    new = not Path(path).exists()
    with Path(path).open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)


def ghl_file_names() -> list[str]:
    import yaml

    manifest = yaml.safe_load(DEFAULT_INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    return sorted(entry["name"] for entry in manifest["datasets"]["GHL"]["files"])


def ghl_series(csv_name: str) -> str:
    return f"{int(csv_name.split('_GHL_id_')[1].split('_')[0]):02d}"


def run_paths(series: str) -> tuple[Path, Path]:
    return RUN_DIRECTORY / f"GHL_{series}_runs.csv", RUN_DIRECTORY / f"GHL_{series}_vus.csv"


def build_ghl_specs(seeds=None) -> list[dict]:
    """Dev18 후보 spec을 GHL 실행 역할과 registry의 final seed로 옮긴다."""
    registry, _ = load_model_registry_with_sha()
    final_seeds = registry["seeds"]["final"]
    seeds = set(seeds or final_seeds)
    if not seeds <= set(final_seeds):
        raise SystemExit(f"seed는 registry final seed {final_seeds} 중에서 고른다")
    physical = [
        spec for spec in build_specs("development")
        if spec["seed"] == registry["seeds"]["development"][0]
        and not (spec["model"] == "PCA_LEGACY" and spec["hyperparameters"]["n_components"] is None)
        and not (spec["model"] == "GDN" and spec["hyperparameters"].get("topk", 0) > GHL_SENSOR_COUNT)
        and (spec["target_use"] != "fit_full_prefix" or spec["ratio"] in SERVICE_RATIOS)
    ]
    plan = [(spec, seed) for spec in physical
            for seed in (final_seeds[:1] if registry["models"][spec["model"]]["deterministic"] else final_seeds)]
    # GHL 검증은 선택표 없이 웹사이트 후보 전체를 돌린다. 실행 목록의 SHA를 membership 자리에 둔다.
    plan_sha256 = hashlib.sha256(json.dumps(sorted(
        run_key({**spec, "seed": seed}) for spec, seed in plan)).encode("utf-8")).hexdigest()
    identity = {
        "dataset_role": "final", "split_role": "ghl25_final",
        "input_manifest_sha256": validate_input_manifest_role(DEFAULT_INPUT_MANIFEST_PATH, "final", "ghl25_final"),
        "final_policy_membership_sha256": plan_sha256,
    }
    specs = [{**spec, **identity, "seed": seed} for spec, seed in plan
             if seed in seeds or registry["models"][spec["model"]]["deterministic"]]
    return sorted(specs, key=lambda spec: (spec["seed"], MODEL_ORDER.index(spec["model"]),
                                           spec["ratio"], spec["config_id"]))


def file_complete(series: str, specs) -> bool:
    """이 파일에서 모든 spec이 끝났고 그 점수를 모두 채점했는가."""
    runs_path, vus_path = run_paths(series)
    complete = [row for row in read_rows(runs_path) if row["status"] == "complete"]
    scored = {row["score_file"] for row in read_rows(vus_path)}
    return ({run_key(spec) for spec in specs} <= {run_key(row) for row in complete}
            and all(row["score_file"] in scored for row in complete))


def require_clean_commit() -> str:
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=REPOSITORY_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()

    status = git("status", "--porcelain")
    if status:
        raise SystemExit(f"작업 트리가 clean하지 않다. 커밋한 코드로만 GHL을 실행한다:\n{status}")
    return git("rev-parse", "HEAD")


def run_spec(spec, inputs, *, series: str) -> list[dict]:
    """한 spec을 실행해 head마다 점수를 저장하고 기록 행을 만든다."""
    result = execute_registered_model(
        spec, normal_training=inputs["normal_training"], test_sessions=inputs["test_sessions"], device="cuda",
    )
    evidence = build_execution_evidence(
        result["split"], result["timing"], spec=spec, measurement_protocol_id=FULL_PREFIX_MEASUREMENT_PROTOCOL_ID,
        retry_count=0, training_session_durations=[dict(UNMEASURED_DURATION)] if result["split"] else [],
        test_input_sessions=inputs["test_sessions"], test_session_durations=[dict(UNMEASURED_DURATION)],
        peak_memory_mb=None, model_artifact_bytes=0,
        resource_usage={"status": "not_measured", "reason": "GHL 검증은 웹사이트 비용 추정을 쓰고 실행 자원을 재지 않는다"},
    )
    rows = []
    for variant in spec["score_variants"]:
        saved = save_execution_result(
            result, build_output_directory(EXPERIMENT_DIRECTORY, spec, score_variant=variant or None),
            spec=spec, dataset="GHL", series=int(series), score_variant=variant or None,
            execution_evidence=evidence,
        )
        raw_score = next(Path(path) for path in saved["score_paths"]
                         if Path(path).name.endswith("__raw__trainnorm.npy"))
        rows.append({
            "model": spec["model"], "target_use": spec["target_use"], "config_id": spec["config_id"],
            "ratio": spec["ratio"], "seed": spec["seed"], "score_variant": variant,
            "status": "complete", "status_reason": "",
            "score_file": raw_score.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
        })
    return rows


_LABELS = None
_L_MAX = None


def _set_labels(labels, l_max) -> None:
    global _LABELS, _L_MAX
    _LABELS, _L_MAX = labels, l_max


def score_file(relative_path: str) -> dict:
    started = time.perf_counter()
    scores, _, score_metadata = load_and_validate_score(REPOSITORY_ROOT / relative_path)
    start, end = score_metadata["label_slice"]
    value = vus_pr(scores, _LABELS[start:end], _L_MAX, n_thresholds=N_THRESHOLDS)
    return {"score_file": relative_path, "vus_pr": value, "l_max_samples": _L_MAX,
            "n_thresholds": N_THRESHOLDS, "vus_seconds": time.perf_counter() - started}


def collect_scores(futures, vus_path, *, wait: bool) -> None:
    """끝난 채점을 기록한다. 실패한 채점은 남기지 않아 다음 제출에서 다시 채점한다."""
    finished = as_completed(list(futures)) if wait else [future for future in futures if future.done()]
    for future in finished:
        futures.discard(future)
        try:
            append_row(vus_path, VUS_FIELDS, future.result())
        except Exception as error:
            print(f"채점 실패: {type(error).__name__}: {error}", flush=True)


def write_environment(path, *, commit: str, l_max: int) -> None:
    import torch

    payload = {
        "project_commit": commit, "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "cpu_count": CPU_COUNT, "l_max_samples": l_max,
        "n_thresholds": N_THRESHOLDS,
        "packages": sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_file(file_index: int, seeds, data_directory: Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA를 쓸 수 없다. CPU로 넘어가지 않고 멈춘다")
    commit = require_clean_commit()
    csv_name = ghl_file_names()[file_index]
    csv_path = data_directory / csv_name
    series = ghl_series(csv_name)
    specs = build_ghl_specs(seeds)
    inputs = load_registered_inputs(spec=specs[0], csv_path=csv_path)
    boundary = len(inputs["normal_training"])
    labels = pandas.read_csv(csv_path, usecols=["Label"])["Label"].to_numpy(dtype=int)[boundary:]
    l_max = summarize_training_periods(inputs["normal_training"], inputs["feature_names"])["l_max_samples"]

    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    runs_path, vus_path = run_paths(series)
    write_environment(RUN_DIRECTORY / f"GHL_{series}_environment.json", commit=commit, l_max=l_max)
    context = {"series": series, "csv_file": csv_name, "project_commit": commit}
    rows = read_rows(runs_path)
    done = {run_key(row) for row in rows if row["status"] == "complete"}
    scored = {row["score_file"] for row in read_rows(vus_path)}
    unscored = [row["score_file"] for row in rows if row["status"] == "complete" and row["score_file"] not in scored]
    pending = [spec for spec in specs if run_key(spec) not in done]
    workers = max(1, CPU_COUNT - 1)
    print(f"GHL {series} ({csv_name}): 실행 {len(pending)}/{len(specs)}건, 미채점 {len(unscored)}건 남음, "
          f"l_max={l_max}, 채점 프로세스 {workers}개", flush=True)
    # 학습 중인 CUDA 프로세스를 fork하지 않도록 spawn으로 띄운다.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_set_labels, initargs=(labels, l_max)) as executor:
        futures = {executor.submit(score_file, path) for path in unscored}
        for number, spec in enumerate(pending, 1):
            try:
                rows = run_spec(spec, inputs, series=series)
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
        print(f"GHL {series}: 실행 끝, 남은 채점 {len(futures)}건을 기다린다", flush=True)
        collect_scores(futures, vus_path, wait=True)
    print(f"GHL {series}: 끝, 완료 여부 {file_complete(series, specs)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=int, choices=range(25),
                        help="configs/input_manifest.yaml의 GHL 파일을 이름순으로 센 위치")
    parser.add_argument("--check-complete", action="store_true",
                        help="25개 파일의 실행·채점이 모두 끝났으면 0, 아니면 1로 끝난다")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="PaAno·GDN seed. 기본은 registry final seed 전부, 결정론 모델은 첫 seed 하나")
    parser.add_argument("--data-directory", type=Path, default=DATA_DIRECTORY)
    arguments = parser.parse_args()
    if arguments.check_complete:
        specs = build_ghl_specs(arguments.seeds)
        raise SystemExit(0 if all(file_complete(ghl_series(name), specs) for name in ghl_file_names()) else 1)
    if arguments.file_index is None:
        parser.error("--file-index 또는 --check-complete가 필요하다")
    run_file(arguments.file_index, arguments.seeds, arguments.data_directory)


if __name__ == "__main__":
    main()
