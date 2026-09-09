"""실행마다 독립 이력을 남기고 이전 완료 영수증을 보존한다."""

import datetime
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_ACTIVE_STARTS = {}


def _write_history_file(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def save_run_history(record):
    """작업을 시작하기 전에 신원과 현재 상태를 디스크에 기록한다."""
    if record["run_id"] in _ACTIVE_STARTS:
        record["elapsed_seconds_lower_bound"] = (
            record["elapsed_seconds"] if record.get("elapsed_seconds") is not None
            else time.perf_counter() - _ACTIVE_STARTS[record["run_id"]]
        )
    content = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _write_history_file(record["history_file"], content.encode("utf-8"))


@contextmanager
def record_run_history(directory, *, identity):
    """강제 종료는 running으로 남기며 확인하지 못한 종료시간을 추정하지 않는다."""
    started = time.perf_counter()
    run_id = uuid.uuid4().hex
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "history_file": str(Path(directory) / f"{run_id}.json"),
        "identity": dict(identity),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "finished_at": None,
        "elapsed_seconds": None,
        "status": "running",
    }
    _ACTIVE_STARTS[run_id] = started
    try:
        save_run_history(record)
        yield record
    except BaseException as error:
        record.update(
            status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
            error_type=type(error).__name__, error_message=str(error),
        )
        raise
    else:
        record["status"] = "complete"
    finally:
        record["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record["elapsed_seconds"] = time.perf_counter() - started
        try:
            save_run_history(record)
        finally:
            _ACTIVE_STARTS.pop(run_id, None)


@contextmanager
def record_run_stage(history, name):
    """단계 시간은 명령 전체 시간의 내역이며 별도 비용으로 더하지 않는다."""
    started = time.perf_counter()
    stage = {"status": "running", "elapsed_seconds": None}
    history.setdefault("stages", {})[name] = stage
    save_run_history(history)
    try:
        yield stage
    except BaseException as error:
        stage["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        raise
    else:
        stage["status"] = "complete"
    finally:
        stage["elapsed_seconds"] = time.perf_counter() - started
        save_run_history(history)


def _load_histories(directory, budget_id):
    for path in sorted(Path(directory).glob("*.json")):
        content = path.read_bytes()
        record = json.loads(content)
        if record["identity"].get("budget_id") == budget_id:
            yield path, content, record


def has_run_persistence_failure(record):
    return (
        record.get("error_type") == "RunResultPersistenceError"
        or record.get("training_complete", {}).get("status") == "saving"
        or record.get("stages", {}).get("save_training", {}).get("status") in {"running", "failed", "interrupted"}
    )


def load_attempt_counts(directory, budget_id, *, computed_attempts=None):
    """manifest를 쓰기 전에 종료된 시도도 재시도 예산에서 빼지 않는다."""
    counts, next_indices = {}, {}
    for path, _, record in _load_histories(directory, budget_id):
        identity = record["identity"]
        if identity.get("kind") != "model_attempt":
            continue
        index = identity["attempt"]
        if type(index) is not int or index < 0:
            raise ValueError("실행 이력의 시도 번호가 잘못됐다")
        key = tuple(map(str, (identity["series"], identity["model"], identity["config_id"],
                             identity["ratio"], identity["seed"])))
        counts[key] = counts.get(key, 0) + 1
        next_indices[key] = max(next_indices.get(key, 0), index + 1)
        if computed_attempts is not None and (
                record.get("model_execution_complete") is True or has_run_persistence_failure(record)):
            if index >= computed_attempts.get(key, {}).get("attempt", -1):
                computed_attempts[key] = {
                    "attempt": index, "history_file": str(path), "result": record.get("result"),
                }
    return {key: max(count, next_indices[key]) for key, count in counts.items()}


def summarize_run_history(directory, budget_id):
    """측정된 시간과 강제 종료로 끝을 모르는 구간을 분리한다."""
    known_seconds, lower_bound = 0.0, 0.0
    unknown, references = [], []
    for path, content, record in _load_histories(directory, budget_id):
        elapsed = record.get("elapsed_seconds")
        if elapsed is None:
            unknown.append(record["run_id"])
            measured = record.get("elapsed_seconds_lower_bound", 0.0)
        else:
            measured = elapsed
        if type(measured) not in (int, float) or not math.isfinite(measured) or measured < 0:
            raise ValueError("실행 이력의 측정 시간이 잘못됐다")
        lower_bound += measured
        if elapsed is not None:
            known_seconds += elapsed
        references.append({"file": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                           "run_id": record["run_id"], "status": record["status"]})
    return {"known_elapsed_seconds": known_seconds,
            "elapsed_seconds_lower_bound": lower_bound,
            "total_elapsed_seconds": None if unknown else known_seconds,
            "unknown_elapsed_run_ids": unknown, "history_files": references}


def preserve_run_receipt(receipt_path, directory):
    """같은 완료 영수증은 한 번 보존하고 내용이 바뀌면 새 파일로 남긴다."""
    receipt_path = Path(receipt_path)
    if not receipt_path.exists():
        return None
    content = receipt_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    archived = Path(directory) / f"{receipt_path.stem}__{digest}.json"
    if archived.exists():
        if archived.read_bytes() != content:
            raise ValueError("보존한 실행 영수증의 내용이 SHA와 다르다")
    else:
        _write_history_file(archived, content)
    return archived


@contextmanager
def hold_tuning_lock(path):
    """프로세스 종료 때 운영체제가 해제하는 잠금으로 동시 덮어쓰기를 막는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            acquire = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(lock, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as error:
            raise RuntimeError("다른 튜닝 명령이 실행 중이다. 같은 산출물을 동시에 수정하지 않는다") from error
        try:
            yield
        finally:
            release()
