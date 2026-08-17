"""실행 입력 파일을 봉인 manifest와 대조한다."""

import hashlib
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_file_state(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def verify_input_file_state(data_dir, verified_files) -> None:
    data_dir = Path(data_dir)
    for verified in verified_files:
        path = data_dir / verified["name"]
        expected = verified["size_bytes"], verified["mtime_ns"]
        try:
            actual = read_file_state(path)
        except FileNotFoundError as error:
            raise RuntimeError(f"입력 파일이 검증 뒤 바뀌었다: {path.name}") from error
        if actual != expected:
            raise RuntimeError(f"입력 파일이 검증 뒤 바뀌었다: {path.name}")
    # ponytail: SHA-256은 시작할 때 한 번만 계산하고, 이후 실제 로드 경계는 stat으로 지킨다.


def verify_input_files(dataset: str, data_dir, manifest_path=None) -> tuple[dict, ...]:
    manifest_path = Path(manifest_path or REPOSITORY_ROOT / "configs" / "input_manifest.yaml")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = yaml.safe_load(file)
    try:
        expected_files = manifest["datasets"][dataset]["files"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"입력 manifest에 dataset이 없다: {dataset}") from error

    data_dir = Path(data_dir)
    verified = []
    for expected in expected_files:
        path = data_dir / expected["name"]
        if not path.is_file():
            raise FileNotFoundError(f"입력 파일이 없다: {path}")
        state_before_hash = read_file_state(path)
        size_bytes = state_before_hash[0]
        if size_bytes != expected["size_bytes"]:
            raise ValueError(
                f"입력 파일 크기가 manifest와 다르다: {path.name} "
                f"{size_bytes} != {expected['size_bytes']}"
            )
        sha256 = sha256_file(path)
        if read_file_state(path) != state_before_hash:
            raise RuntimeError(f"입력 파일이 SHA-256 검증 중 바뀌었다: {path.name}")
        if sha256 != expected["sha256"]:
            raise ValueError(f"입력 파일 SHA-256이 manifest와 다르다: {path.name}")
        verified.append({
            "name": path.name,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "mtime_ns": state_before_hash[1],
        })
    return tuple(verified)
