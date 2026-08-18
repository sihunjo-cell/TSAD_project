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
        size_bytes = path.stat().st_size
        if size_bytes != expected["size_bytes"]:
            raise ValueError(
                f"입력 파일 크기가 manifest와 다르다: {path.name} "
                f"{size_bytes} != {expected['size_bytes']}"
            )
        sha256 = sha256_file(path)
        if sha256 != expected["sha256"]:
            raise ValueError(f"입력 파일 SHA-256이 manifest와 다르다: {path.name}")
        verified.append({
            "name": path.name,
            "size_bytes": size_bytes,
            "sha256": sha256,
        })
    return tuple(verified)
