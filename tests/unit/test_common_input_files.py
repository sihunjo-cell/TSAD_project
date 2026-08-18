"""실행 입력 파일이 봉인 manifest와 같은지 검증한다."""

import tempfile
import unittest
from pathlib import Path

import yaml

from src.common.verify_input_files import verify_input_files


class TestVerifyInputFiles(unittest.TestCase):
    def write_manifest(self, root: Path) -> Path:
        manifest_path = root / "input_manifest.yaml"
        manifest_path.write_text(yaml.safe_dump({
            "datasets": {
                "TEST": {
                    "files": [{
                        "name": "input.csv",
                        "size_bytes": 3,
                        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                    }],
                },
            },
        }), encoding="utf-8")
        return manifest_path

    def test_returns_exact_fingerprint_for_matching_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "input.csv"
            input_path.write_bytes(b"abc")
            manifest_path = self.write_manifest(root)

            result = verify_input_files("TEST", root, manifest_path)

        self.assertEqual(result, ({
            "name": "input.csv",
            "size_bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },))

    def test_rejects_same_size_file_with_different_content(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "input.csv").write_bytes(b"abd")
            manifest_path = self.write_manifest(root)

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_input_files("TEST", root, manifest_path)


if __name__ == "__main__":
    unittest.main()
