"""활성 모델 환경과 선택 프로필의 고정 의존성 계약을 검증한다."""

import re
import unittest
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIRECTORY = REPOSITORY_ROOT / "src" / "models"
ACTIVE_PACKAGE_PINS = {
    "accelerate": "1.14.0",
    "datasets": "4.4.2",
    "deprecated": "1.2.18",
    "einops": "0.8.1",
    "filelock": "3.20.3",
    "huggingface-hub": "0.36.0",
    "jaxtyping": "0.3.2",
    "matplotlib": "3.10.5",
    "numpy": "2.3.2",
    "pandas": "2.3.3",
    "pyyaml": "6.0.3",
    "safetensors": "0.7.0",
    "scikit-learn": "1.7.1",
    "scipy": "1.16.1",
    "tokenizers": "0.22.2",
    "torch": "2.10.0",
    "torch-geometric": "2.7.0",
    "transformers": "4.57.6",
    "urllib3": "2.6.3",
}
ACTIVE_SOURCES = {
    "git+https://github.com/thu-sail-lab/Time-RCD.git@372bb980426b2f67007311c6f3165ab789c79bef",
    "git+https://github.com/ibm-granite/granite-tsfm.git@fe7a35697723e2a2f5246ae979474bfc554e26c0",
}
def read_profile(filename: str) -> tuple[list[str], dict[str, str], set[str], list[str]]:
    includes = []
    pins = {}
    sources = set()
    invalid = []
    for raw_line in (MODELS_DIRECTORY / filename).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            includes.append(line[3:].strip())
            continue
        if line.startswith("git+"):
            sources.add(line)
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
        else:
            invalid.append(line)
    return includes, pins, sources, invalid


class TestModelEnvironment(unittest.TestCase):
    def test_active_profile_contains_only_required_exact_pins_and_sources(self):
        includes, pins, sources, invalid = read_profile("requirements.txt")

        self.assertEqual(includes, [])
        self.assertEqual(pins, ACTIVE_PACKAGE_PINS)
        self.assertEqual(sources, ACTIVE_SOURCES)
        self.assertEqual(invalid, [])

    def test_yaml_declares_one_active_environment(self):
        environment = yaml.safe_load(
            (REPOSITORY_ROOT / "configs" / "environment.yaml").read_text(encoding="utf-8"),
        )

        self.assertEqual(environment["python"], "3.11.14")
        self.assertEqual(
            {name.lower(): version for name, version in environment["packages"].items()},
            ACTIVE_PACKAGE_PINS,
        )
        self.assertEqual(set(environment["sources"].values()), ACTIVE_SOURCES)
        self.assertEqual(environment["dependency_declaration"], {
            "scope": "top_level_compatibility_pins",
            "full_resolver_lock": False,
            "installed_snapshot": "pip_freeze_after_install",
        })
        self.assertNotIn("optional_profiles", environment)
        self.assertEqual(environment["runtime_snapshot"], {
            "record_pip_freeze": True,
            "record_torch_build_suffix": True,
            "record_torch_cuda": True,
            "record_cudnn": True,
        })


if __name__ == "__main__":
    unittest.main()
