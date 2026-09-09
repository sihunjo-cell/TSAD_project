"""현행 v5 공통 YAML과 registry 점수 의미가 갈라지지 않는지 검증한다."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.common.experiment_config import (
    load_dataset_ratios,
    load_validation_fraction,
)
from src.common.model_registry import load_model_registry


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class TestExperimentConfig(unittest.TestCase):
    def test_current_datasets_use_the_same_seven_ratios(self):
        expected = (5, 10, 20, 40, 60, 80, 100)
        self.assertEqual(load_dataset_ratios("GHL"), expected)
        self.assertEqual(load_dataset_ratios("HAI"), expected)
        self.assertEqual(load_validation_fraction(), 0.0)

    def test_rejects_ratio_or_validation_drift(self):
        with TemporaryDirectory() as directory:
            config_directory = Path(directory) / "configs"
            config_directory.mkdir()
            config = {
                "common": {"validation_fraction": 0.2},
                "datasets": {"GHL": {"ratios": [7]}},
            }
            path = config_directory / "data_preprocessing.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "비율 계약"):
                load_dataset_ratios("GHL", directory)
            config["datasets"]["GHL"]["ratios"] = [5, 10, 20, 40, 60, 80, 100]
            config["common"]["validation_fraction"] = 1.0
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_fraction"):
                load_validation_fraction(directory)

    def test_registry_common_recipe_matches_active_yaml_semantics(self):
        config_directory = REPOSITORY_ROOT / "configs"
        preprocessing = yaml.safe_load(
            (config_directory / "data_preprocessing.yaml").read_text(encoding="utf-8")
        )
        scoring = yaml.safe_load(
            (config_directory / "scoring_pipeline.yaml").read_text(encoding="utf-8")
        )
        recipe = load_model_registry()["common_recipe"]
        self.assertEqual(recipe["input_dispatch"], preprocessing["common"]["input_dispatch"])
        self.assertEqual(recipe["methodology_revision"], "paper_tuning_v4")
        self.assertEqual(recipe["methodology_revision"], scoring["methodology_revision"])
        self.assertEqual(recipe["input_scaling"], preprocessing["common"]["input_scaling"])
        self.assertEqual(recipe["score_calibration"], {
            "fit_validation": "model_native",
            "full_prefix_scalar": "model_native",
            "full_prefix_channels": "model_native",
            "target_free": scoring["normalization"]["target_free_method"],
            "epsilon": scoring["normalization"]["epsilon"],
        })
        self.assertEqual(recipe["smoothing"], {
            "kind": scoring["smoothing"]["kind"],
            "window": scoring["smoothing"]["window"],
            "boundary": scoring["smoothing"]["boundary"],
        })
        self.assertEqual(recipe["aggregation"]["channel_scores"],
                         scoring["aggregation"]["mode"])

    def test_recipe_comparison_detects_config_drift(self):
        registry_recipe = load_model_registry()["common_recipe"]
        changed = copy.deepcopy(registry_recipe)
        changed["smoothing"]["window"] = 5
        self.assertNotEqual(changed, registry_recipe)


if __name__ == "__main__":
    unittest.main()
