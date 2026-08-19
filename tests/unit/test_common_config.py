"""YAML 선언과 GDN 실행 계약이 갈라지지 않는지 검증한다."""

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from src.common.experiment_config import (
    load_dataset_ratios,
    load_planned_ratios_and_seeds,
    validate_gdn_training_contract,
    validate_pipeline_contract,
)


class TestLoadPlannedRatiosAndSeeds(unittest.TestCase):
    def test_reads_the_shared_ratio_contract_and_seeds(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "data_preprocessing.yaml").write_text(yaml.safe_dump({
                "datasets": {"GHL": {"ratios": [5, 10, 20, 40, 60, 80, 100]}},
            }), encoding="utf-8")
            (config_dir / "gdn_hyperparams.yaml").write_text(yaml.safe_dump({
                "train_params": {"seeds_by_dataset": {"GHL": [2, 4]}},
            }), encoding="utf-8")

            ratios, seeds = load_planned_ratios_and_seeds("GHL", root)

        self.assertEqual(ratios, (5, 10, 20, 40, 60, 80, 100))
        self.assertEqual(seeds, (2, 4))

    def test_current_datasets_use_the_same_seven_ratios(self):
        expected = (5, 10, 20, 40, 60, 80, 100)
        self.assertEqual(load_dataset_ratios("GHL"), expected)
        self.assertEqual(load_dataset_ratios("HAI"), expected)

    def test_rejects_ratio_outside_the_filename_and_split_contract(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "data_preprocessing.yaml").write_text(yaml.safe_dump({
                "datasets": {"GHL": {"ratios": [7]}},
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "비율 계약"):
                load_planned_ratios_and_seeds("GHL", root)


class TestValidatePipelineContract(unittest.TestCase):
    def load_configs(self):
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        with (config_dir / "scoring_pipeline.yaml").open(encoding="utf-8") as file:
            scoring = yaml.safe_load(file)
        with (config_dir / "data_preprocessing.yaml").open(encoding="utf-8") as file:
            preprocessing = yaml.safe_load(file)
        return scoring, preprocessing

    def test_accepts_current_ghl_contract(self):
        scoring, preprocessing = self.load_configs()
        validate_pipeline_contract(scoring, preprocessing, "GHL", feature_count=19)

    def test_rejects_declared_aggregation_not_used_by_runner(self):
        scoring, preprocessing = self.load_configs()
        changed = copy.deepcopy(scoring)
        changed["aggregation"]["mode"] = "mean"

        with self.assertRaisesRegex(ValueError, "aggregation.mode"):
            validate_pipeline_contract(changed, preprocessing, "GHL", feature_count=19)

    def test_rejects_fixed_scoring_value_drift(self):
        scoring, preprocessing = self.load_configs()
        changes = (
            (("normalization", "epsilon"), 0.1, "normalization.epsilon"),
            (("normalization", "author_original_appendix"), False,
             "normalization.author_original_appendix"),
            (("smoothing", "window"), 5, "smoothing.window"),
        )
        for keys, value, message in changes:
            with self.subTest(setting=".".join(keys)):
                changed = copy.deepcopy(scoring)
                changed[keys[0]][keys[1]] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_pipeline_contract(
                        changed, preprocessing, "GHL", feature_count=19,
                    )

    def test_rejects_hidden_trainer_contract_drift(self):
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        with (config_dir / "gdn_hyperparams.yaml").open(encoding="utf-8") as file:
            contract = yaml.safe_load(file)["local_training_contract"]
        changed = copy.deepcopy(contract)
        changed["scheduler"] = "reduce_lr_on_plateau"

        with self.assertRaisesRegex(ValueError, "scheduler"):
            validate_gdn_training_contract(changed)


if __name__ == "__main__":
    unittest.main()
