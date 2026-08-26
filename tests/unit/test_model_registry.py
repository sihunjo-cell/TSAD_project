"""봉인한 모델 recipe와 config_id 계약을 검증한다."""

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.common.build_config_id import build_common_recipe_id, build_config_id
from src.common.model_registry import (
    _expand_candidates,
    load_model_registry,
    load_model_registry_with_sha,
    model_registry_sha256,
    validate_primary_hpo_seal,
)


class TestBuildConfigId(unittest.TestCase):
    def setUp(self):
        self.recipe = {
            "model": "MWVAR",
            "source_commit": "dcbbd9fbeaabfb27ad084ffa4351a2418ea1dab9",
            "source_checkpoint_sha256": "none",
            "hyperparameters": {"window": 96},
            "preprocess_recipe": {"calibration": "none"},
            "common_recipe": {"input_dispatch": "registered_executor_v1"},
        }

    def test_hashes_canonical_recipe(self):
        self.assertEqual(build_config_id(**self.recipe), "c31a126ede742")
        reordered = dict(reversed(tuple(self.recipe.items())))
        self.assertEqual(build_config_id(**reordered), "c31a126ede742")

    def test_runtime_fields_cannot_change_identity(self):
        with self.assertRaises(TypeError):
            build_config_id(**self.recipe, dataset="GHL", ratio=5, seed=3)

    def test_hyperparameter_change_changes_identity(self):
        changed = {**self.recipe, "hyperparameters": {"window": 95}}
        self.assertNotEqual(build_config_id(**changed), build_config_id(**self.recipe))

    def test_checkpoint_config_change_changes_identity(self):
        first = build_config_id(
            **self.recipe, checkpoint_config_sha256="a" * 64,
        )
        second = build_config_id(
            **self.recipe, checkpoint_config_sha256="b" * 64,
        )
        self.assertNotEqual(first, second)

    def test_common_recipe_change_changes_identity(self):
        changed = deepcopy(self.recipe)
        changed["common_recipe"]["input_dispatch"] = "different"
        self.assertNotEqual(build_config_id(**changed), build_config_id(**self.recipe))

    def test_common_recipe_id_hashes_only_canonical_semantics(self):
        recipe = {"b": {"x": 1}, "a": "fixed"}
        reordered = {"a": "fixed", "b": {"x": 1}}
        self.assertEqual(build_common_recipe_id(recipe), build_common_recipe_id(reordered))
        self.assertRegex(build_common_recipe_id(recipe), r"^r[0-9a-f]{12}$")


class TestModelRegistry(unittest.TestCase):
    def test_registry_sha_is_stable_file_identity(self):
        digest = model_registry_sha256()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, model_registry_sha256())

    def test_registry_and_sha_are_loaded_from_one_snapshot(self):
        registry, digest = load_model_registry_with_sha()
        self.assertEqual(digest, model_registry_sha256())
        self.assertEqual(tuple(registry["models"]), tuple(load_model_registry()["models"]))

    def test_checkpoint_models_are_ready_after_real_source_smoke(self):
        models = load_model_registry()["models"]
        self.assertEqual(models["TimeRCD"]["execution_status"], "ready")
        self.assertEqual(models["TSPulse"]["execution_status"], "ready")

    def test_checkpoint_config_is_part_of_checkpoint_model_identity(self):
        registry = load_model_registry()["models"]
        time_rcd = registry["TimeRCD"]
        changed = build_config_id(
            model="TimeRCD",
            source_commit=time_rcd["source_commit"],
            source_checkpoint_sha256=time_rcd["source_checkpoint_sha256"],
            checkpoint_config_sha256="0" * 64,
            hyperparameters=time_rcd["candidates"][0]["hyperparameters"],
            preprocess_recipe=time_rcd["preprocess_recipe"],
            common_recipe=load_model_registry()["common_recipe"],
        )
        self.assertNotEqual(changed, time_rcd["candidates"][0]["config_id"])

    def test_checkpoint_identities_require_lowercase_sha256_values(self):
        registry = load_model_registry()["models"]
        for model_name in ("TimeRCD", "TSPulse"):
            self.assertRegex(
                registry[model_name]["source_checkpoint_sha256"], r"^[0-9a-f]{64}$",
            )
            self.assertRegex(
                registry[model_name]["checkpoint_config_sha256"], r"^[0-9a-f]{64}$",
            )

        with TemporaryDirectory() as directory:
            config_directory = Path(directory) / "configs"
            config_directory.mkdir()
            (config_directory / "model_registry.yaml").write_text(
                "common_recipe:\n"
                "  input_dispatch: registered_executor_v1\n"
                "models:\n"
                "  Broken:\n"
                "    execution_status: ready\n"
                "    source_commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
                "    source_checkpoint_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
                "    checkpoint_config_sha256: ABC\n"
                "    preprocess_recipe: {}\n"
                "    fixed: {}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint_config_sha256"):
                load_model_registry(directory)

    def test_tier3_transfer_recipes_are_explicit(self):
        registry = load_model_registry()["models"]
        self.assertEqual(
            registry["TimeRCD"]["preprocess_recipe"]["padding"],
            "short_none_long_tail_last_value",
        )
        self.assertEqual(
            registry["TSPulse"]["candidates"][0]["hyperparameters"]["heads"],
            ["time", "fft", "pred", "raw_max"],
        )
        self.assertEqual(
            registry["TSPulse"]["preprocess_recipe"]["raw_max"],
            "common_native_intersection_max_then_edge_repeat",
        )

    def test_context_and_multivariate_transfer_policies_are_hashed(self):
        models = load_model_registry()["models"]
        self.assertEqual(
            models["MWVAR"]["preprocess_recipe"]["official_scope"], "univariate",
        )
        self.assertEqual(
            models["SQDIFF_LAST3"]["preprocess_recipe"]["multivariate_extension"],
            "channelwise_then_project_max",
        )
        self.assertEqual(
            models["GDN"]["preprocess_recipe"]["aggregation"],
            "channelwise_then_project_max",
        )
        for model in models.values():
            self.assertIn(model["preprocess_recipe"]["evaluation_mode"], {
                "offline_noncausal", "causal",
            })
            self.assertIsInstance(model["preprocess_recipe"]["context_policy"], str)
            self.assertTrue(model["preprocess_recipe"]["context_policy"])

    def test_registry_roster_status_and_candidate_order_are_sealed(self):
        registry = load_model_registry()

        self.assertEqual(
            tuple(
                (
                    name,
                    model["execution_status"],
                    tuple(candidate["config_id"] for candidate in model["candidates"]),
                )
                for name, model in registry["models"].items()
            ),
            (
                ("MWVAR", "ready", ("c43024819503c",)),
                ("SQDIFF_LAST3", "ready", ("c472288b30428",)),
                ("PCA_LEGACY", "ready", (
                    "c24a495574d36", "c2fe6a5279bc1", "cc1e2a9937297",
                    "cfc7d9fddbf38",
                )),
                ("PaAno", "ready", (
                    "c486c198d51af", "c867da53d531b", "c78142611e61b",
                    "c0b08c90aa5f8", "c7dd56f14a04e", "c00ba5b22d349",
                    "c274acf78c2ae", "cabe0daecefe0", "c4edb5b522474",
                )),
                ("ALoRa", "ready", (
                    "c6d63d2174286", "cedb42f4d95c7", "cf3cfcf2b1d07",
                    "ca5e2c8657215", "c8c072cb432b9",
                )),
                ("GDN", "ready", ("cf1a967db6cfe", "c1168c94d4dfc")),
                ("TimeRCD", "ready", ("c1c5aeea6f7d3",)),
                ("TSPulse", "ready", (
                    "c12c5e6196ea5", "cb3bd93b0152d", "c04a7985c3759",
                )),
            ),
        )
        self.assertEqual(registry["seeds"], {
            "development": [0, 1, 2],
            "final": [3, 4, 5, 6, 7],
        })
        self.assertRegex(registry["common_recipe_id"], r"^r[0-9a-f]{12}$")

    def test_equal_trial_budget_and_primary_score_variant_are_sealed(self):
        registry = load_model_registry()
        validate_primary_hpo_seal(registry)
        self.assertEqual(registry["selection"]["primary_hpo_regime"], "equal_trial")
        self.assertEqual(registry["selection"]["budget_id"], "b5367ad431093")
        self.assertEqual(
            registry["selection"]["primary_score_variants"],
            {"TSPulse": ["raw_max"]},
        )
        for field, value in (
            ("budget_id", "not-sealed"),
            ("selection_status", "pending_budget_seal"),
            ("primary_score_variants", {"TSPulse": ["time"]}),
        ):
            changed = deepcopy(registry)
            changed["selection"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_primary_hpo_seal(changed)

    def test_active_models_record_unambiguous_source_provenance(self):
        models = load_model_registry()["models"]
        expected = {
            "MWVAR": (
                "https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/publications/iclr2026_timeseriesfoundationmodelsad",
                "MIT", "behavioral_reimplementation",
            ),
            "SQDIFF_LAST3": (
                "https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/publications/iclr2026_timeseriesfoundationmodelsad",
                "MIT", "behavioral_reimplementation",
            ),
            "PCA_LEGACY": (
                "https://github.com/TheDatumOrg/TSB-AD",
                "Apache-2.0", "official_source_adaptation",
            ),
            "PaAno": (
                "https://github.com/jinnnju/PaAno",
                "MIT", "official_source_adaptation",
            ),
            "ALoRa": (
                "https://github.com/CharisShimillas/ALoRa",
                "EUPL-1.2", "official_source_adaptation",
            ),
            "GDN": (
                "https://github.com/d-ailin/GDN",
                "MIT", "official_source_adaptation",
            ),
            "TimeRCD": (
                "https://github.com/thu-sail-lab/Time-RCD",
                "Apache-2.0", "package_backed_adapter",
            ),
            "TSPulse": (
                "https://github.com/ibm-granite/granite-tsfm",
                "Apache-2.0", "package_backed_adapter",
            ),
        }
        for model_name, (source_url, source_license, relationship) in expected.items():
            with self.subTest(model_name=model_name):
                model = models[model_name]
                self.assertNotIn("license", model)
                self.assertEqual(model["source_url"], source_url)
                self.assertEqual(model["source_license"], source_license)
                self.assertEqual(model["local_relationship"], relationship)
        self.assertEqual(
            {name for name, model in models.items() if "checkpoint_license" in model},
            {"TimeRCD", "TSPulse"},
        )
        self.assertTrue(all(
            models[name]["checkpoint_license"] == "Apache-2.0"
            for name in ("TimeRCD", "TSPulse")
        ))

    def test_execution_status_is_validated_when_registry_loads(self):
        registry_path = Path(__file__).resolve().parents[2] / "configs" / "model_registry.yaml"
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        cases = (
            ("misspelled_ready", "reason", "execution_status"),
            ("pending_checkpoint_smoke", "", "status_reason"),
            ("unavailable", None, "status_reason"),
        )
        for status, reason, message in cases:
            with self.subTest(status=status, reason=reason):
                changed = deepcopy(raw)
                changed["models"]["MWVAR"]["execution_status"] = status
                if reason is None:
                    changed["models"]["MWVAR"].pop("status_reason", None)
                else:
                    changed["models"]["MWVAR"]["status_reason"] = reason
                with TemporaryDirectory() as directory:
                    config_directory = Path(directory) / "configs"
                    config_directory.mkdir()
                    (config_directory / "model_registry.yaml").write_text(
                        yaml.safe_dump(changed, sort_keys=False), encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        load_model_registry(directory)

    def test_execution_status_cannot_be_implicit(self):
        registry_path = Path(__file__).resolve().parents[2] / "configs/model_registry.yaml"
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        raw["models"]["MWVAR"].pop("execution_status", None)
        with TemporaryDirectory() as directory:
            config_directory = Path(directory) / "configs"
            config_directory.mkdir()
            (config_directory / "model_registry.yaml").write_text(
                yaml.safe_dump(raw, sort_keys=False), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "execution_status"):
                load_model_registry(directory)

    def test_common_recipe_change_rotates_every_candidate_id(self):
        registry = load_model_registry()
        registry_path = Path(__file__).resolve().parents[2] / "configs" / "model_registry.yaml"
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        changed_recipe = deepcopy(registry["common_recipe"])
        changed_recipe["smoothing"]["window"] = 5
        for model_name, model in registry["models"].items():
            rebuilt = _expand_candidates(
                model_name, raw["models"][model_name], changed_recipe,
            )
            self.assertEqual(len(rebuilt), len(model["candidates"]))
            self.assertTrue(all(
                left["config_id"] != right["config_id"]
                for left, right in zip(model["candidates"], rebuilt)
            ))

    def test_candidates_have_unique_stable_ids_and_source_identity(self):
        registry = load_model_registry()
        config_ids = []
        for model_name, model in registry["models"].items():
            self.assertEqual(len(model["source_commit"]), 40, model_name)
            for candidate in model["candidates"]:
                self.assertRegex(candidate["config_id"], r"^c[0-9a-f]{12}$")
                config_ids.append(candidate["config_id"])
        self.assertEqual(len(config_ids), len(set(config_ids)))

    def test_paano_and_alora_resolved_fields_are_explicit(self):
        registry = load_model_registry()
        paano = registry["models"]["PaAno"]
        self.assertTrue(all(
            candidate["hyperparameters"]["use_revin"]
            for candidate in paano["candidates"]
        ))
        self.assertTrue(all(
            candidate["hyperparameters"]["memory_seed"] == 42
            for candidate in paano["candidates"]
        ))
        alora = registry["models"]["ALoRa"]
        self.assertTrue(all(
            candidate["hyperparameters"]["rank_threshold"] == 0.01
            for candidate in alora["candidates"]
        ))

    def test_training_defaults_that_change_a_recipe_are_in_each_config_id(self):
        registry = load_model_registry()["models"]

        pca_recipe = registry["PCA_LEGACY"]["preprocess_recipe"]
        self.assertEqual(
            pca_recipe,
            {
                "window_normalization": "window_row_zscore_ddof1",
                "scaler": "window_feature_standardscaler_fit_only",
                "calibration": "validation_median_iqr",
                "zero_pruning": False,
                "aggregation": "weighted_component_distance_scalar",
                "evaluation_mode": "offline_noncausal",
                "context_policy": "centered_window_edge_repeat",
            },
        )

        paano_recipe = registry["PaAno"]["preprocess_recipe"]
        self.assertEqual(
            paano_recipe["aggregation"], "scalar_patch_top3_cosine",
        )
        self.assertEqual(paano_recipe["memory_count"], "floor_fit_patch_fraction")
        self.assertEqual(paano_recipe["memory_cap"], "none")
        alora_model = registry["ALoRa"]
        alora = alora_model["candidates"][0]["hyperparameters"]
        self.assertEqual(
            {key: alora[key] for key in ("epochs", "batch_size", "patience")},
            {"epochs": 5, "batch_size": 256, "patience": 3},
        )
        self.assertEqual(alora_model["preprocess_recipe"]["pair_selection"], "fit_only_spearman")
        self.assertEqual(alora_model["preprocess_recipe"]["batch_order"], "fixed")
        self.assertEqual(alora_model["preprocess_recipe"]["scheduler"], "constant")
        gdn = registry["GDN"]["candidates"][0]["hyperparameters"]
        self.assertEqual(
            {key: gdn[key] for key in ("embedding", "hidden", "rho")},
            {"embedding": 64, "hidden": 128, "rho": 0.3},
        )
        self.assertEqual(gdn["batch_size"], 128)
        self.assertEqual(gdn["learning_rate"], 0.001)
        self.assertEqual(gdn["weight_decay"], 0.0)
        self.assertEqual(
            registry["GDN"]["preprocess_recipe"]["batch_policy"],
            "project_transfer_128",
        )

    def test_candidate_cannot_override_a_fixed_parameter(self):
        model = {
            "source_commit": "a" * 40,
            "source_checkpoint_sha256": "none",
            "preprocess_recipe": {},
            "fixed": {"window": 96},
            "candidates": [{"window": 95}],
        }
        with self.assertRaisesRegex(ValueError, "fixed.*window"):
            _expand_candidates("MWVAR", model, {"input_dispatch": "registered_executor_v1"})


if __name__ == "__main__":
    unittest.main()
