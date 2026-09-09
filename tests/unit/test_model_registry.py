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
    resolve_gdn_topk,
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
    def test_gdn_keeps_official_code_and_paper_tuples_without_project_crosses(self):
        candidates = [
            candidate["hyperparameters"]
            for candidate in load_model_registry()["models"]["GDN"]["candidates"]
        ]
        official = next(candidate for candidate in candidates if candidate.get("topk") == 5)
        self.assertEqual(
            {key: official[key] for key in (
                "embedding", "hidden", "topk", "batch_size", "window", "epochs",
                "stride", "learning_rate", "weight_decay", "out_layer_num", "optimizer",
            )},
            {"embedding": 64, "hidden": 128, "topk": 5, "batch_size": 32,
             "window": 5, "epochs": 30, "stride": 1, "learning_rate": 0.001,
             "weight_decay": 0.0, "out_layer_num": 1, "optimizer": "Adam"},
        )
        self.assertFalse(any("rho" in candidate or candidate.get("topk") == 2 for candidate in candidates))
        self.assertEqual({
            (candidate["embedding"], candidate["hidden"], candidate["topk"], candidate["epochs"],
             candidate["patience"], candidate["validation_ratio"], tuple(candidate["optimizer_betas"]))
            for candidate in candidates
        }, {
            (64, 128, 5, 30, 15, 0.2, (0.9, 0.999)),
            (64, 64, 15, 50, 15, 0.1, (0.9, 0.999)),
            (128, 128, 30, 50, 15, 0.1, (0.9, 0.999)),
        })
        self.assertEqual(len(candidates), 3)
        self.assertEqual(load_model_registry()["models"]["GDN"]["preprocess_recipe"]["checkpoint_selection"],
                         "validation_loss_early_stopping")

    def test_gdn_topk_uses_exact_fixed_count_or_legacy_rho(self):
        self.assertEqual(resolve_gdn_topk(2, topk=2), 2)
        self.assertEqual(resolve_gdn_topk(2, rho=0.3), 1)
        for arguments in ({}, {"topk": 2, "rho": 0.3}, {"topk": True}, {"topk": 0}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                resolve_gdn_topk(3, **arguments)

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
            ["time", "fft", "pred", "ensemble"],
        )
        self.assertEqual(
            registry["TSPulse"]["preprocess_recipe"]["ensemble"],
            "max_after_head_minmax_and_native_smoothing_then_output_minmax",
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
            "model_native_channel_max",
        )
        for model in models.values():
            self.assertIn(model["preprocess_recipe"]["evaluation_mode"], {
                "offline_noncausal", "causal",
            })
            self.assertIsInstance(model["preprocess_recipe"]["context_policy"], str)
            self.assertTrue(model["preprocess_recipe"]["context_policy"])

    def test_registry_roster_and_full_prefix_candidate_counts_are_explicit(self):
        registry = load_model_registry()

        self.assertEqual(
            tuple(
                (
                    name,
                    model["execution_status"],
                    len(model["candidates"]),
                )
                for name, model in registry["models"].items()
            ),
            (
                ("MWVAR", "ready", 11),
                ("SQDIFF_LAST1", "ready", 1),
                ("SQDIFF_LAST3", "ready", 1),
                ("SQDIFF_CENTERED5", "ready", 1),
                ("MWVAR96_SQDIFF_LAST3", "ready", 1),
                ("MWVAR96_SQDIFF_CENTERED5", "ready", 1),
                ("PCA_LEGACY", "ready", 4),
                ("PaAno", "ready", 9),
                ("GDN", "ready", 3),
                ("TimeRCD", "ready", 1),
                ("TSPulse", "ready", 3),
            ),
        )
        self.assertEqual(registry["seeds"], {
            "development": [0, 1, 2],
            "final": [3, 4, 5, 6, 7],
        })
        self.assertRegex(registry["common_recipe_id"], r"^r[0-9a-f]{12}$")
        self.assertEqual(sum(len(model["candidates"]) for model in registry["models"].values()), 36)
        from src.common.equal_trial_budget import _score_variants

        self.assertEqual(sum(len(_score_variants(registry["selection"], name, candidate["hyperparameters"])[0])
                             for name, model in registry["models"].items() for candidate in model["candidates"]), 45)
        self.assertEqual(registry["common_recipe"]["methodology_revision"], "paper_tuning_v4")

    def test_full_prefix_selection_requires_a_new_budget(self):
        registry = load_model_registry()
        self.assertEqual(registry["selection"]["primary_hpo_regime"], "full_prefix_per_ratio")
        self.assertIsNone(registry["selection"]["budget_id"])
        self.assertEqual(registry["selection"]["selection_status"], "pending_full_prefix_budget")
        self.assertEqual(registry["selection"]["selection_rule_id"],
                         "full_prefix_tspulse_two_stage_family_lofo_v3")
        self.assertEqual(
            registry["selection"]["primary_score_variants"],
            {"TSPulse": ["time", "fft", "pred", "ensemble"]},
        )
        with self.assertRaisesRegex(ValueError, "ready"):
            validate_primary_hpo_seal(registry)

    def test_paper_native_statistics_and_duplicate_raw_smoothed_contract_are_explicit(self):
        registry = load_model_registry()
        common = registry["common_recipe"]
        self.assertEqual(common["smoothing"], {"kind": "model_native", "window": 0, "boundary": "model_native"})
        self.assertEqual(common["order"]["smoothed"], "identical_to_raw_native_score")
        self.assertNotIn("tspulse_prediction_aggregation_window", registry["selection"])
        self.assertEqual(registry["selection"]["diagnostic_score_variants"], {})
        models = registry["models"]
        self.assertEqual(models["PCA_LEGACY"]["target_use"], "training_free")
        for name in ("PCA_LEGACY", "MWVAR96_SQDIFF_LAST3", "MWVAR96_SQDIFF_CENTERED5", "GDN", "TimeRCD", "TSPulse"):
            self.assertTrue(models[name]["preprocess_recipe"]["target_statistics_fit"], name)
        self.assertEqual(models["TimeRCD"]["preprocess_recipe"]["calibration"], "none")

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

    def test_paano_resolved_fields_are_explicit(self):
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

    def test_training_defaults_that_change_a_recipe_are_in_each_config_id(self):
        registry = load_model_registry()["models"]

        pca_recipe = registry["PCA_LEGACY"]["preprocess_recipe"]
        self.assertEqual(
            pca_recipe,
            {
                "official_procedure": True,
                "fit_source": "full_evaluation",
                "target_statistics_fit": True,
                "window_normalization": "official_multi_row_ddof1_uni_column_ddof0",
                "scaler": "window_feature_standardscaler_full_evaluation",
                "calibration": "none",
                "zero_pruning": True,
                "aggregation": "weighted_component_distance_scalar",
                "evaluation_mode": "offline_noncausal",
                "context_policy": "centered_window_edge_repeat",
            },
        )

        paano_recipe = registry["PaAno"]["preprocess_recipe"]
        self.assertEqual(
            paano_recipe["aggregation"], "scalar_patch_top3_cosine",
        )
        self.assertEqual(paano_recipe["memory_count"], "official_minimum")
        self.assertEqual(paano_recipe["memory_cap"], "none")
        gdn = registry["GDN"]["candidates"][0]["hyperparameters"]
        self.assertEqual(
            {key: gdn[key] for key in ("embedding", "hidden", "topk")},
            {"embedding": 64, "hidden": 128, "topk": 5},
        )
        self.assertEqual(gdn["batch_size"], 32)
        self.assertEqual(gdn["learning_rate"], 0.001)
        self.assertEqual(gdn["weight_decay"], 0.0)
        self.assertEqual(
            registry["GDN"]["preprocess_recipe"]["batch_policy"],
            "candidate_batch_size",
        )

    def test_gdn_fixed_topk_is_forwarded_as_a_distinct_adapter_argument(self):
        from src.common.run_registered_model import build_entrypoint_arguments

        registry = load_model_registry()
        model = registry["models"]["GDN"]
        for candidate in model["candidates"]:
            parameters = candidate["hyperparameters"]
            arguments = build_entrypoint_arguments(
                {"model": "GDN", "target_use": model["target_use"],
                 "hyperparameters": parameters, "seed": 0,
                 "common_recipe": registry["common_recipe"]},
                device="cuda", channel_count=32,
            )
            if "topk" in parameters:
                self.assertEqual(arguments["fixed_topk"], parameters["topk"])
                self.assertNotIn("rho", arguments)
            else:
                self.assertEqual(arguments["rho"], parameters["rho"])
                self.assertNotIn("fixed_topk", arguments)
            self.assertEqual(arguments["batch_size"], parameters["batch_size"])

    def test_mwvar_passes_all_eleven_official_comparison_windows(self):
        from src.common.run_registered_model import build_entrypoint_arguments

        model = load_model_registry()["models"]["MWVAR"]
        self.assertEqual(
            [candidate["hyperparameters"]["window"] for candidate in model["candidates"]],
            [5, 10, 32, 50, 60, 64, 96, 100, 256, 512, 1024],
        )
        for candidate in model["candidates"]:
            self.assertEqual(
                build_entrypoint_arguments(
                    {"model": "MWVAR", "hyperparameters": candidate["hyperparameters"]},
                    device="cuda", channel_count=2,
                ),
                {"window": candidate["hyperparameters"]["window"]},
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

    def test_candidate_declarations_reject_silent_omission_and_duplicate_trials(self):
        base = {
            "source_commit": "a" * 40, "source_checkpoint_sha256": "none",
            "preprocess_recipe": {}, "fixed": {},
        }
        for declaration in (
            {"candidates": [{"window": 5}], "grid": {"window": [10]}},
            {"candidates": []}, {"grid": {"window": []}},
            {"candidates": [{"window": 5}, {"window": 5}]},
            {"grid": {"window": [5, 5]}},
        ):
            with self.subTest(declaration=declaration), self.assertRaises(ValueError):
                _expand_candidates("MWVAR", {**base, **declaration}, {})


if __name__ == "__main__":
    unittest.main()
