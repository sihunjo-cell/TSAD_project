"""추천 자료가 빠진 완료 판정과 폐기 baseline 승계를 막는다."""

import csv
import json
import sqlite3
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ghl_main import run_ratio_tuning as runner


class TestRecommendationCommand(unittest.TestCase):
    def test_full_prefix_prepare_ignores_existing_legacy_results(self):
        from tests.unit.test_ratio_tuning_full_prefix import make_inputs

        registry, entries, feasibility, summary = make_inputs()
        feasibility = [{key: value for key, value in row.items() if key != "name"} for row in feasibility]
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs/input_manifest.yaml").write_text(
                json.dumps({"datasets": {"DEV18": {"files": entries}}}), encoding="utf-8",
            )
            old_manifest = root / "old_manifest.csv"
            old_ledger = root / "old_ledger.csv"
            old_manifest.write_text("discarded scores", encoding="utf-8")
            old_ledger.write_text("discarded winners", encoding="utf-8")
            paths = {"budget": root / "snapshots/budget.json", "result": root / "results",
                     "recommendation": root / "results/recommendation_evidence"}
            evidence = MagicMock()
            evidence.__enter__.return_value = evidence
            evidence.status.return_value = {"expected_prefixes": 126, "completed_prefixes": 0}
            evidence.contract = {"expected_prefixes": 126, "schema_version": "test"}
            stack.enter_context(patch.object(runner.tuning, "REPOSITORY_ROOT", root))
            stack.enter_context(patch.object(runner, "load_model_registry_with_sha",
                                            return_value=(registry, "r" * 64)))
            stack.enter_context(patch.object(runner, "_paths", return_value=paths))
            stack.enter_context(patch.object(runner, "open_recommendation_evidence", return_value=evidence))
            stack.enter_context(patch.object(runner.tuning, "validate_vus_evidence"))
            stack.enter_context(patch.object(runner.tuning, "_validate_ell_max"))
            stack.enter_context(patch(
                "tests.ghl_main.build_dev18_feasibility._load_approved_inventory",
                return_value=(None, None, "i" * 64, None),
            ))
            stack.enter_context(patch("src.common.model_feasibility.build_dev18_feasibility_rows",
                                      return_value=feasibility))
            stack.enter_context(patch("src.common.model_feasibility.summarize_dev18_feasibility",
                                      return_value=summary))

            def hash_current_file(path):
                if Path(path) in {old_manifest, old_ledger}:
                    raise AssertionError("폐기 결과를 읽었다")
                return "d" * 64

            stack.enter_context(patch.object(runner, "file_sha256", side_effect=hash_current_file))
            _, _, preserved, reused, _, sources = runner.prepare_ratio_tuning(
                baseline_manifest=old_manifest, baseline_ledger=old_ledger,
            )
            self.assertEqual((preserved, reused, sources), ([], [], {}))
            artifact = json.loads(paths["budget"].read_text(encoding="utf-8"))
            self.assertNotIn("preserved_sources", artifact)
            self.assertEqual(old_manifest.read_text(encoding="utf-8"), "discarded scores")
            columns = (paths["budget"].parent / "dev18_feasibility_ledger.csv").read_text(
                encoding="utf-8").splitlines()[0].split(",")
            self.assertIn("observed_row", columns)
            self.assertFalse({"available_count", "fit_count", "validation_count"} & set(columns))

    def test_missing_features_stop_finish_before_scoring(self):
        evidence = MagicMock()
        evidence.__enter__.return_value = evidence
        evidence.require_features.side_effect = ValueError("입력 특징이 미완료다")
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch.object(runner.tuning, "_require_clean_worktree"))
            stack.enter_context(patch.object(runner.tuning, "_git_head", return_value="a" * 40))
            stack.enter_context(patch.object(runner, "open_recommendation_evidence", return_value=evidence))
            stack.enter_context(patch.object(runner.tuning, "build_trial_score_ledger",
                                            side_effect=AssertionError("채점이 열렸다")))
            with self.assertRaisesRegex(ValueError, "입력 특징이 미완료"):
                runner.finish_ratio_tuning(
                    {}, {"experiment_mode": "full_prefix_v2"}, [], [],
                    {"result": root, "recommendation": root / "recommendation_evidence"},
                    data_root=root, selection_only=True, write_plots=False,
                )

    def test_command_exports_real_database_and_resumes_without_repeating_models(self):
        from tests.ghl_main import store_recommendation_evidence as evidence
        from tests.unit.test_recommendation_evidence import (
            completed_rows, fixture, loader, verified_metadata, write_source,
        )

        entries, registry, budget, identity = fixture()
        for order, entry in enumerate(entries):
            entry.update(row_count=50, order=order)
        budget.update(experiment_mode="full_prefix_v2", series_ids=[row["series"] for row in entries],
                      physical_execution_count=1, physical_run_count=18, primary_logical_score_row_count=14,
                      input_manifest_sha256="i" * 64, selection_rule_id="synthetic_selection",
                      model_panels=[], failure_rules={"maximum_total_attempts": 1})
        manifest, ledger = completed_rows(entries, budget)
        for row in manifest:
            row.update(family="synthetic", tier="t3", retry_count=0)
        panel = budget["execution_panel"][0]
        specification = {"model": "TSPulse", "config_id": "cActual", "ratio": 100, "seed": 0,
                         "tier": "t3", "hyperparameters": {}, "target_use": "strict_zero_shot",
                         "dataset_role": "development", "split_role": "dev18_selection",
                         "common_recipe": registry["common_recipe"]}
        events = []

        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            paths = {"budget": root / "budget.json", "manifest": root / "manifest.csv",
                     "result": root / "results", "recommendation": root / "results/recommendation_evidence",
                     "checkpoint": root / "checkpoints"}
            paths["result"].mkdir()
            paths["budget"].write_text(json.dumps(budget), encoding="utf-8")

            def open_store(*arguments, directory, **keywords):
                return evidence.RecommendationEvidence(directory, identity=identity, registry=registry,
                                                       budget=budget, entries=entries)

            def prepare(**keywords):
                with open_store(directory=paths["recommendation"]) as store:
                    runner._seal_json(paths["budget"].parent / "recommendation_contract.json", store.contract)
                return registry, budget, [], [], paths, {}

            def load_features(entry, data_root):
                events.append(("features", entry["series"]))
                return loader(entry)

            def run_model(*arguments, series, **keywords):
                with open_store(directory=paths["recommendation"]) as store:
                    self.assertTrue(store.status()["feature_complete"])
                events.append(("model", f"{series:02d}"))
                return [row for row in manifest if row["series"] == f"{series:02d}"]

            def completed(rows, *, spec, series, **keywords):
                return sum(row["series"] == f"{series:02d}" and row["status"] == "complete"
                           for row in rows) == 3

            def score(*arguments, output_path, **keywords):
                events.append(("score", "panel"))
                task = {"estimated_cost": 1, "manifest_row": manifest[0]}
                with patch.object(runner.tuning, "_score_primary_row", return_value={
                    "manifest_key": list(runner.tuning._manifest_key(manifest[0])), "reused": False,
                }):
                    runner.tuning._score_primary_rows(
                        [task], {}, keywords["workers"], scoring_history=keywords["scoring_history"],
                    )
                write_source(output_path, ledger)
                return ledger

            def load_scores(path, *arguments, **keywords):
                with path.open(encoding="utf-8", newline="") as source:
                    return list(csv.DictReader(source))

            def write_reports(selection, budget, result_directory, **keywords):
                for name in ("model_support_limits.csv", "model_ratio_policy.csv", "tier_adaptive.csv",
                             "ratio_family_lofo.csv", "candidate_audit.csv", "matched_model_comparison.csv",
                             "structural_exclusions.csv", "tspulse_official_heads.csv",
                             "tuning_support.json", "conditional_selection.json"):
                    (result_directory / name).write_text("{}\n", encoding="utf-8")

            stack.enter_context(patch.object(runner, "prepare_ratio_tuning", side_effect=prepare))
            stack.enter_context(patch.object(runner, "prepare_tuning_environment"))
            stack.enter_context(patch.object(runner, "open_recommendation_evidence", side_effect=open_store))
            stack.enter_context(patch.object(evidence, "open_recommendation_evidence", side_effect=open_store))
            stack.enter_context(patch.object(evidence.RecommendationEvidence, "_verify_execution_files",
                                            side_effect=verified_metadata))
            for name, value in (
                ("REPOSITORY_ROOT", root), ("_git_head", identity["project_commit"]),
                ("_require_clean_worktree", None), ("_require_same_worktree", None),
                ("_require_cuda_or_remote", None), ("prepare_tuning", {"environment": {"backend": "remote-test"}}),
                ("load_input_manifest_role", ({"datasets": {"DEV18": {"files": entries}}}, "i" * 64)),
                ("load_model_registry_with_sha", (registry, "r" * 64)),
                ("_specs_for_budget", ([specification], {("TSPulse", "cActual", 100, 0): panel})),
                ("_load_completion_receipt", []),
                ("validate_vus_evidence", {"evaluator_sha256": "e" * 64}),
                ("_validate_ell_max", {"ell_max_id": "ellNew"}), ("build_conditional_membership_rows", []),
            ):
                stack.enter_context(patch.object(runner.tuning, name, **(
                    {"new": value} if name == "REPOSITORY_ROOT" else {"return_value": value})))
            stack.enter_context(patch.object(runner.tuning, "_completed_run", side_effect=completed))
            stack.enter_context(patch.object(runner.tuning, "_run_one_spec", side_effect=run_model))
            stack.enter_context(patch.object(runner.tuning, "build_trial_score_ledger", side_effect=score))
            stack.enter_context(patch.object(runner.tuning, "load_trial_score_ledger", side_effect=load_scores))
            stack.enter_context(patch.object(runner, "select_ratio_tuning_policies",
                                            return_value={"model_ratio": [], "tier_adaptive": []}))
            stack.enter_context(patch.object(runner, "write_full_prefix_reports", side_effect=write_reports))
            stack.enter_context(patch("tests.ghl_main.build_tuning_support.build_tuning_support", return_value={}))
            stack.enter_context(patch("src.common.load_final_membership.load_final_membership"))
            stack.enter_context(patch("src.common.set_reproducible_seed.set_reproducible_seed"))
            stack.enter_context(patch("src.data_split.load_dev18_series.load_dev18_registered_inputs",
                                      side_effect=load_features))
            stack.enter_context(patch("tests.ghl_main.run_registered_models.load_registered_inputs",
                                      return_value={"family": "synthetic"}))

            def command(**mode):
                arguments = SimpleNamespace(**{
                    "prepare": False, "execute_only": False, "finish_only": False, "selection_only": False,
                    "baseline_manifest": root / "discarded.csv", "baseline_ledger": root / "discarded_ledger.csv",
                    "data_root": root, "workers": 1, "no_plots": True, "remote_cpu": False, **mode,
                })
                with runner.record_run_history(root / "commands", identity={"budget_id": None}) as history:
                    return runner._run_ratio_tuning_command(arguments, history)

            prepared = command(prepare=True)
            self.assertFalse(prepared["recommendation_evidence"]["feature_complete"])
            self.assertEqual(events, [])
            with patch.object(evidence.RecommendationEvidence, "_export_csv", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    command()
            self.assertEqual(events, [(kind, entry["series"]) for kind in ("features", "model") for entry in entries]
                             + [("score", "panel")])
            self.assertFalse((paths["result"] / "selection_complete.json").exists())
            histories = [json.loads(path.read_text(encoding="utf-8")) for path in (root / "commands").glob("*.json")]
            failed = [history for history in histories if history["status"] == "failed"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["stages"]["score_select_export"]["status"], "failed")
            self.assertEqual(failed[0]["scoring_environment"]["actual_workers"], 1)

            finished = command()
            self.assertEqual(finished["status"], "ready_for_handoff")
            self.assertEqual(finished["recommendation_status"], "complete")
            recommendation = finished["recommendation_evidence"]
            self.assertEqual(recommendation["completed_prefix_count"], 126)
            self.assertEqual(recommendation["completed_result_count"], 378)
            with closing(sqlite3.connect(recommendation["database"]["file"])) as backup:
                self.assertEqual(backup.execute("SELECT count(vus_pr) FROM results").fetchone()[0], 252)
                self.assertEqual(backup.execute("SELECT count(*) FROM recommendation_inputs").fetchone()[0], 126)
            exported = next(item for item in recommendation["exports"]
                            if Path(item["file"]).name == "recommendation_inputs.csv")
            with Path(exported["file"]).open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                self.assertNotIn("channel_std_median", reader.fieldnames)
                self.assertNotIn("channel_interquartile_range_median", reader.fieldnames)
                self.assertEqual(len(list(reader)), 126)
            for mode in ({"finish_only": True}, {"selection_only": True}):
                resumed = command(**mode)
                self.assertEqual(resumed["recommendation_status"], "complete")
                saved = json.loads(Path(resumed["history_file"]).read_text(encoding="utf-8"))
                self.assertEqual(saved["scoring_environment"]["actual_workers"], 0)
                self.assertEqual(saved["scoring_environment"]["scoring_status"], "no_new_scoring")
            self.assertEqual(events, [(kind, entry["series"]) for kind in ("features", "model") for entry in entries]
                             + [("score", "panel")])


if __name__ == "__main__":
    unittest.main()
