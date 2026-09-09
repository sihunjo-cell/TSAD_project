"""작은 메타데이터로 조건별 HPO와 재개 계약을 확인한다."""

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from tests.ghl_main.build_ratio_tuning_budget import build_full_prefix_budget
from tests.ghl_main.select_ratio_tuning import select_ratio_tuning_policies


def make_inputs():
    registry = {"selection": {}, "seeds": {"development": [0, 1, 2]}, "models": {}}
    for model, deterministic, parameters in (
        ("PaAno", False, [{"patch_size": 32}, {"patch_size": 96}]),
        ("GDN", False, [{"window": 5, "topk": 5}, {"window": 5, "topk": 15}]),
        ("TimeRCD", True, [{"context_length": 5000}]),
    ):
        registry["models"][model] = {
            "tier": "t3" if deterministic else "t2", "deterministic": deterministic,
            "target_use": "strict_zero_shot" if deterministic else "fit_full_prefix",
            "source_commit": "a" * 40, "source_checkpoint_sha256": "none",
            "preprocess_recipe": {"training": "full_prefix"},
            "candidates": [{"config_id": f"c{index:012x}", "hyperparameters": item}
                           for index, item in enumerate(parameters, 1)],
        }
    entries = [{"series": f"{index:02d}", "name": f"sample_{index}.csv",
                "family": "ABC"[(index - 1) % 3], "feature_count": 2 if index <= 6 else 19,
                "training_boundary": 1000 if index <= 6 else 10000}
               for index in range(1, 19)]
    rows = []
    for model, details in registry["models"].items():
        for candidate in details["candidates"]:
            for ratio in (5, 10, 20, 40, 60, 80, 100):
                for entry in entries:
                    available = entry["training_boundary"] * ratio // 100
                    feasible = not (
                        model == "GDN" and entry["feature_count"] == 2
                        or model == "PaAno" and candidate["hyperparameters"]["patch_size"] == 96
                        and ratio <= 40 and entry["feature_count"] == 2
                    )
                    rows.append({**entry, "model": model, "tier": details["tier"],
                                 "config_id": candidate["config_id"], "logical_ratio": ratio,
                                 "available_count": available, "fit_count": available,
                                 "validation_count": 0, "status": "feasible" if feasible
                                 else "structurally_infeasible", "status_reason": "" if feasible
                                 else "too_short_or_few_channels"})
    summary = {field: "d" * 64 for field in (
        "input_manifest_sha256", "inventory_sha256", "data_preprocessing_sha256",
        "feasibility_decision_sha256",
    )}
    return registry, entries, rows, summary


def make_scores(registry, entries, budget):
    families = {entry["series"]: entry["family"] for entry in entries}
    rows = []
    for execution in budget["execution_panel"]:
        for series in execution["series_ids"]:
            for ratio in execution["logical_ratios"]:
                for variant in execution["primary_score_variants"]:
                    score = .2
                    if execution["model"] == "PaAno":
                        preferred = "c000000000001" if ratio == 60 else "c000000000002"
                        score = .8 if execution["config_id"] == preferred else .2
                    rows.append({**execution, "series": series, "family": families[series],
                                 "ratio": ratio, "score_variant": variant, "vus_pr": score,
                                 "status": "complete"})
    return rows


class TestFullPrefixTuning(unittest.TestCase):
    def setUp(self):
        self.registry, self.entries, self.feasibility, self.summary = make_inputs()
        self.budget = build_full_prefix_budget(
            self.registry, self.feasibility, self.entries, self.summary,
        )

    def test_candidates_are_available_per_series_without_all_panel_veto(self):
        large = [item for item in self.budget["execution_panel"] if item["model"] == "PaAno"
                 and item["config_id"] == "c000000000002" and item["physical_ratio"] == 5]
        self.assertEqual(len(large), 3)
        self.assertEqual(large[0]["series_ids"], [f"{index:02d}" for index in range(7, 19)])
        pulse = [item for item in self.budget["execution_panel"] if item["model"] == "TimeRCD"]
        self.assertEqual(len(pulse), 1)
        self.assertEqual(pulse[0]["seed"], 0)
        self.assertEqual(pulse[0]["logical_ratios"], [5, 10, 20, 40, 60, 80, 100])
        self.assertEqual(self.budget["expected_ledger_rows"], len(make_scores(
            self.registry, self.entries, self.budget,
        )))

    def test_budget_stores_one_observed_count_and_checks_the_full_prefix(self):
        import json

        serialized = json.dumps(self.budget)
        for field in ("available_count", "fit_count", "validation_count"):
            self.assertNotIn(field, serialized)
        group = next(group for panel in self.budget["model_panels"]
                     for group in panel["groups"] if group["ratio"] == 5)
        self.assertEqual(group["support"][0]["observed_row"], 50)
        invalid = deepcopy(self.feasibility)
        invalid[0].update(fit_count=40, validation_count=10)
        with self.assertRaises(ValueError):
            build_full_prefix_budget(self.registry, invalid, self.entries, self.summary)

    def test_conditional_config_changes_by_ratio_and_never_uses_holdout_to_select(self):
        rows = make_scores(self.registry, self.entries, self.budget)
        for row in rows:
            if row["model"] == "PaAno" and row["ratio"] == 80 and row["family"] == "C":
                row["vus_pr"] = .99 if row["config_id"] == "c000000000001" else .1
        selection = select_ratio_tuning_policies(rows, self.registry, self.budget, "e" * 64)
        policies = {row["ratio"]: row for row in selection["model_ratio"]
                    if row["model"] == "PaAno" and row["ratio"] in (60, 80)}
        self.assertEqual(policies[60]["config_id"], "c000000000001")
        self.assertEqual(policies[80]["config_id"], "c000000000002")
        fold = next(row for row in selection["adaptive_lofo"] if row["model"] == "PaAno"
                    and row["ratio"] == 80 and row["holdout_family"] == "C")
        self.assertEqual(fold["selected_config_id"], "c000000000002")
        self.assertAlmostEqual(fold["holdout_vus_pr"], .1)
        self.assertTrue(selection["tier_adaptive"])
        for policy in selection["tier_adaptive"]:
            if policy["selection_status"] == "selected":
                self.assertIn(policy["config_id"], policy["candidate_ids_by_model"][policy["model"]])
        for row in selection["model_comparison"]:
            self.assertEqual(row["left_series_ids"], row["right_series_ids"])
            for fold in row["folds"]:
                self.assertIn("left_score_variant", fold)
                self.assertIn("right_score_variant", fold)

    def test_incomplete_or_duplicate_trials_are_rejected(self):
        rows = make_scores(self.registry, self.entries, self.budget)
        for invalid in (rows[:-1], rows + [deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                select_ratio_tuning_policies(invalid, self.registry, self.budget, "e" * 64)
        with self.assertRaises(ValueError):
            build_full_prefix_budget(self.registry, self.feasibility[:-1], self.entries,
                                     self.summary)

    def test_file_mean_evidence_preserves_family_selection_in_matched_groups(self):
        import csv
        import json

        from tests.ghl_main.run_ratio_tuning import write_full_prefix_reports

        for entry in self.entries:
            entry["family"] = "A" if int(entry["series"]) <= 16 else "B"
        budget = build_full_prefix_budget(
            self.registry, self.feasibility, self.entries, self.summary,
        )
        rows = make_scores(self.registry, self.entries, budget)
        first, second = "c000000000001", "c000000000002"
        for row in rows:
            if row["model"] == "PaAno" and row["ratio"] == 100:
                scores = ((.1, .2, .6) if row["family"] == "A" else (.8, .9, 1))
                row["vus_pr"] = (scores if row["config_id"] == first else (.2, .5, .8))[row["seed"]]
        selection = select_ratio_tuning_policies(rows, self.registry, budget, "e" * 64)
        expected = {
            ("model_ratio", 18, first): (.6, 11 / 30, 2, True),
            ("model_ratio", 18, second): (.5, .5, 2, False),
            ("tier_adaptive", 12, first): (.6, .4, 2, True),
            ("tier_adaptive", 12, second): (.5, .5, 2, False),
            ("tier_adaptive", 6, first): (.3, .3, 1, False),
            ("tier_adaptive", 6, second): (.5, .5, 1, True),
        }
        audits = {(row["analysis_kind"], len(row["series_ids"]), row["config_id"]): row
                  for row in selection["candidate_audit"]
                  if row["model"] == "PaAno" and row["ratio"] == 100}
        self.assertEqual(audits.keys(), expected.keys())
        for key, (family_mean, file_mean, family_count, selected) in expected.items():
            row = audits[key]
            self.assertAlmostEqual(row["family_macro_vus_pr"], family_mean)
            self.assertAlmostEqual(row["series_macro_vus_pr"], file_mean)
            self.assertEqual((row["file_count"], row["family_count"]), (key[1], family_count))
            self.assertEqual(row["selected"], selected)

        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_full_prefix_reports(selection, budget, output, write_plots=False)
            for filename in ("candidate_audit.csv", "PaAno.csv"):
                with (output / filename).open(encoding="utf-8", newline="") as source:
                    exported = [row for row in csv.DictReader(source)
                                if row["model"] == "PaAno" and int(row["ratio"]) == 100]
                self.assertEqual(len(exported), 6 if filename == "candidate_audit.csv" else 2)
                for row in exported:
                    key = (row["analysis_kind"], len(json.loads(row["series_ids"])), row["config_id"])
                    family_mean, file_mean, family_count, _ = expected[key]
                    self.assertAlmostEqual(float(row["family_macro_vus_pr"]), family_mean)
                    self.assertAlmostEqual(float(row["series_macro_vus_pr"]), file_mean)
                    self.assertEqual((int(row["file_count"]), int(row["family_count"])),
                                     (key[1], family_count))

    def test_tier_joint_selection_uses_only_its_matched_files_and_training_families(self):
        rows = make_scores(self.registry, self.entries, self.budget)
        for row in rows:
            if row["ratio"] != 80:
                continue
            if row["model"] == "PaAno":
                row["vus_pr"] = .99 if int(row["series"]) <= 6 or row["family"] == "C" else .3
            elif row["model"] == "GDN":
                row["vus_pr"] = .1 if row["family"] == "C" else .8
        selection = select_ratio_tuning_policies(rows, self.registry, self.budget, "e" * 64)
        policies = [row for row in selection["tier_adaptive"] if row["tier"] == "t2" and row["ratio"] == 80]
        self.assertEqual(len(policies), 2)
        policy = next(row for row in policies if len(row["series_ids"]) == 12)
        self.assertEqual(policy["selected_model"], "GDN")
        fold = next(row for row in selection["adaptive_lofo"] if row["group_id"] == policy["group_id"]
                    and row["holdout_family"] == "C")
        self.assertEqual(fold["model"], "GDN")
        self.assertAlmostEqual(fold["holdout_vus_pr"], .1)

    def test_partial_manifest_accepts_missing_work_but_rejects_other_identity(self):
        from tests.ghl_main.run_ratio_tuning import _validate_partial_manifest

        execution = self.budget["execution_panel"][0]
        row = {**execution, "series": execution["series_ids"][0],
               "score_variant": execution["primary_score_variants"][0],
               "status": "complete", "budget_id": self.budget["budget_id"]}
        _validate_partial_manifest([row], self.budget)
        for invalid in ([row, row], [{**row, "budget_id": "old"}], [{**row, "series": "99"}]):
            with self.assertRaises(ValueError):
                _validate_partial_manifest(invalid, self.budget)

    def test_finish_resumes_partial_ledger_instead_of_treating_it_as_complete(self):
        from contextlib import ExitStack
        from tests.ghl_main import run_ratio_tuning as runner

        partial = [{"evaluator_sha256": "e", "ell_max_id": "l"}]
        manifest = [{"primary_score": "true"}]
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            evidence = MagicMock()
            evidence.__enter__.return_value = evidence
            stack.enter_context(patch.object(runner, "open_recommendation_evidence", return_value=evidence))
            (root / "dev18_trial_score_ledger.csv").write_text("partial\n", encoding="utf-8")
            for name in ("_require_clean_worktree", "_require_same_worktree", "prepare_tuning",
                         "_validate_primary_manifest_rows", "_write_csv"):
                stack.enter_context(patch.object(runner.tuning, name))
            stack.enter_context(patch.object(runner.tuning, "_git_head", return_value="a" * 40))
            stack.enter_context(patch.object(runner.tuning, "validate_vus_evidence", return_value={"evaluator_sha256": "e"}))
            stack.enter_context(patch.object(runner.tuning, "_validate_ell_max", return_value={"ell_max_id": "l"}))
            load = stack.enter_context(patch.object(runner.tuning, "load_trial_score_ledger", return_value=partial))
            build = stack.enter_context(patch.object(runner.tuning, "build_trial_score_ledger", return_value=partial))
            stack.enter_context(patch.object(runner.tuning, "_reuse_trial_scores", return_value={"key": partial[0]}))
            stack.enter_context(patch.object(runner, "select_ratio_tuning_policies", side_effect=RuntimeError("selection reached")))
            with self.assertRaisesRegex(RuntimeError, "selection reached"):
                runner.finish_ratio_tuning(self.registry, self.budget, manifest, [],
                                           {"result": root, "checkpoint": root / "cache",
                                            "recommendation": root / "recommendation_evidence"},
                                           data_root=root, write_plots=False)
            self.assertTrue(load.call_args.kwargs["allow_partial"])
            self.assertEqual(build.call_args.kwargs["reused_ledger"], partial)


if __name__ == "__main__":
    unittest.main()
