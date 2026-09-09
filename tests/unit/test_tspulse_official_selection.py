"""파일 평균 공통 창 선택 뒤 공식 family별 head 선택을 검증한다."""

import unittest

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from tests.ghl_main.select_ratio_tuning import select_ratio_tuning_policies, select_tspulse_official_heads


class TestTspulseOfficialSelection(unittest.TestCase):
    def setUp(self):
        self.registry = {"models": {"TSPulse": {
            "source_commit": "a" * 40,
            "candidates": [{"config_id": f"c{window}", "hyperparameters": {"aggregation_window": window}}
                           for window in (64, 96, 128)],
        }}}

    def select(self, rows):
        return select_tspulse_official_heads(rows, self.registry, budget_id="btest", evaluator_sha256="e" * 64)

    def rows(self, scores, *, ratio=100, families=("SMD",)):
        return [{"model": "TSPulse", "config_id": f"c{window}", "ratio": ratio,
                 "family": family, "series": f"{index + 1:02d}", "score_variant": head,
                 "vus_pr": scores.get(window, {}).get(head, [.1] * len(families))[index]}
                for window in (64, 96, 128) for head in ("time", "fft", "pred", "ensemble")
                for index, family in enumerate(families)]

    def test_selects_window_by_file_mean_before_family_heads_and_unseen_fallback(self):
        rows = self.rows({
            64: {"time": [.9, .9, 0], "fft": [.9, .9, 0],
                 "pred": [.95, .95, .2], "ensemble": [.2, .2, .8]},
            96: {"time": [.5, .5, .7], "fft": [.5, .5, .7]},
            128: {"pred": [1., 1., 1.], "ensemble": [1., 1., 1.]},
        }, families=("SMD", "SMD", "MSL"))
        selected = {row["family"]: row for row in self.select(rows)}
        self.assertEqual(selected["SMD"]["score_variant"], "pred")
        self.assertEqual(selected["SMD"]["official_mode"], "forecast")
        self.assertEqual(selected["SMD"]["tuning_file_count"], 2)
        self.assertEqual(selected["SMD"]["config_id"], "c64")
        self.assertEqual(selected["MSL"]["score_variant"], "ensemble")
        self.assertAlmostEqual(selected["SMD"]["window_scores"]["64"], .6)
        self.assertEqual(selected["SMD"]["window_rule_id"], "tspulse_file_mean_time_fft_v1")
        self.assertEqual(selected["SMD"]["window_series_ids"], ["01", "02", "03"])
        self.assertIsNone(selected["SMD"]["window_metric_decimal_places"])
        self.assertEqual(selected["GHL"]["score_variant"], "time")
        self.assertEqual(selected["GHL"]["selection_status"], "unseen_family_time_fallback")
        self.assertIsNone(selected["GHL"]["family_mean_vus_pr"])
        self.assertEqual(selected["HAI"]["score_variant"], "time")
        self.assertEqual(selected["HAI"]["selection_status"], "unseen_family_time_fallback")

    def test_rounds_per_file_as_official_csv_before_argmax_in_readme_order(self):
        selected = self.select(self.rows({96: {"time": [.500001], "fft": [.500002],
                                              "pred": [.4], "ensemble": [.3]}}))
        family = next(row for row in selected if row["family"] == "SMD")
        self.assertEqual(family["score_variant"], "time")
        self.assertEqual(family["time_vus_pr"], .5)
        self.assertEqual(family["fft_vus_pr"], .5)
        self.assertEqual(family["head_order"], ["time", "fft", "pred", "ensemble"])

    def test_selects_each_logical_ratio_independently_and_breaks_window_ties_numerically(self):
        rows = self.rows({}, ratio=5)
        rows += self.rows({128: {"time": [.8], "fft": [.8]}}, ratio=100)
        selected = {row["ratio"]: row for row in self.select(rows) if row["family"] == "SMD"}
        self.assertEqual(selected[5]["aggregation_window"], 64)
        self.assertEqual(selected[100]["aggregation_window"], 128)

    def test_rejects_missing_head_files_and_duplicate_seed_mean_rows(self):
        rows = self.rows({})
        with self.assertRaisesRegex(ValueError, "파일 구성"):
            self.select(rows[:-1])
        with self.assertRaisesRegex(ValueError, "고유 키"):
            self.select(rows + rows[:1])

    def test_tier_groups_consume_the_same_global_window_as_model_policy(self):
        registry = {"common_recipe": {"methodology_revision": "paper_tuning_v4"}, "models": {
            "TSPulse": {**self.registry["models"]["TSPulse"], "tier": "t3", "source_checkpoint_sha256": "b" * 64},
            "TimeRCD": {"tier": "t3", "source_commit": "a" * 40, "source_checkpoint_sha256": "b" * 64,
                        "candidates": [{"config_id": "crcd", "hyperparameters": {}}]},
        }}
        support = [{"series": series, "family": family, "observed_row": 100,
                    "training_boundary": 100, "feature_count": channels}
                   for series, family, channels in (("01", "A", 1), ("02", "A", 1), ("03", "B", 2))]
        panels, executions, rows = [], [], []
        for model, details in registry["models"].items():
            configs = [candidate["config_id"] for candidate in details["candidates"]]
            variants = ["time", "fft", "pred", "ensemble"] if model == "TSPulse" else [""]
            available = support if model == "TSPulse" else support[2:]
            groups = []
            for ratio in SUPPORTED_RATIO_PERCENTS:
                for members, candidate_ids in ((available, configs), (support[:2], [])):
                    if model == "TSPulse" and not candidate_ids:
                        continue
                    groups.append({"group_id": f"{model}_{ratio}_{bool(candidate_ids)}", "model": model,
                                   "tier": "t3", "ratio": ratio, "series_ids": [item["series"] for item in members],
                                   "candidate_ids": candidate_ids,
                                   "support": [{**item, "observed_row": ratio} for item in members]})
            panels.append({"model": model, "tier": "t3", "groups": groups, "primary_score_variants": variants})
            executions.extend({"model": model, "config_id": config, "logical_ratios": list(SUPPORTED_RATIO_PERCENTS),
                               "seed": 0, "series_ids": [item["series"] for item in available],
                               "primary_score_variants": variants} for config in configs)
        for ratio in SUPPORTED_RATIO_PERCENTS:
            rows.extend({**row, "tier": "t3", "seed": 0, "status": "complete"} for row in self.rows({
                64: {"time": [.9, .9, 0.], "fft": [.9, .9, 0.], "pred": [1., 1., .8]},
                96: {"time": [0., 0., .9], "fft": [0., 0., .9], "ensemble": [.1, .1, 1.]},
            }, ratio=ratio, families=("A", "A", "B")))
            rows.append({"model": "TimeRCD", "tier": "t3", "config_id": "crcd", "ratio": ratio,
                         "seed": 0, "series": "03", "family": "B", "score_variant": "",
                         "vus_pr": .1, "status": "complete"})
        budget = {"experiment_mode": "full_prefix_v2", "schema_version": 3,
                  "primary_hpo_regime": "full_prefix_per_ratio", "budget_id": "btest",
                  "tie_rule": {"tolerance": 1e-6}, "series_ids": ["01", "02", "03"],
                  "model_panels": panels, "execution_panel": executions}
        selection = select_ratio_tuning_policies(rows, registry, budget, "e" * 64)
        policies = [row for kind in ("model_ratio", "tier_adaptive") for row in selection[kind]
                    if row["model"] == "TSPulse"]
        self.assertTrue(policies)
        self.assertEqual({row["config_id"] for row in policies}, {"c64"})
        singleton = next(row for row in selection["tier_adaptive"] if row["series_ids"] == ["03"])
        self.assertEqual(singleton["score_variant_by_family"]["B"], "pred")
        self.assertAlmostEqual(singleton["family_macro_vus_pr"], .8)


if __name__ == "__main__":
    unittest.main()
