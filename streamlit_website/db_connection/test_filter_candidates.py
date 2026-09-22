"""원격에서 작은 입력과 임시 DB로 1차 후보 축소를 확인한다."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.model_registry import load_model_registry
from streamlit_website.db_connection.filter_candidates import (
    build_ml_input, filter_candidates, load_candidates, profile_normal_data,
)


class TestCandidateFilter(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "evidence.sqlite3"
        self.registry = load_model_registry()
        with sqlite3.connect(self.database) as connection:
            connection.executescript("""
                CREATE TABLE model_configs (config_id TEXT, model TEXT, settings_json TEXT);
            """)
            for model, settings in self.registry["models"].items():
                for candidate in settings["candidates"]:
                    saved = {**settings, **candidate, "common_recipe": self.registry["common_recipe"]}
                    connection.execute("INSERT INTO model_configs VALUES (?, ?, ?)",
                                       (candidate["config_id"], model, json.dumps(saved)))
            connection.execute("INSERT INTO model_configs VALUES ('old', 'ALoRa', '{}')")

    def test_registered_pool_expands_heads_and_filters_fixed_sensor_count(self):
        candidates = load_candidates(self.database)
        self.assertEqual(len(candidates), 45)
        self.assertNotIn("ALoRa", {row["model"] for row in candidates})
        self.assertEqual(sum(row["model"] == "TSPulse" for row in candidates), 12)
        rows = filter_candidates(candidates, channel_count=19)
        graph = [row for row in rows if row["model"] == "GDN"]
        self.assertEqual(sum(row["status"] == "eligible" for row in graph), 2)
        self.assertEqual(sum(row["status"] == "infeasible" for row in graph), 1)
        single = filter_candidates(candidates, channel_count=1)
        self.assertTrue(all(row["status"] == "infeasible" for row in single
                            if row["model"] in {"GDN", "TimeRCD"}))
        self.assertTrue(all(row["status"] == "eligible" for row in single
                            if row["model"] not in {"GDN", "TimeRCD"}))
        self.assertTrue(all(row["status"] == "eligible" for row in
                            filter_candidates(candidates, channel_count=30)))

    def test_data_growth_and_changing_features_do_not_reduce_plan_pool(self):
        candidates = load_candidates(self.database)
        expected = filter_candidates(candidates, channel_count=2)
        for frame in (
            pd.DataFrame({"first": [1.], "second": [1.]}),
            pd.DataFrame({"first": [np.nan] * 4, "second": [1.] * 4}),
            pd.DataFrame({"first": range(2000), "second": range(2000)}),
        ):
            summary, _ = profile_normal_data(frame, list(frame.columns))
            self.assertEqual(filter_candidates(candidates, channel_count=summary["input_column"]), expected)
        self.assertTrue(all(row["status"] == "eligible" for row in expected if row["model"] != "GDN"))

    def test_ml_input_keeps_full_prefix_model_state_and_registered_settings(self):
        candidates = load_candidates(self.database)
        current_model = {"model": "GDN", "settings": "config-1", "checkpoint": "model.pt",
                         "last_trained_at": "2026-09-22 14:00 KST"}
        conditions = {"operating_days": 365, "collection_rows_per_second": 1.0}
        frame = pd.DataFrame({"time": [1, 2, 3], "sensor": [5., 5., 6.]})
        rows = filter_candidates(candidates, channel_count=1)
        previous = build_ml_input(frame.iloc[:1], ["sensor"], rows,
                                  current_model=current_model, operating_conditions=conditions)
        updated = build_ml_input(frame, ["sensor"], rows,
                                 current_model=current_model, operating_conditions=conditions)
        self.assertEqual(previous["input"]["row_count"], 1)
        self.assertEqual(updated["input"]["row_count"], 3)
        self.assertEqual(updated["input"]["channel_count"], 1)
        self.assertEqual(updated["input"]["prefix_mode"], "cumulative_full")
        pd.testing.assert_frame_equal(updated["normal_prefix"], frame[["sensor"]])
        self.assertEqual(previous["candidates"], updated["candidates"])
        self.assertEqual(updated["current_model"], current_model)
        self.assertEqual(updated["operating_conditions"], conditions)
        self.assertTrue(all(row["status"] == "eligible" for row in updated["candidates"]))
        self.assertNotIn("GDN", {row["model"] for row in updated["candidates"]})
        candidate = updated["candidates"][0]
        self.assertEqual(candidate["settings"]["hyperparameters"], candidate["parameters"])
        self.assertIn("preprocess_recipe", candidate["settings"])
        self.assertIn("source_checkpoint_sha256", candidate["settings"])

    def test_missing_recipe_and_unknown_model_stay_unverified(self):
        with sqlite3.connect(self.database) as connection:
            connection.execute("UPDATE model_configs SET settings_json='{}' WHERE model='MWVAR'")
            connection.execute("INSERT INTO model_configs VALUES ('unknown', 'UnknownModel', '{}')")
        candidates = load_candidates(self.database)
        template = next(row for row in candidates if row["model"] == "GDN")
        candidates.extend([
            {**template, "model": "UnknownModel"},
            {**template, "parameters": {"window": 5}},
            {**template, "model": "TSPulse", "head": "unknown",
             "parameters": {"context_length": 512, "patch_size": 8, "aggregation_window": 64}},
        ])
        rows = filter_candidates(candidates, channel_count=40)
        self.assertTrue(all(row["status"] == "unverified" for row in rows
                            if row["model"] in {"UnknownModel", "MWVAR"}))
        self.assertTrue(all(row["status"] == "unverified" for row in rows[-3:]))
        self.assertTrue(any(row["status"] == "eligible" for row in rows))

    def test_features_use_only_selected_current_data_and_keep_undefined_values(self):
        frame = pd.DataFrame({"sensor": [1., 2., 3., 4.], "constant": [2.] * 4,
                              "Label": [0, 0, 0, 0]})
        summary, channels = profile_normal_data(frame, ["sensor", "constant"])
        self.assertEqual((summary["observed_row"], summary["input_column"]), (4, 2))
        self.assertAlmostEqual(channels[0]["std"], np.sqrt(1.25))
        self.assertAlmostEqual(channels[0]["acf_lag1"], 0.25)
        self.assertIsNone(channels[1]["spectral_entropy"])
        self.assertIsNone(summary["absolute_correlation_median"])
        self.assertEqual(summary["finite_value_fraction"], 1.)
        frame.loc[0, "sensor"] = np.nan
        summary, _ = profile_normal_data(frame, ["sensor", "constant"])
        self.assertLess(summary["finite_value_fraction"], 1.)

    def test_read_only_connection_does_not_create_missing_database(self):
        missing = self.database.with_name("missing.sqlite3")
        with self.assertRaises(sqlite3.OperationalError):
            load_candidates(missing)
        self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
