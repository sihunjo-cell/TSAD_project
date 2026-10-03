"""작은 손계산 입력으로 보고서가 결론·일정을 담고 입력값을 escape하는지 확인한다."""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from streamlit_website.DP.planner import optimize_plan
from streamlit_website.report.build_report import build_report_html


class TestBuildReport(unittest.TestCase):
    def test_report_shows_action_schedule_and_escapes_input(self):
        payload = {
            "metric": "VUS-PR", "cost_unit": "seconds", "budget": None,
            "candidates": [{"candidate_id": "a::", "model": "MWVAR", "config_id": "a", "head": ""},
                           {"candidate_id": "b::", "model": "PaAno", "config_id": "b", "head": ""}],
            "stages": [{"stage": 0, "weight": 10.0, "elapsed_days": 0.0, "ratio_percent": 5,
                        "available_rows": 100, "inference_rows": 1000.0},
                       {"stage": 1, "weight": 20.0, "elapsed_days": 10.0, "ratio_percent": 10,
                        "available_rows": 200, "inference_rows": 2000.0}],
            "training_options": [{"stage": 0, "candidate_id": "a::", "predicted_performance": 0.3, "training_cost": 0.0},
                                 {"stage": 0, "candidate_id": "b::", "predicted_performance": 0.5, "training_cost": 5.0}],
            "inference_options": [{"stage": stage, "candidate_id": candidate, "trained_stage": 0, "inference_cost": cost}
                                  for stage in (0, 1) for candidate, cost in (("a::", 0.1), ("b::", 1.0))],
            "current_checkpoint": None,
        }
        ml_input = {
            "input": {"row_count": 100, "channel_count": 2, "sensor_columns": ["s1", "s2"],
                      "source_name": "<script>alert(1)</script>.csv"},
            "operating_conditions": {"operating_days": 30, "collection_rows_per_second": 0.0001,
                                     "inference_rows_per_day": 100, "gpu_memory_gib": 24.0},
            "current_model": {}, "excluded_candidates": [],
            "candidates": [{"config_id": "a", "head": "", "model": "MWVAR", "target_use": "training_free",
                            "status": "eligible"},
                           {"config_id": "b", "head": "", "model": "PaAno", "target_use": "fit_full_prefix",
                            "status": "eligible"}],
        }
        ml_output = SimpleNamespace(similarity_matches=[], stage_candidates_excluded={}, metadata={})
        html = build_report_html(payload=payload, result=optimize_plan(payload), ml_input=ml_input, ml_output=ml_output,
                                 generated_at=datetime(2026, 10, 3, 15, 2, tzinfo=timezone(timedelta(hours=9))))
        self.assertIn("PaAno 최초 도입", html)
        self.assertIn("5 → 10%", html)
        self.assertIn("계획 번호 P-20261003-1502", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;.csv", html)
        self.assertNotIn("<script>alert(1)", html)


if __name__ == "__main__":
    unittest.main()
