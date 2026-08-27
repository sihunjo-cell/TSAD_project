"""활성 모델의 source-only 합성 smoke runner를 검증한다."""

import unittest


class TestModelSmoke(unittest.TestCase):
    def test_all_active_models_pass_without_labels_or_real_data(self):
        from tests.checks.run_model_smoke import run_source_only_smoke

        report = run_source_only_smoke()

        self.assertEqual(
            set(report["models"]),
            {
                "MWVAR", "SQDIFF_LAST3", "PCA_LEGACY", "PaAno",
                "ALoRa", "GDN", "TimeRCD", "TSPulse",
            },
        )
        self.assertTrue(all(
            result["status"] == "passed" for result in report["models"].values()
        ))
        self.assertEqual(report["input_kind"], "synthetic_only")
        self.assertFalse(report["uses_labels"])
        self.assertFalse(report["downloads_checkpoints"])

if __name__ == "__main__":
    unittest.main()
