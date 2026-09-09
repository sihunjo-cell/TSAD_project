"""활성 모델의 source-only 합성 smoke runner를 검증한다."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestModelSmoke(unittest.TestCase):
    def test_official_smoke_accepts_scalar_and_native_channel_pairs(self):
        from tests.checks.run_model_smoke import run_full_prefix_smoke

        report = run_full_prefix_smoke(device="cpu", remote_execution=True)

        self.assertEqual(report["protocol"], "paper_tuning_v4")
        self.assertEqual(report["status"], "passed", report["models"])
        for name in ("PaAno", "GDN"):
            self.assertTrue(report["models"][name]["native_saved_scores"])

    def test_official_report_cannot_reuse_the_legacy_smoke_protocol(self):
        from tests.checks.run_model_smoke import validate_full_prefix_smoke

        identity = {"registry_sha256": "a" * 64, "source_sha256": {}, "methodology_revision": "paper_tuning_v4"}
        report = {"schema_version": 1, "protocol": "full_prefix_v2", "status": "passed",
                  "input_kind": "synthetic_only", **identity,
                  "models": {name: {"status": "passed", "registered_arguments": True,
                                    "native_saved_scores": True} for name in ("PaAno", "GDN")}}
        with tempfile.TemporaryDirectory() as directory, patch("tests.checks.run_model_smoke._full_prefix_identity", return_value=identity):
            path = Path(directory) / "smoke.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_full_prefix_smoke(path)
            report.update(schema_version=2, protocol="paper_tuning_v4")
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(validate_full_prefix_smoke(path)["status"], "passed")

    def test_full_prefix_smoke_rejects_local_cpu_and_missing_cuda_before_models(self):
        from tests.checks.run_model_smoke import run_full_prefix_smoke

        with self.assertRaisesRegex(RuntimeError, "원격"):
            run_full_prefix_smoke(device="cpu")
        with patch("tests.checks.run_model_smoke.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                run_full_prefix_smoke(device="cuda", remote_execution=True)

    def test_full_prefix_report_requires_current_code_and_all_models(self):
        from tests.checks.run_model_smoke import validate_full_prefix_smoke

        identity = {"registry_sha256": "a" * 64, "source_sha256": {"model.py": "b" * 64}}
        report = {"schema_version": 1, "protocol": "full_prefix_v2", "status": "passed",
                  "input_kind": "synthetic_only", **identity,
                  "models": {model: {"status": "passed"} for model in ("PaAno", "GDN")}}
        with tempfile.TemporaryDirectory() as directory, patch("tests.checks.run_model_smoke._full_prefix_identity", return_value=identity):
            path = Path(directory) / "smoke.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(validate_full_prefix_smoke(path)["status"], "passed")
            report["models"].pop("GDN")
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_full_prefix_smoke(path)

    def test_all_active_models_pass_without_labels_or_real_data(self):
        from tests.checks.run_model_smoke import run_source_only_smoke

        report = run_source_only_smoke()

        self.assertEqual(
            set(report["models"]),
            {
                "MWVAR", "SQDIFF_LAST1", "SQDIFF_LAST3", "SQDIFF_CENTERED5",
                "MWVAR96_SQDIFF_LAST3", "MWVAR96_SQDIFF_CENTERED5",
                "PCA_LEGACY", "PaAno", "GDN", "TimeRCD", "TSPulse",
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
