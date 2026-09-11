"""기존 실측과 추정 비교 시간을 구분하고 q 중복·타 모델 혼합을 막는다."""

import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.ghl_main import compare_execution_runtimes as comparison


def reference(name, config, seconds, *, threads=1, model="PCA_LEGACY"):
    return {
        "snapshot": {
            "spec": {"model": model, "config_id": config, "source_commit": "source", "seed": 0,
                     "common_recipe": {"methodology_revision": "paper_tuning_v4"},
                     "preprocess_recipe": {"fit_source": "full_evaluation"}},
            "input_identity": {"sha256": name}, "source_ranges": {"test_sessions": [[100, 300]]},
            "environment": {"packages": "same", "cuda_device": "L4"},
            "execution_resources": {"pca_fit_blas_threads_requested": threads},
        },
        "runtime_seconds": seconds,
        "reference": {"file": f"{name}-{config}-{threads}.json", "sha256": "sha",
                      "run_id": f"{name}-{config}-{threads}"},
    }


class TestRuntimeComparisons(unittest.TestCase):
    def compare(self, target, references):
        return comparison.compare_execution_runtime(
            target["snapshot"], target["runtime_seconds"], references, repository_root=Path("."),
        )

    def test_exact_serial_measurement_wins_and_other_models_keep_their_runtime(self):
        serial = reference("target", "A", 100)
        target = reference("target", "A", 25, threads=8)
        actual = self.compare(target, [serial])
        self.assertEqual(actual["comparison_runtime_seconds"], 100)
        self.assertEqual(actual["actual_runtime_seconds"], 25)
        self.assertEqual(actual["comparison_runtime_status"], "measured")
        self.assertEqual(actual["comparison_runtime_sources"], [serial["reference"]])
        for model in ("TSPulse", "PaAno", "GDN", "MWVAR"):
            with self.subTest(model=model):
                measured = self.compare(reference("target", "A", 7, model=model), [serial])
                self.assertEqual(measured["comparison_runtime_seconds"], 7)
                self.assertEqual(measured["comparison_runtime_basis"], "recorded_execution")

    def test_historical_configuration_ratios_fill_missing_serial_runtime(self):
        references = [reference("target", "A", 100),
                      reference("donor1", "A", 10), reference("donor1", "B", 20),
                      reference("donor2", "A", 20), reference("donor2", "B", 60),
                      reference("target", "B", 9999, model="TSPulse")]
        repeated = deepcopy(references[1])
        repeated["reference"]["run_id"] = "repeat"
        references.append(repeated)
        target = reference("target", "B", 40, threads=8)
        actual = self.compare(target, references)
        self.assertEqual(actual["comparison_runtime_seconds"], 250)
        self.assertEqual(actual["actual_runtime_seconds"], 40)
        self.assertEqual(actual["comparison_runtime_status"], "estimated")
        self.assertEqual(actual["comparison_runtime_basis"], "estimated_from_historical_pca_config_ratios")
        details = actual["comparison_runtime_details"]
        self.assertEqual(details["reference_prediction_min_seconds"], 200)
        self.assertEqual(details["reference_prediction_max_seconds"], 300)
        self.assertEqual(len(details["reference_predictions"]), 2)
        self.assertNotIn(references[5]["reference"], actual["comparison_runtime_sources"])

    def test_paired_thread_timings_precede_configuration_transfer(self):
        references = [reference("donor", "B", 120), reference("donor", "B", 30, threads=8),
                      reference("target", "A", 100), reference("donor", "A", 10)]
        actual = self.compare(reference("target", "B", 50, threads=8), references)
        self.assertEqual(actual["comparison_runtime_seconds"], 200)
        self.assertEqual(actual["comparison_runtime_basis"], "estimated_from_paired_pca_thread_timings")

    def test_sixteen_threads_use_only_matching_thread_pairs_and_keep_the_serial_reference(self):
        target = reference("target", "B", 20, threads=16)
        references = [reference("donor", "B", 120), reference("donor", "B", 30, threads=8)]
        self.assertIsNone(self.compare(target, references)["comparison_runtime_seconds"])
        references.append(reference("donor", "B", 15, threads=16))
        estimated = self.compare(target, references)
        self.assertEqual(estimated["comparison_runtime_seconds"], 160)
        self.assertEqual(estimated["actual_runtime_seconds"], 20)
        self.assertNotIn(references[1]["reference"], estimated["comparison_runtime_sources"])
        references.append(reference("target", "B", 150))
        measured = self.compare(target, references)
        self.assertEqual(measured["comparison_runtime_seconds"], 150)
        self.assertEqual(measured["comparison_runtime_status"], "measured")

    def test_missing_or_incompatible_references_do_not_scale_or_block_execution(self):
        target = reference("target", "A", 25, threads=8)
        serial = reference("target", "A", 100)
        for field, value in (("environment", {"cuda_device": "other"}),
                             ("source_ranges", {"test_sessions": [[100, 400]]})):
            changed = deepcopy(serial)
            changed["snapshot"][field] = value
            with self.subTest(field=field):
                actual = self.compare(target, [changed])
                self.assertIsNone(actual["comparison_runtime_seconds"])
                self.assertEqual(actual["actual_runtime_seconds"], 25)
        unknown = deepcopy(serial)
        unknown["snapshot"].pop("execution_resources")
        self.assertIsNone(self.compare(target, [unknown])["comparison_runtime_seconds"])
        with patch.object(comparison, "_legacy_single_thread", return_value=True):
            self.assertEqual(self.compare(target, [unknown])["comparison_runtime_seconds"], 100)

    def test_distance_parallelism_uses_its_own_pairs_but_keeps_historical_serial_evidence(self):
        target = reference("target", "B", 20, threads=16)
        target["snapshot"]["execution_resources"]["pca_distance_workers_requested"] = 16
        references = [reference("donor", "B", 120), reference("donor", "B", 60, threads=16)]
        self.assertIsNone(self.compare(target, references)["comparison_runtime_seconds"])
        paired = deepcopy(references[1])
        paired["snapshot"]["execution_resources"]["pca_distance_workers_requested"] = 16
        paired["runtime_seconds"] = 15
        paired["reference"]["run_id"] = "parallel-distances"
        references.append(paired)
        actual = self.compare(target, references)
        self.assertEqual(actual["comparison_runtime_seconds"], 160)
        self.assertNotIn(references[1]["reference"], actual["comparison_runtime_sources"])
        historical = [reference("target", "A", 100), reference("donor", "A", 10), references[0]]
        self.assertEqual(self.compare(target, historical)["comparison_runtime_seconds"], 1200)
        references.append(reference("target", "B", 150))
        self.assertEqual(self.compare(target, references)["comparison_runtime_seconds"], 150)
        self.assertEqual(self.compare(target, references)["comparison_runtime_status"], "measured")
        distance_only = deepcopy(target)
        distance_only["snapshot"]["execution_resources"]["pca_fit_blas_threads_requested"] = 1
        self.assertIsNone(self.compare(distance_only, [distance_only])["comparison_runtime_seconds"])

    def test_loader_uses_model_timing_and_deduplicates_only_completed_physical_runs(self):
        source = reference("target", "A", 10)
        record = {"run_id": "once", "identity": {"budget_id": "budget"}, "status": "complete",
                  "model_execution_complete": True, "run_snapshot": source["snapshot"],
                  "model_timing": {field: 2.0 for field in comparison.FULL_PREFIX_TIMING_FIELDS},
                  "model_execution_seconds": 999.0}
        with TemporaryDirectory() as directory:
            for name, status, identifier in (("first", "complete", "once"), ("q_copy", "complete", "once"),
                                             ("failed", "failed", "failed"), ("stopped", "interrupted", "stopped")):
                Path(directory, name + ".json").write_text(
                    json.dumps({**record, "status": status, "run_id": identifier}), encoding="utf-8",
                )
            loaded = comparison.load_runtime_references(directory, "budget")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["runtime_seconds"], 10)
            self.assertEqual(len(loaded[0]["reference"]["sha256"]), 64)
            self.assertEqual(comparison.load_runtime_references(directory, "other"), [])


if __name__ == "__main__":
    unittest.main()
