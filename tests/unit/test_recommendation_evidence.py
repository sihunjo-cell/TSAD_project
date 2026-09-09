"""원격에서 prefix 캐시·SQL 연결·원자 내보내기를 검사한다."""

import csv
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import closing
from unittest import mock

import numpy

from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS
from tests.ghl_main import store_recommendation_evidence as evidence


def fixture():
    entries = [dict(series=f"{index:02d}", name=f"input_{index:02d}.csv", source_directory="tuning", family="synthetic",
                    training_boundary=40, feature_count=2, sha256=f"{index:064x}",
                    feature_names_sha256=evidence._digest(["sensor", "sensor"])) for index in range(1, 19)]
    registry = {"common_recipe": {"training_split": "full_prefix_v2"}, "models": {
        "TSPulse": {"target_use": "strict_zero_shot", "candidates": [
            {"config_id": "cActual", "hyperparameters": {"aggregation_window": 64}},
            {"config_id": "cExcluded", "hyperparameters": {"aggregation_window": 128}},
        ]},
    }}
    budget = {"budget_id": "bNew", "budget_sha256": "b" * 64, "expected_ledger_rows": 252,
              "execution_panel": [{"model": "TSPulse", "config_id": "cActual", "seed": 0,
                  "physical_ratio": 100, "logical_ratios": list(SUPPORTED_RATIO_PERCENTS),
                  "series_ids": [entry["series"] for entry in entries],
                  "primary_score_variants": ["time", "fft"], "diagnostic_score_variants": ["raw_max"],
                  "training_group_id": "tShared"}],
              "structural_exclusions": [{"series": entry["series"], "model": "TSPulse", "config_id": "cExcluded", "ratio": ratio,
                  "status_reason": "structural_constraint"} for entry in entries for ratio in SUPPORTED_RATIO_PERCENTS]}
    identity = {"project_commit": "new_source", "budget_sha256": budget["budget_sha256"], "schema_version": evidence.SCHEMA_VERSION,
                "feasibility_ledger": {"file": "fixture_feasibility.csv", "sha256": "f" * 64}}
    return entries, registry, budget, identity


def loader(entry):
    return {"normal_training": numpy.arange(80, dtype=numpy.float32).reshape(40, 2),
            "feature_names": ["sensor", "sensor"], "input_identity": {"sha256": entry["sha256"]}}


def completed_rows(entries, budget):
    manifest, ledger = [], []
    panel = budget["execution_panel"][0]
    primary = panel["primary_score_variants"]
    for entry in entries:
        for variant in primary + panel["diagnostic_score_variants"]:
            row = {"series": entry["series"], "model": "TSPulse", "config_id": "cActual", "physical_ratio": 100,
                   "seed": 0, "score_variant": variant, "primary_score": str(variant in primary).lower(),
                   "budget_id": budget["budget_id"], "status": "complete", "status_reason": "",
                   "score_file": f"score_{entry['series']}_{variant}.npy", "score_sha256": "a" * 64,
                   "metadata_file": f"metadata_{entry['series']}_{variant}.json", "metadata_sha256": "d" * 64}
            manifest.append(row)
            if variant in primary:
                ledger.extend({**row, "ratio": ratio, "vus_pr": 0.25 if variant == "time" else 0.75,
                               "normalization": "trainnorm", "evaluator_sha256": "e" * 64, "ell_max_id": "ellNew"}
                              for ratio in SUPPORTED_RATIO_PERCENTS)
    return manifest, ledger


def write_source(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified_metadata(row):
    return {"execution_attempt": {"run_id": f"attempt_{row['series']}"}}


class TestRecommendationEvidence(unittest.TestCase):
    def test_database_verifier_rejects_missing_tspulse_numeric_calibration(self):
        from src.common.execution_evidence import FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, FULL_PREFIX_STORAGE_SCHEMA_VERSION

        settings = {"model": "TSPulse", "config_id": "cActual", "target_use": "strict_zero_shot",
                    "common_recipe": {"methodology_revision": "paper_tuning_v4"}}
        identity = {"input_manifest_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.json"
            snapshot.write_text(json.dumps({"spec": {**settings, "ratio": 100, "seed": 0,
                "dataset_role": "development", "split_role": "dev18_selection", **identity}}), encoding="utf-8")
            score = root / "score.npy"
            numpy.save(score, numpy.arange(10, dtype=float))
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"dataset": "DEV18", "series": 1, "model": "TSPulse",
                "config_id": "cActual", "seed": 0, "ratio": 100, "score_variant": "time",
                "run_snapshot": {"file": snapshot.name}, "training_files": {},
                "storage_schema_version": FULL_PREFIX_STORAGE_SCHEMA_VERSION,
                "execution_evidence": {"measurement_protocol_id": FULL_PREFIX_MEASUREMENT_PROTOCOL_ID}}), encoding="utf-8")
            row = {"series": "01", "model": "TSPulse", "config_id": "cActual", "physical_ratio": 100,
                   "seed": 0, "score_variant": "time", "score_file": score.name, "metadata_file": metadata.name,
                   "score_sha256": evidence.file_sha256(score), "metadata_sha256": evidence.file_sha256(metadata)}
            connection = mock.Mock()
            connection.execute.return_value.fetchone.return_value = [json.dumps(settings)]
            store = SimpleNamespace(connection=connection, identity=identity, _verified_execution_files={},
                entries={"01": {"row_count": 20, "training_boundary": 10, "feature_count": 2}})
            with mock.patch.object(evidence, "REPOSITORY_ROOT", root), mock.patch(
                "src.common.execution_evidence.validate_execution_evidence_for_run", return_value={}), \
                    self.assertRaisesRegex(ValueError, "TSPulse native_calibration"):
                evidence.RecommendationEvidence._verify_execution_files(store, row)

    def test_recommendation_inputs_are_invariant_to_each_channel_unit(self):
        entries, registry, budget, identity = fixture()
        original = numpy.column_stack((numpy.tile([0, 0, 0, 0, 4], 8), numpy.arange(40))).astype(numpy.float32)
        transformed = original * numpy.array([-3, 8], dtype=numpy.float32) + numpy.array([1024, -256], dtype=numpy.float32)
        with tempfile.TemporaryDirectory() as temporary:
            inputs, raw = [], []
            for name, values in (("original", original), ("transformed", transformed)):
                with evidence.RecommendationEvidence(Path(temporary) / name, identity=identity,
                                                     registry=registry, budget=budget, entries=entries) as store:
                    store.bind_environment({"backend": "remote-test"})
                    store.prepare_features(lambda entry: {**loader(entry), "normal_training": values})
                    inputs.append(dict(store.connection.execute(
                        "SELECT * FROM recommendation_inputs WHERE series='01' AND q_percent=100").fetchone()))
                    raw.append(store.connection.execute(
                        "SELECT channel_std_median FROM prefix_features WHERE series='01' AND q_percent=100").fetchone()[0])
                    short = store.connection.execute(
                        "SELECT channel_interquartile_range_std_ratio_median,"
                        "channel_interquartile_range_std_ratio_valid_channel_count "
                        "FROM recommendation_inputs WHERE series='01' AND q_percent=5").fetchone()
                    self.assertEqual(tuple(short), (1.0, 1))
            self.assertNotEqual(raw[0], raw[1])
            for field in evidence.RECOMMENDATION_FEATURE_COLUMNS:
                self.assertAlmostEqual(inputs[0][field], inputs[1][field], msg=field)
            for field in evidence.RECOMMENDATION_QUALITY_COLUMNS:
                self.assertEqual(inputs[0][field], inputs[1][field], field)
            expected_shape = 0.5 * 19.5 / numpy.std(numpy.arange(40), ddof=0)
            self.assertAlmostEqual(inputs[0]["channel_interquartile_range_std_ratio_median"], expected_shape)
            self.assertEqual(inputs[0]["channel_interquartile_range_std_ratio_valid_channel_count"], 2)
            self.assertNotIn("channel_std_median", inputs[0])
            self.assertNotIn("channel_interquartile_range_median", inputs[0])

    def test_recommendation_inputs_keep_missingness_and_reject_changed_view(self):
        entries, registry, budget, identity = fixture()
        values = numpy.column_stack((numpy.ones(40), numpy.full(40, numpy.nan))).astype(numpy.float32)
        with tempfile.TemporaryDirectory() as temporary:
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                store.bind_environment({"backend": "remote-test"})
                store.prepare_features(lambda entry: {**loader(entry), "normal_training": values})
                row = dict(store.connection.execute(
                    "SELECT * FROM recommendation_inputs WHERE series='01' AND q_percent=100").fetchone())
                self.assertEqual(tuple(row), evidence.RECOMMENDATION_HEADERS)
                self.assertEqual(row["family"], "synthetic")
                self.assertEqual(row["constant_channel_fraction"], 1.0)
                self.assertEqual(row["constant_evaluable_channel_fraction"], 0.5)
                self.assertEqual(row["finite_value_fraction"], 0.5)
                self.assertIsNone(row["channel_interquartile_range_std_ratio_median"])
                self.assertEqual(row["channel_interquartile_range_std_ratio_median_reason"], "no_valid_channel_ratios")
                self.assertEqual(row["channel_interquartile_range_std_ratio_valid_channel_count"], 0)
                self.assertEqual(row["absolute_correlation_valid_pair_fraction"], 0.0)
                self.assertIsNone(row["channel_spectral_entropy_median"])
                self.assertEqual(row["channel_spectral_entropy_valid_channel_fraction"], 0.0)
                self.assertEqual(row["channel_spectral_entropy_median_reason"], "no_valid_values")
                final = store.finalize(require_complete=False)
                export = next(item for item in final["exports"] if Path(item["file"]).name == "recommendation_inputs.csv")
                self.assertEqual(export["rows"], 126)
                self.assertEqual(final["recommendation_input_contract"], store.contract["recommendation_input_contract"])
                with Path(export["file"]).open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(tuple(reader.fieldnames), evidence.RECOMMENDATION_HEADERS)
                    exported = next(item for item in reader if item["prefix_feature_id"] == row["prefix_feature_id"])
                self.assertEqual(exported["channel_spectral_entropy_median"], "")
                self.assertEqual(exported["family"], "synthetic")
                self.assertEqual(exported["channel_spectral_entropy_valid_channel_fraction"], "0.0")
                self.assertEqual(exported["channel_spectral_entropy_median_reason"], "no_valid_values")
                with store.connection:
                    store.connection.execute("DROP VIEW recommendation_inputs")
                    store.connection.execute("CREATE VIEW recommendation_inputs AS SELECT * FROM prefix_features")
                with self.assertRaisesRegex(ValueError, "recommendation input view"):
                    store.require_features()

    def test_extended_features_survive_export_resume_and_detect_corruption(self):
        entries, registry, budget, identity = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                store.bind_environment({"backend": "remote-test"})
                store.prepare_features(loader)
                prefix = store.connection.execute(
                    "SELECT * FROM prefix_features WHERE series='01' AND q_percent=10").fetchone()
                channel = store.connection.execute(
                    "SELECT * FROM channel_features WHERE prefix_feature_id=? AND channel_index=0",
                    (prefix["prefix_feature_id"],)).fetchone()
                expected = {"interquartile_range": 3.0, "difference_q90_iqr_ratio": 2 / 3,
                            "median_shift_iqr_ratio": 4 / 3, "spectral_entropy": 0.9182958340544896}
                for field, value in expected.items():
                    self.assertAlmostEqual(channel[field], value)
                    self.assertIsNone(channel[field + "_reason"])
                    self.assertAlmostEqual(prefix[f"channel_{field}_median"], value)
                    self.assertIsNone(prefix[f"channel_{field}_median_reason"])
                    self.assertEqual(prefix[f"channel_{field}_valid_channel_count"], 2)
                short = store.connection.execute(
                    "SELECT channel_spectral_entropy_median,channel_spectral_entropy_valid_channel_count "
                    "FROM prefix_features WHERE series='01' AND q_percent=5").fetchone()
                self.assertEqual(tuple(short), (None, 0))
                final = store.finalize(require_complete=False)
                for filename, original_headers, fields in (
                    ("prefix_features.csv", ("prefix_feature_id", "csv_file", "q_percent", "observed_row",
                        "상수채널수", "채널std median", "채널 ACF lag1 median", "채널간 절대 상관 중앙값"),
                        {f"channel_{field}_median": value for field, value in expected.items()}),
                    ("channel_features.csv", ("prefix_feature_id", "channel(col)", "mean", "std", "median", "ACF lag1"),
                        expected),
                ):
                    export = next(item for item in final["exports"] if Path(item["file"]).name == filename)
                    with Path(export["file"]).open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.DictReader(stream)
                        self.assertEqual(tuple(reader.fieldnames[:len(original_headers)]), original_headers)
                        exported = next(row for row in reader if row["prefix_feature_id"] == prefix["prefix_feature_id"])
                    for field, value in fields.items():
                        self.assertAlmostEqual(float(exported[field]), value)
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as resumed:
                resumed.prepare_features(mock.Mock(side_effect=AssertionError("cached loader called")))
                with self.assertRaises(sqlite3.IntegrityError), resumed.connection:
                    resumed.connection.execute("UPDATE channel_features SET spectral_entropy=1.5")
                with resumed.connection:
                    resumed.connection.execute("UPDATE channel_features SET median_shift_iqr_ratio=99")
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    resumed.require_features()

    def test_obsolete_schema_is_rejected_before_creating_database(self):
        entries, registry, budget, identity = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            for version in (1, 2):
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "schema version"):
                    evidence.RecommendationEvidence(temporary, identity={**identity, "schema_version": version},
                                                    registry=registry, budget=budget, entries=entries)
            self.assertFalse((Path(temporary) / "recommendation.sqlite3").exists())

    def test_paper_ensemble_is_scored_and_exported_for_all_q(self):
        entries, registry, budget, identity = fixture()
        registry["common_recipe"]["methodology_revision"] = "paper_tuning_v4"
        budget["execution_panel"][0].update(primary_score_variants=["time", "fft", "pred", "ensemble"],
                                            diagnostic_score_variants=[])
        budget["expected_ledger_rows"] = 18 * 7 * 4
        manifest, ledger = completed_rows(entries, budget)
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, ledger_path = Path(temporary) / "manifest.csv", Path(temporary) / "ledger.csv"
            write_source(manifest_path, manifest)
            write_source(ledger_path, ledger)
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                store.bind_environment({"backend": "remote-test"})
                store.prepare_features(loader)
                with mock.patch.object(store, "_verify_execution_files", side_effect=verified_metadata):
                    store.sync_results(manifest, ledger_rows=ledger, manifest_path=manifest_path, ledger_path=ledger_path)
                    final = store.finalize()
                self.assertEqual(final["status"], "complete")
                self.assertEqual(store.connection.execute(
                    "SELECT count(vus_pr) FROM results WHERE score_variant='ensemble'").fetchone()[0], 18 * 7)
                self.assertEqual(len([row for row in final["exports"] if "__ensemble.csv" in row["file"]]), 18)

    def test_prepare_cache_resume_and_source_rejection(self):
        entries, registry, budget, identity = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                self.assertFalse(store.status()["feature_complete"])
                with self.assertRaisesRegex(ValueError, "environment"):
                    store.prepare_features(loader)
                store.bind_environment({"backend": "remote-test"})
                with mock.patch.object(evidence, "compute_prefix_features", wraps=evidence.compute_prefix_features) as extractor:
                    store.prepare_features(loader)
                    store.prepare_features(mock.Mock(side_effect=AssertionError("cached loader called")))
                    self.assertEqual(extractor.call_count, 126)
                self.assertEqual(store.status()["completed_channel_count"], 252)
                self.assertEqual(store.status()["completed_exclusion_count"], 126)
                self.assertEqual(store.connection.execute("SELECT count(*) FROM results").fetchone()[0], 0)
                self.assertEqual(store.connection.execute("SELECT typeof(q_percent),typeof(channel_std_median) FROM prefix_features LIMIT 1").fetchone()[:], ("integer", "real"))
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute("INSERT INTO channel_features SELECT * FROM channel_features LIMIT 1")
                with self.assertRaises(sqlite3.IntegrityError):
                    store.connection.execute("UPDATE channel_features SET prefix_feature_id='missing' WHERE channel_index=0")
                with self.assertRaisesRegex(ValueError, "environment"):
                    store.bind_environment({"backend": "another"})
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as resumed:
                resumed.prepare_features(mock.Mock(side_effect=AssertionError("resume loader called")))
            with self.assertRaisesRegex(ValueError, "another experiment"):
                evidence.RecommendationEvidence(temporary, identity={**identity, "project_commit": "different"},
                                                registry=registry, budget=budget, entries=entries)

    def test_heads_ledger_reconciliation_atomic_backup_and_exports(self):
        entries, registry, budget, identity = fixture()
        manifest, ledger = completed_rows(entries, budget)
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, ledger_path = Path(temporary) / "manifest.csv", Path(temporary) / "ledger.csv"
            write_source(manifest_path, manifest)
            write_source(ledger_path, ledger)
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                store.bind_environment({"backend": "remote-test"})
                store.prepare_features(loader)
                with mock.patch.object(store, "_verify_execution_files", side_effect=verified_metadata):
                    store.sync_results(manifest, manifest_path=manifest_path)
                    self.assertFalse(store.status()["scoring_complete"])
                    self.assertTrue(store.status()["execution_complete"])
                    self.assertEqual(store.connection.execute("SELECT count(vus_pr) FROM results").fetchone()[0], 0)
                    partial = store.finalize(require_complete=False)
                    self.assertEqual(partial["status"], "incomplete")
                    store.sync_results(manifest, ledger_rows=ledger, manifest_path=manifest_path, ledger_path=ledger_path)
                    store.sync_results(list(reversed(manifest)), ledger_rows=list(reversed(ledger)), manifest_path=manifest_path, ledger_path=ledger_path)
                    self.assertEqual(store.status()["completed_result_count"], 378)
                    with self.assertRaises(sqlite3.IntegrityError), store.connection:
                        store.connection.execute("UPDATE results SET score_file='' WHERE status='complete'")
                    with self.assertRaises(sqlite3.IntegrityError), store.connection:
                        store.connection.execute("UPDATE results SET evaluator_sha256=' ' WHERE status='complete'")
                    bad = [{**ledger[0], "vus_pr": 0.9}]
                    with self.assertRaisesRegex(ValueError, "differs"):
                        store.sync_results([manifest[0]], ledger_rows=bad, manifest_path=manifest_path, ledger_path=ledger_path)
                with mock.patch.object(store, "_verify_execution_files", side_effect=verified_metadata):
                    final = store.finalize()
                self.assertEqual(final["status"], "complete")
                self.assertEqual(final["missing_primary_result_count"], 0)
                self.assertEqual(len([item for item in final["exports"] if "performance" in item["file"]]), 54)
                for item in final["exports"]:
                    path = Path(item["file"])
                    self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.reader(stream)
                        headers = tuple(next(reader))
                        rows = list(reader)
                    self.assertEqual(len(rows), item["rows"])
                    if "performance" in str(path):
                        self.assertEqual(headers, evidence.PERFORMANCE_HEADERS)
                        self.assertEqual({row[0] for row in rows}, {path.name.split("__")[0]})
                        if "__raw_max" in str(path):
                            self.assertTrue(all(row[-1] == "" and row[-2] == "executed_unscored" for row in rows))
                with closing(sqlite3.connect(final["database"]["file"])) as backup:
                    self.assertEqual(backup.execute("SELECT count(*) FROM results").fetchone()[0], 378)
                    self.assertEqual(backup.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                original_receipt = Path(final["receipt_file"]).read_bytes()
                with mock.patch.object(store, "_verify_execution_files", side_effect=verified_metadata), \
                        mock.patch.object(store, "_export_csv", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        store.finalize()
                self.assertEqual(Path(final["receipt_file"]).read_bytes(), original_receipt)
                with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                     budget=budget, entries=entries) as resumed, \
                        mock.patch.object(resumed, "_verify_execution_files", side_effect=verified_metadata):
                    retried = resumed.finalize()
                self.assertEqual(retried["status"], "complete")
                self.assertEqual(retried["completed_result_count"], 378)
                self.assertEqual(retried["table_fingerprints"], final["table_fingerprints"])
                self.assertEqual(
                    [(Path(item["file"]).name, item["sha256"], item["rows"]) for item in retried["exports"]],
                    [(Path(item["file"]).name, item["sha256"], item["rows"]) for item in final["exports"]],
                )
                with store.connection:
                    store.connection.execute("""UPDATE results SET metadata_sha256='changed' WHERE score_variant='raw_max'
                        AND prefix_feature_id=(SELECT prefix_feature_id FROM prefix_features WHERE series='01' AND q_percent=5)""")
                with self.assertRaisesRegex(ValueError, "shared physical"):
                    store.finalize()

    def test_partial_transaction_is_recoverable_and_corruption_is_rejected(self):
        entries, registry, budget, identity = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                store.bind_environment({"backend": "remote-test"})
                original = evidence.compute_prefix_features
                calls = 0

                def interrupt_after_one(*arguments):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise OSError("collection interrupted")
                    return original(*arguments)

                with mock.patch.object(evidence, "compute_prefix_features", side_effect=interrupt_after_one):
                    with self.assertRaises(OSError):
                        store.prepare_features(loader)
                self.assertEqual(store.status()["completed_prefix_count"], 1)
                with mock.patch.object(evidence, "compute_prefix_features", wraps=original) as extractor:
                    store.prepare_features(loader)
                    self.assertEqual(extractor.call_count, 125)
                with store.connection:
                    store.connection.execute("UPDATE channel_features SET mean=999 WHERE channel_index=0")
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    store.require_features()

    def test_channel_write_failure_rolls_back_only_the_unfinished_prefix(self):
        entries, registry, budget, identity = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                store.bind_environment({"backend": "remote-test"})
                store.connection.execute("""CREATE TEMP TRIGGER interrupt_channels
                    BEFORE INSERT ON channel_features
                    WHEN NEW.channel_index=1 AND NEW.prefix_feature_id=(
                        SELECT prefix_feature_id FROM prefix_features WHERE series='01' AND q_percent=10)
                    BEGIN SELECT RAISE(ABORT, 'channel write interrupted'); END""")
                with self.assertRaisesRegex(sqlite3.IntegrityError, "channel write interrupted"):
                    store.prepare_features(loader)
                self.assertEqual(store.status()["completed_prefix_count"], 1)
                self.assertEqual(store.status()["completed_channel_count"], 2)
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as resumed:
                state = resumed.prepare_features(loader)
                self.assertTrue(state["feature_complete"])
                self.assertEqual(state["completed_prefix_count"], 126)
                self.assertEqual(state["completed_channel_count"], 252)

    def test_result_write_failure_preserves_pending_rows_and_resume_is_idempotent(self):
        entries, registry, budget, identity = fixture()
        manifest, ledger = completed_rows(entries, budget)
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, ledger_path = Path(temporary) / "manifest.csv", Path(temporary) / "ledger.csv"
            write_source(manifest_path, manifest)
            write_source(ledger_path, ledger)
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as store:
                store.bind_environment({"backend": "remote-test"})
                store.prepare_features(loader)
                with mock.patch.object(store, "_verify_execution_files", side_effect=verified_metadata):
                    store.sync_results(manifest, manifest_path=manifest_path)
                    store.connection.execute("""CREATE TEMP TRIGGER interrupt_results
                        BEFORE INSERT ON results WHEN NEW.score_variant='fft' AND NEW.status='complete'
                        BEGIN SELECT RAISE(ABORT, 'result write interrupted'); END""")
                    with self.assertRaisesRegex(sqlite3.IntegrityError, "result write interrupted"):
                        store.sync_results(manifest, ledger_rows=ledger, manifest_path=manifest_path,
                                           ledger_path=ledger_path)
                self.assertEqual(store.status()["completed_result_count"], 378)
                self.assertEqual(store.connection.execute("SELECT count(vus_pr) FROM results").fetchone()[0], 0)
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as resumed, \
                    mock.patch.object(resumed, "_verify_execution_files", side_effect=verified_metadata):
                for _ in range(2):
                    resumed.sync_results(manifest, ledger_rows=ledger, manifest_path=manifest_path,
                                         ledger_path=ledger_path)
                resumed.sync_results(manifest, manifest_path=manifest_path)
                self.assertTrue(resumed.status()["recommendation_complete"])
                self.assertEqual(resumed.status()["recorded_result_count"], 378)
                self.assertEqual(resumed.connection.execute("SELECT count(vus_pr) FROM results").fetchone()[0], 252)

    def test_interrupted_schema_creation_rolls_back(self):
        entries, registry, budget, identity = fixture()
        original_connect = sqlite3.connect

        class InterruptIdentityConnection(sqlite3.Connection):
            def execute(self, statement, parameters=()):
                if statement.startswith("INSERT INTO experiment_identity"):
                    raise OSError("identity write interrupted")
                return super().execute(statement, parameters)

        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(evidence.sqlite3, "connect", side_effect=lambda path:
                                   original_connect(path, factory=InterruptIdentityConnection)):
                with self.assertRaises(OSError):
                    evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                    budget=budget, entries=entries)
            with evidence.RecommendationEvidence(temporary, identity=identity, registry=registry,
                                                 budget=budget, entries=entries) as resumed:
                self.assertEqual(resumed.status()["completed_prefix_count"], 0)


if __name__ == "__main__":
    unittest.main()
