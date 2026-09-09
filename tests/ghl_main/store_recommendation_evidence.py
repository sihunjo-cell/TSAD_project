"""새 Dev18 실험의 작은 추천 원표를 SQLite로 연결한다."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import time
from contextlib import closing
from pathlib import Path

import numpy

from src.common.compute_prefix_features import (
    EXTRACTOR_CONTRACT, compute_prefix_features, prefix_feature_identity,
)
from src.common.equal_trial_budget import registry_space_sha256
from src.common.execution_identity import file_sha256, load_input_manifest_role
from src.common.experiment_config import SUPPORTED_RATIO_PERCENTS


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECOMMENDATION_DIRECTORY = (
    REPOSITORY_ROOT / "experiments/01_ghl_main/results/dev18_tuning/full_prefix_v2/recommendation_evidence"
)
FULL_PREFIX_SNAPSHOT_DIRECTORY = REPOSITORY_ROOT / "experiments/01_ghl_main/snapshots/dev18_selection/full_prefix_v2"
SCHEMA_VERSION = 3
DESCRIPTOR_COLUMNS = ("interquartile_range", "difference_q90_iqr_ratio", "median_shift_iqr_ratio", "spectral_entropy")
PERFORMANCE_HEADERS = ("CSV_file", "q_percent", "training_boundary", "observed_row", "input_column",
                       "prefix_feature_id", "model", "config_id", "seed", "status", "vus_pr")
PREFIX_HEADERS = ("prefix_feature_id", "csv_file", "q_percent", "observed_row", "상수채널수",
                  "채널std median", "채널 ACF lag1 median", "채널간 절대 상관 중앙값",
                  *(f"channel_{field}_median" for field in DESCRIPTOR_COLUMNS))
CHANNEL_HEADERS = ("prefix_feature_id", "channel(col)", "mean", "std", "median", "ACF lag1", *DESCRIPTOR_COLUMNS)
CONFIG_HEADERS = ("config_id", "model", "설정 예시")
RECOMMENDATION_CHANNEL_DESCRIPTORS = ("acf_lag1", "interquartile_range_std_ratio", *DESCRIPTOR_COLUMNS[1:])
RECOMMENDATION_CONTEXT_COLUMNS = ("prefix_feature_id", "csv_id", "csv_file", "series", "family", "q_percent", "training_boundary")
RECOMMENDATION_FEATURE_COLUMNS = (
    "observed_row", "input_column", "constant_channel_fraction", "absolute_correlation_median",
    *(f"channel_{field}_median" for field in RECOMMENDATION_CHANNEL_DESCRIPTORS),
)
RECOMMENDATION_QUALITY_COLUMNS = (
    "finite_value_fraction",
    "constant_channel_count", "constant_evaluable_channel_count", "constant_channel_count_reason",
    "constant_evaluable_channel_fraction", "absolute_correlation_median_reason",
    "absolute_correlation_valid_pair_count", "absolute_correlation_eligible_pair_count",
    "absolute_correlation_valid_pair_fraction",
    *(column for field in RECOMMENDATION_CHANNEL_DESCRIPTORS for column in (
        f"channel_{field}_median_reason", f"channel_{field}_valid_channel_count",
        f"channel_{field}_valid_channel_fraction")),
)
RECOMMENDATION_HEADERS = (*RECOMMENDATION_CONTEXT_COLUMNS, *RECOMMENDATION_FEATURE_COLUMNS, *RECOMMENDATION_QUALITY_COLUMNS)
RECOMMENDATION_VIEW_SQL = f"""CREATE VIEW recommendation_inputs AS
    WITH ranked_ratios AS (
        SELECT prefix_feature_id, interquartile_range / std AS ratio,
            ROW_NUMBER() OVER (PARTITION BY prefix_feature_id ORDER BY interquartile_range / std) AS row_index,
            COUNT(*) OVER (PARTITION BY prefix_feature_id) AS valid_channel_count
        FROM channel_features WHERE std > 0 AND interquartile_range IS NOT NULL
    ), ratio_summary AS (
        SELECT prefix_feature_id, AVG(ratio) AS channel_interquartile_range_std_ratio_median,
            MAX(valid_channel_count) AS channel_interquartile_range_std_ratio_valid_channel_count
        FROM ranked_ratios WHERE 2 * row_index BETWEEN valid_channel_count AND valid_channel_count + 2
        GROUP BY prefix_feature_id
    ), sample_coverage AS (
        SELECT prefix_feature_id, SUM(valid_value_count) AS finite_value_count
        FROM channel_features GROUP BY prefix_feature_id
    ), input_features AS (
        SELECT prefix_features.*, finite_value_count, channel_interquartile_range_std_ratio_median,
            COALESCE(channel_interquartile_range_std_ratio_valid_channel_count, 0)
                AS channel_interquartile_range_std_ratio_valid_channel_count,
            CASE WHEN channel_interquartile_range_std_ratio_median IS NULL THEN 'no_valid_channel_ratios' END
                AS channel_interquartile_range_std_ratio_median_reason
        FROM prefix_features LEFT JOIN ratio_summary USING (prefix_feature_id)
            LEFT JOIN sample_coverage USING (prefix_feature_id)
    )
    SELECT {','.join(RECOMMENDATION_CONTEXT_COLUMNS)}, observed_row, input_column,
        1.0 * constant_channel_count / NULLIF(constant_evaluable_channel_count, 0) AS constant_channel_fraction,
        absolute_correlation_median,
        {','.join(f'channel_{field}_median' for field in RECOMMENDATION_CHANNEL_DESCRIPTORS)},
        1.0 * finite_value_count / NULLIF(1.0 * observed_row * input_column, 0) AS finite_value_fraction,
        constant_channel_count, constant_evaluable_channel_count, constant_channel_count_reason,
        1.0 * constant_evaluable_channel_count / input_column AS constant_evaluable_channel_fraction,
        absolute_correlation_median_reason, absolute_correlation_valid_pair_count, absolute_correlation_eligible_pair_count,
        2.0 * absolute_correlation_valid_pair_count / NULLIF(1.0 * input_column * (input_column - 1), 0)
            AS absolute_correlation_valid_pair_fraction,
        {','.join(f'channel_{field}_median_reason,channel_{field}_valid_channel_count,'
                  f'1.0 * channel_{field}_valid_channel_count / input_column AS channel_{field}_valid_channel_fraction'
                  for field in RECOMMENDATION_CHANNEL_DESCRIPTORS)}
    FROM input_features"""
SUMMARY_COLUMNS = (
    "constant_channel_count", "constant_evaluable_channel_count", "constant_channel_count_reason",
    "channel_std_median", "channel_std_median_reason", "channel_std_valid_channel_count",
    "channel_acf_lag1_median", "channel_acf_lag1_median_reason", "channel_acf_lag1_valid_channel_count",
    "absolute_correlation_median", "absolute_correlation_median_reason",
    "absolute_correlation_valid_pair_count", "absolute_correlation_eligible_pair_count",
    *(column for field in DESCRIPTOR_COLUMNS for column in (
        f"channel_{field}_median", f"channel_{field}_median_reason", f"channel_{field}_valid_channel_count")),
)
CHANNEL_COLUMNS = ("channel_index", "channel_name", "valid_value_count", "is_constant", "constant_reason",
                   "mean", "mean_reason", "std", "std_reason", "median", "median_reason",
                   "acf_lag1", "acf_lag1_reason",
                   *(column for field in DESCRIPTOR_COLUMNS for column in (field, field + "_reason")))
RESULT_COLUMNS = ("prefix_feature_id", "config_id", "seed", "score_variant", "primary_score", "status",
                  "status_reason", "vus_pr", "physical_execution_id", "training_group_id", "score_file",
                  "score_sha256", "metadata_file", "metadata_sha256", "manifest_reference",
                  "ledger_reference", "evaluator_sha256", "ell_max_id")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def build_recommendation_identity(registry, budget, *, project_commit):
    if budget.get("experiment_mode") != "full_prefix_v2":
        raise ValueError("recommendation evidence requires a new full_prefix_v2 experiment")
    if not project_commit or registry_space_sha256(registry) != budget["registry_space_sha256"]:
        raise ValueError("recommendation source or registry differs from the sealed budget")
    feasibility_path = FULL_PREFIX_SNAPSHOT_DIRECTORY / "dev18_feasibility_ledger.csv"
    summary = json.loads((FULL_PREFIX_SNAPSHOT_DIRECTORY / "dev18_feasibility_summary.json").read_text(encoding="utf-8"))
    if (summary["feasibility_decision_sha256"] != budget["feasibility_decision_sha256"]
            or file_sha256(feasibility_path) != summary["ledger_sha256"]):
        raise ValueError("recommendation feasibility evidence differs from the sealed budget")
    return {
        "schema_version": SCHEMA_VERSION, "experiment_mode": "full_prefix_v2",
        "project_commit": project_commit,
        **{key: budget[key] for key in ("budget_id", "budget_sha256", "registry_space_sha256",
                                       "input_manifest_sha256", "data_preprocessing_sha256",
                                       "feasibility_decision_sha256")},
        "common_recipe": registry["common_recipe"], "extractor": EXTRACTOR_CONTRACT,
        "extractor_source_sha256": file_sha256(REPOSITORY_ROOT / "src/common/compute_prefix_features.py"),
        "storage_source_sha256": file_sha256(Path(__file__)),
        "feasibility_ledger": {"file": feasibility_path.relative_to(REPOSITORY_ROOT).as_posix(),
                               "sha256": summary["ledger_sha256"]},
    }


def open_recommendation_evidence(registry, budget, *, directory=DEFAULT_RECOMMENDATION_DIRECTORY,
                                 project_commit=None):
    manifest, manifest_sha256 = load_input_manifest_role(
        REPOSITORY_ROOT / "configs/input_manifest.yaml", "development", "dev18_selection",
    )
    if manifest_sha256 != budget["input_manifest_sha256"]:
        raise ValueError("recommendation manifest differs from the sealed budget")
    if project_commit is None:
        project_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True,
        ).strip()
    return RecommendationEvidence(directory, identity=build_recommendation_identity(
        registry, budget, project_commit=project_commit,
    ), registry=registry, budget=budget, entries=manifest["datasets"]["DEV18"]["files"])


class RecommendationEvidence:
    def __init__(self, directory, *, identity, registry, budget, entries):
        if identity.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("recommendation schema version differs; create a separately sealed database")
        self.directory = Path(directory)
        self.database_path = self.directory / "recommendation.sqlite3"
        self.budget = budget
        entries = list(entries)
        self.entries = {str(entry["series"]).zfill(2): dict(entry) for entry in entries}
        if (len(entries) != 18 or len(self.entries) != 18 or sorted(self.entries) != [f"{index:02d}" for index in range(1, 19)]
                or len({entry["name"] for entry in self.entries.values()}) != 18):
            raise ValueError("recommendation inputs must contain the 18 distinct Dev18 CSV files")
        if any(not isinstance(entry.get("family"), str) or not entry["family"].strip() for entry in entries):
            raise ValueError("recommendation inputs require the sealed dataset family for grouped validation")
        self.identity = {**identity, "manifest_entries_sha256": _digest(list(self.entries.values()))}
        self.owner_process = os.getpid()
        self._verified_execution_files = {}
        self._features_ready = False
        self.prefixes = {}
        for series, entry in self.entries.items():
            for ratio in SUPPORTED_RATIO_PERCENTS:
                self.prefixes[series, ratio] = prefix_feature_identity(
                    csv_id=f"{entry['source_directory']}/{entry['name']}", source_sha256=entry["sha256"],
                    ordered_schema_sha256=entry["feature_names_sha256"], q_percent=ratio,
                    source_start=0, source_end_exclusive=entry["training_boundary"] * ratio // 100,
                )
        self.panel = {}
        self.expected_results = {}
        for panel in budget["execution_panel"]:
            for series in panel["series_ids"]:
                for variant in (*panel["primary_score_variants"], *panel["diagnostic_score_variants"]):
                    if not isinstance(variant, str):
                        raise ValueError("score variant must be the canonical non-NULL string")
                    key = (str(series).zfill(2), panel["model"], panel["config_id"],
                           int(panel["physical_ratio"]), int(panel["seed"]), variant)
                    if key in self.panel:
                        raise ValueError("duplicate physical score key in recommendation budget")
                    self.panel[key] = panel
                    for ratio in panel["logical_ratios"]:
                        logical_key = (self.prefixes[key[0], int(ratio)]["prefix_feature_id"],
                                       panel["config_id"], int(panel["seed"]), variant)
                        if logical_key in self.expected_results:
                            raise ValueError("duplicate logical score key in recommendation budget")
                        self.expected_results[logical_key] = int(variant in panel["primary_score_variants"])
        self.contract = {
            "identity": self.identity, "schema_version": SCHEMA_VERSION,
            "expected_prefix_count": len(self.prefixes),
            "expected_channel_count": len(SUPPORTED_RATIO_PERCENTS) * sum(entry["feature_count"] for entry in self.entries.values()),
            "expected_result_count": len(self.expected_results),
            "expected_primary_result_count": sum(self.expected_results.values()),
            "expected_exclusion_count": len(budget.get("structural_exclusions", [])),
            "policy": "one unscaled float32 observed prefix per CSV/q; model-native training, validation and evaluation scopes are recorded in execution metadata",
            "feature_measurement_scope": {
                "collection_seconds": "extractor call", "storage_seconds": "SQL writes before commit",
                "feature_bytes": "UTF-8 canonical summary and channel JSON payload",
            },
            "recommendation_input_contract": {
                "version": "recommendation_inputs.v1", "view": "recommendation_inputs",
                "export": "recommendation_inputs.csv", "context_columns": list(RECOMMENDATION_CONTEXT_COLUMNS),
                "feature_columns": list(RECOMMENDATION_FEATURE_COLUMNS), "quality_columns": list(RECOMMENDATION_QUALITY_COLUMNS),
                "unit_policy": "channel descriptors invariant to independent nonzero affine channel transformations in exact arithmetic; raw mean/std/IQR excluded",
                "shape_ratio": "median of per-channel IQR/std; std=0 omitted, IQR=0 with std>0 is valid zero",
                "coverage": "finite values divided by n*d; valid channels divided by d; valid correlations divided by all d*(d-1)/2 pairs; empty denominators yield NULL",
                "missing_values": "NULL and reason retained; fit imputation and scaling on training families only",
                "sampling": "lags and differences are per observation; sampling interval unknown; temporal estimates depend on observed_row",
                "validation": "group all CSV/q prefixes from the same family in one fold; IDs, family and q/training_boundary are context, not default features",
            },
        }
        if self.contract["expected_primary_result_count"] != budget["expected_ledger_rows"]:
            raise ValueError("recommendation expected scores differ from the sealed ledger budget")
        self.expected_exclusions = {}
        feasible_pairs = {key[:2] for key in self.expected_results}
        for row in budget.get("structural_exclusions", []):
            key = (self.prefixes[str(row["series"]).zfill(2), int(row["ratio"])]["prefix_feature_id"], row["config_id"])
            if key in self.expected_exclusions or key in feasible_pairs or not row["status_reason"]:
                raise ValueError("structural exclusion is duplicated, feasible or has no reason")
            feasibility = self.identity["feasibility_ledger"]
            reference_key = (row["model"], row["config_id"], int(row["ratio"]), str(row["series"]).zfill(2))
            self.expected_exclusions[key] = (row["status_reason"],
                f"{feasibility['file']}@{feasibility['sha256']}#key={_json(reference_key)}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            tables = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and "experiment_identity" not in tables:
                raise ValueError("existing SQLite file belongs to another schema; refusing to overwrite")
            if not tables:
                self._create_schema()
            stored = self.connection.execute("SELECT identity_json FROM experiment_identity WHERE singleton=1").fetchone()
            if stored is None:
                raise ValueError("recommendation database has no sealed experiment identity")
            if stored[0] != _json(self.identity):
                raise ValueError("recommendation database belongs to another experiment identity")
            with self.connection:
                for model, settings in registry["models"].items():
                    for candidate in settings["candidates"]:
                        complete_settings = {**{key: value for key, value in settings.items()
                                                if key not in ("candidates", "execution_status", "status_reason")},
                                             **candidate, "model": model, "common_recipe": registry["common_recipe"]}
                        values = (candidate["config_id"], model, settings["target_use"], _json(complete_settings))
                        previous = self.connection.execute("SELECT * FROM model_configs WHERE config_id=?", values[:1]).fetchone()
                        if previous is not None and tuple(previous) != values:
                            raise ValueError("config_id already refers to different complete settings")
                        self.connection.execute("INSERT OR IGNORE INTO model_configs VALUES (?,?,?,?)", values)
            self._validate_cached_features()
        except BaseException:
            self.connection.close()
            raise

    def _create_schema(self):
        self.connection.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE experiment_identity (
                singleton INTEGER PRIMARY KEY CHECK(singleton=1), identity_json TEXT NOT NULL,
                environment_json TEXT
            ) STRICT;
            CREATE TABLE prefix_features (
                prefix_feature_id TEXT PRIMARY KEY NOT NULL, csv_id TEXT NOT NULL, csv_file TEXT NOT NULL,
                family TEXT NOT NULL CHECK(trim(family)<>''),
                series TEXT NOT NULL, source_sha256 TEXT NOT NULL, ordered_schema_sha256 TEXT NOT NULL,
                q_percent INTEGER NOT NULL CHECK(q_percent IN (5,10,20,40,60,80,100)),
                training_boundary INTEGER NOT NULL CHECK(training_boundary>0),
                observed_row INTEGER NOT NULL CHECK(observed_row>=0), input_column INTEGER NOT NULL CHECK(input_column>0),
                source_start INTEGER NOT NULL CHECK(source_start=0), source_end_exclusive INTEGER NOT NULL,
                identity_json TEXT NOT NULL, feature_sha256 TEXT NOT NULL,
                constant_channel_count INTEGER, constant_evaluable_channel_count INTEGER NOT NULL,
                constant_channel_count_reason TEXT,
                channel_std_median REAL, channel_std_median_reason TEXT, channel_std_valid_channel_count INTEGER NOT NULL,
                channel_acf_lag1_median REAL, channel_acf_lag1_median_reason TEXT,
                channel_acf_lag1_valid_channel_count INTEGER NOT NULL,
                absolute_correlation_median REAL, absolute_correlation_median_reason TEXT,
                absolute_correlation_valid_pair_count INTEGER NOT NULL, absolute_correlation_eligible_pair_count INTEGER NOT NULL,
                channel_interquartile_range_median REAL CHECK(channel_interquartile_range_median>=0),
                channel_interquartile_range_median_reason TEXT,
                channel_interquartile_range_valid_channel_count INTEGER NOT NULL CHECK(channel_interquartile_range_valid_channel_count>=0),
                channel_difference_q90_iqr_ratio_median REAL CHECK(channel_difference_q90_iqr_ratio_median>=0),
                channel_difference_q90_iqr_ratio_median_reason TEXT,
                channel_difference_q90_iqr_ratio_valid_channel_count INTEGER NOT NULL CHECK(channel_difference_q90_iqr_ratio_valid_channel_count>=0),
                channel_median_shift_iqr_ratio_median REAL CHECK(channel_median_shift_iqr_ratio_median>=0),
                channel_median_shift_iqr_ratio_median_reason TEXT,
                channel_median_shift_iqr_ratio_valid_channel_count INTEGER NOT NULL CHECK(channel_median_shift_iqr_ratio_valid_channel_count>=0),
                channel_spectral_entropy_median REAL CHECK(channel_spectral_entropy_median BETWEEN 0 AND 1),
                channel_spectral_entropy_median_reason TEXT,
                channel_spectral_entropy_valid_channel_count INTEGER NOT NULL CHECK(channel_spectral_entropy_valid_channel_count>=0),
                collection_seconds REAL NOT NULL, storage_seconds REAL NOT NULL, feature_bytes INTEGER NOT NULL,
                UNIQUE(csv_id,q_percent), UNIQUE(csv_file,q_percent), UNIQUE(series,q_percent),
                CHECK(observed_row=training_boundary*q_percent/100), CHECK(source_end_exclusive=observed_row)
            ) STRICT;
            CREATE TABLE channel_features (
                prefix_feature_id TEXT NOT NULL REFERENCES prefix_features(prefix_feature_id),
                channel_index INTEGER NOT NULL CHECK(channel_index>=0), channel_name TEXT NOT NULL,
                valid_value_count INTEGER NOT NULL CHECK(valid_value_count>=0),
                is_constant INTEGER CHECK(is_constant IN (0,1)), constant_reason TEXT,
                mean REAL, mean_reason TEXT, std REAL, std_reason TEXT, median REAL, median_reason TEXT,
                acf_lag1 REAL, acf_lag1_reason TEXT,
                interquartile_range REAL CHECK(interquartile_range>=0), interquartile_range_reason TEXT,
                difference_q90_iqr_ratio REAL CHECK(difference_q90_iqr_ratio>=0), difference_q90_iqr_ratio_reason TEXT,
                median_shift_iqr_ratio REAL CHECK(median_shift_iqr_ratio>=0), median_shift_iqr_ratio_reason TEXT,
                spectral_entropy REAL CHECK(spectral_entropy BETWEEN 0 AND 1), spectral_entropy_reason TEXT,
                PRIMARY KEY(prefix_feature_id,channel_index)
            ) STRICT;
            CREATE TABLE model_configs (
                config_id TEXT PRIMARY KEY NOT NULL, model TEXT NOT NULL, target_use TEXT NOT NULL,
                settings_json TEXT NOT NULL
            ) STRICT;
            CREATE TABLE results (
                prefix_feature_id TEXT NOT NULL REFERENCES prefix_features(prefix_feature_id),
                config_id TEXT NOT NULL REFERENCES model_configs(config_id), seed INTEGER NOT NULL,
                score_variant TEXT NOT NULL, primary_score INTEGER NOT NULL CHECK(primary_score IN (0,1)),
                status TEXT NOT NULL CHECK(status IN ('complete','executed_pending_score','executed_unscored',
                    'failed','interrupted','running','timeout')), status_reason TEXT, vus_pr REAL,
                physical_execution_id TEXT NOT NULL, training_group_id TEXT NOT NULL,
                score_file TEXT, score_sha256 TEXT, metadata_file TEXT, metadata_sha256 TEXT,
                manifest_reference TEXT, ledger_reference TEXT, evaluator_sha256 TEXT, ell_max_id TEXT,
                PRIMARY KEY(prefix_feature_id,config_id,seed,score_variant),
                CHECK((status='complete' AND primary_score=1 AND vus_pr IS NOT NULL AND vus_pr>=0 AND vus_pr<=1
                    AND evaluator_sha256 IS NOT NULL AND trim(evaluator_sha256)<>''
                    AND ell_max_id IS NOT NULL AND trim(ell_max_id)<>''
                    AND ledger_reference IS NOT NULL AND trim(ledger_reference)<>'')
                    OR (status!='complete' AND vus_pr IS NULL)),
                CHECK(status NOT IN ('complete','executed_pending_score','executed_unscored') OR
                    (trim(physical_execution_id)<>'' AND score_file IS NOT NULL AND trim(score_file)<>''
                    AND score_sha256 IS NOT NULL AND trim(score_sha256)<>''
                    AND metadata_file IS NOT NULL AND trim(metadata_file)<>''
                    AND metadata_sha256 IS NOT NULL AND trim(metadata_sha256)<>''
                    AND manifest_reference IS NOT NULL AND trim(manifest_reference)<>''))
            ) STRICT;
            CREATE TABLE structural_exclusions (
                prefix_feature_id TEXT NOT NULL REFERENCES prefix_features(prefix_feature_id),
                config_id TEXT NOT NULL REFERENCES model_configs(config_id), reason TEXT NOT NULL,
                feasibility_reference TEXT NOT NULL, PRIMARY KEY(prefix_feature_id,config_id)
            ) STRICT;
        """)
        with self.connection:
            self.connection.execute(RECOMMENDATION_VIEW_SQL)
            self.connection.execute("INSERT INTO experiment_identity VALUES (1,?,NULL)", (_json(self.identity),))

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        self.close()

    def close(self):
        self.connection.close()

    def _require_parent(self):
        if os.getpid() != self.owner_process:
            raise RuntimeError("only the parent process may write recommendation evidence")

    def bind_environment(self, environment):
        self._require_parent()
        if not isinstance(environment, dict) or not environment:
            raise ValueError("actual execution environment is required")
        serialized = _json(environment)
        existing = self.connection.execute("SELECT environment_json FROM experiment_identity").fetchone()[0]
        if existing is not None and existing != serialized:
            raise ValueError("recommendation execution environment changed on resume")
        with self.connection:
            self.connection.execute("UPDATE experiment_identity SET environment_json=? WHERE singleton=1", (serialized,))

    def _environment(self):
        serialized = self.connection.execute("SELECT environment_json FROM experiment_identity").fetchone()[0]
        if serialized is None:
            raise ValueError("recommendation execution environment is not sealed")
        return json.loads(serialized)

    def _validate_cached_features(self):
        view = self.connection.execute("SELECT sql FROM sqlite_master WHERE type='view' AND name='recommendation_inputs'").fetchone()
        if view is None or view[0] != RECOMMENDATION_VIEW_SQL:
            raise ValueError("recommendation input view differs from the sealed feature contract")
        if self.connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("recommendation database has broken foreign keys")
        for row in self.connection.execute("SELECT * FROM prefix_features"):
            expected = self.prefixes.get((row["series"], row["q_percent"]))
            if expected is None or any(row[key] != expected[key] for key in ("prefix_feature_id", "identity_json")):
                raise ValueError("cached prefix source, range, schema or extractor identity changed")
            entry = self.entries[row["series"]]
            count = entry["training_boundary"] * row["q_percent"] // 100
            expected_fields = {
                "csv_id": f"{entry['source_directory']}/{entry['name']}", "csv_file": entry["name"],
                "family": entry["family"],
                "source_sha256": entry["sha256"], "ordered_schema_sha256": entry["feature_names_sha256"],
                "training_boundary": entry["training_boundary"], "input_column": entry["feature_count"],
                "observed_row": count, "source_start": 0, "source_end_exclusive": count,
            }
            if any(row[field] != value for field, value in expected_fields.items()):
                raise ValueError("cached prefix source identity or dimensions differ from the manifest")
            channels = [{key: item[key] for key in CHANNEL_COLUMNS} for item in self.connection.execute(
                "SELECT * FROM channel_features WHERE prefix_feature_id=? ORDER BY channel_index", (row["prefix_feature_id"],),
            )]
            if [channel["channel_index"] for channel in channels] != list(range(entry["feature_count"])):
                raise ValueError("cached prefix channel rows are incomplete")
            if _digest([channel["channel_name"] for channel in channels]) != entry["feature_names_sha256"]:
                raise ValueError("cached ordered channel schema differs from the manifest")
            if _digest({"summary": {key: row[key] for key in SUMMARY_COLUMNS}, "channels": channels}) != row["feature_sha256"]:
                raise ValueError("cached prefix statistics fingerprint mismatch")

    def prepare_features(self, load_inputs):
        self._require_parent()
        self._environment()
        self._validate_cached_features()
        cached = {row[0] for row in self.connection.execute("SELECT prefix_feature_id FROM prefix_features")}
        for series, entry in self.entries.items():
            ratios = [ratio for ratio in SUPPORTED_RATIO_PERCENTS
                      if self.prefixes[series, ratio]["prefix_feature_id"] not in cached]
            if not ratios:
                continue
            inputs = load_inputs(entry)
            training = inputs["normal_training"]
            names = list(inputs["feature_names"])
            if (not isinstance(training, numpy.ndarray) or training.dtype != numpy.float32
                    or training.shape != (entry["training_boundary"], entry["feature_count"])
                    or _digest(names) != entry["feature_names_sha256"]
                    or inputs["input_identity"]["sha256"] != entry["sha256"]):
                raise ValueError("feature loader input differs from the sealed manifest")
            for ratio in ratios:
                count = entry["training_boundary"] * ratio // 100
                started = time.perf_counter()
                summary, channels = compute_prefix_features(training[:count], names)
                collection_seconds = time.perf_counter() - started
                identity = self.prefixes[series, ratio]
                payload = {"summary": summary, "channels": channels}
                values = {
                    **identity, "csv_id": f"{entry['source_directory']}/{entry['name']}",
                    "family": entry["family"],
                    "csv_file": entry["name"], "series": series, "source_sha256": entry["sha256"],
                    "ordered_schema_sha256": entry["feature_names_sha256"], "q_percent": ratio,
                    "training_boundary": entry["training_boundary"], "observed_row": count,
                    "input_column": entry["feature_count"], "source_start": 0, "source_end_exclusive": count,
                    "feature_sha256": _digest(payload), **summary, "collection_seconds": collection_seconds,
                    "storage_seconds": 0.0, "feature_bytes": len(_json(payload).encode("utf-8")),
                }
                started = time.perf_counter()
                with self.connection:
                    self.connection.execute(f"INSERT INTO prefix_features ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
                                            tuple(values.values()))
                    self.connection.executemany(
                        f"INSERT INTO channel_features (prefix_feature_id,{','.join(CHANNEL_COLUMNS)}) "
                        f"VALUES ({','.join('?' for _ in range(len(CHANNEL_COLUMNS) + 1))})",
                        [(identity["prefix_feature_id"], *(channel[key] for key in CHANNEL_COLUMNS)) for channel in channels],
                    )
                    self.connection.execute("UPDATE prefix_features SET storage_seconds=? WHERE prefix_feature_id=?",
                                            (time.perf_counter() - started, identity["prefix_feature_id"]))
            del training, inputs
        self._sync_exclusions()
        self.require_features()
        return self.status()

    def _sync_exclusions(self):
        with self.connection:
            for key, details in self.expected_exclusions.items():
                values = (*key, *details)
                previous = self.connection.execute("SELECT * FROM structural_exclusions WHERE prefix_feature_id=? AND config_id=?",
                                                   values[:2]).fetchone()
                if previous is not None and tuple(previous) != values:
                    raise ValueError("structural exclusion differs from the sealed budget")
                self.connection.execute("INSERT OR IGNORE INTO structural_exclusions VALUES (?,?,?,?)", values)

    def require_features(self):
        self._environment()
        self._validate_cached_features()
        state = self.status()
        if not state["feature_complete"]:
            raise ValueError("recommendation prefix/channel features are incomplete")
        self._features_ready = True
        return state

    def _verify_execution_files(self, row):
        from tests.ghl_main.run_dev18_tuning import _validate_bound_run_files
        from src.common.execution_evidence import (
            FULL_PREFIX_MEASUREMENT_PROTOCOL_ID, FULL_PREFIX_STORAGE_SCHEMA_VERSION,
            validate_execution_evidence_for_run,
        )

        paths = []
        for field in ("score", "metadata"):
            path = (REPOSITORY_ROOT / row[field + "_file"]).resolve()
            path.relative_to(REPOSITORY_ROOT)
            if not path.is_file():
                raise ValueError("recommendation execution artifact is missing")
            paths.append(path)
        metadata = json.loads(paths[1].read_text(encoding="utf-8"))
        references = [metadata.get("run_snapshot"), *(metadata.get("training_files") or {}).values(),
                      *metadata.get("score_files", [])]
        if any(not isinstance(reference, dict) or not reference.get("file") for reference in references):
            raise ValueError("recommendation snapshot/training evidence references are missing")
        reference_paths = [(REPOSITORY_ROOT / reference["file"]).resolve() for reference in references]
        for path in reference_paths:
            path.relative_to(REPOSITORY_ROOT)
        signature = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in (*paths, *reference_paths))
        cache_key = tuple(str(row[field]) for field in ("series", "model", "config_id", "physical_ratio",
            "seed", "score_variant", "score_file", "score_sha256", "metadata_file", "metadata_sha256"))
        if self._verified_execution_files.get(cache_key) == signature:
            return metadata
        if any(file_sha256(path) != row[field + "_sha256"] for field, path in zip(("score", "metadata"), paths)):
            raise ValueError("recommendation artifact SHA-256 differs from the score manifest")
        expected = {"dataset": "DEV18", "series": int(row["series"]),
                    "model": row["model"], "config_id": row["config_id"],
                    "seed": int(row["seed"]), "ratio": int(row["physical_ratio"]),
                    "score_variant": row["score_variant"] or None}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("recommendation metadata key differs from the score manifest")
        settings = json.loads(self.connection.execute("SELECT settings_json FROM model_configs WHERE config_id=?", (row["config_id"],)).fetchone()[0])
        snapshot = json.loads(reference_paths[0].read_text(encoding="utf-8"))
        specification = snapshot.get("spec", {})
        if (any(specification.get(field) != settings.get(field) for field in (
                "model", "config_id", "source_commit", "source_checkpoint_sha256", "checkpoint_config_sha256",
                "checkpoint_revision", "preprocess_recipe", "common_recipe", "hyperparameters", "target_use", "tier"))
                or any(specification.get(field) != expected[field] for field in ("seed", "ratio"))
                or specification.get("dataset_role") != "development"
                or specification.get("split_role") != "dev18_selection"
                or specification.get("input_manifest_sha256") != self.identity["input_manifest_sha256"]):
            raise ValueError("recommendation snapshot settings differ from the sealed model config")
        if (metadata.get("storage_schema_version") != FULL_PREFIX_STORAGE_SCHEMA_VERSION
                or metadata.get("execution_evidence", {}).get("measurement_protocol_id") != FULL_PREFIX_MEASUREMENT_PROTOCOL_ID):
            raise ValueError("recommendation metadata uses an obsolete execution/storage schema")
        run_evidence = validate_execution_evidence_for_run(metadata["execution_evidence"],
            dataset_role="development", target_use=settings["target_use"])
        entry = self.entries[str(row["series"]).zfill(2)]
        if (settings["model"] == "TSPulse"
                and settings.get("common_recipe", {}).get("methodology_revision") == "paper_tuning_v4"):
            from src.common.save_model_artifacts import validate_tspulse_native_calibration

            validate_tspulse_native_calibration(metadata.get("native_calibration"), source_start=0,
                source_end_exclusive=entry["row_count"] - entry["training_boundary"],
                score_variant=row["score_variant"], input_column=entry["feature_count"])
        if (len(run_evidence["test_sessions"]) != 1
                or run_evidence["test_sessions"][0]["observation_count"] != entry["row_count"] - entry["training_boundary"]):
            raise ValueError("recommendation execution test extent differs from the input manifest")
        if settings["target_use"] == "fit_full_prefix" and (
            len(run_evidence["training_sessions"]) != 1
            or run_evidence["training_sessions"][0]["training_boundary"] != entry["training_boundary"]
            or run_evidence["training_sessions"][0]["observed_row"] != entry["training_boundary"] * int(row["physical_ratio"]) // 100
        ):
            raise ValueError("recommendation execution training prefix differs from the input manifest")
        _validate_bound_run_files(metadata, expected_project_commit=self.identity["project_commit"],
                                  expected_environment=self._environment(), allow_compatible_history=False)
        if (str(snapshot["spec"].get("series")).zfill(2) != str(row["series"]).zfill(2)
                or snapshot["spec"].get("series_input_sha256") != entry["sha256"]):
            raise ValueError("recommendation snapshot source differs from the Dev18 input")
        input_identity = snapshot.get("input_identity", {})
        if (any(input_identity.get(field) != entry[field] for field in ("name", "sha256", "source_directory", "size_bytes"))
                or input_identity.get("input_manifest_sha256") != self.identity["input_manifest_sha256"]
                or snapshot.get("source_ranges") != {
                    "source": entry["name"], "source_directory": entry["source_directory"],
                    "normal_training": [0, entry["training_boundary"]],
                    "test_sessions": [[entry["training_boundary"], entry["row_count"]]],
                }):
            raise ValueError("recommendation snapshot input identity or source ranges differ from the manifest")
        attempt = metadata.get("execution_attempt", {})
        if not attempt.get("run_id") or not attempt.get("history_file"):
            raise ValueError("recommendation execution attempt reference is missing")
        history_path = (REPOSITORY_ROOT / attempt["history_file"]).resolve()
        history_path.relative_to(REPOSITORY_ROOT)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history.get("run_id") != attempt["run_id"] or history.get("identity", {}).get("budget_id") != self.budget["budget_id"]:
            raise ValueError("recommendation execution attempt identity differs from the new budget")
        self._verified_execution_files[cache_key] = signature
        return metadata

    def sync_results(self, manifest_rows, *, ledger_rows=(), manifest_path=None, ledger_path=None):
        self._require_parent()
        if not self._features_ready:
            self.require_features()
        manifest_rows, ledger_rows = list(manifest_rows), list(ledger_rows)
        if manifest_rows and (manifest_path is None or not Path(manifest_path).is_file()):
            raise ValueError("recommendation results require the persisted score manifest")
        if ledger_rows and (ledger_path is None or not Path(ledger_path).is_file()):
            raise ValueError("recommendation scores require the persisted trial ledger")
        if ledger_rows:
            self._compare_source_rows(ledger_rows, ledger_path,
                ("series", "config_id", "ratio", "seed", "score_variant"),
                ("model", "status", "score_file", "score_sha256", "normalization", "evaluator_sha256", "ell_max_id", "vus_pr"))
            self._compare_source_rows(manifest_rows, manifest_path,
                ("series", "model", "config_id", "physical_ratio", "seed", "score_variant"),
                ("budget_id", "primary_score", "status", "score_file", "score_sha256", "metadata_file", "metadata_sha256"))
        ledger = {}
        for row in ledger_rows:
            key = (str(row["series"]).zfill(2), row["config_id"], int(row["ratio"]),
                   int(row["seed"]), row["score_variant"])
            if key in ledger or row.get("status") != "complete":
                raise ValueError("recommendation ledger has duplicate or unfinished rows")
            ledger[key] = row
        seen_manifest, used_ledger = set(), set()
        with self.connection:
            for row in manifest_rows:
                key = (str(row["series"]).zfill(2), row["model"], row["config_id"],
                       int(row["physical_ratio"]), int(row["seed"]), row["score_variant"])
                if key in seen_manifest or key not in self.panel or row.get("budget_id") != self.budget["budget_id"]:
                    raise ValueError("recommendation manifest key/budget differs from the new execution panel")
                seen_manifest.add(key)
                panel = self.panel[key]
                primary = int(key[-1] in panel["primary_score_variants"])
                if str(row.get("primary_score")).lower() != str(bool(primary)).lower():
                    raise ValueError("recommendation score head role differs from the sealed panel")
                complete = row["status"] == "complete"
                metadata = {}
                if complete:
                    metadata = self._verify_execution_files(row) or {}
                elif row["status"] not in ("failed", "interrupted", "running", "timeout"):
                    raise ValueError("structural exclusions and unknown states are not result rows")
                physical_id = metadata.get("execution_attempt", {}).get("run_id") or (
                    "e" + _digest({"budget": self.budget["budget_sha256"], "key": list(key[:-1])}))
                for ratio in panel["logical_ratios"]:
                    ledger_key = (key[0], key[2], int(ratio), key[4], key[5])
                    score = ledger.get(ledger_key)
                    if score is not None:
                        if (not complete or not primary or score.get("model") != key[1]
                                or score.get("normalization") != "trainnorm"
                                or any(score.get(field) != row.get(field) for field in ("score_file", "score_sha256"))
                                or not score.get("evaluator_sha256") or not score.get("ell_max_id")):
                            raise ValueError("recommendation ledger score identity differs from the executed score")
                        value = float(score["vus_pr"])
                        if not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError("recommendation VUS must be finite and between zero and one")
                        used_ledger.add(ledger_key)
                    prefix = self.prefixes[key[0], int(ratio)]["prefix_feature_id"]
                    status = ("complete" if score is not None else "executed_pending_score" if primary else "executed_unscored") if complete else row["status"]
                    values = dict(prefix_feature_id=prefix, config_id=key[2], seed=key[4], score_variant=key[5],
                                  primary_score=primary, status=status, status_reason=row.get("status_reason") or None,
                                  vus_pr=float(score["vus_pr"]) if score is not None else None,
                                  physical_execution_id=physical_id, training_group_id=panel.get("training_group_id") or physical_id,
                                  **{field: row.get(field) or None for field in ("score_file", "score_sha256", "metadata_file", "metadata_sha256")},
                                  manifest_reference=f"{Path(manifest_path).resolve()}#key={_json(key)}",
                                  ledger_reference=f"{Path(ledger_path).resolve()}#key={_json(ledger_key)}" if score is not None else None,
                                  evaluator_sha256=score["evaluator_sha256"] if score is not None else None,
                                  ell_max_id=score["ell_max_id"] if score is not None else None)
                    previous = self.connection.execute("SELECT * FROM results WHERE prefix_feature_id=? AND config_id=? AND seed=? AND score_variant=?",
                                                       (prefix, key[2], key[4], key[5])).fetchone()
                    if previous is not None and previous["status"] == "complete":
                        if not complete or any(previous[field] != values[field] for field in ("score_file", "score_sha256", "metadata_file", "metadata_sha256")):
                            raise ValueError("refusing to replace an already scored result with different execution evidence")
                        if score is None:
                            continue
                        if any(previous[field] != values[field] for field in ("vus_pr", "evaluator_sha256", "ell_max_id")):
                            raise ValueError("already scored recommendation row differs from the current ledger")
                    self.connection.execute(
                        f"INSERT INTO results ({','.join(RESULT_COLUMNS)}) VALUES ({','.join('?' for _ in RESULT_COLUMNS)}) "
                        "ON CONFLICT(prefix_feature_id,config_id,seed,score_variant) DO UPDATE SET "
                        + ",".join(f"{field}=excluded.{field}" for field in RESULT_COLUMNS[4:]),
                        tuple(values[field] for field in RESULT_COLUMNS),
                    )
            if used_ledger != ledger.keys():
                raise ValueError("recommendation ledger includes scores absent from the actual manifest")

    def status(self):
        for row in self.connection.execute("SELECT * FROM structural_exclusions"):
            if self.expected_exclusions.get(tuple(row)[:2]) != tuple(row)[2:]:
                raise ValueError("recommendation exclusion differs from the sealed feasibility evidence")
        actual_results = {}
        for row in self.connection.execute("SELECT prefix_feature_id,config_id,seed,score_variant,primary_score,status FROM results"):
            key = tuple(row)[:4]
            if key not in self.expected_results or row["primary_score"] != self.expected_results[key]:
                raise ValueError("recommendation database contains an unexpected result key")
            actual_results[key] = row["status"]
        counts = {"prefix": self.connection.execute("SELECT count(*) FROM prefix_features").fetchone()[0],
                  "channel": self.connection.execute("SELECT count(*) FROM channel_features").fetchone()[0],
                  "exclusion": self.connection.execute("SELECT count(*) FROM structural_exclusions").fetchone()[0],
                  "result": sum(value in ("complete", "executed_pending_score", "executed_unscored") for value in actual_results.values()),
                  "primary_result": sum(state == "complete" for state in actual_results.values())}
        state = {"schema_version": SCHEMA_VERSION, "identity_sha256": _digest(self.identity),
                 "recorded_result_count": len(actual_results)}
        for name, count in counts.items():
            expected = self.contract[f"expected_{name}_count"]
            if count > expected:
                raise ValueError("recommendation database exceeds its sealed expected row counts")
            state.update({f"expected_{name}_count": expected, f"completed_{name}_count": count,
                          f"missing_{name}_count": expected - count})
        state["environment_bound"] = self.connection.execute("SELECT environment_json IS NOT NULL FROM experiment_identity").fetchone()[0] == 1
        state["feature_complete"] = all(state[f"missing_{name}_count"] == 0 for name in ("prefix", "channel", "exclusion"))
        state["execution_complete"] = state["missing_result_count"] == 0 and all(
            value in ("complete", "executed_pending_score", "executed_unscored") for value in actual_results.values())
        state["scoring_complete"] = state["missing_primary_result_count"] == 0
        state["recommendation_complete"] = all(state[key] for key in (
            "environment_bound", "feature_complete", "execution_complete", "scoring_complete"))
        return state

    def _compare_source_rows(self, rows, path, key_fields, value_fields):
        with Path(path).open(encoding="utf-8-sig", newline="") as source:
            originals = {}
            for original in csv.DictReader(source):
                key = tuple(str(original[field]) for field in key_fields)
                if key in originals:
                    raise ValueError("recommendation source table contains duplicate keys")
                originals[key] = original
        for row in rows:
            key = tuple(str(row[field]) for field in key_fields)
            original = originals.get(key)
            if original is None:
                raise ValueError("recommendation result is absent from its persisted source table")
            for field in value_fields:
                same = (float(original[field]) == float(row[field]) if field == "vus_pr"
                        else original[field] == str(row.get(field) or ""))
                if not same:
                    raise ValueError("recommendation result differs from its persisted source table")

    def _validate_result_sources(self):
        manifests, ledgers = {}, {}
        for row in self.connection.execute("""SELECT r.*,p.series,p.q_percent,c.model FROM results r
                JOIN prefix_features p USING(prefix_feature_id) JOIN model_configs c USING(config_id)"""):
            path, encoded_key = row["manifest_reference"].rsplit("#key=", 1)
            key = tuple(json.loads(encoded_key))
            if key not in self.panel or (key[0], key[1], key[2], key[4], key[5]) != (
                row["series"], row["model"], row["config_id"], row["seed"], row["score_variant"]
            ) or row["q_percent"] not in self.panel[key]["logical_ratios"]:
                raise ValueError("recommendation database source reference has a different logical key")
            manifest = {**dict(row), "physical_ratio": key[3], "budget_id": self.budget["budget_id"],
                        "primary_score": str(bool(row["primary_score"])).lower(),
                        "status": "complete" if row["status"] in ("complete", "executed_pending_score", "executed_unscored") else row["status"]}
            by_key = manifests.setdefault(path, {})
            previous = by_key.get(key)
            if previous is not None and any(previous[field] != manifest[field] for field in (
                "status", "score_file", "score_sha256", "metadata_file", "metadata_sha256",
                "physical_execution_id", "training_group_id", "primary_score",
            )):
                raise ValueError("logical q copies disagree about their shared physical execution evidence")
            by_key[key] = manifest
            if row["status"] == "complete":
                ledger_path, encoded_ledger_key = row["ledger_reference"].rsplit("#key=", 1)
                ledger_key = (row["series"], row["config_id"], row["q_percent"], row["seed"], row["score_variant"])
                if tuple(json.loads(encoded_ledger_key)) != ledger_key:
                    raise ValueError("recommendation ledger reference has a different logical key")
                ledgers.setdefault(ledger_path, []).append({**dict(row), "ratio": row["q_percent"], "normalization": "trainnorm"})
        source_files = []
        for path, rows in manifests.items():
            self._compare_source_rows(rows.values(), path,
                ("series", "model", "config_id", "physical_ratio", "seed", "score_variant"),
                ("budget_id", "primary_score", "status", "score_file", "score_sha256", "metadata_file", "metadata_sha256"))
            for key, row in rows.items():
                if row["status"] == "complete":
                    metadata = self._verify_execution_files(row)
                    execution_id = metadata["execution_attempt"]["run_id"]
                    if (row["physical_execution_id"] != execution_id
                            or row["training_group_id"] != (self.panel[key].get("training_group_id") or execution_id)):
                        raise ValueError("recommendation execution IDs differ from the saved run evidence")
            source_files.append({"file": path, "sha256": file_sha256(path)})
        for path, rows in ledgers.items():
            self._compare_source_rows(rows, path, ("series", "config_id", "ratio", "seed", "score_variant"),
                ("model", "status", "score_file", "score_sha256", "normalization", "evaluator_sha256", "ell_max_id", "vus_pr"))
            source_files.append({"file": path, "sha256": file_sha256(path)})
        return source_files

    def finalize(self, *, require_complete=True):
        self._require_parent()
        self.require_features()
        state = self.status()
        if require_complete and not state["recommendation_complete"]:
            raise ValueError("recommendation execution/scoring evidence is incomplete")
        source_files = self._validate_result_sources()
        self.connection.commit()
        backup_path = self.directory / "recommendation_backup.sqlite3"
        temporary = backup_path.with_name(f".{backup_path.name}.{os.getpid()}.tmp")
        try:
            with closing(sqlite3.connect(temporary)) as backup:
                self.connection.backup(backup)
                if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("recommendation SQLite backup failed integrity verification")
            backup_sha256 = file_sha256(temporary)
            backup_path = self.directory / f"recommendation_backup_{backup_sha256}.sqlite3"
            if backup_path.exists():
                if file_sha256(backup_path) != backup_sha256:
                    raise ValueError("existing recommendation backup fingerprint is corrupt")
            else:
                temporary.replace(backup_path)
        finally:
            temporary.unlink(missing_ok=True)
        exported = []
        with closing(sqlite3.connect(f"{backup_path.resolve().as_uri()}?mode=ro", uri=True)) as snapshot:
            queries = (
                ("recommendation_inputs.csv", RECOMMENDATION_HEADERS,
                 f"SELECT {','.join(RECOMMENDATION_HEADERS)} FROM recommendation_inputs ORDER BY series,q_percent"),
                ("prefix_features.csv", PREFIX_HEADERS,
                 "SELECT prefix_feature_id,csv_file,q_percent,observed_row,constant_channel_count,"
                 "channel_std_median,channel_acf_lag1_median,absolute_correlation_median,"
                 + ",".join(f"channel_{field}_median" for field in DESCRIPTOR_COLUMNS)
                 + " FROM prefix_features ORDER BY series,q_percent"),
                ("channel_features.csv", CHANNEL_HEADERS,
                 "SELECT c.prefix_feature_id,c.channel_name,c.mean,c.std,c.median,c.acf_lag1,"
                 + ",".join(f"c.{field}" for field in DESCRIPTOR_COLUMNS)
                 + " FROM channel_features c JOIN prefix_features p USING(prefix_feature_id) ORDER BY p.series,p.q_percent,c.channel_index"),
                ("model_configs.csv", CONFIG_HEADERS, "SELECT config_id,model,settings_json FROM model_configs ORDER BY model,config_id"),
            )
            for filename, headers, query in queries:
                export = self._export_csv(f"{backup_sha256}/{filename}", headers, snapshot.execute(query))
                if filename == "recommendation_inputs.csv" and export["rows"] != self.contract["expected_prefix_count"]:
                    raise ValueError("recommendation input export has missing or duplicate prefix rows")
                exported.append(export)
            combinations = snapshot.execute("SELECT DISTINCT p.csv_file,r.score_variant FROM results r JOIN prefix_features p USING(prefix_feature_id) ORDER BY p.csv_file,r.score_variant").fetchall()
            for csv_file, variant in combinations:
                if variant not in ("", "time", "fft", "pred", "raw_max", "ensemble"):
                    raise ValueError("unknown score head cannot be exported")
                filename = f"performance/{csv_file}__{variant or 'default'}.csv"
                rows = snapshot.execute("""SELECT p.csv_file,p.q_percent,p.training_boundary,p.observed_row,p.input_column,
                    p.prefix_feature_id,c.model,r.config_id,r.seed,r.status,r.vus_pr FROM results r
                    JOIN prefix_features p USING(prefix_feature_id) JOIN model_configs c USING(config_id)
                    WHERE p.csv_file=? AND r.score_variant=? ORDER BY p.q_percent,c.model,r.config_id,r.seed""", (csv_file, variant))
                exported.append(self._export_csv(f"{backup_sha256}/{filename}", PERFORMANCE_HEADERS, rows))
            fingerprints = {}
            for table, order in (("prefix_features", "series,q_percent"), ("channel_features", "prefix_feature_id,channel_index"),
                                 ("model_configs", "config_id"), ("results", "prefix_feature_id,config_id,seed,score_variant"),
                                 ("structural_exclusions", "prefix_feature_id,config_id")):
                digest = hashlib.sha256()
                for row in snapshot.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                    digest.update((_json(list(row)) + "\n").encode("utf-8"))
                fingerprints[table] = digest.hexdigest()
        receipt = {**state, "status": "complete" if state["recommendation_complete"] else "incomplete",
                   "identity": self.identity, "environment": self._environment(),
                   "recommendation_input_contract": self.contract["recommendation_input_contract"],
                   "database": {"file": str(backup_path.resolve()), "sha256": file_sha256(backup_path), "bytes": backup_path.stat().st_size},
                   "table_fingerprints": fingerprints, "exports": exported,
                   "source_tables": source_files,
                   "export_sha256": _digest(exported), "feature_scope": "overlapping CSV/q prefixes, not independent datasets"}
        receipt_path = self.directory / "recommendation_receipt.json"
        temporary = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as output:
                output.write(_json(receipt) + "\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(receipt_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {**receipt, "receipt_file": str(receipt_path.resolve()), "receipt_sha256": file_sha256(receipt_path)}

    def _export_csv(self, filename, headers, rows):
        path = self.directory / "exports" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        count = 0
        try:
            with temporary.open("w", encoding="utf-8-sig", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(headers)
                for row in rows:
                    writer.writerow(row)
                    count += 1
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"file": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size, "rows": count}
