"""Dev18 등록 입력이 봉인된 이질 CSV를 label-free로 읽는지 검증한다."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy
import pandas
import yaml

from src.data_split.load_dev18_series import load_dev18_registered_inputs
from tests.ghl_main.run_registered_models import load_registered_inputs


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_DEV18 = (
    ("01", "MSL", "004_MSL_id_3_Sensor_tr_530_1st_630.csv", 530, 577095,
     "8c66d4a5525db15840d4d58d881d647ce1e61a5388ebc603ecf2eadbd6c046a8"),
    ("02", "MSL", "011_MSL_id_10_Sensor_tr_1525_1st_4590.csv", 1525, 1445433,
     "f266d167f0db1fd709423ecb59c01b567acd99f4d97e2df81d74d7d22f0b7b1c"),
    ("03", "MITDB", "023_MITDB_id_5_Medical_tr_25000_1st_36913.csv", 25000, 1484351,
     "130f7c4404f8bd5807cde20e69478dd176c8da976bccef82c2f02c5386a377b7"),
    ("04", "MITDB", "028_MITDB_id_10_Medical_tr_37500_1st_39948.csv", 37500, 2140360,
     "b7bcf3bbe8215bb35da728da1b2fc9d49c2200f696bf8daba8a32f792b9b0dc7"),
    ("05", "SMD", "062_SMD_id_6_Facility_tr_7180_1st_15131.csv", 7180, 7811204,
     "dda20568da0041b19f12eaf05d6f5dc88f37283c83d4b2f3a3816212849de577"),
    ("06", "SMD", "072_SMD_id_16_Facility_tr_7119_1st_15849.csv", 7119, 8399285,
     "2c1a273cc2e8bb9742d00c8be0632217275053b528007d1430a59c1f01e3d8e9"),
    ("07", "LTDB", "082_LTDB_id_4_Medical_tr_4456_1st_4556.csv", 4456, 1261256,
     "698edd323bc804f797cf5d8e4222d70844f10202081d4379543f162315f43183"),
    ("08", "SVDB", "090_SVDB_id_7_Medical_tr_12157_1st_12257.csv", 12157, 628354,
     "443e3e64f65cb7c4002c4e149dc12997889d7f2e74cfdb275b023a3a2b4ab846"),
    ("09", "SVDB", "107_SVDB_id_24_Medical_tr_32805_1st_32905.csv", 32805, 3200090,
     "79af5662ff1534500c68e74263822019bd2f0ed25d82f46e46f8779888730219"),
    ("10", "SVDB", "113_SVDB_id_30_Medical_tr_4552_1st_4652.csv", 4552, 2945634,
     "c79e1884bc009b1b0d11ca8b27c5c4d33a535f949de2ee4851ae81e5e422eda7"),
    ("11", "TAO", "120_TAO_id_5_Environment_tr_500_1st_3.csv", 500, 211593,
     "5926d15432d2b67d7348b6107c06daa7707b78f3994e0d9fb11b70afc74a6763"),
    ("12", "TAO", "126_TAO_id_11_Environment_tr_500_1st_7.csv", 500, 211767,
     "bb5df8d457b6b06ae6d9386651252294e22f9a5c3ffe58264c47a9b3b1b7679e"),
    ("13", "OPPORTUNITY", "131_OPPORTUNITY_id_3_HumanActivity_tr_7016_1st_26691.csv", 7016, 41279685,
     "da34489a3117911ac3ca04dcd4e0e2c153e499cd96bcb6add38b050124d796f4"),
    ("14", "CATSv2", "140_CATSv2_id_3_Sensor_tr_28307_1st_28407.csv", 28307, 105415516,
     "c4e9706efd923a595d93ade46d20cad0ac586bc704eec5ec98009afda585db7b"),
    ("15", "SMAP", "149_SMAP_id_6_Sensor_tr_2128_1st_5000.csv", 2128, 903500,
     "06ec1ec91cb267026426f796a26efb9f2b8000a9a7d91384fea187308b968970"),
    ("16", "SMAP", "164_SMAP_id_21_Sensor_tr_1976_1st_4200.csv", 1976, 806601,
     "263494508921547457ab3340b3ee92ec949650b21e2e886021011b348cf510e3"),
    ("17", "Exathlon", "178_Exathlon_id_5_Facility_tr_12538_1st_12638.csv", 12538, 21892023,
     "ade5f3cdbffa2b3464497babe566041dff871bb9d5e9da9d1e17f3fefdedc16c"),
    ("18", "Exathlon", "195_Exathlon_id_22_Facility_tr_10766_1st_12590.csv", 10766, 11893438,
     "2825858c31806a7ed49befa912d5cd078ea36d4dd9f7dae2b7925d7e2c4b1b9d"),
)
EXPECTED_OFFICIAL20_REFERENCES = (
    *(("DEV18", "series", f"{order:02d}") for order in range(1, 5)),
    ("GHL", "name", "040_GHL_id_9_Sensor_tr_50000_1st_92001.csv"),
    ("GHL", "name", "049_GHL_id_18_Sensor_tr_50000_1st_109001.csv"),
    *(("DEV18", "series", f"{order:02d}") for order in range(5, 19)),
)
EXPECTED_DEV18_SHAPES = (
    (2430, 55), (6100, 55), (100000, 2), (150000, 2),
    (28722, 38), (28479, 38), (100000, 2), (50000, 2),
    (230400, 2), (230400, 2), (10000, 3), (10000, 3),
    (28066, 248), (400000, 17), (8512, 25), (7907, 25),
    (129197, 19), (43066, 31),
)


def write_registered_csv(
    data_root: Path, *, feature_count: int = 2, boundary: int = 3,
    rows: int = 6, values=None,
) -> tuple[dict, Path]:
    source_directory = "TSB-AD-M-other-datasets"
    directory = data_root / source_directory
    directory.mkdir(parents=True, exist_ok=True)
    name = f"004_MSL_id_3_Sensor_tr_{boundary}_1st_630.csv"
    path = directory / name
    if values is None:
        values = {
            f"feature_{column:02d}": numpy.arange(rows, dtype=float) + column * 100
            for column in range(feature_count)
        }
    feature_count = len(values)
    frame = pandas.DataFrame(values)
    frame["Label"] = ["must-not-be-read"] * len(frame)
    frame.to_csv(path, index=False)
    feature_names = list(values)
    return {
        "series": "01",
        "family": "MSL",
        "source_directory": source_directory,
        "name": name,
        "training_boundary": boundary,
        "row_count": rows,
        "feature_count": feature_count,
        "feature_names_sha256": hashlib.sha256(json.dumps(
            feature_names, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, path


class TestDev18Manifest(unittest.TestCase):
    def test_preserves_all_non_ghl_files_in_official_relative_order(self):
        manifest = yaml.safe_load(
            (REPOSITORY_ROOT / "configs" / "input_manifest.yaml").read_text(
                encoding="utf-8"
            )
        )
        rows = manifest["datasets"]["DEV18"]["files"]

        actual = tuple(
            (
                row["series"], row["family"], row["name"],
                row["training_boundary"], row["size_bytes"], row["sha256"],
            )
            for row in rows
        )
        self.assertEqual(actual, EXPECTED_DEV18)
        self.assertTrue(all(row["source_directory"] == "tuning" for row in rows))
        self.assertEqual(
            tuple((row["row_count"], row["feature_count"]) for row in rows),
            EXPECTED_DEV18_SHAPES,
        )
        self.assertTrue(all(
            len(row["feature_names_sha256"]) == 64 for row in rows
        ))
        self.assertFalse(any("_GHL_" in row["name"] for row in rows))
        self.assertEqual(
            tuple((row["order"], row["series"]) for row in rows),
            tuple((order, f"{order:02d}") for order in range(1, 19)),
        )

    def test_roles_resolve_the_three_dataset_roles_without_copying_fingerprints(self):
        manifest = yaml.safe_load(
            (REPOSITORY_ROOT / "configs" / "input_manifest.yaml").read_text(
                encoding="utf-8"
            )
        )
        roles = manifest["roles"]
        self.assertEqual(roles["dev18_selection"], {"dataset": "DEV18"})
        self.assertEqual(roles["ghl25_final"], {"dataset": "GHL"})

        references = roles["official20_provenance"]["ordered_members"]
        self.assertEqual(
            tuple(
                (reference["dataset"], *(item for item in reference.items() if item[0] != "dataset"))
                for reference in references
            ),
            tuple(
                (dataset, (field, value))
                for dataset, field, value in EXPECTED_OFFICIAL20_REFERENCES
            ),
        )

        resolved_names = []
        for dataset, field, value in EXPECTED_OFFICIAL20_REFERENCES:
            rows = manifest["datasets"][dataset]["files"]
            matches = [row for row in rows if row.get(field) == value]
            self.assertEqual(len(matches), 1, (dataset, field, value))
            resolved_names.append(matches[0]["name"])
        self.assertEqual(
            resolved_names,
            [row[2] for row in EXPECTED_DEV18[:4]]
            + [
                "040_GHL_id_9_Sensor_tr_50000_1st_92001.csv",
                "049_GHL_id_18_Sensor_tr_50000_1st_109001.csv",
            ]
            + [row[2] for row in EXPECTED_DEV18[4:]],
        )


class TestLoadDev18RegisteredInputs(unittest.TestCase):
    def test_accepts_one_two_and_nineteen_features_without_reading_label_values(self):
        for feature_count in (1, 2, 19):
            with self.subTest(feature_count=feature_count):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    data_root = Path(temporary_directory)
                    entry, path = write_registered_csv(
                        data_root, feature_count=feature_count
                    )
                    calls = []
                    original_read_csv = pandas.read_csv

                    def record_read(*args, **kwargs):
                        calls.append(dict(kwargs))
                        return original_read_csv(*args, **kwargs)

                    with patch(
                        "src.data_split.load_dev18_series.pandas.read_csv",
                        side_effect=record_read,
                    ):
                        inputs = load_dev18_registered_inputs(entry, data_root)

                self.assertEqual(inputs["feature_names"], tuple(
                    f"feature_{column:02d}" for column in range(feature_count)
                ))
                self.assertEqual(inputs["normal_training"].shape, (3, feature_count))
                self.assertEqual(inputs["normal_training"].dtype, numpy.float32)
                self.assertEqual(inputs["test_sessions"][0].shape, (3, feature_count))
                self.assertEqual(inputs["family"], "MSL")
                self.assertEqual(inputs["series"], "01")
                self.assertEqual(inputs["source_ranges"], {
                    "source": path.name,
                    "source_directory": "TSB-AD-M-other-datasets",
                    "normal_training": (0, 3),
                    "test_sessions": ((3, 6),),
                })
                self.assertEqual(inputs["input_identity"], {
                    "name": path.name,
                    "source_directory": "TSB-AD-M-other-datasets",
                    "size_bytes": entry["size_bytes"],
                    "sha256": entry["sha256"],
                })
                value_reads = [call for call in calls if call.get("nrows") != 0]
                self.assertEqual(len(value_reads), 1)
                self.assertNotIn("Label", value_reads[0]["usecols"])

    def test_rejects_missing_or_invalid_manifest_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)
            valid, _ = write_registered_csv(data_root)
            cases = (
                None,
                {key: value for key, value in valid.items() if key != "family"},
                {**valid, "series": 1},
                {**valid, "series": "1"},
                {**valid, "family": ""},
                {**valid, "source_directory": ""},
                {**valid, "name": "nested/" + valid["name"]},
                {**valid, "training_boundary": True},
                {**valid, "row_count": True},
                {**valid, "feature_count": 0},
                {**valid, "feature_names_sha256": "not-a-sha"},
                {**valid, "size_bytes": 0},
                {**valid, "sha256": "not-a-sha"},
            )
            for entry in cases:
                with self.subTest(entry=entry):
                    with self.assertRaises(ValueError):
                        load_dev18_registered_inputs(entry, data_root)

    def test_rejects_boundary_size_sha_and_path_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            parent = Path(temporary_directory)
            data_root = parent / "data"
            valid, _ = write_registered_csv(data_root)

            cases = (
                ({**valid, "training_boundary": 2}, "경계"),
                ({**valid, "row_count": valid["row_count"] + 1}, "행 수"),
                ({**valid, "feature_count": valid["feature_count"] + 1}, "feature 수"),
                ({**valid, "feature_names_sha256": "0" * 64}, "feature 순서"),
                ({**valid, "size_bytes": valid["size_bytes"] + 1}, "크기"),
                ({**valid, "sha256": "0" * 64}, "SHA-256"),
            )
            for entry, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        load_dev18_registered_inputs(entry, data_root)

            outside_entry, _ = write_registered_csv(parent)
            outside_entry["source_directory"] = "..\\TSB-AD-M-other-datasets"
            with self.assertRaisesRegex(ValueError, "root"):
                load_dev18_registered_inputs(outside_entry, data_root)

    def test_rejects_bad_header_values_and_row_boundary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory)

            nonnumeric, _ = write_registered_csv(
                data_root, values={"feature_00": [0, 1, "bad", 3, 4, 5]}
            )
            with self.assertRaisesRegex(ValueError, "숫자"):
                load_dev18_registered_inputs(nonnumeric, data_root)

            nonfinite, _ = write_registered_csv(
                data_root, values={"feature_00": [0, 1, numpy.inf, 3, 4, 5]}
            )
            with self.assertRaisesRegex(ValueError, "비유한"):
                load_dev18_registered_inputs(nonfinite, data_root)

            entry, path = write_registered_csv(data_root)
            pandas.DataFrame({"feature_00": range(6)}).to_csv(path, index=False)
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "Label"):
                load_dev18_registered_inputs(entry, data_root)

            pandas.DataFrame({"Label": range(6)}).to_csv(path, index=False)
            entry["size_bytes"] = path.stat().st_size
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "feature"):
                load_dev18_registered_inputs(entry, data_root)

            boundary_entry, _ = write_registered_csv(data_root, boundary=6, rows=6)
            with self.assertRaisesRegex(ValueError, "행 범위"):
                load_dev18_registered_inputs(boundary_entry, data_root)


class TestRegisteredInputDispatch(unittest.TestCase):
    @staticmethod
    def write_manifest(path: Path, entry: dict, *, rows=None):
        manifest = {
            "roles": {"dev18_selection": {"dataset": "DEV18"}},
            "datasets": {"DEV18": {"files": list(rows or [entry])}},
        }
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    @staticmethod
    def spec(manifest_path: Path) -> dict:
        return {
            "dataset_role": "development", "split_role": "dev18_selection",
            "input_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "final_policy_membership_sha256": None,
        }

    def test_sealed_spec_resolves_dev_series_from_the_same_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            entry, _ = write_registered_csv(root)
            entry["order"] = 1
            manifest_path = root / "manifest.yaml"
            self.write_manifest(manifest_path, entry)
            with patch(
                "tests.ghl_main.run_registered_models.load_dev18_registered_inputs",
                return_value={"loaded": True},
            ) as development_loader:
                inputs = load_registered_inputs(
                    spec=self.spec(manifest_path), series="01", data_root=root,
                    input_manifest_path=manifest_path,
                )

            development_loader.assert_called_once_with(entry, root)
            self.assertTrue(inputs["loaded"])
            self.assertEqual(
                inputs["input_identity"]["input_manifest_sha256"],
                self.spec(manifest_path)["input_manifest_sha256"],
            )

    def test_development_dispatch_rejects_unsealed_or_duplicate_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            entry, _ = write_registered_csv(root)
            entry["order"] = 1
            manifest_path = root / "manifest.yaml"
            self.write_manifest(manifest_path, entry)

            with self.assertRaises(TypeError):
                load_registered_inputs(
                    dataset_role="development", series="01", data_root=root,
                    input_manifest_path=manifest_path,
                )
            with self.assertRaisesRegex(ValueError, "manifest"):
                load_registered_inputs(
                    spec={
                        **self.spec(manifest_path),
                        "input_manifest_sha256": "0" * 64,
                    },
                    series="01", data_root=root,
                    input_manifest_path=manifest_path,
                )

            self.write_manifest(manifest_path, entry, rows=[entry, dict(entry)])
            with self.assertRaisesRegex(ValueError, "유일"):
                load_registered_inputs(
                    spec=self.spec(manifest_path), series="01", data_root=root,
                    input_manifest_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
