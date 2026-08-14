"""src/common/naming.py 단위 테스트. 저장소 루트에서 python -m unittest discover -s tests 로 실행."""

import unittest

from src.common.naming import build_score_filename, parse_score_filename


class TestBuildScoreFilename(unittest.TestCase):
    def test_matches_claude_md_example(self):
        filename = build_score_filename("GHL", 3, "GDN", "t2", 10, 1, "raw", "trainnorm")
        self.assertEqual(filename, "GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy")

    def test_zero_padding_boundaries(self):
        filename = build_score_filename("GHL", 1, "GDN", "t2", 5, 1, "raw", "trainnorm")
        self.assertIn("__01__", filename)
        self.assertIn("__r005__", filename)

    def test_channels_suffix(self):
        filename = build_score_filename("GHL", 3, "GDN", "t2", 10, 1, "smoothed", "testnorm", channels=True)
        self.assertTrue(filename.endswith("__smoothed__testnorm__channels.npy"))

    def test_rejects_invalid_arguments(self):
        valid = dict(dataset="GHL", series=3, model="GDN", tier="t2", ratio=10, seed=1,
                     smoothing_kind="raw", norm_kind="trainnorm")
        for broken in (
            {**valid, "smoothing_kind": "smooth"},
            {**valid, "norm_kind": "valnorm"},
            {**valid, "ratio": 7},
            {**valid, "series": 100},
            {**valid, "dataset": "GH__L"},
        ):
            with self.assertRaises(ValueError):
                build_score_filename(**broken)


class TestParseScoreFilename(unittest.TestCase):
    def test_round_trip_identity(self):
        for channels in (False, True):
            original = build_score_filename("HAI", 12, "GDN", "t2", 50, 7, "smoothed", "testnorm", channels=channels)
            self.assertEqual(build_score_filename(**parse_score_filename(original)), original)

    def test_parsed_fields(self):
        parsed = parse_score_filename("GHL__03__GDN__t2__r010__s1__raw__trainnorm.npy")
        self.assertEqual(parsed, {
            "dataset": "GHL", "series": 3, "model": "GDN", "tier": "t2",
            "ratio": 10, "seed": 1, "smoothing_kind": "raw", "norm_kind": "trainnorm",
            "channels": False,
        })

    def test_rejects_malformed_filenames(self):
        for broken in (
            "GHL__03__GDN__t2__r010__s1__raw__trainnorm.txt",
            "GHL__03__GDN__t2__r010__s1__raw.npy",
            "GHL__3__GDN__t2__r010__s1__raw__trainnorm.npy",
            "GHL__03__GDN__t2__r07__s1__raw__trainnorm.npy",
            "GHL__03__GDN__t2__r010__s1__smooth__trainnorm.npy",
        ):
            with self.assertRaises(ValueError):
                parse_score_filename(broken)


if __name__ == "__main__":
    unittest.main()
