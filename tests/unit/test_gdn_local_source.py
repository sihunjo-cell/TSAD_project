"""프로젝트에 포함한 GraGOD GDN 소스의 최소 계약."""

import unittest
from pathlib import Path


GDN_DIRECTORY = Path("src/models/tier2/GDN")


class TestGdnLocalSource(unittest.TestCase):
    def test_sources_are_flat_and_keep_the_axis_fix(self):
        expected = {
            "model.py", "modules.py", "trainer.py", "callbacks.py",
            "dataset.py", "graph.py", "types.py", "LICENSE",
        }
        self.assertTrue(expected <= {path.name for path in GDN_DIRECTORY.iterdir()})
        self.assertFalse((GDN_DIRECTORY / "vendor").exists())

        model_source = (GDN_DIRECTORY / "model.py").read_text(encoding="utf-8")
        self.assertIn("from .trainer import PLBaseModule", model_source)
        self.assertIn("from .modules import GNNLayer, OutLayer", model_source)
        self.assertEqual(model_source.count("x.permute(0, 2, 1).contiguous()"), 2)
        self.assertNotIn("x.reshape(-1, x.size(2), x.size(1))", model_source)
        self.assertEqual(model_source.count("SlidingWindowDataset"), 2)

        self.assertIn(
            "from .types import PathType",
            (GDN_DIRECTORY / "trainer.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "from .types import CleanMethods",
            (GDN_DIRECTORY / "dataset.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "MIT License",
            (GDN_DIRECTORY / "LICENSE").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
