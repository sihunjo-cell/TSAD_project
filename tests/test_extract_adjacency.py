"""src/gdn_runner/extract_adjacency.py 단위 테스트 — 손계산 embedding 대조."""

import tempfile
import unittest
from pathlib import Path

import torch

from src.gdn_runner.extract_adjacency import extract_best_adjacency


class TestExtractBestAdjacency(unittest.TestCase):
    def test_topk_and_self_edge_removal_match_hand_calculation(self):
        # N=4, 유사도 순위 손계산:
        # e0=[1,0], e1=[1,0.1], e2=[0,1], e3=[-1,0]
        # cos(e0,e1)≈0.995, cos(e0,e2)=0, cos(e0,e3)=-1, cos(e1,e2)≈0.0995, cos(e2,e3)=0
        # 이 fixture에서는 self 유사도 1이 유일한 최댓값인 topk=2:
        #   노드0 → {0,1} / 노드1 → {1,0} / 노드2 → {2,1} / 노드3 → {3,2}
        embedding = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [-1.0, 0.0]])
        checkpoint = {"state_dict": {"model.embedding.weight": embedding}}

        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "best.ckpt"
            torch.save(checkpoint, checkpoint_path)
            edges_with_self, edges_without_self = extract_best_adjacency(str(checkpoint_path), topk=2)

        expected_with_self = {
            (0, 0), (1, 0),   # 노드0 의 이웃 {0,1}
            (1, 1), (0, 1),   # 노드1 의 이웃 {1,0}
            (2, 2), (1, 2),   # 노드2 의 이웃 {2,1}
            (3, 3), (2, 3),   # 노드3 의 이웃 {3,2}
        }
        self.assertEqual(edges_with_self, expected_with_self)
        self.assertEqual(edges_without_self, {(1, 0), (0, 1), (1, 2), (2, 3)})

    def test_rejects_checkpoint_with_undefined_cosine_similarity(self):
        checkpoint = {
            "state_dict": {
                "model.embedding.weight": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            },
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "best.ckpt"
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "embedding norm"):
                extract_best_adjacency(str(checkpoint_path), topk=1)


if __name__ == "__main__":
    unittest.main()
