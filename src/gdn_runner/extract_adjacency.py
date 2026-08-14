"""best.ckpt 의 embedding 으로 인접 edge 집합 재계산.

근거: D-08 — best validation 시점의 그래프는 embedding 에만 의존하므로 forward 없이
결정적으로 재계산 가능(SHAPEFLOW 1부 a). 로직 출처: gragod-fork
models/gdn/model.py:137-144 (cos-sim → torch.topk) + 146-155 (edge = (j=이웃, i=대상)).
self-edge 제거 전/후를 모두 반환한다 — Jaccard 계산은 제거 후 집합 기준(D-08·D-18).

exp03 에서 쓰는 함수지만 GDN 체크포인트 구조 의존이라 러너 패키지에 둔다.
"""

import torch


def extract_best_adjacency(checkpoint_path: str, topk: int) -> tuple[set, set]:
    """(self-edge 포함 집합, 제거 집합) 반환. edge 는 (source j, target i) 튜플."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # GDN_PLModule 은 GDN 을 self.model 로 들고 있으므로 키가 "model." 접두 (trainer.py:41)
    embedding_weight = checkpoint["state_dict"]["model.embedding.weight"]

    # 이하 model.py:139-144 재현: cos-sim 행렬 → topk 이웃 인덱스
    cos_similarity = torch.matmul(embedding_weight, embedding_weight.T)
    norms = embedding_weight.norm(dim=-1)
    cos_similarity = cos_similarity / torch.matmul(norms.view(-1, 1), norms.view(1, -1))
    topk_indices = torch.topk(cos_similarity, topk, dim=-1)[1]

    # model.py:146-155: 대상 노드 i 의 이웃 j → edge (j, i)
    edges_with_self = {
        (int(neighbor_index), int(node_index))
        for node_index in range(embedding_weight.shape[0])
        for neighbor_index in topk_indices[node_index]
    }
    edges_without_self = {  # D-08: 상수 self-edge 는 Jaccard 를 부풀리므로 제거본 별도 제공
        (source, target) for source, target in edges_with_self if source != target
    }
    return edges_with_self, edges_without_self
