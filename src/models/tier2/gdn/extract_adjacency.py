"""best checkpoint의 embedding으로 GDN edge 집합을 다시 계산한다.

HAI 확장 실험에서 쓰지만 GDN 체크포인트 구조에 의존하므로 GDN 패키지에 둔다.
"""

import torch


def extract_best_adjacency(checkpoint_path: str, topk: int) -> tuple[set, set]:
    """(self-edge 포함 집합, 제거 집합) 반환. edge 는 (source j, target i) 튜플."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # checkpoint에서 내부 GDN 모델의 embedding만 꺼낸다.
    embedding_weight = checkpoint["state_dict"]["model.embedding.weight"]
    if embedding_weight.ndim != 2 or not torch.isfinite(embedding_weight).all():
        raise ValueError("GDN embedding은 유한한 2차원 배열이어야 한다")
    if not 1 <= topk <= embedding_weight.shape[0]:
        raise ValueError(f"topk는 1~채널 수 범위여야 한다: {topk}")

    # cosine similarity가 큰 이웃을 노드마다 topk개 고른다.
    cos_similarity = torch.matmul(embedding_weight, embedding_weight.T)
    norms = embedding_weight.norm(dim=-1)
    if torch.any(norms == 0):
        raise ValueError("GDN embedding norm이 0이면 cosine graph를 정의할 수 없다")
    cos_similarity = cos_similarity / torch.matmul(norms.view(-1, 1), norms.view(1, -1))
    topk_indices = torch.topk(cos_similarity, topk, dim=-1)[1]

    # 대상 노드 i의 이웃 j를 방향 edge (j, i)로 저장한다.
    edges_with_self = {
        (int(neighbor_index), int(node_index))
        for node_index in range(embedding_weight.shape[0])
        for neighbor_index in topk_indices[node_index]
    }
    edges_without_self = {  # self-edge를 빼야 seed 간 관계만 비교할 수 있다.
        (source, target) for source, target in edges_with_self if source != target
    }
    return edges_with_self, edges_without_self
