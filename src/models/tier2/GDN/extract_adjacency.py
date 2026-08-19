"""저장한 GDN embedding에서 방향 TopK edge 집합을 복원한다."""

import torch


def extract_best_adjacency(checkpoint_path: str, topk: int) -> tuple[set, set]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
    embedding_weight = state_dict.get(
        "embedding.weight", state_dict.get("model.embedding.weight"),
    )
    if not isinstance(embedding_weight, torch.Tensor) or embedding_weight.ndim != 2:
        raise ValueError("GDN embedding은 2차원 tensor여야 한다")
    if not torch.isfinite(embedding_weight).all():
        raise ValueError("GDN embedding은 모두 유한해야 한다")
    if not 1 <= topk <= embedding_weight.shape[0]:
        raise ValueError(f"topk는 1~채널 수 범위여야 한다: {topk}")

    norms = embedding_weight.norm(dim=-1)
    if torch.any(norms == 0):
        raise ValueError("GDN embedding norm이 0이면 cosine graph를 정의할 수 없다")
    similarities = torch.matmul(embedding_weight, embedding_weight.T)
    similarities /= torch.matmul(norms.view(-1, 1), norms.view(1, -1))
    neighbor_indices = torch.topk(similarities, topk, dim=-1).indices
    edges_with_self = {
        (int(neighbor), node)
        for node in range(embedding_weight.shape[0])
        for neighbor in neighbor_indices[node]
    }
    return edges_with_self, {
        edge for edge in edges_with_self if edge[0] != edge[1]
    }
