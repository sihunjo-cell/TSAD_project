"""github.com/jinnnju/PaAno commit d4c67116190efa4592dc6a8a157ced0def68b6af의 최소 코어."""

import copy
import math

import numpy
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset


_MEMORY_DISTANCE_BLOCK_SIZE = 1024


class RevIN1d(nn.Module):
    def __init__(self, channel_count: int, epsilon: float = 1e-5):
        super().__init__()
        self.channel_count = channel_count
        self.epsilon = epsilon

    def forward(self, values):
        means = values.mean(dim=-1, keepdim=True)
        variances = values.var(dim=-1, unbiased=False, keepdim=True)
        return (values - means) / (variances + self.epsilon).sqrt().clamp_min(1e-5)


class PatchEncoder(nn.Module):
    """원 구현의 RevIN-1D-CNN encoder와 두 학습 head."""

    def __init__(
        self,
        in_channels: int = 1,
        projection_dim: int = 256,
        use_revin: bool = True,
    ):
        super().__init__()
        widths = (128, 256, 128, 64)
        kernels = (7, 5, 3, 3)
        blocks = []
        input_width = in_channels
        for output_width, kernel_size in zip(widths, kernels):
            blocks.append(nn.Sequential(
                nn.Conv1d(
                    input_width,
                    output_width,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(output_width),
                nn.ReLU(inplace=True),
            ))
            input_width = output_width
        self.revin = RevIN1d(in_channels) if use_revin else None
        self.convblocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection_head = nn.Sequential(
            nn.Linear(widths[-1], projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.classification_head = nn.Linear(widths[-1] * 2, 1)

    def embedding(self, patches):
        hidden = self.revin(patches) if self.revin is not None else patches
        for block in self.convblocks:
            hidden = block(hidden)
        return self.pool(hidden).flatten(start_dim=1)

    def projection(self, embeddings):
        return self.projection_head(embeddings)

    def forward(self, patches):
        return self.embedding(patches)


def choose_training_batch_size(patch_count: int, requested_batch_size: int):
    if patch_count < 3 or requested_batch_size < 2:
        raise ValueError("PaAno batch에는 patch가 2개 이상 들어가야 한다")
    batch_size = min(patch_count, requested_batch_size)
    if patch_count > batch_size and patch_count % batch_size == 1:
        batch_size = batch_size - 1 if batch_size > 2 else batch_size + 1
    return batch_size


def train_encoder(
    model,
    fit_patches,
    *,
    iterations: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    official_procedure: bool = False,
):
    """원 구현의 triplet·초기 pretext 목적을 정해진 iteration만큼 학습한다."""
    if len(fit_patches) < 3:
        raise ValueError("PaAno 학습에는 patch가 3개 이상 필요하다")
    indices = torch.arange(len(fit_patches), dtype=torch.long)
    effective_batch_size = batch_size if official_procedure else choose_training_batch_size(
        len(fit_patches), batch_size,
    )
    if official_procedure and (batch_size < 2 or (
        len(fit_patches) % batch_size == 1
        and iterations >= (len(fit_patches) + batch_size - 1) // batch_size
    )):
        raise ValueError("PaAno official batches cannot contain a singleton; the requested batch size is preserved")
    loader = DataLoader(
        TensorDataset(fit_patches, indices),
        batch_size=effective_batch_size,
        shuffle=True,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay,
    )
    classification_loss = nn.BCEWithLogitsLoss(reduction="none")
    neighbor_offsets = torch.tensor((-2, -1, 1, 2), dtype=torch.long)
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    selected_iteration = None
    loss_history = []
    completed = 0

    while completed < iterations:
        for anchors, anchor_indices in loader:
            if completed >= iterations:
                break
            completed += 1
            progress = min(completed, iterations) / iterations
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            current_rate = learning_rate / 10 + learning_rate * 0.9 * cosine
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = current_rate

            anchors = anchors.to(device)
            anchor_indices = anchor_indices.squeeze()
            anchor_count = len(anchors)
            candidates = anchor_indices[:, None] + neighbor_offsets[None, :]
            valid = (candidates >= 0) & (candidates < len(fit_patches))
            choices = torch.where(
                valid,
                torch.rand(candidates.shape),
                torch.full(candidates.shape, -1.0),
            ).argmax(dim=1)
            positive_indices = candidates.gather(1, choices[:, None]).squeeze(1)
            positives = fit_patches[positive_indices].to(device)
            pretext_weight = max(0.0, 1 - completed / max(1, iterations / 5))
            prior_indices = anchor_indices - anchors.shape[-1]
            usable_prior = prior_indices >= 0
            if pretext_weight:
                pretext_patches = torch.zeros_like(anchors)
                pretext_patches[usable_prior] = fit_patches[
                    prior_indices[usable_prior]
                ].to(device)
                all_patches = torch.cat((anchors, positives, pretext_patches), dim=0)
                all_embeddings = model.embedding(all_patches)
                anchor_embeddings = all_embeddings[:anchor_count]
                positive_embeddings = all_embeddings[
                    anchor_count:2 * anchor_count
                ]
                pretext_embeddings = all_embeddings[2 * anchor_count:]
            else:
                all_embeddings = model.embedding(torch.cat((anchors, positives), dim=0))
                anchor_embeddings = all_embeddings[:anchor_count]
                positive_embeddings = all_embeddings[anchor_count:]
            anchor_projection = functional.normalize(model.projection(anchor_embeddings), dim=1)
            positive_projection = functional.normalize(model.projection(positive_embeddings), dim=1)
            similarities = anchor_projection @ positive_projection.T
            positive_distance = 1 - similarities.diagonal()
            negative_similarities = similarities.clone()
            negative_similarities.diagonal().fill_(float("inf"))
            hardest_negative_distance = (1 - negative_similarities).max(dim=1).values
            triplet_loss = functional.relu(
                positive_distance - hardest_negative_distance + 0.1,
            ).mean() / 10

            if pretext_weight:
                adjacent = torch.cat((
                    anchor_embeddings[usable_prior],
                    pretext_embeddings[usable_prior],
                ), dim=1)
                anchor_rows = torch.arange(anchor_count, device=device).repeat_interleave(5)
                random_offsets = torch.randint(
                    1, anchor_count, (anchor_count * 5,), device=device,
                )
                nonadjacent_rows = (anchor_rows + random_offsets) % anchor_count
                nonadjacent = torch.cat((
                    anchor_embeddings.repeat_interleave(5, dim=0),
                    anchor_embeddings[nonadjacent_rows],
                ), dim=1)
                features = torch.cat((adjacent, nonadjacent), dim=0)
                targets = torch.cat((
                    torch.ones(len(adjacent), device=device),
                    torch.zeros(len(nonadjacent), device=device),
                ))
                losses = classification_loss(
                    model.classification_head(features).squeeze(1), targets,
                )
                adjacent_loss = (
                    losses[:len(adjacent)].mean()
                    if len(adjacent) or official_procedure
                    else torch.zeros((), device=device)
                )
                nonadjacent_loss = losses[len(adjacent):].mean()
                pretext_loss = adjacent_loss + nonadjacent_loss
            else:
                pretext_loss = torch.zeros((), device=device)

            loss = triplet_loss + pretext_weight * pretext_loss
            if official_procedure and not torch.isfinite(loss):
                raise RuntimeError("PaAno official training loss must remain finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss = loss.item()
            loss_history.append({
                "iteration": completed,
                "total_loss": total_loss,
                "triplet_loss": triplet_loss.item(),
                "pretext_loss": pretext_loss.item(),
                "pretext_weight": pretext_weight,
                "learning_rate": current_rate,
            })
            if total_loss < best_loss:
                best_loss = total_loss
                best_state = copy.deepcopy(model.state_dict())
                selected_iteration = completed

    model.load_state_dict(best_state)
    return {
        "optimizer_updates": completed,
        "best_training_loss": best_loss,
        "iterations_completed": completed,
        "selected_iteration": selected_iteration,
        "loss_history": loss_history,
    }


@torch.inference_mode()
def encode_patches(model, patches, device: str, *, batch_size: int = 512,
                   shuffle: bool = False):
    model.to(device).eval()
    embeddings = []
    for (batch,) in DataLoader(
        TensorDataset(patches), batch_size=batch_size, shuffle=shuffle,
    ):
        embeddings.append(model.embedding(batch.to(device)).detach().cpu().float())
    return torch.cat(embeddings)


def select_memory_count(patch_count: int, fraction: float = 0.1,
                        memory_policy: str = "legacy_fraction") -> int:
    if patch_count < 1 or not 0 < fraction <= 1:
        raise ValueError("PaAno memory 입력이나 fraction이 잘못됐다")
    if memory_policy == "legacy_fraction":
        memory_count = int(patch_count * fraction)
    elif memory_policy == "paper_fraction":
        memory_count = min(round(fraction * patch_count), patch_count - 1)
    elif memory_policy == "official_minimum":
        memory_count = max(
            min(500, max(1, patch_count - 1)),
            min(round(fraction * patch_count), patch_count - 1),
        )
    else:
        raise ValueError(f"알 수 없는 PaAno memory_policy: {memory_policy}")
    if memory_count < 3:
        raise ValueError("PaAno top-3 점수에는 memory가 3개 이상이어야 한다")
    return memory_count


def select_memory_bank(embeddings, fraction: float = 0.1, random_seed: int = 42,
                       memory_policy: str = "legacy_fraction"):
    """MiniBatchKMeans 중심마다 가장 가까운 fit embedding 하나를 고른다."""
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32).detach().cpu()
    if embeddings.ndim != 2 or not 0 < fraction <= 1:
        raise ValueError("PaAno memory 입력이나 fraction이 잘못됐다")
    memory_count = select_memory_count(len(embeddings), fraction, memory_policy)
    normalized = functional.normalize(embeddings, dim=1, eps=1e-12)
    clustering = MiniBatchKMeans(
        n_clusters=memory_count,
        init="k-means++",
        random_state=random_seed,
        batch_size=max(8192, memory_count),
        max_iter=50,
        n_init=1,
        reassignment_ratio=0.01,
    ).fit(normalized.numpy())
    centers = torch.from_numpy(clustering.cluster_centers_).to(normalized)
    representatives = torch.empty(memory_count, dtype=torch.long)
    # Edge blocks use the same distance algorithm as the full input.
    compute_mode = (
        "use_mm_for_euclid_dist" if len(normalized) > 25 or memory_count > 25
        else "donot_use_mm_for_euclid_dist"
    )
    for center_start in range(0, memory_count, _MEMORY_DISTANCE_BLOCK_SIZE):
        center_block = centers[center_start:center_start + _MEMORY_DISTANCE_BLOCK_SIZE]
        best_distances = normalized.new_full((len(center_block),), float("inf"))
        best_indices = torch.zeros(len(center_block), dtype=torch.long)
        for patch_start in range(0, len(normalized), _MEMORY_DISTANCE_BLOCK_SIZE):
            distances = torch.cdist(
                normalized[patch_start:patch_start + _MEMORY_DISTANCE_BLOCK_SIZE],
                center_block, compute_mode=compute_mode,
            )
            block_distances, block_indices = distances.min(dim=0)
            improved = block_distances < best_distances
            best_distances[improved] = block_distances[improved]
            best_indices[improved] = patch_start + block_indices[improved]
        representatives[center_start:center_start + len(center_block)] = best_indices
    return embeddings[representatives].clone()


@torch.inference_mode()
def _score_batch(embeddings, normalized_memory, top_k: int):
    embeddings = functional.normalize(
        torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0),
        dim=1,
        eps=1e-12,
    )
    embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    if normalized_memory.ndim != 2 or len(normalized_memory) < top_k:
        raise ValueError("PaAno memory bank must contain at least top_k representatives")
    similarities = torch.nan_to_num(
        embeddings @ normalized_memory.T, nan=-1.0, posinf=1.0, neginf=-1.0,
    )
    nearest = torch.topk(similarities, k=top_k, dim=1).values
    return torch.nan_to_num(
        (1 - nearest).mean(dim=1), nan=1.0, posinf=1.0, neginf=0.0,
    ).cpu()


@torch.inference_mode()
def score_embeddings(embeddings, memory_bank, top_k: int = 3):
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    normalized_memory = functional.normalize(
        torch.as_tensor(memory_bank, dtype=torch.float32, device=embeddings.device),
        dim=1, eps=1e-12,
    )
    return _score_batch(embeddings, normalized_memory, top_k)


@torch.inference_mode()
def score_patches(model, patches, memory_bank, *, device: str,
                  top_k: int = 3, batch_size: int = 512):
    model.to(device).eval()
    normalized_memory = functional.normalize(
        torch.as_tensor(memory_bank, dtype=torch.float32, device=device), dim=1, eps=1e-12,
    )
    scores = []
    for (batch,) in DataLoader(TensorDataset(patches), batch_size=batch_size, shuffle=False):
        embeddings = model.embedding(batch.to(device=device, dtype=torch.float32))
        scores.append(_score_batch(embeddings, normalized_memory, top_k))
    return torch.cat(scores)


__all__ = [
    "PatchEncoder",
    "choose_training_batch_size",
    "encode_patches",
    "score_embeddings",
    "score_patches",
    "select_memory_bank",
    "select_memory_count",
    "train_encoder",
]
