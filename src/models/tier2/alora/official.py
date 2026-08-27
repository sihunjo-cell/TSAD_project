"""ALoRa-T architecture adapted from commit 97dcc4a337710e6dc72c1a67893717c9538bae1a.

The upstream EUPL-1.2 architecture is kept here without its dataset, device,
thresholding, or evaluation controller. Project-owned code supplies selected
fit-normal channel pairs and consumes only continuous scores.
"""

import math

import numpy
import torch
from torch import nn
from torch.nn import functional


SOURCE_COMMIT = "97dcc4a337710e6dc72c1a67893717c9538bae1a"
ATTENTION_RANK_THRESHOLD = 0.01


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dimension: int, maximum_length: int = 5000):
        super().__init__()
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.arange(0, embedding_dimension, 2, dtype=torch.float32)
        frequencies = torch.exp(frequencies * -(math.log(10000.0) / embedding_dimension))
        embedding = torch.zeros(maximum_length, embedding_dimension)
        embedding[:, 0::2] = torch.sin(positions * frequencies)
        cosine_columns = embedding[:, 1::2].shape[1]
        embedding[:, 1::2] = torch.cos(positions * frequencies[:cosine_columns])
        self.register_buffer("pe", embedding.unsqueeze(0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : inputs.shape[1]]


class PairTokenEmbedding(nn.Module):
    def __init__(
        self,
        channel_count: int,
        selected_pairs,
        *,
        pair_embedding_dimension: int = 512,
        kernel_size: int = 3,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("ALoRa token kernel_size must be odd")
        pairs = torch.as_tensor(selected_pairs, dtype=torch.long)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
            raise ValueError("ALoRa selected_pairs must have shape (K, 2)")
        if pair_embedding_dimension < 1:
            raise ValueError("ALoRa pair_embedding_dimension must be positive")
        pairs = pairs[:pair_embedding_dimension]
        if torch.any(pairs < 0) or torch.any(pairs >= channel_count):
            raise ValueError("ALoRa selected pair index is out of range")
        if torch.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError("ALoRa selected pairs must be ordered unique-channel pairs")

        self.c_in = channel_count
        self.d_model = len(pairs)
        self.padding = kernel_size // 2
        self.register_buffer("pairs_idx", pairs)
        self.weights = nn.Parameter(torch.randn(self.d_model, 2))
        averaging_kernel = torch.ones(channel_count, 1, kernel_size) / kernel_size
        self.register_buffer("avg_kernel", averaging_kernel)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[2] != self.c_in:
            raise ValueError("ALoRa input must have shape (batch, time, channels)")
        channel_first = inputs.permute(0, 2, 1)
        padded = functional.pad(
            channel_first, (self.padding, self.padding), mode="circular",
        )
        smoothed = functional.conv1d(
            padded,
            self.avg_kernel.to(dtype=inputs.dtype),
            groups=self.c_in,
        )
        first = smoothed[:, self.pairs_idx[:, 0], :]
        second = smoothed[:, self.pairs_idx[:, 1], :]
        first_weights = self.weights[:, 0].reshape(1, -1, 1).to(inputs.dtype)
        second_weights = self.weights[:, 1].reshape(1, -1, 1).to(inputs.dtype)
        return (first_weights * first + second_weights * second).permute(0, 2, 1)


class DataEmbedding(nn.Module):
    def __init__(
        self,
        channel_count: int,
        selected_pairs,
        *,
        pair_embedding_dimension: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.value_embedding = PairTokenEmbedding(
            channel_count,
            selected_pairs,
            pair_embedding_dimension=pair_embedding_dimension,
        )
        self.d_model = self.value_embedding.d_model
        self.position_embedding = PositionalEmbedding(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.value_embedding(inputs)
        return self.dropout(values + self.position_embedding(values))


class ALoRaAttention(nn.Module):
    def __init__(self, *, scale=None, dropout: float = 0.0):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.scale or 1.0 / math.sqrt(queries.shape[-1])
        logits = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
        attention = self.dropout(torch.softmax(logits, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values)
        return output.contiguous(), attention


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dimension: int, attention_heads: int, dropout: float = 0.0):
        super().__init__()
        head_dimension = embedding_dimension // attention_heads
        if head_dimension < 1:
            raise ValueError("ALoRa embedding dimension must be at least the head count")
        projected_dimension = head_dimension * attention_heads
        self.n_heads = attention_heads
        self.norm = nn.LayerNorm(embedding_dimension)
        self.query_projection = nn.Linear(embedding_dimension, projected_dimension)
        self.key_projection = nn.Linear(embedding_dimension, projected_dimension)
        self.value_projection = nn.Linear(embedding_dimension, projected_dimension)
        self.out_projection = nn.Linear(projected_dimension, embedding_dimension)
        self.inner_attention = ALoRaAttention(dropout=dropout)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, time_count, _ = inputs.shape
        queries = self.query_projection(inputs).reshape(
            batch_size, time_count, self.n_heads, -1,
        )
        keys = self.key_projection(inputs).reshape(
            batch_size, time_count, self.n_heads, -1,
        )
        values = self.value_projection(inputs).reshape(
            batch_size, time_count, self.n_heads, -1,
        )
        output, attention = self.inner_attention(queries, keys, values)
        output = output.reshape(batch_size, time_count, -1)
        return self.out_projection(output), attention


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        window_size: int,
        attention_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = AttentionLayer(embedding_dimension, attention_heads, dropout)
        self.norm1 = nn.LayerNorm(window_size)
        self.norm2 = nn.LayerNorm(window_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attention(inputs)
        residual = inputs + self.dropout(attended)
        normalized = self.norm1(residual.permute(0, 2, 1)).permute(0, 2, 1)
        # The upstream layer applies its second temporal normalization to x + x.
        output = self.norm2((normalized + normalized).permute(0, 2, 1))
        return output.permute(0, 2, 1), attention


class Encoder(nn.Module):
    def __init__(self, layers, embedding_dimension: int):
        super().__init__()
        self.attn_layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embedding_dimension)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        attentions = []
        output = inputs
        for layer in self.attn_layers:
            output, attention = layer(output)
            attentions.append(attention)
        return self.norm(output), tuple(attentions)


class ALoRaT(nn.Module):
    def __init__(
        self,
        *,
        window_size: int,
        channel_count: int,
        output_channels: int,
        selected_pairs,
        pair_embedding_dimension: int = 512,
        attention_heads: int = 8,
        encoder_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = DataEmbedding(
            channel_count,
            selected_pairs,
            pair_embedding_dimension=pair_embedding_dimension,
            dropout=dropout,
        )
        embedding_dimension = self.embedding.d_model
        self.encoder = Encoder(
            [
                EncoderLayer(
                    embedding_dimension,
                    window_size,
                    attention_heads,
                    dropout,
                )
                for _ in range(encoder_layers)
            ],
            embedding_dimension,
        )
        self.projection = nn.Linear(embedding_dimension, output_channels)

    def forward(self, inputs: torch.Tensor):
        embedded = self.embedding(inputs)
        encoded, attentions = self.encoder(embedded)
        return self.projection(encoded), attentions, embedded


def calculate_low_rank_loss(attentions) -> torch.Tensor:
    """Return the upstream TNN-Geman penalty averaged over encoder layers."""
    penalties = []
    for attention in attentions:
        singular_values = torch.linalg.svdvals(attention.mean(dim=1))
        truncated = singular_values[:, 1:]
        penalties.append(torch.sum(truncated / (truncated + 1.0)))
    if not penalties:
        raise ValueError("ALoRa low-rank loss needs at least one attention layer")
    return torch.stack(penalties).mean()


def calculate_attention_rank(attention: torch.Tensor) -> torch.Tensor:
    """Count last-layer head-mean singular values above the fixed official h1."""
    if attention.ndim != 4:
        raise ValueError("ALoRa attention must have shape (batch, heads, time, time)")
    singular_values = torch.linalg.svdvals(attention.mean(dim=1))
    return (singular_values > ATTENTION_RANK_THRESHOLD).sum(dim=1).to(attention.dtype)
