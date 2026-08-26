"""GDN architecture adapted from commit 9853899da860682669a134e4af315d036aab4eca.

The model and parameter names follow the MIT upstream implementation. Device,
session splitting, scaling, validation, and score persistence remain project
responsibilities.
"""

import math

import torch
from torch import nn
from torch.nn import functional
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


SOURCE_COMMIT = "9853899da860682669a134e4af315d036aab4eca"


def build_fully_connected_edge_index(channel_count: int, device: str) -> torch.Tensor:
    if channel_count < 1:
        raise ValueError("GDN channel_count must be positive")
    indices = torch.arange(channel_count, device=device)
    source = indices.repeat_interleave(channel_count)
    target = indices.repeat(channel_count)
    return torch.stack((source, target))


def get_batch_edge_index(
    original_edge_index: torch.Tensor,
    batch_size: int,
    node_count: int,
) -> torch.Tensor:
    edge_count = original_edge_index.shape[1]
    batched = original_edge_index.repeat(1, batch_size).contiguous()
    for batch_index in range(batch_size):
        start = batch_index * edge_count
        batched[:, start : start + edge_count] += batch_index * node_count
    return batched.long()


class OutLayer(nn.Module):
    def __init__(self, input_dimension: int, layer_count: int, intermediate_dimension: int):
        super().__init__()
        modules = []
        for layer_index in range(layer_count):
            if layer_index == layer_count - 1:
                modules.append(nn.Linear(
                    input_dimension if layer_count == 1 else intermediate_dimension,
                    1,
                ))
            else:
                layer_input = input_dimension if layer_index == 0 else intermediate_dimension
                modules.extend((
                    nn.Linear(layer_input, intermediate_dimension),
                    nn.BatchNorm1d(intermediate_dimension),
                    nn.ReLU(),
                ))
        self.mlp = nn.ModuleList(modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for module in self.mlp:
            if isinstance(module, nn.BatchNorm1d):
                output = module(output.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                output = module(output)
        return output


class GraphLayer(MessagePassing):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        heads: int = 1,
        concatenate: bool = False,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.heads = heads
        self.concatenate = concatenate
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin = Linear(input_channels, heads * output_channels, bias=False)
        self.att_i = Parameter(torch.empty(1, heads, output_channels))
        self.att_j = Parameter(torch.empty(1, heads, output_channels))
        self.att_em_i = Parameter(torch.empty(1, heads, output_channels))
        self.att_em_j = Parameter(torch.empty(1, heads, output_channels))
        if bias:
            bias_size = heads * output_channels if concatenate else output_channels
            self.bias = Parameter(torch.empty(bias_size))
        else:
            self.register_parameter("bias", None)
        self._attention = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        if self.bias is not None:
            zeros(self.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        embedding: torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ):
        transformed = self.lin(inputs)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=transformed.size(self.node_dim))
        self._attention = None
        output = self.propagate(
            edge_index,
            x=(transformed, transformed),
            embedding=embedding,
            source_indices=edge_index[0],
        )
        if self.concatenate:
            output = output.reshape(-1, self.heads * self.output_channels)
        else:
            output = output.mean(dim=1)
        if self.bias is not None:
            output = output + self.bias
        if return_attention_weights:
            attention = self._attention
            self._attention = None
            return output, (edge_index, attention)
        return output

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        index: torch.Tensor,
        size_i: int,
        embedding: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        target_values = x_i.reshape(-1, self.heads, self.output_channels)
        source_values = x_j.reshape(-1, self.heads, self.output_channels)
        target_embedding = embedding[index].unsqueeze(1).repeat(1, self.heads, 1)
        source_embedding = embedding[source_indices].unsqueeze(1).repeat(1, self.heads, 1)
        target_keys = torch.cat((target_values, target_embedding), dim=-1)
        source_keys = torch.cat((source_values, source_embedding), dim=-1)
        target_attention = torch.cat((self.att_i, self.att_em_i), dim=-1)
        source_attention = torch.cat((self.att_j, self.att_em_j), dim=-1)
        attention = (
            (target_keys * target_attention).sum(dim=-1)
            + (source_keys * source_attention).sum(dim=-1)
        ).reshape(-1, self.heads, 1)
        attention = functional.leaky_relu(attention, self.negative_slope)
        attention = softmax(attention, index, num_nodes=size_i)
        self._attention = attention
        attention = functional.dropout(attention, p=self.dropout, training=self.training)
        return source_values * attention


class GNNLayer(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.gnn = GraphLayer(input_channels, output_channels, heads=1, concatenate=False)
        self.bn = nn.BatchNorm1d(output_channels)

    def forward(
        self,
        inputs: torch.Tensor,
        edge_index: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        output, (new_edge_index, attention) = self.gnn(
            inputs,
            edge_index,
            embedding,
            return_attention_weights=True,
        )
        self.attention_weights = attention
        self.attention_edge_index = new_edge_index
        return functional.relu(self.bn(output))


class GDN(nn.Module):
    def __init__(
        self,
        edge_index_sets,
        node_num: int,
        dim: int = 64,
        out_layer_inter_dim: int = 256,
        input_dim: int = 10,
        out_layer_num: int = 1,
        topk: int = 20,
    ):
        super().__init__()
        if not 0 < topk <= node_num:
            raise ValueError("GDN topk is outside the node range")
        self.edge_index_sets = tuple(edge_index_sets)
        self.embedding = nn.Embedding(node_num, dim)
        self.bn_outlayer_in = nn.BatchNorm1d(dim)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim) for _ in self.edge_index_sets
        ])
        self.topk = topk
        self.learned_graph = None
        self.out_layer = OutLayer(
            dim * len(self.edge_index_sets),
            out_layer_num,
            out_layer_inter_dim,
        )
        self.cache_edge_index_sets = [None] * len(self.edge_index_sets)
        self.dp = nn.Dropout(0.2)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(
        self,
        data: torch.Tensor,
        original_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del original_edge_index
        features = data.clone().detach()
        batch_size, node_count, window_size = features.shape
        features = features.reshape(-1, window_size).contiguous()
        graph_outputs = []

        for graph_index, edge_index in enumerate(self.edge_index_sets):
            edge_count = edge_index.shape[1]
            cached = self.cache_edge_index_sets[graph_index]
            if cached is None or cached.shape[1] != edge_count * batch_size:
                self.cache_edge_index_sets[graph_index] = get_batch_edge_index(
                    edge_index,
                    batch_size,
                    node_count,
                ).to(data.device)

            node_embedding = self.embedding(torch.arange(node_count, device=data.device))
            repeated_embedding = node_embedding.repeat(batch_size, 1)
            detached_embedding = node_embedding.detach().clone()
            cosine = detached_embedding @ detached_embedding.T
            norms = detached_embedding.norm(dim=-1)
            cosine = cosine / (norms[:, None] @ norms[None, :])
            top_indices = torch.topk(cosine, self.topk, dim=-1).indices
            self.learned_graph = top_indices
            targets = torch.arange(node_count, device=data.device).repeat_interleave(self.topk)
            sources = top_indices.reshape(-1)
            learned_edge_index = torch.stack((sources, targets))
            batch_edges = get_batch_edge_index(
                learned_edge_index,
                batch_size,
                node_count,
            ).to(data.device)
            graph_outputs.append(self.gnn_layers[graph_index](
                features,
                batch_edges,
                repeated_embedding,
            ))

        output = torch.cat(graph_outputs, dim=1).reshape(batch_size, node_count, -1)
        node_embedding = self.embedding(torch.arange(node_count, device=data.device))
        output = output * node_embedding
        output = functional.relu(self.bn_outlayer_in(output.permute(0, 2, 1)))
        output = self.dp(output.permute(0, 2, 1))
        return self.out_layer(output).reshape(-1, node_count)
