"""Neural encoders and scoring heads for SLHMM."""

from __future__ import annotations

import math
from typing import Iterable

import dgl.nn as dglnn
import torch
from torch import nn
import torch.nn.functional as F

from slhmm.configs.training import TrainingConfig


class RelationalRoadEncoder(nn.Module):
    """Encode OSM directed edges on a single-node-type relational line graph."""

    def __init__(
        self,
        numeric_dim: int,
        highway_count: int,
        relation_names: Iterable[str],
        config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.relation_names = tuple(relation_names)
        self.hidden_dim = config.road_hidden_dim
        self.layer_count = config.road_layers
        self.highway_embedding = nn.Embedding(
            highway_count, config.highway_embedding_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(numeric_dim + config.highway_embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.layer_count):
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        relation: dglnn.GraphConv(
                            self.hidden_dim,
                            self.hidden_dim,
                            norm="both",
                            allow_zero_in_degree=True,
                        )
                        for relation in self.relation_names
                    },
                    aggregate="sum",
                )
            )
            self.norms.append(nn.LayerNorm(self.hidden_dim))
        self.dropout = nn.Dropout(config.road_dropout)

    def encode_input(
        self, numeric_features: torch.Tensor, highway_ids: torch.Tensor
    ) -> torch.Tensor:
        highway = self.highway_embedding(highway_ids)
        return self.input_projection(torch.cat((numeric_features, highway), dim=-1))

    def forward(
        self,
        graph_or_blocks,
        numeric_features: torch.Tensor,
        highway_ids: torch.Tensor,
    ) -> torch.Tensor:
        blocks = (
            list(graph_or_blocks)
            if isinstance(graph_or_blocks, (list, tuple))
            else [graph_or_blocks] * self.layer_count
        )
        if len(blocks) != self.layer_count:
            raise ValueError(
                f"Expected {self.layer_count} graph blocks, got {len(blocks)}"
            )
        hidden = self.encode_input(numeric_features, highway_ids)
        for block, convolution, norm in zip(blocks, self.layers, self.norms):
            output = convolution(block, {"road": hidden})["road"]
            destination_count = int(block.num_dst_nodes("road"))
            residual = hidden[:destination_count]
            hidden = norm(residual + self.dropout(F.relu(output)))
        return hidden


class RoadPretrainingHeads(nn.Module):
    def __init__(self, hidden_dim: int, numeric_dim: int, highway_count: int):
        super().__init__()
        self.numeric_decoder = nn.Linear(hidden_dim, numeric_dim)
        self.highway_decoder = nn.Linear(hidden_dim, highway_count)
        self.link_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def reconstruct(self, embeddings: torch.Tensor):
        return self.numeric_decoder(embeddings), self.highway_decoder(embeddings)

    def link_score(
        self, source_embeddings: torch.Tensor, target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.link_bilinear(source_embeddings, target_embeddings).squeeze(-1)


class CoordinateEncoder(nn.Module):
    def __init__(
        self,
        frequencies: int,
        output_dim: int,
        coordinate_mean: tuple[float, float],
        coordinate_std: tuple[float, float],
    ) -> None:
        super().__init__()
        self.frequencies = frequencies
        self.register_buffer(
            "coordinate_mean", torch.tensor(coordinate_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "coordinate_std",
            torch.tensor(coordinate_std, dtype=torch.float32).clamp_min(1e-6),
        )
        input_dim = 2 + 4 * frequencies
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        normalized = (positions - self.coordinate_mean) / self.coordinate_std
        values = [normalized]
        for index in range(self.frequencies):
            scaled = normalized * (2.0**index) * math.pi
            values.extend((torch.sin(scaled), torch.cos(scaled)))
        return self.network(torch.cat(values, dim=-1))


class CellEncoder(nn.Module):
    """Generate cell representations without a separate cellular graph."""

    def __init__(
        self,
        cell_count: int,
        coordinate_mean: tuple[float, float],
        coordinate_std: tuple[float, float],
        config: TrainingConfig,
    ) -> None:
        super().__init__()
        hidden = config.road_hidden_dim
        self.coordinate_encoder = CoordinateEncoder(
            config.coordinate_frequencies,
            hidden,
            coordinate_mean,
            coordinate_std,
        )
        self.cell_id_embedding = nn.Embedding(
            cell_count, config.cell_id_embedding_dim
        )
        nn.init.zeros_(self.cell_id_embedding.weight)
        self.id_projection = nn.Linear(config.cell_id_embedding_dim, hidden)
        self.query_projection = nn.Linear(hidden, hidden, bias=False)
        self.key_projection = nn.Linear(hidden, hidden, bias=False)
        self.distance_network = nn.Sequential(
            nn.Linear(8, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

    @staticmethod
    def distance_rbf(distances_m: torch.Tensor) -> torch.Tensor:
        centers = torch.tensor(
            [0.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0],
            dtype=distances_m.dtype,
            device=distances_m.device,
        )
        widths = torch.tensor(
            [50.0, 75.0, 125.0, 250.0, 400.0, 750.0, 1250.0, 2000.0],
            dtype=distances_m.dtype,
            device=distances_m.device,
        )
        return torch.exp(
            -((distances_m.unsqueeze(-1) - centers) / widths).square()
        )

    def forward(
        self,
        cell_ids: torch.Tensor,
        positions: torch.Tensor,
        local_road_embeddings: torch.Tensor,
        local_road_distances_m: torch.Tensor,
        local_road_mask: torch.Tensor,
    ) -> torch.Tensor:
        coordinate = self.coordinate_encoder(positions)
        if hasattr(self, "h3_prior"):
            coordinate = self.h3_prior(coordinate, cell_ids)
        identifier = self.id_projection(self.cell_id_embedding(cell_ids))
        query = self.query_projection(coordinate).unsqueeze(-2)
        keys = self.key_projection(local_road_embeddings)
        attention = (query * keys).sum(dim=-1) / math.sqrt(keys.shape[-1])
        attention = attention + self.distance_network(
            self.distance_rbf(local_road_distances_m)
        ).squeeze(-1)
        attention = attention.masked_fill(~local_road_mask, float("-inf"))
        empty_rows = ~local_road_mask.any(dim=-1)
        if empty_rows.any():
            attention = torch.where(
                empty_rows.unsqueeze(-1),
                torch.zeros_like(attention),
                attention,
            )
        weights = torch.softmax(attention, dim=-1)
        weights = torch.where(local_road_mask, weights, torch.zeros_like(weights))
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = weights / denominator
        context = torch.sum(weights.unsqueeze(-1) * local_road_embeddings, dim=-2)
        return self.output(torch.cat((coordinate, identifier, context), dim=-1))


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int, config: TrainingConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.trajectory_hidden_dim,
            num_layers=config.trajectory_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if config.trajectory_layers > 1 else 0.0,
        )
        self.output_dim = 2 * config.trajectory_hidden_dim

    def forward(
        self, cell_embeddings: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            cell_embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.gru(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=cell_embeddings.shape[1],
        )
        return padded


class EmitHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.distance_projection = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
        )
        self.network = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(
        self,
        trajectory_context: torch.Tensor,
        road_embeddings: torch.Tensor,
        distances_m: torch.Tensor,
        physical_ranks: torch.Tensor,
    ) -> torch.Tensor:
        distance = torch.cat(
            (
                CellEncoder.distance_rbf(distances_m),
                torch.log1p(physical_ranks.float()).unsqueeze(-1),
            ),
            dim=-1,
        )
        context = trajectory_context.unsqueeze(-2).expand_as(road_embeddings)
        features = torch.cat(
            (
                context,
                road_embeddings,
                context * road_embeddings,
                torch.abs(context - road_embeddings),
                self.distance_projection(distance),
            ),
            dim=-1,
        )
        return self.network(features).squeeze(-1)


class TransHead(nn.Module):
    """Score macro-transitions between roads at adjacent cellular observations."""

    def __init__(self, hidden_dim: int, pair_feature_dim: int) -> None:
        super().__init__()
        self.pair_projection = nn.Sequential(
            nn.Linear(pair_feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.network = nn.Sequential(
            nn.Linear(7 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(
        self,
        previous_road: torch.Tensor,
        current_road: torch.Tensor,
        previous_context: torch.Tensor,
        current_context: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            torch.cat(
                (
                    previous_road,
                    current_road,
                    previous_road * current_road,
                    torch.abs(previous_road - current_road),
                    previous_context,
                    current_context,
                    self.pair_projection(pair_features),
                ),
                dim=-1,
            )
        ).squeeze(-1)


class SLHMMModel(nn.Module):
    def __init__(
        self,
        *,
        numeric_dim: int,
        highway_count: int,
        relation_names: Iterable[str],
        cell_count: int,
        coordinate_mean: tuple[float, float],
        coordinate_std: tuple[float, float],
        pair_feature_dim: int,
        config: TrainingConfig,
    ) -> None:
        super().__init__()
        self.road_encoder = RelationalRoadEncoder(
            numeric_dim, highway_count, relation_names, config
        )
        self.road_pretraining_heads = RoadPretrainingHeads(
            config.road_hidden_dim, numeric_dim, highway_count
        )
        self.cell_encoder = CellEncoder(
            cell_count, coordinate_mean, coordinate_std, config
        )
        self.trajectory_encoder = TrajectoryEncoder(
            config.road_hidden_dim, config
        )
        self.emit_head = EmitHead(config.road_hidden_dim)
        self.trans_head = TransHead(config.road_hidden_dim, pair_feature_dim)
