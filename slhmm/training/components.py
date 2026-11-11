"""Observation-independent context, gated emissions, and physical penalties."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from slhmm.training.models import CellEncoder


class IndependentDirectionalEncoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        H = config.trajectory_hidden_dim
        self.forward_gru = nn.GRU(input_dim, H, config.trajectory_layers, batch_first=True,
                                 dropout=0.1 if config.trajectory_layers > 1 else 0.)
        self.backward_gru = nn.GRU(input_dim, H, config.trajectory_layers, batch_first=True,
                                  dropout=0.1 if config.trajectory_layers > 1 else 0.)
        self.initial_forward = nn.Parameter(torch.zeros(H))
        self.initial_backward = nn.Parameter(torch.zeros(H))

    @staticmethod
    def reverse_valid(x, lengths):
        steps = torch.arange(x.shape[1], device=x.device)[None, :]
        indices = (lengths.to(x.device)[:, None] - 1 - steps).clamp_min(0)
        return x.gather(1, indices[:, :, None].expand(-1, -1, x.shape[-1]))

    @staticmethod
    def encode(gru, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = gru(packed)
        return nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])[0]

    def forward(self, x, lengths):
        forward = self.encode(self.forward_gru, x, lengths)
        backward = self.reverse_valid(self.encode(self.backward_gru, self.reverse_valid(x, lengths), lengths), lengths)
        B, T, H = forward.shape
        left = torch.cat([self.initial_forward.expand(B, 1, H), forward[:, :-1]], 1)
        right = torch.cat([backward[:, 1:], self.initial_backward.expand(B, 1, H)], 1)
        end = torch.arange(T, device=x.device)[None, :] == lengths.to(x.device)[:, None] - 1
        right = torch.where(end[:, :, None], self.initial_backward.expand(B, T, H), right)
        return torch.cat([forward, backward], -1), torch.cat([left, right], -1)


class ReliabilityGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.deviation_projection = nn.Linear(dim, dim, bias=False)
        self.network = nn.Sequential(nn.Linear(2*dim+8, dim), nn.ReLU(), nn.Linear(dim, 1))
        nn.init.constant_(self.network[-1].bias, 2.)

    def forward(self, z, full, loo, q):
        deviation = 1-F.cosine_similarity(self.deviation_projection(z), loo, dim=-1)
        return self.network(torch.cat([z, full, q, deviation.unsqueeze(-1)], -1)).squeeze(-1)


class DecoupledEmitHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.context = nn.Linear(dim, dim)
        self.road = nn.Linear(dim, dim, bias=False)
        self.context_score = nn.Linear(dim, 1, bias=False)
        self.distance = nn.Sequential(nn.Linear(10, dim), nn.ReLU())
        self.observation = nn.Sequential(nn.Linear(4*dim, 2*dim), nn.ReLU(), nn.Dropout(.1), nn.Linear(2*dim, 1))

    def forward(self, full, roads, distances, ranks, offsets, loo, gate):
        context_score = self.context_score(torch.tanh(self.context(loo).unsqueeze(-2) + self.road(roads))).squeeze(-1)
        distance = self.distance(torch.cat([CellEncoder.distance_rbf(distances), torch.log1p(ranks.float()).unsqueeze(-1), offsets.unsqueeze(-1)], -1))
        h = full.unsqueeze(-2).expand_as(roads)
        observation = self.observation(torch.cat([h, roads, h*roads, distance], -1)).squeeze(-1)
        return context_score + gate.unsqueeze(-1)*observation


class PhysicalCosts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cost_logits = nn.Parameter(torch.full((4,), -1.))
        self.rank_logits = nn.Parameter(torch.full((config.rank_levels, config.rank_levels), -2.))
        self.register_buffer("off_diagonal", 1-torch.eye(config.rank_levels))

    def forward(self, features, rho_previous, rho_current, length_previous, length_current, dref, same):
        c = self.config
        features = features.float()
        delta = rho_current.float() - rho_previous.float()
        same_distance = (delta.clamp_min(0)+c.mu_back*(-delta).clamp_min(0))*length_previous
        other_distance = (1-rho_previous)*length_previous + features[:, 1] + rho_current*length_current
        dnet = torch.where(same, same_distance, other_distance).float().clamp_min(0)
        matrix = F.softplus(self.rank_logits.float())*self.off_diagonal
        costs = torch.stack([
            F.relu(dnet-c.detour_gamma*dref)/dref + c.lambda_len*dnet/1000,
            features[:, 2]/torch.pi,
            features[:, 3]+c.mu_uturn*features[:, 4],
            features[:, 7:] @ matrix.flatten() + c.beta_tunnel*features[:, 5]], -1).clamp(max=c.cost_clamp)
        return (F.softplus(self.cost_logits.float())*costs).sum(-1), costs, dnet
