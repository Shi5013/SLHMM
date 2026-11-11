"""Numerically stable weak-label losses and ranking metrics."""

from __future__ import annotations

import torch


def multi_positive_marginal_loss(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    positive_log_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log probability assigned to a non-empty soft-positive set."""

    if scores.shape != candidate_mask.shape or scores.shape != positive_mask.shape:
        raise ValueError("scores, candidate_mask and positive_mask must share shape")
    valid_rows = positive_mask.any(dim=-1) & candidate_mask.any(dim=-1)
    if not valid_rows.any():
        return scores.sum() * 0.0
    all_scores = scores.masked_fill(~candidate_mask, float("-inf"))
    positive_scores = scores
    if positive_log_weights is not None:
        positive_scores = positive_scores + positive_log_weights
    positive_scores = positive_scores.masked_fill(~positive_mask, float("-inf"))
    losses = torch.logsumexp(all_scores, dim=-1) - torch.logsumexp(
        positive_scores, dim=-1
    )
    return losses[valid_rows].mean()


def soft_positive_recall_at_k(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    k = min(k, scores.shape[-1])
    ranked = scores.masked_fill(~candidate_mask, float("-inf")).topk(k, dim=-1).indices
    hits = torch.gather(positive_mask, -1, ranked).any(dim=-1)
    valid = positive_mask.any(dim=-1)
    if not valid.any():
        return torch.zeros((), device=scores.device)
    return hits[valid].float().mean()


def soft_positive_mrr(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    ranked = scores.masked_fill(~candidate_mask, float("-inf")).argsort(
        dim=-1, descending=True
    )
    ranked_positive = torch.gather(positive_mask, -1, ranked)
    positions = torch.arange(
        1, scores.shape[-1] + 1, device=scores.device, dtype=torch.float32
    )
    reciprocal = (ranked_positive.float() / positions).max(dim=-1).values
    valid = positive_mask.any(dim=-1)
    if not valid.any():
        return torch.zeros((), device=scores.device)
    return reciprocal[valid].mean()


def ragged_multi_positive_marginal_loss(
    scores: torch.Tensor,
    offsets: torch.Tensor,
    positive_mask: torch.Tensor,
    positive_log_weights: torch.Tensor,
) -> torch.Tensor:
    """Multi-positive marginal loss for flattened variable-length pair groups."""

    losses = []
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        group_scores = scores[start:end]
        group_positive = positive_mask[start:end]
        if not group_positive.any():
            continue
        losses.append(
            torch.logsumexp(group_scores, dim=0)
            - torch.logsumexp(
                group_scores[group_positive]
                + positive_log_weights[start:end][group_positive],
                dim=0,
            )
        )
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def ragged_positive_recall_at_k(
    scores: torch.Tensor,
    offsets: torch.Tensor,
    positive_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    hits = []
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        group_scores = scores[start:end]
        group_positive = positive_mask[start:end]
        if not group_positive.any():
            continue
        count = min(k, end - start)
        top = group_scores.topk(count).indices
        hits.append(group_positive[top].any().float())
    if not hits:
        return torch.zeros((), device=scores.device)
    return torch.stack(hits).mean()
