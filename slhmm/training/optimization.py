"""Weighted reductions with a fixed epoch scale for stochastic training."""
from __future__ import annotations

import numpy as np


def weighted_reduction(values, weights, normalizer=None):
    """A fixed denominator preserves weights across different mini-batches.

    With N batches and total source weight W, normalizer=W/N makes the
    average batch objective equal to the weighted dataset objective. Splitting
    a logical batch into micro-batches must keep this same denominator.
    Evaluation instead uses the sum of the supplied weights.
    """
    denominator = weights.sum().clamp_min(1e-12) if normalizer is None else max(float(normalizer), 1e-12)
    return (values * weights).sum() / denominator


def observation_weight_totals(alignments):
    emit = gate = sequence = 0.
    for a in alignments:
        repeat = a.get("input_repeat_weights", [1.] * len(a["cell_ids"]))
        gate += sum(repeat)
        sequence += float(np.mean(repeat))
        for o in a["observations"]:
            if o["positive_state_ids"] and o.get("qualified", True):
                emit += max(0., o.get("loss_weight", 1.))
    return dict(emit=emit, gate=gate, sequence=sequence)


def transition_weight_total(groups, alignments):
    total = 0.
    for g in groups:
        a = alignments[g["trajectory_id"]]
        mask = a.get("transition_loss_mask", [True] * (len(a["cell_ids"]) - 1))
        for r in g["transitions"]:
            if mask[r["transition_index"]] and r["positive_state_pairs"]:
                total += max(0., r.get("loss_weight", 1.))
    return total
