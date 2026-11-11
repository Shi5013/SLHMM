"""Exact monotonic anchor marginalization with a determinized occurrence lattice.

Each anchor prefix carries the SET of feasible reference occurrences. Merging
identical (last anchor, occurrence set) nodes sums disjoint anchor prefixes;
alternative occurrence assignments never multiply an anchor path's mass.
Wildcard steps preserve the last occurrence set. Backward viability excludes
dead ends before deciding whether to relax supervision at a step.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch


def safe_logsumexp(values, dim):
    finite = torch.isfinite(values).any(dim=dim, keepdim=True)
    safe = torch.where(finite, values, torch.zeros_like(values))
    reduced = torch.logsumexp(safe, dim=dim)
    return reduced.masked_fill(~finite.squeeze(dim), -torch.inf)


def segment_logsumexp(values, indices, count):
    maximum = values.new_full((count,), -torch.inf)
    maximum.scatter_reduce_(0, indices, values.detach(), reduce="amax", include_self=True)
    finite = torch.isfinite(maximum)
    safe_max = torch.where(finite, maximum, torch.zeros_like(maximum))
    exp_values = torch.exp(values - safe_max[indices])
    total = values.new_zeros(count).scatter_add(0, indices, exp_values)
    result = safe_max + total.clamp_min(torch.finfo(values.dtype).tiny).log()
    return result.masked_fill(~finite, -torch.inf)


@dataclass
class PositiveAutomaton:
    initial_candidates: np.ndarray
    layers: list[dict]
    wildcard_steps: int
    total_steps: int
    valid: bool
    max_nodes: int


def build_positive_automaton(pools, reachable, observations, delta_max, transition_mask=None):
    T = len(pools)
    if not T:
        return PositiveAutomaton(np.asarray([], dtype=int), [], 0, 0, False, 0)
    viable = [None] * T
    viable[-1] = np.ones(len(pools[-1]), dtype=bool)
    for t in range(T-2, -1, -1):
        viable[t] = (reachable[t] & viable[t+1][None, :]).any(1)
    if not viable[0].any():
        return PositiveAutomaton(np.asarray([], dtype=int), [], 0, T, False, 0)
    allowed = []
    for pool, observation in zip(pools, observations):
        occurrence = defaultdict(set)
        if observation.get("qualified", bool(observation["positive_state_ids"])):
            positives = set(observation["positive_state_ids"])
            for p, state, *_ in observation.get("allowed_occurrences", []):
                if state in positives:
                    occurrence[int(state)].add(int(p))
        allowed.append([tuple(sorted(occurrence[int(s)])) for s in pool])
    nodes = [(j, occ) for j, occ in enumerate(allowed[0]) if occ and viable[0][j]]
    wildcards = 0
    last_positive = 0
    if not nodes:
        nodes = [(j, ()) for j in range(len(pools[0])) if viable[0][j]]
        wildcards += 1
        last_positive = -1
    initial = np.asarray([n[0] for n in nodes], dtype=np.int64)
    layers = []
    maximum_nodes = len(nodes)
    for t in range(1, T):
        trusted_transition = transition_mask is None or bool(transition_mask[t-1])
        gap_limit = int(delta_max) * max(1, t-last_positive)

        def expand(strict):
            mapping = {}
            new_nodes, src, dst, prev_cand, cur_cand = [], [], [], [], []
            for i, (p, memory) in enumerate(nodes):
                next_candidates = np.flatnonzero(reachable[t-1][p] & viable[t])
                for j in next_candidates:
                    j = int(j)
                    if strict:
                        occ = allowed[t][j]
                        if memory:
                            occ = tuple(q for q in occ if any(0 <= q-r <= gap_limit for r in memory))
                        if not occ:
                            continue
                    else:
                        occ = memory
                    key = (j, occ)
                    if key not in mapping:
                        mapping[key] = len(new_nodes); new_nodes.append(key)
                    src.append(i); dst.append(mapping[key]); prev_cand.append(p); cur_cand.append(j)
            return new_nodes, dict(source=np.asarray(src, dtype=np.int64), destination=np.asarray(dst, dtype=np.int64),
                                   previous_candidate=np.asarray(prev_cand, dtype=np.int64), current_candidate=np.asarray(cur_cand, dtype=np.int64))

        new_nodes, arcs = expand(strict=True) if trusted_transition else ([], {})
        wildcard = not new_nodes
        if wildcard:
            new_nodes, arcs = expand(strict=False)
            wildcards += 1
        else:
            last_positive = t
        if not new_nodes:
            raise RuntimeError("Positive lattice lost a suffix-viable prefix")
        arcs["node_candidates"] = np.asarray([n[0] for n in new_nodes], dtype=np.int64)
        arcs["wildcard"] = wildcard
        maximum_nodes = max(maximum_nodes, len(new_nodes))
        if len(new_nodes) > 100_000:
            raise RuntimeError("Exact occurrence lattice exceeds 100000 nodes; no approximate truncation performed")
        layers.append(arcs)
        nodes = new_nodes
    return PositiveAutomaton(initial, layers, wildcards, T, True, maximum_nodes)


def lattice_crf_loss(emit, trans, automaton):
    """One unpadded trajectory. Inputs are the SAME potentials as Viterbi."""
    emit = [e.float() for e in emit]
    trans = [a.float() for a in trans]
    zero = emit[0][torch.isfinite(emit[0])].sum() * 0.0
    if not automaton.valid:
        return zero, {"invalid_lattice": 1., "wildcard_ratio": 1.}
    all_alpha = emit[0]
    for e, a in zip(emit[1:], trans):
        all_alpha = e + safe_logsumexp(all_alpha[:, None] + a, 0)
    log_z = safe_logsumexp(all_alpha, 0)
    alpha = emit[0][torch.as_tensor(automaton.initial_candidates, device=emit[0].device)]
    for t, layer in enumerate(automaton.layers, 1):
        arrays = {k: torch.as_tensor(layer[k], device=emit[0].device) for k in (
            "source", "destination", "previous_candidate", "current_candidate", "node_candidates")}
        values = alpha[arrays["source"]] + trans[t-1][arrays["previous_candidate"], arrays["current_candidate"]]
        alpha = emit[t][arrays["node_candidates"]] + segment_logsumexp(values, arrays["destination"], len(arrays["node_candidates"]))
    log_z_pos = safe_logsumexp(alpha, 0)
    if not torch.isfinite(log_z) or not torch.isfinite(log_z_pos):
        raise FloatingPointError("Nonfinite partition function in viable lattice")
    difference = log_z - log_z_pos
    if difference.detach().item() < -1e-3:
        raise AssertionError(f"Positive partition exceeds full partition: {difference.item()}")
    return difference / len(emit), {"invalid_lattice": 0., "wildcard_ratio": automaton.wildcard_steps / automaton.total_steps,
                                    "log_z": float(log_z.detach()), "log_z_positive": float(log_z_pos.detach()),
                                    "max_positive_nodes": float(automaton.max_nodes)}
