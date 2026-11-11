"""Batch extensions; original labels, split files, and observations stay immutable."""
from __future__ import annotations
from collections import OrderedDict
import random
import numpy as np
import torch
from scipy.spatial import cKDTree
from slhmm.training.common import load_pickle, read_json
from slhmm.training.data import build_emit_batch


class ObservationData:
    def __init__(self, resources, config):
        self.resources, self.config = resources, config
        self.offsets = load_pickle(config.feature_root / f"cell_offsets_k{config.candidate_k}.obj")
        self.lookups = {}
        self.training_cells = np.asarray(sorted({c for a in resources.alignment_train.values() for c in a["cell_ids"]}), dtype=np.int64)
        self.positions = resources.cell_positions_metric.numpy()
        self.tree = cKDTree(self.positions[self.training_cells])
        self.rings = {}

    def lookup(self, cell):
        cell = int(cell)
        if cell not in self.lookups:
            candidate = self.resources.cell_candidates[cell]
            self.lookups[cell] = {int(s): i for i, s in enumerate(candidate["state_ids"])}
        return self.lookups[cell]

    def attributes(self, cell, states):
        lookup = self.lookup(cell)
        indices = np.asarray([lookup[int(s)] for s in states])
        candidate = self.resources.cell_candidates[int(cell)]
        return (np.asarray(candidate["state_distances_m"], dtype=np.float32)[indices],
                np.asarray(candidate["state_physical_ranks"], dtype=np.int64)[indices],
                self.offsets[int(cell)][indices])

    def ring(self, cell):
        if cell not in self.rings:
            inner, outer = self.config.perturb_ring_m
            idx = np.asarray(self.tree.query_ball_point(self.positions[cell], outer), dtype=np.int64)
            cells = self.training_cells[idx]
            distances = np.linalg.norm(self.positions[cells] - self.positions[cell], axis=1)
            self.rings[cell] = cells[distances >= inner].tolist()
        return self.rings[cell]

    def augment(self, items):
        output, flags = [], []
        for tid, alignment in items:
            item = dict(alignment)
            item["cell_ids"] = list(alignment["cell_ids"])
            item["observations"] = [dict(o) for o in alignment["observations"]]
            perturbed = [False]*len(item["cell_ids"])
            for t in range(1, len(perturbed)-1):
                if random.random() >= self.config.p_perturb:
                    continue
                ring = self.ring(int(item["cell_ids"][t]))
                if not ring:
                    continue
                cell = random.choice(ring)
                item["cell_ids"][t] = cell
                observation = item["observations"][t]
                observation["cell_id"] = cell
                present = self.lookup(cell)
                positives = [s for s in observation["positive_state_ids"] if s in present]
                # Never infer extra labels from the route or promote masked labels.
                observation["positive_state_ids"] = positives
                observation["qualified"] = bool(positives) and observation.get("qualified", True)
                observation["allowed_occurrences"] = [o for o in observation["allowed_occurrences"] if o[1] in positives]
                perturbed[t] = True
            output.append((tid, item)); flags.append(perturbed)
        return output, flags

    def batch(self, items, augment=False):
        flags = None
        if augment:
            items, flags = self.augment(items)
        batch = build_emit_batch(items, self.resources, self.config)
        B, T, K = batch["candidate_ids"].shape
        rho = torch.zeros(B, T, K)
        weights = torch.zeros(B, T)
        repeat = torch.zeros(B, T)
        synthetic = torch.zeros(B, T, dtype=torch.bool)
        for b, (_, a) in enumerate(items):
            for t, observation in enumerate(a["observations"]):
                cell = int(a["cell_ids"][t])
                offsets = self.offsets[cell]
                rho[b, t, :len(offsets)] = torch.from_numpy(offsets)
                trusted = observation.get("qualified", bool(observation["positive_state_ids"]))
                weights[b, t] = observation.get("loss_weight", 1.) if trusted else 0.
                repeat[b, t] = a.get("input_repeat_weights", [1.]*len(a["cell_ids"]))[t]
                if not trusted:
                    batch["positive_mask"][b, t] = False
            if flags:
                synthetic[b, :len(flags[b])] = torch.tensor(flags[b])
        xy = self.resources.cell_positions_metric[batch["cell_ids"]]
        before = torch.zeros_like(xy); after = torch.zeros_like(xy)
        before[:, 1:] = xy[:, 1:]-xy[:, :-1]
        after[:, :-1] = xy[:, 1:]-xy[:, :-1]
        left = batch["observation_mask"].clone(); left[:, 0] = False
        right = torch.zeros_like(left); right[:, :-1] = batch["observation_mask"][:, 1:]
        before *= left.unsqueeze(-1); after *= right.unsqueeze(-1)
        dleft, dright = before.norm(dim=-1), after.norm(dim=-1)
        jump_ratio = (dleft+dright)/(before+after).norm(dim=-1).clamp_min(self.config.d_min_m)
        cosine = (before*after).sum(-1)/(dleft*dright).clamp_min(1e-6)
        repeated = torch.zeros_like(left); repeated[:, 1:] = batch["cell_ids"][:, 1:] == batch["cell_ids"][:, :-1]
        q = torch.stack([torch.log1p(jump_ratio), cosine, torch.log1p(dleft), torch.log1p(dright),
                         repeated.float(), (~left).float(), (~right).float()], -1)
        batch.update(candidate_offsets=rho, loss_weights=weights, repeat_weights=repeat,
                     q=q, perturbed=synthetic,
                     gate_supervised=(synthetic | ((jump_ratio < 1.5) & left & right)) & batch["observation_mask"],
                     items=items)
        return batch

    def compact_batch(self, items, pools):
        # Context needs only K_local roads. Build the unchanged context batch,
        # then replace scoring candidates with the recorded pure Top-K pools.
        batch = self.batch(items)
        B = len(items); T = batch["cell_ids"].shape[1]
        K = max(len(layer) for pool in pools for layer in pool)
        ids = torch.zeros(B, T, K, dtype=torch.long)
        distances = torch.zeros(B, T, K); ranks = torch.ones(B, T, K, dtype=torch.long)
        rho = torch.zeros(B, T, K); mask = torch.zeros(B, T, K, dtype=torch.bool)
        positive = torch.zeros_like(mask)
        for b, ((_, a), pool) in enumerate(zip(items, pools)):
            for t, states in enumerate(pool):
                n = len(states); d, r, offset = self.attributes(a["cell_ids"][t], states)
                ids[b,t,:n] = torch.tensor(states); mask[b,t,:n] = True
                distances[b,t,:n] = torch.from_numpy(d); ranks[b,t,:n] = torch.from_numpy(r); rho[b,t,:n] = torch.from_numpy(offset)
                allowed = set(a["observations"][t]["positive_state_ids"])
                positive[b,t,:n] = torch.tensor([s in allowed for s in states])
        batch.update(candidate_ids=ids, candidate_distances_m=distances, candidate_ranks=ranks,
                     candidate_offsets=rho, candidate_mask=mask, positive_mask=positive)
        return batch
