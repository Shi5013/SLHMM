"""One route lattice and one scoring implementation for training and decoding."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from slhmm.training.structured import build_positive_automaton


class LatticeBuilder:
    def __init__(self, engine):
        self.engine = engine
        self.cache = OrderedDict()
        self.automata = OrderedDict()
        self.shared_root = engine.config.feature_root / "lattices" / engine.router.fingerprint
        namespace = os.environ.get("SLHMM_CACHE_NAMESPACE")
        if namespace and (Path(namespace).name != namespace or namespace in ('.','..')):
            raise ValueError('Invalid lattice cache namespace')
        self.root = self.shared_root / "workers" / namespace if namespace else self.shared_root
        self.root.mkdir(parents=True, exist_ok=True)

    def routes(self, trajectory_id, alignment, pools):
        e = self.engine; c = e.config
        # Hash actual candidates AND observation IDs: independent of model
        # parameters, but never shared across incompatible observation streams.
        identity = json.dumps([e.hashes["training_manifest"], trajectory_id, alignment["cell_ids"], pools,
                               c.use_route_cutoff, c.route_gamma_cut, c.route_abs_slack_m, c.d_min_m])
        key = hashlib.sha256(identity.encode()).hexdigest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return key, self.cache[key]
        path = self.root / key[:2] / (key + ".npz")
        existing = path if path.exists() else self.shared_root / key[:2] / (key + ".npz")
        if existing.exists():
            with np.load(existing, allow_pickle=False) as z:
                result = dict(features=[z[f"features_{i}"] for i in range(len(pools)-1)],
                              reachable=[z[f"reachable_{i}"] for i in range(len(pools)-1)],
                              fallback_unbounded_steps=int(z["fallback"]), drefs=z["drefs"])
        else:
            features, reachable, drefs = [], [], []
            previous_reachable = np.ones(len(pools[0]), dtype=bool)
            fallback = 0
            xy = e.resources.cell_positions_metric.numpy()
            for t, (left, right) in enumerate(zip(pools, pools[1:])):
                previous = np.repeat(left, len(right)); current = np.tile(right, len(left))
                dref = max(float(np.linalg.norm(xy[alignment["cell_ids"][t+1]]-xy[alignment["cell_ids"][t]])), c.d_min_m)
                limit = max(c.route_gamma_cut*dref, dref+c.route_abs_slack_m) if c.use_route_cutoff else math.inf
                f, r = e.router.route_pairs(previous, current, limit)
                r = r.reshape(len(left), len(right))
                # A nonempty K*K matrix can still have no surviving prefix.
                if previous_reachable.any() and not (r & previous_reachable[:, None]).any() and math.isfinite(limit):
                    f, r = e.router.route_pairs(previous, current, math.inf)
                    r = r.reshape(len(left), len(right)); fallback += 1
                previous_reachable = (r & previous_reachable[:, None]).any(0)
                features.append(f); reachable.append(r); drefs.append(dref)
            result = dict(features=features, reachable=reachable, fallback_unbounded_steps=fallback, drefs=np.asarray(drefs, dtype=np.float32))
            arrays = {f"features_{i}": f for i, f in enumerate(features)}
            arrays.update({f"reachable_{i}": r for i, r in enumerate(reachable)})
            arrays.update(fallback=np.asarray(fallback), drefs=result["drefs"])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(tmp, path)
        self.cache[key] = result
        while len(self.cache) > 24:
            self.cache.popitem(last=False)
        return key, result

    def positive_automaton(self, key, alignment, pools, routes):
        if key not in self.automata:
            self.automata[key] = build_positive_automaton(pools, routes["reachable"], alignment["observations"],
                                                        self.engine.delta_max, alignment.get("transition_loss_mask"))
            while len(self.automata) > 256:
                self.automata.popitem(last=False)
        self.automata.move_to_end(key)
        return self.automata[key]


def score_lattice(engine, items, pools):
    """Return per-trajectory unpadded emit/trans potentials and route metadata."""
    batch = engine.observation_data.compact_batch(items, pools)
    context, roads, state_map = engine.encode(batch, [batch["candidate_ids"][batch["candidate_mask"]]])
    emit, _ = engine.score_emit(batch, context, roads, state_map)
    output = []
    for b, ((tid, alignment), pool) in enumerate(zip(items, pools)):
        key, routes = engine.lattice_builder.routes(tid, alignment, pool)
        transitions, dnet_values = [], []
        for t, (left, right) in enumerate(zip(pool, pool[1:])):
            n, m = len(left), len(right)
            previous = torch.tensor(np.repeat(left, m), dtype=torch.long)
            current = torch.tensor(np.tile(right, n), dtype=torch.long)
            pc = torch.full_like(previous, alignment["cell_ids"][t])
            cc = torch.full_like(current, alignment["cell_ids"][t+1])
            previous_context = context["full"][b, t].expand(n*m, -1)
            current_context = context["full"][b, t+1].expand(n*m, -1)
            endpoint_offsets = None
            if engine.config.use_path_costs:
                endpoint_offsets = (
                    np.repeat(batch["candidate_offsets"][b, t, :n].numpy(), m),
                    np.tile(batch["candidate_offsets"][b, t+1, :m].numpy(), n))
            scores, aux = engine.score_trans(previous, current, pc, cc, previous_context, current_context,
                                             roads, state_map, route_features=routes["features"][t],
                                             reachable=routes["reachable"][t].reshape(-1), endpoint_offsets=endpoint_offsets)
            transitions.append(scores.reshape(n, m)*engine.config.viterbi_trans_weight)
            dnet_values.append(aux.get("dnet"))
        output.append(dict(emit=[emit[b,t,:len(s)] for t,s in enumerate(pool)], trans=transitions,
                           routes=routes, key=key, dnet=dnet_values))
    return output
