"""Deterministic directed line-graph Dijkstra and versioned pair features.

Feature layout: impedance, between_len, sum_theta, n_sharp, n_uturn,
n_tunnel_switch, n_hops, then R*R rank-pair counts. All features use fp32;
counts are not silently truncated to uint8 and lengths cannot overflow fp16.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class GeneralizedRouter:
    def __init__(self, road_features, config, graph_hash=""):
        self.config = config
        self.lengths = road_features["length_m"].numpy().astype(np.float64)
        self.tin = road_features["tangent_in"].numpy().astype(np.float64)
        self.tout = road_features["tangent_out"].numpy().astype(np.float64)
        self.rank = road_features["rank"].numpy().astype(np.int64) - 1
        self.tunnel = road_features["tunnel_raw"].numpy()
        self.R = config.rank_levels
        self.N = len(self.lengths)
        self.width = 7 + self.R * self.R
        weights = (config.eta_turn, config.eta_uturn, config.eta_tunnel) if config.use_path_costs else (0., 0., 0.)
        self.weights = weights
        identity = dict(schema=3, graph=graph_hash, weights=weights, ranks=self.R,
                        geometry_features=hashlib.sha256(b"".join(
                            road_features[k].numpy().tobytes() for k in
                            ("length_m","tangent_in","tangent_out","rank","tunnel_raw","edge_u","edge_v")
                        )).hexdigest())
        self.identity = identity
        self.fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        self.shared_cache_root = config.feature_root / "paths" / self.fingerprint
        namespace = os.environ.get("SLHMM_CACHE_NAMESPACE")
        if namespace and (Path(namespace).name != namespace or namespace in ('.','..')):
            raise ValueError('Invalid route cache namespace')
        self.cache_root = self.shared_cache_root / "workers" / namespace if namespace else self.shared_cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        read_namespace = os.environ.get("SLHMM_ROUTE_CACHE_READ_NAMESPACE")
        if read_namespace and (Path(read_namespace).name != read_namespace or read_namespace in ('.', '..')):
            raise ValueError('Invalid read-only route cache namespace')
        self.read_cache_root = self.shared_cache_root / "workers" / read_namespace if read_namespace else None
        by_start = defaultdict(list)
        for s, u in enumerate(road_features["edge_u"].tolist()):
            by_start[u].append(s)
        sources, targets = [], []
        for s, v in enumerate(road_features["edge_v"].tolist()):
            for t in by_start[v]:
                if s != t:
                    sources.append(s); targets.append(t)
        sources, targets = np.asarray(sources), np.asarray(targets)
        theta = np.arccos(np.clip(np.sum(self.tout[sources] * self.tin[targets], axis=1), -1, 1))
        costs = self.lengths[targets] + weights[0] * theta + weights[1] * (theta > 5 * np.pi / 6) + weights[2] * (self.tunnel[sources] != self.tunnel[targets])
        self.graph = csr_matrix((costs, (sources, targets)), shape=(self.N, self.N))
        self.graph.sort_indices()
        self.cache = OrderedDict()
        self.disk_sources = OrderedDict()
        # Cache the decoded immutable arrays, not repeatedly decompressed NPZs.
        # This is a runtime memory budget, independent of model/checkpoint config.
        cache_mib = int(os.environ.get("SLHMM_ROUTE_CACHE_MIB", "4096"))
        if cache_mib < 0:
            raise ValueError("SLHMM_ROUTE_CACHE_MIB must be nonnegative")
        self.disk_cache_budget = cache_mib * 1024 * 1024
        self.disk_cache_bytes = 0
        self.stats = defaultdict(int)

    @staticmethod
    def _source_bytes(value):
        # Include a conservative allowance for array/container/key overhead.
        return 512 if value is None else value[0].nbytes + value[1].nbytes + 512

    def _drop_disk_source(self, source):
        if source in self.disk_sources:
            self.disk_cache_bytes -= self._source_bytes(self.disk_sources.pop(source))

    def clear_disk_cache(self):
        self.disk_sources.clear()
        self.disk_cache_bytes = 0

    def _cache_disk_source(self, source, value):
        self._drop_disk_source(source)
        size = self._source_bytes(value)
        if size > self.disk_cache_budget:
            return
        while self.disk_sources and (self.disk_cache_bytes + size > self.disk_cache_budget
                                     or len(self.disk_sources) >= 65536):
            _, evicted = self.disk_sources.popitem(last=False)
            self.disk_cache_bytes -= self._source_bytes(evicted)
            self.stats["disk_cache_evictions"] += 1
        self.disk_sources[source] = value
        self.disk_cache_bytes += size

    def _source_file(self, source):
        return self.cache_root / f"{source // 1000:03d}" / f"{source:07d}.npz"

    def _read_source_file(self, source):
        p = self._source_file(source)
        if p.exists():
            return p
        if self.read_cache_root is not None:
            p = self.read_cache_root / f"{source // 1000:03d}" / f"{source:07d}.npz"
            if p.exists():
                return p
        return self.shared_cache_root / f"{source // 1000:03d}" / f"{source:07d}.npz"

    def _disk(self, source):
        if source in self.disk_sources:
            self.disk_sources.move_to_end(source)
            self.stats["disk_cache_hits"] += 1
            return self.disk_sources[source]
        p = self._read_source_file(source)
        value = None
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                value = (z["targets"], z["features"], float(z["search_limit"]))
            self.stats["disk_source_reads"] += 1
        self._cache_disk_source(source, value)
        return value

    def _lookup(self, s, t, limit):
        key = (s, t)
        if key in self.cache:
            self.cache.move_to_end(key)
            feature, searched = self.cache[key]
            if feature is not None or searched >= limit:
                return True, feature if feature is not None and feature[0] <= limit + 1e-5 else None
        source = self._disk(s)
        if source is not None:
            targets, features, searched = source
            i = np.searchsorted(targets, t)
            if i < len(targets) and targets[i] == t:
                feature = features[i]
                if np.isfinite(feature[0]):
                    return True, feature if feature[0] <= limit + 1e-5 else None
                if searched >= limit:
                    return True, None
        return False, None

    def _remember(self, s, t, feature, limit):
        self.cache[(s, t)] = (feature, limit)
        self.cache.move_to_end((s, t))
        while len(self.cache) > self.config.route_cache_pairs:
            self.cache.popitem(last=False)

    def _backtrace(self, source, target, predecessors):
        path = [int(target)]
        while path[-1] != source:
            previous = int(predecessors[path[-1]])
            if previous < 0:
                return None
            path.append(previous)
            if len(path) > self.N:
                raise RuntimeError("Cycle in Dijkstra predecessor chain")
        return path[::-1]

    def features_for_path(self, path, impedance=None):
        result = np.zeros(self.width, dtype=np.float32)
        if len(path) == 1:
            return result
        a, b = np.asarray(path[:-1]), np.asarray(path[1:])
        angles = np.arccos(np.clip((self.tout[a] * self.tin[b]).sum(1), -1, 1))
        tunnel = (self.tunnel[a] != self.tunnel[b]).sum()
        turns = (angles > 5 * np.pi / 6).sum()
        route_cost = self.lengths[b].sum() + self.weights[0] * angles.sum() + self.weights[1] * turns + self.weights[2] * tunnel
        result[:7] = [route_cost if impedance is None else impedance,
                      self.lengths[np.asarray(path[1:-1], dtype=int)].sum(),
                      angles.sum(), (angles > np.pi / 2).sum(), turns, tunnel, len(b)]
        result[7:] = np.bincount(self.rank[a] * self.R + self.rank[b], minlength=self.R*self.R)
        return result

    def compute_source(self, source, targets, limit):
        distances, predecessors = dijkstra(self.graph, directed=True, indices=int(source), limit=limit, return_predecessors=True)
        out = np.full((len(targets), self.width), np.inf, dtype=np.float32)
        for i, t in enumerate(targets):
            if source == t:
                out[i] = 0
            elif math.isfinite(distances[t]):
                path = self._backtrace(source, int(t), predecessors)
                out[i] = self.features_for_path(path, distances[t])
        return out

    def prepare_source(self, source, targets, limit):
        targets = np.asarray(sorted(set(targets)), dtype=np.int64)
        p = self._source_file(source)
        existing = self._read_source_file(source)
        if existing.exists():
            with np.load(existing, allow_pickle=False) as z:
                if float(z["search_limit"]) >= limit and np.isin(targets, z["targets"]).all():
                    return len(targets)
                targets = np.union1d(targets, z["targets"])
                limit = max(limit, float(z["search_limit"]))
        features = self.compute_source(source, targets, limit)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, targets=targets, features=features, search_limit=np.asarray(limit))
        os.replace(tmp, p)
        self._drop_disk_source(source)
        return len(targets)

    def route_pairs(self, sources, targets, limits):
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        limits = np.broadcast_to(limits, sources.shape)
        output = np.zeros((len(sources), self.width), dtype=np.float32)
        reachable = np.zeros(len(sources), dtype=bool)
        missing = defaultdict(list)
        for i, (s, t, limit) in enumerate(zip(sources, targets, limits)):
            s, t = int(s), int(t)
            if s == t:
                reachable[i] = True
                continue
            known, feature = self._lookup(s, t, float(limit))
            if known:
                self.stats["cache_hits"] += 1
                if feature is not None:
                    output[i] = feature; reachable[i] = True
            else:
                missing[s].append(i)
        for s, indices in missing.items():
            maximum = float(limits[indices].max())
            requested = np.unique(targets[indices])
            features = self.compute_source(s, requested, maximum)
            self.stats["dijkstra_sources"] += 1
            for t, feature in zip(requested, features):
                self._remember(s, int(t), feature if np.isfinite(feature[0]) else None, maximum)
            for i in indices:
                f = features[np.searchsorted(requested, targets[i])]
                if f[0] <= limits[i] + 1e-5:
                    output[i] = f; reachable[i] = True
        return output, reachable

    def path_states(self, source, target):
        if source == target:
            return [int(source)]
        distance, predecessors = dijkstra(self.graph, directed=True, indices=int(source), return_predecessors=True)
        if not math.isfinite(distance[target]):
            return None
        return self._backtrace(int(source), int(target), predecessors)
