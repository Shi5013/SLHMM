"""Directed edge-based HMM map matching for OSMnx road graphs.

The legacy matcher treats road-network nodes as HMM states.  This module is
deliberately separate: its states are directed OSM edges and candidate
distances are computed against the complete edge geometry in a metric CRS.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Hashable, Iterable

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform
from shapely.strtree import STRtree


EdgeId = tuple[Hashable, Hashable, Hashable]


@dataclass(frozen=True)
class EdgeCandidate:
    edge: EdgeId
    distance_m: float
    offset_m: float
    edge_length_m: float


@dataclass
class MatchResult:
    success: bool
    reason: str | None
    edges: list[EdgeId]
    selected_candidates: list[EdgeCandidate]
    valid_point_indices: list[int]
    point_candidate_counts: list[int]
    skipped_point_indices: list[int]
    log_probability: float | None


def _edge_linestring(
    graph: nx.MultiDiGraph,
    edge: EdgeId,
    geometry_transformer: Transformer | None = None,
) -> LineString:
    u, v, key = edge
    data = graph.edges[u, v, key]
    geom = data.get("geometry")
    if geom is None:
        geom = LineString(
            [(graph.nodes[u]["x"], graph.nodes[u]["y"]),
             (graph.nodes[v]["x"], graph.nodes[v]["y"])]
        )
    elif geom.geom_type != "LineString":
        geom = LineString([coord for part in geom.geoms for coord in part.coords])

    u_point = Point(graph.nodes[u]["x"], graph.nodes[u]["y"])
    if geometry_transformer is not None:
        start = Point(geom.coords[0])
        end = Point(geom.coords[-1])
        # OSMnx projects edge geometry on simplified graphs.  On unsimplified
        # or hand-built graphs it can retain lon/lat geometry while projecting
        # nodes, so handle both representations safely.
        if min(start.distance(u_point), end.distance(u_point)) > 10_000:
            geom = transform(geometry_transformer.transform, geom)

    # OSM geometries are normally u -> v, but explicitly normalize orientation
    # because candidate offsets and directed transition distances depend on it.
    start = Point(geom.coords[0])
    end = Point(geom.coords[-1])
    if end.distance(u_point) < start.distance(u_point):
        geom = LineString(list(geom.coords)[::-1])
    return geom


class OSMEdgeHMMMatcher:
    """Map match WGS84 GPS coordinates to directed OSM edge sequences."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        *,
        candidate_radius_m: float = 200.0,
        max_candidates: int = 8,
        sigma_m: float = 20.0,
        beta_m: float = 50.0,
        minimum_candidate_fraction: float = 0.8,
    ) -> None:
        if candidate_radius_m <= 0 or max_candidates <= 0:
            raise ValueError("candidate radius and count must be positive")
        if sigma_m <= 0 or beta_m <= 0:
            raise ValueError("sigma_m and beta_m must be positive")

        self.graph_wgs84 = graph
        graph_crs = CRS.from_user_input(graph.graph.get("crs", "EPSG:4326"))
        self.graph = ox.project_graph(graph) if graph_crs.is_geographic else graph.copy()
        self.crs = CRS.from_user_input(self.graph.graph["crs"])
        self.to_metric = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

        self.candidate_radius_m = float(candidate_radius_m)
        self.max_candidates = int(max_candidates)
        self.sigma_m = float(sigma_m)
        self.beta_m = float(beta_m)
        self.minimum_candidate_fraction = float(minimum_candidate_fraction)

        self.edge_ids: list[EdgeId] = list(self.graph.edges(keys=True))
        geometry_transformer = self.to_metric if graph_crs.is_geographic else None
        self.edge_geometries = [
            _edge_linestring(self.graph, edge, geometry_transformer) for edge in self.edge_ids
        ]
        self.edge_lengths = [float(geom.length) for geom in self.edge_geometries]
        self.edge_index = STRtree(self.edge_geometries)
        self._edge_to_index = {edge: index for index, edge in enumerate(self.edge_ids)}
        self._shortest_distance_cache: dict[tuple[Hashable, Hashable], float] = {}
        self._shortest_path_cache: dict[tuple[Hashable, Hashable], list[Hashable] | None] = {}
        self._weak_component = {}
        for component_id, nodes in enumerate(nx.weakly_connected_components(self.graph)):
            for node in nodes:
                self._weak_component[node] = component_id

    def project_point(self, position: tuple[float, float] | tuple[float, float, object]) -> Point:
        x, y = self.to_metric.transform(float(position[0]), float(position[1]))
        return Point(x, y)

    def candidates(self, position: tuple[float, float] | tuple[float, float, object]) -> list[EdgeCandidate]:
        point = self.project_point(position)
        indices = self.edge_index.query(point.buffer(self.candidate_radius_m))
        ranked: list[EdgeCandidate] = []
        for index in indices:
            idx = int(index)
            geom = self.edge_geometries[idx]
            distance = float(point.distance(geom))
            if distance <= self.candidate_radius_m:
                ranked.append(
                    EdgeCandidate(
                        edge=self.edge_ids[idx],
                        distance_m=distance,
                        offset_m=float(geom.project(point)),
                        edge_length_m=self.edge_lengths[idx],
                    )
                )
        ranked.sort(key=lambda item: (item.distance_m, repr(item.edge)))
        return ranked[: self.max_candidates]

    def _shortest_distance(self, source: Hashable, target: Hashable) -> float:
        if source == target:
            return 0.0
        cache_key = (source, target)
        if cache_key not in self._shortest_distance_cache:
            try:
                value = nx.shortest_path_length(self.graph, source, target, weight="length")
                self._shortest_distance_cache[cache_key] = float(value)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self._shortest_distance_cache[cache_key] = math.inf
        return self._shortest_distance_cache[cache_key]

    def _shortest_distances(
        self, source: Hashable, targets: Iterable[Hashable]
    ) -> dict[Hashable, float]:
        """Resolve several nearby targets with one early-stopping Dijkstra.

        A Viterbi layer has at most eight candidates. Running an independent
        NetworkX shortest-path search for every candidate pair dominates full
        dataset runtime. This search stops as soon as all reachable target
        nodes have been settled and also records paths for later expansion.
        """
        unique_targets = set(targets)
        result: dict[Hashable, float] = {}
        unresolved = set()
        for target in unique_targets:
            key = (source, target)
            if source == target:
                result[target] = 0.0
                self._shortest_distance_cache[key] = 0.0
                self._shortest_path_cache[key] = [source]
            elif key in self._shortest_distance_cache:
                result[target] = self._shortest_distance_cache[key]
            elif self._weak_component.get(source) != self._weak_component.get(target):
                result[target] = math.inf
                self._shortest_distance_cache[key] = math.inf
                self._shortest_path_cache[key] = None
            else:
                unresolved.add(target)
        if not unresolved:
            return result

        distances = {source: 0.0}
        predecessors: dict[Hashable, Hashable] = {}
        settled: set[Hashable] = set()
        counter = itertools.count()
        queue = [(0.0, next(counter), source)]
        while queue and unresolved:
            distance, _, node = heapq.heappop(queue)
            if node in settled:
                continue
            settled.add(node)
            if node in unresolved:
                path = [node]
                cursor = node
                while cursor != source:
                    cursor = predecessors[cursor]
                    path.append(cursor)
                path.reverse()
                result[node] = distance
                self._shortest_distance_cache[(source, node)] = distance
                self._shortest_path_cache[(source, node)] = path
                unresolved.remove(node)
                if not unresolved:
                    break
            for neighbor, keyed_edges in self.graph[node].items():
                weight = min(
                    float(attributes.get("length", 1.0))
                    for attributes in keyed_edges.values()
                )
                candidate_distance = distance + weight
                if candidate_distance < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate_distance
                    predecessors[neighbor] = node
                    heapq.heappush(queue, (candidate_distance, next(counter), neighbor))

        for target in unresolved:
            result[target] = math.inf
            self._shortest_distance_cache[(source, target)] = math.inf
            self._shortest_path_cache[(source, target)] = None
        return result

    def _route_distance_matrix(
        self,
        previous_layer: list[EdgeCandidate],
        current_layer: list[EdgeCandidate],
    ) -> np.ndarray:
        matrix = np.full((len(previous_layer), len(current_layer)), math.inf, dtype=float)
        targets = {candidate.edge[0] for candidate in current_layer}
        distances_by_source = {
            source: self._shortest_distances(source, targets)
            for source in {candidate.edge[1] for candidate in previous_layer}
        }
        for previous_index, previous in enumerate(previous_layer):
            for current_index, current in enumerate(current_layer):
                if previous.edge == current.edge and current.offset_m >= previous.offset_m:
                    matrix[previous_index, current_index] = current.offset_m - previous.offset_m
                    continue
                between = distances_by_source[previous.edge[1]][current.edge[0]]
                if math.isfinite(between):
                    matrix[previous_index, current_index] = (
                        max(0.0, previous.edge_length_m - previous.offset_m)
                        + between
                        + max(0.0, current.offset_m)
                    )
        return matrix

    def _route_distance(self, previous: EdgeCandidate, current: EdgeCandidate) -> float:
        if previous.edge == current.edge and current.offset_m >= previous.offset_m:
            return current.offset_m - previous.offset_m
        between = self._shortest_distance(previous.edge[1], current.edge[0])
        if not math.isfinite(between):
            return math.inf
        return (
            max(0.0, previous.edge_length_m - previous.offset_m)
            + between
            + max(0.0, current.offset_m)
        )

    def _shortest_nodes(self, source: Hashable, target: Hashable) -> list[Hashable] | None:
        if source == target:
            return [source]
        cache_key = (source, target)
        if cache_key not in self._shortest_path_cache:
            try:
                self._shortest_path_cache[cache_key] = nx.shortest_path(
                    self.graph, source, target, weight="length"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self._shortest_path_cache[cache_key] = None
        return self._shortest_path_cache[cache_key]

    def _best_edge(self, u: Hashable, v: Hashable) -> EdgeId:
        keys = self.graph[u][v]
        key = min(
            keys,
            key=lambda candidate_key: float(keys[candidate_key].get("length", math.inf)),
        )
        return (u, v, key)

    def _expand_candidates(self, selected: list[EdgeCandidate]) -> list[EdgeId]:
        if not selected:
            return []
        result: list[EdgeId] = [selected[0].edge]
        for previous, current in zip(selected, selected[1:]):
            if previous.edge == current.edge and current.offset_m >= previous.offset_m:
                continue
            nodes = self._shortest_nodes(previous.edge[1], current.edge[0])
            if nodes is None:
                return []
            for u, v in zip(nodes, nodes[1:]):
                edge = self._best_edge(u, v)
                if result[-1] != edge:
                    result.append(edge)
            if result[-1] != current.edge:
                result.append(current.edge)
        return result

    def match(self, positions: Iterable[tuple[float, float] | tuple[float, float, object]]) -> MatchResult:
        # Candidate-node pairs vary strongly between trajectories. Keeping all
        # pairwise routes across 10k trajectories grows without bound and has
        # little reuse, so scope the routing cache to one trajectory.
        self._shortest_distance_cache.clear()
        self._shortest_path_cache.clear()
        positions = list(positions)
        if len(positions) < 2:
            return MatchResult(False, "fewer_than_two_points", [], [], [], [0] * len(positions), list(range(len(positions))), None)

        all_candidates = [self.candidates(position) for position in positions]
        counts = [len(layer) for layer in all_candidates]
        valid_indices = [index for index, layer in enumerate(all_candidates) if layer]
        skipped = [index for index, layer in enumerate(all_candidates) if not layer]
        required = max(2, math.ceil(len(positions) * self.minimum_candidate_fraction))
        if len(valid_indices) < required:
            return MatchResult(False, "insufficient_candidate_coverage", [], [], valid_indices, counts, skipped, None)

        layers = [all_candidates[index] for index in valid_indices]
        valid_points = [self.project_point(positions[index]) for index in valid_indices]
        scores: list[np.ndarray] = []
        backpointers: list[np.ndarray] = []

        first = np.array([-0.5 * (candidate.distance_m / self.sigma_m) ** 2 for candidate in layers[0]])
        scores.append(first)
        backpointers.append(np.full(len(layers[0]), -1, dtype=int))

        for layer_index in range(1, len(layers)):
            previous_layer = layers[layer_index - 1]
            current_layer = layers[layer_index]
            gps_distance = float(valid_points[layer_index - 1].distance(valid_points[layer_index]))
            current_scores = np.full(len(current_layer), -math.inf, dtype=float)
            current_backpointers = np.full(len(current_layer), -1, dtype=int)
            route_distances = self._route_distance_matrix(previous_layer, current_layer)
            for current_index, current in enumerate(current_layer):
                emission = -0.5 * (current.distance_m / self.sigma_m) ** 2
                for previous_index, previous in enumerate(previous_layer):
                    route_distance = route_distances[previous_index, current_index]
                    if not math.isfinite(route_distance):
                        continue
                    transition = -abs(route_distance - gps_distance) / self.beta_m
                    score = scores[-1][previous_index] + transition + emission
                    if score > current_scores[current_index]:
                        current_scores[current_index] = score
                        current_backpointers[current_index] = previous_index
            if not np.isfinite(current_scores).any():
                return MatchResult(False, "no_connected_candidate_path", [], [], valid_indices, counts, skipped, None)
            scores.append(current_scores)
            backpointers.append(current_backpointers)

        selected_indices = [int(np.nanargmax(scores[-1]))]
        for layer_index in range(len(layers) - 1, 0, -1):
            previous_index = int(backpointers[layer_index][selected_indices[-1]])
            if previous_index < 0:
                return MatchResult(False, "broken_viterbi_backpointer", [], [], valid_indices, counts, skipped, None)
            selected_indices.append(previous_index)
        selected_indices.reverse()
        selected = [layer[index] for layer, index in zip(layers, selected_indices)]
        edges = self._expand_candidates(selected)
        if not edges:
            return MatchResult(False, "failed_to_expand_route", [], selected, valid_indices, counts, skipped, None)
        return MatchResult(
            True,
            None,
            edges,
            selected,
            valid_indices,
            counts,
            skipped,
            float(np.max(scores[-1])),
        )

    def edge_geometry_wgs84(self, edge: EdgeId) -> LineString:
        projected_edge = (edge[0], edge[1], edge[2])
        index = self._edge_to_index[projected_edge]
        return transform(self.to_wgs84.transform, self.edge_geometries[index])

    def route_geometries_metric(self, edges: Iterable[EdgeId]) -> list[LineString]:
        return [self.edge_geometries[self._edge_to_index[tuple(edge)]] for edge in edges]


def edge_id_to_json(edge: EdgeId) -> list[object]:
    """Convert numpy/OSM scalar edge IDs to JSON-safe values."""

    def scalar(value: object) -> object:
        return value.item() if hasattr(value, "item") else value

    return [scalar(value) for value in edge]
