"""Build portable road features and a relation-aware directed line graph."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Hashable

import dgl
import numpy as np
import osmnx as ox
from pyproj import Transformer
import torch

from slhmm.configs.training import TrainingConfig
from slhmm.training.common import (
    atomic_torch_save,
    load_pickle,
    sha256_file,
    write_json,
)


WGS84 = "EPSG:4326"
UTM51N = "EPSG:32651"
TURN_RELATIONS = ("straight", "left", "right", "u_turn")
RELATION_NAMES = TURN_RELATIONS + tuple(f"reverse_{name}" for name in TURN_RELATIONS) + (
    "same_physical",
)


def _first(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _category(value, default: str = "missing") -> str:
    value = _first(value)
    return default if value is None else str(value).strip().lower()


def _number(value, default: float = 0.0) -> tuple[float, float]:
    value = _first(value)
    if value is None:
        return default, 1.0
    text = str(value).lower().replace("km/h", "").strip()
    try:
        return float(text), 0.0
    except ValueError:
        digits = "".join(character for character in text if character.isdigit() or character == ".")
        try:
            return float(digits), 0.0
        except ValueError:
            return default, 1.0


def _boolean(value) -> float:
    value = _first(value)
    if isinstance(value, bool):
        return float(value)
    return float(str(value).strip().lower() in {"1", "yes", "true", "y"})


def _bearing_degrees(dx: float, dy: float) -> float:
    # Compass bearing: north=0, east=90.
    return float((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)


def _turn_relation(previous_bearing: float, current_bearing: float) -> str:
    delta = (current_bearing - previous_bearing + 180.0) % 360.0 - 180.0
    magnitude = abs(delta)
    if magnitude <= 30.0:
        return "straight"
    if magnitude >= 150.0:
        return "u_turn"
    return "right" if delta > 0 else "left"


def _edge_xy(
    graph,
    edge: tuple[Hashable, Hashable, Hashable],
    transformer: Transformer,
) -> tuple[float, float, float, float]:
    u, v, _ = edge
    ux, uy = transformer.transform(
        float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])
    )
    vx, vy = transformer.transform(
        float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])
    )
    return float(ux), float(uy), float(vx), float(vy)


def _build_feature_tensors(graph, mappings: dict, metric_epsg="EPSG:32651") -> tuple[dict, dict]:
    transformer = Transformer.from_crs(WGS84, metric_epsg, always_xy=True)
    state_edges = [tuple(edge) for edge in mappings["state_id_to_edge"]]
    highway_values = sorted(
        {
            _category(graph.get_edge_data(*edge).get("highway"))
            for edge in state_edges
        }
    )
    highway_to_id = {name: index for index, name in enumerate(highway_values)}
    rows = []
    bearings = []
    feature_names = [
        "log1p_length",
        "oneway",
        "bearing_sin",
        "bearing_cos",
        "midpoint_x",
        "midpoint_y",
        "curvature",
        "bridge",
        "tunnel",
        "lanes",
        "lanes_missing",
        "maxspeed",
        "maxspeed_missing",
        "in_degree",
        "out_degree",
        "has_reverse",
    ]
    highway_ids = []
    midpoint_xy = []
    start_xy = []
    end_xy = []

    directed_endpoint_pairs = {(edge[0], edge[1]) for edge in state_edges}
    for edge in state_edges:
        u, v, _ = edge
        attributes = graph.get_edge_data(*edge)
        ux, uy, vx, vy = _edge_xy(graph, edge, transformer)
        dx, dy = vx - ux, vy - uy
        chord = max(math.hypot(dx, dy), 1.0)
        length = float(attributes.get("length", chord))
        bearing = _bearing_degrees(dx, dy)
        bearings.append(bearing)
        midpoint_xy.append(((ux + vx) / 2.0, (uy + vy) / 2.0))
        start_xy.append((ux, uy))
        end_xy.append((vx, vy))
        lanes, lanes_missing = _number(attributes.get("lanes"))
        maxspeed, maxspeed_missing = _number(attributes.get("maxspeed"))
        has_reverse = (v, u) in directed_endpoint_pairs
        rows.append(
            [
                math.log1p(max(length, 0.0)),
                _boolean(attributes.get("oneway")),
                math.sin(math.radians(bearing)),
                math.cos(math.radians(bearing)),
                (ux + vx) / 2.0,
                (uy + vy) / 2.0,
                max(length / chord, 1.0),
                _boolean(attributes.get("bridge")),
                _boolean(attributes.get("tunnel")),
                lanes,
                lanes_missing,
                maxspeed,
                maxspeed_missing,
                float(graph.in_degree(u)),
                float(graph.out_degree(v)),
                float(has_reverse),
            ]
        )
        highway_ids.append(highway_to_id[_category(attributes.get("highway"))])

    raw = np.asarray(rows, dtype=np.float32)
    means = raw.mean(axis=0)
    stds = raw.std(axis=0)
    stds[stds < 1e-6] = 1.0
    normalized = (raw - means) / stds
    payload = {
        "numeric": torch.from_numpy(normalized),
        "highway_id": torch.tensor(highway_ids, dtype=torch.long),
        "bearing_degrees": torch.tensor(bearings, dtype=torch.float32),
        "midpoint_xy": torch.tensor(midpoint_xy, dtype=torch.float32),
        "start_xy": torch.tensor(start_xy, dtype=torch.float32),
        "end_xy": torch.tensor(end_xy, dtype=torch.float32),
        "edge_u": torch.tensor([int(edge[0]) for edge in state_edges], dtype=torch.int64),
        "edge_v": torch.tensor([int(edge[1]) for edge in state_edges], dtype=torch.int64),
        "physical_id": torch.as_tensor(
            mappings["state_to_physical_id"], dtype=torch.int64
        ),
    }
    metadata = {
        "feature_names": feature_names,
        "numeric_mean": means.tolist(),
        "numeric_std": stds.tolist(),
        "highway_vocabulary": highway_values,
    }
    return payload, metadata


def _build_relation_graph(mappings: dict, bearings: torch.Tensor):
    state_edges = [tuple(edge) for edge in mappings["state_id_to_edge"]]
    by_start: dict[Hashable, list[int]] = defaultdict(list)
    for state_id, edge in enumerate(state_edges):
        by_start[edge[0]].append(state_id)

    relation_pairs: dict[str, tuple[list[int], list[int]]] = {
        name: ([], []) for name in RELATION_NAMES
    }
    bearing_values = bearings.tolist()
    for previous_id, previous_edge in enumerate(state_edges):
        for current_id in by_start.get(previous_edge[1], ()):
            relation = _turn_relation(
                bearing_values[previous_id], bearing_values[current_id]
            )
            relation_pairs[relation][0].append(previous_id)
            relation_pairs[relation][1].append(current_id)
            reverse_relation = f"reverse_{relation}"
            relation_pairs[reverse_relation][0].append(current_id)
            relation_pairs[reverse_relation][1].append(previous_id)

    for state_ids in mappings["physical_to_state_ids"]:
        for source in state_ids:
            for target in state_ids:
                if source != target:
                    relation_pairs["same_physical"][0].append(int(source))
                    relation_pairs["same_physical"][1].append(int(target))

    graph_data = {
        ("road", relation, "road"): (
            torch.tensor(sources, dtype=torch.int64),
            torch.tensor(targets, dtype=torch.int64),
        )
        for relation, (sources, targets) in relation_pairs.items()
    }
    graph = dgl.heterograph(
        graph_data,
        num_nodes_dict={"road": len(state_edges)},
    )
    counts = {
        relation: int(graph.num_edges(("road", relation, "road")))
        for relation in RELATION_NAMES
    }
    return graph, counts


def prepare_road_artifacts(config: TrainingConfig) -> dict:
    """Create road feature tensors and the DGL directed line graph."""

    output_root = config.feature_root / "road"
    output_root.mkdir(parents=True, exist_ok=True)
    mappings_path = config.training_data_root / "mappings.obj"
    mappings = load_pickle(mappings_path)
    graph = ox.load_graphml(config.graph_path)
    features, feature_metadata = _build_feature_tensors(graph, mappings, config.metric_epsg)
    from slhmm.training.geometry import prepare_geometry
    features.update(prepare_geometry(config, graph, mappings))
    line_graph, relation_counts = _build_relation_graph(
        mappings, features["bearing_degrees"]
    )

    feature_path = output_root / "road_features.pt"
    graph_path = output_root / "road_relations.dgl"
    atomic_torch_save(feature_path, features)
    temporary_graph = graph_path.with_suffix(".dgl.tmp")
    dgl.save_graphs(str(temporary_graph), [line_graph])
    temporary_graph.replace(graph_path)
    metadata = {
        "schema_version": 2,
        "metric_epsg": config.metric_epsg,
        "snapshot": config.snapshot,
        "graph_sha256": sha256_file(config.graph_path),
        "mappings_sha256": sha256_file(mappings_path),
        "state_count": len(mappings["state_id_to_edge"]),
        "physical_segment_count": len(mappings["physical_to_state_ids"]),
        "relation_counts": relation_counts,
        "relation_names": list(RELATION_NAMES),
        **feature_metadata,
    }
    write_json(output_root / "metadata.json", metadata)
    return metadata


def load_road_artifacts(config: TrainingConfig):
    root = config.feature_root / "road"
    graphs, _ = dgl.load_graphs(str(root / "road_relations.dgl"))
    try:
        features = torch.load(
            root / "road_features.pt", map_location="cpu", weights_only=True
        )
    except TypeError:
        features = torch.load(root / "road_features.pt", map_location="cpu")
    return graphs[0], features
