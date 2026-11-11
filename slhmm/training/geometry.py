"""Immutable-source geometry preparation shared by routing and evaluation."""
from __future__ import annotations

import numpy as np
import torch
import shapely
from pyproj import CRS

from slhmm.matching.osm_hmm import OSMEdgeHMMMatcher
from slhmm.training.common import load_pickle, save_pickle


def road_rank(value):
    values = value if isinstance(value, (tuple, list)) else [value]
    ranks = {"motorway": 1, "trunk": 2, "primary": 3, "secondary": 4,
             "tertiary": 5, "unclassified": 6, "residential": 6}
    return min(ranks.get(str(v).removesuffix("_link"), 7) for v in values)


def boolean_attribute(value):
    values = value if isinstance(value, (tuple, list)) else [value]
    return any(str(v).lower() not in ("none", "no", "false", "0", "", "nan") for v in values)


def tangents(line):
    diffs = np.diff(np.asarray(line.coords)[:, :2], axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    valid = norms > 1e-9
    if not valid.any():
        raise ValueError("A road geometry has no nonzero segment")
    unit = diffs[valid] / norms[valid, None]
    return unit[0], unit[-1]


def prepare_geometry(config, graph, mappings):
    matcher = OSMEdgeHMMMatcher(graph)
    if matcher.crs != CRS.from_user_input(config.metric_epsg):
        raise ValueError(f"Inconsistent metric CRS: {matcher.crs} vs {config.metric_epsg}")
    edges = [tuple(e) for e in mappings["state_id_to_edge"]]
    lines = np.asarray([matcher.edge_geometries[matcher._edge_to_index[e]] for e in edges], dtype=object)
    lengths = shapely.length(lines)
    directions = np.asarray([tangents(line) for line in lines], dtype=np.float32)
    attrs = [graph.get_edge_data(*e) for e in edges]
    root = config.feature_root / "road"
    save_pickle(root / "geometries.obj", list(lines))
    candidates = load_pickle(config.training_data_root / f"cell_candidates_k{config.candidate_k}.obj")
    positions = mappings["cell_id_to_position"]
    offsets = {}
    for cell, candidate in candidates.items():
        ids = np.asarray(candidate["state_ids"], dtype=np.int64)
        point = matcher.project_point(positions[int(cell)])
        offsets[int(cell)] = np.clip(shapely.line_locate_point(lines[ids], point, normalized=True), 0, 1).astype(np.float32)
    save_pickle(config.feature_root / f"cell_offsets_k{config.candidate_k}.obj", offsets)
    return {
        "length_m": torch.from_numpy(lengths.astype(np.float32)),
        "tangent_in": torch.from_numpy(directions[:, 0]),
        "tangent_out": torch.from_numpy(directions[:, 1]),
        "rank": torch.tensor([road_rank(a.get("highway")) for a in attrs]),
        "tunnel_raw": torch.tensor([boolean_attribute(a.get("tunnel")) for a in attrs]),
        "bridge_raw": torch.tensor([boolean_attribute(a.get("bridge")) for a in attrs]),
        "is_link": torch.tensor(["_link" in str(a.get("highway", "")) for a in attrs]),
    }
