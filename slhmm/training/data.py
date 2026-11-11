"""Data preparation and trajectory-oriented batching for Emit/Trans."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Iterator

import numpy as np
import osmnx as ox
from pyproj import Transformer
import torch

from slhmm.configs.training import TrainingConfig
from slhmm.training.common import (
    load_pickle,
    portable_shard_path,
    read_json,
    save_pickle,
    sha256_file,
    write_json,
)


WGS84 = "EPSG:4326"
UTM51N = "EPSG:32651"
PAIR_FEATURE_DIM = 9


class _Reservoir:
    def __init__(self, maximum: int, seed: int):
        self.maximum = maximum
        self.random = random.Random(seed)
        self.values: list[float] = []
        self.seen = 0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.seen += 1
        if len(self.values) < self.maximum:
            self.values.append(float(value))
            return
        position = self.random.randrange(self.seen)
        if position < self.maximum:
            self.values[position] = float(value)

    def median(self, minimum: float = 1.0) -> float:
        if not self.values:
            return minimum
        return max(float(np.median(np.asarray(self.values))), minimum)


def _cell_metric_positions(mappings: dict, metric_epsg="EPSG:32651") -> np.ndarray:
    transformer = Transformer.from_crs(WGS84, metric_epsg, always_xy=True)
    positions = mappings["cell_id_to_position"]
    xs, ys = transformer.transform(
        [position[0] for position in positions],
        [position[1] for position in positions],
    )
    return np.column_stack((xs, ys)).astype(np.float32)


def _road_lengths(config: TrainingConfig, mappings: dict) -> np.ndarray:
    graph = ox.load_graphml(config.graph_path)
    return np.asarray(
        [
            float(graph.get_edge_data(*tuple(edge)).get("length", 0.0))
            for edge in mappings["state_id_to_edge"]
        ],
        dtype=np.float64,
    )


def _cumulative_route_lengths(
    truth_state_ids: list[int], road_lengths: np.ndarray
) -> np.ndarray:
    if not truth_state_ids:
        return np.zeros(1, dtype=np.float64)
    values = road_lengths[np.asarray(truth_state_ids, dtype=np.int64)]
    return np.concatenate((np.zeros(1), np.cumsum(values)))


def _subpath_length(
    cumulative: np.ndarray, previous_index: int, current_index: int
) -> float:
    if current_index <= previous_index:
        return 0.0
    return float(cumulative[current_index] - cumulative[previous_index])


def _rank_distance(cell_candidates: dict, cell_id: int, rank: int) -> float:
    distances = cell_candidates[cell_id]["physical_distances_m"]
    if rank <= 0 or rank > len(distances):
        return float("inf")
    return float(distances[rank - 1])


def _positive_pair_details(
    record: dict,
    *,
    truth_state_ids: list[int],
    cumulative_lengths: np.ndarray,
    cell_positions_metric: np.ndarray,
    cell_candidates: dict,
    endpoint_scale: float,
    motion_scale: float,
) -> tuple[list[float], list[dict]]:
    previous_by_state: dict[int, list[tuple[int, int]]] = defaultdict(list)
    current_by_state: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for route_index, state_id, rank in record["previous_allowed_occurrences"]:
        previous_by_state[int(state_id)].append((int(route_index), int(rank)))
    for route_index, state_id, rank in record["current_allowed_occurrences"]:
        current_by_state[int(state_id)].append((int(route_index), int(rank)))

    previous_cell = int(record["previous_cell_id"])
    current_cell = int(record["current_cell_id"])
    cell_distance = float(
        np.linalg.norm(
            cell_positions_metric[current_cell] - cell_positions_metric[previous_cell]
        )
    )
    log_weights = []
    metadata = []
    for previous_state, current_state in record["positive_state_pairs"]:
        best = None
        for previous_index, previous_rank in previous_by_state[int(previous_state)]:
            for current_index, current_rank in current_by_state[int(current_state)]:
                if previous_index > current_index:
                    continue
                route_length = _subpath_length(
                    cumulative_lengths, previous_index, current_index
                )
                previous_distance = _rank_distance(
                    cell_candidates, previous_cell, previous_rank
                )
                current_distance = _rank_distance(
                    cell_candidates, current_cell, current_rank
                )
                log_weight = -(
                    (previous_distance + current_distance)
                    / (2.0 * endpoint_scale)
                    + abs(route_length - cell_distance) / motion_scale
                )
                candidate = (
                    log_weight,
                    previous_index,
                    current_index,
                    route_length,
                    previous_distance,
                    current_distance,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            raise ValueError(
                f"No ordered occurrence supports positive pair "
                f"{previous_state, current_state} in {record['trajectory_id']}"
            )
        log_weights.append(float(best[0]))
        metadata.append(
            {
                "previous_route_index": int(best[1]),
                "current_route_index": int(best[2]),
                "route_index_gap": int(best[2] - best[1]),
                "ground_truth_subpath_length_m": float(best[3]),
                "previous_cell_road_distance_m": float(best[4]),
                "current_cell_road_distance_m": float(best[5]),
                "cell_displacement_m": cell_distance,
            }
        )
    normalizer = float(np.logaddexp.reduce(np.asarray(log_weights)))
    return [value - normalizer for value in log_weights], metadata


def _iter_manifest_records(
    config: TrainingConfig, family: str, split: str
) -> Iterator[dict]:
    manifest = read_json(config.training_data_root / "manifest.json")
    for shard in manifest[f"{family}_shards"][split]:
        path = portable_shard_path(
            shard["path"],
            root=config.training_data_root,
            split=split,
            family=family,
        )
        yield from load_pickle(path)


def _estimate_trans_scales(
    config: TrainingConfig,
    *,
    alignments: dict,
    cell_candidates: dict,
    cell_positions_metric: np.ndarray,
    road_lengths: np.ndarray,
) -> tuple[float, float]:
    endpoint = _Reservoir(200_000, config.seed)
    motion = _Reservoir(200_000, config.seed + 1)
    cumulative_cache = {}
    for record in _iter_manifest_records(config, "trans", "train"):
        trajectory_id = record["trajectory_id"]
        alignment = alignments[trajectory_id]
        cumulative = cumulative_cache.get(trajectory_id)
        if cumulative is None:
            cumulative = _cumulative_route_lengths(
                alignment["truth_state_ids"], road_lengths
            )
            cumulative_cache = {trajectory_id: cumulative}
        previous_cell = int(record["previous_cell_id"])
        current_cell = int(record["current_cell_id"])
        cell_distance = float(
            np.linalg.norm(
                cell_positions_metric[current_cell]
                - cell_positions_metric[previous_cell]
            )
        )
        for route_index, _, rank in record["previous_allowed_occurrences"]:
            endpoint.add(_rank_distance(cell_candidates, previous_cell, int(rank)))
        for route_index, _, rank in record["current_allowed_occurrences"]:
            endpoint.add(_rank_distance(cell_candidates, current_cell, int(rank)))
        previous = record["previous_allowed_occurrences"]
        current = record["current_allowed_occurrences"]
        for previous_index, _, _ in previous:
            for current_index, _, _ in current:
                if previous_index <= current_index:
                    route_length = _subpath_length(
                        cumulative, int(previous_index), int(current_index)
                    )
                    motion.add(abs(route_length - cell_distance))
    return endpoint.median(50.0), motion.median(100.0)


def prepare_enriched_trans(config: TrainingConfig) -> dict:
    """Add occurrence-level macro-transition weights without overwriting source data."""

    root = config.training_data_root
    mappings = load_pickle(root / "mappings.obj")
    cell_candidates = load_pickle(root / f"cell_candidates_k{config.candidate_k}.obj")
    cell_positions_metric = _cell_metric_positions(mappings, config.metric_epsg)
    road_lengths = _road_lengths(config, mappings)
    alignments = {
        "train": load_pickle(root / "alignment_train.obj"),
        "val": load_pickle(root / "alignment_val.obj"),
    }
    endpoint_scale, motion_scale = _estimate_trans_scales(
        config,
        alignments=alignments["train"],
        cell_candidates=cell_candidates,
        cell_positions_metric=cell_positions_metric,
        road_lengths=road_lengths,
    )
    output_root = config.feature_root / "trans_enriched"
    shard_manifest = {}

    for split in ("train", "val"):
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        output_shards = []
        trajectory_groups = []
        current_id = None
        current_records = []
        shard_index = 0

        def flush_shard() -> None:
            nonlocal trajectory_groups, shard_index
            if not trajectory_groups:
                return
            path = split_root / f"trans_{shard_index:02d}.obj"
            save_pickle(path, trajectory_groups)
            output_shards.append(
                {
                    "path": str(path.relative_to(config.feature_root)),
                    "trajectory_count": len(trajectory_groups),
                    "transition_count": sum(
                        len(group["transitions"]) for group in trajectory_groups
                    ),
                }
            )
            trajectory_groups = []
            shard_index += 1

        def flush_trajectory() -> None:
            nonlocal current_records
            if current_id is None:
                return
            trajectory_groups.append(
                {"trajectory_id": current_id, "transitions": current_records}
            )
            current_records = []
            if len(trajectory_groups) >= 1000:
                flush_shard()

        cumulative_cache = {}
        for record in _iter_manifest_records(config, "trans", split):
            trajectory_id = record["trajectory_id"]
            if current_id is not None and trajectory_id != current_id:
                flush_trajectory()
            current_id = trajectory_id
            alignment = alignments[split][trajectory_id]
            cumulative = cumulative_cache.get(trajectory_id)
            if cumulative is None:
                cumulative = _cumulative_route_lengths(
                    alignment["truth_state_ids"], road_lengths
                )
                cumulative_cache = {trajectory_id: cumulative}
            log_weights, positive_metadata = _positive_pair_details(
                record,
                truth_state_ids=alignment["truth_state_ids"],
                cumulative_lengths=cumulative,
                cell_positions_metric=cell_positions_metric,
                cell_candidates=cell_candidates,
                endpoint_scale=endpoint_scale,
                motion_scale=motion_scale,
            )
            enriched = dict(record)
            enriched["positive_log_weights"] = log_weights
            enriched["positive_metadata"] = positive_metadata
            current_records.append(enriched)
        flush_trajectory()
        flush_shard()
        shard_manifest[split] = output_shards

    manifest = {
        "schema_version": 1,
        "source_manifest_sha256": sha256_file(root / "manifest.json"),
        "graph_sha256": sha256_file(config.graph_path),
        "endpoint_distance_scale_m": endpoint_scale,
        "motion_mismatch_scale_m": motion_scale,
        "positive_weight_rule": (
            "-(prev_cell_road + curr_cell_road)/(2*endpoint_scale) "
            "- abs(truth_subpath_length-cell_displacement)/motion_scale"
        ),
        "model_input_policy": (
            "ground-truth subpath fields are supervision metadata only and "
            "must never be passed to the Trans scorer"
        ),
        "shards": shard_manifest,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def _refreshed_negative_pairs(
    previous_candidates: list[int],
    current_candidates: list[int],
    positive_pairs: list[tuple[int, int]],
    *,
    maximum: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Build hard negatives from the same Emit pools used by Viterbi."""

    positives = {(int(previous), int(current)) for previous, current in positive_pairs}
    positive_previous = {pair[0] for pair in positives}
    positive_current = {pair[1] for pair in positives}
    ordered = []

    # Prefer one-correct-endpoint confounders, then fill with the joint pool.
    for previous in previous_candidates:
        for current in current_candidates:
            pair = (int(previous), int(current))
            if pair in positives:
                continue
            if previous in positive_previous or current in positive_current:
                ordered.append(pair)
    for previous in previous_candidates:
        for current in current_candidates:
            pair = (int(previous), int(current))
            if pair not in positives and pair not in ordered:
                ordered.append(pair)
    random.Random(seed).shuffle(ordered)
    return ordered[:maximum]


def refresh_trans_from_emit_candidates(config: TrainingConfig) -> dict:
    """Rebuild Trans pair pools after Emit Top-K refresh.

    Training keeps every soft-positive endpoint even when Emit misses it, while
    validation remains honest: it uses only the states actually returned by
    Emit and reports candidate misses separately.
    """

    source_root = config.feature_root / "trans_enriched"
    source_manifest = read_json(source_root / "manifest.json")
    emit_root = (
        config.feature_root / "emit_candidates" / config.experiment_id
    )
    output_root = (
        config.feature_root / "trans_refreshed" / config.experiment_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_manifest: dict[str, list[dict]] = {}
    statistics = {}

    for split in ("train", "val"):
        emit_candidates = load_pickle(emit_root / f"{split}.obj")
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        output_shards = []
        trajectory_count = transition_count = kept_count = candidate_misses = 0

        for shard_index, source_shard in enumerate(
            source_manifest["shards"][split]
        ):
            groups = load_pickle(config.feature_root / source_shard["path"])
            refreshed_groups = []
            for group in groups:
                trajectory_id = group["trajectory_id"]
                if trajectory_id not in emit_candidates:
                    continue
                trajectory_count += 1
                per_observation = emit_candidates[trajectory_id]
                transitions = []
                for record in group["transitions"]:
                    transition_count += 1
                    previous_index = int(record["previous_observation_index"])
                    current_index = int(record["current_observation_index"])
                    previous_pool = list(
                        dict.fromkeys(per_observation[previous_index])
                    )
                    current_pool = list(
                        dict.fromkeys(per_observation[current_index])
                    )
                    source_pairs = [
                        (int(previous), int(current))
                        for previous, current in record["positive_state_pairs"]
                    ]
                    source_weights = [
                        float(value)
                        for value in record["positive_log_weights"]
                    ]

                    if split == "train":
                        for previous, current in source_pairs:
                            if previous not in previous_pool:
                                previous_pool.append(previous)
                            if current not in current_pool:
                                current_pool.append(current)
                        selected = list(range(len(source_pairs)))
                    else:
                        selected = [
                            index
                            for index, pair in enumerate(source_pairs)
                            if pair[0] in previous_pool
                            and pair[1] in current_pool
                        ]
                        if not selected:
                            candidate_misses += 1
                            continue

                    positive_pairs = [source_pairs[index] for index in selected]
                    positive_weights = [source_weights[index] for index in selected]
                    normalizer = float(
                        np.logaddexp.reduce(np.asarray(positive_weights))
                    )
                    positive_weights = [
                        value - normalizer for value in positive_weights
                    ]
                    negatives = _refreshed_negative_pairs(
                        previous_pool,
                        current_pool,
                        positive_pairs,
                        maximum=config.trans_negative_count,
                        seed=(
                            config.seed
                            + shard_index * 1_000_003
                            + transition_count
                        ),
                    )
                    if not negatives:
                        continue
                    refreshed = dict(record)
                    refreshed["positive_state_pairs"] = positive_pairs
                    refreshed["positive_log_weights"] = positive_weights
                    refreshed["positive_metadata"] = [
                        record["positive_metadata"][index] for index in selected
                    ]
                    refreshed["negative_state_pairs"] = negatives
                    refreshed["emit_previous_candidates"] = previous_pool
                    refreshed["emit_current_candidates"] = current_pool
                    refreshed["candidate_policy"] = (
                        "top_k_plus_soft_positive"
                        if split == "train"
                        else "top_k_only"
                    )
                    transitions.append(refreshed)
                    kept_count += 1
                if transitions:
                    refreshed_groups.append(
                        {
                            "trajectory_id": trajectory_id,
                            "transitions": transitions,
                        }
                    )
            path = split_root / f"trans_{shard_index:02d}.obj"
            save_pickle(path, refreshed_groups)
            output_shards.append(
                {
                    "path": str(path.relative_to(config.feature_root)),
                    "trajectory_count": len(refreshed_groups),
                    "transition_count": sum(
                        len(group["transitions"])
                        for group in refreshed_groups
                    ),
                }
            )
        output_manifest[split] = output_shards
        statistics[split] = {
            "trajectory_count": trajectory_count,
            "source_transition_count": transition_count,
            "kept_transition_count": kept_count,
            "candidate_miss_count": candidate_misses,
            "candidate_recall": (
                (transition_count - candidate_misses) / transition_count
                if transition_count
                else 0.0
            ),
        }

    manifest = {
        "schema_version": 1,
        "source_manifest_sha256": sha256_file(
            source_root / "manifest.json"
        ),
        "emit_metadata_sha256": sha256_file(emit_root / "metadata.json"),
        "emit_top_k": config.emit_top_k,
        "negative_count": config.trans_negative_count,
        "training_candidate_policy": "top_k_plus_soft_positive",
        "validation_candidate_policy": "top_k_only",
        "statistics": statistics,
        "shards": output_manifest,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


@dataclass
class TrainingResources:
    mappings: dict
    cell_candidates: dict
    alignment_train: dict
    alignment_val: dict
    cell_positions_wgs84: torch.Tensor
    cell_positions_metric: torch.Tensor

    @classmethod
    def load(cls, config: TrainingConfig) -> "TrainingResources":
        root = config.training_data_root
        mappings = load_pickle(root / "mappings.obj")
        wgs84 = torch.tensor(
            mappings["cell_id_to_position"], dtype=torch.float32
        )
        metric = torch.from_numpy(_cell_metric_positions(mappings, config.metric_epsg))
        result = cls(
            mappings=mappings,
            cell_candidates=load_pickle(
                root / f"cell_candidates_k{config.candidate_k}.obj"
            ),
            alignment_train=load_pickle(root / "alignment_train.obj"),
            alignment_val=load_pickle(root / "alignment_val.obj"),
            cell_positions_wgs84=wgs84,
            cell_positions_metric=metric,
        )
        normalizer = root / "coordinate_normalizer.json"
        result._coordinate_normalizer = read_json(normalizer) if normalizer.exists() else None
        result.config = config
        return result

    @property
    def coordinate_mean(self) -> tuple[float, float]:
        if self._coordinate_normalizer is not None:
            return tuple(self._coordinate_normalizer["mean"])
        values = self.cell_positions_wgs84.mean(dim=0)
        return float(values[0]), float(values[1])

    @property
    def coordinate_std(self) -> tuple[float, float]:
        if self._coordinate_normalizer is not None:
            return tuple(self._coordinate_normalizer["std"])
        values = self.cell_positions_wgs84.std(dim=0)
        return float(values[0]), float(values[1])


def iter_trajectory_batches(
    alignments: dict,
    cell_candidates: dict,
    *,
    candidate_budget: int,
    seed: int,
    shuffle: bool,
    limit: int | None = None,
) -> Iterator[list[tuple[str, dict]]]:
    items = list(alignments.items())
    if limit is not None:
        items = items[:limit]
    if shuffle:
        random.Random(seed).shuffle(items)
    batch = []
    cost = 0
    for trajectory_id, alignment in items:
        trajectory_cost = sum(
            len(cell_candidates[cell_id]["state_ids"])
            for cell_id in alignment["cell_ids"]
        )
        if batch and cost + trajectory_cost > candidate_budget:
            yield batch
            batch = []
            cost = 0
        batch.append((trajectory_id, alignment))
        cost += trajectory_cost
    if batch:
        yield batch


def build_emit_batch(
    batch: list[tuple[str, dict]],
    resources: TrainingResources,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_time = max(len(alignment["cell_ids"]) for _, alignment in batch)
    max_candidates = max(
        len(resources.cell_candidates[cell_id]["state_ids"])
        for _, alignment in batch
        for cell_id in alignment["cell_ids"]
    )
    max_local = max(
        sum(
            rank <= config.local_road_k
            for rank in resources.cell_candidates[cell_id][
                "state_physical_ranks"
            ]
        )
        for _, alignment in batch
        for cell_id in alignment["cell_ids"]
    )
    cell_ids = torch.zeros((batch_size, max_time), dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    observation_mask = torch.zeros((batch_size, max_time), dtype=torch.bool)
    candidate_ids = torch.zeros(
        (batch_size, max_time, max_candidates), dtype=torch.long
    )
    candidate_distances = torch.zeros(
        (batch_size, max_time, max_candidates), dtype=torch.float32
    )
    candidate_ranks = torch.ones(
        (batch_size, max_time, max_candidates), dtype=torch.long
    )
    candidate_mask = torch.zeros(
        (batch_size, max_time, max_candidates), dtype=torch.bool
    )
    positive_mask = torch.zeros_like(candidate_mask)
    local_ids = torch.zeros((batch_size, max_time, max_local), dtype=torch.long)
    local_distances = torch.zeros(
        (batch_size, max_time, max_local), dtype=torch.float32
    )
    local_mask = torch.zeros((batch_size, max_time, max_local), dtype=torch.bool)

    trajectory_ids = []
    for batch_index, (trajectory_id, alignment) in enumerate(batch):
        trajectory_ids.append(trajectory_id)
        length = len(alignment["cell_ids"])
        lengths[batch_index] = length
        observation_mask[batch_index, :length] = True
        cell_ids[batch_index, :length] = torch.tensor(alignment["cell_ids"])
        for time_index, observation in enumerate(alignment["observations"]):
            candidate = resources.cell_candidates[observation["cell_id"]]
            count = len(candidate["state_ids"])
            ids = torch.tensor(candidate["state_ids"], dtype=torch.long)
            candidate_ids[batch_index, time_index, :count] = ids
            candidate_distances[batch_index, time_index, :count] = torch.tensor(
                candidate["state_distances_m"], dtype=torch.float32
            )
            candidate_ranks[batch_index, time_index, :count] = torch.tensor(
                candidate["state_physical_ranks"], dtype=torch.long
            )
            candidate_mask[batch_index, time_index, :count] = True
            positive = set(observation["positive_state_ids"])
            positive_mask[batch_index, time_index, :count] = torch.tensor(
                [state_id in positive for state_id in candidate["state_ids"]],
                dtype=torch.bool,
            )
            local_indices = [
                index
                for index, rank in enumerate(candidate["state_physical_ranks"])
                if rank <= config.local_road_k
            ]
            local_count = len(local_indices)
            local_ids[batch_index, time_index, :local_count] = ids[local_indices]
            local_distances[
                batch_index, time_index, :local_count
            ] = candidate_distances[
                batch_index, time_index, local_indices
            ]
            local_mask[batch_index, time_index, :local_count] = True
    return {
        "trajectory_ids": trajectory_ids,
        "cell_ids": cell_ids,
        "lengths": lengths,
        "observation_mask": observation_mask,
        "positions": resources.cell_positions_wgs84[cell_ids],
        "candidate_ids": candidate_ids,
        "candidate_distances_m": candidate_distances,
        "candidate_ranks": candidate_ranks,
        "candidate_mask": candidate_mask,
        "positive_mask": positive_mask,
        "local_ids": local_ids,
        "local_distances_m": local_distances,
        "local_mask": local_mask,
    }


def iter_enriched_trans_groups(
    config: TrainingConfig, split: str
) -> Iterator[dict]:
    """Read one shard at a time without retaining unused audit metadata.

    The occurrence-level metadata remains in the immutable shard files. Only
    preparation/statistics consume it; training uses the recorded pairs,
    log-weights and masks, all of which are kept unchanged.
    """
    refreshed = (
        config.feature_root
        / "trans_refreshed"
        / config.experiment_id
    )
    root = (
        refreshed
        if (refreshed / "manifest.json").exists()
        else config.feature_root / "trans_enriched"
    )
    manifest = read_json(root / "manifest.json")
    emitted = 0
    for shard in manifest["shards"][split]:
        groups = load_pickle(config.feature_root / shard["path"])
        for group in groups:
            for record in group["transitions"]:
                record.pop("positive_metadata", None)
        for group in groups:
            if config.limit_trajectories is not None and emitted >= config.limit_trajectories:
                return
            yield group
            emitted += 1


def load_enriched_trans_groups(
    config: TrainingConfig, split: str
) -> list[dict]:
    return list(iter_enriched_trans_groups(config, split))


def pair_features(
    previous_state_ids: torch.Tensor,
    current_state_ids: torch.Tensor,
    previous_cell_ids: torch.Tensor,
    current_cell_ids: torch.Tensor,
    *,
    road_features: dict[str, torch.Tensor],
    cell_positions_metric: torch.Tensor,
) -> torch.Tensor:
    previous_cell_xy = cell_positions_metric[previous_cell_ids]
    current_cell_xy = cell_positions_metric[current_cell_ids]
    cell_delta = current_cell_xy - previous_cell_xy
    cell_distance = torch.linalg.vector_norm(cell_delta, dim=-1).clamp_min(1e-6)
    cell_bearing = torch.atan2(cell_delta[:, 0], cell_delta[:, 1])

    previous_midpoint = road_features["midpoint_xy"][previous_state_ids]
    current_midpoint = road_features["midpoint_xy"][current_state_ids]
    midpoint_distance = torch.linalg.vector_norm(
        current_midpoint - previous_midpoint, dim=-1
    )
    previous_bearing = torch.deg2rad(
        road_features["bearing_degrees"][previous_state_ids]
    )
    current_bearing = torch.deg2rad(
        road_features["bearing_degrees"][current_state_ids]
    )
    bearing_difference = current_bearing - previous_bearing
    same_state = previous_state_ids == current_state_ids
    direct_successor = (
        road_features["edge_v"][previous_state_ids]
        == road_features["edge_u"][current_state_ids]
    )
    same_physical = (
        road_features["physical_id"][previous_state_ids]
        == road_features["physical_id"][current_state_ids]
    )
    return torch.stack(
        (
            torch.log1p(cell_distance),
            torch.sin(cell_bearing),
            torch.cos(cell_bearing),
            torch.log1p(midpoint_distance),
            torch.sin(bearing_difference),
            torch.cos(bearing_difference),
            same_state.float(),
            direct_successor.float(),
            same_physical.float(),
        ),
        dim=-1,
    )
