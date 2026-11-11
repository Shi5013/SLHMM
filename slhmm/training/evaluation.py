"""Full-path Viterbi validation and route-level model selection metrics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import osmnx as ox
from shapely.geometry import MultiLineString
import torch

from slhmm.matching.osm_hmm import EdgeCandidate, OSMEdgeHMMMatcher
from slhmm.training.checkpoint import load_checkpoint
from slhmm.training.common import load_pickle, write_json
from slhmm.training.data import build_emit_batch, pair_features


PAPER_CORRIDOR_TOLERANCE_M = 50.0


def _lcs_length(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    row = [0] * (len(right) + 1)
    for left_value in left:
        previous = 0
        for index, right_value in enumerate(right, start=1):
            saved = row[index]
            if left_value == right_value:
                row[index] = previous + 1
            else:
                row[index] = max(row[index], row[index - 1])
            previous = saved
    return row[-1]


def _sequence_lcs_f1(predicted: list[int], truth: list[int]) -> float:
    if not predicted or not truth:
        return 0.0
    overlap = _lcs_length(predicted, truth)
    precision = overlap / len(predicted)
    recall = overlap / len(truth)
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _weighted_lcs_length(
    predicted: list[int],
    truth: list[int],
    state_lengths_m: list[float],
) -> float:
    """Return the ordered common directed-edge length in metres.

    DMM/LHMM define a correctly matched route by its physical length but do
    not specify how repeated roads are paired. A length-weighted LCS makes
    that pairing deterministic, preserves traversal order, and counts a
    repeated directed edge at most once per occurrence in either route.
    """
    if len(predicted) < len(truth):
        predicted, truth = truth, predicted
    row = [0.0] * (len(truth) + 1)
    for predicted_state in predicted:
        previous = 0.0
        for index, truth_state in enumerate(truth, start=1):
            saved = row[index]
            if predicted_state == truth_state:
                row[index] = previous + float(state_lengths_m[predicted_state])
            else:
                row[index] = max(row[index], row[index - 1])
            previous = saved
    return row[-1]


def _road_length_metrics(
    predicted: list[int],
    truth: list[int],
    state_lengths_m: list[float],
) -> dict[str, float]:
    """Compute DMM/LHMM road-length precision, recall, F1, and RMF."""
    predicted_length = float(sum(state_lengths_m[state] for state in predicted))
    truth_length = float(sum(state_lengths_m[state] for state in truth))
    correct_length = _weighted_lcs_length(predicted, truth, state_lengths_m)
    missing_length = max(0.0, truth_length - correct_length)
    redundant_length = max(0.0, predicted_length - correct_length)
    precision = correct_length / predicted_length if predicted_length > 0 else 0.0
    recall = correct_length / truth_length if truth_length > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    rmf = (
        (missing_length + redundant_length) / truth_length
        if truth_length > 0
        else 0.0
    )
    return {
        "road_length_precision": precision,
        "road_length_recall": recall,
        "road_length_f1": f1,
        "rmf": rmf,
        "correctly_matched_length_m": correct_length,
        "predicted_path_length_m": predicted_length,
        "ground_truth_path_length_m": truth_length,
        "missing_length_m": missing_length,
        "redundant_length_m": redundant_length,
    }


def _geometry_length_metrics(
    predicted_lines: list,
    truth_lines: list,
    tolerance_m: float,
) -> dict[str, float]:
    """Compute buffered geometric length precision/recall and F1."""
    empty = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "truth_length_m": 0.0,
        "truth_uncovered_length_m": 0.0,
    }
    if not predicted_lines or not truth_lines:
        if truth_lines:
            truth = MultiLineString(truth_lines)
            empty["truth_length_m"] = float(truth.length)
            empty["truth_uncovered_length_m"] = float(truth.length)
        return empty
    predicted = MultiLineString(predicted_lines)
    truth = MultiLineString(truth_lines)
    if predicted.length <= 0 or truth.length <= 0:
        return empty
    covered_predicted = min(
        float(predicted.length),
        float(predicted.intersection(truth.buffer(tolerance_m)).length),
    )
    covered_truth = min(
        float(truth.length),
        float(truth.intersection(predicted.buffer(tolerance_m)).length),
    )
    precision = covered_predicted / float(predicted.length)
    recall = covered_truth / float(truth.length)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
        "truth_length_m": float(truth.length),
        "truth_uncovered_length_m": max(0.0, float(truth.length) - covered_truth),
    }


def _geometry_f1(
    predicted_lines: list,
    truth_lines: list,
    tolerance_m: float,
) -> float:
    return _geometry_length_metrics(
        predicted_lines, truth_lines, tolerance_m
    )["f1"]


class FullViterbiEvaluator:
    """Decode every validation trajectory with Emit Top-K and reachable Trans."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.config = engine.config
        graph = ox.load_graphml(self.config.graph_path)
        self.matcher = OSMEdgeHMMMatcher(graph)
        self.state_edges = [
            tuple(edge)
            for edge in engine.resources.mappings["state_id_to_edge"]
        ]
        self.edge_to_state = {
            edge: index for index, edge in enumerate(self.state_edges)
        }
        self.state_lengths_m = [
            self.matcher.edge_lengths[self.matcher._edge_to_index[edge]]
            for edge in self.state_edges
        ]
        candidate_path = (
            self.config.feature_root
            / "emit_candidates"
            / self.config.experiment_id
            / "val.obj"
        )
        if not candidate_path.exists():
            raise FileNotFoundError(
                "Emit candidates are missing; run refresh-candidates first: "
                f"{candidate_path}"
            )
        self.candidates = load_pickle(candidate_path)

    def _failed_decode_result(
        self,
        trajectory_id: str,
        alignment: dict,
        reason: str,
    ) -> dict:
        """Represent a failed decode as a fully missed ground-truth route."""
        truth_states = [int(value) for value in alignment["truth_state_ids"]]
        road_metrics = _road_length_metrics([], truth_states, self.state_lengths_m)
        truth_lines = self.matcher.route_geometries_metric(
            [self.state_edges[state_id] for state_id in truth_states]
        )
        corridor_metrics = _geometry_length_metrics(
            [], truth_lines, PAPER_CORRIDOR_TOLERANCE_M
        )
        return {
            "trajectory_id": trajectory_id,
            "success": False,
            "reason": reason,
            "directed_edge_lcs_f1": 0.0,
            "geometry_length_f1": 0.0,
            **road_metrics,
            "cmf_50": 1.0 if corridor_metrics["truth_length_m"] > 0 else 0.0,
            "corridor_uncovered_length_50m": corridor_metrics[
                "truth_uncovered_length_m"
            ],
            "corridor_truth_length_m": corridor_metrics["truth_length_m"],
            "topology_continuity": 0.0,
            "composite": 0.0,
        }

    def _candidate_tensor(self, trajectory_id: str, alignment: dict):
        layers = self.candidates[trajectory_id]
        maximum = max(len(layer) for layer in layers)
        state_ids = torch.zeros((1, len(layers), maximum), dtype=torch.long)
        distances = torch.zeros_like(state_ids, dtype=torch.float32)
        ranks = torch.ones_like(state_ids)
        mask = torch.zeros_like(state_ids, dtype=torch.bool)
        for time_index, (cell_id, layer) in enumerate(
            zip(alignment["cell_ids"], layers)
        ):
            candidate = self.engine.resources.cell_candidates[cell_id]
            lookup = {
                int(state_id): (float(distance), int(rank))
                for state_id, distance, rank in zip(
                    candidate["state_ids"],
                    candidate["state_distances_m"],
                    candidate["state_physical_ranks"],
                )
            }
            for candidate_index, state_id in enumerate(layer):
                if int(state_id) not in lookup:
                    raise ValueError(
                        f"Refreshed state {state_id} is absent from cell {cell_id}"
                    )
                state_ids[0, time_index, candidate_index] = int(state_id)
                distances[0, time_index, candidate_index] = lookup[int(state_id)][0]
                ranks[0, time_index, candidate_index] = lookup[int(state_id)][1]
                mask[0, time_index, candidate_index] = True
        return state_ids, distances, ranks, mask

    def _reachable_matrix(
        self,
        previous_states: torch.Tensor,
        current_states: torch.Tensor,
    ) -> torch.Tensor:
        previous_edges = [
            self.state_edges[int(state_id)] for state_id in previous_states
        ]
        current_edges = [
            self.state_edges[int(state_id)] for state_id in current_states
        ]
        # The legacy mask asks only whether a directed path exists. Condensing
        # SCCs answers the exact same question without thousands of unnecessary
        # full-city Dijkstra searches for one-way unreachable targets.
        if not hasattr(self, "_directed_reachability"):
            from slhmm.routing.reachability import DirectedReachability
            self._directed_reachability = DirectedReachability(self.matcher.graph)
        return torch.tensor(
            [
                [
                    previous == current
                    or self._directed_reachability(previous[1], current[0])
                    for current in current_edges
                ]
                for previous in previous_edges
            ],
            dtype=torch.bool,
            device=self.engine.device,
        )

    def decode(self, trajectory_id: str, alignment: dict) -> dict:
        engine = self.engine
        emit_batch = build_emit_batch(
            [(trajectory_id, alignment)], engine.resources, self.config
        )
        state_ids, distances, ranks, mask = self._candidate_tensor(
            trajectory_id, alignment
        )
        requested = state_ids[mask]
        context, road_embeddings, state_map, _ = engine._trajectory_context(
            emit_batch, additional_state_ids=[requested]
        )
        candidate_embeddings = engine.road_batch_encoder.gather(
            road_embeddings, state_map, state_ids, mask
        )
        emit_scores = engine.model.emit_head(
            context,
            candidate_embeddings,
            distances.to(engine.device),
            ranks.to(engine.device),
        )[0]
        emit_scores = emit_scores.masked_fill(
            ~mask[0].to(engine.device), float("-inf")
        )

        length = len(alignment["cell_ids"])
        scores = emit_scores[0, : len(self.candidates[trajectory_id][0])]
        backpointers = []
        for time_index in range(1, length):
            previous_layer = state_ids[0, time_index - 1][
                mask[0, time_index - 1]
            ]
            current_layer = state_ids[0, time_index][mask[0, time_index]]
            previous_count = len(previous_layer)
            current_count = len(current_layer)
            previous_grid = (
                previous_layer[:, None].expand(previous_count, current_count).reshape(-1)
            )
            current_grid = (
                current_layer[None, :].expand(previous_count, current_count).reshape(-1)
            )
            pair_mask = self._reachable_matrix(
                previous_layer, current_layer
            ).reshape(-1)
            all_true = torch.ones_like(previous_grid, dtype=torch.bool)
            previous_road = engine.road_batch_encoder.gather(
                road_embeddings, state_map, previous_grid, all_true
            )
            current_road = engine.road_batch_encoder.gather(
                road_embeddings, state_map, current_grid, all_true
            )
            previous_cell = torch.full_like(
                previous_grid, int(alignment["cell_ids"][time_index - 1])
            )
            current_cell = torch.full_like(
                current_grid, int(alignment["cell_ids"][time_index])
            )
            features = pair_features(
                previous_grid,
                current_grid,
                previous_cell,
                current_cell,
                road_features=engine.road_features,
                cell_positions_metric=engine.resources.cell_positions_metric,
            ).to(engine.device)
            transition = engine.model.trans_head(
                previous_road,
                current_road,
                context[0, time_index - 1]
                .expand(previous_count * current_count, -1),
                context[0, time_index].expand(previous_count * current_count, -1),
                features,
            ).reshape(previous_count, current_count)
            transition = transition.masked_fill(~pair_mask.reshape_as(transition), float("-inf"))
            values = (
                scores[:, None]
                + self.config.viterbi_trans_weight * transition
                + emit_scores[time_index, :current_count][None, :]
            )
            scores, pointers = values.max(dim=0)
            if not torch.isfinite(scores).any():
                return self._failed_decode_result(
                    trajectory_id,
                    alignment,
                    f"no_reachable_path_at_observation_{time_index}",
                )
            backpointers.append(pointers.cpu())

        selected_indices = [int(scores.argmax())]
        for pointers in reversed(backpointers):
            selected_indices.append(int(pointers[selected_indices[-1]]))
        selected_indices.reverse()
        selected_states = [
            int(self.candidates[trajectory_id][time_index][candidate_index])
            for time_index, candidate_index in enumerate(selected_indices)
        ]
        selected_candidates = []
        for state_id in selected_states:
            edge = self.state_edges[state_id]
            edge_index = self.matcher._edge_to_index[edge]
            selected_candidates.append(
                EdgeCandidate(
                    edge=edge,
                    distance_m=0.0,
                    offset_m=0.0,
                    edge_length_m=self.matcher.edge_lengths[edge_index],
                )
            )
        expanded_edges = self.matcher._expand_candidates(selected_candidates)
        if not expanded_edges:
            return self._failed_decode_result(
                trajectory_id, alignment, "route_expansion_failed"
            )
        expanded_states = [self.edge_to_state[tuple(edge)] for edge in expanded_edges]
        truth_states = [int(value) for value in alignment["truth_state_ids"]]
        predicted_lines = self.matcher.route_geometries_metric(expanded_edges)
        truth_lines = self.matcher.route_geometries_metric(
            [self.state_edges[state_id] for state_id in truth_states]
        )
        continuity = sum(
            left[1] == right[0]
            for left, right in zip(expanded_edges, expanded_edges[1:])
        ) / max(1, len(expanded_edges) - 1)
        if len(expanded_edges) == 1:
            continuity = 1.0
        lcs_f1 = _sequence_lcs_f1(expanded_states, truth_states)
        geometry_f1 = _geometry_f1(
            predicted_lines,
            truth_lines,
            self.config.geometry_tolerance_m,
        )
        road_metrics = _road_length_metrics(
            expanded_states, truth_states, self.state_lengths_m
        )
        corridor_metrics = _geometry_length_metrics(
            predicted_lines,
            truth_lines,
            PAPER_CORRIDOR_TOLERANCE_M,
        )
        return {
            "trajectory_id": trajectory_id,
            "success": True,
            "selected_state_ids": selected_states,
            "expanded_state_ids": expanded_states,
            "directed_edge_lcs_f1": lcs_f1,
            "geometry_length_f1": geometry_f1,
            **road_metrics,
            "cmf_50": 1.0 - corridor_metrics["recall"],
            "corridor_uncovered_length_50m": corridor_metrics[
                "truth_uncovered_length_m"
            ],
            "corridor_truth_length_m": corridor_metrics["truth_length_m"],
            "topology_continuity": continuity,
            "composite": 0.5 * lcs_f1 + 0.5 * geometry_f1,
        }

    def evaluate(
        self,
        output_path: str | Path | None = None,
        *,
        visualization_dir: str | Path | None = None,
        visualization_limit: int | None = None,
    ) -> dict:
        if visualization_limit is not None and visualization_limit < 0:
            raise ValueError("visualization_limit must be non-negative")
        if visualization_dir is not None:
            from slhmm.evaluation.prediction_visualization import (
                visualize_prediction,
            )

            visualization_dir = Path(visualization_dir)
            visualization_dir.mkdir(parents=True, exist_ok=True)
        self.engine.model.eval()
        results = []
        visualization_count = 0
        with torch.no_grad():
            for index, (trajectory_id, alignment) in enumerate(
                self.engine.resources.alignment_val.items()
            ):
                if self.config.limit_trajectories is not None and index >= self.config.limit_trajectories:
                    break
                if trajectory_id not in self.candidates:
                    continue
                self.matcher._shortest_path_cache.clear()
                self.matcher._shortest_distance_cache.clear()
                result = self.decode(trajectory_id, alignment)
                results.append(result)
                should_visualize = (
                    visualization_dir is not None
                    and (
                        visualization_limit is None
                        or visualization_count < visualization_limit
                    )
                )
                if should_visualize:
                    try:
                        path = visualize_prediction(
                            trajectory_id=trajectory_id,
                            alignment=alignment,
                            result=result,
                            cell_positions_metric=(
                                self.engine.resources.cell_positions_metric
                            ),
                            state_edges=self.state_edges,
                            matcher=self.matcher,
                            output_dir=visualization_dir,
                        )
                        result["visualization_path"] = str(path)
                        visualization_count += 1
                    except Exception as error:
                        result["visualization_error"] = (
                            f"{type(error).__name__}: {error}"
                        )
        successful = [result for result in results if result.get("success")]
        metrics = {
            "trajectory_count": len(results),
            "success_count": len(successful),
            "failure_count": len(results) - len(successful),
            "decode_success_rate": (
                len(successful) / len(results) if results else 0.0
            ),
        }
        for name in (
            "directed_edge_lcs_f1",
            "geometry_length_f1",
            "road_length_precision",
            "road_length_recall",
            "road_length_f1",
            "topology_continuity",
            "composite",
        ):
            metrics[name] = (
                float(
                    np.mean(
                        [
                            result.get(name, 0.0)
                            for result in results
                        ]
                    )
                )
                if results
                else 0.0
            )
        correct_length = sum(
            result.get("correctly_matched_length_m", 0.0) for result in results
        )
        predicted_length = sum(
            result.get("predicted_path_length_m", 0.0) for result in results
        )
        truth_length = sum(
            result.get("ground_truth_path_length_m", 0.0) for result in results
        )
        missing_length = sum(
            result.get("missing_length_m", 0.0) for result in results
        )
        redundant_length = sum(
            result.get("redundant_length_m", 0.0) for result in results
        )
        corridor_uncovered = sum(
            result.get("corridor_uncovered_length_50m", 0.0)
            for result in results
        )
        corridor_truth = sum(
            result.get("corridor_truth_length_m", 0.0) for result in results
        )
        metrics.update(
            {
                "road_length_precision_micro": (
                    correct_length / predicted_length
                    if predicted_length > 0
                    else 0.0
                ),
                "road_length_recall_micro": (
                    correct_length / truth_length if truth_length > 0 else 0.0
                ),
                "rmf": (
                    (missing_length + redundant_length) / truth_length
                    if truth_length > 0
                    else 0.0
                ),
                "cmf_50": (
                    corridor_uncovered / corridor_truth
                    if corridor_truth > 0
                    else 0.0
                ),
                "paper_metric_corridor_radius_m": PAPER_CORRIDOR_TOLERANCE_M,
            }
        )
        micro_precision = metrics["road_length_precision_micro"]
        micro_recall = metrics["road_length_recall_micro"]
        metrics["road_length_f1_micro"] = (
            2.0
            * micro_precision
            * micro_recall
            / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        metrics["checkpoint_eligible"] = (
            metrics["topology_continuity"]
            >= self.config.minimum_topology_continuity
        )
        if visualization_dir is not None:
            metrics["visualization_count"] = visualization_count
        report = {"metrics": metrics, "trajectories": results}
        if visualization_dir is not None:
            report["visualization"] = {
                "directory": str(visualization_dir),
                "image_count": visualization_count,
                "limit": visualization_limit,
            }
        if output_path is not None:
            write_json(output_path, report)
        return report


def evaluate_checkpoint(
    engine,
    checkpoint_path: str | Path,
    output_path=None,
    *,
    visualization_dir: str | Path | None = None,
    visualization_limit: int | None = None,
) -> dict:
    load_checkpoint(
        checkpoint_path,
        model=engine.model,
        map_location=engine.device,
        restore_rng=False,
    )
    return FullViterbiEvaluator(engine).evaluate(
        output_path,
        visualization_dir=visualization_dir,
        visualization_limit=visualization_limit,
    )
