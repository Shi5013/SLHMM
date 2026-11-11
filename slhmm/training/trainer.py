"""Staged Road/Emit/Trans training loops."""

from __future__ import annotations

from collections import defaultdict
import contextlib
import itertools
import math
from pathlib import Path
import random
import shutil
from typing import Iterator

import dgl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from slhmm.configs.training import TrainingConfig
from slhmm.training.checkpoint import load_checkpoint, save_checkpoint
from slhmm.training.common import (
    load_pickle,
    read_json,
    resolve_device,
    save_pickle,
    seed_everything,
    sha256_file,
    write_json,
)
from slhmm.training.data import (
    PAIR_FEATURE_DIM,
    TrainingResources,
    build_emit_batch,
    iter_trajectory_batches,
    load_enriched_trans_groups,
    pair_features,
    refresh_trans_from_emit_candidates,
)
from slhmm.training.features import load_road_artifacts
from slhmm.training.losses import (
    multi_positive_marginal_loss,
    ragged_multi_positive_marginal_loss,
    ragged_positive_recall_at_k,
    soft_positive_mrr,
    soft_positive_recall_at_k,
)
from slhmm.training.models import SLHMMModel


def build_model(
    config: TrainingConfig,
    resources: TrainingResources,
    road_features: dict[str, torch.Tensor],
) -> SLHMMModel:
    metadata = read_json(config.feature_root / "road" / "metadata.json")
    return SLHMMModel(
        numeric_dim=int(road_features["numeric"].shape[1]),
        highway_count=len(metadata["highway_vocabulary"]),
        relation_names=metadata["relation_names"],
        cell_count=len(resources.mappings["cell_id_to_position"]),
        coordinate_mean=resources.coordinate_mean,
        coordinate_std=resources.coordinate_std,
        pair_feature_dim=PAIR_FEATURE_DIM,
        config=config,
    )


class RoadBatchEncoder:
    """Sample full two-hop relation blocks and preserve global-state indexing."""

    def __init__(
        self,
        graph,
        features: dict[str, torch.Tensor],
        layer_count: int,
        device: torch.device,
    ) -> None:
        self.graph = graph
        self.features = features
        self.device = device
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(layer_count)
        self.node_count = int(graph.num_nodes("road"))

    def encode(
        self, model: SLHMMModel, requested_state_ids: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = [value.detach().cpu().reshape(-1) for value in requested_state_ids]
        seeds = torch.unique(torch.cat(flattened)).to(torch.int64)
        input_nodes, output_nodes, blocks = self.sampler.sample_blocks(
            self.graph, {"road": seeds}
        )
        input_ids = (
            input_nodes["road"]
            if isinstance(input_nodes, dict)
            else input_nodes
        )
        output_ids = (
            output_nodes["road"]
            if isinstance(output_nodes, dict)
            else output_nodes
        )
        blocks = [block.to(self.device) for block in blocks]
        embeddings = model.road_encoder(
            blocks,
            self.features["numeric"][input_ids].to(self.device),
            self.features["highway_id"][input_ids].to(self.device),
        )
        global_to_local = torch.full(
            (self.node_count,), -1, dtype=torch.long
        )
        global_to_local[output_ids] = torch.arange(len(output_ids))
        return embeddings, global_to_local

    def gather(
        self,
        embeddings: torch.Tensor,
        global_to_local: torch.Tensor,
        state_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        local = global_to_local[state_ids.cpu()]
        local = torch.where(mask.cpu(), local, torch.zeros_like(local))
        if (local[mask.cpu()] < 0).any():
            raise RuntimeError("A requested road state is missing from sampled output nodes")
        result = embeddings[local.to(self.device)]
        return result * mask.to(self.device).unsqueeze(-1)


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def _data_hashes(config: TrainingConfig) -> dict[str, str]:
    return {
        "graph": sha256_file(config.graph_path),
        "training_manifest": sha256_file(
            config.training_data_root / "manifest.json"
        ),
        "road_metadata": sha256_file(
            config.feature_root / "road" / "metadata.json"
        ),
    }


def _copy_best(checkpoint_path: Path, stage_root: Path) -> None:
    shutil.copy2(checkpoint_path, stage_root / "best.pt")


class TrainingEngine:
    def __init__(self, config: TrainingConfig) -> None:
        seed_everything(config.seed)
        self.config = config
        self.device = resolve_device(config.device)
        self.amp = bool(config.amp and self.device.type == "cuda")
        self.resources = TrainingResources.load(config)
        self.road_graph, self.road_features = load_road_artifacts(config)
        self.model = build_model(config, self.resources, self.road_features).to(
            self.device
        )
        self.road_batch_encoder = RoadBatchEncoder(
            self.road_graph,
            self.road_features,
            config.road_layers,
            self.device,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.global_step = 0
        self.hashes = _data_hashes(config)

    def initialize_from(self, checkpoint_path: str | Path | None) -> None:
        if checkpoint_path:
            load_checkpoint(
                checkpoint_path,
                model=self.model,
                map_location=self.device,
                restore_rng=False,
            )

    def _task_optimizer(self, stage: str):
        if stage == "joint":
            road_lr = self.config.joint_road_lr
            other_lr = self.config.joint_head_lr
        else:
            road_lr = self.config.task_road_lr
            other_lr = self.config.task_head_lr
        road_parameters = list(self.model.road_encoder.parameters())
        road_ids = {id(parameter) for parameter in road_parameters}
        other_parameters = [
            parameter
            for parameter in self.model.parameters()
            if id(parameter) not in road_ids
        ]
        return torch.optim.AdamW(
            [
                {"params": road_parameters, "lr": road_lr},
                {"params": other_parameters, "lr": other_lr},
            ],
            weight_decay=self.config.weight_decay,
        )

    def _trajectory_context(
        self,
        emit_batch: dict,
        *,
        additional_state_ids: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        requested = [
            emit_batch["local_ids"][emit_batch["local_mask"]],
        ]
        if additional_state_ids:
            requested.extend(additional_state_ids)
        road_embeddings, state_map = self.road_batch_encoder.encode(
            self.model, requested
        )
        local_embeddings = self.road_batch_encoder.gather(
            road_embeddings,
            state_map,
            emit_batch["local_ids"],
            emit_batch["local_mask"],
        )
        cell_ids = emit_batch["cell_ids"].to(self.device)
        cell_embeddings = self.model.cell_encoder(
            cell_ids,
            emit_batch["positions"].to(self.device),
            local_embeddings,
            emit_batch["local_distances_m"].to(self.device),
            emit_batch["local_mask"].to(self.device),
        )
        trajectory_context = self.model.trajectory_encoder(
            cell_embeddings, emit_batch["lengths"]
        )
        return trajectory_context, road_embeddings, state_map, local_embeddings

    def emit_step(self, batch_items: list[tuple[str, dict]]) -> tuple[torch.Tensor, dict]:
        emit_batch = build_emit_batch(batch_items, self.resources, self.config)
        candidate_valid_ids = emit_batch["candidate_ids"][
            emit_batch["candidate_mask"]
        ]
        context, road_embeddings, state_map, _ = self._trajectory_context(
            emit_batch, additional_state_ids=[candidate_valid_ids]
        )
        candidate_embeddings = self.road_batch_encoder.gather(
            road_embeddings,
            state_map,
            emit_batch["candidate_ids"],
            emit_batch["candidate_mask"],
        )
        scores = self.model.emit_head(
            context,
            candidate_embeddings,
            emit_batch["candidate_distances_m"].to(self.device),
            emit_batch["candidate_ranks"].to(self.device),
        )
        candidate_mask = emit_batch["candidate_mask"].to(self.device)
        positive_mask = emit_batch["positive_mask"].to(self.device)
        loss = multi_positive_marginal_loss(
            scores, candidate_mask, positive_mask
        )
        metrics = {
            "loss": float(loss.detach()),
            "recall_at_30": float(
                soft_positive_recall_at_k(
                    scores, candidate_mask, positive_mask, self.config.emit_top_k
                ).detach()
            ),
            "mrr": float(
                soft_positive_mrr(scores, candidate_mask, positive_mask).detach()
            ),
        }
        return loss, metrics

    def _iter_trans_batches(
        self, groups: list[dict], epoch: int, shuffle: bool
    ) -> Iterator[list[dict]]:
        groups = list(groups)
        if shuffle:
            random.Random(self.config.seed + epoch).shuffle(groups)
        batch = []
        cost = 0
        for group in groups:
            group_cost = sum(
                len(record["positive_state_pairs"])
                + len(record["negative_state_pairs"])
                for record in group["transitions"]
            )
            if batch and cost + group_cost > self.config.batch_candidate_budget:
                yield batch
                batch = []
                cost = 0
            batch.append(group)
            cost += group_cost
        if batch:
            yield batch

    def trans_step(self, groups: list[dict]) -> tuple[torch.Tensor, dict]:
        batch_items = [
            (
                group["trajectory_id"],
                self.resources.alignment_train.get(group["trajectory_id"])
                or self.resources.alignment_val[group["trajectory_id"]],
            )
            for group in groups
        ]
        emit_batch = build_emit_batch(batch_items, self.resources, self.config)
        previous_states = []
        current_states = []
        previous_cells = []
        current_cells = []
        previous_observations = []
        current_observations = []
        batch_indices = []
        positive_flags = []
        log_weights = []
        offsets = [0]
        for batch_index, group in enumerate(groups):
            for record in group["transitions"]:
                pairs = list(record["positive_state_pairs"]) + list(
                    record["negative_state_pairs"]
                )
                previous_states.extend(pair[0] for pair in pairs)
                current_states.extend(pair[1] for pair in pairs)
                count = len(pairs)
                previous_cells.extend([record["previous_cell_id"]] * count)
                current_cells.extend([record["current_cell_id"]] * count)
                previous_observations.extend(
                    [record["previous_observation_index"]] * count
                )
                current_observations.extend(
                    [record["current_observation_index"]] * count
                )
                batch_indices.extend([batch_index] * count)
                positive_count = len(record["positive_state_pairs"])
                positive_flags.extend(
                    [True] * positive_count + [False] * (count - positive_count)
                )
                log_weights.extend(
                    record["positive_log_weights"]
                    + [0.0] * (count - positive_count)
                )
                offsets.append(offsets[-1] + count)
        previous_states = torch.tensor(previous_states, dtype=torch.long)
        current_states = torch.tensor(current_states, dtype=torch.long)
        context, road_embeddings, state_map, _ = self._trajectory_context(
            emit_batch,
            additional_state_ids=[previous_states, current_states],
        )
        all_true = torch.ones_like(previous_states, dtype=torch.bool)
        previous_road = self.road_batch_encoder.gather(
            road_embeddings, state_map, previous_states, all_true
        )
        current_road = self.road_batch_encoder.gather(
            road_embeddings, state_map, current_states, all_true
        )
        batch_indices_tensor = torch.tensor(batch_indices, device=self.device)
        previous_observation_tensor = torch.tensor(
            previous_observations, device=self.device
        )
        current_observation_tensor = torch.tensor(
            current_observations, device=self.device
        )
        pair_feature_values = pair_features(
            previous_states,
            current_states,
            torch.tensor(previous_cells, dtype=torch.long),
            torch.tensor(current_cells, dtype=torch.long),
            road_features=self.road_features,
            cell_positions_metric=self.resources.cell_positions_metric,
        ).to(self.device)
        scores = self.model.trans_head(
            previous_road,
            current_road,
            context[batch_indices_tensor, previous_observation_tensor],
            context[batch_indices_tensor, current_observation_tensor],
            pair_feature_values,
        )
        offsets_tensor = torch.tensor(offsets, dtype=torch.long)
        positive_mask = torch.tensor(
            positive_flags, dtype=torch.bool, device=self.device
        )
        positive_log_weights = torch.tensor(
            log_weights, dtype=torch.float32, device=self.device
        )
        loss = ragged_multi_positive_marginal_loss(
            scores,
            offsets_tensor,
            positive_mask,
            positive_log_weights,
        )
        metrics = {
            "loss": float(loss.detach()),
            "recall_at_10": float(
                ragged_positive_recall_at_k(
                    scores, offsets_tensor, positive_mask, 10
                ).detach()
            ),
        }
        return loss, metrics

    def _optimize_loss(
        self,
        loss: torch.Tensor,
        optimizer,
        batch_index: int,
    ) -> bool:
        scaled_loss = loss / self.config.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        should_step = (
            (batch_index + 1) % self.config.gradient_accumulation_steps == 0
        )
        if should_step:
            self._optimizer_step(optimizer)
        return should_step

    def _optimizer_step(self, optimizer) -> None:
        self.scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    @staticmethod
    def _mean_metrics(values: list[dict]) -> dict[str, float]:
        keys = sorted({key for value in values for key in value})
        return {
            key: sum(value[key] for value in values if key in value)
            / sum(key in value for value in values)
            for key in keys
        }

    def pretrain_road(self, resume: str | Path | None = None) -> Path:
        stage = "road"
        stage_root = self.config.checkpoint_root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(self.config.run_root / "tensorboard" / stage))
        optimizer = torch.optim.AdamW(
            itertools.chain(
                self.model.road_encoder.parameters(),
                self.model.road_pretraining_heads.parameters(),
            ),
            lr=self.config.road_pretrain_lr,
            weight_decay=self.config.weight_decay,
        )
        start_epoch = 0
        if resume:
            checkpoint = load_checkpoint(
                resume,
                model=self.model,
                optimizer=optimizer,
                scaler=self.scaler,
                map_location=self.device,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            self.global_step = int(checkpoint["global_step"])
        graph = self.road_graph.to(self.device)
        numeric = self.road_features["numeric"].to(self.device)
        highway = self.road_features["highway_id"].to(self.device)
        link_sources = []
        link_targets = []
        for relation in ("straight", "left", "right", "u_turn"):
            source, target = graph.edges(etype=("road", relation, "road"))
            link_sources.append(source)
            link_targets.append(target)
        link_sources = torch.cat(link_sources)
        link_targets = torch.cat(link_targets)
        best_loss = math.inf
        best_path = stage_root / "best.pt"

        if resume and best_path.exists():
            best_loss = float(torch.load(best_path, map_location="cpu", weights_only=False)["metrics"]["loss"])

        for epoch in range(start_epoch, self.config.road_pretrain_epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            masked = (
                torch.rand(numeric.shape[0], device=self.device)
                < self.config.masked_attribute_rate
            )
            masked_numeric = numeric.clone()
            masked_highway = highway.clone()
            masked_numeric[masked] = 0.0
            masked_highway[masked] = 0
            with _autocast(self.device, self.amp):
                embeddings = self.model.road_encoder(
                    graph, masked_numeric, masked_highway
                )
                decoded_numeric, decoded_highway = (
                    self.model.road_pretraining_heads.reconstruct(embeddings)
                )
                reconstruction = F.mse_loss(
                    decoded_numeric[masked], numeric[masked]
                ) + F.cross_entropy(
                    decoded_highway[masked], highway[masked]
                )
                maximum = min(100_000, len(link_sources))
                selection = torch.randperm(
                    len(link_sources), device=self.device
                )[:maximum]
                source = link_sources[selection]
                target = link_targets[selection]
                negative_target = torch.randint(
                    0, embeddings.shape[0], (maximum,), device=self.device
                )
                positive_score = self.model.road_pretraining_heads.link_score(
                    embeddings[source], embeddings[target]
                )
                negative_score = self.model.road_pretraining_heads.link_score(
                    embeddings[source], embeddings[negative_target]
                )
                link_loss = F.binary_cross_entropy_with_logits(
                    torch.cat((positive_score, negative_score)),
                    torch.cat(
                        (
                            torch.ones_like(positive_score),
                            torch.zeros_like(negative_score),
                        )
                    ),
                )
                loss = reconstruction + link_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_norm
            )
            self.scaler.step(optimizer)
            self.scaler.update()
            self.global_step += 1
            metrics = {
                "loss": float(loss.detach()),
                "reconstruction_loss": float(reconstruction.detach()),
                "link_loss": float(link_loss.detach()),
            }
            for key, value in metrics.items():
                writer.add_scalar(key, value, epoch)
            path = stage_root / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                path,
                stage=stage,
                epoch=epoch,
                global_step=self.global_step,
                model=self.model,
                optimizer=optimizer,
                scheduler=None,
                scaler=self.scaler,
                config=self.config,
                data_hashes=self.hashes,
                metrics=metrics,
            )
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                _copy_best(path, stage_root)
        writer.close()
        return best_path

    def _evaluate_emit(self, split: str) -> dict[str, float]:
        alignments = (
            self.resources.alignment_train
            if split == "train"
            else self.resources.alignment_val
        )
        metrics = []
        self.model.eval()
        with torch.no_grad():
            for batch_items in iter_trajectory_batches(
                alignments,
                self.resources.cell_candidates,
                candidate_budget=self.config.batch_candidate_budget,
                seed=self.config.seed,
                shuffle=False,
                limit=self.config.limit_trajectories,
            ):
                with _autocast(self.device, self.amp):
                    _, batch_metrics = self.emit_step(batch_items)
                metrics.append(batch_metrics)
        return self._mean_metrics(metrics)

    def train_emit(
        self,
        *,
        initialize: str | Path | None,
        resume: str | Path | None = None,
    ) -> Path:
        stage = "emit"
        self.initialize_from(initialize)
        return self._train_task_stage(stage, resume=resume)

    def train_trans(
        self,
        *,
        initialize: str | Path | None,
        resume: str | Path | None = None,
    ) -> Path:
        stage = "trans"
        self.initialize_from(initialize)
        return self._train_task_stage(stage, resume=resume)

    def _train_task_stage(
        self, stage: str, *, resume: str | Path | None
    ) -> Path:
        stage_root = self.config.checkpoint_root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(self.config.run_root / "tensorboard" / stage))
        optimizer = self._task_optimizer(stage)
        optimizer.zero_grad(set_to_none=True)
        start_epoch = 0
        if resume:
            checkpoint = load_checkpoint(
                resume,
                model=self.model,
                optimizer=optimizer,
                scaler=self.scaler,
                map_location=self.device,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            self.global_step = int(checkpoint["global_step"])
        epochs = (
            self.config.emit_epochs
            if stage == "emit"
            else self.config.trans_epochs
        )
        trans_train = (
            load_enriched_trans_groups(self.config, "train")
            if stage == "trans"
            else None
        )
        trans_val = (
            load_enriched_trans_groups(self.config, "val")
            if stage == "trans"
            else None
        )
        best_loss = math.inf
        best_path = stage_root / "best.pt"
        if resume and best_path.exists():
            best_loss = float(torch.load(best_path, map_location="cpu", weights_only=False)["metrics"]["val_loss"])
        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            self.model.train()
            training_metrics = []
            if stage == "emit":
                iterator = iter_trajectory_batches(
                    self.resources.alignment_train,
                    self.resources.cell_candidates,
                    candidate_budget=self.config.batch_candidate_budget,
                    seed=self.config.seed + epoch,
                    shuffle=True,
                    limit=self.config.limit_trajectories,
                )
                iterator = list(iterator)
                if hasattr(self, "configure_training_normalizers"):
                    self.configure_training_normalizers(stage, len(iterator))
                for batch_index, batch_items in enumerate(iterator):
                    with _autocast(self.device, self.amp):
                        loss, metrics = self.emit_step(batch_items)
                    self._optimize_loss(loss, optimizer, batch_index)
                    training_metrics.append(metrics)
            else:
                iterator = self._iter_trans_batches(
                    trans_train, epoch, shuffle=True
                )
                iterator = list(iterator)
                if hasattr(self, "configure_training_normalizers"):
                    self.configure_training_normalizers(stage, len(iterator), trans_train)
                for batch_index, groups in enumerate(iterator):
                    with _autocast(self.device, self.amp):
                        loss, metrics = self.trans_step(groups)
                    self._optimize_loss(loss, optimizer, batch_index)
                    training_metrics.append(metrics)
            completed_batches = len(training_metrics)
            if (
                completed_batches
                and completed_batches
                % self.config.gradient_accumulation_steps
            ):
                self._optimizer_step(optimizer)
            train_metrics = self._mean_metrics(training_metrics)
            if stage == "emit":
                val_metrics = self._evaluate_emit("val")
            else:
                val_values = []
                self.model.eval()
                with torch.no_grad():
                    for groups in self._iter_trans_batches(
                        trans_val, epoch, shuffle=False
                    ):
                        with _autocast(self.device, self.amp):
                            _, metrics = self.trans_step(groups)
                        val_values.append(metrics)
                val_metrics = self._mean_metrics(val_values)
            metrics = {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            for key, value in metrics.items():
                writer.add_scalar(key, value, epoch)
            path = stage_root / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                path,
                stage=stage,
                epoch=epoch,
                global_step=self.global_step,
                model=self.model,
                optimizer=optimizer,
                scheduler=None,
                scaler=self.scaler,
                config=self.config,
                data_hashes=self.hashes,
                metrics=metrics,
            )
            if metrics["val_loss"] < best_loss:
                best_loss = metrics["val_loss"]
                _copy_best(path, stage_root)
            write_json(self.config.run_root / "epochs" / f"{stage}_{epoch:03d}.json", metrics)
            print({"stage": stage, "epoch": epoch, **metrics}, flush=True)
        writer.close()
        return best_path

    def refresh_candidates(
        self, checkpoint_path: str | Path
    ) -> dict[str, Path]:
        self.initialize_from(checkpoint_path)
        self.model.eval()
        output_root = (
            self.config.feature_root
            / "emit_candidates"
            / self.config.experiment_id
        )
        output_root.mkdir(parents=True, exist_ok=True)
        paths = {}
        with torch.no_grad():
            for split, alignments in (
                ("train", self.resources.alignment_train),
                ("val", self.resources.alignment_val),
            ):
                output = {}
                for batch_items in iter_trajectory_batches(
                    alignments,
                    self.resources.cell_candidates,
                    candidate_budget=self.config.batch_candidate_budget,
                    seed=self.config.seed,
                    shuffle=False,
                    limit=self.config.limit_trajectories,
                ):
                    emit_batch = build_emit_batch(
                        batch_items, self.resources, self.config
                    )
                    candidate_ids = emit_batch["candidate_ids"]
                    with _autocast(self.device, self.amp):
                        context, embeddings, state_map, _ = (
                            self._trajectory_context(
                                emit_batch,
                                additional_state_ids=[
                                    candidate_ids[emit_batch["candidate_mask"]]
                                ],
                            )
                        )
                        candidate_embeddings = self.road_batch_encoder.gather(
                            embeddings,
                            state_map,
                            candidate_ids,
                            emit_batch["candidate_mask"],
                        )
                        scores = self.model.emit_head(
                            context,
                            candidate_embeddings,
                            emit_batch["candidate_distances_m"].to(self.device),
                            emit_batch["candidate_ranks"].to(self.device),
                        )
                        scores = scores.masked_fill(
                            ~emit_batch["candidate_mask"].to(self.device),
                            float("-inf"),
                        )
                    for batch_index, (trajectory_id, alignment) in enumerate(
                        batch_items
                    ):
                        length = len(alignment["cell_ids"])
                        layers = []
                        for time_index in range(length):
                            valid_count = int(
                                emit_batch["candidate_mask"][
                                    batch_index, time_index
                                ].sum()
                            )
                            take = min(self.config.emit_top_k, valid_count)
                            top = scores[
                                batch_index, time_index, :valid_count
                            ].topk(take).indices.cpu()
                            layers.append(
                                candidate_ids[
                                    batch_index, time_index, :valid_count
                                ][top].tolist()
                            )
                        output[trajectory_id] = layers
                path = output_root / f"{split}.obj"
                save_pickle(path, output)
                paths[split] = path
        write_json(
            output_root / "metadata.json",
            {
                "schema_version": 1,
                "checkpoint": str(checkpoint_path),
                "top_k": self.config.emit_top_k,
                "training_forces_soft_positives": True,
                "validation_forces_soft_positives": False,
            },
        )
        return paths

    def refresh_trans_candidates(self) -> dict:
        return refresh_trans_from_emit_candidates(self.config)

    def finetune(
        self,
        *,
        initialize: str | Path | None,
        resume: str | Path | None = None,
    ) -> Path:
        """Alternate Emit and Trans batches, selecting epochs by full Viterbi."""

        from slhmm.training.evaluation import FullViterbiEvaluator

        stage = "joint"
        self.initialize_from(initialize)
        stage_root = self.config.checkpoint_root / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(self.config.run_root / "tensorboard" / stage))
        optimizer = self._task_optimizer(stage)
        optimizer.zero_grad(set_to_none=True)
        start_epoch = 0
        if resume:
            checkpoint = load_checkpoint(
                resume,
                model=self.model,
                optimizer=optimizer,
                scaler=self.scaler,
                map_location=self.device,
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            self.global_step = int(checkpoint["global_step"])

        trans_train = load_enriched_trans_groups(self.config, "train")
        trans_val = load_enriched_trans_groups(self.config, "val")
        evaluator = FullViterbiEvaluator(self)
        best_composite = -math.inf
        best_path = stage_root / "best.pt"
        if resume and best_path.exists():
            best_composite = float(torch.load(best_path, map_location="cpu", weights_only=False)["metrics"]["viterbi_composite"])
        patience = 0

        for epoch in range(start_epoch, self.config.joint_epochs):
            self.model.train()
            emit_iterator = iter_trajectory_batches(
                self.resources.alignment_train,
                self.resources.cell_candidates,
                candidate_budget=self.config.batch_candidate_budget,
                seed=self.config.seed + epoch,
                shuffle=True,
                limit=self.config.limit_trajectories,
            )
            trans_iterator = self._iter_trans_batches(
                trans_train, epoch, shuffle=True
            )
            emit_iterator = list(emit_iterator)
            trans_iterator = list(trans_iterator)
            if hasattr(self, "configure_training_normalizers"):
                self.configure_training_normalizers("emit", len(emit_iterator))
                self.configure_training_normalizers("trans", len(trans_iterator), trans_train)
            task_metrics = {"emit": [], "trans": []}
            update_index = 0
            for emit_items, trans_groups in itertools.zip_longest(
                emit_iterator, trans_iterator
            ):
                if emit_items is not None:
                    with _autocast(self.device, self.amp):
                        loss, metrics = self.emit_step(emit_items)
                    self._optimize_loss(loss, optimizer, update_index)
                    task_metrics["emit"].append(metrics)
                    update_index += 1
                if trans_groups is not None:
                    with _autocast(self.device, self.amp):
                        loss, metrics = self.trans_step(trans_groups)
                    self._optimize_loss(loss, optimizer, update_index)
                    task_metrics["trans"].append(metrics)
                    update_index += 1
            if update_index and update_index % self.config.gradient_accumulation_steps:
                self._optimizer_step(optimizer)

            emit_val = self._evaluate_emit("val")
            trans_values = []
            self.model.eval()
            with torch.no_grad():
                for groups in self._iter_trans_batches(
                    trans_val, epoch, shuffle=False
                ):
                    with _autocast(self.device, self.amp):
                        _, values = self.trans_step(groups)
                    trans_values.append(values)
            trans_val_metrics = self._mean_metrics(trans_values)
            viterbi_path = (
                self.config.run_root
                / "viterbi"
                / f"joint_epoch_{epoch:03d}.json"
            )
            viterbi_report = evaluator.evaluate(viterbi_path)
            viterbi = viterbi_report["metrics"]
            metrics = {
                **{
                    f"train_emit_{key}": value
                    for key, value in self._mean_metrics(
                        task_metrics["emit"]
                    ).items()
                },
                **{
                    f"train_trans_{key}": value
                    for key, value in self._mean_metrics(
                        task_metrics["trans"]
                    ).items()
                },
                **{f"val_emit_{key}": value for key, value in emit_val.items()},
                **{
                    f"val_trans_{key}": value
                    for key, value in trans_val_metrics.items()
                },
                **{f"viterbi_{key}": value for key, value in viterbi.items()},
            }
            for key, value in metrics.items():
                if isinstance(value, (int, float, bool)):
                    writer.add_scalar(key, float(value), epoch)
            path = stage_root / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                path,
                stage=stage,
                epoch=epoch,
                global_step=self.global_step,
                model=self.model,
                optimizer=optimizer,
                scheduler=None,
                scaler=self.scaler,
                config=self.config,
                data_hashes=self.hashes,
                metrics=metrics,
            )
            if (
                viterbi["checkpoint_eligible"]
                and viterbi["composite"] > best_composite
            ):
                best_composite = float(viterbi["composite"])
                _copy_best(path, stage_root)
                patience = 0
            else:
                patience += 1
            if patience >= self.config.early_stopping_patience:
                break
        writer.close()
        if not best_path.exists():
            raise RuntimeError(
                "No joint checkpoint met the minimum topology continuity "
                f"{self.config.minimum_topology_continuity:.1%}"
            )
        return best_path


def latest_stage_checkpoint(config: TrainingConfig, stage: str) -> Path:
    paths = sorted((config.checkpoint_root / stage).glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {stage} checkpoints below {config.checkpoint_root}")
    return paths[-1]
