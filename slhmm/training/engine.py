"""Path-aware SLHMM engine, preserving the original stage implementations."""
from __future__ import annotations

from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from slhmm.training.trainer import TrainingEngine, _autocast, _copy_best
from slhmm.training.common import load_pickle, save_pickle, read_json, write_json, seed_everything
from slhmm.training.checkpoint import load_checkpoint, save_checkpoint
from slhmm.training.data import pair_features, iter_trajectory_batches, load_enriched_trans_groups
from slhmm.training.components import IndependentDirectionalEncoder, ReliabilityGate, DecoupledEmitHead, PhysicalCosts
from slhmm.training.extended_data import ObservationData
from slhmm.training.structured import safe_logsumexp, lattice_crf_loss
from slhmm.routing.path_router import GeneralizedRouter
from slhmm.training.lattice import LatticeBuilder, score_lattice
from slhmm.training.optimization import weighted_reduction, observation_weight_totals, transition_weight_total


def weighted_mean(values, weights):
    return weighted_reduction(values, weights)


class EnhancedEngine(TrainingEngine):
    def __init__(self, config):
        super().__init__(config)
        self.observation_data = ObservationData(self.resources, config)
        if config.use_h3:
            from slhmm.training.h3_prior import H3Prior
            root = config.feature_root / "h3"
            ids = torch.load(root / "cell_h3_ids.pt", map_location="cpu", weights_only=True)
            metadata = read_json(root / "metadata.json")
            self.model.cell_encoder.h3_prior = H3Prior(ids, metadata["vocab_sizes"], config)
        if config.emit_mode == "decoupled":
            self.model.trajectory_encoder = IndependentDirectionalEncoder(config.road_hidden_dim, config)
            self.model.emit_head = DecoupledEmitHead(config.road_hidden_dim)
        if not config.use_trajectory_context:
            self.model.trajectory_encoder = nn.Identity()
        if config.use_reliability_gate:
            self.model.reliability_gate = ReliabilityGate(config.road_hidden_dim)
        if config.variant != "baseline":
            self.model.physical_costs = PhysicalCosts(config)
        self.model.to(self.device)
        self.router = GeneralizedRouter(self.road_features, config, self.hashes["graph"])
        self.lattice_builder = LatticeBuilder(self)
        statistics = read_json(config.feature_root / "training_statistics.json")
        self.delta_max = int(statistics["delta_max"])
        self._evaluator = None
        self.stage = None
        self.epoch = 0
        self._last_progress = 0.
        alignments = list(self.resources.alignment_train.values())
        if config.limit_trajectories is not None:
            alignments = alignments[:config.limit_trajectories]
        self._training_weight_totals = observation_weight_totals(alignments)
        self._loss_normalizers = {}

    def configure_training_normalizers(self, stage, batch_count, trans_groups=None):
        if trans_groups is not None:
            self._training_weight_totals["trans"] = transition_weight_total(trans_groups, self.resources.alignment_train)
        tasks = ("emit", "gate") if stage == "emit" else (("trans",) if stage == "trans" else ("emit", "gate", "trans", "sequence"))
        for task in tasks:
            self._loss_normalizers[task] = self._training_weight_totals[task] / max(1, batch_count)

    def reduce_loss(self, values, weights, task):
        normalizer = self._loss_normalizers.get(task) if self.model.training else None
        return weighted_reduction(values, weights, normalizer)

    def initialize_from(self, checkpoint_path):
        if not checkpoint_path:
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint["data_hashes"]["graph"] != self.hashes["graph"]:
            raise ValueError("Checkpoint graph does not match the selected dataset")
        if checkpoint["stage"] != "road" and checkpoint["data_hashes"]["training_manifest"] != self.hashes["training_manifest"]:
            raise ValueError("Checkpoint was trained with a different data split or supervision manifest")
        if checkpoint["stage"] == "road":
            state = checkpoint["model_state_dict"]
            for name in ("road_encoder", "road_pretraining_heads"):
                getattr(self.model, name).load_state_dict({k[len(name)+1:]: v for k,v in state.items() if k.startswith(name+".")})
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

    def encode(self, batch, additional=None):
        requested = [batch["local_ids"][batch["local_mask"]]] + (additional or [])
        roads, state_map = self.road_batch_encoder.encode(self.model, requested)
        local = self.road_batch_encoder.gather(roads, state_map, batch["local_ids"], batch["local_mask"])
        z = self.model.cell_encoder(batch["cell_ids"].to(self.device), batch["positions"].to(self.device),
                                    local, batch["local_distances_m"].to(self.device), batch["local_mask"].to(self.device))
        if not self.config.use_trajectory_context:
            full = loo = z
        else:
            encoded = self.model.trajectory_encoder(z, batch["lengths"])
            full, loo = encoded if isinstance(encoded, tuple) else (encoded, encoded)
        return {"full": full, "loo": loo, "z": z}, roads, state_map

    def _trajectory_context(self, emit_batch, *, additional_state_ids=None):
        # Compatibility with the unmodified legacy decoder for the baseline.
        context, roads, state_map = self.encode(emit_batch, additional_state_ids)
        return context["full"], roads, state_map, None

    def score_emit(self, batch, context, roads, state_map):
        embedded = self.road_batch_encoder.gather(roads, state_map, batch["candidate_ids"], batch["candidate_mask"])
        d = batch["candidate_distances_m"].to(self.device)
        r = batch["candidate_ranks"].to(self.device)
        gate_logits = None
        if self.config.use_reliability_gate:
            gate_logits = self.model.reliability_gate(context["z"], context["full"], context["loo"], batch["q"].to(self.device))
            gate = torch.sigmoid(gate_logits)
        else:
            gate = torch.ones(context["full"].shape[:2], device=self.device)
        if self.config.emit_mode == "decoupled":
            scores = self.model.emit_head(context["full"], embedded, d, r, batch["candidate_offsets"].to(self.device), context["loo"], gate)
        else:
            scores = self.model.emit_head(context["full"], embedded, d, r)
        return scores.float().masked_fill(~batch["candidate_mask"].to(self.device), -torch.inf), gate_logits

    def emit_step(self, batch_items):
        batch = self.observation_data.batch(batch_items, augment=self.model.training and self.config.use_reliability_gate)
        context, roads, state_map = self.encode(batch, [batch["candidate_ids"][batch["candidate_mask"]]])
        scores, gate_logits = self.score_emit(batch, context, roads, state_map)
        positive = batch["positive_mask"].to(self.device)
        weights = batch["loss_weights"].to(self.device)
        valid = positive.any(-1) & (weights > 0)
        zero = scores[torch.isfinite(scores)].sum()*0.
        if valid.any():
            chosen = scores[valid]
            loss_rows = torch.logsumexp(chosen, -1) - safe_logsumexp(chosen.masked_fill(~positive[valid], -torch.inf), -1)
            mean_loss = weighted_mean(loss_rows, weights[valid])
            loss = self.reduce_loss(loss_rows, weights[valid], "emit")
            order = chosen.argsort(dim=-1, descending=True)
            ranked_positive = positive[valid].gather(-1, order)
            hits = ranked_positive[:, :self.config.emit_top_k].any(-1).float()
            reciprocals = (ranked_positive.float()/torch.arange(1, chosen.shape[-1]+1, device=self.device)).max(-1).values
        else:
            loss = mean_loss = zero; hits = reciprocals = scores.new_zeros(0)
        metrics = {"emit_loss": float(mean_loss.detach()), "valid_observations": int(valid.sum()),
                   "recall_at_30": float(hits.mean()) if hits.numel() else 0.,
                   "mrr": float(reciprocals.mean()) if reciprocals.numel() else 0.,
                   "recall_hits": float(hits.sum()), "reciprocal_sum": float(reciprocals.sum()),
                   "supervision_weight": float(weights[valid].sum())}
        if gate_logits is not None:
            supervised = batch["gate_supervised"].to(self.device)
            labels = (~batch["perturbed"]).float().to(self.device)
            if supervised.any():
                gate_losses = F.binary_cross_entropy_with_logits(gate_logits.float()[supervised], labels[supervised], reduction="none")
                gate_weights = batch["repeat_weights"].to(self.device)[supervised]
                gate_loss = self.reduce_loss(gate_losses, gate_weights, "gate")
                # Validation loss used to choose an emission checkpoint remains
                # the untouched observation ranking loss, not pseudo-clean BCE.
                if self.model.training:
                    gate_coefficient = self.config.lambda_gate
                    if self.stage == "joint":
                        gate_coefficient /= self.config.lambda_emit
                    loss = loss + gate_coefficient*gate_loss
                metrics["gate_loss"] = float(weighted_mean(gate_losses, gate_weights).detach())
            gates = torch.sigmoid(gate_logits.detach())
            for name, selected in (("clean", supervised & (labels > 0)), ("perturbed", supervised & (labels == 0))):
                if selected.any():
                    metrics[f"gate_mean_{name}"] = float(gates[selected].mean())
            if not self.model.training:
                metrics["gate_mean_val"] = float(gates[batch["observation_mask"].to(self.device)].mean())
        metrics["loss"] = float(mean_loss.detach())
        metrics["optimization_loss"] = float(loss.detach())
        return loss, metrics

    def score_trans(self, previous, current, previous_cells, current_cells, previous_context, current_context,
                    roads, state_map, *, route_features=None, reachable=None, endpoint_offsets=None):
        previous, current = previous.cpu(), current.cpu()
        previous_cells, current_cells = previous_cells.cpu(), current_cells.cpu()
        all_true = torch.ones_like(previous, dtype=torch.bool)
        pr = self.road_batch_encoder.gather(roads, state_map, previous, all_true)
        cr = self.road_batch_encoder.gather(roads, state_map, current, all_true)
        features = pair_features(previous, current, previous_cells, current_cells,
                                 road_features=self.road_features, cell_positions_metric=self.resources.cell_positions_metric).to(self.device)
        parts = []
        for start in range(0, len(previous), self.config.trans_chunk_size):
            sl = slice(start, start+self.config.trans_chunk_size)
            parts.append(self.model.trans_head(pr[sl], cr[sl], previous_context[sl], current_context[sl], features[sl]))
        neural = torch.cat(parts).float()
        aux = {}
        # The historical baseline trains its endpoint-only head without a route
        # mask. Its original decoder still enforces physical reachability.
        if self.config.variant == "baseline" and reachable is None:
            return neural, aux
        xy = self.resources.cell_positions_metric
        dref = (xy[current_cells]-xy[previous_cells]).norm(dim=-1).clamp_min(self.config.d_min_m)
        if route_features is None:
            limits = np.maximum(self.config.route_gamma_cut*dref.numpy(), dref.numpy()+self.config.route_abs_slack_m)
            if not self.config.use_route_cutoff:
                limits[:] = np.inf
            route_features, reachable = self.router.route_pairs(previous.numpy(), current.numpy(), limits)
        if self.config.use_path_costs:
            if endpoint_offsets is None:
                po, co = np.zeros(len(previous), dtype=np.float32), np.zeros(len(previous), dtype=np.float32)
                for i, (s,t,pc,cc) in enumerate(zip(previous.tolist(), current.tolist(), previous_cells.tolist(), current_cells.tolist())):
                    po[i] = self.observation_data.offsets[pc][self.observation_data.lookup(pc)[s]]
                    co[i] = self.observation_data.offsets[cc][self.observation_data.lookup(cc)[t]]
            else:
                po, co = endpoint_offsets
            penalty, costs, dnet = self.model.physical_costs(
                torch.as_tensor(route_features, device=self.device), torch.as_tensor(po, device=self.device), torch.as_tensor(co, device=self.device),
                self.road_features["length_m"][previous].to(self.device), self.road_features["length_m"][current].to(self.device),
                dref.to(self.device), (previous==current).to(self.device))
            neural = neural - penalty
            aux = {"costs": costs, "dnet": dnet, "dref": dref}
        return neural.masked_fill(~torch.as_tensor(reachable, device=self.device), -torch.inf), aux

    def trans_step(self, groups):
        items = [(g["trajectory_id"], self.resources.alignment_train.get(g["trajectory_id"]) or self.resources.alignment_val[g["trajectory_id"]]) for g in groups]
        batch = self.observation_data.batch(items)
        previous, current, pc, cc, bi, ti, tj, positives, logw, weights, boundaries = [], [], [], [], [], [], [], [], [], [], [0]
        for b, group in enumerate(groups):
            a = items[b][1]
            for record in group["transitions"]:
                t = int(record["transition_index"])
                if not a.get("transition_loss_mask", [True]*(len(a["cell_ids"])-1))[t] or record.get("loss_weight", 1.) <= 0:
                    continue
                pairs = list(record["positive_state_pairs"])+list(record["negative_state_pairs"])
                n = len(pairs); npos = len(record["positive_state_pairs"])
                previous.extend(p[0] for p in pairs); current.extend(p[1] for p in pairs)
                pc.extend([record["previous_cell_id"]]*n); cc.extend([record["current_cell_id"]]*n)
                bi.extend([b]*n); ti.extend([record["previous_observation_index"]]*n); tj.extend([record["current_observation_index"]]*n)
                positives.extend([True]*npos+[False]*(n-npos)); logw.extend(record["positive_log_weights"]+[0.]*(n-npos))
                weights.append(record.get("loss_weight",1.)); boundaries.append(boundaries[-1]+n)
        if not previous:
            return next(self.model.parameters()).sum()*0., {"loss": 0., "recall_at_10": 0., "valid_transitions": 0.}
        previous = torch.tensor(previous); current = torch.tensor(current)
        context, roads, state_map = self.encode(batch, [previous,current])
        bi = torch.tensor(bi, device=self.device); ti = torch.tensor(ti, device=self.device); tj = torch.tensor(tj, device=self.device)
        scores, auxiliary = self.score_trans(previous,current,torch.tensor(pc),torch.tensor(cc),context["full"][bi,ti],context["full"][bi,tj],roads,state_map)
        positives = torch.tensor(positives,device=self.device); logw = torch.tensor(logw,device=self.device)
        losses, valid_weights, hits = [], [], []
        discarded = 0
        for i,(start,end) in enumerate(zip(boundaries,boundaries[1:])):
            s = scores[start:end]; positive = positives[start:end] & torch.isfinite(s)
            if not positive.any():
                discarded += 1
                continue
            # Re-normalize the original prior only over reachable positive pairs.
            w = logw[start:end][positive]; w = w-torch.logsumexp(w,0)
            losses.append(torch.logsumexp(s,0)-torch.logsumexp(s[positive]+w,0))
            valid_weights.append(weights[i]); hits.append(positive[s.topk(min(10,len(s))).indices].any().float())
        if losses:
            row_losses = torch.stack(losses)
            row_weights = scores.new_tensor(valid_weights)
            loss = self.reduce_loss(row_losses, row_weights, "trans")
            mean_loss = weighted_mean(row_losses, row_weights)
        else:
            loss = mean_loss = scores[torch.isfinite(scores)].sum()*0.
        diagnostics = {}
        finite = torch.isfinite(scores)
        if "costs" in auxiliary and finite.any():
            means = auxiliary["costs"][finite].detach().mean(0).tolist()
            learned = F.softplus(self.model.physical_costs.cost_logits.detach()).tolist()
            diagnostics.update({f"cost_mean_{i+1}":v for i,v in enumerate(means)})
            diagnostics.update({f"physical_weight_{i+1}":v for i,v in enumerate(learned)})
        return loss, {"loss":float(mean_loss.detach()), "optimization_loss":float(loss.detach()),
                      "supervision_weight":sum(valid_weights), "recall_at_10":float(torch.stack(hits).mean()) if hits else 0.,
                      "valid_transitions":len(losses), "unreachable_positive_groups":discarded,
                      "transition_hits":float(torch.stack(hits).sum()) if hits else 0., **diagnostics}

    @staticmethod
    def _mean_metrics(values):
        if not values:
            return {"loss":0.}
        result = TrainingEngine._mean_metrics(values)
        count = sum(v.get("valid_observations",0) for v in values)
        if count:
            result["recall_at_30"] = sum(v.get("recall_hits",0) for v in values)/count
            result["mrr"] = sum(v.get("reciprocal_sum",0) for v in values)/count
        trans = sum(v.get("valid_transitions",0) for v in values)
        if trans:
            result["recall_at_10"] = sum(v.get("transition_hits",0) for v in values)/trans
        for k in ("valid_observations","valid_transitions","unreachable_positive_groups","recall_hits","reciprocal_sum","transition_hits"):
            if any(k in v for v in values):
                result[k] = sum(v.get(k,0) for v in values)
        if any("supervision_weight" in v for v in values):
            denominator = sum(v.get("supervision_weight",0) for v in values)
            result["supervision_weight"] = denominator
            if denominator:
                result["loss"] = sum(v["loss"]*v.get("supervision_weight",0) for v in values)/denominator
                if any("emit_loss" in v for v in values):
                    result["emit_loss"] = sum(v.get("emit_loss",0)*v.get("supervision_weight",0) for v in values)/denominator
        return result

    def _optimize_loss(self, loss, optimizer, batch_index):
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss in {self.stage}, epoch {self.epoch}")
        result = super()._optimize_loss(loss,optimizer,batch_index)
        now=time.monotonic()
        if now-self._last_progress>30:
            print(json.dumps({"stage":self.stage,"epoch":self.epoch,"batch":batch_index,"loss":float(loss.detach()),"global_step":self.global_step}),flush=True)
            self._last_progress=now
        return result

    def _optimizer_step(self, optimizer):
        self.scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm, error_if_nonfinite=not self.amp)
        if not torch.isfinite(norm):
            print({"amp_overflow_skipped": True, "stage": self.stage, "scale": self.scaler.get_scale()}, flush=True)
        self.scaler.step(optimizer); self.scaler.update(); optimizer.zero_grad(set_to_none=True); self.global_step += 1

    def _train_task_stage(self, stage, *, resume=None):
        # Keep the inherited, checked stage schedule and checkpoint selection.
        self.stage = stage
        return super()._train_task_stage(stage,resume=resume)

    def _iter_trans_batches(self, groups, epoch, shuffle):
        self.epoch=epoch
        return super()._iter_trans_batches(groups,epoch,shuffle)

    def refresh_candidates(self, checkpoint_path):
        self.initialize_from(checkpoint_path); self.model.eval()
        root = self.config.feature_root / "emit_candidates" / self.config.experiment_id
        paths = {}
        with torch.no_grad():
            for split, alignments in (("train",self.resources.alignment_train),("val",self.resources.alignment_val)):
                output={}
                for items in iter_trajectory_batches(alignments,self.resources.cell_candidates,candidate_budget=self.config.batch_candidate_budget,
                                                    seed=self.config.seed,shuffle=False,limit=self.config.limit_trajectories):
                    batch=self.observation_data.batch(items)
                    with _autocast(self.device,self.amp):
                        context,roads,state_map=self.encode(batch,[batch["candidate_ids"][batch["candidate_mask"]]])
                        scores,_=self.score_emit(batch,context,roads,state_map)
                    for b,(tid,a) in enumerate(items):
                        layers=[]
                        for t in range(len(a["cell_ids"])):
                            n=int(batch["candidate_mask"][b,t].sum()); take=min(self.config.emit_top_k,n)
                            top=scores[b,t,:n].topk(take).indices.cpu()
                            layers.append(batch["candidate_ids"][b,t,:n][top].tolist())
                        output[tid]=layers
                p=root/f"{split}.obj"; save_pickle(p,output); paths[split]=p
        write_json(root/"metadata.json",{"schema_version":2,"checkpoint":str(checkpoint_path),"top_k":self.config.emit_top_k,
                                        "training_forces_soft_positives":False,"validation_forces_soft_positives":False,
                                        "refresh_policy":"once after emission warmup; frozen through transition and joint stages",
                                        "local_trans_training_policy":"top_k_plus_original_trusted_positives"})
        return paths

    def evaluator(self):
        if self._evaluator is None:
            from slhmm.training.unified_evaluation import UnifiedEvaluator
            self._evaluator=UnifiedEvaluator(self)
        return self._evaluator

    def sequence_batches(self, epoch):
        items=list(self.resources.alignment_train.items())
        if self.config.limit_trajectories is not None:
            items=items[:self.config.limit_trajectories]
        random.Random(self.config.seed+epoch).shuffle(items)
        batch=[]; budget=0
        for item in items:
            cost=max(1,len(item[1]["cell_ids"])-1)*self.config.emit_top_k**2
            if batch and budget+cost>self.config.lattice_pair_budget:
                yield batch; batch=[]; budget=0
            batch.append(item); budget+=cost
        if batch: yield batch

    def sequence_step(self, items, candidate_pools):
        pools=[candidate_pools[tid] for tid,_ in items]
        scored=score_lattice(self,items,pools)
        losses=[]; weights=[]; metrics=[]
        for (tid,a),pool,result in zip(items,pools,scored):
            automaton=self.lattice_builder.positive_automaton(result["key"],a,pool,result["routes"])
            loss,stat=lattice_crf_loss(result["emit"],result["trans"],automaton)
            metrics.append(stat)
            if automaton.valid:
                losses.append(loss)
                # Overlapping views retain their source-observation repeat
                # correction; local losses use exact point/pair weights.
                weights.append(float(np.mean(a.get("input_repeat_weights",[1.]))))
        if not losses:
            zero=next(self.model.parameters()).sum()*0.
            return zero,self._mean_metrics(metrics)
        rows = torch.stack(losses); row_weights = losses[0].new_tensor(weights)
        loss = self.reduce_loss(rows, row_weights, "sequence")
        return loss,{**self._mean_metrics(metrics), "loss":float(weighted_mean(rows,row_weights).detach()),
                     "optimization_loss":float(loss.detach()), "supervision_weight":sum(weights)}

    def finetune(self, *, initialize, resume=None):
        if self.config.variant == "baseline":
            self.stage = "joint"
            return super().finetune(initialize=initialize, resume=resume)
        self.initialize_from(initialize)
        self.stage="joint"
        optimizer=self._task_optimizer("joint"); optimizer.zero_grad(set_to_none=True)
        start=0
        if resume:
            checkpoint=load_checkpoint(resume,model=self.model,optimizer=optimizer,scaler=self.scaler,map_location=self.device)
            start=int(checkpoint["epoch"])+1; self.global_step=int(checkpoint["global_step"])
        trans_train=load_enriched_trans_groups(self.config,"train")
        trans_by_trajectory = {g["trajectory_id"]: g for g in trans_train}
        trans_val=load_enriched_trans_groups(self.config,"val")
        candidates=load_pickle(self.config.feature_root/"emit_candidates"/self.config.experiment_id/"train.obj")
        stage_root=self.config.checkpoint_root/"joint"
        best=-math.inf; patience=0
        oldbest=stage_root/"best.pt"
        if resume and oldbest.exists():
            best=float(torch.load(oldbest,map_location="cpu",weights_only=False)["metrics"]["viterbi_composite"])
        for epoch in range(start,self.config.joint_epochs):
            self.epoch=epoch; self.model.train()
            joint_batches = list(self.sequence_batches(epoch))
            self.configure_training_normalizers("joint", len(joint_batches), trans_train)
            records={"emit":[], "trans":[], "sequence":[]}; updates=0
            # Full and no_lattice use identical trajectory groups and optimizer
            # updates. Each micro-batch uses its logical group's fixed scale.
            for group_index, items in enumerate(joint_batches):
                stats={}
                emit_batches = iter_trajectory_batches(dict(items), self.resources.cell_candidates,
                    candidate_budget=self.config.batch_candidate_budget, seed=self.config.seed, shuffle=False)
                trans_batches = self._iter_trans_batches(
                    [trans_by_trajectory[tid] for tid,_ in items if tid in trans_by_trajectory], epoch, shuffle=False)
                accumulation_count = min(self.config.gradient_accumulation_steps,
                    len(joint_batches) - (group_index // self.config.gradient_accumulation_steps)*self.config.gradient_accumulation_steps)
                tasks = [("emit", emit_batches, self.config.lambda_emit), ("trans", trans_batches, self.config.lambda_trans)]
                if self.config.use_lattice_loss:
                    tasks.append(("sequence", [items], 1.))
                for task, micro_batches, coefficient in tasks:
                    values_list=[]
                    for micro in micro_batches:
                        with _autocast(self.device,self.amp):
                            if task=="emit": loss,values=self.emit_step(micro)
                            elif task=="trans": loss,values=self.trans_step(micro)
                            else: loss,values=self.sequence_step(micro,candidates)
                        if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite {task} joint loss")
                        self.scaler.scale(coefficient*loss/accumulation_count).backward()
                        values_list.append(values); records[task].append(values)
                        del loss
                    stats.update({f"{task}_{k}":v for k,v in self._mean_metrics(values_list).items()})
                updates+=1
                if updates%self.config.gradient_accumulation_steps==0: self._optimizer_step(optimizer)
                if time.monotonic()-self._last_progress>30:
                    print(json.dumps({"stage":"joint","epoch":epoch,"batch":updates,**stats}),flush=True); self._last_progress=time.monotonic()
            if updates%self.config.gradient_accumulation_steps: self._optimizer_step(optimizer)
            emit_val=self._evaluate_emit("val")
            self.model.eval()
            with torch.no_grad():
                values=[]
                for groups in self._iter_trans_batches(trans_val,epoch,shuffle=False):
                    with _autocast(self.device,self.amp): _,v=self.trans_step(groups)
                    values.append(v)
                trans_metrics=self._mean_metrics(values)
            report=self.evaluator().evaluate(self.config.run_root/"viterbi"/f"joint_epoch_{epoch:03d}.json")
            viterbi=report["metrics"]
            metrics={**{f"train_{task}_{k}":v for task,values in records.items() for k,v in self._mean_metrics(values).items()},
                     "joint_trajectory_groups":len(joint_batches),
                     **{f"val_emit_{k}":v for k,v in emit_val.items()},
                     **{f"val_trans_{k}":v for k,v in trans_metrics.items()},
                     **{f"viterbi_{k}":v for k,v in viterbi.items()}}
            path=stage_root/f"epoch_{epoch:03d}.pt"
            save_checkpoint(path,stage="joint",epoch=epoch,global_step=self.global_step,model=self.model,optimizer=optimizer,
                            scheduler=None,scaler=self.scaler,config=self.config,data_hashes=self.hashes,metrics=metrics)
            write_json(self.config.run_root/"epochs"/f"joint_{epoch:03d}.json",metrics)
            print(json.dumps({"stage":"joint_epoch_done","epoch":epoch,**viterbi}),flush=True)
            if viterbi["checkpoint_eligible"] and viterbi["composite"]>best:
                best=viterbi["composite"]; _copy_best(path,stage_root); patience=0
            else: patience+=1
            if patience>=self.config.early_stopping_patience: break
        if not (stage_root/"best.pt").exists():
            raise RuntimeError("No joint checkpoint passed topology continuity >=99%; results not silently substituted")
        return stage_root/"best.pt"
