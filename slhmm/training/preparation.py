"""Prepare isolated static artifacts and training-only supervision statistics."""
from __future__ import annotations
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import gc
import multiprocessing
import numpy as np
from slhmm.training.common import load_pickle, read_json, write_json, sha256_file
from slhmm.training.features import prepare_road_artifacts
from slhmm.training.data import prepare_enriched_trans, iter_enriched_trans_groups


def available(config):
    root=config.feature_root
    needed=[root/"road/metadata.json",root/"road/road_features.pt",root/"road/road_relations.dgl",root/"road/geometries.obj",
            root/"trans_enriched/manifest.json",root/"h3/metadata.json",root/"h3/cell_h3_ids.pt",root/"training_statistics.json",
            root/f"cell_offsets_k{config.candidate_k}.obj"]
    if not all(p.is_file() for p in needed): return False
    metadata=read_json(root/"road/metadata.json"); stats=read_json(root/"training_statistics.json")
    return (metadata.get("schema_version")==2 and metadata.get("metric_epsg")==config.metric_epsg
            and metadata["graph_sha256"]==sha256_file(config.graph_path)
            and metadata["mappings_sha256"]==sha256_file(config.training_data_root/"mappings.obj")
            and stats["alignment_train_sha256"]==sha256_file(config.training_data_root/"alignment_train.obj")
            and stats["source_manifest_sha256"]==sha256_file(config.training_data_root/"manifest.json"))


def prepare(config):
    if available(config): return {"reused_verified_artifacts":True,"feature_root":str(config.feature_root)}
    print("Preparing geometry and projection offsets",flush=True)
    prepare_road_artifacts(config)
    print("Preparing trusted transition supervision",flush=True)
    prepare_enriched_trans(config)
    from slhmm.training.h3_prior import prepare_h3_artifacts
    h3=prepare_h3_artifacts(config)
    # Read original enriched shards, never previously refreshed model pools.
    manifest=read_json(config.feature_root/"trans_enriched/manifest.json")
    gaps=[]; weights=[]
    for shard in manifest["shards"]["train"]:
        for group in load_pickle(config.feature_root/shard["path"]):
            for r in group["transitions"]:
                for p in r["positive_metadata"]:
                    gaps.append(p["route_index_gap"]); weights.append(r.get("loss_weight",1.))
    order=np.argsort(gaps); values=np.asarray(gaps)[order]; w=np.asarray(weights)[order]
    q=int(values[np.searchsorted(w.cumsum(), .99*w.sum())])
    stats={"schema_version":2,"delta_max":max(1,q),"delta_quantile":.99,"fit_split":"train",
           "quantile_weighting":"source transition inverse repeat count where supplied",
           "source_manifest_sha256":sha256_file(config.training_data_root/"manifest.json"),
           "alignment_train_sha256":sha256_file(config.training_data_root/"alignment_train.obj"),"h3":h3}
    write_json(config.feature_root/"training_statistics.json",stats)
    return stats


_PATH_ROUTER=None
def _prepare_source(task):
    source,targets,limit=task
    return _PATH_ROUTER.prepare_source(source,targets,limit)


def _path_worker_initialize():
    # The inherited engine/target sets are read-only. Avoid cyclic GC scanning
    # them in every child and turning shared fork pages into private copies.
    gc.disable()


def bounded_map(executor, function, tasks, maximum_pending):
    """Preserve input order while bounding queued task payloads in Python 3.11."""
    if maximum_pending < 1:
        raise ValueError('maximum_pending must be positive')
    tasks = iter(tasks)
    pending = deque()
    for _ in range(maximum_pending):
        try: task = next(tasks)
        except StopIteration: break
        pending.append(executor.submit(function, task))
    while pending:
        yield pending.popleft().result()
        try: task = next(tasks)
        except StopIteration: continue
        pending.append(executor.submit(function, task))


def prepare_paths(engine,workers=8):
    global _PATH_ROUTER
    _PATH_ROUTER=engine.router
    targets=defaultdict(set); limits=defaultdict(float)
    config=engine.config; xy=engine.resources.cell_positions_metric.numpy()
    for split,alignments in (("train",engine.resources.alignment_train),("val",engine.resources.alignment_val)):
        pools=load_pickle(config.feature_root/"emit_candidates"/config.experiment_id/f"{split}.obj")
        for tid,pool in pools.items():
            a=alignments[tid]
            for t,(left,right) in enumerate(zip(pool,pool[1:])):
                ref=max(float(np.linalg.norm(xy[a["cell_ids"][t+1]]-xy[a["cell_ids"][t]])),config.d_min_m)
                limit=max(config.route_gamma_cut*ref,ref+config.route_abs_slack_m) if config.use_route_cutoff else np.inf
                for s in left: targets[s].update(right); limits[s]=max(limits[s],limit)
        for group in iter_enriched_trans_groups(config,split):
            for r in group["transitions"]:
                ref=max(float(np.linalg.norm(xy[r["current_cell_id"]]-xy[r["previous_cell_id"]])),config.d_min_m)
                limit=max(config.route_gamma_cut*ref,ref+config.route_abs_slack_m) if config.use_route_cutoff else np.inf
                for s,t in r["positive_state_pairs"]+r["negative_state_pairs"]:
                    targets[s].add(t); limits[s]=max(limits[s],limit)
    tasks=((s,sorted(ts),limits[s]) for s,ts in sorted(targets.items()))
    total=len(targets); pairs=0
    print({"prepare_path_sources":total,"workers":workers},flush=True)
    gc.collect()
    # Workers only run CPU SciPy and NumPy. Fork shares the read-only CSR graph;
    # they never access or initialize the parent's CUDA model.
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("fork"),
                             initializer=_path_worker_initialize) as pool:
        for i,count in enumerate(bounded_map(pool,_prepare_source,tasks,maximum_pending=2*workers),1):
            pairs+=count
            if i%500==0: print({"path_sources_done":i,"total":total,"pairs":pairs},flush=True)
    engine.router.clear_disk_cache()
    report={"schema_version":3,"router_identity":engine.router.identity,"sources":total,"pairs":pairs,"candidate_experiment":config.experiment_id}
    write_json(config.feature_root/"emit_candidates"/config.experiment_id/"paths_prepared.json",report)
    return report
