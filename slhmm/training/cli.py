"""Reproducible dual-dataset SLHMM ablation command line."""
from __future__ import annotations
import argparse
from dataclasses import fields
from pathlib import Path
import json
import os
import torch
from slhmm.configs.training import TrainingConfig
from slhmm.training.preparation import prepare,available,prepare_paths
from slhmm.training.common import write_json,read_json

VARIANTS=('baseline','full','no_road_pretrain','no_context_gate','no_h3','no_path_costs','no_lattice','no_joint')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('hangzhou','xian'),default='hangzhou')
    p.add_argument('--variant',choices=VARIANTS,default='full')
    p.add_argument('--experiment-id'); p.add_argument('--snapshot'); p.add_argument('--config')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto')
    p.add_argument('--no-amp',action='store_true'); p.add_argument('--limit-trajectories',type=int)
    p.add_argument('--emit-mode',choices=('concat','decoupled'))
    for name in ('road_pretrain','trajectory_context','h3','reliability_gate','path_costs','lattice_loss','route_cutoff'):
        p.add_argument('--no-'+name.replace('_','-'),action='store_true')
    p.add_argument('--skip-joint',action='store_true')
    for name in ('road_pretrain_epochs','emit_epochs','trans_epochs','joint_epochs','batch_candidate_budget','gradient_accumulation_steps','emit_top_k','trans_negative_count','lattice_pair_budget','trans_chunk_size','num_workers'):
        p.add_argument('--'+name.replace('_','-'),type=int)
    for name in ('eta_turn','eta_uturn','eta_tunnel','route_gamma_cut','route_abs_slack_m','d_min_m','detour_gamma','lambda_len','mu_back','mu_uturn','beta_tunnel','cost_clamp','lambda_emit','lambda_trans','lambda_gate','p_perturb'):
        p.add_argument('--'+name.replace('_','-'),type=float)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    for cmd in ('pretrain-road','train-emit','train-trans','finetune'):
        s=sub.add_parser(cmd); s.add_argument('--initialize'); s.add_argument('--resume')
    s=sub.add_parser('refresh-candidates'); s.add_argument('--checkpoint')
    s=sub.add_parser('prepare-paths'); s.add_argument('--workers',type=int,default=8)
    s=sub.add_parser('evaluate'); s.add_argument('--checkpoint'); s.add_argument('--output')
    s.add_argument('--visualize',action='store_true'); s.add_argument('--visualization-limit',type=int,default=20)
    s=sub.add_parser('run-all'); s.add_argument('--workers',type=int,default=8)
    return p


def _config(args):
    valid={f.name for f in fields(TrainingConfig)}
    values={k:v for k,v in (read_json(args.config) if args.config else {}).items() if k in valid}
    for k,v in vars(args).items():
        if k in valid and v is not None: values[k]=v
    values['amp']=not args.no_amp
    for name in ('road_pretrain','trajectory_context','h3','reliability_gate','path_costs','lattice_loss','route_cutoff'):
        if getattr(args,'no_'+name): values['use_'+name]=False
    if args.skip_joint: values['skip_joint']=True
    if not values.get('experiment_id'):
        values['experiment_id']=f"{args.dataset}_{args.variant}_seed{args.seed}"
    return TrainingConfig(**values)


def _checkpoint(c,stage):
    p=c.checkpoint_root/stage/'best.pt'
    if not p.exists(): raise FileNotFoundError(p)
    return p


def _prepared_artifacts_available(c): return available(c)


def main(argv=None):
    args=build_parser().parse_args(argv); c=_config(args)
    from slhmm.training.delegation import delegated_phase
    if delegated_phase(c, args.command):
        return
    torch.set_num_threads(int(os.environ.get('SLHMM_TORCH_THREADS','4')))
    write_json(c.run_root/'config.json',c.to_dict())
    if args.command=='prepare': print(prepare(c),flush=True); return
    if not available(c): raise RuntimeError('Run prepare before training; static artifacts are missing or incompatible')
    from slhmm.training.engine import EnhancedEngine
    e=EnhancedEngine(c)
    if args.command=='pretrain-road': print(e.pretrain_road(resume=args.resume),flush=True)
    elif args.command=='train-emit':
        initial=args.initialize or (_checkpoint(c,'road') if c.use_road_pretrain else None)
        print(e.train_emit(initialize=initial,resume=args.resume),flush=True)
    elif args.command=='refresh-candidates':
        print(e.refresh_candidates(args.checkpoint or _checkpoint(c,'emit')),flush=True)
        print(e.refresh_trans_candidates(),flush=True)
    elif args.command=='prepare-paths': print(prepare_paths(e,args.workers),flush=True)
    elif args.command=='train-trans': print(e.train_trans(initialize=args.initialize or _checkpoint(c,'emit'),resume=args.resume),flush=True)
    elif args.command=='finetune': print(e.finetune(initialize=args.initialize or _checkpoint(c,'trans'),resume=args.resume),flush=True)
    elif args.command=='evaluate':
        e.initialize_from(args.checkpoint or _checkpoint(c,'trans' if c.skip_joint else 'joint'))
        kwargs={}
        if args.visualize: kwargs=dict(visualization_dir=c.run_root/'visualizations',visualization_limit=args.visualization_limit)
        print(e.evaluator().evaluate(args.output or c.run_root/'viterbi/evaluation.json',**kwargs)['metrics'],flush=True)
    elif args.command=='run-all':
        road=e.pretrain_road() if c.use_road_pretrain else None
        emit=e.train_emit(initialize=road)
        e.refresh_candidates(emit); e.refresh_trans_candidates()
        if c.variant!='baseline': prepare_paths(e,args.workers)
        trans=e.train_trans(initialize=emit)
        final=trans if c.skip_joint else e.finetune(initialize=trans)
        e.initialize_from(final)
        print(e.evaluator().evaluate(c.run_root/'viterbi/evaluation.json')['metrics'],flush=True)
    else: raise AssertionError(args.command)


if __name__=='__main__': main()
