"""Coordinate one explicitly assigned experiment without a second trainer."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import uuid

from slhmm.configs.runtime import PROJECT_ROOT

PHASES = {'pretrain-road','train-emit','refresh-candidates','prepare-paths','train-trans','finetune','evaluate'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def delegated_phase(config, phase, *, root=None, poll_seconds=20):
    """A non-owner joins the assigned phase; it never launches another trainer.

    The old serial driver imports this hook in each newly started CLI process,
    so an already-running training process does not need to be interrupted.
    """
    if phase not in PHASES:
        return False
    path = Path(root or PROJECT_ROOT) / 'results/delegated' / (config.experiment_id + '.json')
    if not path.exists():
        return False
    announced = False
    while True:
        state = json.loads(path.read_text())
        for name in ('dataset','variant','seed'):
            if getattr(config, name) != state[name]:
                raise ValueError(f'Delegated experiment has incompatible {name}')
        if config.limit_trajectories is not None:
            raise ValueError('Cannot run a limited subset under a formal delegated experiment ID')
        expected = state.get('model_config', {})
        actual = json.loads(json.dumps(config.to_dict()))
        for name, value in expected.items():
            if name not in ('device', 'num_workers') and actual.get(name) != value:
                raise ValueError(f'Delegated experiment has incompatible setting: {name}')
        if os.environ.get('SLHMM_ASSIGNMENT_OWNER') == state['owner']:
            return False
        progress = state.get('phases', {}).get(phase, {})
        if progress.get('status') == 'complete':
            print(json.dumps({'delegated_phase_reused':phase,'experiment':config.experiment_id,
                              'owner':state['owner'],'log':progress.get('log')},ensure_ascii=False),flush=True)
            return True
        if state.get('status') == 'failed' or progress.get('status') == 'failed':
            raise RuntimeError(f"Assigned worker failed: {state.get('error', progress)}")
        if state.get('status') == 'complete':
            raise RuntimeError(f'Completed assigned experiment has no completed {phase} record')
        if not announced:
            print(json.dumps({'waiting_for_assigned_phase':phase,'owner':state['owner']},ensure_ascii=False),flush=True)
            announced = True
        time.sleep(poll_seconds)
