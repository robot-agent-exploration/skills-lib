"""Run a task composition end to end, from / up to a step, or checkpoint by checkpoint.

From the workspace root:
    python3 code/rby1_tasks/run_task.py pnp_box --checkpoints            # new run, pauses at checkpoints
    python3 code/rby1_tasks/run_task.py pnp_box --checkpoints --resume output/runs/<run> --start <next>
    python3 code/rby1_tasks/run_task.py pnp_box                          # unattended (no pauses)
Writes output/runs/<time>_<task>/{log.txt, report_*.json, world.json, path_*.jpg, table_sides_*.jpg}.
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
WORKSPACE = next((p for p in HERE.parents if (p / 'api' / 'runtime.py').exists()), HERE.parents[1])
sys.path.insert(0, str(WORKSPACE))

from rby1_tasks.core.runner import Runtime, run_steps   # noqa: E402
from rby1_tasks.core.world import World                 # noqa: E402


def main(ctx, task='pnp_box', start=None, stop_after=None, resume=None, checkpoints=False, outdir=None, params=None,
         **task_kwargs):
    """-> {ok, why, checkpoint, next, outdir, measured}"""
    mod = importlib.import_module(f'rby1_tasks.tasks.{task}')
    if resume:
        outdir = Path(resume)
        world = World.load(outdir / 'world.json') if (outdir / 'world.json').exists() else World()
    else:
        outdir = Path(outdir or WORKSPACE / 'output' / 'runs' / f'{time.strftime("%Y%m%d_%H%M%S")}_{task}')
        world = World()
    rt = Runtime(ctx, outdir, world)
    rep = run_steps(rt, mod.steps(**task_kwargs), start=start, stop_after=stop_after, checkpoints=checkpoints,
                    params=params)
    return {'ok': rep['ok'], 'why': rep['why'], 'checkpoint': rep['checkpoint'], 'next': rep['next'],
            'outdir': str(outdir), 'measured': {s['step']: s.get('result') for s in rep['steps'] if s['ok']}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('task')
    ap.add_argument('--start')
    ap.add_argument('--stop-after')
    ap.add_argument('--resume', help='run directory to continue (world.json, map.npz)')
    ap.add_argument('--checkpoints', action='store_true')
    ap.add_argument('--param', action='append', default=[], help='key=value for the step started (agent choice)')
    a = ap.parse_args()
    params = {}
    for kv in a.param:
        k, v = kv.split('=', 1)
        params[k] = json.loads(v) if v[:1] in '[{0123456789-' or v in ('true', 'false') else v
    from api.runtime import Context
    with Context() as session:
        res = main(session, a.task, start=a.start, stop_after=a.stop_after, resume=a.resume, checkpoints=a.checkpoints,
                   params=params)
    print(json.dumps(res, indent=1, default=str)[:4000])
