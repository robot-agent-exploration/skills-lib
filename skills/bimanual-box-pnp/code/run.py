"""Skill entry (bimanual-box-pnp): pick and place a closed cardboard box with the bimanual closed-ball squeeze,
floor <-> black-top table, step by step with checkpoints (agent / subagent / Gemini perception in the loop).

    main(ctx, task='floor_to_table' | 'table_to_floor', start=None, resume=None, checkpoints=True, params=None,
         stop_after=None, **task_kwargs) -> {ok, measured, why, checkpoint: {step, next, brief}, next, outdir}
A checkpoint returns a brief (frames, question, pass_if, on_fail) for a subagent / Gemini / the agent; resume
with main(ctx, task, resume=r['outdir'], start=r['next']). ok is True only when the task's last step finished."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from boxpnp.core.runner import run_task, workspace_root     # noqa: E402
from boxpnp.tasks import floor_to_table, table_to_floor     # noqa: E402

TASKS = {'floor_to_table': floor_to_table.steps, 'table_to_floor': table_to_floor.steps}


def main(ctx, task='floor_to_table', start=None, resume=None, checkpoints=True, params=None, stop_after=None,
         **task_kwargs):
    if task not in TASKS:
        return {'ok': False, 'measured': {}, 'why': f'unknown task {task}; one of {sorted(TASKS)}'}
    return run_task(ctx, TASKS[task](**task_kwargs), task=task, start=start, stop_after=stop_after, resume=resume,
                    checkpoints=checkpoints, params=params)


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=sorted(TASKS))
    ap.add_argument('--start')
    ap.add_argument('--resume')
    ap.add_argument('--stop-after')
    ap.add_argument('--no-checkpoints', action='store_true')
    ap.add_argument('--params', help='JSON kwargs for the step the run starts at, e.g. {"edge_x": 0.45}')
    a = ap.parse_args()
    sys.path.insert(0, str(workspace_root()))
    from api.runtime import Context
    with Context() as session:
        r = main(session, a.task, start=a.start, resume=a.resume, checkpoints=not a.no_checkpoints,
                 stop_after=a.stop_after, params=json.loads(a.params) if a.params else None)
    print(json.dumps(r, indent=1, default=str)[:6000])
