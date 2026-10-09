"""Skill entry: pick the cardboard box off the floor and put it on the black-top table (RB-Y1), checkpoint by
checkpoint with the agent in the loop.

    from the workspace root, in Python:
        spec = importlib.util.spec_from_file_location('skill', 'skill-library/skills/box-floor-to-table-squeeze/code/run.py')
        ...; result = skill.main(ctx)                                    # runs to the first checkpoint
        result = skill.main(ctx, resume=result['outdir'], start=result['next'])          # after reviewing
        result = skill.main(ctx, resume=..., start='plan_table', params={'side': 1})       # agent's choice

main() returns {'ok', 'measured', 'why', 'checkpoint', 'next', 'outdir'}; ok is True only when the last step
(retreat) finished. Each checkpoint leaves pictures (path_*.jpg, table_sides_*.jpg) and log.txt in outdir.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rby1_tasks import run_task   # noqa: E402


def main(ctx, start=None, resume=None, checkpoints=True, params=None, task='pnp_box', stop_after=None, **task_kwargs):
    r = run_task.main(ctx, task, start=start, stop_after=stop_after, resume=resume, checkpoints=checkpoints,
                      params=params, **task_kwargs)
    done = r['ok'] and r['checkpoint'] is None and stop_after is None
    why = r['why'] or (f"checkpoint after {r['checkpoint']}: review the pictures in {r['outdir']}, then resume with "
                       f"start={r['next']!r}" if r['checkpoint'] else 'finished')
    return {'ok': bool(done), 'measured': r['measured'], 'why': why, 'checkpoint': r['checkpoint'], 'next': r['next'],
            'outdir': r['outdir']}


if __name__ == '__main__':
    import json
    WORKSPACE = next(p for p in HERE.parents if (p / 'api' / 'runtime.py').exists())
    sys.path.insert(0, str(WORKSPACE))
    from api.runtime import Context
    with Context() as session:
        print(json.dumps(main(session), indent=1, default=str)[:3000])
