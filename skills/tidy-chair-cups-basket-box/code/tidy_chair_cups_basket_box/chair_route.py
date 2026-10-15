"""Move the held chair along a base route with guards: pull (right caged), turns with firm relief and grip-loss
checks, drives with load / wrist guards. Usage, from the workspace root with <skill>/code on PYTHONPATH:
python -m tidy_chair_cups_basket_box.chair_route <moves json> e.g. '[["pull",0.7],["turn",-1.5708],["drive",1.42],["turn",1.5708]]'"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]          # workspace root (skill-library/skills/<skill>/code/<package>)
sys.path.insert(0, str(ROOT))
from office_chair_tuck import chair_ctl as C  # noqa: E402
from office_chair_tuck import chair_moves as CM  # noqa: E402

from .params import CHAIR_EXTRA  # hyper-parameters: params.py


def run(ctx, moves, log=print):
    yaw = C.status(ctx)['odometry']['yaw']
    for kind, v in moves:
        lost = CM.lost_grip(C.status(ctx))
        if lost:
            return {'ok': False, 'why': f'grip lost before {kind}: {lost}'}
        if kind == 'pull':
            r = C.pull(ctx, v, log=log)
            ctx.value('rby1_gripper', side='right', action='close', current_a=C.GRIP_A)
        elif kind == 'turn':
            yaw = yaw + v
            r = CM.turn_to_relieved(ctx, yaw, log=log)
        elif kind == 'drive':
            r = C.base_drive(ctx, v, step=0.15, speed=0.05, wrist_base=C.wrist(C.status(ctx)), log=log)
        log(f'== {kind} {v}: {r}')
        if not r.get('ok'):
            return {'ok': False, 'why': f'{kind} {v}: {r.get("why")}'}
    return {'ok': True, 'odometry': C.status(ctx)['odometry']}


if __name__ == '__main__':
    from api.runtime import Context
    with Context() as ctx:
        print(json.dumps(run(ctx, json.loads(sys.argv[1])), default=str))
