"""Move the held chair along a base route with guards: pull (right caged), turns with firm relief and grip-loss
checks, drives with load / wrist guards. Usage: python chair_route.py <moves json> e.g. '[["pull",0.7],["turn",-1.5708],["drive",1.42],["turn",1.5708]]'"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]          # workspace root (skill-library/skills/<skill>/code)
sys.path.insert(0, str(HERE.parents[1] / 'office-chair-tuck/code'))
sys.path.insert(0, str(ROOT))
import chair_ctl as C  # noqa: E402
import chair_moves as CM  # noqa: E402

CHAIR_EXTRA = (0.50, -0.36, 1.24, 0.36)   # held chair box (x0, y0, x1, y1) in the base frame for sweep_check


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
