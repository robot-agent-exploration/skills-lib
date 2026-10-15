"""Reset helpers: put a held object down and line the cups up.

    python <skill>/code/arrange.py [--object-at X Y] [--home]     (from the workspace root)
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))                       # workspace root (api/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_the_cube import controller as C  # noqa: E402

from find_the_cube.params import ROW_X, ROW_Y  # hyper-parameters: find_the_cube/params.py


def place_object(ctx, xy):
    """Carry the held object to xy and set it down (keeps the current gripper yaw)."""
    st = ctx.value('franka_status')
    q = st['tcp_quaternion']
    C.move_far(ctx, [xy[0], xy[1], 0.25], q)
    C.move(ctx, [xy[0], xy[1], C.TABLE_Z + 0.025], q, tol=6, accept_mm=10)
    ctx.value('franka_gripper', action='open')
    C.wait_gripper(ctx)
    C.move(ctx, [xy[0], xy[1], 0.25], q)


def locate_cups(ctx):
    C.goto_watch(ctx)
    fr = C.observe(ctx)
    n = C.count_cups(fr['image'])
    centres, _ = C.cup_clusters(fr['image'], k=n)
    pc, valid, us, vs = C.backproject(fr, step=1)
    cups = []
    for u, v in centres:
        u, v = int(u), int(v)
        win = pc[v - 12:v + 13, u - 12:u + 13].reshape(-1, 3)
        ok = valid[v - 12:v + 13, u - 12:u + 13].reshape(-1) & (win[:, 2] > C.CUP_TOP_MIN_Z)
        if ok.sum() > 20:
            cups.append(np.median(win[ok], 0).tolist())
    return cups


def line_up(ctx, obstacles=()):
    """Move every cup onto a slot of a straight row (least total travel), nearest moves first."""
    cups = locate_cups(ctx)
    slots = [(ROW_X, y) for y in ROW_Y][:len(cups)]
    best = min(itertools.permutations(range(len(slots))),
               key=lambda perm: sum(np.hypot(cups[i][0] - slots[j][0], cups[i][1] - slots[j][1])
                                    for i, j in enumerate(perm)))
    moves = sorted(((i, slots[j]) for i, j in enumerate(best)),
                   key=lambda m: np.hypot(cups[m[0]][0] - m[1][0], cups[m[0]][1] - m[1][1]))
    done = []
    for i, slot in moves:
        cup = cups[i]
        if np.hypot(cup[0] + 0.02 - slot[0], cup[1] - slot[1]) < 0.015:   # watch-pose x reads ~2 cm short
            done.append({'cup': cup, 'slot': slot, 'moved': False})
            continue
        others = [c for k, c in enumerate(cups) if k != i] + [list(o) for o in obstacles]
        r = C.remove_cup(ctx, others + [cup], cup, dest=slot)
        C.log('cup', np.round(cup[:2], 3).tolist(), '->', slot, r.get('ok'), r.get('why'))
        if not r.get('ok'):
            return {'ok': False, 'why': f'moving cup {np.round(cup[:2], 3).tolist()}: {r.get("why")}', 'done': done}
        cups[i] = [slot[0], slot[1], cup[2]]
        done.append({'cup': cup, 'slot': slot, 'moved': True})
    return {'ok': True, 'done': done}


if __name__ == '__main__':
    from api.runtime import Context
    ap = argparse.ArgumentParser()
    ap.add_argument('--object-at', nargs=2, type=float)
    ap.add_argument('--home', action='store_true')
    args = ap.parse_args()
    with Context() as ctx:
        obstacles = []
        if args.object_at:
            place_object(ctx, args.object_at)
            obstacles.append([args.object_at[0], args.object_at[1], 0.12])
        r = line_up(ctx, obstacles)
        if args.home and r['ok']:
            ctx.value('franka_gripper', action='open')
            ctx.value('franka_move_joints', preset='home')
    print(json.dumps(r, default=str))
