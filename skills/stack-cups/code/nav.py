"""Base routes in the odometry frame with the axle model: in-place turns pivot about the drive axle 0.228 m ahead of
the base origin; straight drives move the axle along the heading."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'push_chair'))
import chair_ctl as C  # noqa: E402

AXLE = 0.228
REAR_R = 0.62        # rear sweep radius about the axle


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def heading_vec(yaw):
    return np.array([math.cos(yaw), math.sin(yaw)])


def plan_pose(odom, target_xy, target_yaw, reverse=0.0, turn_dir=None):
    """Moves [(kind, value)] from odom {x, y, yaw} to the base origin at target_xy with heading target_yaw:
    optional straight reverse, turn about the axle to face the target axle point, drive, turn to target_yaw.
    turn_dir 'left' / 'right' forces the direction of the first turn (the rear swings the other way)."""
    yaw0 = odom['yaw']
    b0 = np.array([odom['x'], odom['y']]) - reverse * heading_vec(yaw0)
    a0 = b0 + AXLE * heading_vec(yaw0)
    a1 = np.asarray(target_xy, float) + AXLE * heading_vec(target_yaw)
    v = a1 - a0
    dist = float(np.linalg.norm(v))
    h = math.atan2(v[1], v[0]) if dist > 1e-3 else yaw0
    t1 = wrap(h - yaw0)
    if turn_dir == 'left' and t1 < 0:
        t1 += 2 * math.pi
    if turn_dir == 'right' and t1 > 0:
        t1 -= 2 * math.pi
    moves = []
    if reverse:
        moves.append(('drive', -round(reverse, 3)))
    moves += [('turn', round(t1, 3)), ('drive', round(dist, 3)), ('turn_to', round(target_yaw, 4))]
    return {'moves': moves, 'axle_start': a0.round(3).tolist(), 'axle_end': a1.round(3).tolist(),
            'heading': round(h, 3)}


def execute(ctx, moves, log=print, look=None):
    """Run planned moves; 'turn_to' is an absolute odometry yaw (corrected for the ~5 % short turns)."""
    frames = []
    for kind, val in moves:
        if kind == 'turn_to':
            r = C.base_turn_to(ctx, val, log=log)
        elif abs(val) < 0.01:
            continue
        elif kind == 'turn':
            r = C.base_turn(ctx, val, step=0.4, speed=0.2, log=log)
        else:
            r = C.base_drive(ctx, val, step=0.4, speed=0.1, log=log)
        if not r.get('ok', True):
            return {'ok': False, 'why': r.get('why'), 'frames': frames}
        if look and kind == 'drive':
            frames.append(C.observe(ctx, head=look))
    return {'ok': True, 'frames': frames, 'odom': C.status(ctx)['odometry']}
