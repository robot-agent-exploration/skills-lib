"""Base moves with the office chair held by both grippers, with load relief (compose task).

Why: turning 90 deg with both grips twisted left_arm_4 to 113 % (2026-10-09, park run): the casters resist the
rotation and the rigid grips take the torque. Relief = open the loaded side to 4 cm (the bar stays caged) and
re-close FIRMLY at GRIP_A 0.4 A (a soft 0.25 A re-close let the chair slip out during the next turn).
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chair_ctl as C  # noqa: E402

RELIEVE_PCT = 90.0       # a wrist / forearm joint above this between turn steps -> relieve that side
ABORT_PCT = 125.0        # still above this after relief -> stop
WRIST_ABORT_N = 55.0


def relieve(ctx, side, log=print):
    ctx.value('rby1_gripper', side=side, action='width', width_m=0.04)
    r = ctx.value('rby1_gripper', side=side, action='close', current_a=C.GRIP_A)   # firm: 0.25 A let it slip
    log(f'  relieved {side}: re-closed at {r["measured_width_m"]:.4f} m')
    return r


def hot_sides(s, pct=RELIEVE_PCT):
    lo = C.loads(s)
    out = {}
    for j, v in lo.items():
        for side in ('left', 'right'):
            if j.startswith(side) and v > pct and j[-1] in '3456':
                out[side] = max(out.get(side, 0), v)
    return out


def turn_to_relieved(ctx, yaw_target, step=0.15, speed=0.1, log=print, tol=0.012):
    """Turn to an absolute odometry yaw holding the chair, relieving a twisted grip between steps."""
    wb = C.wrist(C.status(ctx))
    if lost_grip(C.status(ctx)):
        return {'ok': False, 'why': f'not holding: {lost_grip(C.status(ctx))}'}
    for _ in range(40):
        s = C.status(ctx)
        err = math.atan2(math.sin(yaw_target - s['odometry']['yaw']), math.cos(yaw_target - s['odometry']['yaw']))
        if abs(err) < tol:
            return {'ok': True, 'yaw': s['odometry']['yaw'], 'err': round(err, 4)}
        hot = hot_sides(s)
        for side in hot:
            log(f'  {side} loaded {hot[side]:.0f}% -> relieve')
            relieve(ctx, side, log=log)
            wb = C.wrist(C.status(ctx))
        s = C.status(ctx)
        if C.max_load(s) > ABORT_PCT:
            return {'ok': False, 'why': f'loads after relief {C.loads(s)}'}
        d = math.copysign(min(step, abs(err)), err)
        C._base_move(ctx, log, turn=round(d, 4), max_angular=speed)
        s = C.status(ctx)
        dw = {k: float(abs(C.wrist(s)[k] - wb[k]).max()) for k in wb}
        log(f'  turn: yaw {s["odometry"]["yaw"]:+.3f} (target {yaw_target:+.3f}) max load {C.max_load(s):.0f}% '
            f'{C.loads(s)} wrist d {dw}')
        if max(dw.values()) > WRIST_ABORT_N:
            return {'ok': False, 'why': f'wrist force change {dw}'}
        lost = lost_grip(s)
        if lost:
            return {'ok': False, 'why': f'grip lost: {lost}'}
    return {'ok': False, 'why': 'turn did not converge'}


def lost_grip(s, min_w=0.012):
    """Sides whose gripper closed below min_w (on air: the bar slipped out). 2026-10-09: the chair slipped out of
    two 0.25 A grips during a turn and the loop kept turning for 1.1 rad."""
    w = (s.get('grippers') or {}).get('openings_m') or {}
    return {k: v for k, v in w.items() if v is not None and v < min_w}


if __name__ == '__main__':
    import json
    sys.path.insert(0, str(HERE.parents[3]))
    from api.runtime import Context
    with Context() as ctx:
        if sys.argv[1] == 'turn_to':
            print(json.dumps(turn_to_relieved(ctx, float(sys.argv[2])), default=str))
