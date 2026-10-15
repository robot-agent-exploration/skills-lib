"""Small, guarded motion primitives on the MCP robot tools."""
import json
import time

import numpy as np

DOWN = [1.0, 0.0, 0.0, 0.0]  # TCP z along base -z, yaw 0 (fingers close along base y)


def quat_down(yaw):
    """Gripper pointing straight down, rotated by yaw (rad) about base z. [x, y, z, w]."""
    # q = Rz(yaw) * Rx(pi) = [cos(y/2), sin(y/2), 0, 0] in (x, y, z, w) order
    return [float(np.cos(yaw / 2)), float(np.sin(yaw / 2)), 0.0, 0.0]


def status(ctx):
    return ctx.value('franka_status')


def wrench(ctx, n=3, dt=0.05):
    """Mean tcp_wrench over n status reads."""
    w = []
    for _ in range(n):
        w.append(status(ctx)['tcp_wrench'])
        time.sleep(dt)
    return np.mean(np.array(w), axis=0)


class MotionError(RuntimeError):
    """A motion failed in a way that leaves control uncertain or was refused."""


def move(ctx, pos, quat=None, tol_mm=None, timeout_s=None, allow_stall=True, **extra):
    """franka_move_pose. A clean timeout (arm holding, control not uncertain) returns
    {'reached': False, 'stalled': True, 'measured_position': ...} when allow_stall.
    extra: integral=False, settle='time' etc. are passed through."""
    args = {'position': [float(v) for v in pos], **extra}
    if quat is not None:
        args['quaternion'] = [float(v) for v in quat]
    if tol_mm is not None:
        args['tolerance_mm'] = tol_mm
    if timeout_s is not None:
        args['timeout_s'] = timeout_s
    try:
        return ctx.value('franka_move_pose', **args)
    except RuntimeError as e:
        text = str(e)
        i = text.find('{')
        err = json.loads(text[i:]).get('error', {}) if i >= 0 else {}
        if (allow_stall and err.get('type') == 'TimeoutError' and err.get('control_uncertain') is False
                and 'robot_state' in err):
            rs = err['robot_state']
            return {'reached': False, 'stalled': True, 'measured_position': rs['tcp_position'],
                    'measured_quaternion': rs['tcp_quaternion'], 'distance_mm': rs.get('distance_to_target_mm')}
        raise MotionError(text) from e


def touch_down(ctx, xy, z_start, z_min, quat=None, step=0.0015, coarse_until=None, coarse_step=0.004,
               df_thresh=4.0, log=print):
    """Descend at xy from z_start in small steps until fz drops by df_thresh N below baseline.

    Returns dict(ok, z_contact (measured TCP z at first contact step), dfz, steps).
    Backs off 3 mm upward after contact. Stops (ok False) at z_min without contact."""
    move(ctx, [xy[0], xy[1], z_start], quat)
    base = wrench(ctx)
    log(f'touch_down baseline wrench {np.round(base, 2).tolist()}')
    z = z_start
    hist = []
    while z > z_min:
        s = coarse_step if (coarse_until is not None and z > coarse_until) else step
        z = max(z - s, z_min)
        r = move(ctx, [xy[0], xy[1], z], quat, tol_mm=1.5, timeout_s=3)
        w = wrench(ctx, n=2)
        dfz = w[2] - base[2]
        mz = r['measured_position'][2]
        hist.append((z, mz, float(dfz)))
        log(f'  cmd z={z:.4f} meas z={mz:.4f} dfz={dfz:+.2f} dfxy=({w[0]-base[0]:+.2f},{w[1]-base[1]:+.2f})'
            + (' STALLED' if r.get('stalled') else ''))
        if dfz < -df_thresh:
            move(ctx, [xy[0], xy[1], mz + 0.003], quat, tol_mm=1.5, timeout_s=4)
            return {'ok': True, 'z_contact': mz, 'z_cmd': z, 'dfz': float(dfz), 'hist': hist}
    return {'ok': False, 'reason': 'no contact before z_min', 'hist': hist}
