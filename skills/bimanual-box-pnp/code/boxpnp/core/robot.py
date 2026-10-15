"""Thin wrappers around the robot tools (api.runtime.Context) and the body constants every primitive uses."""
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from ..params import TORSO_READY, TORSO_ZERO, TORSO_READY_LEAN, ARM_READY, MAX_FORWARD_CALL, TORSO_MAX_STEP  # hyper-parameters: boxpnp/params.py
AXLE_X = 0.228          # the base turns in place about this point (base x)
BUMPER_X = 0.304


def status(ctx):
    return ctx.value('rby1_status')


def check(ctx, tool, arguments):
    """rby1_check (its `tool` argument collides with Context.value's own parameter name)."""
    result = ctx.request('tools/call', {'name': 'rby1_check', 'arguments': {'tool': tool, 'arguments': arguments}})
    text = next(b['text'] for b in result['content'] if b.get('type') == 'text')
    if result.get('isError'):
        raise RuntimeError('MCP tool failed: ' + text)
    return json.loads(text)


def look(ctx, point):
    """Aim the head; False when the head cannot point there. A ~1 deg short stop counts as done."""
    try:
        ctx.value('rby1_look_at', point=[float(p) for p in point])
    except RuntimeError as e:
        if 'cannot point' in str(e):
            return False
        if 'did not reach' not in str(e):
            raise
    return True


def observe(ctx):
    return ctx.value('rby1_observe')['frame_id']


def observe_full(ctx):
    """-> (frame_id, odometry at capture)"""
    r = ctx.value('rby1_observe')
    return r['frame_id'], r['odometry']


def _base_call(ctx, log, **kw):
    """One rby1_base_move. Any error (refusal, stop, backend error) propagates: the run stops and reports
    (an automatic retry hid a misbehaving backend in run A2)."""
    return ctx.value('rby1_base_move', **kw)


def base_move(ctx, log, forward=0.0, turn=0.0, slow=False):
    """Straight move (split into <= 0.8 m calls), then an in-place turn. Returns odometry-measured motions."""
    out = []
    remaining = float(forward)
    while abs(remaining) > 1e-3:
        step = float(np.clip(remaining, -MAX_FORWARD_CALL, MAX_FORWARD_CALL))
        r = _base_call(ctx, log, forward=step, max_linear=0.08 if slow else 0.12)
        out.append(r['measured_by_odometry'])
        log(f"base forward {step:+.3f} -> {r['measured_by_odometry']}")
        remaining -= step
    if abs(turn) > 1e-3:
        r = _base_call(ctx, log, turn=float(turn), max_angular=0.2)
        out.append(r['measured_by_odometry'])
        log(f"base turn {turn:+.3f} -> {r['measured_by_odometry']}")
    return out


def torso_to(ctx, target, log=print):
    """Torso joint move split so no joint changes more than TORSO_MAX_STEP per call."""
    cur = np.array(status(ctx)['torso'])
    target = np.array(target, float)
    n = max(1, int(np.ceil(np.abs(target - cur).max() / TORSO_MAX_STEP)))
    for k in range(1, n + 1):
        ctx.value('rby1_move_joints', component='torso', positions=(cur + (target - cur) * k / n).tolist())
    log(f'torso -> {np.round(target, 3).tolist()}')


def arms_to_ready(ctx, log=print, max_step=1.1):
    """Both arms to the ready preset in joint space, split so no joint moves more than max_step per call."""
    for _ in range(4):
        s = status(ctx)
        cur = {sd: np.array(s[f'{sd}_arm']) for sd in ('right', 'left')}
        diff = max(float(np.abs(cur[sd] - ARM_READY[sd]).max()) for sd in cur)
        if diff < 2e-3:
            return
        f = min(1.0, max_step / diff)
        ctx.value('rby1_move_body', targets={f'{sd}_arm': {'positions': (cur[sd] + (np.array(ARM_READY[sd]) - cur[sd]) * f).tolist()}
                                             for sd in cur})
        log(f'arms towards ready ({f:.2f} of {diff:.2f} rad)')
        if f >= 1.0:
            return


def ready_posture(ctx, log=print):
    s = status(ctx)
    if np.abs(np.array(s['torso']) - TORSO_READY).max() > 1.1:
        torso_to(ctx, [0, 0.8, -1.6, 0.87, 0, 0], log)
    arms_to_ready(ctx, log)
    torso_to(ctx, TORSO_READY, log)


def wrist_force(ctx, samples=3, dt=0.05):
    """Mean wrist force per side over a few status reads (N, sensor frame = ee orientation)."""
    acc = {'right': [], 'left': []}
    for _ in range(samples):
        s = status(ctx)
        for side in acc:
            acc[side].append(s['wrist_force_n'][side])
        time.sleep(dt)
    return {k: np.mean(v, axis=0) for k, v in acc.items()}


def forces_base(ctx):
    """Wrist forces rotated into the base frame, plus the status."""
    F = wrist_force(ctx)
    s = status(ctx)
    return {side: Rotation.from_quat(s[f'{side}_ee']['quaternion']).as_matrix() @ F[side] for side in F}, s


def loads(s):
    return {k: float(v.rstrip('%')) / 100 for k, v in s['safety']['highest_torque_load'].items()}


def hand_step(ctx, side, delta, speed=0.02):
    """One hand straight by `delta` (base frame), orientation kept (controller Cartesian, arm alone)."""
    return ctx.value('rby1_cartesian', component=f'{side}_arm', linear_velocity=speed,
                     targets=[{'link': f'ee_{side}', 'position': [float(d) for d in delta], 'relative': True}])


def torso_pins(torso):
    return [{'joint': f'torso_{i}', 'position': float(v)} for i, v in enumerate(torso)]


def hands_step(ctx, delta, torso, speed=0.02, delta_left=None):
    """Both hands in one Cartesian body command with the torso pinned (no arm-alone reach limit)."""
    dr = [float(x) for x in delta]
    dl = dr if delta_left is None else [float(x) for x in delta_left]
    return ctx.value('rby1_cartesian', component='body', linear_velocity=speed,
                     targets=[{'link': 'ee_right', 'position': dr, 'relative': True},
                              {'link': 'ee_left', 'position': dl, 'relative': True}],
                     joint_targets=torso_pins(torso))
