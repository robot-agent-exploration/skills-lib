"""Placing a squeezed object on a table: hands forward over the measured edge, lower until the wrist forces
show the table took the weight, release, clear the hands."""
import numpy as np

from ..core import robot
from ..core.runner import StepFailed

from . import carry, squeeze

TOUCH_DFZ_N = 4.0           # vertical (base frame) wrist force change per side = the surface took the object
STEP_DOWN = 0.01


def lower_until_contact(rt, torso, z_min, fast_to=None):
    """Lower both hands (torso pinned) until a side's base-frame Fz changes by > TOUCH_DFZ_N, or ee z_min."""
    if fast_to is not None:
        s = rt.status()
        z = min(s['right_ee']['position'][2], s['left_ee']['position'][2])
        if z - fast_to > 0.002:
            robot.hands_step(rt.ctx, [0, 0, -(z - fast_to)], torso)
    Fh, s = robot.forces_base(rt.ctx)
    rt.log(f"holding Fz {[round(float(v[2]), 1) for v in Fh.values()]} at ee z {s['right_ee']['position'][2]:.3f}")
    while True:
        z = min(s['right_ee']['position'][2], s['left_ee']['position'][2])
        if z - STEP_DOWN < z_min:
            return {'ok': False, 'why': f'no contact down to ee z {z:.3f}', 'ee_z': z}
        try:
            robot.hands_step(rt.ctx, [0, 0, -STEP_DOWN], torso, speed=0.01)
        except RuntimeError as e:
            F, s = robot.forces_base(rt.ctx)
            return {'ok': False, 'stopped': True, 'why': str(e)[:300], 'ee_z': s['right_ee']['position'][2]}
        F, s = robot.forces_base(rt.ctx)
        dFz = {k: round(float(F[k][2] - Fh[k][2]), 1) for k in F}
        rt.log(f"ee z {s['right_ee']['position'][2]:.3f} dFz {dFz} loads {robot.loads(s)}")
        if max(abs(v) for v in dFz.values()) > TOUCH_DFZ_N:
            return {'ok': True, 'ee_z': s['right_ee']['position'][2], 'dFz': dFz}


def place_on_surface(rt, name, surface, edge_margin_in=0.065, ee_to_near_face=0.025, max_forward=0.16):
    """Precondition: carrying `name` (torso zero) at a stance facing `surface` with its edge measured
    (world.notes[surface + '_edge_x']). The object's near face goes edge_margin_in past the edge."""
    if rt.world.held != name:
        raise StepFailed(f'not holding {name}')
    edge = rt.world.notes.get(f'{surface}_edge_x')
    if edge is None:
        raise StepFailed(f'{surface} edge unknown')
    torso = rt.world.grip.get('torso', robot.TORSO_ZERO)
    s = rt.status()
    dx = edge + edge_margin_in + ee_to_near_face - s['right_ee']['position'][0]
    if dx > max_forward:
        raise StepFailed(f'hands would need {dx:.2f} m forward: stance too far from the edge')
    n = int(np.ceil(abs(dx) / 0.05)) if abs(dx) > 0.005 else 0
    for _ in range(n):
        robot.hands_step(rt.ctx, [dx / n, 0, 0], torso)
    st = carry.squeeze_state(rt.ctx, rt.world.grip['F0'])
    rt.log(f'over {surface}: {st["dF"]} loads {st["loads"]}')
    if min(st['dF'].values()) < carry.SQUEEZE_LOW_N:
        raise StepFailed(f'squeeze weak before lowering: {st["dF"]}')
    top = rt.world.notes.get(f'{surface}_top_z', 0.70)
    fid = rt.observe([edge + 0.2, 0.0, top])
    c = lower_until_contact(rt, torso, z_min=0.95, fast_to=max(1.10, top + carry.EE_TO_BOX_BOTTOM + 0.10))
    if not c['ok']:
        raise StepFailed(f'no contact with {surface}: {c}')
    squeeze.release_cartesian(rt)
    rt.world.held, rt.world.grip = None, {}
    robot.hands_step(rt.ctx, [0, 0, 0.10], torso, speed=0.04)
    robot.arms_to_ready(rt.ctx, rt.log)
    return {'contact_ee_z': c['ee_z'], 'dFz': c.get('dFz'), 'frame_before_lowering': fid}
