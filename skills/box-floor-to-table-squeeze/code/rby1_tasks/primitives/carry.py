"""Moving a squeezed object up and down: torso paths with the arms fixed (squeeze preserved exactly),
Cartesian two-hand moves with the torso pinned (squeeze regulated), lift from the floor to carry height,
re-centre / retract while carrying, set down on the floor."""
import numpy as np

from ..core import robot
from ..core.runner import StepFailed
from . import squeeze

SQUEEZE_LOW_N = 8.0         # below: grip at risk -> stop
SQUEEZE_TIGHTEN_N = 11.0    # below: 2 mm tighter (~1.4 N/mm measured)
SQUEEZE_HIGH_N = 30.0       # above: 1 mm looser (lowering the arms raised it 16 -> 24 N)
MAX_TIGHTEN_MM = 8
MAX_LOAD = 1.30             # stop before the backend's 150 % torque stop
EE_TO_BOX_BOTTOM = 0.355    # ee z - box bottom z for the floor grasp at BALL_Z 0.18 (table touchdowns 0.35-0.36)


def squeeze_state(ctx, F0):
    F = robot.wrist_force(ctx)
    s = robot.status(ctx)
    return {'dF': {k: round(float(np.linalg.norm(F[k] - np.asarray(F0[k]))), 1) for k in F},
            'loads': robot.loads(s), 'torso': s['torso'],
            'ee': {k: s[f'{k}_ee']['position'] for k in ('right', 'left')}}


def squeeze_adjust(ctx, torso, dy_left):
    """Left hand dy_left along base y (negative = tighter), right hand held, torso pinned (body command)."""
    return robot.hands_step(ctx, [0, 0, 0], torso, speed=0.01, delta_left=[0, dy_left, 0])


def torso_path(rt, poses, check=True):
    F0 = rt.world.grip['F0']
    rows = []
    for pose in poses:
        rt.v('rby1_move_joints', component='torso', positions=[float(p) for p in pose])
        if not check:
            continue
        st = squeeze_state(rt.ctx, F0)
        rows.append(st)
        rt.log(f'torso {pose}: dF {st["dF"]} loads {st["loads"]}')
        if min(st['dF'].values()) < SQUEEZE_LOW_N:
            raise StepFailed(f'squeeze dropped during the torso move: {st["dF"]}')
        if max(st['loads'].values()) > MAX_LOAD:
            raise StepFailed(f'load too high during the torso move: {st["loads"]}')
    return rows


def arm_move_z(rt, torso, z_target, step=0.04):
    """Both hands to ee z = z_target (up or down) in Cartesian body steps, regulating the squeeze."""
    F0 = rt.world.grip['F0']
    tightened = 0
    for _ in range(60):
        st = squeeze_state(rt.ctx, F0)
        z = min(st['ee']['right'][2], st['ee']['left'][2])
        rt.log(f'ee z {z:.3f}: dF {st["dF"]} loads {st["loads"]}')
        if min(st['dF'].values()) < SQUEEZE_LOW_N:
            raise StepFailed(f'squeeze low {st["dF"]}')
        if max(st['loads'].values()) > MAX_LOAD:
            raise StepFailed(f'load high {st["loads"]}')
        if min(st['dF'].values()) < SQUEEZE_TIGHTEN_N and tightened < MAX_TIGHTEN_MM:
            squeeze_adjust(rt.ctx, torso, -0.002)
            tightened += 2
            rt.log('tightened 2 mm')
            continue
        if max(st['dF'].values()) > SQUEEZE_HIGH_N:
            squeeze_adjust(rt.ctx, torso, +0.001)
            rt.log('opened 1 mm')
            continue
        if abs(z_target - z) < 2e-3:
            return st
        robot.hands_step(rt.ctx, [0, 0, float(np.clip(z_target - z, -step, step))], torso)
    raise StepFailed('arm move did not converge')


def lift_from_floor(rt, carry_z=1.17):
    """Squeezed at the lowest squat -> carry posture: torso up to ready+lean (arms fixed), hands to ee z 0.92,
    torso to zero, hands to carry_z (box bottom ~ carry_z - 0.355). Drivable at the end."""
    if rt.world.held is None:
        raise StepFailed('nothing held')
    torso_path(rt, [[0, 1.1, -2.2, 1.17, 0, 0], [0, 0.9, -1.8, 0.97, 0, 0], [0, 0.7, -1.4, 0.77, 0, 0],
                    robot.TORSO_READY_LEAN])
    arm_move_z(rt, robot.TORSO_READY_LEAN, 0.92)
    torso_path(rt, [[0, 0.26, -0.52, 0.30, 0, 0], robot.TORSO_ZERO])
    st = arm_move_z(rt, robot.TORSO_ZERO, carry_z)
    rt.world.grip['torso'] = robot.TORSO_ZERO
    fid = rt.observe([0.5, 0.0, 0.8])
    return {'squeeze': st['dF'], 'loads': st['loads'], 'frame': fid}


def recentre_and_retract(rt, x_target=0.375):
    """While carrying (torso zero): shift the held object onto the base x axis and pull the hands to x_target."""
    s = rt.status()
    cy = (s['right_ee']['position'][1] + s['left_ee']['position'][1]) / 2
    x = s['right_ee']['position'][0]
    robot.hands_step(rt.ctx, [x_target - x, -cy, 0], robot.TORSO_ZERO, speed=0.02)
    st = squeeze_state(rt.ctx, rt.world.grip['F0'])
    rt.log(f'recentred by {-cy:+.3f}, hands at x {x_target}: {st["dF"]}')
    return {'shift_y': -cy, 'squeeze': st['dF']}


def set_down_on_floor(rt, clearance=0.025):
    """Carry posture -> box on the floor in front: hands to ee z 1.044, torso to ready+lean, hands to 0.70,
    squat until the box bottom is ~`clearance` above the floor (torso_1 from the shoulder geometry,
    capped at 1.2), release with one IK joint move (the Cartesian step down at the squat tripped the
    left wrist-roll guard), rise. The box drops the last ~2-3 cm."""
    arm_move_z(rt, robot.TORSO_ZERO, 1.044)
    torso_path(rt, [[0, 0.26, -0.52, 0.30, 0, 0], robot.TORSO_READY_LEAN])
    arm_move_z(rt, robot.TORSO_READY_LEAN, 0.70)
    torso_path(rt, [[0, 0.7, -1.4, 0.77, 0, 0], [0, 0.9, -1.8, 0.97, 0, 0]])
    s = rt.status()
    ee_z = min(s['right_ee']['position'][2], s['left_ee']['position'][2])
    t1 = s['torso'][1]
    drop = ee_z - EE_TO_BOX_BOTTOM - clearance                 # shoulders z = 0.67 + 0.7 cos(torso_1)
    c = np.clip(np.cos(t1) - drop / 0.7, np.cos(1.2), 1.0)
    t1_new = float(np.arccos(c))
    rt.log(f'ee z {ee_z:.3f} -> squat torso_1 {t1:.3f} -> {t1_new:.3f} (box bottom ~{clearance} m)')
    torso_path(rt, [[0, t1_new, -2 * t1_new, t1_new + 0.07, 0, 0]])
    squeeze.release_joint_space(rt, out=0.025, up=0.005)
    F = robot.wrist_force(rt.ctx, samples=5)
    dF = {k: round(float(np.linalg.norm(F[k] - np.asarray(rt.world.grip['F0'][k]))), 1) for k in F}
    rt.log(f'released: dF {dF}')
    held = rt.world.held
    rt.world.held, rt.world.grip = None, {}
    robot.torso_to(rt.ctx, robot.TORSO_READY_LEAN, rt.log)
    robot.ready_posture(rt.ctx, rt.log)
    if max(dF.values()) > 4.0:
        raise StepFailed(f'{held} may still touch a hand after the release: dF {dF}')
    return {'released_dF': dF, 'torso_1_at_release': t1_new}
