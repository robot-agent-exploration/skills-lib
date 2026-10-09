"""Bimanual closed-ball squeeze of a box (both grippers closed into 'balls', centre 0.195 m along -z_ee):
grasp from the floor (squat), grasp from a table top, release (Cartesian at table height, joint space when low).

Gripper orientations are the ones tested by the library skill bimanual-box-squeeze (fingers ~61 deg
forward-down); pre-grasp joints come from our IK (core.kin), the controller's Cartesian moves do the touch."""
import numpy as np
from scipy.spatial.transform import Rotation

from ..core import kin, robot
from ..core.runner import StepFailed
from . import navigation, sensing
from .touch import step_until_contact

SQUAT = [[0, 0.785, -1.571, 0.855, 0, 0], [0, 1.047, -2.094, 1.117, 0, 0], [0, 1.2, -2.4, 1.27, 0, 0]]
TORSO_LOW = SQUAT[-1]                       # lowest + 0.07 lean
BALL_Z_PRE = 0.20                           # balls above the support after the squat (box top 0.263)
BALL_Z = 0.18                               # touch / squeeze height above the support (tested 0.18)
STANDOFF = 0.09                             # ball centre starts this far outside the depth end
MAX_TOUCH_TRAVEL = 0.11
KNEE_Y = 0.205                              # |ee y| below this when squatting: wrist meets the knee link (model)
SQUEEZE_MM = 4                              # extra squeeze after both contacts (~1.4 N/mm)
MIN_SQUEEZE_N = 9.0
Q_EE = {'right': [-0.187, -0.1618, 0.7327, 0.634], 'left': [0.107, -0.2229, -0.4199, 0.8732]}     # floor grasp
SEED = {'right': [-0.38, -0.5064, 0.9074, -0.9777, -0.215, 0.4984, 0.793],
        'left': [-0.4087, 0.6193, -0.9807, -0.8881, 0.4448, 0.3421, -0.103]}
Q_CARRY = {'right': [-0.2127, -0.1838, 0.726, 0.6276], 'left': [0.1222, -0.2528, -0.4153, 0.8653]}   # torso zero
SEED_TABLE = {'right': [-0.6116, -0.5768, 0.7147, -1.4269, -0.4398, 1.1185, 0.9395],
              'left': [-0.6109, 0.8671, -0.8153, -1.3949, 0.7088, 0.8888, -0.2031]}


def _box_yaw(box):
    la = np.array(box['long_axis'])
    return float(np.arctan2(-la[0], la[1]))


def plan_balls(box, torso, z, quats, seeds, standoff=STANDOFF):
    """Arm joints putting the balls `standoff` outside the box ends at height z, gripper orientations `quats`
    turned by the box yaw. -> {ok, right, left, targets} or {ok: False, why}."""
    c, la = np.array(box['center']), np.array(box['long_axis'])
    Rz = Rotation.from_euler('z', _box_yaw(box))
    half = box['size'][0] / 2 + standoff
    q = kin.q_from(torso=torso, right=seeds['right'], left=seeds['left'])
    out = {'targets': {}}
    for side, sgn in (('right', -1), ('left', 1)):
        target = np.array([*(c + sgn * la * half), z])
        R = (Rz * Rotation.from_quat(quats[side])).as_matrix()
        q, ep, er = kin.solve_ball(q, side, target, R)
        if ep > 0.002 or er > 0.01:
            return {'ok': False, 'why': f'IK {side} misses by {ep * 1000:.1f} mm / {er:.3f} rad'}
        out[side] = [round(float(v), 4) for v in kin.arm_q(q, side)]
        out['targets'][side] = target.tolist()
    out['ok'] = True
    return out


def checked_move_body(rt, targets):
    chk = robot.check(rt.ctx, 'rby1_move_body', {'targets': targets})
    if not chk['accepted'] or chk['after']['self_collisions']:
        raise StepFailed(f'posture refused: {chk.get("reason") or chk["after"]["self_collisions"]}')
    rt.v('rby1_move_body', targets=targets)


def touch_and_squeeze(rt, box, knee_y=KNEE_Y, squeeze_side=None):
    """Both balls in along the box's long axis until wrist-force contact, then SQUEEZE_MM more with one hand
    (the one further from the knee, or `squeeze_side`). -> dict with F0 baselines and the squeeze reading."""
    la = np.array([*box['long_axis'], 0.0])
    F0 = robot.wrist_force(rt.ctx, samples=5)
    res = {'F0': {k: v.tolist() for k, v in F0.items()}}
    for side, sign in (('right', 1.0), ('left', -1.0)):
        r = step_until_contact(rt.ctx, side, sign * la, max_travel=MAX_TOUCH_TRAVEL, knee_y=knee_y, log=rt.log)
        res[side] = {k: r[k] for k in ('ok', 'travel', 'dF') if k in r}
        if not r['ok']:
            res.update(ok=False, why=f'{side} touch: {r.get("why")}')
            return res
    s = rt.status()
    y = {k: s[f'{k}_ee']['position'][1] for k in ('right', 'left')}
    res['contact_ee_y'] = y
    res['contact_centre_y'] = (y['right'] + y['left']) / 2
    side = squeeze_side or ('left' if knee_y is None else ('right' if abs(y['right']) > abs(y['left']) else 'left'))
    sign = 1.0 if side == 'right' else -1.0
    if knee_y is not None and abs(y[side]) - SQUEEZE_MM / 1000 < knee_y:
        res.update(ok=False, why=f'no room to squeeze next to the knee: ee y {y}')
        return res
    for extra in (SQUEEZE_MM, 2):
        robot.hand_step(rt.ctx, side, sign * la * extra / 1000, speed=0.01)
        F = robot.wrist_force(rt.ctx, samples=5)
        dF = {k: round(float(np.linalg.norm(F[k] - F0[k])), 1) for k in F}
        rt.log(f'squeeze +{extra} mm with {side}: dF {dF}')
        if min(dF.values()) >= MIN_SQUEEZE_N:
            break
    res.update(squeeze_dF=dF, squeezed_with=side, ok=min(dF.values()) >= MIN_SQUEEZE_N)
    if not res['ok']:
        res['why'] = f'squeeze too weak {dF}'
    return res


def recover_from_squat(rt):
    """Hands empty, torso low: rise with the arms fixed (balls go straight up), then ready."""
    if rt.status()['torso'][1] > 0.9:
        rt.v('rby1_move_joints', component='torso', positions=[0, 0.8, -1.6, 0.87, 0, 0])
    robot.ready_posture(rt.ctx, rt.log)


def grasp_from_floor(rt, name, stance_d=0.47, stance_y=-0.025, y_tol=0.045):
    """Precondition: torso / arms ready, grippers closed, the box `name` on the floor at the grasp stance.
    Squat with the balls outside the ends, touch both ends, squeeze. Leaves the box squeezed at the floor."""
    obj = sensing.relocate(rt, name, radius=0.4)
    probs = navigation.pose_problems(obj, stance_d, stance_y, y_tol)
    if probs:
        raise StepFailed(f'{name} not at the grasp stance: {probs}')
    pre = plan_balls(obj, TORSO_LOW, BALL_Z_PRE, Q_EE, SEED)
    if not pre['ok']:
        raise StepFailed(pre['why'])
    checked_move_body(rt, {'torso': {'positions': robot.TORSO_READY_LEAN}, 'right_arm': {'positions': pre['right']},
                           'left_arm': {'positions': pre['left']}})
    for t in SQUAT:                                     # arms fixed: the balls drop beside the box ends
        rt.v('rby1_move_joints', component='torso', positions=t)
    for side in ('right', 'left'):
        robot.hand_step(rt.ctx, side, [0, 0, BALL_Z - BALL_Z_PRE])
    g = touch_and_squeeze(rt, obj)
    if not g['ok']:
        recover_from_squat(rt)
        raise StepFailed(f'grasp failed (recovered to ready, {name} left on the floor): {g.get("why")}')
    rt.world.held = name
    rt.world.grip = {'F0': g['F0'], 'contact_ee_y': g['contact_ee_y'], 'squeeze_dF': g['squeeze_dF'],
                     'torso': TORSO_LOW, 'depth_centre_y': obj['center'][1]}
    rt.log(f'{name} squeezed {g["squeeze_dF"]}; depth centre y {obj["center"][1]:+.3f} vs touch {g["contact_centre_y"]:+.3f}')
    return {'squeeze_dF': g['squeeze_dF'], 'depth_vs_touch_y': [obj['center'][1], g['contact_centre_y']]}


def grasp_from_surface(rt, name, surface, above=0.11, lift_off=0.08):
    """Precondition: torso zero or ready, arms free, the box `name` on table `surface` in reach (~0.55 ahead).
    Hands above the ends (own IK, joint move), down beside them, touch, squeeze, lift `lift_off`."""
    support = rt.world.objects[surface].get('top_z', 0.68)
    robot.torso_to(rt.ctx, robot.TORSO_ZERO, rt.log)
    obj = sensing.relocate(rt, name, radius=0.4)
    p = plan_balls(obj, robot.TORSO_ZERO, obj['top_z'] + above, Q_CARRY, SEED_TABLE)
    if not p['ok']:
        raise StepFailed(p['why'])
    checked_move_body(rt, {'right_arm': {'positions': p['right']}, 'left_arm': {'positions': p['left']}})
    s = rt.status()
    ball_z = s['right_ee']['position'][2] - kin.BALL_D * Rotation.from_quat(s['right_ee']['quaternion']).as_matrix()[2, 2]
    robot.hands_step(rt.ctx, [0, 0, support + BALL_Z_PRE - ball_z], robot.TORSO_ZERO, speed=0.03)
    robot.hands_step(rt.ctx, [0, 0, BALL_Z - BALL_Z_PRE], robot.TORSO_ZERO, speed=0.02)
    g = touch_and_squeeze(rt, obj, knee_y=None, squeeze_side='left')
    if not g['ok']:
        raise StepFailed(f'grasp on {surface} failed: {g.get("why")}')
    robot.hands_step(rt.ctx, [0, 0, lift_off], robot.TORSO_ZERO, speed=0.02)
    rt.world.held = name
    rt.world.grip = {'F0': g['F0'], 'contact_ee_y': g['contact_ee_y'], 'squeeze_dF': g['squeeze_dF'],
                     'torso': robot.TORSO_ZERO, 'contact_centre_y': g['contact_centre_y']}
    return {'squeeze_dF': g['squeeze_dF'], 'contact_centre_y': g['contact_centre_y']}


def release_cartesian(rt, out=0.025, up=0.08):
    """Open the squeeze at table height: each hand `out` outward along y, then up (arm-alone Cartesian)."""
    robot.hand_step(rt.ctx, 'left', [0.0, out, 0.0])
    robot.hand_step(rt.ctx, 'right', [0.0, -out, 0.0])
    robot.hand_step(rt.ctx, 'left', [0.0, 0.0, up], speed=0.03)
    robot.hand_step(rt.ctx, 'right', [0.0, 0.0, up], speed=0.03)


def release_joint_space(rt, out=0.025, up=0.005):
    """Open the squeeze with one joint move from our IK (for low, stretched arms where the controller's
    Cartesian solver tripped the wrist-roll torque guard): each ee `out` outward along y and `up`."""
    s = rt.status()
    q = kin.q_from(torso=s['torso'], right=s['right_arm'], left=s['left_arm'])
    fr = kin.frames(q, ['ee_right', 'ee_left'])
    goal = {}
    for side, dy in (('right', -out), ('left', out)):
        M = fr[f'ee_{side}']
        p, R = M.translation.copy() + np.array([0, dy, up]), M.rotation.copy()
        q, ep, er = kin.solve_ee(q, side, p, R)
        if ep > 0.002:
            raise StepFailed(f'release IK {side} misses by {ep * 1000:.1f} mm')
        goal[side] = [round(float(v), 4) for v in kin.arm_q(q, side)]
    checked_move_body(rt, {'right_arm': {'positions': goal['right']}, 'left_arm': {'positions': goal['left']}})
