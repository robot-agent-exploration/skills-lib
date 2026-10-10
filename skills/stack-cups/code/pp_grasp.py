"""Grasp plans for one arm (no motion): base-frame end-effector poses per object kind, checked with local IK
(pinocchio) from given joints. The robot tools stay the authority (rby1_check before every move).

Finger model (custom round fingers, measured 2026-10-09 by desk contact with open fingers pointing down):
the open fingertips' lowest point is TIP_LOW = 0.192 m below ee along the fingers; objects are held around
GRIP_C = 0.175 m. (robot-interface's 0.20 / 0.225 put the tips ~3 cm too low: a banana grasp planned with
it closed above the banana.)"""
import math

import _deps  # noqa: F401
import numpy as np
import pinocchio as pin

import kin

TIP_LOW = 0.192
GRIP_C = 0.175
OPEN_HALF = 0.05          # open fingers' inner faces at ±0.05 m along ee x (opening 0.10)
DESK_CLEAR = 0.004        # tips above the support when closing on a lying object (tested 0.003-0.004)
TOUCH_MARGIN = 0.012      # the force-guarded descent may go this far below the planned height (support error)
PRE_UP = 0.09             # pre-grasp above the grasp (top grasps)
BOX_PITCHES = (0, -30, -20, -45)  # fingers down when reachable (near boxes), else tilted forward (deg); tested -30
BOX_GRIP_DOWN = 0.035     # grip centre below the box top (tested: the tips pinched the top flaps)
RIM_DEPTH = 0.024         # tips' lowest point below the rim for a rim pinch (tested 0.024 on 4.4-5 cm bowls)
RIM_IN = 0.010            # wall at that depth, inward of the rim (tapered bowls)
RIM_ANGLES = (180, 157.5, -157.5, 135, -135, 112.5, -112.5, 90, -90)   # rim points tried, deg from +x (180 nearest)


def quat(R):
    q = pin.Quaternion(np.asarray(R))
    return [q.x, q.y, q.z, q.w]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _ik(side, torso, joints, pos, R):
    other = 'left' if side == 'right' else 'right'
    arm, pe, re = kin.ik(side, torso, joints[f'{side}_arm'], list(pos), quat(R), joints[f'{other}_arm'])
    return (pe < 0.003 and re < 0.01), arm, pe


def _fingers(pos, R):
    x = np.asarray(R)[:, 0]
    tip = np.asarray(pos) + TIP_LOW * (np.asarray(R) @ [0, 0, -1.0])
    return [(tip[:2] + OPEN_HALF * x[:2]).round(3).tolist(), (tip[:2] - OPEN_HALF * x[:2]).round(3).tolist()]


def plan_top(obj, side, torso, joints):
    """Lying elongated object: fingers straight down, closing across the local axis at the ridge point; close
    with the tips DESK_CLEAR above the support (the descent's last part is force-guarded to find the support)."""
    x, y = obj['grasp_xy']
    z = obj['support_z'] + TIP_LOW + DESK_CLEAR
    base = wrap(obj['axis_yaw'] + math.pi / 2)
    tried = []
    for yaw in sorted({round(wrap(base), 4), round(wrap(base + math.pi), 4)}, key=abs):
        R = kin.Rz(yaw)
        ok_g, arm_g, e_g = _ik(side, torso, joints, (x, y, z), R)
        ok_p, arm_p, e_p = _ik(side, torso, joints, (x, y, z + PRE_UP), R)
        tried.append((round(math.degrees(yaw), 1), round(e_g * 1000, 1), round(e_p * 1000, 1)))
        if ok_g and ok_p:
            return {'ok': True, 'kind': 'top', 'side': side, 'yaw': round(yaw, 4),
                    'pre': {'pos': [x, y, round(z + PRE_UP, 4)], 'quat': quat(R), 'arm': arm_p.round(4).tolist()},
                    'grasp': {'pos': [x, y, round(z, 4)], 'quat': quat(R), 'arm': arm_g.round(4).tolist()},
                    'approach': [[x, y, round(z + 0.035, 4)], [x, y, round(z + 0.010, 4)]],
                    'touch': {'z_min': round(z - TOUCH_MARGIN, 4), 'step': 0.005, 'back': DESK_CLEAR},
                    'fingers': _fingers((x, y, z + PRE_UP), R), 'why': ''}
    return {'ok': False, 'why': f'no IK for a top grasp at ({x:.3f}, {y:.3f}): drive closer? tried {tried}'}


def plan_box(obj, side, torso, joints):
    """Upright box: fingers pitched forward, closing across its short side, grip centre BOX_GRIP_DOWN below the
    top at the top-face centre; approach along the fingers from 8 cm back and 4 cm up."""
    cx, cy = obj['centre']
    gc = np.array([cx, cy, obj['top_z'] - BOX_GRIP_DOWN])
    base = wrap(obj['long_yaw'] + math.pi / 2)
    tried = []
    for pitch in BOX_PITCHES:
        for yaw in sorted({round(wrap(base), 4), round(wrap(base + math.pi), 4)}, key=lambda a: abs(a - math.pi / 2)):
            R = kin.Ry(math.radians(pitch)) @ kin.Rz(yaw)
            f = R @ np.array([0, 0, -1.0])
            ee = gc - GRIP_C * f
            pre = ee - 0.08 * f + np.array([0, 0, 0.04])
            ok_g, arm_g, e_g = _ik(side, torso, joints, ee, R)
            ok_p, arm_p, e_p = _ik(side, torso, joints, pre, R)
            tried.append((pitch, round(math.degrees(yaw), 1), round(e_g * 1000, 1)))
            if ok_g and ok_p:
                return {'ok': True, 'kind': 'box', 'side': side, 'yaw': round(yaw, 4), 'pitch': pitch,
                        'pre': {'pos': pre.round(4).tolist(), 'quat': quat(R), 'arm': arm_p.round(4).tolist()},
                        'grasp': {'pos': ee.round(4).tolist(), 'quat': quat(R), 'arm': arm_g.round(4).tolist()},
                        'approach': [((pre + ee) / 2).round(4).tolist()], 'touch': None,
                        'fingers': _fingers(pre, R), 'why': ''}
    return {'ok': False, 'why': f'no IK for the box top grasp: tried {tried}'}


RIM_PITCHES = (0, -20, -35)   # fingers tilted forward about a sideways (y) closing axis, for far rims (tested -35)


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rim_rotation(yaw, pitch_deg):
    """Fingers down closing along yaw, tilted forward by |pitch| about the CLOSING AXIS itself, so the clamping
    line stays horizontal (a tilted clamping line lets the cup swing about it: a cup pinched with Ry(-35)Rz(-67.5)
    swung almost horizontal and fell, 2026-10-09)."""
    if not pitch_deg:
        return kin.Rz(yaw)
    best = None
    for sgn in (1, -1):
        R = kin.Rz(yaw) @ Rx(sgn * math.radians(abs(pitch_deg)))
        fx = (R @ np.array([0, 0, -1.0]))[0]
        if best is None or fx > best[0]:
            best = (fx, R)
    return best[1]


def plan_rim(obj, side, torso, joints, rim_in=None, angles=RIM_ANGLES, pitches=RIM_PITCHES, depth=None):
    """Bowl / cup: pinch the wall at a rim point (one finger inside, one outside), fingers down, closing radially;
    the first rim point in `angles` the arm reaches. rim_in: wall inward of the rim at the pinch depth (bowls
    0.010, straighter cups 0.004). depth: fingertips below the rim (0.024 single bowl / cup; a stack of n nested
    cups needs (n-1) x the nesting rise + 0.012 to clamp every wall)."""
    c = np.array(obj['centre'])
    rim_in = obj.get('rim_in', RIM_IN) if rim_in is None else rim_in
    tip_z = obj['rim_z'] - (obj.get('depth', RIM_DEPTH) if depth is None else depth)   # fingertips' lowest point
    tried = []
    for pitch in pitches:
        for ang in angles:
            a = math.radians(ang)
            if pitch and abs(math.sin(a)) < 0.9:          # tilt forward only where the closing axis is sideways
                continue
            p = c + (obj['r'] - rim_in) * np.array([math.cos(a), math.sin(a)])
            for yaw in (wrap(a), wrap(a + math.pi)):
                R = rim_rotation(yaw, pitch)
                f = R @ np.array([0, 0, -1.0])
                ee = np.array([p[0], p[1], tip_z]) - TIP_LOW * f
                ok_g, arm_g, e_g = _ik(side, torso, joints, ee, R)
                if not ok_g:
                    tried.append((pitch, ang, round(e_g * 1000)))
                    continue
                pre = ee + np.array([0, 0, PRE_UP])
                ok_p, arm_p, _ = _ik(side, torso, joints, pre, R)
                if ok_p:
                    return {'ok': True, 'kind': 'rim', 'side': side, 'yaw': round(yaw, 4), 'rim_angle': ang,
                            'pitch': pitch, 'pinch': [round(float(p[0]), 4), round(float(p[1]), 4), round(tip_z, 4)],
                            'pre': {'pos': pre.round(4).tolist(), 'quat': quat(R), 'arm': arm_p.round(4).tolist()},
                            'grasp': {'pos': ee.round(4).tolist(), 'quat': quat(R), 'arm': arm_g.round(4).tolist()},
                            'approach': [(ee + np.array([0, 0, 0.04])).round(4).tolist()], 'touch': None,
                            'fingers': _fingers(pre, R), 'why': ''}
    return {'ok': False, 'why': f'no reachable rim point: tried {tried[:6]}'}


PLANNERS = {'elongated': plan_top, 'box': plan_box, 'round': plan_rim}


def plan(obj, side, torso, joints):
    """obj: a pp_perception description; joints: {'left_arm': [...], 'right_arm': [...]} (IK seeds)."""
    return PLANNERS[obj['kind']](obj, side, torso, joints)
