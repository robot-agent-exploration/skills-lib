"""Grasp plans for one arm (no motion): base-frame end-effector poses per object kind. plan_rim (v2) is CLOSED FORM:
one rim point and yaw from the geometry, no IK and no search; whether the arm gets there is decided by the robot
controller's optimal control when the pose is commanded (pp_motion.move_arm_oc). plan_top / plan_box and the v1
plan_rim_search still check their candidates with local IK (pinocchio). The robot tools stay the authority
(rby1_check before every move).

Finger model (custom round fingers, measured 2026-10-09 by desk contact with open fingers pointing down):
the open fingertips' lowest point is TIP_LOW = 0.192 m below ee along the fingers; objects are held around
GRIP_C = 0.175 m. (robot-interface's 0.20 / 0.225 put the tips ~3 cm too low: a banana grasp planned with
it closed above the banana.)"""
import math

import numpy as np
import pinocchio as pin

from office_chair_tuck import kin

from .params import (  # hyper-parameters: params.py
    TIP_LOW, GRIP_C, OPEN_HALF, DESK_CLEAR, TOUCH_MARGIN, PRE_UP, BOX_PITCHES, BOX_GRIP_DOWN, RIM_DEPTH,
    RIM_IN, RIM_ANGLES, RIM_PITCHES, RIM_TILT_X, YAW_TIE, REACH_M,
)


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


def _arm_frame(side, torso, joints, cur_quat=None):
    """Shoulder (link_<side>_arm_0, base frame) and the hand's current closing-axis yaw (ee x projected on the floor;
    None when unknown or nearly vertical), by local forward kinematics (no IK)."""
    joints = joints or {}
    q = kin.q_from(torso if torso is not None else [0.0] * 6, joints.get('right_arm'), joints.get('left_arm'))
    sh = kin.fk(q, f'link_{side}_arm_0')[0]
    if cur_quat is not None:
        R = np.asarray(kin.R_from_quat(cur_quat))
    elif joints.get(f'{side}_arm') is not None:
        R = kin.fk(q, kin.EE[side])[1]
    else:
        return sh, None
    x = R[:, 0]
    return sh, (math.atan2(x[1], x[0]) if math.hypot(x[0], x[1]) > 0.2 else None)


def plan_rim(obj, side, torso, joints, rim_in=None, angles=RIM_ANGLES, pitches=RIM_PITCHES, depth=None,
             cur_quat=None):
    """Bowl / cup: pinch the wall at one rim point (one finger inside, one outside), closing radially. CLOSED FORM
    (v2), no IK and no search:
    - rim point: the one facing the hand's shoulder (link_<side>_arm_0), on the line from the cup centre towards the
      shoulder = the rim point nearest to that arm;
    - yaw (closing axis): radial, a or a + pi, whichever is nearer the hand's current closing-axis yaw (from `joints`
      or cur_quat): the smaller wrist turn. On a tie (within YAW_TIE, e.g. the sideways tilted pinch from a hand
      facing forward) or with no current yaw: the one nearer the shoulder-to-pinch bearing (a forward reach; for the
      fingers-down pinch that bearing IS the radial direction);
    - pitch: fingers straight down, unless that rim point is more than RIM_TILT_X ahead of the base origin and
      `pitches` allows a tilt: then pitches[-1] about the closing axis (rim_rotation: the clamping line stays
      horizontal) at the SIDEWAYS rim point (+-90 deg) on the shoulder's side - only a sideways closing axis lets a
      tilt about it point the fingers forward (v1 tilted only there too);
    - ok False only when the grasp pose is beyond the arm's reach (more than REACH_M from the shoulder, the backend's
      own rule). Whether the arm gets there is decided by the controller: a refused / failed optimal-control move.
    rim_in: wall inward of the rim at the pinch depth (bowls 0.010, straighter cups 0.004). depth: fingertips below
    the rim (0.024 single bowl / cup; a stack of n nested cups needs (n-1) x the nesting rise + 0.012 to clamp every
    wall). pitches=(0,): fingers down only. `angles` is ignored (kept for callers; plan_rim_search is the v1 search
    over it). Same keys as v1; 'arm' is None (the controller chooses the joints)."""
    c = np.array(obj['centre'][:2], float)
    rim_in = obj.get('rim_in', RIM_IN) if rim_in is None else rim_in
    tip_z = obj['rim_z'] - (obj.get('depth', RIM_DEPTH) if depth is None else depth)   # fingertips' lowest point
    rr = obj['r'] - rim_in
    sh, cur_yaw = _arm_frame(side, torso, joints, cur_quat)
    a = math.atan2(sh[1] - c[1], sh[0] - c[0])                  # rim point facing the shoulder
    tilt = pitches[-1] if pitches else 0
    if tilt and (0 not in pitches or c[0] + rr * math.cos(a) > RIM_TILT_X):
        a = math.copysign(math.pi / 2, sh[1] - c[1])            # sideways rim point on the shoulder's side
    else:
        tilt = 0
    p = c + rr * np.array([math.cos(a), math.sin(a)])
    yaws = (wrap(a), wrap(a + math.pi))
    bearing = math.atan2(p[1] - sh[1], p[0] - sh[0])
    d = [abs(wrap(y - (bearing if cur_yaw is None else cur_yaw))) for y in yaws]
    yaw = yaws[int(np.argmin(d))] if abs(d[0] - d[1]) > YAW_TIE else min(yaws, key=lambda y: abs(wrap(y - bearing)))
    R = rim_rotation(yaw, tilt)
    f = R @ np.array([0, 0, -1.0])
    ee = np.array([p[0], p[1], tip_z]) - TIP_LOW * f
    pre = ee + np.array([0, 0, PRE_UP])
    reach = float(np.linalg.norm(ee - sh))
    if reach > REACH_M:
        return {'ok': False, 'why': f'{side} hand: rim point ({p[0]:.3f}, {p[1]:.3f}) puts the hand {reach:.2f} m '
                                     f'from the shoulder (reach {REACH_M} m)', 'reach_m': round(reach, 3)}
    return {'ok': True, 'kind': 'rim', 'side': side, 'yaw': round(yaw, 4), 'rim_angle': round(math.degrees(a), 2),
            'pitch': tilt, 'pinch': [round(float(p[0]), 4), round(float(p[1]), 4), round(tip_z, 4)],
            'pre': {'pos': pre.round(4).tolist(), 'quat': quat(R), 'arm': None},
            'grasp': {'pos': ee.round(4).tolist(), 'quat': quat(R), 'arm': None},
            'approach': [(ee + np.array([0, 0, 0.04])).round(4).tolist()], 'touch': None,
            'fingers': _fingers(pre, R), 'reach_m': round(reach, 3), 'why': ''}


def plan_rim_search(obj, side, torso, joints, rim_in=None, angles=RIM_ANGLES, pitches=RIM_PITCHES, depth=None):
    """v1 (unused by default): pinch the wall at a rim point, fingers down, closing radially; the first rim point
    in `angles` (x pitches x two yaws) for which local IK reaches the grasp and the pre-grasp. rim_in / depth as in
    plan_rim."""
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
    """obj: a pp_perception description; joints: {'left_arm': [...], 'right_arm': [...]} (current joints: IK seeds
    for plan_top / plan_box, the current hand yaw for plan_rim)."""
    return PLANNERS[obj['kind']](obj, side, torso, joints)


if __name__ == '__main__':     # offline self-test of the closed-form plan_rim, no robot: python -m stack_cups.pp_grasp
    import sys
    MID = [0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0]
    READY = {'right_arm': [-0.7854, -0.5236, 0.0, -1.5708, 0.0, 0.7854, 0.0],
             'left_arm': [-0.7854, 0.5236, 0.0, -1.5708, 0.0, 0.7854, 0.0]}
    CLEAR_Q = quat(kin.Ry(-math.pi / 4))                 # pp_motion.clear_view pose: pitched -45, yaw 0 (after scan)
    CASES = [((0.42, 0.04), 'left'), ((0.399, 0.276), 'left'), ((0.42, 0.04), 'right'), ((0.40, -0.20), 'right'),
             ((0.35, 0.30), 'left'), ((0.56, 0.10), 'left'), ((0.56, -0.12), 'right')]
    bad = []
    for (cx, cy), side in CASES:
        obj = {'centre': [cx, cy], 'r': 0.047, 'rim_z': 0.848, 'rim_in': 0.004}
        for pitches in ((0,), RIM_PITCHES):
            p = plan_rim(obj, side, MID, READY, pitches=pitches, cur_quat=CLEAR_Q)
            tag = f'{side:5s} cup ({cx:.3f}, {cy:+.3f}) pitches {pitches}'
            if not p['ok']:
                print(tag, 'NOT OK', p['why'])
                bad.append(tag)
                continue
            R = kin.R_from_quat(p['grasp']['quat'])
            f, x = R @ np.array([0, 0, -1.0]), R[:, 0]
            pin_ = np.array(p['pinch'])
            sh = kin.fk(kin.q_from(MID), f'link_{side}_arm_0')[0]
            ring = np.array([cx, cy]) + 0.043 * np.c_[np.cos(np.linspace(-math.pi, math.pi, 721)),
                                                       np.sin(np.linspace(-math.pi, math.pi, 721))]
            checks = {
                'tip on pinch': np.linalg.norm(np.array(p['grasp']['pos']) + TIP_LOW * f - pin_) < 1e-3,
                'pinch on wall': abs(np.linalg.norm(pin_[:2] - [cx, cy]) - 0.043) < 1e-3,
                'closing axis horizontal + radial': abs(x[2]) < 1e-6 and abs(x[0] * (pin_[1] - cy) - x[1] * (pin_[0] - cx)) < 1e-3,
                'small wrist turn': abs(wrap(p['yaw'] - 0.0)) <= math.pi / 2 + 1e-3,     # current yaw 0 (CLEAR_Q)
                'pre 9 cm above': np.allclose(np.array(p['pre']['pos']) - p['grasp']['pos'], [0, 0, PRE_UP], atol=1e-3),
            }
            if p['pitch']:
                checks['tilted forward, sideways axis'] = f[0] > 0.5 and abs(x[0]) < 1e-6
            else:
                checks['nearest rim point to the shoulder'] = (np.linalg.norm(pin_[:2] - sh[:2])
                                                               <= np.linalg.norm(ring - sh[:2], axis=1).min() + 1e-3)
                checks['far rims tilt when allowed'] = len(pitches) == 1 or pin_[0] <= RIM_TILT_X
            ik = [kin.ik(side, MID, READY[f'{side}_arm'], p[k]['pos'], p[k]['quat'], READY[f"{'left' if side == 'right' else 'right'}_arm"])
                  for k in ('pre', 'grasp')]
            v1 = plan_rim_search(obj, side, MID, READY, pitches=pitches)
            fails = [k for k, v in checks.items() if not v]
            bad += [f'{tag}: {k}' for k in fails]
            print(f"{tag}: rim {p['rim_angle']:7.2f} yaw {math.degrees(p['yaw']):7.1f} pitch {p['pitch']:3d} "
                  f"reach {p['reach_m']:.3f} | local-IK oracle pre/grasp {ik[0][1] * 1000:.1f}/{ik[1][1] * 1000:.1f} mm "
                  f"| v1 search: {(v1['rim_angle'], round(math.degrees(v1['yaw'])), v1['pitch']) if v1['ok'] else 'none'} "
                  f"| {'PASS' if not fails else 'FAIL ' + str(fails)}")
    obj = {'centre': [0.42, 0.04], 'r': 0.047, 'rim_z': 0.848, 'rim_in': 0.004}
    pj, pn = plan_rim(obj, 'left', MID, READY, pitches=(0,)), plan_rim(obj, 'left', MID, None, pitches=(0,))
    far = plan_rim({'centre': [0.75, 0.0], 'r': 0.047, 'rim_z': 0.848}, 'left', MID, READY, pitches=(0,))
    print(f"current yaw from the ready joints: yaw {math.degrees(pj['yaw']):.1f}; no joints (bearing): yaw "
          f"{math.degrees(pn['yaw']):.1f}; cup 0.75 m ahead: ok {far['ok']} ({far['why']})")
    if not (pj['ok'] and pn['ok'] and abs(wrap(pn['yaw'] - pj['yaw'])) < 1e-3 and not far['ok']):
        bad.append('joints / no-joints / reach cases')
    print('SELFTEST', 'PASS' if not bad else f'FAIL {bad}')
    sys.exit(1 if bad else 0)
