"""Hand moves solved by the RB-Y1 controller's whole-body optimal control (rby1_optimal_control), without any IK
seed, yaw or orientation search on our side.

Why: the earlier skills solved IK locally and searched (office-chair-tuck tried 3 seeds per arm and once took a
flipped-shoulder solution, joint1 -3.1, that made the next Cartesian descent time out; stack-cups searched grasp
orientations; the compositional session searched gripper yaws for a natural elbow). The controller solves one
weighted problem on the robot instead: pose targets for the hands (and the hand that should stay put), joint
targets that hold the torso, and weak joint targets (a posture prior) that pull each moving arm towards an
elbow-down shape. The redundancy is resolved by the weights, continuously from the current joints, so the
solution cannot jump to another IK branch.

    r = move_hands(ctx, {'right': {'pos': [0.5, -0.1, 1.05], 'quat': quat_yaw_pitch(0.2, -0.785)}})
    r['ok'], r['measured']['right']['err_pos_m']

Contract
- start: armed, no fault; any posture. The torso is held where it is unless `torso` is given.
- targets: {side: {'pos': [x, y, z] base frame (m), 'quat': [x, y, z, w] or None (keep the current orientation),
  'w_pos': 0-10 (default 10), 'w_rot': 0-10 (default 5)}}. Sides not listed are held at their current pose.
  A low w_rot (< 1) means "position matters, orientation loosely": the posture prior then picks the wrist (use it
  for hands-up / out-of-view poses: a high hand pitched -45 deg at torso zero needs the wrist at its limit) and
  the final orientation is not checked against tol_rad.
- torso: None (hold: joint targets on torso_1/2/3/5 at the measured values, weight 10; the backend adds torso_0 =
  torso_4 = 0), a 6-vector to reach together with the hands (weight w_torso), or 'free' (no torso targets).
- posture: 'natural' (NATURAL below; per-joint weights PRIOR scaled by w_posture / 0.2), 'current' (the live joints
  of each moving arm: no pull; use it when the arm starts far from NATURAL - trial 2, 2026-10-10: from `ready` or a
  chair-grasp pose the natural prior moved several joints 0.5-0.9 rad in the FIRST waypoint and the backend
  stopped it: left_arm_6 166 % / 168 %, right_arm_5 at its limit), None, or {side: 7-vector}.
- path: split into waypoints within the per-call caps (<= step_m, <= step_rad per target; caps 0.25 m / 1.0 rad):
  linear position, slerp orientation. Each call is checked with rby1_check first.
- end: the hands at the targets within tol_m (default 5 mm) or ok False with why and the measured poses.
- return {ok, why, measured: {side: {pos, quat, err_pos_m, err_rot_rad}}, calls, seconds, loads}.
Nothing here decides where the hands should go; callers keep their own approach geometry.
"""
import math
import time

import numpy as np

NATURAL = {'right': [-0.6, -0.45, 0.0, -1.9, 0.0, 1.2, 0.0],
           'left': [-0.6, 0.45, 0.0, -1.9, 0.0, 1.2, 0.0]}     # elbow-down reference (compositional session)
# posture prior: joint -> weight. Shoulder roll (joint 1) is the one that runs into its limit when both hands come
# close together in front (right_arm_1 reached -0.012 rad with a 0.2 weight and the backend stopped the move,
# 2026-10-09): 1.5 at -/+0.45 keeps it inside; the elbow (joint 3) only gets a weak pull; the upper-arm twist is
# left free so the elbow can swivel. The wrist pitch (joint 5) ran to 1.892 rad (limit) for a high hand pitched -45
# deg (chair hands_up) with only joints 1 / 3 weighted: 0.6 towards 1.2 makes the swivel keep it lower.
# Budget: <= 10 joint targets per call = torso 4 + 3 per moving arm.
PRIOR = {1: 1.5, 3: 0.2, 5: 0.6}
CAP_M, CAP_RAD = 0.25, 1.0      # backend per-call caps per target


# ---------- small geometry helpers (no robot) ----------
def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def quat_from_R(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()


def R_from_quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_yaw_pitch(yaw, pitch):
    """Gripper orientation R = Rz(yaw) Ry(pitch): the fingers (ee -z) point forward-down at pitch < 0 (pitch -pi/2:
    straight down; -pi/4: 45 deg forward-down), turned by yaw about the vertical. -> [x, y, z, w]"""
    return quat_from_R(Rz(yaw) @ Ry(pitch))


def slerp(q0, q1, t):
    q0, q1 = np.array(q0, float), np.array(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)
    return (q / np.linalg.norm(q)).tolist()


def angle_between(q0, q1):
    d = abs(float(np.dot(q0, q1)))
    return 2 * math.acos(min(1.0, d))


def yaw_toward(ctx, side, pos):
    """Closed-form gripper yaw that points the fingers from the shoulder towards `pos` (base frame): the yaw a
    natural forward reach takes. Replaces searching yaws for the smallest joint cost. Uses rby1_fk (no motion)."""
    fk = ctx.value('rby1_fk', frames=[f'link_{side}_arm_1'])
    sh = _frame_pos(fk, f'link_{side}_arm_1')
    return math.atan2(pos[1] - sh[1], pos[0] - sh[0])


def _frame_pos(fk, name):
    fr = fk.get('frames') or fk.get('links') or {}
    v = fr.get(name)
    if isinstance(v, dict):
        return v.get('position')
    return v


# ---------- the motion primitive ----------
def _call(ctx, name, args):
    """ctx.value with a dict of arguments (rby1_check takes an argument named 'tool')."""
    import json
    res = ctx.request('tools/call', {'name': name, 'arguments': args})
    text = next((b.get('text', '') for b in res.get('content', []) if b.get('type') == 'text'), '')
    if res.get('isError'):
        raise RuntimeError(f'{name} failed: {text}')
    return json.loads(text)


def _loads(s):
    return {k: float(str(v).rstrip('%')) for k, v in (s.get('safety', {}).get('highest_torque_load') or {}).items()}


def build_command(s, goal, torso=None, posture='natural', w_posture=0.2, w_torso=10.0, w_hold=10.0):
    """One rby1_optimal_control argument dict from the measured status `s` and per-side goals
    {side: {'pos', 'quat', 'w_pos', 'w_rot'}}; sides not in goal are held at their measured pose."""
    targets = []
    for side in ('right', 'left'):
        e = s[f'{side}_ee']
        if side in goal:
            g = goal[side]
            targets.append({'link': f'ee_{side}', 'position': [round(float(v), 4) for v in g['pos']],
                            'quaternion': [round(float(v), 6) for v in (g.get('quat') or e['quaternion'])],
                            'translation_weight': float(g.get('w_pos', 10.0)),
                            'rotation_weight': float(g.get('w_rot', 5.0))})
        else:
            targets.append({'link': f'ee_{side}', 'position': e['position'], 'quaternion': e['quaternion'],
                            'translation_weight': w_hold, 'rotation_weight': w_hold})
    jt = []
    if torso is None or isinstance(torso, (list, tuple)):
        tv = s['torso'] if torso is None else torso
        for i in (1, 2, 3, 5):
            jt.append({'joint': f'torso_{i}', 'position': round(float(tv[i]), 4), 'weight': w_torso})
    if posture:
        if posture == 'current':      # anchor each moving arm at its live joints: no pull towards NATURAL
            ref = {side: s[f'{side}_arm'] for side in goal}
        else:
            ref = NATURAL if posture == 'natural' else posture
        for side in goal:
            for j, w in PRIOR.items():
                if len(jt) < 10:
                    jt.append({'joint': f'{side}_arm_{j}', 'position': float(ref[side][j]),
                               'weight': w * w_posture / 0.2})
    return {'component': 'body', 'targets': targets, 'joint_targets': jt}


def move_hands(ctx, targets, torso=None, posture='natural', w_posture=0.2, w_torso=10.0, step_m=0.20,
               step_rad=0.35, tol_m=0.005, tol_rad=0.03, max_load=130.0, log=print):
    t0 = time.time()
    s = ctx.value('rby1_status')
    if torso is None:      # hold the torso at its START value: re-reading it per waypoint let a 0.01-0.03 rad drift
        torso_hold = [round(float(v), 4) for v in s['torso']]          # of the soft hold accumulate (2026-10-09)
    start = {side: (np.array(s[f'{side}_ee']['position'], float), s[f'{side}_ee']['quaternion']) for side in targets}
    goal_q = {side: (targets[side].get('quat') or start[side][1]) for side in targets}
    n = 1
    for side, g in targets.items():
        d = float(np.linalg.norm(np.array(g['pos'], float) - start[side][0]))
        a = angle_between(start[side][1], goal_q[side])
        n = max(n, int(math.ceil(d / min(step_m, CAP_M - 0.02))), int(math.ceil(a / min(step_rad, CAP_RAD - 0.1))))
    calls = 0
    for k in range(1, n + 1):
        t = k / n
        goal = {side: {'pos': (start[side][0] + t * (np.array(targets[side]['pos'], float) - start[side][0])).tolist(),
                       'quat': slerp(start[side][1], goal_q[side], t),
                       'w_pos': targets[side].get('w_pos', 10.0), 'w_rot': targets[side].get('w_rot', 5.0)}
                for side in targets}
        if torso is None:
            tv = torso_hold
        elif isinstance(torso, (list, tuple)):
            tv = (np.array(s['torso']) + t * (np.array(torso, float) - np.array(s['torso']))).tolist()
        else:
            tv = torso                                             # 'free'
        cmd = build_command(ctx.value('rby1_status') if k > 1 else s, goal, torso=tv, posture=posture,
                            w_posture=w_posture, w_torso=w_torso)
        chk = _call(ctx, 'rby1_check', {'tool': 'rby1_optimal_control', 'arguments': cmd})
        if not chk.get('accepted'):
            return _result(ctx, targets, False, f"waypoint {k}/{n} refused: {chk.get('reason')}", calls, t0)
        try:
            ctx.value('rby1_optimal_control', **cmd)
        except RuntimeError as e:
            return _result(ctx, targets, False, f'waypoint {k}/{n}: {e}', calls, t0)
        calls += 1
        st = ctx.value('rby1_status')
        lo = _loads(st)
        log(f'  oc waypoint {k}/{n}: ' + ', '.join(
            f"{sd} {np.linalg.norm(np.array(st[f'{sd}_ee']['position']) - np.array(goal[sd]['pos'])) * 1000:.1f} mm"
            for sd in goal) + f' loads {lo}')
        if lo and max(lo.values()) > max_load:
            return _result(ctx, targets, False, f'load {lo} over {max_load} %', calls, t0)
    r = _result(ctx, targets, True, '', calls, t0)
    bad = {sd: m for sd, m in r['measured'].items()
           if m['err_pos_m'] > tol_m or (m['err_rot_rad'] > tol_rad and targets[sd].get('w_rot', 5.0) >= 1.0)}
    if bad:
        r['ok'], r['why'] = False, f'final error above tolerance: {bad}'
    return r


def _result(ctx, targets, ok, why, calls, t0):
    s = ctx.value('rby1_status')
    meas = {}
    for side, g in targets.items():
        e = s[f'{side}_ee']
        meas[side] = {'pos': e['position'], 'quat': e['quaternion'],
                      'err_pos_m': round(float(np.linalg.norm(np.array(e['position']) - np.array(g['pos']))), 4),
                      'err_rot_rad': round(angle_between(e['quaternion'], g.get('quat') or e['quaternion']), 4),
                      'arm': s[f'{side}_arm']}
    return {'ok': ok, 'why': why, 'measured': meas, 'calls': calls, 'seconds': round(time.time() - t0, 1),
            'loads': _loads(s), 'torso': s['torso']}
