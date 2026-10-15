"""Chair manipulation primitives for the RB-Y1 (push_chair task).

Measured recipe (2026-10-09, memory/TIPS.md):
- stand square behind the backrest, torso ZERO, chest 0; the backrest top bar ~0.6 m ahead at z≈1.0
- the bar is CURVED: grip it with both hands at its centre ±0.10 m, each gripper yawed to the local tangent,
  pitched -45° (fingers forward-down), open tips' middle (0.20 m along ee -z) on the bar centre
- v4: hand poses (incl. gripper yaw changes) by the controller's whole-body optimal control (oc-hand-reach:
  rotation <= 0.35 rad per call; v1's large Cartesian rotations tripped left_arm_6); Cartesian = translation
- close left 0.4 A then right 0.4 A; a wrist joint over 85 % -> open that side to 4 cm, re-close at 0.25 A
- move the chair with the base (forward/back 3-5 cm steps at 0.03 m/s, turns in 0.15 rad steps) while holding

Every function takes ctx (api.runtime.Context), returns a dict with ok / why / measured values, and stops on the
first unexpected reply. Nothing here decides where the chair should go: the task script does.
"""
import math
import time

import numpy as np
import pinocchio as pin

import geom
import kin

import importlib.util as _ilu
from pathlib import Path as _P
_spec = _ilu.spec_from_file_location('oc_reach', _P(__file__).resolve().parents[2] / 'oc-hand-reach' / 'code' / 'oc_reach.py')
OC = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(OC)

from office_chair_tuck_params import (  # hyper-parameters: office_chair_tuck_params.py
    TORSO_ZERO, PITCH, FINGER, HAND_HALF, GRASP_DEEPER, RELIEVE_PCT, ABORT_PCT, WRIST_ABORT_N, UP_POSE,
    READY_ARM, ROLL_STEP, STANDOFF, PRE_DIST, CLEAR_R,
)
PRE_BACK, PRE_UP = 0.10, 0.04
GRIP_A, GRIP_SOFT_A = 0.4, 0.25


def log_default(*a):
    print(*a, flush=True)


# ---------- robot state helpers ----------
def call(ctx, name, args):
    """Like ctx.value but with the arguments as a dict (rby1_check has an argument named 'tool')."""
    import json
    res = ctx.request('tools/call', {'name': name, 'arguments': args})
    text = next((b.get('text', '') for b in res.get('content', []) if b.get('type') == 'text'), '')
    if res.get('isError'):
        raise RuntimeError('MCP tool failed: ' + text)
    return json.loads(text)


def status(ctx):
    return ctx.value('rby1_status')


def loads(s):
    return {k: float(v.rstrip('%')) for k, v in (s.get('safety', {}).get('highest_torque_load') or {}).items()}


def max_load(s):
    lo = loads(s)
    return max(lo.values()) if lo else 0.0


def wrist(s):
    return {k: np.array(v) for k, v in s['wrist_force_n'].items()}


def quat_pitch_yaw(yaw, pitch=PITCH):
    R = kin.Rz(yaw) @ kin.Ry(pitch)
    q = pin.Quaternion(R)
    return [q.x, q.y, q.z, q.w], R


def finger_dir(yaw, pitch=PITCH):
    return quat_pitch_yaw(yaw, pitch)[1] @ np.array([0, 0, -1.0])


# ---------- perception (geometry) ----------
def observe(ctx, head=None):
    if head is not None:
        ctx.value('rby1_move_joints', component='head', positions=list(head))
    r = ctx.value('rby1_observe')
    return r['frame_id']


def bar_profile(frame, x_range=(0.45, 1.1), y_half=0.45, z_band=(0.95, 1.06), v_max=90):
    """Backrest top bar from depth + colour in the robot frame: per-2.5 cm y bins of the visible (front) surface x
    and the top z. Centre = middle of the flat top (z within 1.5 cm of the max). -> dict or ok False."""
    import cv2
    f = geom.load(frame)
    pts, px, cols = geom.cloud(f, step=1, max_range=3)
    hsv = cv2.cvtColor(cols.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sel = ((hsv[:, 2] < v_max) & (pts[:, 2] > z_band[0]) & (pts[:, 2] < z_band[1]) & (pts[:, 0] > x_range[0])
           & (pts[:, 0] < x_range[1]) & (np.abs(pts[:, 1]) < y_half))
    q = pts[sel]
    if len(q) < 300:
        return {'ok': False, 'why': f'only {len(q)} bar points in the band', 'frame': str(f['dir'])}
    ys, xs, zs = [], [], []
    for yb in np.arange(-y_half, y_half, 0.025):
        s = q[(q[:, 1] >= yb) & (q[:, 1] < yb + 0.025)]
        if len(s) > 15:
            ys.append(yb + 0.0125)
            xs.append(float(np.median(s[:, 0])))
            zs.append(float(np.median(s[:, 2])))
    ys, xs, zs = np.array(ys), np.array(xs), np.array(zs)
    if len(ys) < 8:
        return {'ok': False, 'why': f'bar seen in only {len(ys)} bins', 'frame': str(f['dir'])}
    # robust: the bar is the dominant surface; drop bins far (in x) from the central bins' median and bins whose
    # x jumps from the neighbours (fingers, background)
    order = np.argsort(np.abs(ys - np.median(ys)))
    xc = np.median(xs[order[:max(3, len(order) // 3)]])
    med = np.array([np.median(xs[max(0, i - 2):i + 3]) for i in range(len(xs))])
    keep = (np.abs(xs - xc) < 0.08) & (np.abs(xs - med) < 0.025)
    ys, xs, zs = ys[keep], xs[keep], zs[keep]
    if len(ys) < 6:
        return {'ok': False, 'why': f'bar consistent in only {len(ys)} bins', 'frame': str(f['dir'])}
    flat = zs > zs.max() - 0.015
    yc = float((ys[flat].min() + ys[flat].max()) / 2)
    c = np.polyfit(ys, xs, 2)
    if c[0] < 0:
        return {'ok': False, 'why': f'bar curvature inverted ({c[0]:.2f}): contaminated profile', 'frame': str(f['dir'])}
    return {'ok': True, 'frame': str(f['dir']), 'centre_y': round(yc, 3), 'z_top': round(float(zs.max()) + 0.005, 3),
            'flat': [round(float(ys[flat].min()), 3), round(float(ys[flat].max()), 3)],
            'x_at_centre': round(float(np.polyval(c, yc)), 3), 'poly': [float(v) for v in c],
            'ys': ys.round(3).tolist(), 'xs': np.round(xs, 3).tolist(), 'zs': np.round(zs, 3).tolist()}


def bar_grasp_targets(prof, half=HAND_HALF, finger=FINGER):
    """Per hand: bar point, tangent yaw, quaternion, pre-grasp and grasp ee positions (robot frame)."""
    c = np.array(prof['poly'])
    out = {}
    for side, sgn in (('right', -1), ('left', 1)):
        y = prof['centre_y'] + sgn * half
        x_surf = float(np.polyval(c, y))
        slope = float(np.polyval(np.polyder(c), y))
        yaw = -math.atan(slope)                      # closing axis (1, -slope) ⊥ bar tangent (slope, 1)
        quat, R = quat_pitch_yaw(yaw)
        f = R @ np.array([0, 0, -1.0])
        bar_c = np.array([x_surf + 0.015, y, prof['z_top'] - 0.015])
        grasp = bar_c - (finger - GRASP_DEEPER) * f
        pre = grasp - PRE_BACK * f + np.array([0, 0, PRE_UP])
        out[side] = {'bar': bar_c.round(4).tolist(), 'yaw_deg': round(math.degrees(yaw), 1), 'quat': quat,
                     'pre': pre.round(4).tolist(), 'grasp': grasp.round(4).tolist(), 'f': f.round(4).tolist()}
    return out


def arm_ik(ctx, targets, torso=TORSO_ZERO):
    """targets {side: (pos, quat)} -> {side: joints}: ONE local solve from the current joints (analysis only; the
    robot path uses move_arms_joint = the controller's optimal control). v4 removed v3's seed fallback (current,
    mirrored other arm, ready): searching seeds once returned a flipped-shoulder solution (joint1 -3.1)."""
    s = status(ctx)
    sol = {}
    for side, (pos, quat) in targets.items():
        a, ep, er = kin.ik(side, torso, np.array(s[f'{side}_arm']), pos, quat)
        if ep > 0.002 or er > 0.01:
            raise RuntimeError(f'IK {side} {pos}: pos err {ep:.4f} rot err {er:.4f}')
        sol[side] = [round(float(v), 4) for v in a]
    return sol


def move_arms_joint(ctx, targets, log=log_default, w_rot=5.0, tol_m=0.005, mode='oc'):
    """One or both hands to (pos, quat) in ONE whole-body optimal-control problem (oc-hand-reach.move_hands): the
    torso held where it is, a hand not listed held at its pose, elbow-down posture prior, waypoints within the
    per-call caps (<= 0.20 m, <= 0.35 rad). No local IK, no seed search (v4; the name is kept for callers).
    Tested 2026-10-09: both hands ready -> pitched pre-grasp shape in 3 calls / 23 s, 1.1 mm, 0.01 rad.
    -> oc result dict; raises RuntimeError when the controller refuses or stops."""
    if mode == 'joint':
        r = {'ok': False, 'why': 'mode joint (limit)'}
    else:
        r = OC.move_hands(ctx, {side: {'pos': [float(v) for v in pos], 'quat': [float(v) for v in quat],
                                       'w_rot': w_rot} for side, (pos, quat) in targets.items()}, tol_m=tol_m, log=log)
    if not r['ok'] and w_rot >= 1.0 and ('150%' in str(r['why']) or 'limit' in str(r['why'])):
        # v5 (trial 3): the OC posture prior's first waypoint tripped left_arm_6 165 % in free air (as in trial 2's
        # release / cups pre-grasp) -> one local IK solve from the stopped pose + a joint move (split <= 1.1 rad)
        log(f"  optimal control stopped ({str(r['why'])[:120]}) -> IK + joint move")
        sol = arm_ik(ctx, targets, torso=status(ctx)['torso'])
        s = status(ctx)
        cur = {k: np.array(s[f'{k}_arm']) for k in sol}
        n = max(1, int(math.ceil(max(float(np.max(np.abs(np.array(sol[k]) - cur[k]))) for k in cur) / 1.1)))
        for i in range(1, n + 1):
            tg = {f'{k}_arm': {'positions': (cur[k] + (np.array(sol[k]) - cur[k]) * i / n).round(4).tolist()}
                  for k in cur}
            ctx.value('rby1_move_body', targets=tg)
        return {'ok': True, 'fallback': 'ik_joint', 'joints': sol}
    if not r['ok']:
        raise RuntimeError(f"optimal-control move {sorted(targets)}: {r['why']}")
    return r


def move_hands_cartesian(ctx, deltas, lin=0.03):
    """Translate one or both hands (orientation unchanged) in one controller command, torso pinned."""
    s = status(ctx)
    tg = []
    for side, d in deltas.items():
        e = s[f'{side}_ee']
        tg.append({'link': f'ee_{side}', 'position': (np.array(e['position']) + np.array(d)).round(4).tolist(),
                   'quaternion': e['quaternion']})
    pin_torso = [{'joint': f'torso_{i}', 'position': 0.0} for i in range(6)]
    return ctx.value('rby1_cartesian', component='body', targets=tg, joint_targets=pin_torso,
                     linear_velocity=lin, angular_velocity=0.1)


def hands_up(ctx, log=log_default):
    """Both hands to the 'up' positions (fingers well above the chair). v4: optimal control with a loose
    orientation (w_rot 0.3): at torso zero the exact -45 deg pitch there needs the wrist at its limit (right_arm_5
    reached 1.892 rad and the backend stopped the move, 2026-10-09)."""
    q0, _ = quat_pitch_yaw(0.0)
    return move_arms_joint(ctx, {k: (list(v), q0) for k, v in UP_POSE.items()}, log=log, w_rot=0.3)


def ensure_torso_zero(ctx, log=log_default):
    """The arm IK and the pinned-torso Cartesian moves assume torso ZERO. v3: a grasp started at torso `ready` put
    the pre-grasp 9 cm low and the descent (torso pinned to 0) drove the fingers under the bar (2026-10-09)."""
    s = status(ctx)
    if max(abs(v) for v in s['torso']) > 0.01:
        log(f"  torso {s['torso']} -> zero (precondition)")
        ctx.value('rby1_move_joints', component='torso', preset='zero')


def grasp_bar(ctx, prof, log=log_default):
    """Two-hand grasp of the backrest bar from a fresh bar profile. Starts with hands up/clear and grippers free.
    -> {ok, widths, loads, targets}"""
    ensure_torso_zero(ctx, log=log)
    T = bar_grasp_targets(prof)
    log('  grasp targets', {k: (v['yaw_deg'], v['grasp']) for k, v in T.items()})
    for side in ('right', 'left'):
        ctx.value('rby1_gripper', side=side, action='open')
    # pre-grasp: both hands in one optimal-control problem (v4; v3 moved right then left by searched-seed IK).
    # Two hands + posture prior settle 5-7 mm off (2026-10-09 trial 2): accept 12 mm, the descent is absolute.
    # v5 (trial 3): OC tripped left_arm_6 165 % here -> pre-grasp by one IK solve + joint move (mode='joint')
    move_arms_joint(ctx, {k: (T[k]['pre'], T[k]['quat']) for k in ('right', 'left')}, log=log, tol_m=0.012,
                    mode='joint')
    base = wrist(status(ctx))
    # descend along each hand's finger direction (+ the PRE_UP) in two translation steps to ABSOLUTE targets
    for i in (1, 2):
        s = status(ctx)
        d = {k: (np.array(T[k]['pre']) + (np.array(T[k]['grasp']) - np.array(T[k]['pre'])) * i / 2
                 - np.array(s[f'{k}_ee']['position'])).round(4).tolist() for k in T}
        move_hands_cartesian(ctx, d)
        s = status(ctx)
        dw = {k: float(np.linalg.norm(wrist(s)[k] - base[k])) for k in base}
        if max(dw.values()) > 25:
            return {'ok': False, 'why': f'contact while descending: wrist Δ {dw}', 'targets': T}
    widths = {}
    for side in ('left', 'right'):
        r = ctx.value('rby1_gripper', side=side, action='close', current_a=GRIP_A)
        widths[side] = r['measured_width_m']
        s = status(ctx)
        lo = loads(s)
        hot = {j: v for j, v in lo.items() if v > RELIEVE_PCT}
        if r['measured_width_m'] < 0.008 or not r.get('blocked_by_object'):
            ctx.value('rby1_gripper', side=side, action='open')
            return {'ok': False, 'why': f'{side} closed on air ({r["measured_width_m"]})', 'targets': T}
        hot_roll = [j for j in hot if j.endswith('_6')]
        if hot and not hot_roll:                     # an elbow / wrist-pitch joint: soft re-close of this side
            log(f'  {side} closed: loads {hot} > {RELIEVE_PCT}% -> relieve and re-close soft')
            ctx.value('rby1_gripper', side=side, action='width', width_m=0.04)
            r = ctx.value('rby1_gripper', side=side, action='close', current_a=GRIP_SOFT_A)
            widths[side] = r['measured_width_m']
    rr = relieve_wrist_roll(ctx, log=log)            # v4: the hot wrist roll is often the OTHER hand's
    widths.update(rr.get('widths', {}))
    s = status(ctx)
    return {'ok': max_load(s) < 100, 'why': '' if max_load(s) < 100 else f'loads {loads(s)}', 'widths': widths,
            'loads': loads(s), 'targets': T, 'wrist_base': {k: v.round(1).tolist() for k, v in base.items()}}


def relieve_wrist_roll(ctx, log=log_default):
    """After both grippers closed on the curved bar, a wrist ROLL joint (arm_6, rated 8 N m) can sit at 90-113 %:
    closing the second gripper twists the chair against the first grip, and the tangent yaw from the bar fit is a few
    degrees off (trial 2, 2026-10-10: left_arm_6 108-113 % twice; re-closing softer or relieving the side just closed
    did not help). Fix: open the hot side to 4 cm (load drops to ~40 %), roll its joint 6 by ROLL_STEP in the
    direction the closed grip pushed it (closed - open joint reading; + when that is below noise), re-close firm
    (GRIP_A); if the load did not drop, undo and try the other direction once (a sign test, not a search).
    -> {widths, loads, rolled}"""
    s = status(ctx)
    lo = loads(s)
    hot = sorted((v, j) for j, v in lo.items() if j.endswith('_6') and j.split('_')[0] in ('left', 'right')
                 and v > RELIEVE_PCT)
    if not hot:
        return {'loads': lo}
    before, joint = hot[-1]
    side = joint.split('_')[0]
    q_closed = status(ctx)[f'{side}_arm'][6]
    ctx.value('rby1_gripper', side=side, action='width', width_m=0.04)
    q_open = status(ctx)[f'{side}_arm'][6]
    first = 1.0 if (q_closed - q_open) >= -0.0002 else -1.0
    widths = {}
    for sign in (first, -first):
        ctx.value('rby1_jog', joint=f'{side}_arm_6', mode='relative', value=round(sign * ROLL_STEP, 3))
        g = ctx.value('rby1_gripper', side=side, action='close', current_a=GRIP_A)
        widths[side] = g['measured_width_m']
        after = loads(status(ctx)).get(joint, 0.0)
        log(f'  wrist roll relief: {joint} {before:.0f} % -> roll {sign * ROLL_STEP:+.2f} rad -> {after:.0f} %')
        if after < min(before, RELIEVE_PCT):
            return {'widths': widths, 'loads': loads(status(ctx)), 'rolled': {joint: sign * ROLL_STEP}}
        ctx.value('rby1_gripper', side=side, action='width', width_m=0.04)
        ctx.value('rby1_jog', joint=f'{side}_arm_6', mode='relative', value=round(-sign * ROLL_STEP, 3))
    g = ctx.value('rby1_gripper', side=side, action='close', current_a=GRIP_SOFT_A)
    widths[side] = g['measured_width_m']
    return {'widths': widths, 'loads': loads(status(ctx)), 'rolled': None}


def arms_ready_joint(ctx, max_step=1.1):
    """Both arms to READY_ARM by joint moves split so no joint moves more than max_step per call (backend cap 1.2)."""
    s = status(ctx)
    cur = {k: np.array(s[f'{k}_arm']) for k in READY_ARM}
    n = max(1, int(math.ceil(max(float(np.max(np.abs(np.array(READY_ARM[k]) - cur[k]))) for k in cur) / max_step)))
    for i in range(1, n + 1):
        tg = {f'{k}_arm': {'positions': (cur[k] + (np.array(READY_ARM[k]) - cur[k]) * i / n).round(4).tolist()}
              for k in cur}
        ctx.value('rby1_move_body', targets=tg)


def release(ctx, log=log_default):
    """Open both grippers, back the hands out along their finger axes and lift; then arms to ready by split joint
    moves (v4, trial 2: the optimal-control hands_up from here tripped left_arm_6 at 168 % on its first waypoint)."""
    s = status(ctx)
    for side in ('left', 'right'):
        ctx.value('rby1_gripper', side=side, action='open')
    out = {}
    for side in ('left', 'right'):
        R = kin.R_from_quat(s[f'{side}_ee']['quaternion'])
        f = R @ np.array([0, 0, -1.0])
        out[side] = (-0.10 * f + np.array([0, 0, 0.04])).round(4).tolist()
    move_hands_cartesian(ctx, out)
    arms_ready_joint(ctx)
    return {'ok': True}


# ---------- base motions with the chair held ----------
def _guard(ctx, base, log):
    s = status(ctx)
    ml = max_load(s)
    dw = {k: float(np.linalg.norm(wrist(s)[k] - base[k])) for k in base} if base else {}
    if ml > ABORT_PCT or (dw and max(dw.values()) > WRIST_ABORT_N):
        return False, f'load {loads(s)} wrist Δ {dw}'
    return True, f'max load {ml:.0f}% wrist Δ ' + ' '.join(f'{k}:{v:.0f}' for k, v in dw.items())


def _base_move(ctx, log, **kw):
    """rby1_base_move with one retry on a transient SDK stream error (seen once: 'This stream is expired')."""
    try:
        return ctx.value('rby1_base_move', **kw)
    except RuntimeError as e:
        if 'stream is expired' not in str(e):
            raise
        log('  base move: transient SDK stream error, retrying once')
        time.sleep(1.0)
        return ctx.value('rby1_base_move', **kw)


def base_drive(ctx, distance, step=0.05, speed=0.03, wrist_base=None, log=log_default):
    """Drive straight (±) in steps, checking loads / wrist forces after each step. -> {ok, moved}"""
    moved = 0.0
    n = max(1, int(math.ceil(abs(distance) / step)))
    for i in range(n):
        d = distance / n
        r = _base_move(ctx, log, forward=round(d, 4), max_linear=speed)
        moved += r['measured_by_odometry']['forward']
        ok, msg = _guard(ctx, wrist_base, log)
        log(f'  drive {i + 1}/{n}: moved {moved:+.3f} {msg}')
        if not ok:
            return {'ok': False, 'moved': moved, 'why': msg}
    return {'ok': True, 'moved': moved}


def base_turn(ctx, angle, step=0.15, speed=0.1, wrist_base=None, log=log_default):
    turned = 0.0
    n = max(1, int(math.ceil(abs(angle) / step)))
    for i in range(n):
        r = _base_move(ctx, log, turn=round(angle / n, 4), max_angular=speed)
        turned += r['measured_by_odometry']['turn']
        ok, msg = _guard(ctx, wrist_base, log)
        log(f'  turn {i + 1}/{n}: {turned:+.3f} rad {msg}')
        if not ok:
            return {'ok': False, 'turned': turned, 'why': msg}
    return {'ok': True, 'turned': turned}


# ---------- task-level helpers (added after practice runs P1/P2) ----------
def base_turn_to(ctx, yaw_target, wrist_base=None, log=log_default, tol=0.01):
    """Turn to an absolute odometry yaw (steps land ~5% short: re-measure and correct, at most 3 passes)."""
    out = None
    for _ in range(3):
        yaw = status(ctx)['odometry']['yaw']
        err = math.atan2(math.sin(yaw_target - yaw), math.cos(yaw_target - yaw))
        if abs(err) < tol:
            break
        out = base_turn(ctx, err, wrist_base=wrist_base, log=log)
        if not out['ok']:
            return out
    yaw = status(ctx)['odometry']['yaw']
    return {'ok': True, 'yaw': yaw, 'err': round(math.atan2(math.sin(yaw_target - yaw), math.cos(yaw_target - yaw)), 4)}


def pull(ctx, distance, log=log_default):
    """Pull the held chair backwards: the right gripper is caged (pulling with both grips overloaded right_arm_4
    to 98-113%), the left keeps the grip; the right re-closes softly afterwards."""
    ctx.value('rby1_gripper', side='right', action='width', width_m=0.045)
    r = base_drive(ctx, -abs(distance), step=0.12, speed=0.04, wrist_base=wrist(status(ctx)), log=log)
    ctx.value('rby1_gripper', side='right', action='close', current_a=GRIP_SOFT_A)
    return r


def push(ctx, distance, step=0.10, speed=0.04, stop_force=30.0, log=log_default):
    """Push the held chair forward; stops early when a wrist force rises by stop_force (contact)."""
    base = wrist(status(ctx))
    moved = 0.0
    n = max(1, int(math.ceil(abs(distance) / step)))
    for i in range(n):
        r = _base_move(ctx, log, forward=round(distance / n, 4), max_linear=speed)
        moved += r['measured_by_odometry']['forward']
        s = status(ctx)
        dw = {k: float(np.linalg.norm(wrist(s)[k] - base[k])) for k in base}
        log(f'  push {i + 1}/{n}: {moved:+.3f} m, wrist Δ ' + ' '.join(f'{k}:{v:.0f}' for k, v in dw.items())
            + f', max load {max_load(s):.0f}%')
        if max(dw.values()) > stop_force or max_load(s) > ABORT_PCT:
            return {'ok': False, 'moved': moved, 'why': f'force/load rise: wrist Δ {dw}, loads {loads(s)}'}
    return {'ok': True, 'moved': moved}


def chair_yaw(prof, ends=2):
    """Chair facing yaw in the robot frame from the bar's two visible ends (same x when square).
    Left end further away -> the chair is turned clockwise (negative). None when the bar span is short."""
    ys, xs = np.array(prof['ys']), np.array(prof['xs'])
    if len(ys) < 2 * ends + 2 or ys[-1] - ys[0] < 0.30:
        return None
    xl, yl = xs[-ends:].mean(), ys[-ends:].mean()
    xr, yr = xs[:ends].mean(), ys[:ends].mean()
    return float(-math.atan2(xl - xr, yl - yr))


def lane_change(ctx, lateral, angle=0.61, wrist_base=None, log=log_default):
    """Shift the robot sideways by `lateral` (m, + left) keeping its heading: back up, turn ±angle, drive, turn
    back to the stored yaw. Net forward motion ≈ 0. Sweeps the robot's rear: check the space first."""
    D = abs(lateral) / math.sin(angle)
    back = D * math.cos(angle)
    yaw0 = status(ctx)['odometry']['yaw']
    for step in (lambda: base_drive(ctx, -back, step=0.2, speed=0.05, wrist_base=wrist_base, log=log),
                 lambda: base_turn(ctx, math.copysign(angle, lateral), step=0.2, speed=0.15, wrist_base=wrist_base, log=log),
                 lambda: base_drive(ctx, D, step=0.2, speed=0.05, wrist_base=wrist_base, log=log),
                 lambda: base_turn_to(ctx, yaw0, wrist_base=wrist_base, log=log)):
        r = step()
        if not r['ok']:
            return r
    return {'ok': True, 'lateral': lateral, 'D': round(D, 3)}


def measure(ctx, log=log_default):
    """Hands wide (out of the camera's view of the bar), look at the chair; bar profile + desk front + chair yaw."""
    import perception as P
    ensure_torso_zero(ctx, log=log)
    q0, _ = quat_pitch_yaw(0.0)
    move_arms_joint(ctx, {'left': ([0.36, 0.46, 1.32], q0), 'right': ([0.36, -0.42, 1.32], q0)}, log=log, w_rot=0.3)
    fid = observe(ctx, head=(0.0, 0.66))
    prof = bar_profile(fid)
    desk = P.desk_front(fid)
    yaw = chair_yaw(prof) if prof.get('ok') else None
    return {'ok': bool(prof.get('ok')), 'frame': fid, 'profile': prof, 'desk': desk,
            'chair_yaw': None if yaw is None else round(yaw, 3), 'why': prof.get('why', '')}


def find_chair(ctx, log=log_default, heads=((0.0, 0.5), (-0.45, 0.5), (0.45, 0.5))):
    """Look around with the head until Gemini + depth localize the chair. -> chair_pose dict (base frame)."""
    import perception as P
    last = None
    for h in heads:
        fid = observe(ctx, head=h)
        r = P.chair_pose(ctx, fid, log=log)
        last = r
        if r.get('ok'):
            r['frame_id'] = fid
            return r
    return last or {'ok': False, 'why': 'no observation'}


def _seg_point_dist(a, b, p):
    a, b, p = map(np.asarray, (a, b, p))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(a + t * ab - p))


def plan_approach(chair):
    """Base moves [(kind, value)] from the current pose (origin, heading 0) to stand square behind the chair."""
    col = np.array(chair['column'])
    phi = chair['facing_rad']
    d = np.array([math.cos(phi), math.sin(phi)])
    pre = col - PRE_DIST * d
    axle0 = np.array([0.228, 0.0])
    pre_axle = pre + 0.228 * d
    waypoints = [pre_axle]
    if _seg_point_dist(axle0, pre_axle, col) < CLEAR_R:
        side = np.array([-d[1], d[0]])
        if np.dot(axle0 - col, side) < 0:
            side = -side
        waypoints = [col + 1.2 * side - 0.3 * d, pre_axle]
    moves, heading, pos = [], 0.0, axle0
    for w in waypoints:
        v = w - pos
        h = math.atan2(v[1], v[0])
        moves += [('turn', math.atan2(math.sin(h - heading), math.cos(h - heading))), ('drive', float(np.linalg.norm(v)))]
        heading, pos = h, w
    moves += [('turn', math.atan2(math.sin(phi - heading), math.cos(phi - heading))),
              ('drive', PRE_DIST - STANDOFF)]
    return {'moves': [(k, round(v, 3)) for k, v in moves], 'pre': pre.round(3).tolist(),
            'waypoints': [np.round(w, 3).tolist() for w in waypoints]}


def execute_moves(ctx, moves, log=log_default, look=True):
    """Run planned base moves (robot not holding anything); observe after each drive for the agent's review."""
    frames = []
    for kind, val in moves:
        if abs(val) < 0.01:
            continue
        if kind == 'turn':
            r = base_turn(ctx, val, step=0.6, speed=0.2, log=log)
        else:
            r = base_drive(ctx, val, step=0.5, speed=0.1, log=log)
        if not r['ok']:
            return {'ok': False, 'why': r.get('why'), 'frames': frames}
        if look and kind == 'drive':
            frames.append(observe(ctx, head=(0.0, 0.5)))
    return {'ok': True, 'frames': frames}


# ---------- desk memory (odometry) and transport planning with the chair held ----------
def to_odom(odom, p):
    """Robot-frame point -> odometry frame (odom = {'x','y','yaw'} of the base origin)."""
    c, s = math.cos(odom['yaw']), math.sin(odom['yaw'])
    return [odom['x'] + c * p[0] - s * p[1], odom['y'] + s * p[0] + c * p[1]]


def from_odom(odom, q):
    c, s = math.cos(odom['yaw']), math.sin(odom['yaw'])
    dx, dy = q[0] - odom['x'], q[1] - odom['y']
    return [c * dx + s * dy, -s * dx + c * dy]


def plan_transport(desk_front_centre, desk_yaw_rel, pull_back=0.30, pre_gap=0.80):
    """Moves for the robot holding the chair (column STANDOFF ahead) from its current pose (origin, heading 0) to
    square in front of the desk with the bar `pre_gap` from the desk edge, ready for `tuck`.
    desk_front_centre: robot-frame point on the desk's front edge; desk_yaw_rel: heading that faces the desk."""
    E = np.array(desk_front_centre, float)
    n = np.array([math.cos(desk_yaw_rel), math.sin(desk_yaw_rel)])
    bar_ahead = STANDOFF - 0.25                         # bar front ≈0.25 m behind the column
    base_t = E - (pre_gap + bar_ahead) * n
    axle_t = base_t + 0.228 * n
    axle0 = np.array([0.228 - pull_back, 0.0])
    v = axle_t - axle0
    h = math.atan2(v[1], v[0])
    dist = float(np.linalg.norm(v))
    if abs(h) > math.pi / 2:                           # target behind: reverse instead of turning around
        h -= math.copysign(math.pi, h)
        dist = -dist
    moves = [('drive', -pull_back), ('turn', round(h, 3)), ('drive', round(dist, 3)),
             ('turn_to_rel', round(desk_yaw_rel, 3))]
    return {'moves': moves, 'target_base': base_t.round(3).tolist(), 'heading_to_target': round(h, 3)}
