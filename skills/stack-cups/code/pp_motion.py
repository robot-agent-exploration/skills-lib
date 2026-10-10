"""Motion primitives for desk -> bucket pick & place on the RB-Y1 (robot armed). One function per primitive:
primitive(ctx, ..., log=print) -> {ok, measured, why, frames[, checkpoint]}. Each re-observes what it acts on and
stops in a safe state (holding still, nothing pressed) when a check fails.

Tested values (2026-10-09, torso mid, custom round fingers): grip 0.5 A; touch-down 5 mm steps, contact = wrist
force change > 5 N; first lift 4 cm then a depth rise check; carry poses below; gentle drop: hand pitched -30°,
torso lowest, ee (bucket_x - 0.09, bucket_y, 0.69) -> object just above the rim (0.36 m), open."""
import math

import _deps  # noqa: F401
import numpy as np

import chair_ctl as C
import kin
import nav
import pp_grasp as G
import pp_perception as PP
import pp_vis

CLOSE_A = 0.5
TOUCH_N = 5.0
STOP_N = 8.0
LIFT_CHECK = 0.04
HELD_RISE = 0.02
CARRY = {'left': (0.40, 0.30, 1.12), 'right': (0.40, -0.30, 1.12)}
CLEAR = {'left': (0.34, 0.33, 1.15), 'right': (0.34, -0.33, 1.15)}     # pitched -45°: arms out of the camera view
DROP_DX, DROP_Z, DROP_PITCH = 0.09, 0.69, -30
TORSO = {'mid': [0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0], 'lowest': [0.0, 1.2, -2.4, 1.2, 0.0, 0.0]}
BUCKET_DIST = 0.62        # bucket centre ahead of the base origin at the drop stance (bumper 0.304, near rim 0.47)
DESK_EDGE = 0.215         # desk front edge ahead of the base origin for reaching (bumper under the top; torso clear)
SIGN = {'left': 1.0, 'right': -1.0}


def _res(ok, why='', **kw):
    return dict({'ok': ok, 'why': why, 'measured': {}, 'frames': []}, **kw)


def quat_pitch_yaw(pitch_deg, yaw):
    return G.quat(kin.Ry(math.radians(pitch_deg)) @ kin.Rz(yaw))


def ee_yaw(q):
    """Yaw of the gripper's closing axis (ee x) projected on the floor."""
    x = np.asarray(kin.R_from_quat(q))[:, 0]
    return math.atan2(x[1], x[0])


def force(ctx, side):
    return np.array(C.status(ctx)['wrist_force_n'][side])


def move_arm(ctx, side, pos, quat, log=print):
    """Joint-space move of one arm to an ee pose: local IK from the live joints and torso, split so that no joint
    moves more than 1.1 rad per call, each part rby1_check'ed first."""
    s = C.status(ctx)
    other = 'left' if side == 'right' else 'right'
    arm, pe, re = kin.ik(side, s['torso'], s[f'{side}_arm'], list(pos), list(quat), s[f'{other}_arm'])
    if pe > 0.003 or re > 0.02:                       # 1 deg of wrist orientation is harmless; position exact
        return _res(False, f'IK {side} {list(pos)}: {pe * 1000:.1f} mm / {re:.3f} rad')
    cur = np.array(s[f'{side}_arm'])
    n = max(1, int(math.ceil(float(np.max(np.abs(arm - cur))) / 1.1)))
    for i in range(1, n + 1):
        q = (cur + (arm - cur) * i / n).round(4).tolist()
        chk = C.call(ctx, 'rby1_check', {'tool': 'rby1_move_joints',
                                         'arguments': {'component': f'{side}_arm', 'positions': q}})
        if not chk.get('accepted'):
            return _res(False, f"rby1_check refused {side} move: {chk.get('reason')}")
        r = ctx.value('rby1_move_joints', component=f'{side}_arm', positions=q)
        if r.get('blocked_rad') or r.get('warning'):
            return _res(False, f"{side} arm blocked: {r.get('warning')}")
        log(f'  {side} joint move {i}/{n}: {r.get("finish")}')
    return _res(True, measured={'ee': C.status(ctx)[f'{side}_ee']})


def cart(ctx, side, pos, speed=0.03, stop_n=STOP_N, log=print):
    """Cartesian translation of one hand (orientation kept) in parts of at most 0.24 m; stops when the wrist force
    changes by more than stop_n after a part."""
    f0 = force(ctx, side)
    cur = np.array(C.status(ctx)[f'{side}_ee']['position'])
    tgt = np.array(pos, float)
    n = max(1, int(math.ceil(np.linalg.norm(tgt - cur) / 0.24)))
    for i in range(1, n + 1):
        p = (cur + (tgt - cur) * i / n).round(4).tolist()
        r = ctx.value('rby1_cartesian', component=f'{side}_arm', targets=[{'link': f'ee_{side}', 'position': p}],
                      linear_velocity=speed)
        df = float(np.linalg.norm(force(ctx, side) - f0))
        if r.get('finish') != 'Ok' or df > stop_n:
            return _res(False, f'{side} cartesian stopped at part {i}/{n}: finish {r.get("finish")}, wrist Δ{df:.1f} N',
                        measured={'df': df})
    return _res(True, measured={'df': df})


def touch_down(ctx, side, z_from, z_min, step=0.005, touch_n=TOUCH_N, back=G.DESK_CLEAR, log=print):
    """Lower the hand in `step`s from z_from until the wrist force changes by touch_n (fingertips on the support)
    or z_min is reached; after a contact go back up by `back`. -> measured {z, contact}"""
    ee = C.status(ctx)[f'{side}_ee']['position']
    f0 = force(ctx, side)
    z, contact = z_from, False
    while z - step >= z_min - 1e-6:
        z = round(z - step, 4)
        ctx.value('rby1_cartesian', component=f'{side}_arm', targets=[{'link': f'ee_{side}',
                                                                        'position': [ee[0], ee[1], z]}],
                  linear_velocity=0.01)
        df = float(np.linalg.norm(force(ctx, side) - f0))
        if df > touch_n:
            contact = True
            z = round(z + back + (step if df > 2 * touch_n else 0.0), 4)     # pressed hard: one more step
            ctx.value('rby1_cartesian', component=f'{side}_arm', targets=[{'link': f'ee_{side}',
                                                                            'position': [ee[0], ee[1], z]}],
                      linear_velocity=0.01)
            log(f'  touch at Δ{df:.1f} N -> ee z {z}')
            break
    return _res(True, measured={'z': z, 'contact': contact})


def object_z(ctx, frame_id, name, log=print):
    """Median height of the named object's depth points in a frame (Gemini mask) or None."""
    seg = PP.segment_named(ctx, frame_id, [name], log=log)
    if name not in seg['masks']:
        return None
    P, _ = pp_vis.mask_points(frame_id, seg['masks'][name])
    return float(np.median(P[:, 2])) if len(P) > 30 else None


def pick(ctx, name, kind, side, look=None, stow=True, log=print):
    """Localize `name` (kind elongated / box / round) in a fresh frame, plan, pre-grasp, approach (touch-down for
    lying objects), close, lift 4 cm, check by depth that the object rose with the hand, stow at CARRY[side].
    Start: torso mid, the hand free and open or openable, the object within reach (else why = 'advance ...').
    End (ok): the object held at CARRY[side]. End (fail): hand above the object, open, nothing held."""
    frames = []
    if look:
        ctx.value('rby1_look_at', point=list(look))
    fid = C.observe(ctx)
    frames.append(fid)
    loc = PP.localize(ctx, fid, {name: kind}, log=log)
    obj = loc['objects'][name]
    if not obj.get('ok'):
        return _res(False, f'localize {name}: {obj.get("why")}', frames=frames, measured={'overlay': loc['overlay']})
    s = C.status(ctx)
    plan = G.plan(obj, side, s['torso'], {'left_arm': s['left_arm'], 'right_arm': s['right_arm']})
    if not plan['ok']:
        near_x = (obj['centre'][0] - obj.get('r', 0.0)) if kind == 'round' else (obj.get('grasp_xy') or obj['centre'])[0]
        return _res(False, f"{plan['why']}; advance ~{max(0.0, near_x - 0.47):.2f} m", frames=frames,
                    measured={'object': obj})
    z0 = object_z(ctx, fid, name, log=log)
    ctx.value('rby1_gripper', side=side, action='open')
    r = move_arm(ctx, side, plan['pre']['pos'], plan['pre']['quat'], log=log)
    if not r['ok']:
        return _res(False, f"pre-grasp: {r['why']}", frames=frames, measured={'plan': plan})
    fid = C.observe(ctx)
    frames.append(fid)
    td = pp_vis.topdown_check(fid, obj.get('grasp_xy') or obj['centre'], fingers=plan['fingers'])
    for p in plan['approach'] + ([] if plan['touch'] else [plan['grasp']['pos']]):
        r = cart(ctx, side, p, speed=0.02, log=log)
        if not r['ok']:
            cart(ctx, side, plan['pre']['pos'], speed=0.02, log=log)
            return _res(False, f"approach: {r['why']}", frames=frames, measured={'plan': plan, 'topdown': td})
    if plan['touch']:
        t = touch_down(ctx, side, plan['approach'][-1][2], plan['touch']['z_min'], log=log)
        log(f"  touch-down: {t['measured']}")
    g = ctx.value('rby1_gripper', side=side, action='close', current_a=CLOSE_A)
    width = g['measured_width_m']
    log(f"  closed: {width * 1000:.1f} mm blocked {g.get('blocked_by_object')}")
    if plan['kind'] == 'top' and width < 0.003:
        ctx.value('rby1_gripper', side=side, action='open')
        cart(ctx, side, plan['pre']['pos'], speed=0.03, log=log)
        return _res(False, 'closed on air (fingers above or beside the object)', frames=frames,
                    measured={'plan': plan, 'width': width, 'topdown': td})
    ee = C.status(ctx)[f'{side}_ee']['position']
    cart(ctx, side, [ee[0], ee[1], ee[2] + LIFT_CHECK], speed=0.02, log=log)
    fid = C.observe(ctx)
    frames.append(fid)
    z1 = object_z(ctx, fid, name, log=log)
    rise = None if z0 is None or z1 is None else round(z1 - z0, 3)
    log(f'  object rise {rise} m')
    if rise is not None and rise < HELD_RISE:
        ctx.value('rby1_gripper', side=side, action='open')
        cart(ctx, side, plan['pre']['pos'], speed=0.03, log=log)
        return _res(False, f'object did not rise with the hand ({rise} m)', frames=frames,
                    measured={'plan': plan, 'width': width, 'rise': rise, 'topdown': td})
    if not stow:                                        # stay lifted above the pick point (place it elsewhere)
        return _res(True, frames=frames, measured={'plan': {k: plan[k] for k in ('kind', 'yaw', 'grasp', 'pre')},
                                                   'width': width, 'rise': rise, 'topdown': td,
                                                   'object': {k: v for k, v in obj.items() if k != 'ridge'}})
    ee = C.status(ctx)[f'{side}_ee']['position']
    cart(ctx, side, [ee[0], ee[1], max(ee[2], CARRY[side][2])], speed=0.04, log=log)
    r = cart(ctx, side, CARRY[side], speed=0.04, log=log)
    ctx.value('rby1_look_at', point=[CARRY[side][0] + 0.1, CARRY[side][1], CARRY[side][2] - 0.25])
    frames.append(C.observe(ctx))                       # the held object hanging at the carry pose: checkpoint
    out = _res(r['ok'], r['why'], frames=frames,
               measured={'plan': {k: plan[k] for k in ('kind', 'yaw', 'grasp', 'pre')}, 'width': width,
                         'rise': rise, 'topdown': td, 'object': {k: v for k, v in obj.items() if k != 'ridge'}})
    out['checkpoint'] = {
        'subagent': 'haiku', 'frames': [f'runtime/frames/{frames[-1]}/head_camera.jpg'],
        'question': f"Is the robot's {side} gripper holding the {name}?",
        'answer': 'yes / no', 'pass_if': 'yes (and the depth rise check in measured agrees)',
        'on_fail': 'open the gripper over the desk, re-run pick once; then stop and report'}
    return out


def place_down(ctx, side, xy, z_release, lift=0.04, touch_n=TOUCH_N, log=print):
    """Put a held object down on a surface: carry at the current height to above xy, lower in 1 cm steps to
    z_release (the ee height at which it was picked from the same surface, + a few mm) or until the wrist force
    changes by touch_n, open, rise by 0.10. End: hand open above the placed object."""
    ee = C.status(ctx)[f'{side}_ee']['position']
    r = cart(ctx, side, [xy[0], xy[1], max(ee[2], z_release + lift)], speed=0.03, log=log)
    if not r['ok']:
        return _res(False, f'carry to place: {r["why"]}')
    f0 = force(ctx, side)
    z = C.status(ctx)[f'{side}_ee']['position'][2]
    while z > z_release + 1e-4:
        z = round(max(z_release, z - 0.01), 4)
        ctx.value('rby1_cartesian', component=f'{side}_arm', targets=[{'link': f'ee_{side}', 'position': [xy[0], xy[1], z]}],
                  linear_velocity=0.015)
        df = float(np.linalg.norm(force(ctx, side) - f0))
        if df > touch_n:
            log(f'  placed: contact at ee z {z} (Δ{df:.1f} N)')
            break
    ctx.value('rby1_gripper', side=side, action='open')
    cart(ctx, side, [xy[0], xy[1], z + 0.10], speed=0.03, log=log)
    return _res(True, measured={'z': z})


def free_spot(f, xy, half, support_z, max_frac=0.02):
    """Is a desk rectangle (centre xy, half sizes (hx, hy)) free? Share of depth points more than 1.5 cm above the
    support. -> (free, share)"""
    P, _, _ = pp_vis.geom.cloud(pp_vis.frame(f), step=2, max_range=2.0)
    k = (np.abs(P[:, 0] - xy[0]) < half[0]) & (np.abs(P[:, 1] - xy[1]) < half[1])
    if k.sum() < 30:
        return False, None
    z = P[k, 2]
    z = z[z < support_z + 0.25]                       # not the robot's own arms above the desk
    if len(z) < 30:
        return False, None
    share = float(np.mean(z > support_z + 0.015))
    return share < max_frac, round(share, 3)


def clear_view(ctx, held=(), log=print):
    """Both free hands to the raised wide pose (pitched -45°) so the head camera sees the desk."""
    for side in ('left', 'right'):
        if side not in held:
            r = move_arm(ctx, side, CLEAR[side], quat_pitch_yaw(-45, 0.0), log=log)
            if not r['ok']:
                return r
    return _res(True)


def approach_desk(ctx, edge=DESK_EDGE, max_drive=1.0, log=print):
    """Drive straight until the desk front edge is `edge` ahead (2 legs, re-measured; never closer)."""
    frames = []
    for leg in range(2):
        ctx.value('rby1_look_at', point=[0.9, 0.0, 0.7])
        fid = C.observe(ctx)
        frames.append(fid)
        d = pp_vis.desk_front(fid)
        if not d['ok']:
            return _res(False, d['why'], frames=frames)
        gap = d['x'] - edge
        log(f'  desk edge {d["x"]} (slope {d["slope"]}), drive {gap:+.3f}')
        if gap < 0.01:
            break
        if gap > max_drive:
            return _res(False, f'desk edge {d["x"]} m: farther than expected', frames=frames)
        r = C.base_drive(ctx, gap if leg == 1 or gap < 0.25 else gap * 0.7, step=0.15, speed=0.06, log=log)
        if not r['ok']:
            return _res(False, r.get('why'), frames=frames)
    return _res(True, measured={'edge': d['x'], 'z': d['z']}, frames=frames)


def find_bucket(ctx, look=(1.2, 0.0, 0.2), name='orange bucket', log=print):
    ctx.value('rby1_look_at', point=list(look))
    fid = C.observe(ctx)
    loc = PP.localize(ctx, fid, {name: 'round'}, log=log)
    b = loc['objects'][name]
    return _res(b.get('ok', False), b.get('why', ''), frames=[fid], measured={'bucket': b, 'overlay': loc['overlay']})


def approach_bucket(ctx, dist=BUCKET_DIST, name='orange bucket', log=print):
    """Face the bucket (turn about the axle) and drive until its centre is `dist` ahead; re-localizes after each
    leg. End: bucket centre ≈(dist, 0)."""
    frames = []
    for leg in range(3):
        r = find_bucket(ctx, look=(0.8 if leg else 1.2, 0.0, 0.2), name=name, log=log)
        frames += r['frames']
        if not r['ok']:
            return _res(False, f"bucket: {r['why']}", frames=frames)
        bx, by = r['measured']['bucket']['centre']
        bearing = math.atan2(by, bx - nav.AXLE)
        gap = math.hypot(bx - nav.AXLE, by) + nav.AXLE - dist
        log(f'  bucket ({bx:.3f}, {by:.3f}): turn {bearing:+.3f}, drive {gap:+.3f}')
        if abs(bearing) < 0.03 and abs(gap) < 0.02:
            break
        if abs(bearing) >= 0.03:
            C.base_turn(ctx, bearing, step=0.2, speed=0.1, log=log)
        if gap > 0.01:
            C.base_drive(ctx, gap if gap < 0.35 else gap * 0.7, step=0.3, speed=0.08, log=log)
        elif gap < -0.01:
            C.base_drive(ctx, gap, step=0.1, speed=0.05, log=log)
    return _res(True, measured={'bucket': [bx, by]}, frames=frames)


def drop_gently(ctx, sides, bucket_xy, confirm=None, drop_z=DROP_Z, log=print):
    """Release held objects into a floor bucket from just above its rim, one hand after the other: hands pitched
    -30° at standing height, torso lowest, each hand in turn to (bx - 0.09, by, 0.69), open, out; torso mid.
    confirm(frame_id, side) -> bool is asked before each release (e.g. a Gemini / agent check that the object is
    above the opening); False keeps holding, stands up and stops.
    Start: torso mid, objects held, bucket centre ≈(0.62, 0) ahead. End: torso mid, hands open and out."""
    bx, by = bucket_xy
    frames = []
    for i, side in enumerate(sides):
        q = C.status(ctx)[f'{side}_ee']['quaternion']
        yaw = ee_yaw(q)
        pos = (bx - 0.12, by + SIGN[side] * 0.10, 0.95) if i == 0 else (0.40, SIGN[side] * 0.25, 1.05)
        r = move_arm(ctx, side, pos, quat_pitch_yaw(DROP_PITCH, yaw), log=log)
        if not r['ok']:
            return _res(False, f'drop pose {side}: {r["why"]}')
    chk = C.call(ctx, 'rby1_check', {'tool': 'rby1_move_joints', 'arguments': {'component': 'torso', 'preset': 'lowest'}})
    if not chk.get('accepted'):
        return _res(False, f"torso lowest refused: {chk.get('reason')}")
    ctx.value('rby1_move_joints', component='torso', preset='lowest')
    for side in sides:
        r = cart(ctx, side, (bx - DROP_DX, by, drop_z), speed=0.04, log=log)
        if not r['ok']:
            break
        frames.append(C.observe(ctx))
        if confirm is not None and not confirm(frames[-1], side):
            cart(ctx, side, (bx - 0.17, SIGN[side] * 0.20, 0.80), speed=0.04, log=log)
            ctx.value('rby1_move_joints', component='torso', preset='mid')
            return _res(False, f'{side}: release not confirmed (object not above the opening?)', frames=frames)
        ctx.value('rby1_gripper', side=side, action='open')
        cart(ctx, side, (bx - 0.17, SIGN[side] * 0.20, 0.75), speed=0.04, log=log)
    ctx.value('rby1_move_joints', component='torso', preset='mid')
    if not r['ok']:
        return _res(False, f'release pose: {r["why"]}', frames=frames)
    ctx.value('rby1_look_at', point=[bx, by, 0.2])
    frames.append(C.observe(ctx))
    out = _res(True, frames=frames, measured={'bucket': [bx, by]})
    out['checkpoint'] = {
        'subagent': 'haiku', 'frames': [f'runtime/frames/{frames[-1]}/head_camera.jpg'],
        'question': 'Look into the orange bucket and at the floor around it. Which of the dropped objects are inside '
                    'the bucket, and is any object lying on the floor outside it?',
        'answer': 'list inside / outside', 'pass_if': 'all dropped objects inside, none on the floor',
        'on_fail': 'stop and report the frame (an object on the floor needs a floor pick or the operator)'}
    return out


def route(ctx, target_xy, target_yaw, reverse=0.0, turn_dir=None, log=print):
    """Base route in odometry (reverse, turn about the axle, drive, turn to target_yaw)."""
    od = C.status(ctx)['odometry']
    p = nav.plan_pose(od, target_xy, target_yaw, reverse=reverse, turn_dir=turn_dir)
    log(f'  route {p["moves"]}')
    r = nav.execute(ctx, p['moves'], log=log)
    return _res(r['ok'], r.get('why') or '', measured={'odom': r.get('odom'), 'moves': p['moves']})
