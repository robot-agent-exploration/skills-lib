"""Stack one upright open cup into another with a one-hand rim pinch, perception-checked. Tested 2026-10-09 on
silver aluminium cups (rim radius 0.045-0.049 m, height 0.11 m, ~40 g) on a 0.74 m desk, robot square to the desk
(desk edge 0.215 m ahead), torso mid, left hand.

Every primitive returns {ok, why, measured, frames} and, at the decisive points, a `checkpoint` brief for the agent
(or a perception subagent / Gemini): the held cup hangs 15-20 deg tilted from a one-sided rim pinch, so where its
bottom will go is perceived, not assumed.

  locate(ctx, names, look)            fresh frame; per cup: Gemini rim-lip points ray-cast at the depth rim height
  pick_cup(ctx, name, side, look)     fingers-down rim pinch (closed form: the rim point facing the hand's shoulder),
                                      lift 6 cm, held check (never releases on doubt), raise to the carry height
  align_over(ctx, side, target, pinch) hand over the target so the held cup's bottom is above its opening (pinch
                                      geometry, then closed-loop depth correction); stands up to 'ready' when the
                                      carry pose is not reached at 'mid'; returns the alignment checkpoint
  insert(ctx, side, target, pinch)    lower until the cups wedge / the outer finger meets the rim (wrist force) or
                                      the nest depth, release, rise; returns the nesting checkpoint

v2 (2026-10-09, compositional session): hand moves (pre-grasp, lift, raise, transfer, rise) are M.move_arm = the
controller's optimal control (oc-hand-reach), grasp planning is closed form (G.plan_rim): no IK seed / angle search.
Not yet re-run on the robot with this change.
"""
import math
import time

import _deps  # noqa: F401
import numpy as np

import chair_ctl as C
import geom
import kin
import pp_grasp as G
import pp_motion as M
import pp_perception as PP
import pp_vis

CUP_RIM_IN = 0.004        # cup wall inward of the rim 2.4 cm down (straight-ish aluminium cups)
NEST_RISE = 0.024         # each nested cup's rim stands this much above the one below (measured 0.024-0.026)
CARRY_CLEAR = 0.06        # held cup's bottom above the target rim while moving over it (0.018 knocked the target over)
CONTACT_N = 5.0           # wrist force change = the outer finger met the target rim -> release
ALIGN_TOL = 0.012         # held cup bottom centre within this of the target axis before inserting
ALIGN_STEPS = 3           # closed-loop xy corrections at most
CUP_H = 0.112             # cup height if the depth support estimate fails
HELD = 'silver cup held by the robot gripper'


def _res(ok, why='', **kw):
    return dict({'ok': ok, 'why': why, 'measured': {}, 'frames': []}, **kw)


def _down_quat(yaw):
    return G.quat(kin.Rz(yaw))


TORSO_POS = {'mid': [0.0, 0.7854, -1.5708, 0.7854, 0.0, 0.0], 'ready': [0.0, 0.5236, -1.0472, 0.5236, 0.0, 0.0]}


def set_torso(ctx, goal, log=print):
    """Torso preset: 'mid' to pick, 'ready' (standing up a bit: the high carry pose over a cup with the left wrist at
    yaw 180 had no IK at mid; v2: used when the optimal-control move does not reach it). rby1_check first; no-op
    when already there."""
    pos = TORSO_POS[goal]
    if np.allclose(C.status(ctx)['torso'], pos, atol=0.02):
        return True
    cur = np.array(C.status(ctx)['torso'])
    n = max(1, int(math.ceil(float(np.max(np.abs(np.array(pos) - cur))) / 1.1)))   # <= 1.2 rad per call (zero -> mid 1.57)
    for i in range(1, n + 1):
        q = (cur + (np.array(pos) - cur) * i / n).round(4).tolist()
        chk = C.call(ctx, 'rby1_check', {'tool': 'rby1_move_joints', 'arguments': {'component': 'torso', 'positions': q}})
        if not chk.get('accepted'):
            log(f'  torso {goal} refused: {chk.get("reason")}')
            return False
        ctx.value('rby1_move_joints', component='torso', positions=q)
    log(f'  torso -> {goal} loads {C.loads(C.status(ctx))}')
    return True


def locate(ctx, names, look=None, log=print):
    """Fresh frame -> {name: cup description} (PP.cup) for every name."""
    if look:
        M.look_at(ctx, list(look))
    fid = C.observe(ctx)
    cups = {n: PP.cup(ctx, fid, n, log=log) for n in names}
    bad = [n for n, c in cups.items() if not c.get('ok')]
    return _res(not bad, f'not located: {bad}' if bad else '', frames=[fid], measured={'cups': cups})


def axis_offset(pinch):
    """Held cup axis xy relative to the hand (ee) xy for a rim pinch, fingers down: pinch point is straight below
    the hand; the cup centre lies (r - rim_in) inward of the pinched rim point."""
    a = math.radians(pinch['rim_angle'])
    return -(pinch['r'] - pinch['rim_in']) * np.array([math.cos(a), math.sin(a)])


def ee_z_for_bottom(bottom_z, pinch):
    """Hand height (fingers down) that puts the held cup's bottom at bottom_z."""
    return bottom_z + pinch['h'] - pinch['depth'] + G.TIP_LOW


def pick_cup(ctx, name, side, look=None, depth=G.RIM_DEPTH, carry_bottom=None, log=print):
    """Rim-pinch one cup and lift it. Start: hand free, torso any, cup localizable within fingers-down reach.
    End (ok): cup held fingers-down with its bottom at carry_bottom (default its rim + CARRY_CLEAR + one nest rise),
    measured.pinch for align_over. ok None: holding but the held check is unsure (checkpoint). ok False: hand open,
    back at the pre-grasp (or nothing moved; after a failed pre-grasp move, where the controller stopped it).
    Reachability is decided by the controller: a refused / unreached optimal-control move is the failure."""
    set_torso(ctx, 'mid', log=log)
    loc = locate(ctx, [name], look=look, log=log)
    cup = loc['measured']['cups'][name]
    frames = list(loc['frames'])
    if not cup.get('ok'):
        return _res(False, f'locate {name}: {cup.get("why")}', frames=frames)
    s = C.status(ctx)
    obj = dict(cup, rim_in=CUP_RIM_IN, depth=depth)
    plan = G.plan_rim(obj, side, s['torso'], {'left_arm': s['left_arm'], 'right_arm': s['right_arm']}, pitches=(0,),
                      cur_quat=s[f'{side}_ee']['quaternion'])
    if not plan['ok']:                     # fingers-down only (a tilted pinch let a cup swing off and fall)
        return _res(False, f"{name} out of fingers-down reach: {plan['why']}", frames=frames, measured={'cup': cup})
    # closed form (v2): the rim point facing the shoulder, yaw nearer the hand's current yaw; no IK feasibility here
    log(f"  {name}: rim {plan['rim_angle']} deg, pitch {plan['pitch']}, yaw {math.degrees(plan['yaw']):.0f}")
    ctx.value('rby1_gripper', side=side, action='open')
    r = M.move_arm(ctx, side, plan['pre']['pos'], plan['pre']['quat'], log=log)
    if not r['ok']:
        return _res(False, f"pre-grasp: {r['why']}", frames=frames)
    frames.append(C.observe(ctx))
    for p in plan['approach'] + [plan['grasp']['pos']]:
        r = M.cart(ctx, side, p, speed=0.015, stop_n=CONTACT_N, log=log)
        if not r['ok']:
            M.move_arm(ctx, side, plan['pre']['pos'], plan['pre']['quat'], log=log)
            return _res(False, f"approach: {r['why']}", frames=frames, measured={'plan': plan})
    g = ctx.value('rby1_gripper', side=side, action='close', current_a=M.CLOSE_A)
    ee = C.status(ctx)[f'{side}_ee']['position']
    M.move_arm(ctx, side, [ee[0], ee[1], ee[2] + LIFT_CUP], plan['grasp']['quat'], log=log)
    fid = C.observe(ctx)
    frames.append(fid)
    support = cup.get('support_z') or 0.74
    verdict, low = held_check(ctx, fid, support, log=log)
    log(f"  closed {g['measured_width_m'] * 1000:.1f} mm, held check: {verdict} (lowest {low}, desk {support:.3f})")
    if verdict == 'not held':                             # clearly still on the desk: let go, back off
        ctx.value('rby1_gripper', side=side, action='open')
        M.cart(ctx, side, plan['pre']['pos'], speed=0.02, stop_n=30, log=log)
        return _res(False, f'{name} stayed on the desk (lowest point {low})', frames=frames, measured={'plan': plan})
    if verdict == 'unsure':                               # never release on doubt: keep holding, ask
        out = _res(None, 'held check unsure: agent decides (still holding)', frames=frames,
                   measured={'plan': plan, 'lowest': low})
        out['checkpoint'] = {'subagent': 'agent (you)', 'frames': [f'runtime/frames/{fid}/head_camera.jpg'],
                             'question': f"Is the robot's {side} gripper holding the cup off the desk?",
                             'answer': 'yes / no', 'pass_if': 'yes -> continue with align',
                             'on_fail': 'lower it back to the desk, open, re-locate, pick again'}
        return out
    rise = round(low - support, 3)
    h = cup['height'] if cup.get('height') and 0.05 < cup['height'] < 0.25 else CUP_H
    pinch = {'rim_angle': plan['rim_angle'], 'r': cup['r'], 'rim_in': CUP_RIM_IN, 'depth': depth, 'h': h,
             'yaw': plan['yaw'], 'pitch': plan['pitch']}
    # raise (optimal control, same orientation) so the bottom clears the desk's other cups, then fingers down
    bottom = carry_bottom if carry_bottom is not None else cup['rim_z'] + CARRY_CLEAR + NEST_RISE
    ee = C.status(ctx)[f'{side}_ee']
    tip_z_now = ee['position'][2] + (np.asarray(kin.R_from_quat(ee['quaternion'])) @ [0, 0, -G.TIP_LOW])[2]
    lift = bottom - (tip_z_now + depth - h)
    if lift > 0:
        r = M.move_arm(ctx, side, [ee['position'][0], ee['position'][1], ee['position'][2] + lift], ee['quaternion'],
                       log=log)
        if not r['ok']:
            return _res(False, f'raise: {r["why"]}', frames=frames, measured={'pinch': pinch})
    return _res(True, frames=frames, measured={'pinch': pinch, 'rise': rise, 'width': g['measured_width_m'],
                                               'cup': {k: cup[k] for k in ('centre', 'r', 'rim_z', 'height')}})


LIFT_CUP = 0.06           # lift before the held check (a 4 cm lift was misjudged as "not risen")
HELD_CLEAR = 0.025        # held cup's lowest point at least this above the desk = lifted
LEFT_ON_DESK = 0.015      # lowest point within this of the desk = clearly not lifted -> release and back off


def held_check(ctx, fid, support_z, log=print):
    """Is a cup hanging from the gripper? Segment 'the cup held by the robot gripper' (not the cup's scene name:
    a held cup no longer matches 'standing alone' and that name picked the wrong cup) and compare its lowest depth
    point with the desk. -> 'held' | 'not held' | 'unsure', bottom z"""
    seg = PP.segment_named(ctx, fid, [HELD], log=log)
    if HELD not in seg['masks']:
        return 'unsure', None
    P, _ = pp_vis.mask_points(geom.load(fid), seg['masks'][HELD])
    if len(P) < 50:
        return 'unsure', None
    low = float(np.percentile(P[:, 2], 2))
    if low >= support_z + HELD_CLEAR:
        return 'held', low
    if low <= support_z + LEFT_ON_DESK:
        return 'not held', low
    return 'unsure', low


def held_bottom(ctx, fid, target_rim_z, log=print):
    """Held cup's bottom-region centre (depth points of the held cup within 2 cm of its lowest point), corrected
    for the camera seeing only its near side (+0.6 r away from the camera). -> (xy, low_z) or (None, None)"""
    seg = PP.segment_named(ctx, fid, [HELD], log=log)
    if HELD not in seg['masks']:
        return None, None
    f = geom.load(fid)
    P, _ = pp_vis.mask_points(f, seg['masks'][HELD])
    P = P[P[:, 2] > target_rim_z + 0.005]
    if len(P) < 50:
        return None, None
    low = float(np.percentile(P[:, 2], 2))
    B = P[P[:, 2] < low + 0.02][:, :2].mean(0)
    cam = f['T'][:2, 3]
    away = (B - cam) / max(1e-6, np.linalg.norm(B - cam))
    return B + 0.6 * 0.033 * away, low


def align_over(ctx, side, target, pinch, nudge=None, log=print):
    """Bring the held cup's bottom above the target cup (description: centre, rim_z): move at a safe
    height to the geometric pose, then correct from depth until within ALIGN_TOL. Ends holding, above the target,
    with the checkpoint for the agent."""
    T = np.array(target['centre'])
    q = _down_quat(pinch['yaw'])
    z_safe = ee_z_for_bottom(target['rim_z'] + CARRY_CLEAR, pinch)
    ee = C.status(ctx)[f'{side}_ee']
    R = np.asarray(kin.R_from_quat(ee['quaternion']))
    tip = np.asarray(ee['position']) + R @ [0, 0, -G.TIP_LOW]
    bottom = tip[2] + pinch['depth'] - pinch['h']
    if bottom < target['rim_z'] + CARRY_CLEAR - 0.005:    # never move sideways with the bottom lower than that
        r = M.move_arm(ctx, side, list(np.asarray(ee['position']) + [0, 0, target['rim_z'] + CARRY_CLEAR - bottom]),
                       ee['quaternion'], log=log)
        if not r['ok']:
            return _res(False, f'raise before transfer: {r["why"]}')
    xy = T - axis_offset(pinch)
    frames, off = [], None
    for i in range(ALIGN_STEPS + 1):
        r = M.move_arm(ctx, side, [float(xy[0]), float(xy[1]), z_safe], q, log=log)
        if not r['ok'] and M.unreached(r) and set_torso(ctx, 'ready', log=log):     # not reached, nothing pressed
            r = M.move_arm(ctx, side, [float(xy[0]), float(xy[1]), z_safe], q, log=log)   # stood up a bit
        if not r['ok']:
            return _res(False, f'over target: {r["why"]}', frames=frames)
        M.look_at(ctx, [float(T[0]), float(T[1]), target['rim_z']])
        fid = C.observe(ctx)
        frames.append(fid)
        b, low = held_bottom(ctx, fid, target['rim_z'], log=log)
        if b is None:
            break
        off = b - T
        log(f'  align {i}: held bottom {np.round(b, 3)} low z {low:.3f} -> offset {np.round(off, 3)}')
        if nudge is not None or np.linalg.norm(off) < ALIGN_TOL or i == ALIGN_STEPS:
            break
        xy = xy - np.clip(off, -0.04, 0.04)
    if nudge is not None:                              # the agent's correction (it looked at the checkpoint)
        xy = xy + np.clip(np.asarray(nudge, float), -0.04, 0.04)
        r = M.move_arm(ctx, side, [float(xy[0]), float(xy[1]), z_safe], q, log=log)
        if not r['ok']:
            return _res(False, f'nudge: {r["why"]}', frames=frames)
        fid = C.observe(ctx)
        frames.append(fid)
        b, low = held_bottom(ctx, fid, target['rim_z'], log=log)
        off = None if b is None else b - T
    td = pp_vis.topdown_check(frames[-1], T, half=0.12, fingers=[T.tolist()], zdark=(target['rim_z'] + 0.01, 1.3))
    out = _res(off is not None and np.linalg.norm(off) < 2 * ALIGN_TOL,
               '' if off is not None else 'held cup not measured', frames=frames,
               measured={'offset': None if off is None else np.round(off, 3).tolist(), 'hand_xy': np.round(xy, 3).tolist(),
                         'z_safe': round(z_safe, 3), 'topdown': td})
    out['checkpoint'] = {
        'subagent': 'agent (you) or haiku', 'frames': [f'runtime/frames/{frames[-1]}/head_camera.jpg', td],
        'question': f"The robot's {side} gripper holds a silver cup by its rim above another cup on the "
                    'desk. Is the bottom of the hanging cup directly above the opening of the cup below, so that '
                    'lowering it straight down puts it inside (not onto the rim, not beside it)? If not, which way '
                    'should it move (towards / away from the camera, left / right) and about how many cm?',
        'answer': 'yes | no + direction and cm', 'pass_if': 'yes, and the measured offset < 2.4 cm',
        'on_fail': 'move the hand by the stated correction (align_over again with an offset) and re-check; never '
                   'lower a cup that is not over the opening'}
    return out


def nested_check(ctx, fid, target, log=print):
    """After the release: depth says nested when the top rim over the target stands NEST_RISE (+-1.5 cm) above the
    target's rim; Gemini is asked the same. Both agree -> 'nested' / 'not nested'; otherwise 'unsure' (the agent
    looks). -> (verdict, top rim z)"""
    f = geom.load(fid)
    P, _, _ = geom.cloud(f, step=2, max_range=2.0)
    d = np.linalg.norm(P[:, :2] - np.array(target['centre']), axis=1)
    k = (d < 0.06) & (P[:, 2] > target['rim_z'] - 0.05) & (P[:, 2] < target['rim_z'] + 0.12)
    rim = float(np.percentile(P[k, 2], 98)) if k.sum() > 50 else None
    depth_yes = rim is not None and abs(rim - (target['rim_z'] + NEST_RISE)) < 0.015
    g = PP.check(ctx, fid, 'Look at the silver cup(s) under the robot gripper. Two nested cups look like ONE cup with '
                           'TWO concentric rims at the top (the inner cup rim just inside / above the outer rim). Is '
                           'that what you see - a cup inside another cup, upright - and not two separate cups or a cup '
                           'lying on its side?', log=log)
    if g.get('ok') and g.get('yes') == depth_yes:
        return ('nested' if depth_yes else 'not nested'), rim
    return 'unsure', rim


def insert(ctx, side, target, pinch, log=print):
    """Lower the aligned cup into the target until the outer finger meets the target rim (wrist force) or the nest
    depth, release, rise 8 cm. Returns the nesting checkpoint."""
    q = _down_quat(pinch['yaw'])
    ee = C.status(ctx)[f'{side}_ee']['position']
    z_min = ee_z_for_bottom(target['rim_z'] + NEST_RISE - pinch['h'] - 0.005, pinch)
    f0 = M.force(ctx, side)
    z, contact = ee[2], False
    while z > z_min + 1e-4:
        step = 0.02 if z - z_min > 0.05 else 0.01
        z = round(max(z_min, z - step), 4)
        ctx.value('rby1_cartesian', component=f'{side}_arm', targets=[{'link': f'ee_{side}', 'position': [ee[0], ee[1], z]}],
                  linear_velocity=0.015)
        df = float(np.linalg.norm(M.force(ctx, side) - f0))
        log(f'  insert: ee z {z} dF {df:.1f}')
        if df > CONTACT_N:
            contact = True
            break
    ctx.value('rby1_gripper', side=side, action='open')
    time.sleep(0.5)
    # rise 15 cm (trial 2: after an 8 cm rise the open fingers were still in the depth window over the rim -> the
    # nested check read 'not nested' for two nested cups)
    r = M.move_arm(ctx, side, [ee[0], ee[1], z + 0.15], q, log=log)
    M.look_at(ctx, [target['centre'][0], target['centre'][1], target['rim_z']])
    fid = C.observe(ctx)
    nested, rim = nested_check(ctx, fid, target, log=log)
    ok = r['ok'] and (True if nested == 'nested' else None if nested == 'unsure' else False)
    out = _res(ok, r.get('why', '') or ('' if nested == 'nested' else f'nested check: {nested}'), frames=[fid],
               measured={'release_z': z, 'contact': contact, 'nested': nested, 'top_rim_z': rim})
    out['checkpoint'] = {
        'subagent': 'haiku', 'frames': [f'runtime/frames/{fid}/head_camera.jpg'],
        'question': 'Is the released cup now nested inside the other cup (one inside the other, upright, both rims '
                    'together at the top), not lying on its side or standing beside it?',
        'answer': 'yes / no + what is wrong', 'pass_if': 'yes',
        'on_fail': 'stop; re-locate both cups and pick the loose one again'}
    return out
