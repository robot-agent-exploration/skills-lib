"""Pipeline steps for wiping a soy-sauce stain with a towel. Each step takes ctx, checks its
preconditions from fresh observations or the robot state, acts, verifies, and returns a dict
with ok and what it measured (or raises StepError with a reason). Tested values: see SKILL.md."""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules of this skill
import motions as M      # noqa: E402
import perception as P   # noqa: E402

from swipe_table_params import (  # hyper-parameters: swipe_table_params.py
    CARRY_Z, LOWER_Z, SAFE_Z, MAX_SEG, PINCH_PRESS_Z, PINCH_TOL_MM, WAD_FORCE, PINCH_FORCE, MIN_WAD_WIDTH,
    PATH_X, PATH_Y, CARRY_MAX_X, CARRY_MAX_Y, WRIST_MARGIN, HOME_JOINTS, CAMERA_SLACK, CONTACT_PROBE_Z,
    GRASP_X, GRASP_Y,
)


class StepError(RuntimeError):
    pass


def _seg_moves(ctx, start, goal, log, **kw):
    """Straight move split into <= MAX_SEG pieces."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    n = max(1, int(math.ceil(np.linalg.norm(goal - start) / MAX_SEG)))
    reply = None
    for i in range(1, n + 1):
        reply = M.move(ctx, start + (goal - start) * i / n, log=log, **kw)
    return reply


def tcp(ctx):
    return ctx.value('franka_status')['tcp_position']


def gripper_state(ctx, wait_s=6.0, settle_s=0.6):
    """Gripper width / is_grasped once it stopped moving (after a short settle: a status read
    right after a command can still show the previous state)."""
    time.sleep(settle_s)
    t0 = time.time()
    while True:
        g = ctx.value('franka_status')['gripper']
        if not g.get('is_moving') or time.time() - t0 > wait_s:
            return g
        time.sleep(0.3)


def observe(ctx, log, tag=''):
    reply = ctx.value('franka_observe')
    frame = P.load_frame(reply['frame_id'])
    log(f'observe {tag}: frame {reply["frame_id"]} tcp {[round(v, 3) for v in reply["tcp_position"]]}')
    return frame


def go_home(ctx, log):
    r = ctx.value('franka_move_joints', preset='home')
    log(f'home: tcp {[round(v, 3) for v in r["tcp_position"]]}')
    return r


def lift_to(ctx, z, log):
    p = tcp(ctx)
    if p[2] < z - 0.005:
        M.move(ctx, [p[0], p[1], z], log=log)


def carry_xy(x, y):
    """Where to hold the towel at CARRY_Z above/near (x, y): pulled inside the reachable area."""
    return min(x, CARRY_MAX_X), min(max(y, -CARRY_MAX_Y), CARRY_MAX_Y)


def lift_clear(ctx, log):
    """Lift the held towel to CARRY_Z, rising diagonally inwards when far out (straight up
    from a stretched low pose can be out of reach)."""
    p = tcp(ctx)
    cx, cy = carry_xy(p[0], p[1])
    if p[2] < CARRY_Z - 0.005 or (cx, cy) != (p[0], p[1]):
        _seg_moves(ctx, p, [cx, cy, CARRY_Z], log)


def recentre(ctx, pt, log):
    """Go to pt with a joint move to the IK solution seeded from HOME_JOINTS (elbow up, J3 ~ 0):
    resets the arm's redundant posture so later straight-line moves near the side cameras start
    from the posture pose_ok() checked (trial 5: a swung elbow cost 2-3 cm of camera clearance).
    Falls back to straight moves when the joint path is not accepted."""
    r = ctx.value('franka_ik', position=[float(v) for v in pt], quaternion=M.Q_DOWN, seed=HOME_JOINTS)
    if r.get('move_joints_path') == 'ok' and r.get('solution'):
        ctx.value('franka_move_joints', positions=r['solution'])
        log(f'recentre at {np.round(pt, 3).tolist()} by joint move (cameras {r.get("camera_clearance_m")})')
    else:
        log(f'recentre: joint path not accepted ({r.get("move_joints_path")}); straight move instead')
        _seg_moves(ctx, tcp(ctx), pt, log)


def pose_ok(ctx, pt):
    """Posture at pt (gripper down, IK seeded from HOME_JOINTS) keeps joint margins and camera keep-out with some slack.
    (move_pose follows its own path; the service still checks it and scrub skips refusals.)"""
    r = ctx.value('franka_ik', position=[float(v) for v in pt], quaternion=M.Q_DOWN, seed=HOME_JOINTS)
    clear = min((r.get('camera_clearance_m') or {}).values() or [0.0])
    ok = r.get('min_margin_rad', 0) > 0.3 and clear > r.get('camera_keepout_m', 0.2) + CAMERA_SLACK
    return ok, r


def check_reachable(ctx, points, log):
    """pose_ok for each (x, y, z); raises StepError if one is not."""
    for pt in points:
        ok, r = pose_ok(ctx, pt)
        if not ok:
            raise StepError(f'target {pt} not safely reachable: margin {r.get("min_margin_rad")} '
                            f'cameras {r.get("camera_clearance_m")}')
    log(f'reachable: {len(points)} targets ok')


def fit_region(ctx, region, log, z=0.12, step=0.02, max_iter=15):
    """Clip region to the path limits, then shrink edges next to corners whose posture is not
    safe (camera keep-out near the side cameras, joint margins far out) until all corners and
    the centre are safe. Raises StepError if nothing usable is left."""
    x0, x1, y0, y1 = region
    x0, x1 = max(PATH_X[0], x0), min(PATH_X[1], x1)
    y0, y1 = max(PATH_Y[0], y0), min(PATH_Y[1], y1)
    for _ in range(max_iter):
        if x1 - x0 < 0.03 or y1 - y0 < 0.03:
            break
        corners = {'x0y0': (x0, y0), 'x1y0': (x1, y0), 'x0y1': (x0, y1), 'x1y1': (x1, y1),
                   'c': ((x0 + x1) / 2, (y0 + y1) / 2)}
        bad = [k for k, (x, y) in corners.items() if not pose_ok(ctx, (x, y, z))[0]]
        if not bad:
            if (x0, x1, y0, y1) != tuple(region):
                log(f'wipe region fitted to reachable area: {np.round(region, 3).tolist()} -> '
                    f'{np.round((x0, x1, y0, y1), 3).tolist()}')
            return (x0, x1, y0, y1)
        if 'c' in bad:
            break
        for k in bad:
            # move this corner's x edge or y edge inwards, whichever fixes it / loses less area
            options = [('x0', step) if k.startswith('x0') else ('x1', -step),
                       ('y0', step) if k.endswith('y0') else ('y1', -step)]
            best = None
            for edge, d in options:
                e = {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
                e[edge] += d
                cx = e['x0'] if k.startswith('x0') else e['x1']
                cy = e['y0'] if k.endswith('y0') else e['y1']
                fixed = pose_ok(ctx, (cx, cy, z))[0]
                loss = step * ((y1 - y0) if edge[0] == 'x' else (x1 - x0))
                if best is None or (not fixed, loss) < best[0]:
                    best = ((not fixed, loss), edge, d)
            _, edge, d = best
            if edge == 'x0':
                x0 += d
            elif edge == 'x1':
                x1 += d
            elif edge == 'y0':
                y0 += d
            else:
                y1 += d
    raise StepError(f'no safely reachable wipe region inside {np.round(region, 3).tolist()}')


# ------------------------------------------------------------------ perception steps

def find_towel(frame, log):
    t = P.locate_towel(frame)
    if not t['found']:
        raise StepError('towel not found on the table')
    log(f"towel: {t['shape']} centre {np.round(t['centre'], 3).tolist()} peak {np.round(t['peak'], 3).tolist()} "
        f"top_h {t['top_h']:.3f} area {t['area_m2']:.4f}")
    x, y = t['peak']
    if not (GRASP_X[0] <= x <= GRASP_X[1] and GRASP_Y[0] <= y <= GRASP_Y[1]):
        raise StepError(f'towel at {t["peak"]} is outside the grasp area {GRASP_X} x {GRASP_Y}')
    return t


def find_stain(frame, ref, log, towel_masks=()):
    """Soy sauce of any shade (dark puddle or yellow-brown film) as one cluster."""
    exclude = np.zeros(P.grid()[0].shape, bool)
    for m in towel_masks:
        exclude |= m
    s = P.locate_stain(frame, ref, exclude=exclude)
    if not s['found']:
        log(f'stain: none found ({"vs reference" if ref is not None else "no reference"})')
        return s
    log(f"stain: centre {np.round(s['centre'], 3).tolist()} bbox {np.round(s['bbox'], 3).tolist()} "
        f"area {s['area_m2'] * 1e4:.1f} cm2 components {s['n_components']} others {len(s['others'])} "
        f"({'vs reference ' + ref['id'] if ref is not None else 'no reference'})")
    return s


def measure_residue(frame, ref, roi, log, towel_masks=(), wiped=None):
    """Residue after a pass: exterior cameras over roi (coarse, wide) and, when a reference is
    available, the wrist camera at home over the wiped area + WRIST_MARGIN (fine: catches
    thin streaks the exterior maps blur). Returns the exterior result with 'wrist' attached and
    a combined 'bbox_all'."""
    exclude = np.zeros(P.grid()[0].shape, bool)
    for m in towel_masks:
        exclude |= m
    r = P.residue(frame, ref, roi=roi, exclude=exclude)
    log(f"residue (exterior) in roi {np.round(roi, 3).tolist()}: area {r['area_m2'] * 1e4:.1f} cm2 "
        f"bbox {np.round(r['bbox'], 3).tolist() if r.get('found') else None} {r.get('error', '')}")
    boxes = [r['bbox']] if r.get('found') else []
    r['wrist'] = None
    if ref is not None and wiped is not None:
        w_roi = (wiped[0] - WRIST_MARGIN, wiped[1] + WRIST_MARGIN, wiped[2] - WRIST_MARGIN, wiped[3] + WRIST_MARGIN)
        w = P.residue_wrist(frame, ref, w_roi)
        r['wrist'] = {k: w.get(k) for k in ('found', 'area_m2', 'coverage', 'bbox', 'db_p99', 'dl_p99', 'error')}
        log(f"residue (wrist) in {np.round(w_roi, 3).tolist()}: area {w['area_m2'] * 1e4:.2f} cm2 coverage "
            f"{w['coverage']:.2f} bbox {np.round(w['bbox'], 3).tolist() if w.get('found') else None} {w.get('error', '')}")
        if w.get('found'):
            boxes.append(w['bbox'])
    if boxes:
        r['bbox_all'] = (min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes), max(b[3] for b in boxes))
    return r


# ------------------------------------------------------------------ manipulation steps

def _press_grasp(ctx, x, y, z_above, z_press, tol_mm, force, log):
    """Open, travel over (x, y) at a safe height, descend to z_above, press to z_press (a stall
    on the cloth within tol_mm keeps pressing; a timeout leaves no residual press), grasp, wait."""
    ctx.value('franka_gripper', action='open')
    lift_to(ctx, z_above, log)
    # joint move to the posture above the target: a long straight line at home height towards a
    # far-out towel hit the controller's joint-step limit (trial 7: 0.205 > 0.200 rad)
    recentre(ctx, [x, y, z_above], log)
    p = tcp(ctx)
    if abs(p[0] - x) > 0.005 or abs(p[1] - y) > 0.005 or abs(p[2] - z_above) > 0.005:
        M.move(ctx, [x, y, z_above], log=log)
    try:
        r = ctx.value('franka_move_pose', position=[x, y, z_press], quaternion=M.Q_DOWN,
                      tolerance_mm=tol_mm, timeout_s=10)
        log(f'press to {z_press:.3f}: stalled at {r["measured_position"][2]:.4f} (err {r["position_error_mm"]} mm)')
    except RuntimeError as exc:
        err, state = M._error_state(exc)
        if err is None or err.get('control_uncertain') or err.get('type') != 'TimeoutError':
            raise
        log(f'press to {z_press:.3f}: timeout at {state["tcp_position"][2]:.4f} (no residual press)')
    ctx.value('franka_gripper', action='grasp', width=0.0, force=force, speed=0.03)
    g = gripper_state(ctx)
    log(f'grasp: width {g["width"] * 1000:.1f} mm is_grasped {g["is_grasped"]}')
    return g


def grasp_towel(ctx, log, attempts=3, lift_z=SAFE_Z):
    """Grasp the towel as a thick wad. Flat towel: pinch its centre, lower it into a heap,
    re-grasp the heap. Ends holding the towel at lift_z (keep it low to keep the underside
    down). Returns pickup xy (towel centre when first seen), grasp xy, width, attempts."""
    pickup = None
    for attempt in range(1, attempts + 1):
        frame = observe(ctx, log, f'towel attempt {attempt}')
        t = find_towel(frame, log)
        if pickup is None:
            pickup = t['centre']
        top_z = P.TABLE_Z + t['top_h']
        if t['shape'] == 'flat':
            cx, cy = t['centre']
            check_reachable(ctx, [(cx, cy, 0.11), (cx, cy, 0.25)], log)
            z_press = max(0.10, top_z - 0.003)
            g = _press_grasp(ctx, cx, cy, 0.20, z_press, PINCH_TOL_MM, PINCH_FORCE, log)
            if not g['is_grasped']:
                ctx.value('franka_gripper', action='open')
                lift_to(ctx, 0.20, log)
                log('pinch failed; retry')
                continue
            M.move(ctx, [cx, cy, 0.22], log=log)
            M.move(ctx, [cx, cy, 0.15], log=log)        # towel collapses into a heap under the gripper
            ctx.value('franka_gripper', action='open')
            gripper_state(ctx, 3.0)
            M.move(ctx, [cx, cy, SAFE_Z], log=log)
            continue                                    # re-observe the heap and wad-grasp it
        x, y = t['peak']
        check_reachable(ctx, [(x, y, 0.12), (x, y, CARRY_Z)], log)
        z_above = max(top_z + 0.06, 0.20)
        z_press = max(M.Z_FLOOR, top_z - 0.045)
        g = _press_grasp(ctx, x, y, z_above, z_press, 20.0, WAD_FORCE, log)
        if g['is_grasped'] and g['width'] >= MIN_WAD_WIDTH:
            M.move(ctx, [x, y, max(lift_z, top_z + 0.03)], log=log)
            g = gripper_state(ctx, 1.0)
            if g['is_grasped']:
                log(f'towel held (width {g["width"] * 1000:.1f} mm)')
                return {'ok': True, 'pickup': pickup, 'grasp_xy': (x, y), 'width': g['width'], 'attempts': attempt}
            log('towel slipped while lifting; retry')
        else:
            log('wad grasp too thin or empty; retry')
        ctx.value('franka_gripper', action='open')
        gripper_state(ctx, 3.0)
        lift_to(ctx, SAFE_Z, log)
    raise StepError(f'could not grasp the towel in {attempts} attempts')


def choose_park(pickup, region, min_gap=0.12):
    """Park where the towel was picked up unless that is within min_gap of the wiped region;
    then move it sideways (in y) away from the region, inside the grasp area."""
    x, y = pickup
    x0, x1, y0, y1 = region
    gap_x = max(x0 - x, 0.0, x - x1)
    gap_y = max(y0 - y, 0.0, y - y1)
    if max(gap_x, gap_y) >= min_gap:
        return (x, y)
    room_right = y0 - GRASP_Y[0]            # free table on the robot's right of the region
    room_left = GRASP_Y[1] - y1
    if room_right >= room_left:
        y_new = max(GRASP_Y[0] + 0.04, y0 - min_gap - 0.04)
    else:
        y_new = min(GRASP_Y[1] - 0.04, y1 + min_gap + 0.04)
    return (min(max(x, 0.40), 0.65), y_new)


def wipe_region(ctx, region, log, mode='carry', centre=None, spacing=0.04):
    """Scrub the rectangle region=(x0, x1, y0, y1) (gripper path) with the held towel.
    mode 'carry': lift to CARRY_Z, go over centre, lower the towel onto it (it piles up),
    blot, then scrub. mode 'drag': keep the towel's underside down (e.g. soap applied there)
    and drag it at contact height to the first waypoint."""
    x0, x1, y0, y1 = fit_region(ctx, region, log)
    cx, cy = centre if centre is not None else ((x0 + x1) / 2, (y0 + y1) / 2)
    cx, cy = min(max(cx, x0), x1), min(max(cy, y0), y1)
    pts = M.boustrophedon(x0, x1, y0, y1, spacing, 'y') + M.boustrophedon(x0, x1, y0, y1, spacing, 'x')
    if mode == 'carry':
        lift_clear(ctx, log)
        recentre(ctx, list(carry_xy(cx, cy)) + [CARRY_Z], log)
        M.move(ctx, [cx, cy, LOWER_Z], log=log)
        meas, _ = M.press_move(ctx, cx, cy, CONTACT_PROBE_Z, log=log)   # blot: find the contact height
        z_contact = max(meas[2], M.Z_FLOOR + M.PRESS_DEPTH)
        time.sleep(1.0)
    else:
        p = tcp(ctx)
        M.move(ctx, [p[0], p[1], 0.17], log=log)
        meas, _ = M.press_move(ctx, p[0], p[1], CONTACT_PROBE_Z, log=log)
        z_contact = max(meas[2], M.Z_FLOOR + M.PRESS_DEPTH)
        # drag in short legs so the pad never runs far ahead of tracking
        n = max(1, int(math.ceil(math.hypot(pts[0][0] - p[0], pts[0][1] - p[1]) / 0.10)))
        legs = [(p[0] + (pts[0][0] - p[0]) * i / n, p[1] + (pts[0][1] - p[1]) * i / n) for i in range(1, n)]
        pts = legs + pts
    log(f'scrub {len(pts)} waypoints over x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}] contact z {z_contact:.4f}')
    meas, z_contact = M.scrub(ctx, pts, z_contact, log=log)
    g = gripper_state(ctx, 1.0)
    if not g['is_grasped']:
        raise StepError('towel lost during scrubbing')
    return {'ok': True, 'region': (x0, x1, y0, y1), 'waypoints': len(pts), 'final_contact_z': z_contact}


def park_towel(ctx, park_xy, log):
    """Carry the held towel to park_xy, lower it there into a heap and release; arm to SAFE_Z."""
    lift_clear(ctx, log)
    _seg_moves(ctx, tcp(ctx), list(carry_xy(*park_xy)) + [CARRY_Z], log)
    M.move(ctx, [park_xy[0], park_xy[1], LOWER_Z], log=log)
    ctx.value('franka_gripper', action='open')
    gripper_state(ctx, 3.0)
    M.move(ctx, [park_xy[0], park_xy[1], SAFE_Z], log=log)
    return {'ok': True, 'park': park_xy}
