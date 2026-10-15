"""Paper towels: locate the tissue box, the sheet popping out of its slot and the waste bin from
the exterior cameras' depth (the wrist depth is unreliable up close); pull sheets and drop them.

Tested 2026-10-06: purple tissue box ~22x11x9.3 cm (top z ~0.182), long axis along x; one sheet
sticks out of the slot; pinch it from above with the fingers closing along y, pull up to z 0.47.
"""
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # sibling modules of this skill (motions, perception)
import motions as M      # noqa: E402
import perception as P   # noqa: E402

GRASP_FORCE = 20.0
TCP_ABOVE_BOX = 0.030     # pinch with the TCP at least this far above the box top (fingertips ~2 cm)
PULL_Z = 0.47             # lift to here: the sheet comes free (sheet ~20 cm long)
DROP_Z = 0.115            # release with the pinch nearly at the table: the sheet crumples there
                          # (released at 0.13 the hanging sheet folded over 5-7 cm to one side)
TRAVEL_Z = 0.45


def tissue_box(frame, near=(0.40, -0.17), half=0.18):
    """Box top height and footprint from purple-ish raised points (left + right depth)."""
    x0, y0 = near
    pts_all = []
    for cam in ('left', 'right'):
        pts, col = P.cloud(frame, cam, stride=2, colours=True)
        hsv = cv.cvtColor(col.reshape(-1, 1, 3), cv.COLOR_BGR2HSV).reshape(-1, 3)
        purple = (hsv[:, 0] > 120) & (hsv[:, 0] < 172) & (hsv[:, 1] > 70) & (hsv[:, 2] > 40)
        sel = purple & (np.abs(pts[:, 0] - x0) < half) & (np.abs(pts[:, 1] - y0) < half) & (pts[:, 2] > P.TABLE_Z + 0.03)
        pts_all.append(pts[sel])
    q = np.concatenate(pts_all)
    if len(q) < 200:
        return {'found': False}
    top_z = float(np.percentile(q[:, 2], 95))
    topq = q[q[:, 2] > top_z - 0.012]
    return {'found': True, 'top_z': top_z, 'centre': (float(np.median(topq[:, 0])), float(np.median(topq[:, 1]))),
            'x': (float(np.percentile(topq[:, 0], 3)), float(np.percentile(topq[:, 0], 97))),
            'y': (float(np.percentile(topq[:, 1], 3)), float(np.percentile(topq[:, 1], 97)))}


def protruding_sheet(frame, box, margin=0.045, exclude_box=None):
    """White points above the box top over the box (+margin for a sheet draping over an edge): the
    sheet to pull. exclude_box=(x0, x1, y0, y1) is never searched (the waste bin: paper sticking out
    of it was once pulled instead of a fresh sheet). Returns the sheet's upper part."""
    pts_all = []
    for cam in ('left', 'right'):
        pts, col = P.cloud(frame, cam, stride=1, colours=True)
        hsv = cv.cvtColor(col.reshape(-1, 1, 3), cv.COLOR_BGR2HSV).reshape(-1, 3)
        white = (hsv[:, 2] > 160) & (hsv[:, 1] < 45)
        sel = white & (pts[:, 0] > box['x'][0] - margin) & (pts[:, 0] < box['x'][1] + margin) & \
            (pts[:, 1] > box['y'][0] - margin) & (pts[:, 1] < box['y'][1] + margin) & \
            (pts[:, 2] > box['top_z'] + 0.012) & (pts[:, 2] < box['top_z'] + 0.20)
        if exclude_box is not None:
            ex0, ex1, ey0, ey1 = exclude_box
            sel &= ~((pts[:, 0] > ex0) & (pts[:, 0] < ex1) & (pts[:, 1] > ey0) & (pts[:, 1] < ey1))
        pts_all.append(pts[sel])
    q = np.concatenate(pts_all)
    if len(q) < 40:
        return {'found': False, 'n': int(len(q))}
    hi = q[q[:, 2] > np.percentile(q[:, 2], 70)]
    out = {'found': True, 'n': int(len(q)), 'top_z': float(np.percentile(q[:, 2], 95)),
           'upper': (float(hi[:, 0].mean()), float(hi[:, 1].mean()), float(hi[:, 2].mean())),
           'centre': (float(q[:, 0].mean()), float(q[:, 1].mean()))}
    ux, uy = out['upper'][:2]
    if not (box['x'][0] - 0.04 <= ux <= box['x'][1] + 0.04 and box['y'][0] - 0.04 <= uy <= box['y'][1] + 0.04):
        return {'found': False, 'n': int(len(q)), 'reason': f'sheet upper part {np.round(out["upper"], 3).tolist()} is not over the box'}
    return out


def grasp_point(box, sheet):
    """TCP target for pinching the sheet: its upper part, low enough to catch it, clear of the box."""
    x, y, zu = sheet['upper']
    z = max(box['top_z'] + TCP_ABOVE_BOX, min(zu - 0.01, sheet['top_z'] - 0.025))
    return (x, y, z)


def gripper_after(ctx, wait_s=4.0, settle_s=0.6):
    """Gripper state once a just-sent command has run: the status read right after a grasp can
    still show the old open width (dry run 2: 80 mm, not grasped, while it was about to close)."""
    time.sleep(settle_s)
    t0 = time.time()
    while True:
        g = ctx.value('franka_status')['gripper']
        if not g.get('is_moving') or time.time() - t0 > wait_s:
            return g
        time.sleep(0.3)


def travel(ctx, goal, log, max_seg=0.35):
    """Straight move to goal split into <= max_seg pieces (move_pose takes <= 0.40 m)."""
    p = np.asarray(ctx.value('franka_status')['tcp_position'], float)
    goal = np.asarray(goal, float)
    n = max(1, int(np.ceil(np.linalg.norm(goal - p) / max_seg)))
    for i in range(1, n + 1):
        M.move(ctx, p + (goal - p) * i / n, log=log)


def pull_sheet(ctx, target, log):
    """Pinch the sheet at target (x, y, z) and pull it up to PULL_Z. Returns gripper state."""
    x, y, z = target
    ctx.value('franka_gripper', action='open')
    travel(ctx, [x, y, max(0.35, z + 0.10)], log)
    M.move(ctx, [x, y, z + 0.05], log=log)
    M.move(ctx, [x, y, z], tol=5, timeout=10, log=log)
    ctx.value('franka_gripper', action='grasp', width=0.0, force=GRASP_FORCE, speed=0.05)
    g = gripper_after(ctx)
    log(f'pinch: width {g["width"] * 1000:.1f} mm is_grasped {g["is_grasped"]}')
    M.move(ctx, [x, y, 0.30], log=log)
    M.move(ctx, [x, y, PULL_Z], log=log)
    return g


def drop_at(ctx, xy, log):
    travel(ctx, [xy[0], xy[1], TRAVEL_Z], log)
    M.move(ctx, [xy[0], xy[1], DROP_Z], log=log)
    ctx.value('franka_gripper', action='open')
    gripper_after(ctx, 2.0)
    M.move(ctx, [xy[0], xy[1], 0.30], log=log)


def bin_opening(frame, region=(0.45, 0.82, -0.46, -0.10), rim_h=0.12):
    """Waste bin: the raised rim ring (bag over the bin edge) and the low hole inside it.
    Returns the opening centre (x, y), its radius estimate and the rim top z."""
    X, Y = P.grid()
    hm = np.nan_to_num(P.height_map(frame, cams=('left', 'right')), nan=0.0)
    reg = P.region_mask(*region)
    rim = (hm > rim_h) & (hm < 0.25) & reg      # < 25 cm: an arm above the bin is not the rim
    rim = cv.morphologyEx(rim.astype(np.uint8), cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv.connectedComponentsWithStats(rim)
    if n < 2:
        return {'found': False}
    i = 1 + int(np.argmax(st[1:, cv.CC_STAT_AREA]))
    ring = (lab == i).astype(np.uint8)
    pts = cv.findNonZero(ring)
    hull = cv.convexHull(pts)
    filled = np.zeros_like(ring)
    cv.fillConvexPoly(filled, hull, 1)
    inner = cv.erode(filled, np.ones((7, 7), np.uint8))        # stay off the hull border
    hole = (inner > 0) & (hm < rim_h - 0.03)
    hole = cv.morphologyEx(hole.astype(np.uint8), cv.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if hole.sum() < 50:
        return {'found': False, 'reason': 'no hole inside the rim'}
    rim_top = float(np.percentile(hm[ring > 0], 90)) + P.TABLE_Z
    r = float(np.sqrt(hole.sum() / np.pi) * P.MAP_RES)
    inner_c = (float(X[inner > 0].mean()), float(Y[inner > 0].mean()))      # rim hull centre
    centre = (float(X[hole].mean()), float(Y[hole].mean()))
    if r < 0.03 or np.hypot(centre[0] - inner_c[0], centre[1] - inner_c[1]) > 0.04:
        centre = inner_c                     # hole hidden by paper already in the bin
    return {'found': True, 'centre': centre, 'radius': r, 'rim_top_z': rim_top, 'hull_centre': inner_c,
            'rim_x': (float(X[ring > 0].min()), float(X[ring > 0].max())), 'rim_y': (float(Y[ring > 0].min()), float(Y[ring > 0].max()))}


def paper_pile(frame, region, margin=0.08):
    """White raised points (paper) on/near region=(x0, x1, y0, y1): centre, upper centre, top z."""
    x0, x1, y0, y1 = region
    pts_all = []
    for cam in ('left', 'right'):
        pts, col = P.cloud(frame, cam, stride=1, colours=True)
        hsv = cv.cvtColor(col.reshape(-1, 1, 3), cv.COLOR_BGR2HSV).reshape(-1, 3)
        white = (hsv[:, 2] > 150) & (hsv[:, 1] < 50)
        sel = white & (pts[:, 0] > x0 - margin) & (pts[:, 0] < x1 + margin) & (pts[:, 1] > y0 - margin) & \
            (pts[:, 1] < y1 + margin) & (pts[:, 2] > P.TABLE_Z + 0.006) & (pts[:, 2] < P.TABLE_Z + 0.16)
        pts_all.append(pts[sel])
    q = np.concatenate(pts_all)
    if len(q) < 200:
        return {'found': False, 'n': int(len(q))}
    hi = q[q[:, 2] > np.percentile(q[:, 2], 60)]
    return {'found': True, 'n': int(len(q)), 'centre': (float(q[:, 0].mean()), float(q[:, 1].mean())),
            'upper': (float(hi[:, 0].mean()), float(hi[:, 1].mean())), 'top_z': float(np.percentile(q[:, 2], 90))}


def paper_on_fingers(frame, tcp, below=(0.012, 0.05), half=(0.04, 0.065), min_pts=120, min_cam=60):
    """Is paper hanging from the (open) fingers? Non-dark side-camera depth points (white or
    soy-stained paper; the bin bag is black) in a thin band (5 cm) right below the fingertips.
    Look from as high as reachable: a full bin's paper heap inside the band reads as 'stuck'
    (false alarm 12:07). Returns (stuck, point count)."""
    x, y, z = tcp
    n = per_cam = 0
    for cam in ('left', 'right'):
        if cam not in frame['depth'] or cam not in frame['meta']['cameras']:
            continue
        pts, col = P.cloud(frame, cam, stride=1, colours=True)
        hsv = cv.cvtColor(col.reshape(-1, 1, 3), cv.COLOR_BGR2HSV).reshape(-1, 3)
        light = hsv[:, 2] > 110
        sel = light & (np.abs(pts[:, 0] - x) < half[0]) & (np.abs(pts[:, 1] - y) < half[1]) & \
            (pts[:, 2] > z - below[1]) & (pts[:, 2] < z - below[0])
        n += int(sel.sum())
        per_cam = max(per_cam, int(sel.sum()))
    return (n >= min_pts or (n > 0 and per_cam >= min_cam)), n


def scrape_fingers(ctx, bin_info, log):
    """Wipe paper stuck to the open fingers on the bin's inner walls (wet paper sticks after
    'open'): put the fingertips ~1.4 cm below the rim 2 cm inside a wall, press 1.5 cm towards
    the wall, lift. Walls: +y side (finger on +y) and the near (-x) side (both fingers)."""
    bx, by = bin_info['centre']
    rim = bin_info['rim_top_z']
    wall_y = bin_info['rim_y'][1] - 0.032          # +y inner wall (rim ring includes the bag fold)
    wall_x = bin_info['rim_x'][0] + 0.035          # near (-x) inner wall
    passes = [((bx, wall_y - 0.06), (bx, wall_y - 0.045)),      # +y finger against the +y wall
              ((wall_x + 0.03, by), (wall_x + 0.015, by))]      # finger fronts against the near wall
    for start_xy, press_xy in passes:
        M.move(ctx, [start_xy[0], start_xy[1], rim + 0.075], log=log)
        M.move(ctx, [start_xy[0], start_xy[1], rim - 0.005], tol=6, timeout=8, log=log)
        try:
            ctx.value('franka_move_pose', position=[press_xy[0], press_xy[1], rim - 0.005], quaternion=M.Q_DOWN,
                      tolerance_mm=10, timeout_s=6)
        except RuntimeError as exc:
            err, _ = M._error_state(exc)
            if err is None or err.get('control_uncertain') or err.get('type') not in ('TimeoutError', 'TargetRejected'):
                raise
            log(f'scrape press stopped ({err.get("type")})')
        M.move(ctx, [press_xy[0], press_xy[1], rim + 0.075], log=log)
    log('fingers scraped on bin walls')


def dispose(ctx, bin_info, log, recentre=None):
    """Carry the held wad over the bin opening, release it with the fingertips just above the
    rim, lift clear, scrape anything stuck to the fingers off on the bin walls, then (fingers
    free) close by force and push the paper down."""
    bx, by = bin_info['centre']
    rim = bin_info['rim_top_z']
    p = ctx.value('franka_status')['tcp_position']
    M.move(ctx, [p[0], p[1], max(p[2], TRAVEL_Z)], log=log)
    if recentre is not None:
        recentre(ctx, [bx, by, TRAVEL_Z], log)
    else:
        M.move(ctx, [bx, by, TRAVEL_Z], log=log)
    M.move(ctx, [bx, by, rim + 0.035], log=log)
    ctx.value('franka_gripper', action='open')
    gripper_after(ctx, 2.0)
    time.sleep(0.5)
    # lift clear of the released paper (closing next to it re-pinches it), make sure nothing stays
    # stuck to the fingers, and only then close (by FORCE: a position close onto stuck wet paper
    # faulted the gripper bridge) and push down
    for attempt in range(3):
        # look from the side cameras with the fingers well above the bin's paper heap
        # as high as reachable over the bin (rim + 20 cm was out of reach; 0.45 works at x - 3 cm)
        M.move(ctx, [bx - 0.03, by, min(rim + 0.20, TRAVEL_Z)], log=log)
        frame = P.load_frame(ctx.value('franka_observe', cameras=['left', 'right'])['frame_id'])
        stuck, n = paper_on_fingers(frame, ctx.value('franka_status')['tcp_position'])
        log(f'paper on fingers after release: {stuck} ({n} pts, frame {frame["id"]})')
        if not stuck:
            break
        if attempt == 2:
            raise RuntimeError('paper still stuck to the fingers after 2 scrapes')
        scrape_fingers(ctx, bin_info, log)       # only when it did not come off (operator)
    M.move(ctx, [bx, by, rim + 0.10], log=log)
    ctx.value('franka_gripper', action='grasp', width=0.0, force=10.0, speed=0.05)
    gripper_after(ctx, 2.0)
    try:
        r = ctx.value('franka_move_pose', position=[bx, by, rim - 0.025], quaternion=M.Q_DOWN,
                      tolerance_mm=15, timeout_s=8)
        log(f'push paper into bin: reached z {r["measured_position"][2]:.3f}')
    except RuntimeError as exc:
        err, state = M._error_state(exc)
        if err is None or err.get('control_uncertain') or err.get('type') != 'TimeoutError':
            raise
        log(f'push paper into bin: stalled at {state["tcp_position"][2]:.3f}')
    M.move(ctx, [bx, by, TRAVEL_Z], log=log)
    ctx.value('franka_gripper', action='open')


def leftover_paper(frame, ref, exclude, roi=(0.33, 0.80, -0.42, 0.42), min_area_m2=6e-4):
    """Paper left on the table: cells that rose > 5 mm (either side camera's depth) or changed
    appearance in BOTH side cameras' top-down maps, versus the clean reference from home.
    exclude: map mask (tissue box, bin). Returns blobs [{centre, high, area_m2}] largest first."""
    X, Y = P.grid()
    rise = np.zeros(X.shape, bool)
    rise_h = np.zeros(X.shape, np.float32)
    for cam in ('left', 'right'):
        h = np.nan_to_num(P.height_map(frame, cams=(cam,)), nan=0.0)
        h0 = np.nan_to_num(P.height_map(ref, cams=(cam,)), nan=0.0)
        h = cv.medianBlur((np.clip(h, 0, 0.25) * 1000).astype(np.float32), 5) / 1000
        h0 = cv.medianBlur((np.clip(h0, 0, 0.25) * 1000).astype(np.float32), 5) / 1000
        rise |= (h - h0) > 0.005
        rise_h = np.maximum(rise_h, h - h0)
    look = None
    for cam in ('left', 'right'):
        g, v, _ = P.grey_map(frame, cam)
        g0, v0, _ = P.grey_map(ref, cam)
        d = np.abs(cv.GaussianBlur(g, (5, 5), 0) - cv.GaussianBlur(g0, (5, 5), 0)) > 15
        d &= v & v0
        look = d if look is None else (look & d)
    cand = (rise | look) & P.region_mask(*roi) & ~exclude
    cand = cv.morphologyEx(cand.astype(np.uint8), cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv.morphologyEx(cand, cv.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv.connectedComponentsWithStats(cand)
    hcur = np.nan_to_num(P.height_map(frame), nan=0.0)
    blobs = []
    for i in range(1, n):
        area = st[i, cv.CC_STAT_AREA] * P.MAP_RES ** 2
        if area < min_area_m2:
            continue
        b = lab == i
        hi = b & (hcur >= np.percentile(hcur[b], 70))
        rise_max = float(np.percentile(np.where(b, rise_h, 0)[b], 95))
        blobs.append({'area_m2': float(area), 'rise': rise_max, 'centre': (float(X[b].mean()), float(Y[b].mean())),
                      'high': (float(X[hi].mean()), float(Y[hi].mean())),
                      'top_z': float(np.percentile(hcur[b], 90)) + P.TABLE_Z})
    return sorted(blobs, key=lambda b: -b['area_m2'])


def pick_paper(ctx, x, y, log, z=0.10):
    """Pinch paper lying on the table at (x, y): press the open fingers to the floor height (stall on
    paper within 10 mm keeps pressing), grasp 20 N, lift to 0.30. True if still held after the lift."""
    ctx.value('franka_gripper', action='open')
    gripper_after(ctx, 2.0)
    travel(ctx, [x, y, 0.30], log)
    M.move(ctx, [x, y, 0.16], log=log)
    try:
        r = ctx.value('franka_move_pose', position=[x, y, z], quaternion=M.Q_DOWN, tolerance_mm=10, timeout_s=8)
        log(f'pick ({x:.3f},{y:.3f}): pinch height {r["measured_position"][2]:.4f}')
    except RuntimeError as exc:
        err, _ = M._error_state(exc)
        if err is None or err.get('control_uncertain') or err.get('type') != 'TimeoutError':
            raise
    ctx.value('franka_gripper', action='grasp', width=0.0, force=20.0, speed=0.05)
    gripper_after(ctx)
    M.move(ctx, [x, y, 0.30], log=log)
    g = gripper_after(ctx, 1.0, 0.2)
    log(f'pick ({x:.3f},{y:.3f}): held {g["is_grasped"]} width {g["width"] * 1000:.1f} mm')
    if not g['is_grasped']:
        ctx.value('franka_gripper', action='open')
    return g['is_grasped']
