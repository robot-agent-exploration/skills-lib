"""find_the_cube: watch the operator hide the cube under a green cup and shuffle the cups
from a top-down wrist view, track the cups at 30 Hz, lift the cup with the cube away,
then grasp and lift the cube.

Run from the workspace root:  python <skill>/code/controller.py   (or code/run.py, the library entry)
Steps (each returns {'ok', ...}): goto_watch, find_cube (depth), wait_shuffle (operator),
identify_cup (stream tracking), remove_cup, grasp_cube.

Library import (2026-10-15): ROOT is the working directory (the workspace root, which holds
api/, runtime/frames/ and output/) instead of two levels above this file, and `api.stream`
is imported where it is used, so the perception helpers import without the harness.
"""
import json
import math
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path.cwd()                 # workspace root: api/, runtime/frames/<id>/, output/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shuffle  # noqa: E402
import tracker  # noqa: E402

Q_DOWN = [1.0, 0.0, 0.0, 0.0]
Q_ALONG_X = [0.7071068, 0.7071068, 0.0, 0.0]   # yaw 90: fingers and the 20 cm hand body along base x,
                                                # clear of neighbour cups in a row along y
WATCH_POS = [0.47, 0.0, 0.62]     # wrist camera sees x 0.35..0.95, y -0.5..0.5 from here
ROI = (0, 0, 1280, 520)           # wrist image region above the gripper fingers
TABLE_Z = 0.09
CUBE_Z = (0.103, 0.16)            # cube top ~0.12 (3 cm cube)
CUP_TOP_MIN_Z = 0.15              # inverted cup top ~0.183 (9.3 cm cup)
N_CUPS = 4
COVER_PX = 45                     # UNUSED (stale): identification uses shuffle.COVER_PX = 65
COVER_FRAMES = 12                 # UNUSED: shuffle.COVER_FRAMES
CUP_GRASP_BELOW_TOP = 0.012       # fingertips 1.2 cm below the cup top: 3 cm / 15 N pinched a cube
CUP_FORCE = 8                     # sitting high inside the cup through the wall (trial 8)
CUBE_FORCE = 20
SMALL_FORCE = 12                  # objects under 45 mm
GENTLE_FORCE = 6                  # re-grasp force after the grip closed in (red cube slid out at 20 N)
SLIP_MM = 4                       # finger closing during the 2 cm test lift that counts as slipping
HAND_HALF_LENGTH = 0.105         # Franka Hand body ~21 cm along the finger axis ...
HAND_HALF_WIDTH = 0.03           # ... and ~6 cm across
HAND_CLEARANCE = 0.04            # hand-body rectangle to other cups' centres (cup radius 3 cm + 1 cm)
RIDGE_DL = 12                    # brightness step (Lab L) between the halves of a ridged top
DROP_ABOVE_GRASP = 0.02          # release the moved cup with its rim ~2 cm above the table
DROP_CLEARANCE = 0.09            # dropped cup centre to other cups / the object (rims touch at 7.5 cm)
SAVE_RECORDING = True            # copy the shuffle's wrist frames to output/recordings/
CLOSEUP_BACK = 0.10              # close-up TCP this far towards the robot from the cup ...
CLOSEUP_Z = 0.50                 # ... at this height (camera ~0.4 m from the cup top)
SIDE_LOOK = 0.10                 # sideways offset of the second look after a blocked descent
LOG = []


def log(*a):
    line = ' '.join(str(x) for x in a)
    LOG.append(line)
    print(line, flush=True)


def tool_error(exc):
    """The error dict of a failed MCP tool call (RuntimeError text 'MCP tool failed: {...}')."""
    text = str(exc)
    try:
        return json.loads(text.split('MCP tool failed: ', 1)[1]).get('error', {})
    except (IndexError, ValueError, AttributeError):
        return {}


def move(ctx, pos, q=None, timeout=30, tol=None, accept_mm=None):
    """Straight-line TCP move. accept_mm: a timeout that ends within this distance of the
    target counts as reached (the impedance controller keeps a ~5 mm offset near contact)."""
    args = dict(position=[float(v) for v in pos], timeout_s=timeout)
    args['quaternion'] = q or Q_DOWN
    if tol:
        args['tolerance_mm'] = tol
    try:
        r = ctx.value('franka_move_pose', **args)
    except RuntimeError as e:
        err = tool_error(e)
        dist = (err.get('robot_state') or {}).get('distance_to_target_mm')
        if accept_mm is not None and err.get('type') == 'TimeoutError' and dist is not None and dist < accept_mm:
            log(f'move: accepted {dist:.1f} mm short of {np.round(pos, 3).tolist()}')
            return {'reached': False, 'near_miss_mm': dist}
        raise
    if not r.get('reached', False):
        raise RuntimeError(f'move not reached: {r}')
    return r


def move_far(ctx, pos, q=None, step=0.35, **kw):
    """move() split into straight segments of at most `step` m (the tool refuses > 0.4 m)."""
    cur = np.array(tcp(ctx))
    pos = np.array(pos, float)
    n = max(1, int(math.ceil(np.linalg.norm(pos - cur) / step)))
    for i in range(1, n + 1):
        r = move(ctx, (cur + (pos - cur) * i / n).tolist(), q, **kw)
    return r


def tcp(ctx):
    return ctx.value('franka_status')['tcp_position']


def safe_move(ctx, pos, q=None):
    """Lift to a safe height first, then travel, then descend."""
    cur = tcp(ctx)
    hi = max(cur[2], pos[2], 0.30)
    if cur[2] < hi - 0.005:
        move(ctx, [cur[0], cur[1], hi], q)
    move(ctx, [pos[0], pos[1], hi], q)
    if pos[2] < hi - 0.005:
        move(ctx, pos, q)


def observe(ctx):
    r = ctx.value('franka_observe', cameras=['wrist'])
    d = ROOT / r['directory']
    meta = json.loads((d / 'meta.json').read_text())['cameras']['wrist']
    image = cv.imread(str(d / 'wrist.jpg'))
    depth = cv.imread(str(d / 'wrist_depth.png'), cv.IMREAD_UNCHANGED).astype(float) / 1000.0
    return dict(id=r['frame_id'], image=image, depth=depth, K=np.array(meta['K']),
                T=np.array(meta['T_base_cam']))


def backproject(frame, step=2):
    """Base-frame xyz for every step-th pixel: arrays (H', W', 3) and valid mask."""
    d = frame['depth'][::step, ::step]
    K, T = frame['K'], frame['T']
    vs, us = np.mgrid[0:frame['depth'].shape[0]:step, 0:frame['depth'].shape[1]:step]
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d], -1) @ T[:3, :3].T + T[:3, 3]
    return pc, d > 0, us, vs


def local_table(z, ok, block=48):
    """Table height map: per-block 40th percentile of usable depth, bilinear upsampled.
    Wrist depth on the white table reads 1-3 cm high with a bias that varies across the image."""
    H, W = z.shape
    hb, wb = -(-H // block), -(-W // block)
    grid = np.full((hb, wb), np.nan)
    for i in range(hb):
        for j in range(wb):
            zz = z[i * block:(i + 1) * block, j * block:(j + 1) * block]
            mm = ok[i * block:(i + 1) * block, j * block:(j + 1) * block]
            if mm.sum() > 0.2 * mm.size:
                grid[i, j] = np.percentile(zz[mm], 40)
    if np.all(np.isnan(grid)):
        return np.full_like(z, TABLE_Z)
    grid[np.isnan(grid)] = np.nanmedian(grid)
    up = cv.resize(grid.astype(np.float32), (wb * block, hb * block), interpolation=cv.INTER_LINEAR)
    return up[:H, :W].astype(float)


def find_cube(frame, near_xy=None, radius=0.12):
    """The object (cube or anything that fits under a cup) from depth: non-green points
    1.5-10 cm above the local table, 12-80 mm across (1.0 cm threshold if nothing at 1.5).
    Returns pixel, xy, z, top_z, table_z, yaw (rad, in [-pi/4, pi/4]), size_mm, area
    (depth px at step 2 = half-res px) and Lab colour, or None."""
    pc, valid, us, vs = backproject(frame)
    hsv = cv.cvtColor(frame['image'], cv.COLOR_BGR2HSV)
    green = cv.inRange(hsv, tracker.HSV_LO, tracker.HSV_HI)
    green = cv.dilate(green, np.ones((25, 25), np.uint8))[::2, ::2] > 0
    base = valid & ~green & (vs < ROI[3])
    if near_xy is not None:
        base &= np.hypot(pc[..., 0] - near_xy[0], pc[..., 1] - near_xy[1]) < radius
    table = local_table(pc[..., 2], base)
    rel = pc[..., 2] - table
    for min_rel in (0.015, 0.010):
        sel = base & (rel > min_rel) & (rel < 0.10)
        sel = cv.morphologyEx(sel.astype(np.uint8), cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv.connectedComponentsWithStats(sel)
        best, best_score = None, 0
        for i in range(1, n):
            a = stats[i, cv.CC_STAT_AREA]
            if a < 25:
                continue
            pts = pc[lab == i]
            (_, _), (w, h), _ = cv.minAreaRect((pts[:, :2] * 1000).astype(np.float32))
            if max(w, h) > 80 or max(w, h) < 12 or min(w, h) < 8:     # table patches / noise
                continue
            u, v = us[lab == i].mean(), vs[lab == i].mean()
            if v > ROI[3] - 30 or u < 30 or u > frame['image'].shape[1] - 30:
                continue                              # flying depth pixels at the finger / image edges
            cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
            if not (0.28 <= cx <= 0.80 and -0.45 <= cy <= 0.45):
                continue                              # off the reachable table area
            if a > best_score:
                best, best_score = i, a
        if best is not None:
            break
    if best is None:
        return None
    m = lab == best
    pts = pc[m]
    xy = np.median(pts[:, :2], 0)
    (_, _), (w, h), ang = cv.minAreaRect((pts[:, :2] * 1000).astype(np.float32))
    yaw = math.radians(((ang + 45) % 90) - 45)
    rgb = frame['image'][::2, ::2][:m.shape[0], :m.shape[1]][m]
    rect = [float(w), float(h), float(ang)]          # side w lies along angle (deg from base x)
    labc = cv.cvtColor(rgb.reshape(-1, 1, 3), cv.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    return dict(px=[float(us[m].mean()), float(vs[m].mean())], xy=xy.tolist(),
                z=float(np.median(pts[:, 2])), top_z=float(np.percentile(pts[:, 2], 90)),
                table_z=float(np.median(table[m])), yaw=yaw, size_mm=[float(w), float(h)],
                area=int(best_score), lab=np.median(labc, 0).round(1).tolist(), rect=rect)


def count_cups(image, default=N_CUPS):
    """Number of cups from the green blobs, touching cups counted by area (median blob = 1 cup)."""
    mask = tracker.green_mask(image, ROI)
    n, _, stats, _ = cv.connectedComponentsWithStats(mask)
    areas = [stats[i, cv.CC_STAT_AREA] for i in range(1, n) if stats[i, cv.CC_STAT_AREA] >= 300]
    if not areas:
        return default
    unit = float(np.clip(np.median(areas), 1500, 2600))     # half-res px per cup (trials 1800-2700)
    k = int(sum(max(1, round(a / unit)) for a in areas))
    return k if 2 <= k <= 6 else default


def cup_clusters(image, seeds=None, k=N_CUPS):
    """k cup centres (full-res px) from green pixels; seeded Lloyd or cv.kmeans."""
    mask = tracker.green_mask(image, ROI)
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    if len(pts) < 200 * k:
        return None, None
    if seeds is None:
        crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, _, c = cv.kmeans(pts, k, None, crit, 8, cv.KMEANS_PP_CENTERS)
    else:
        c, _ = tracker.kmeans_step(pts.astype(float), np.array(seeds) * tracker.SCALE, iters=10)
    c = np.array(c, float) / tracker.SCALE
    d = ((pts[:, None, :] - c[None] * tracker.SCALE) ** 2).sum(-1).argmin(1)
    counts = np.bincount(d, minlength=k)
    return c, counts


# ---------------------------------------------------------------- steps

def goto_watch(ctx):
    """Open and go to the watch pose. Rising straight to 0.62 m off-centre while turning was
    refused by the IK (joint step > 0.2 rad), so: up to 0.30 m without turning, then via a
    mid-height point near the centre while turning to straight-down."""
    ctx.value('franka_gripper', action='open')
    st = ctx.value('franka_status')
    cur, q = st['tcp_position'], st['tcp_quaternion']
    near = np.hypot(cur[0] - WATCH_POS[0], cur[1] - WATCH_POS[1]) < 0.03 and abs(cur[2] - WATCH_POS[2]) < 0.03
    if not near:
        if cur[2] < 0.30:
            move(ctx, [cur[0], cur[1], 0.30], q)
        move(ctx, [0.49, float(np.clip(cur[1], -0.1, 0.1)), 0.45])
    move(ctx, WATCH_POS)
    return {'ok': True, 'tcp': tcp(ctx)}


def step_find_cube(ctx):
    fr = observe(ctx)
    cube = find_cube(fr)
    if cube is None:
        return {'ok': False, 'why': 'object not visible from the watch pose', 'frame': fr['id']}
    n_cups = count_cups(fr['image'])
    centres, counts = cup_clusters(fr['image'], k=n_cups)
    log('cube', cube, 'cups', n_cups, 'px', None if centres is None else np.round(centres).tolist())
    model = {'size_mm': cube['size_mm'], 'area': cube['area'], 'lab': cube['lab'],
             'height': cube['top_z'] - cube['table_z']}
    if colourful(model):                   # depth can see only part of a small object: size it by colour
        blob = colour_footprint(fr, model, cube)
        if blob is not None:
            (bw, bh), barea = blob
            model['size_mm'] = [max(cube['size_mm'][0], bw), max(cube['size_mm'][1], bh)]
            model['area'] = max(cube['area'], barea)
            model['height'] = max(model['height'], 0.5 * min(bw, bh) / 1000.0)
    log('object model', model)
    return {'ok': True, 'cube': cube, 'model': model, 'n_cups': n_cups, 'frame': fr['id']}


def colour_footprint(frame, model, cube):
    """Footprint (w, h mm on the table) and half-res area of the object's colour blob near it."""
    u0, v0 = int(cube['px'][0]), int(cube['px'][1])
    x0, y0 = max(0, u0 - 100), max(0, v0 - 100)
    win = frame['image'][y0:v0 + 100, x0:u0 + 100]
    d = cv.cvtColor(win, cv.COLOR_BGR2LAB).astype(float) - np.array(model['lab'])
    like = (np.sqrt((0.3 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2) < 35).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(like)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == k)
    if len(xs) < 40:
        return None
    box = cv.boxPoints(cv.minAreaRect(np.stack([xs + x0, ys + y0], 1).astype(np.float32)))
    pts = [pixel_to_plane(frame, u, v, TABLE_Z + 0.015) for u, v in box]
    w = 1000 * np.linalg.norm(pts[1] - pts[0])
    h = 1000 * np.linalg.norm(pts[2] - pts[1])
    return (float(w), float(h)), int(len(xs) / 4)


def wait_shuffle(ctx, object_name='white cube'):
    """object_name only changes the request text (the session always sent 'white cube',
    also for the red wedge in trials 10-11)."""
    t0 = int(time.time() * 1000)
    reply = ctx.value('request_operator', kind='reset', timeout_s=900, message=(
        f'Robot is watching from above. Please hide the {object_name} in one of the green cups '
        '(e.g. drop it into an upright cup and turn that cup over, or put a cup over it) and '
        'shuffle the cups, keeping all cups on the table in the area below the gripper. '
        'Step back and reply "done" when finished.'))
    t1 = int(time.time() * 1000)
    log('operator reply', reply)
    return {'ok': True, 't0': t0, 't1': t1, 'reply': reply}


def wrist_camera(frame_id):
    """K and pose of the wrist camera in a saved observation (the arm is still while watching)."""
    m = json.loads((ROOT / 'runtime' / 'frames' / frame_id / 'meta.json').read_text())['cameras']['wrist']
    return np.array(m['K']), np.array(m['T_base_cam'])


def identify_cup(t0, t1, cube_px, cam_frame, frames=None, depth_frames=None, debug_path=None,
                 model=None, n_cups=N_CUPS):
    """Which cup holds the cube after the shuffle (see shuffle.py). cam_frame: the observation
    taken at the watch pose; frames / depth_frames default to the live stream over [t0, t1]."""
    if frames is None or depth_frames is None:
        from api import stream            # the harness's 10-minute recording (runtime/stream)
    if frames is None:
        frames = stream.frames('wrist', since_ms=t0, until_ms=t1)
    if depth_frames is None:
        depth_frames = stream.frames('wrist', since_ms=t0, until_ms=t1, depth=True)
    if len(frames) < 30:
        return {'ok': False, 'why': f'only {len(frames)} wrist frames in the shuffle window'}
    K, T = wrist_camera(cam_frame)
    detect = lambda image, depth: find_cube(dict(image=image, depth=depth, K=K, T=T))
    r = shuffle.identify(frames, depth_frames, cube_px, detect, ROI, n_cups, log, model=model)
    if debug_path is not None:
        shuffle.save_sheet(frames, r, debug_path, ROI)
    log('identify:', r.get('why') or f"cup {r['cover_id']} -> px {np.round(r['cup_px'], 1).tolist()}",
        'events', r['events'])
    return r


def save_recording(t0, t1, dest):
    """Copy the shuffle's wrist frames out of the 10-minute stream for later analysis."""
    import shutil
    from api import stream
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    frames = stream.frames('wrist', since_ms=t0, until_ms=t1)
    frames += stream.frames('wrist', since_ms=t0, until_ms=t1, depth=True)
    for _, path in frames:
        shutil.copy2(path, dest / path.name)
    return len(frames)


def locate_cups_now(ctx, final_px, cover_id):
    fr = observe(ctx)
    centres, counts = cup_clusters(fr['image'], seeds=final_px)
    if centres is None:
        return {'ok': False, 'why': 'cups not visible now', 'frame': fr['id']}
    drift = np.hypot(*(centres - np.array(final_px)).T)
    pc, valid, us, vs = backproject(fr, step=1)
    cups = []
    for k, (u, v) in enumerate(centres):
        u, v = int(u), int(v)
        win = pc[v - 12:v + 13, u - 12:u + 13].reshape(-1, 3)
        ok = valid[v - 12:v + 13, u - 12:u + 13].reshape(-1) & (win[:, 2] > CUP_TOP_MIN_Z)
        if ok.sum() < 20:
            return {'ok': False, 'why': f'no depth on cup {k} top', 'frame': fr['id']}
        cups.append(np.median(win[ok], 0).tolist())
    log('cups now', np.round(cups, 3).tolist(), 'drift px', drift.round(1).tolist())
    if drift.max() > 40:
        return {'ok': False, 'why': f'cups moved after the shuffle (drift {drift.max():.0f}px)',
                'frame': fr['id']}
    return {'ok': True, 'cups': cups, 'target': cups[cover_id], 'frame': fr['id']}


def wait_gripper(ctx, timeout=6.0):
    """franka_gripper replies before the fingers move: wait until they start (or 1.5 s pass),
    then until the width stops changing."""
    t_start = time.time()
    w0, last, stable, started = None, None, 0, False
    g = ctx.value('franka_status')['gripper']
    while time.time() - t_start < timeout:
        w = g['width']
        w0 = w if w0 is None else w0
        started = started or g.get('is_moving') or abs(w - w0) > 0.003
        if started or time.time() - t_start > 1.5:
            if not g.get('is_moving') and last is not None and abs(w - last) < 0.0005:
                stable += 1
                if stable >= 2:
                    return g
            else:
                stable = 0
        last = w
        time.sleep(0.15)
        g = ctx.value('franka_status')['gripper']
    return g


def housing_clearance(center, phi, others, half=HAND_HALF_LENGTH, half_w=HAND_HALF_WIDTH):
    """Smallest distance from other cups' centres to the hand body, a rectangle centred on the
    TCP with its long axis at angle phi (trial 1: the body rested on cups 10-11 cm along it)."""
    d = np.array([math.cos(phi), math.sin(phi)])
    n = np.array([-d[1], d[0]])
    best = 1.0
    for o in others:
        v = np.array(o[:2]) - np.array(center[:2])
        da = max(0.0, abs(float(v @ d)) - half)
        db = max(0.0, abs(float(v @ n)) - half_w)
        best = min(best, float(math.hypot(da, db)))
    return best


CAM_OFFSET_TCP = (-0.071, 0.029)  # wrist camera from the TCP in TCP x/y (it rides behind the fingers)
CAM_CLEARANCE = 0.08              # camera centre to other cups' centres when the TCP is low


def camera_clearance(center, yaw_deg, others):
    """Distance from the wrist camera (at this gripper yaw) to the nearest other cup centre."""
    t = math.radians(yaw_deg)
    off = CAM_OFFSET_TCP[0] * np.array([math.cos(t), math.sin(t)]) + \
        CAM_OFFSET_TCP[1] * np.array([math.sin(t), -math.cos(t)])
    cam = np.array(center[:2]) + off
    return min([float(np.hypot(*(cam - np.array(o[:2])))) for o in others] + [1.0])


def low_grasp_yaw(center, others, closing_deg):
    """Gripper yaw closing along closing_deg (either half turn: the camera sits on one side),
    rated by hand-body and wrist-camera clearance. Returns (yaw, hand_clear, cam_clear)."""
    best = None
    for yaw in (closing_deg - 90, closing_deg + 90):
        yaw = (yaw + 180) % 360 - 180
        if not -120 <= yaw <= 170:                    # wrist joint range
            continue
        hc = housing_clearance(center, math.radians(yaw + 90), list(others))
        cc = camera_clearance(center, yaw, others)
        score = min(hc / HAND_CLEARANCE, cc / CAM_CLEARANCE)
        if best is None or score > best[3]:
            best = (yaw, hc, cc, score)
    return best[:3]


def choose_yaw(center, others, options_deg):
    """Gripper yaw (q = [cos(t/2), sin(t/2), 0, 0]) whose hand body (axis t + 90 deg)
    keeps the most clearance to the other cups; ties keep the first option."""
    return max(options_deg, key=lambda t: round(housing_clearance(center, math.radians(t + 90), others), 3))


def q_yaw(deg):
    t = math.radians(deg)
    return [math.cos(t / 2), math.sin(t / 2), 0.0, 0.0]


def disc_centre(points, cam_xy):
    """Centre and top z of a cup's top disc from its depth points. Depth near the camera's
    range limit can miss the far half of the disc, so the centre is the edge nearest the
    camera minus the radius, the radius taken from the full width across the view."""
    top = float(np.percentile(points[:, 2], 90))
    q = points[points[:, 2] > top - 0.008][:, :2]
    med = np.median(q, 0)
    v = np.array(cam_xy[:2]) - med
    v = v / max(np.linalg.norm(v), 1e-6)          # towards the camera
    w = np.array([-v[1], v[0]])                    # across the view
    a, b = q @ v, q @ w
    b0, b1 = np.percentile(b, [2, 98])
    r = float(np.clip((b1 - b0) / 2, 0.022, 0.035))
    c = (np.percentile(a, 98) - r) * v + (b0 + b1) / 2 * w
    return [float(c[0]), float(c[1]), top], r


def cup_top_near(frame, xy, max_dist=0.06):
    """Top-disc centre and top z of the cup nearest xy: green points above 0.165 m."""
    pc, valid, us, vs = backproject(frame, step=1)
    hsv = cv.cvtColor(frame['image'], cv.COLOR_BGR2HSV)
    green = cv.inRange(hsv, tracker.HSV_LO, tracker.HSV_HI) > 0
    top = (valid & green & (pc[..., 2] > 0.165)).astype(np.uint8)
    n, lab, stats, _ = cv.connectedComponentsWithStats(top)
    best, best_d = None, max_dist
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] < 300:
            continue
        p = pc[lab == i]
        d = float(np.hypot(*(np.median(p[:, :2], 0) - np.array(xy[:2]))))
        if d < best_d:
            best, best_d = p, d
    if best is None:
        return None
    centre, r = disc_centre(best, frame['T'][:3, 3])
    log('disc radius', round(r, 4))
    return centre


def look_at_cup(ctx, target, side=0.0):
    """Close overhead view of one cup (camera ~0.4 m away, inside the depth range)."""
    safe_move(ctx, [target[0] - CLOSEUP_BACK, target[1] + side, CLOSEUP_Z])
    fr = observe(ctx)
    loc = cup_top_near(fr, target)
    log('cup close-up', loc, fr['id'], 'side', side)
    return loc, fr['id']


def remove_cup(ctx, cups, target, prefer='near', dest=None):
    """Lift the target cup and set it down at a free spot: the nearest one with DROP_CLEARANCE
    (prefer='near', uncovering the object) or the most open one (prefer='clear', making room)."""
    loc, fid = look_at_cup(ctx, target)
    if loc is None:
        return {'ok': False, 'why': 'target cup not found in the close-up view', 'frame': fid}
    others = [c for c in cups if np.hypot(c[0] - target[0], c[1] - target[1]) > 0.03]
    for attempt in range(2):
        x, y, top = loc
        if not (TABLE_Z + 0.05 < top < TABLE_Z + 0.15):
            return {'ok': False, 'why': f'cup top {top:.3f} implausible', 'frame': fid}
        yaw = choose_yaw(loc, others, [90, 45, -45, 0])
        q = q_yaw(yaw)
        log('cup yaw', yaw, 'clearance', round(housing_clearance(loc, math.radians(yaw + 90), others), 3))
        grasp_z = top - CUP_GRASP_BELOW_TOP
        move(ctx, [x, y, 0.42], q)
        move(ctx, [x, y, top + 0.03], q)
        try:
            move(ctx, [x, y, grasp_z], q, tol=4, accept_mm=8)
            break
        except RuntimeError as e:
            z = tcp(ctx)[2]
            log('cup descent stalled at z', round(z, 3), tool_error(e).get('reason', str(e))[:120])
            move(ctx, [x, y, top + 0.08], q)
            if attempt == 1:
                return {'ok': False, 'why': f'fingers blocked above the cup twice (z {z:.3f})', 'cup': loc}
            # look again from the side: the other edge of the disc faces the camera
            side = -SIDE_LOOK if y > 0 else SIDE_LOOK
            new, fid = look_at_cup(ctx, loc, side=side)
            if new is None:
                return {'ok': False, 'why': 'cup lost after a blocked descent', 'frame': fid}
            loc = [(loc[0] + new[0]) / 2, (loc[1] + new[1]) / 2, max(loc[2], new[2])]
            log('cup centre re-estimated', np.round(loc, 4).tolist())
    ctx.value('franka_gripper', action='grasp', width=0.04, force=CUP_FORCE, speed=0.05)
    g = wait_gripper(ctx)
    log('cup grasp', g)
    if not (0.035 < g['width'] < 0.078):
        ctx.value('franka_gripper', action='open')
        move(ctx, [x, y, top + 0.12], q)
        return {'ok': False, 'why': f'cup grasp width {g["width"]:.3f}', 'grasp': g}
    move(ctx, [x, y, top + 0.15], q)
    # drop spot: not closer to the base than x=0.42 (x=0.36 at low z hit the J4 limit),
    # clear of every other cup and of the cube spot
    cands = [(cx, cy) for cx in (0.42, 0.47, 0.62, 0.68) for cy in np.arange(-0.30, 0.31, 0.05)]
    # x >= 0.42 (x 0.36 hit the J4 limit at low z); far-side spots with |y| >= 0.23 were refused
    # by the IK (joint step > 0.2 rad)
    cands = [(float(cx), float(cy)) for cx, cy in cands if not (cx > 0.6 and abs(cy) > 0.15)]
    score = lambda p: min([np.hypot(p[0] - c[0], p[1] - c[1]) for c in others] +
                          [np.hypot(p[0] - x, p[1] - y)])
    release_z = grasp_z + DROP_ABOVE_GRASP
    if prefer != 'near':                   # making room: open spots, but not across the table
        cands = [p for p in cands if np.hypot(p[0] - x, p[1] - y) < 0.30]
    if dest is not None:                   # a given destination (arranging cups)
        cands = [tuple(dest[:2])]
        score = lambda p: DROP_CLEARANCE
    order = (lambda p: (score(p) < DROP_CLEARANCE, np.hypot(p[0] - x, p[1] - y))) if prefer == 'near' \
        else (lambda p: -score(p))
    for dx, dy in sorted(cands, key=order)[:3]:
        if score((dx, dy)) < DROP_CLEARANCE:
            break
        log('drop cup at', dx, dy, 'clearance', round(score((dx, dy)), 3))
        try:
            move_far(ctx, [dx, dy, top + 0.15], q)
            move(ctx, [dx, dy, release_z], q, tol=6, accept_mm=10)
        except RuntimeError as e:
            z = tcp(ctx)[2]
            if z - release_z > 0.04:   # stopped high: back up and try the next spot
                log('drop spot refused at z', round(z, 3), tool_error(e).get('reason', str(e))[:160])
                cur = tcp(ctx)
                move(ctx, [cur[0], cur[1], top + 0.15], q)
                continue
            log('drop descent stopped at z', round(z, 3), '- releasing there')
        ctx.value('franka_gripper', action='open')
        wait_gripper(ctx)
        cur = tcp(ctx)
        move(ctx, [cur[0], cur[1], top + 0.15], q)
        return {'ok': True, 'cup': loc, 'drop': [cur[0], cur[1]], 'others': others + [[cur[0], cur[1], top]]}
    return {'ok': False, 'why': 'no reachable drop spot; still holding the cup above the table', 'cup': loc}


def pixel_to_plane(frame, u, v, z):
    """Base-frame xy where the camera ray through pixel (u, v) meets the plane at height z."""
    K, T = frame['K'], frame['T']
    ray = T[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    o = T[:3, 3]
    s = (z - o[2]) / ray[2]
    return (o + s * ray)[:2]


def rect_from_colour(frame, cube, model):
    """Top-face rectangle [w_mm, h_mm, angle_deg (base)] of a colourful object from the RGB
    image: object-coloured pixels near it, brighter half (the lit top face), projected onto the
    plane of its top. None for white/grey objects or when segmentation fails."""
    if model is None or model.get('lab') is None:
        return None
    L, a, b = model['lab']
    if math.hypot(a - 128, b - 128) < 20:           # not colourful enough to separate from the table
        return None
    u0, v0 = int(cube['px'][0]), int(cube['px'][1])
    x0, y0 = max(0, u0 - 120), max(0, v0 - 120)
    win = frame['image'][y0:v0 + 120, x0:u0 + 120]
    lab = cv.cvtColor(win, cv.COLOR_BGR2LAB).astype(float)
    d = lab - np.array(model['lab'])
    like = (np.sqrt((0.3 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2) < 35).astype(np.uint8)
    n, labels, stats, _ = cv.connectedComponentsWithStats(like)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    if stats[k, cv.CC_STAT_AREA] < 150:
        return None
    comp = labels == k
    bright = comp & (lab[..., 0] >= np.median(lab[..., 0][comp]))
    # ridge cue: the whole blob's two halves across its long axis differ in brightness
    cy_, cx_ = np.nonzero(comp)
    (_, _), (cw, ch), cang = cv.minAreaRect(np.stack([cx_, cy_], 1).astype(np.float32))
    ta = math.radians(cang if cw >= ch else cang + 90)               # long axis in the image
    side = (cx_ - cx_.mean()) * -math.sin(ta) + (cy_ - cy_.mean()) * math.cos(ta)
    Lc = lab[..., 0][comp]
    ridge = bool(abs(Lc[side > 0].mean() - Lc[side < 0].mean()) > RIDGE_DL) if (side > 0).any() and (side < 0).any() else False
    if ridge:                                         # use the whole blob: both sloped faces
        bright = comp
    ys, xs = np.nonzero(bright)
    (cu, cv_), (wpx, hpx), ang = cv.minAreaRect(np.stack([xs + x0, ys + y0], 1).astype(np.float32))
    z = cube['top_z']
    t = math.radians(ang)
    c = pixel_to_plane(frame, cu, cv_, z)
    pw = pixel_to_plane(frame, cu + 0.5 * wpx * math.cos(t), cv_ + 0.5 * wpx * math.sin(t), z)
    ph = pixel_to_plane(frame, cu - 0.5 * hpx * math.sin(t), cv_ + 0.5 * hpx * math.cos(t), z)
    w_mm, h_mm = 2000 * np.linalg.norm(pw - c), 2000 * np.linalg.norm(ph - c)
    ang_base = math.degrees(math.atan2(pw[1] - c[1], pw[0] - c[0]))
    return [float(w_mm), float(h_mm), float(ang_base)], [float(c[0]), float(c[1])], ridge


def colourful(model):
    return model is not None and model.get('lab') is not None and \
        math.hypot(model['lab'][1] - 128, model['lab'][2] - 128) >= 20


def find_by_colour(frame, model, near_xy=None, radius=0.10):
    """Object from its colour (small colourful objects can fall through the depth test): the
    largest object-coloured blob near near_xy, centre projected onto its mid-height plane."""
    if not colourful(model):
        return None
    lab = cv.cvtColor(frame['image'], cv.COLOR_BGR2LAB).astype(float)
    d = lab - np.array(model['lab'])
    like = (np.sqrt((0.3 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2) < 35).astype(np.uint8)
    like[ROI[3]:] = 0
    n, labels, stats, cents = cv.connectedComponentsWithStats(like)
    height = float(np.clip(model.get('height', 0.03), 0.012, 0.09))
    best = None
    for i in range(1, n):
        if stats[i, cv.CC_STAT_AREA] < 150:
            continue
        xy = pixel_to_plane(frame, cents[i][0], cents[i][1], TABLE_Z + height / 2)
        if near_xy is not None and np.hypot(*(xy - np.array(near_xy[:2]))) > radius:
            continue
        if best is None or stats[i, cv.CC_STAT_AREA] > best[0]:
            best = (int(stats[i, cv.CC_STAT_AREA]), cents[i], xy)
    if best is None:
        return None
    area, c, xy = best
    return dict(px=[float(c[0]), float(c[1])], xy=[float(xy[0]), float(xy[1])], z=TABLE_Z + height / 2,
                top_z=TABLE_Z + height, table_z=TABLE_Z, yaw=0.0, size_mm=list(model['size_mm']),
                area=area, lab=list(model['lab']), by_colour=True)


def norm_deg(t):
    """Angle in (-90, 90]: a parallel gripper is symmetric under a half turn."""
    t = (t + 90) % 180 - 90
    return 90.0 if t == -90 else t


def grasp_yaws(cube, others, xy):
    """Gripper yaws to try for a low grasp, best first. Square footprint: close on a face pair.
    Ridged (wedge) top: close along the ridge onto the parallel end faces (squeezing the
    sloped faces pushed a wedge away). Other elongated footprints: across the narrow extent.
    Each closing direction is tried at the half turn with the most hand and camera clearance."""
    w, h, ang = cube.get('rect', [cube['size_mm'][0], cube['size_mm'][1], math.degrees(cube['yaw'])])
    if abs(w - h) < 0.15 * max(w, h) and not cube.get('ridge'):
        first = ((ang + 45) % 90) - 45                # a face normal
    elif cube.get('ridge'):
        first = ang if w > h else ang + 90            # along the long extent (the ridge)
    else:
        first = ang if w < h else ang + 90            # across the narrow extent
    cands = []
    for closing in (first, first + 90, first + 30, first - 30):
        yaw, hc, cc = low_grasp_yaw(xy, others, closing)
        cands.append((min(hc / HAND_CLEARANCE, cc / CAM_CLEARANCE) >= 1.0, yaw, hc, cc))
    ok = [c for c in cands if c[0]]
    rest = sorted([c for c in cands if not c[0]], key=lambda c: -min(c[2] / HAND_CLEARANCE, c[3] / CAM_CLEARANCE))
    order = ok + rest
    log('grasp yaw candidates', [(round(c[1], 1), round(c[2], 3), round(c[3], 3)) for c in order])
    return [c[1] for c in order]


def blocking_cup(xy, yaw, others):
    """The cup that most limits a low grasp at this yaw (hand body or wrist camera), or None."""
    worst, worst_score = None, 1.0
    t = math.radians(yaw)
    d = np.array([math.cos(t + math.pi / 2), math.sin(t + math.pi / 2)])
    off = CAM_OFFSET_TCP[0] * np.array([math.cos(t), math.sin(t)]) + \
        CAM_OFFSET_TCP[1] * np.array([math.sin(t), -math.cos(t)])
    cam = np.array(xy[:2]) + off
    for o in others:
        hand = housing_clearance(xy, t + math.pi / 2, [o]) / HAND_CLEARANCE
        camc = float(np.hypot(*(np.array(o[:2]) - cam))) / CAM_CLEARANCE
        sc = min(hand, camc)
        if sc < worst_score:
            worst, worst_score = o, sc
    return worst


def grasp_cube(ctx, near_xy, others=(), model=None, yaws=None):
    """Find the object near near_xy from a close look, grasp it at mid-height and lift it.
    The grip is checked 2 cm up: if the fingers are closing in (the object slides or squashes
    out; a red foam-like cube did at 20 N) it is set down and re-grasped more gently, then on
    the other finger axis."""
    fr_ids, w = [], 0.0
    # (yaw choice, force): after a slip try the other finger axis first - a wedge squeezed on its
    # sloped faces slides out, on its parallel end faces it holds - then gentler grips
    plan = [(0, None), (1, None), (1, GENTLE_FORCE), (0, GENTLE_FORCE)]
    for attempt, (yaw_i, force) in enumerate(plan):
        move(ctx, [near_xy[0] - 0.02, near_xy[1], 0.36], Q_DOWN)
        fr = observe(ctx)
        fr_ids.append(fr['id'])
        cube = find_cube(fr, near_xy=near_xy, radius=0.08) or find_by_colour(fr, model, near_xy)
        log('cube close-up', cube)
        if cube is None:
            why = 'no cube under the lifted cup' if attempt == 0 else 'object lost after a failed grasp'
            return {'ok': False, 'why': why, 'frame': fr['id']}
        x, y = cube['xy']
        h = float(np.clip(cube['top_z'] - cube['table_z'], 0.012, 0.09))   # object height
        rc = rect_from_colour(fr, cube, model)
        if rc is not None:                 # colourful object: orientation from the lit face(s) and
            cube['rect'], cube['ridge'] = rc[0], rc[2]           # centre from the whole colour blob
            blob = find_by_colour(fr, model, [x, y])
            if blob is not None:
                x, y = blob['xy']
            log('rgb rect', np.round(rc[0], 1).tolist(), 'centre', [round(x, 3), round(y, 3)], 'ridge', rc[2])
        yaw = (yaws or grasp_yaws(cube, others, [x, y]))[yaw_i]
        cup = blocking_cup([x, y], yaw, [o for o in others if len(o) >= 3])
        if cup is not None:                # a cup would meet the hand or the wrist camera: move it
            log('making room: moving the cup at', np.round(cup[:2], 3).tolist())
            rest = [o for o in others if o is not cup]
            r = remove_cup(ctx, rest + [[x, y, cube['top_z']], cup], cup, prefer='clear')
            if not r['ok']:
                return {'ok': False, 'why': 'a cup blocks the grasp and could not be moved: ' + r['why']}
            others = rest + [r['others'][-1]]
            move(ctx, [near_xy[0] - 0.02, near_xy[1], 0.36], Q_DOWN)
            yaw = (yaws or grasp_yaws(cube, others, [x, y]))[yaw_i]
        if force is None:
            force = SMALL_FORCE if max(cube['size_mm']) < 45 else CUBE_FORCE
        q = q_yaw(yaw)
        gz = max(TABLE_Z + 0.008, TABLE_Z + h / 2)
        log('cube yaw', round(yaw, 1), 'height', round(h, 3), 'force', force)
        move(ctx, [x, y, 0.25], q)
        move(ctx, [x, y, TABLE_Z + h + 0.05], q)
        move(ctx, [x, y, gz], q, tol=6, accept_mm=10)
        ctx.value('franka_gripper', action='grasp', width=0.0, force=force, speed=0.05)   # close to contact
        g = wait_gripper(ctx)
        w0 = g['width']
        log('cube grasp', g)
        if w0 < 0.006:                     # closed on nothing
            ctx.value('franka_gripper', action='open')
            wait_gripper(ctx)
            move(ctx, [x, y, 0.25], q)
            near_xy = [x, y]
            continue
        move(ctx, [x, y, gz + 0.02], q)    # short test lift
        time.sleep(0.4)
        w1 = ctx.value('franka_status')['gripper']['width']
        if w1 < w0 - SLIP_MM / 1000.0:
            log('grip closing in during the test lift', round(w0, 4), '->', round(w1, 4), '- setting it down')
            try:
                move(ctx, [x, y, gz + 0.003], q, tol=6, accept_mm=10)
            except RuntimeError as e:            # pinched against the table: let go where it stopped
                log('set-down stopped short:', tool_error(e).get('reason', str(e))[:100])
            ctx.value('franka_gripper', action='open')
            wait_gripper(ctx)
            move(ctx, [x, y, 0.25], q)
            near_xy = [x, y]
            continue
        move(ctx, [x, y, 0.25], q)
        g = wait_gripper(ctx)
        w = g['width']
        if 0.006 < w < 0.078:          # something held (empty closes to ~0.0002), not fully open
            return {'ok': True, 'width': w, 'is_grasped': g['is_grasped'], 'yaw': yaw, 'force': force,
                    'attempts': attempt + 1, 'frame': fr_ids}
        log('object slipped out on the way up (width', round(w, 4), ')')
        ctx.value('franka_gripper', action='open')
        wait_gripper(ctx)
        near_xy = [x, y]
    return {'ok': False, 'width': w, 'why': f'could not hold the object ({len(plan)} attempts)', 'frame': fr_ids}


def setup_scene(ctx):
    """Ask the operator to lay out the cups and put the (possibly new) object in view."""
    reply = ctx.value('request_operator', kind='reset', timeout_s=900, message=(
        'Robot is at its watching position. Please set up the next trial: put the object to hide '
        'on the table in plain view (not under or touching a cup) and the green cups on the table '
        'below the gripper. Step back and reply "ready". I will then look at the object and ask '
        'you to hide it and shuffle.'))
    log('setup reply', reply)
    return {'ok': True, 'reply': reply}


def main(ctx, out_dir=None, resume=None, setup=False, object_name='white cube'):
    """Whole task; returns a report even when a step raises.
    resume: {'final_px', 'cover_id'} from an earlier identification (e.g. replay.py) to start
    at locate_cups on the scene the operator left."""
    report = {'steps': {}, 'ok': False, 'failed': None}
    try:
        run_steps(ctx, report, Path(out_dir) if out_dir else None, resume, setup, object_name)
    except Exception as e:  # leave the arm where it stopped and report
        report['error'] = repr(e)
        report['failed'] = report['failed'] or report.get('current', 'exception')
    return report


def run_steps(ctx, report, out_dir, resume=None, setup=False, object_name='white cube'):
    steps = report['steps']

    def step(name, r):
        steps[name] = r
        if not r.get('ok'):
            report['failed'] = name
        return r.get('ok')

    report['current'] = 'goto_watch'
    if not step('goto_watch', goto_watch(ctx)):
        return
    if resume is None:
        if setup:
            report['current'] = 'setup'
            step('setup', setup_scene(ctx))
        report['current'] = 'find_cube'
        r = step_find_cube(ctx)
        if not step('find_cube', r):
            return
        model = r['model']
        report['current'] = 'wait_shuffle'
        w = wait_shuffle(ctx, object_name)
        step('wait_shuffle', w)
        t0, t1, cube_px = w['t0'], w['t1'], r['cube']['px']
        if SAVE_RECORDING and out_dir is not None:
            n = save_recording(t0 - 1000, t1, out_dir / 'wrist')
            json.dump({'t0': t0, 't1': t1, 'cube_px': cube_px, 'cam_frame': r['frame'],
                       'model': r['model'], 'n_cups': r['n_cups']},
                      open(out_dir / 'wrist' / 'meta.json', 'w'))
            log('saved', n, 'shuffle frames')
        report['current'] = 'identify_cup'
        r = identify_cup(t0, t1, cube_px, r['frame'], debug_path=(out_dir / 'tracks.jpg') if out_dir else None,
                         model=r['model'], n_cups=r['n_cups'])
        if not step('identify_cup', {k: v for k, v in r.items() if k != 'hist'}):
            return
    else:
        r = dict(resume)
        model = resume.get('model')
    report['current'] = 'locate_cups'
    r2 = locate_cups_now(ctx, r['final_px'], r['cover_id'])
    if not step('locate_cups', r2):
        return
    report['current'] = 'remove_cup'
    r3 = remove_cup(ctx, r2['cups'], r2['target'])
    if not step('remove_cup', r3):
        return
    report['current'] = 'grasp_cube'
    r4 = grasp_cube(ctx, r3['cup'][:2], r3['others'], model)
    if not r4['ok'] and r4.get('why') == 'no cube under the lifted cup':
        # the cube may have come up inside the cup: lift that cup again from where it was
        # dropped, put it somewhere else and look where it stood
        log('cube not under the cup: rescuing it from the moved cup at', np.round(r3['drop'], 3).tolist())
        steps['grasp_cube_first'] = r4
        dropped = r3['others'][-1]
        report['current'] = 'rescue_cup'
        r5 = remove_cup(ctx, r3['others'][:-1] + [dropped], dropped)
        if not step('rescue_cup', r5):
            return
        report['current'] = 'grasp_cube'
        r4 = grasp_cube(ctx, r5['cup'][:2], r5['others'], model)
    if step('grasp_cube', r4):
        report['ok'] = True
    report.pop('current', None)


if __name__ == '__main__':
    import argparse
    from api.runtime import Context
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resume', help='JSON {"final_px": [[u, v], ...], "cover_id": k} to start at locate_cups')
    parser.add_argument('--setup', action='store_true', help='first ask the operator to lay out the scene')
    args = parser.parse_args()
    out = ROOT / 'output' / 'runs' / time.strftime('%Y%m%dT%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    with Context() as session:
        rep = main(session, out_dir=out, resume=json.loads(args.resume) if args.resume else None,
                   setup=args.setup)
    rep['duration_s'] = round(time.time() - t_start, 1)
    (out / 'report.json').write_text(json.dumps(rep, indent=1, default=str))
    (out / 'log.txt').write_text('\n'.join(LOG))
    print(json.dumps(rep, indent=1, default=str))
    print('run directory:', out)
