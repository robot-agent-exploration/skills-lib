"""Force-guided peg insertion: probe down, spiral search on the surface, descend to depth."""
import time

import numpy as np

from motion import move, status, wrench

# Values for the L1 (3 mm clearance) holes; first tested 2026-10-06 on the hexagon.
APPROACH_ABOVE = 0.008     # start probing this far above the expected surface contact (m)
PROBE_STEP = 0.001         # descent step while probing (m)
CONTACT_DFZ = 3.0          # |delta fz| (N) that counts as contact
FREE_BELOW = 0.006         # no contact this far below the expected surface -> peg is in the hole
PRESS = 0.0035             # command this far below the surface contact while searching (~10 N)
DROP = 0.002               # measured z this far below the surface contact = peg entered the hole
SEARCH_TOL_MM = 4.0        # > PRESS so pressed search moves return instead of winding up force
SPIRAL_PITCH = 0.002       # radial spacing of the spiral (m)
SPIRAL_STEP = 0.002        # arc length per move (m)
SPIRAL_RMAX = 0.010        # search radius (m)
DESCEND_STEP = 0.002       # descent step inside the hole (m)
JAM_DFZ = 8.0              # |delta fz| (N) that counts as blocked while descending in the hole
BOTTOM_SLACK = 0.004       # blocked within this of the expected bottom = bottomed out (ok)


def spiral_points(pitch=SPIRAL_PITCH, step=SPIRAL_STEP, rmax=SPIRAL_RMAX):
    """Archimedean spiral r = pitch * theta / 2pi, points ~step apart, starting at the centre."""
    pts, th = [(0.0, 0.0)], 0.0
    while True:
        r = pitch * th / (2 * np.pi)
        th += step / max(r, step)
        r = pitch * th / (2 * np.pi)
        if r > rmax:
            break
        pts.append((r * np.cos(th), r * np.sin(th)))
    return pts


def settled_z(ctx, dt=0.25):
    time.sleep(dt)
    return status(ctx)['tcp_position']


def probe_down(ctx, xy, quat, z_from, z_free, base, log):
    """Step down at xy. Returns ('contact', z) or ('free', z) once below z_free without contact."""
    z = z_from
    while True:
        z -= PROBE_STEP
        r = move(ctx, [xy[0], xy[1], z], quat, tol_mm=1.2, timeout_s=3)
        mz = r['measured_position'][2]
        dfz = wrench(ctx, n=2)[2] - base[2]
        log(f'    probe cmd {z:.4f} meas {mz:.4f} dfz {dfz:+.2f}{" STALL" if r.get("stalled") else ""}')
        if dfz < -CONTACT_DFZ or (r.get('stalled') and mz > z + 0.0008):
            return 'contact', mz
        if mz < z_free:
            return 'free', mz


def spiral_search(ctx, hole_xy, quat, z_contact, log):
    """Slide the pressed peg along a spiral around hole_xy until it drops. Returns xy or None."""
    hx, hy = hole_xy
    zc = z_contact - PRESS
    for i, (dx, dy) in enumerate(spiral_points()):
        r = move(ctx, [hx + dx, hy + dy, zc], quat, tol_mm=SEARCH_TOL_MM, timeout_s=3)
        m = r['measured_position']
        if m[2] > z_contact - DROP:
            m = settled_z(ctx)
        if i % 5 == 0:
            log(f'    spiral {i} d=({dx*1000:+.1f},{dy*1000:+.1f})mm meas z {m[2]:.4f}')
        if m[2] < z_contact - DROP:
            log(f'  dropped at spiral {i}: cmd offset ({dx*1000:+.1f},{dy*1000:+.1f}) mm, '
                f'meas ({(m[0]-hx)*1000:+.1f},{(m[1]-hy)*1000:+.1f}) mm, z {m[2]:.4f}')
            return (m[0], m[1]), m[2]
    return None, None


HOP_UP = 0.0015            # hover this far above the surface contact between search points (m)
HOP_STEP = 0.002           # candidate spacing (m); diagonal gap 1.41 mm < 1.5 mm per-side clearance
HOP_A_MAX = 0.010          # search half-range along the finger axis (gripper x) (m)
HOP_B_MAX = 0.004          # search half-range along the closing axis (gripper y) (m)


def hop_candidates(a_max=None, b_max=None, step=None):
    """Gripper-frame offsets (a along fingers, b along closing), nearest first; b weighted x2
    because the closing axis is centred by the grasp while the finger axis is not."""
    a_max = HOP_A_MAX if a_max is None else a_max
    b_max = HOP_B_MAX if b_max is None else b_max
    step = HOP_STEP if step is None else step
    na, nb = int(round(a_max / step)), int(round(b_max / step))
    pts = [(i * step, j * step) for i in range(-na, na + 1) for j in range(-nb, nb + 1)]
    return sorted(pts, key=lambda p: (p[0] ** 2 + (2 * p[1]) ** 2, abs(p[1]), -p[0]))


def hop_search(ctx, hole_xy, quat, yaw, z_contact, log):
    """Lift-move-press over candidate points (no sliding, so the board is not dragged).
    Returns ((x, y), z) of the drop or (None, None)."""
    hx, hy = hole_xy
    ax = np.array([np.cos(yaw), np.sin(yaw)])
    ay = np.array([np.sin(yaw), -np.cos(yaw)])
    z_hover, z_press = z_contact + HOP_UP, z_contact - PRESS
    for i, (a, b) in enumerate(hop_candidates()):
        x, y = np.array([hx, hy]) + a * ax + b * ay
        move(ctx, [x, y, z_hover], quat, tol_mm=1.5, timeout_s=3)
        r = move(ctx, [x, y, z_press], quat, tol_mm=SEARCH_TOL_MM, timeout_s=2)
        m = r['measured_position']
        if m[2] > z_contact - DROP:
            m = settled_z(ctx)
        log(f'    hop {i} (a,b)=({a*1000:+.1f},{b*1000:+.1f})mm meas z {m[2]:.4f}')
        if m[2] < z_contact - DROP:
            log(f'  dropped at hop {i}: base offset ({(m[0]-hx)*1000:+.1f},{(m[1]-hy)*1000:+.1f}) mm')
            return (m[0], m[1]), m[2]
    return None, None


SLIDE_PRESS = 0.0008       # command this far below the surface contact while sliding (~2 N)
SLIDE_DROP = 0.0005        # measured z this far below contact = peg entering the hole
SLIDE_STEP = 0.0005        # arc length per slide step (m)
SLIDE_PITCH = 0.0006       # spiral radial spacing (m)
SLIDE_RMAX = 0.0022        # slide search radius (m)
SLIDE_BIAS = (0.0, 0.0)    # optional base-frame offset of the spiral centre (camera hint)


def slide_search(ctx, hole_xy, quat, z_contact, log, rmax=None, step=None, pitch=None):
    """Slide the lightly pressed peg along a fine spiral until it starts to drop.
    Returns ((x, y), z) at the drop or (None, None)."""
    rmax = SLIDE_RMAX if rmax is None else rmax
    step = SLIDE_STEP if step is None else step
    pitch = SLIDE_PITCH if pitch is None else pitch
    cx, cy = hole_xy[0] + SLIDE_BIAS[0], hole_xy[1] + SLIDE_BIAS[1]
    zc = z_contact - SLIDE_PRESS
    for i, (dx, dy) in enumerate(spiral_points(pitch, step, rmax)):
        # tol < press depth: on the surface the call never 'reaches', so it runs its full
        # timeout (xy converges, light press); over the hole z reaches the target -> drop
        r = move(ctx, [cx + dx, cy + dy, zc], quat, tol_mm=0.5, timeout_s=1.2)
        m = r['measured_position'] if r.get('stalled') else settled_z(ctx, 0.1)
        if i % 10 == 0:
            log(f'    slide {i} d=({dx*1000:+.2f},{dy*1000:+.2f})mm z {m[2]:.4f} (contact {z_contact:.4f})')
        if m[2] < z_contact - SLIDE_DROP:
            log(f'  drop at slide {i}: offset ({(m[0]-hole_xy[0])*1000:+.2f},{(m[1]-hole_xy[1])*1000:+.2f}) mm, z {m[2]:.4f}')
            return (m[0], m[1]), m[2]
    return None, None


FINE_STEP = 0.0006         # fine hop grid spacing (m), < 2 x 0.5 mm per-side clearance
FINE_RMAX = 0.0020         # fine hop search radius (m)
FINE_UP = 0.0006           # hover above contact between presses (m)
FINE_PRESS = 0.0015        # press target below contact (m); tol 0.5 mm so it really presses
FINE_DROP = 0.0008         # measured z this far below contact = peg entered the hole


FINE_LIST = None           # optional explicit candidate offsets [(dx, dy), ...] (base frame, m)
FINE_PLACE = False         # correct the hover xy with measured positions before each press


def fine_points(step=None, rmax=None):
    if FINE_LIST is not None:
        return list(FINE_LIST)
    step = FINE_STEP if step is None else step
    rmax = FINE_RMAX if rmax is None else rmax
    n = int(rmax / step) + 1
    pts = [(i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)
           if (i * step) ** 2 + (j * step) ** 2 <= rmax ** 2 + 1e-12]
    return sorted(pts, key=lambda p: (p[0] ** 2 + p[1] ** 2, p[1], p[0]))


def fine_hop_search(ctx, hole_xy, quat, z_contact, log):
    """Lift 1 mm, move in free space (tol 0.5 mm), press to 2 mm below contact (tol 0.5 mm, the
    call stalls on the surface and reaches over the hole). Returns ((x, y), z) or (None, None)."""
    hx, hy = hole_xy
    for i, (dx, dy) in enumerate(fine_points()):
        x, y = hx + dx, hy + dy
        if FINE_PLACE:
            (x, y), _ = place_xy(ctx, (x, y), z_contact + FINE_UP, quat)
        else:
            move(ctx, [x, y, z_contact + FINE_UP], quat, tol_mm=0.5, timeout_s=1.5)
        r = move(ctx, [x, y, z_contact - FINE_PRESS], quat, tol_mm=0.5, timeout_s=1.0)
        m = r['measured_position']
        log(f'    fine {i} d=({dx*1000:+.1f},{dy*1000:+.1f})mm meas ({(m[0]-hx)*1000:+.2f},{(m[1]-hy)*1000:+.2f}) '
            f'z {m[2]:.4f}{" STALL" if r.get("stalled") else " reached"}')
        if m[2] < z_contact - FINE_DROP:
            log(f'  dropped at fine hop {i}')
            return (m[0], m[1]), m[2]
    move(ctx, [hx, hy, z_contact + 0.003], quat, tol_mm=1.0, timeout_s=2)
    return None, None


COVER_R = 0.00045          # a press at p rules out hole positions within this radius (m)
PLACE_TOL = 0.00015        # hover placement accuracy target (m), via measured-position correction
ADAPT_RMAX = 0.0022        # adaptive search radius (m)
TESTED = []                # optional seed of already-pressed xy positions (base frame)


def place_xy(ctx, target_xy, z, quat, tries=3):
    """Free-space move whose measured xy ends within PLACE_TOL of target (re-commands the error)."""
    cmd = np.array(target_xy, float)
    for _ in range(tries):
        r = move(ctx, [cmd[0], cmd[1], z], quat, tol_mm=0.5, timeout_s=1.5)
        m = np.array(r['measured_position'][:2])
        err = np.asarray(target_xy) - m
        if np.linalg.norm(err) < PLACE_TOL:
            break
        cmd = cmd + err
    return cmd, m


def adaptive_hop_search(ctx, centre_xy, quat, z_contact, log, rmax=None):
    """Press at the untested position nearest the centre until the peg drops; positions are the
    measured ones, so tracking error cannot leave gaps. Returns ((x, y), z) or (None, None)."""
    rmax = ADAPT_RMAX if rmax is None else rmax
    c = np.asarray(centre_xy, float)
    g = np.arange(-rmax, rmax + 1e-9, 0.0001)
    G = np.array([(c[0] + a, c[1] + b) for a in g for b in g if a * a + b * b <= rmax * rmax])
    tested = [np.asarray(t, float) for t in TESTED]
    shifts = []
    for i in range(80):
        if tested:
            d = np.min(np.linalg.norm(G[:, None, :] - np.array(tested)[None], axis=2), axis=1)
            free = G[d > COVER_R]
        else:
            free = G
        if len(free) == 0:
            break
        target = free[np.argmin(np.linalg.norm(free - c, axis=1))]
        cmd = target - (np.mean(shifts[-5:], axis=0) if shifts else 0.0)  # pre-compensate press shift
        move(ctx, [cmd[0], cmd[1], z_contact + FINE_UP], quat, tol_mm=0.5, timeout_s=1.2)
        r = move(ctx, [cmd[0], cmd[1], z_contact - FINE_PRESS], quat, tol_mm=0.5, timeout_s=1.0)
        p = r['measured_position']
        shifts.append(np.array(p[:2]) - cmd)
        tested.append(np.array(p[:2]))
        log(f'    adapt {i} target ({(target[0]-c[0])*1000:+.2f},{(target[1]-c[1])*1000:+.2f}) pressed '
            f'({(p[0]-c[0])*1000:+.2f},{(p[1]-c[1])*1000:+.2f}) z {p[2]:.4f} free {len(free)}')
        if p[2] < z_contact - FINE_DROP:
            log(f'  dropped at adaptive hop {i}')
            return (p[0], p[1]), p[2]
    move(ctx, [c[0], c[1], z_contact + 0.003], quat, tol_mm=1.0, timeout_s=2)
    return None, None


RASTER_HALF = 0.0018       # half-size of the raster square (m)
RASTER_LINE = 0.0008       # spacing between raster lines (m) (< 2 x 0.5 mm clearance)
RASTER_SEG = 0.0010        # segment length per call (m): enough spring force to beat friction
RASTER_PRESS = 0.0010      # press target below contact (m) (~3 N)
RASTER_DROP = 0.0005       # measured z this far below contact = entered the hole


def raster_path(yaw, half=None, line=None, seg=None):
    """Boustrophedon path in the gripper frame (lines along the finger axis a), as base offsets."""
    half = RASTER_HALF if half is None else half
    line = RASTER_LINE if line is None else line
    seg = RASTER_SEG if seg is None else seg
    ax = np.array([np.cos(yaw), np.sin(yaw)])
    ay = np.array([np.sin(yaw), -np.cos(yaw)])
    bs = np.arange(-half, half + 1e-9, line)
    bs = sorted(bs, key=abs)  # centre line first, then alternate outwards
    pts = []
    for k, b in enumerate(bs):
        a_vals = np.linspace(-half, half, int(np.ceil(2 * half / seg)) + 1)
        if k % 2:
            a_vals = a_vals[::-1]
        pts += [a * ax + b * ay for a in a_vals]
    return pts


def raster_slide_search(ctx, centre_xy, quat, yaw, z_contact, log):
    """Slide the lightly pressed peg along raster lines, one move per 1 mm segment."""
    c = np.asarray(centre_xy, float)
    zc = z_contact - RASTER_PRESS
    for i, d in enumerate(raster_path(yaw)):
        r = move(ctx, [c[0] + d[0], c[1] + d[1], zc], quat, tol_mm=0.5, timeout_s=1.0)
        m = r['measured_position']
        log(f'    raster {i} cmd ({d[0]*1000:+.1f},{d[1]*1000:+.1f}) meas ({(m[0]-c[0])*1000:+.2f},'
            f'{(m[1]-c[1])*1000:+.2f}) z {m[2]:.4f}')
        if m[2] < z_contact - RASTER_DROP:
            log(f'  dropped at raster {i}')
            return (m[0], m[1]), m[2]
    move(ctx, [c[0], c[1], z_contact + 0.003], quat, tol_mm=1.0, timeout_s=2)
    return None, None


def insert(ctx, hole_xy, quat, z_surface, depth, yaw=None, log=print, search='hop'):
    """Insert a held peg into the hole near hole_xy.

    z_surface: expected TCP z when the peg bottom touches the board top around the hole.
    depth: hole depth (m). yaw: gripper yaw (rad) for the hop search axes (None: sliding spiral).
    Returns dict(ok, xy_found, z_final, inserted_m, offset_mm, ...)."""
    hx, hy = hole_xy
    move(ctx, [hx, hy, z_surface + 0.03], quat)
    move(ctx, [hx, hy, z_surface + APPROACH_ABOVE], quat, tol_mm=1.2)
    base = wrench(ctx)
    log(f'  insert: baseline wrench {np.round(base, 2).tolist()}')
    kind, z = probe_down(ctx, (hx, hy), quat, z_surface + APPROACH_ABOVE, z_surface - FREE_BELOW, base, log)
    found, z_contact = (hx, hy), None
    if kind == 'contact':
        z_contact = z
        log(f'  surface contact at z={z:.4f} (expected {z_surface:.4f}); search')
        if search == 'raster':
            found, z = raster_slide_search(ctx, hole_xy, quat, yaw, z_contact, log)
        elif search == 'adaptive':
            found, z = adaptive_hop_search(ctx, hole_xy, quat, z_contact, log)
        elif search == 'finehop':
            found, z = fine_hop_search(ctx, hole_xy, quat, z_contact, log)
        elif search == 'slide':
            # lift off the probe stall first so the slide starts from a light press
            move(ctx, [hx, hy, z_contact + 0.0005], quat, tol_mm=1.0, timeout_s=2)
            found, z = slide_search(ctx, hole_xy, quat, z_contact, log)
        elif yaw is None:
            found, z = spiral_search(ctx, hole_xy, quat, z_contact, log)
        else:
            found, z = hop_search(ctx, hole_xy, quat, yaw, z_contact, log)
        if found is None:
            move(ctx, [hx, hy, z_surface + 0.03], quat)
            return {'ok': False, 'reason': 'hole not found by spiral search', 'z_contact': z_contact}
        z_top = z_contact
    else:
        z_top = z_surface
        log(f'  no contact down to z={z:.4f}: peg entered the hole directly')
    # descend inside the hole
    base = wrench(ctx)
    z_goal = z_top - depth + 0.0015
    zc = z
    m = [found[0], found[1], z]
    while zc > z_goal:
        zc = max(zc - DESCEND_STEP, z_goal)
        r = move(ctx, [found[0], found[1], zc], quat, tol_mm=1.5, timeout_s=3)
        m = r['measured_position']
        w = wrench(ctx, n=2)
        dfz = w[2] - base[2]
        log(f'    descend cmd {zc:.4f} meas {m[2]:.4f} dfz {dfz:+.2f} fxy ({w[0]:+.1f},{w[1]:+.1f})')
        if dfz < -JAM_DFZ or (r.get('stalled') and m[2] > zc + 0.0015):
            if m[2] < z_goal + BOTTOM_SLACK:
                log('  blocked near the expected bottom: bottomed out')
                break
            return {'ok': False, 'reason': 'blocked while descending', 'z': m[2], 'z_top': z_top,
                    'xy_found': list(found), 'z_contact': z_contact, 'inserted_m': z_top - m[2]}
    return {'ok': True, 'xy_found': [float(found[0]), float(found[1])], 'z_final': float(m[2]),
            'z_top': float(z_top), 'z_contact': z_contact, 'inserted_m': float(z_top - m[2]),
            'offset_mm': [(found[0] - hx) * 1000, (found[1] - hy) * 1000]}
