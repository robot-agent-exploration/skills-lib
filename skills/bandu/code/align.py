"""Visual peg-in-hole for the knob: adaptive tilt and offset correction with the exterior cameras.

Learned 2026-10-10 16:05-16:16 (user: "align yourself with at most 1 mm opening error"):
- The two-view wrist estimate of the urn opening scattered 6.5 mm and a knob grasped 3-5 mm off its
  ball centre hangs ~8 deg tilted; a 3 N descent then stops on the mouth even when the foot is
  centred (the tilted body cannot follow the throat).
- Measured instead, relative and in the same image: hold the foot FOOT_GAP above the rim (foot and
  lip separate in both views), find the foot's silhouette centre (saturation: wood vs grey) and the
  lip's outer outline centre (brightness: lip vs background, bounding box of the ellipse) in the left
  and right exterior images, and solve the base-xy offset from the two cameras' image axes. The
  knob's tilt comes from the silhouette centres along its height (3-23 mm above the foot).
- Rotate the hand about the TCP (the ball) by the opposite tilt, move the TCP so the foot lands on the
  lip centre, repeat until tilt <= MAX_TILT and offset <= MAX_OFFSET (16:15: 7.8 -> 2.2 -> 0.8 deg,
  offset 0.3 mm), then a guarded descent at 5 mm/s seats it (bottom -1.7 mm vs nominal).
"""
import math

import cv2
import numpy as np
from scipy.optimize import least_squares

import sys as _sys                                                   # skill folder: siblings import each other by name
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

from perception import Frame
from pieces import PIECES, check_held, carry_height
from motion import wrap_grasp_yaw

FOOT_GAP = 0.020                 # foot bottom above the rim while measuring
MAX_OFFSET = 0.0008              # m, aimed foot-vs-lip offset (user: at most 1 mm)
ACCEPT_OFFSET = 0.0010           # m, still accepted when the corrections run out
MAX_TILT = math.radians(2.5)     # tilt corrected above this (the fit jitters ~1 deg; the throat takes ~3)
ACCEPT_TILT = math.radians(3.0)
MAX_ITER = 6
FOOT_ROWS = (0.003, 0.004, 0.005, 0.006, 0.007)   # heights above the held piece's bottom for its centre
# per held piece: outer radius of the rim it goes into, heights for the axis fit (below the fingers)
PROFILES = {'knob': dict(lip_r=0.013, tilt_rows=np.arange(0.003, 0.0235, 0.002)),     # into the urn
            'urn': dict(lip_r=0.023, tilt_rows=np.arange(0.003, 0.0155, 0.002))}      # into the goblet's cup
DESCENT_SPEED, DESCENT_FORCE = 0.005, 3.0
SEATED_TOL = 0.006


def _crossing(row, uc, side, inner_is_max, k_in, k_out):
    """Sub-pixel position where the profile, scanned outward from uc, crosses the midpoint between
    its inner level (max or min within k_in) and outer level (opposite extreme within k_out)."""
    n = max(26, k_out[1] + 2)
    w = np.array([row[int(round(uc)) + side * k] for k in range(0, n)], float)
    inner = w[k_in[0]:k_in[1]].max() if inner_is_max else w[k_in[0]:k_in[1]].min()
    outer = w[k_out[0]:k_out[1]].min() if inner_is_max else w[k_out[0]:k_out[1]].max()
    thr = (inner + outer) / 2
    for k in range(1, n):
        if (w[k] <= thr) if inner_is_max else (w[k] >= thr):
            a, b = w[k - 1], w[k]
            return round(uc) + side * (k - 1 + ((thr - a) / (b - a) if b != a else 0.0))
    return None


def _channels(f):
    hsv = cv2.cvtColor(cv2.GaussianBlur(f.rgb, (3, 3), 0), cv2.COLOR_BGR2HSV).astype(float)
    return hsv[:, :, 1], hsv[:, :, 2]


def _lip(f, V, lip_xy, z_rim, lip_r):
    """Centre (u), width and expected width (px) of the lip's outer outline: extremes over the
    ellipse rows. Windows scale with the rim's radius in pixels (tuned on the urn: r 12 px)."""
    u0, v0 = f.project((lip_xy[0], lip_xy[1], z_rim))
    r_px = max(abs(f.project((lip_xy[0] + lip_r * math.cos(a), lip_xy[1] + lip_r * math.sin(a), z_rim))[0] - u0)
               for a in np.linspace(0, math.pi, 13))          # horizontal half-width of the rim ellipse
    sc = lip_r / 0.013                     # windows tuned on the urn's lip (r 13 mm) in both cameras
    k_in, k_out = (0, int(round(10 * sc))), (int(round(8 * sc)), int(round(22 * sc)))
    L, R = [], []
    for dv in range(-int(round(9 * sc)), int(round(6 * sc)) + 1):
        row = V[int(round(v0)) + dv]
        l = _crossing(row, u0, -1, False, k_in, k_out)
        r = _crossing(row, u0, +1, False, k_in, k_out)
        if l is not None and r is not None and r - l > 10 * sc:
            L.append(l)
            R.append(r)
    if not L:
        return None, 0.0, 2 * r_px
    return (min(L) + max(R)) / 2, max(R) - min(L), 2 * r_px


def _foot(f, S, tcp, z_foot, heights):
    """[(height above foot bottom, row, centre u, width)] of the knob silhouette."""
    out = []
    for dz in heights:
        u, v = f.project((tcp[0], tcp[1], z_foot + dz))
        row = S[int(round(v))]
        l, r = _crossing(row, u, -1, True, (0, 8), (6, 24)), _crossing(row, u, +1, True, (0, 8), (6, 24))
        if l is not None and r is not None and 6 < r - l < 30:
            out.append((dz, (l + r) / 2, r - l))
    return out


def measure(fid, tcp, lip_xy, z_rim, foot_drop, lip_r=0.013, tilt_rows=PROFILES['knob']['tilt_rows']):
    """{'offset': foot-bottom minus lip centre (m, base xy), 'tilt': (dx/dz, dy/dz) of the held
    piece's axis (top relative to bottom), 'ok', 'why'}. Both exterior cameras of frame fid."""
    z_foot = tcp[2] - foot_drop
    eqs, axis_data, info = [], {}, {}
    for cam in ('left', 'right'):
        f = Frame(fid, cam)
        S, V = _channels(f)
        lip_u, lip_w, lip_exp = _lip(f, V, lip_xy, z_rim, lip_r)
        foot = _foot(f, S, tcp, z_foot, FOOT_ROWS)
        axis = _foot(f, S, tcp, z_foot, tilt_rows)
        info[cam] = dict(lip_u=lip_u, lip_w=round(lip_w, 1), lip_expected=round(lip_exp, 1), foot_n=len(foot), axis_n=len(axis))
        if lip_u is None or not 0.75 * lip_exp < lip_w < 1.35 * lip_exp or len(foot) < 3 or len(axis) < 5:
            return dict(ok=False, why=f'{cam}: lip/foot not found ({info[cam]})', info=info)
        foot_u = float(np.median([c for _, c, _ in foot]))
        p = np.array([lip_xy[0], lip_xy[1], z_rim])
        jx = (f.project(p + [5e-4, 0, 0])[0] - f.project(p - [5e-4, 0, 0])[0]) / 1e-3
        jy = (f.project(p + [0, 5e-4, 0])[0] - f.project(p - [0, 5e-4, 0])[0]) / 1e-3
        h = f.project((lip_xy[0], lip_xy[1], z_foot + 0.005))[0] - f.project(p)[0]
        eqs.append(([jx, jy], foot_u - lip_u - h))
        axis_data[cam] = (f, axis)
        info[cam].update(foot_u=foot_u)
    d = np.linalg.solve(np.array([e[0] for e in eqs]), np.array([e[1] for e in eqs]))

    def resid(q):
        x0, y0, tx, ty = q
        return [f.project((tcp[0] + x0 + tx * dz, tcp[1] + y0 + ty * dz, z_foot + dz))[0] - uc
                for f, rows in axis_data.values() for dz, uc, _ in rows]
    fit = least_squares(resid, [0.0, 0.0, 0.0, 0.0])
    rms = float(np.sqrt(np.mean(np.square(fit.fun))))
    return dict(ok=rms < 0.8, why='' if rms < 0.8 else f'axis fit rms {rms:.2f} px', offset=(float(d[0]), float(d[1])),
                tilt=(float(fit.x[2]), float(fit.x[3])), rms=rms, info=info)


def _qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]


def untilt(quat, tilt):
    """Hand orientation that makes a knob axis with this tilt vertical (rotation about the TCP)."""
    a = np.array([tilt[0], tilt[1], 1.0])
    a /= np.linalg.norm(a)
    k = np.cross(a, [0.0, 0.0, 1.0])
    s = np.linalg.norm(k)
    if s < 1e-9:
        return list(quat)
    k /= s
    th = math.asin(min(1.0, s))
    q = _qmul([*(k * math.sin(th / 2)), math.cos(th / 2)], quat)
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def insert_visual(ctx, m, name, lip_xy, z_rim, seat_bottom_z, yaw=math.pi / 2, log=print):
    """Carry the held piece over the opening, align foot and tilt visually (iterative), descend,
    release and lift. Returns {'xy', 'seat_err', 'iterations'}; raises RuntimeError (still holding,
    lifted) if alignment or seating fails."""
    spec = PIECES[name]
    drop = spec['grasp']                       # ball centre (TCP) to foot bottom
    carry_z = carry_height(name)
    seat_tcp = seat_bottom_z + drop
    m.lift(z=carry_z)
    check_held(m, name)
    yaw = wrap_grasp_yaw(yaw)
    m.transit(lip_xy[0], lip_xy[1], yaw, z=carry_z)
    gap_z = z_rim + FOOT_GAP + drop
    r = m.descend(gap_z)
    if r.get('contact'):
        m.lift(z=carry_z)
        raise RuntimeError(f"{name}: touched something at TCP z {r['contact_position'][2]:.4f} above the alignment height (still holding)")
    target = [lip_xy[0], lip_xy[1]]
    quat = m.tcp()[1]
    for attempt in range(2):
        for it in range(MAX_ITER):
            m.move([target[0], target[1], gap_z], quat=quat, tolerance_mm=0.3)
            obs = ctx.value('franka_observe', cameras=['left', 'right'])
            meas = measure(obs['frame_id'], obs['tcp_position'], lip_xy, z_rim, drop, **PROFILES[name])
            if not meas['ok']:
                m.lift(z=carry_z)
                raise RuntimeError(f"{name}: visual alignment failed: {meas['why']} (frame {obs['frame_id']}, still holding)")
            dx, dy = meas['offset']
            tx, ty = meas['tilt']
            tilt = math.atan(math.hypot(tx, ty))
            log(f'{name}: align {attempt}.{it}: offset ({1000 * dx:+.2f},{1000 * dy:+.2f}) mm, tilt {math.degrees(tilt):.1f} deg '
                f'(rms {meas["rms"]:.2f} px)  frame {obs["frame_id"]}')
            if math.hypot(dx, dy) <= MAX_OFFSET and tilt <= MAX_TILT:
                break
            tcp = obs['tcp_position']
            # the foot hangs at TCP - drop*tilt; after untilting it is right under the TCP, so the
            # TCP goes to where the lip is: foot - offset
            fx, fy = (-drop * tx, -drop * ty) if tilt > MAX_TILT else (0.0, 0.0)
            if tilt > MAX_TILT:
                quat = untilt(obs['tcp_quaternion'], (tx, ty))
            target = [tcp[0] - dx + fx, tcp[1] - dy + fy]
            if math.dist(target, lip_xy) > 0.012:
                m.lift(z=carry_z)
                raise RuntimeError(f'{name}: alignment wants the TCP {1000 * math.dist(target, lip_xy):.0f} mm from the opening (still holding)')
        else:
            if not (math.hypot(dx, dy) <= ACCEPT_OFFSET and tilt <= ACCEPT_TILT):
                m.lift(z=carry_z)
                raise RuntimeError(f'{name}: not aligned after {MAX_ITER} corrections (still holding)')
            m.move([target[0], target[1], gap_z], quat=quat, tolerance_mm=0.3)    # last measured pose is good enough
        p, _ = m.tcp()
        r = ctx.value('franka_guarded_move', direction=[0, 0, -1],
                      max_distance_m=round(min(0.05, p[2] - seat_tcp + 0.004), 4),
                      speed_mps=DESCENT_SPEED, force_threshold_n=DESCENT_FORCE, retract_m=0.0)
        cz = (r.get('contact_position') or m.tcp()[0])[2]
        err = cz - seat_tcp
        log(f'{name}: descent stopped {1000 * err:+.1f} mm vs seat')
        if abs(err) <= SEATED_TOL:
            ctx.value('franka_gripper', action='move', width=min(0.08, spec['width'] + 0.02), speed=0.01)
            p, q = m.tcp()
            m.move([p[0], p[1], carry_z], quat=q)
            return {'xy': [p[0], p[1]], 'seat_err': err, 'iterations': it + 1}
        p, _ = m.tcp()
        m.move([p[0], p[1], gap_z], quat=quat)          # stuck high: back up and measure again
    m.lift(z=carry_z)
    raise RuntimeError(f'{name}: stuck {1000 * err:+.1f} mm above the seat after visual alignment (still holding)')


def upright(fid, xy, z_bottom, heights):
    """Tilt (deg) and rms (px) of a piece standing at xy (bottom at z_bottom) from the silhouette
    centres at `heights` above its bottom in both exterior cameras; None if not enough rows."""
    data = {}
    for cam in ('left', 'right'):
        f = Frame(fid, cam)
        S, _ = _channels(f)
        rows = _foot(f, S, (xy[0], xy[1], 0.0), z_bottom, heights)
        if len(rows) < 5:
            return None, None
        data[cam] = (f, rows)

    def resid(q):
        x0, y0, tx, ty = q
        return [f.project((xy[0] + x0 + tx * dz, xy[1] + y0 + ty * dz, z_bottom + dz))[0] - uc
                for f, rows in data.values() for dz, uc, _ in rows]
    fit = least_squares(resid, [0.0, 0.0, 0.0, 0.0])
    return math.degrees(math.atan(math.hypot(fit.x[2], fit.x[3]))), float(np.sqrt(np.mean(np.square(fit.fun))))


UPRIGHT_ROWS = {'urn': np.arange(0.016, 0.037, 0.002)}     # above the goblet's rim, below the lip
MAX_SEATED_TILT = 12.0     # deg: a seated urn reads 7-8 (shape/perspective bias; the goblet on the table 2), caught on the rim 22-24


def held_tilt(fid, tcp, drop, rows):
    """((dx/dz, dy/dz), rms px) of a held piece whose bottom is `drop` below the TCP, from its
    silhouette centres in both exterior cameras; None if it is not visible in enough rows."""
    z_bottom = tcp[2] - drop
    data = {}
    for cam in ('left', 'right'):
        f = Frame(fid, cam)
        S, _ = _channels(f)
        rows_found = _foot(f, S, tcp, z_bottom, rows)
        if len(rows_found) < 5:
            return None
        data[cam] = (f, rows_found)

    def resid(q):
        x0, y0, tx, ty = q
        return [f.project((tcp[0] + x0 + tx * dz, tcp[1] + y0 + ty * dz, z_bottom + dz))[0] - uc
                for f, rr in data.values() for dz, uc, _ in rr]
    fit = least_squares(resid, [0.0, 0.0, 0.0, 0.0])
    return (float(fit.x[2]), float(fit.x[3])), float(np.sqrt(np.mean(np.square(fit.fun))))


def straighten(ctx, m, name, log=print, max_steps=3):
    """Rotate the hand about the TCP until the held piece hangs vertical (user 16:47: set a tilted
    piece down straight, or it lands on one edge and can turn over on release). Returns the last
    measured tilt in degrees, or None if it could not be measured (then nothing is changed)."""
    drop, rows = PIECES[name]['grasp'], PROFILES[name]['tilt_rows']
    tilt = None
    for _ in range(max_steps):
        obs = ctx.value('franka_observe', cameras=['left', 'right'])
        r = held_tilt(obs['frame_id'], obs['tcp_position'], drop, rows)
        if r is None or r[1] > 0.8:
            log(f"{name}: held tilt not measurable ({'not visible' if r is None else f'rms {r[1]:.2f} px'}), frame {obs['frame_id']}")
            return tilt
        (tx, ty), rms = r
        tilt = math.degrees(math.atan(math.hypot(tx, ty)))
        log(f'{name}: hangs {tilt:.1f} deg off vertical (rms {rms:.2f} px)')
        if tilt <= math.degrees(MAX_TILT):
            return tilt
        m.move(obs['tcp_position'], quat=untilt(obs['tcp_quaternion'], (tx, ty)), tolerance_mm=2.0)
    return tilt
