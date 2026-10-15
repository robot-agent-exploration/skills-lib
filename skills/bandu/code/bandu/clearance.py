"""Whole-arm clearance: FR3 forward kinematics with coarse link, hand, finger and held-piece volumes,
checked against the pieces (vertical cylinders) and the table.

The robot service checks only the workspace box and the heights of the link ORIGINS
(lowest_link_z_m equals the TCP height for a sideways gripper even though link 7's body reaches
~5 cm lower). With a tilted gripper the hand and wrist are 5-25 cm beside the TCP: on 2026-10-10
15:06 the sideways hand swept the tower. Every tilted motion near pieces is checked here first.

Volumes are spheres sampled along capsules (links, from the franka_description collision capsules,
inflated) and inside boxes (hand housing, camera bracket, fingers: measured, see pieces.HAND_BOXES).
"""
import math

import numpy as np

from .motion import table_z_at

# FR3 (same as Panda) modified DH, Craig convention: (a_{i-1}, d_i, alpha_{i-1})
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2), (0.0825, 0.0, math.pi / 2),
      (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2), (0.088, 0.0, math.pi / 2)]
FLANGE_D, HAND_YAW, TCP_D = 0.107, -math.pi / 4, 0.1034
LIMITS = [(-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007), (-3.0421, -0.1518),
          (-2.8065, 2.8065), (0.5445, 4.5169), (-3.0159, 3.0159)]

from .params import LINK_PAD, TABLE_FROM_BASE  # hyper-parameters: params.py
# (frame, p0, p1, radius) in DH frame 1..7 (frame 8 = flange): link bodies as capsules
LINK_CAPSULES = [
    (1, (0, 0, -0.333), (0, 0, -0.05), 0.06),
    (2, (0, 0, -0.06), (0, 0, 0.06), 0.06),
    (3, (0, 0, -0.22), (0, 0, -0.07), 0.06),
    (4, (0, 0, -0.06), (0, 0, 0.06), 0.06),
    (5, (0, 0, -0.31), (0, 0, -0.21), 0.06),
    (5, (0, 0.08, -0.20), (0, 0.08, -0.06), 0.025),
    (6, (0, 0, -0.07), (0, 0, 0.01), 0.05),
    (7, (0, 0, -0.06), (0, 0, 0.08), 0.04),
]
# boxes in the TCP frame (z along the fingers, y = closing axis): housing and camera bracket
HAND_BOXES_3D = [('housing', (-0.035, 0.035), (-0.105, 0.105), (-TCP_D, -0.035)),
                 ('camera', (-0.095, -0.035), (-0.06, 0.0), (-0.12, -0.05))]
FINGER_HALF_X, FINGER_THICK, FINGERTIP_BEYOND_TCP = 0.011, 0.012, 0.0124


def _dh(a, d, alpha, theta):
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca], [0.0, 0.0, 0.0, 1.0]])


def _rz(t, z=0.0):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, z], [0, 0, 0, 1.0]])


def frames(q):
    """[T1..T7, flange, tcp] as 4x4 base-frame transforms."""
    T, out = np.eye(4), []
    for (a, d, alpha), th in zip(DH, q):
        T = T @ _dh(a, d, alpha, th)
        out.append(T)
    flange = T @ _rz(0.0, FLANGE_D)
    out.append(flange)
    out.append(flange @ _rz(HAND_YAW) @ _rz(0.0, TCP_D))
    return out


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    v = [0.0, 0.0, 0.0]
    v[i] = s / 4
    v[j] = (R[j, i] + R[i, j]) / s
    v[k] = (R[k, i] + R[i, k]) / s
    return v + [(R[k, j] - R[j, k]) / s]


def tcp_pose(q):
    T = frames(q)[-1]
    return T[:3, 3].copy(), R_to_quat(T[:3, :3])


def _capsule_spheres(p0, p1, r):
    n = max(1, math.ceil(np.linalg.norm(p1 - p0) / (0.5 * r)))
    return [(p0 + (p1 - p0) * k / n, r) for k in range(n + 1)]


def _box_spheres(T, xs, ys, zs, cell=0.012):
    """Spheres covering a box given in frame T (each grid cell by its circumscribed sphere)."""
    out = []
    axes = []
    for lo, hi in (xs, ys, zs):
        n = max(1, math.ceil((hi - lo) / cell))
        h = (hi - lo) / n
        axes.append(([lo + h * (k + 0.5) for k in range(n)], h))
    r = 0.5 * math.sqrt(sum(h * h for _, h in axes))
    for x in axes[0][0]:
        for y in axes[1][0]:
            for z in axes[2][0]:
                out.append(((T @ np.array([x, y, z, 1.0]))[:3], r))
    return out


def volumes(q, opening=0.08, held=None):
    """{'arm': spheres, 'hand': ..., 'fingers': ..., 'held': ...}; spheres are (centre, radius).
    held: (p0, p1, radius) of the carried piece's axis in the TCP frame (z along the fingers)."""
    F = frames(q)
    arm = []
    for i, a, b, r in LINK_CAPSULES:
        T = F[i - 1]
        arm += _capsule_spheres((T @ np.array([*a, 1.0]))[:3], (T @ np.array([*b, 1.0]))[:3], r + LINK_PAD)
    tcp = F[-1]
    hand = []
    for _, xs, ys, zs in HAND_BOXES_3D:
        hand += _box_spheres(tcp, xs, ys, zs)
    half = opening / 2
    fingers = []
    for ys in ((half, half + FINGER_THICK), (-half - FINGER_THICK, -half)):
        fingers += _box_spheres(tcp, (-FINGER_HALF_X, FINGER_HALF_X), ys, (-0.035, FINGERTIP_BEYOND_TCP), cell=0.008)
    out = {'arm': arm, 'hand': hand, 'fingers': fingers, 'held': []}
    if held is not None:
        a, b, r = held
        out['held'] = _capsule_spheres((tcp @ np.array([*a, 1.0]))[:3], (tcp @ np.array([*b, 1.0]))[:3], r)
    return out


def _cyl_dist(c, R, p):
    """Gap between sphere (c, R) and vertical cylinder p (dict xy, radius, bottom_z, top_z)."""
    dh = max(0.0, math.hypot(c[0] - p['xy'][0], c[1] - p['xy'][1]) - p['radius'])
    dz = max(p['bottom_z'] - c[2], 0.0, c[2] - p['top_z'])
    return math.hypot(dh, dz) - R


def clearance(q, pieces, opening=0.08, held=None, skip=()):
    """Smallest gaps: {(part, obstacle): gap_m} for every part (arm, hand, fingers, held) against
    every piece (and 'table'). skip: (part, obstacle) pairs not to check (e.g. ('held', 'table')
    while setting a piece down)."""
    vol = volumes(q, opening, held)
    gaps = {}
    for part, spheres in vol.items():
        for c, R in spheres:
            for p in pieces:
                if (part, p['name']) in skip or (part, '*') in skip:
                    continue
                g = _cyl_dist(c, R, p)
                k = (part, p['name'])
                if g < gaps.get(k, 1e9):
                    gaps[k] = g
            if (part, 'table') not in skip and math.hypot(c[0], c[1]) > TABLE_FROM_BASE:
                g = c[2] - R - table_z_at(c)
                if g < gaps.get((part, 'table'), 1e9):
                    gaps[(part, 'table')] = g
    return gaps


def worst(gaps, margins):
    """[(part, obstacle, gap)] that are closer than margins[part] (default margins['*'])."""
    bad = [(p, o, g) for (p, o), g in gaps.items() if g < margins.get(p, margins.get('*', 0.02))]
    return sorted(bad, key=lambda t: t[2])


def jacobian(q):
    F = frames(q)
    p = F[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z, o = F[i][:3, 2], F[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def _rot_err(R_target, R):
    E = R_target @ R.T
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def ik(q0, pos, quat, iters=200, damping=0.02, tol=1e-5):
    """Damped least-squares IK from q0 (stays near q0: mimics an incremental path)."""
    q, Rt, pos = np.array(q0, float), quat_to_R(quat), np.asarray(pos, float)
    for _ in range(iters):
        T = frames(q)[-1]
        e = np.concatenate([pos - T[:3, 3], _rot_err(Rt, T[:3, :3])])
        if np.linalg.norm(e) < tol:
            break
        J = jacobian(q)
        q = q + J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
    return q


def slerp(q0, q1, t):
    a, b = np.array(q0, float), np.array(q1, float)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        v = a + t * (b - a)
        return v / np.linalg.norm(v)
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def pose_path(q0, pos1, quat1, step=0.01):
    """Joint configurations along a straight TCP line with slerped orientation (move_pose)."""
    p0, quat0 = tcp_pose(q0)
    n = max(2, math.ceil(np.linalg.norm(np.asarray(pos1) - p0) / step))
    qs, q = [np.array(q0, float)], np.array(q0, float)
    for k in range(1, n + 1):
        t = k / n
        q = ik(q, p0 + t * (np.asarray(pos1) - p0), slerp(quat0, quat1, t))
        qs.append(q)
    return qs


def joint_path(q0, q1, step=0.02):
    """Configurations along the straight joint path (move_joints, min-jerk on the same line)."""
    q0, q1 = np.array(q0, float), np.array(q1, float)
    n = max(2, math.ceil(np.max(np.abs(q1 - q0)) / step))
    return [q0 + (q1 - q0) * k / n for k in range(n + 1)]


def check_path(qs, pieces, margins=None, opening=0.08, held=None, skip=()):
    """Worst clearances along a path: {'ok', 'min': {(part, obstacle): gap}, 'bad': [...], 'at': index}."""
    margins = margins or {'*': 0.02}
    mins, at = {}, {}
    for i, q in enumerate(qs):
        for k, g in clearance(q, pieces, opening, held, skip).items():
            if g < mins.get(k, 1e9):
                mins[k], at[k] = g, i
    bad = worst(mins, margins)
    return {'ok': not bad, 'bad': [(p, o, round(g, 4), at[(p, o)]) for p, o, g in bad],
            'min': {f'{p}/{o}': round(g, 4) for (p, o), g in sorted(mins.items(), key=lambda kv: kv[1])[:8]}}


def pieces_from(state, shapes):
    """Cylinders from {name: {'xy', 'bottom_z'[, 'radius', 'height']}} and {name: (r, h)}."""
    out = []
    for name, s in state.items():
        r, h = shapes.get(name, (0.02, 0.06))
        r, h = s.get('radius', r), s.get('height', h)
        out.append(dict(name=name, xy=tuple(s['xy']), radius=r, bottom_z=s['bottom_z'], top_z=s['bottom_z'] + h))
    return out
