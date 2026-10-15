"""Coarse clearance check between the FR3 (capsules from franka_description) and the fryer.

The fryer is modelled in its own frame: origin on the vertical axis at table height,
`d` = horizontal unit vector from the axis toward the handle (drawer pull direction),
`t` = z x d. Distances are approximate (axis sampling), good to a few millimetres.
"""
import numpy as np

from .kin import fk_all

# (frame index in fk_all, segment endpoints in that frame, radius). fk_all: 0..6 = links 1..7,
# 7 = flange, 8 = TCP. Link 7 capsules use the link_with_sc rpy z=pi/4 rotation.
_c4, _s4 = np.cos(np.pi / 4), np.sin(np.pi / 4)
CAPSULES = [
    ('link3', 2, [(0, 0, -0.22), (0, 0, -0.07)], 0.06),
    ('link4', 3, [(0, 0, -0.06), (0, 0, 0.06)], 0.06),
    ('link5', 4, [(0, 0, -0.31), (0, 0, -0.21)], 0.06),
    ('link5b', 4, [(0, 0.08, -0.20), (0, 0.08, -0.06)], 0.025),
    ('link6', 5, [(0, 0, -0.07), (0, 0, 0.01)], 0.05),
    ('link7', 6, [(0, 0, -0.06), (0, 0, 0.08)], 0.04),
    ('link7b', 6, [(0.06 * _c4 - 0.005 * _c4, 0.06 * _s4 - 0.005 * _s4, 0.082),
                   (0.06 * _c4 + 0.005 * _c4, 0.06 * _s4 + 0.005 * _s4, 0.082)], 0.03),
    # wrist ZED camera: optical centre measured at TCP-frame (-0.0695, 0.032, -0.082), baseline along hand y
    ('wristcam', 8, [(-0.07, -0.03, -0.082), (-0.07, 0.095, -0.082)], 0.022),
]


def _seg_points(T, a, b, n=12):
    a = T[:3, :3] @ np.array(a) + T[:3, 3]
    b = T[:3, :3] @ np.array(b) + T[:3, 3]
    s = np.linspace(0, 1, n)[:, None]
    return a + s * (b - a)


# Hand housing box in the TCP frame: x +-0.032, y +-0.102, from the flange (z=-0.1034) to 0.066 below it.
HAND_BOX = ((-0.032, 0.032), (-0.102, 0.102), (-0.1034, -0.0374))


def hand_points(T_tcp, n=7):
    (x0, x1), (y0, y1), (z0, z1) = HAND_BOX
    g = np.array(np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, 3 * n), np.linspace(z0, z1, n))).reshape(3, -1).T
    return (T_tcp[:3, :3] @ g.T).T + T_tcp[:3, 3]


def finger_points(T_tcp, width):
    """Sample points of the two fingers (pads and diagonal links) in base frame."""
    pts = []
    for sign in (1, -1):
        for z in np.linspace(-0.045, 0.009, 7):       # along approach, relative to TCP
            for x in (-0.0088, 0.0, 0.0088):
                for dy in (0.0, 0.0076, 0.015):
                    p = np.array([x, sign * (width / 2 + dy + (0.008 if z < -0.02 else 0.0)), z])
                    pts.append(T_tcp[:3, :3] @ p + T_tcp[:3, 3])
    return np.array(pts)


class Fryer:
    def __init__(self, center_xy, d_xy, table_z=0.09, top_z=0.415, seam_z=0.245,
                 r_upper=0.115, r_lower=0.12, grip=None, arm=None, drawer_out=0.0):
        self.c = np.asarray(center_xy, float)
        d = np.asarray(d_xy, float)
        self.d = d / np.linalg.norm(d)
        self.t = np.array([-self.d[1], self.d[0]])
        self.table_z, self.top_z, self.seam_z = table_z, top_z, seam_z
        self.r_upper, self.r_lower = r_upper, r_lower
        # boxes in (d, t, z): (min, max)
        self.grip = grip or ((0.150, 0.185), (-0.017, 0.017), (0.13, 0.215))
        self.arm = arm or ((0.110, 0.185), (-0.017, 0.017), (0.21, 0.25))
        self.out = drawer_out   # drawer pulled out by this much along d

    def local(self, p):
        p = np.atleast_2d(p)
        rel = p[:, :2] - self.c
        return np.c_[rel @ self.d, rel @ self.t, p[:, 2]]

    def distance(self, p):
        """Signed-ish distance (m) of base points to the fryer solids (negative inside)."""
        L = self.local(p)
        dd, tt, zz = L[:, 0], L[:, 1], L[:, 2]
        out = []
        # upper body cylinder
        r = np.hypot(dd, tt)
        out.append(self._cyl(r, zz, self.r_upper, self.seam_z, self.top_z))
        # lower: fixed back half of body + drawer (shifted by self.out along d)
        r2 = np.hypot(dd - self.out, tt)
        out.append(self._cyl(np.minimum(r, r2), zz, self.r_lower, self.table_z, self.seam_z))
        for box in (self.grip, self.arm):
            (d0, d1), (t0, t1), (z0, z1) = box
            q = np.c_[dd - self.out, tt, zz]
            lo = np.array([d0, t0, z0]); hi = np.array([d1, t1, z1])
            outside = np.maximum(np.maximum(lo - q, q - hi), 0)
            dist = np.linalg.norm(outside, axis=1)
            inside = np.all((q > lo) & (q < hi), axis=1)
            dist[inside] = -np.min(np.minimum(q - lo, hi - q), axis=1)[inside]
            out.append(dist)
        return np.min(out, axis=0)

    @staticmethod
    def _cyl(r, z, R, z0, z1):
        dr = r - R
        dz = np.maximum(z0 - z, z - z1)
        outside = np.hypot(np.maximum(dr, 0), np.maximum(dz, 0))
        inside = np.maximum(dr, dz)
        return np.where((dr < 0) & (dz < 0), inside, outside)


# Static obstacles besides fryer/table (e.g. exterior cameras): list of (name, a, b, radius)
OBSTACLES = []


def set_camera_obstacles(F, half_width=0.095, radius=0.045, mount_radius=0.04):
    """Keep-out capsules for the exterior cameras: the left one from the (corrected) pose in the
    loaded frame F; the right one moved after a collision, so a large sphere around its old pose."""
    OBSTACLES.clear()
    if 'left' in F:
        T = np.asarray(F['left']['T'])
        c, x = T[:3, 3], T[:3, 0]
        OBSTACLES.append(('left_cam', c - half_width * x, c + half_width * x, radius))
        OBSTACLES.append(('left_mount', c, np.array([c[0], c[1], 0.09]), mount_radius))
    info = F['meta']['cameras'].get('right')
    if info:
        c = np.array(info['T_base_cam'])[:3, 3]
        OBSTACLES.append(('right_cam_zone', c, c, 0.20))
        OBSTACLES.append(('right_mount', c, np.array([c[0], c[1], 0.09]), 0.12))


def _seg_dist(P, a, b):
    """Distance from points P (N,3) to segment ab."""
    ab = b - a
    s = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(P - (a + s[:, None] * ab), axis=1)


def obstacle_distance(P):
    if not OBSTACLES:
        return np.full(len(P), np.inf)
    return np.min([_seg_dist(P, a, b) - r for _, a, b, r in OBSTACLES], axis=0)


def clearance(q, fryer, width=0.08, table_z=0.09, skip_fingers=False):
    """Minimum clearance (m) of arm capsules and fingers to the fryer, table and OBSTACLES."""
    Ts = fk_all(q)
    worst = (np.inf, None)
    for name, idx, (a, b), rad in CAPSULES:
        pts = _seg_points(Ts[idx], a, b)
        dist = min(fryer.distance(pts).min(), (pts[:, 2] - table_z).min(), obstacle_distance(pts).min()) - rad
        if dist < worst[0]:
            worst = (dist, name)
    pts = hand_points(Ts[-1])
    dist = min(fryer.distance(pts).min(), (pts[:, 2] - table_z).min(), obstacle_distance(pts).min())
    if dist < worst[0]:
        worst = (dist, 'hand')
    if not skip_fingers:
        pts = finger_points(Ts[-1], width)
        dist = min(fryer.distance(pts).min(), (pts[:, 2] - table_z).min())
        if dist < worst[0]:
            worst = (dist, 'fingers')
    return worst


def held_banana_points(T_tcp, half_len=0.09, radius=0.02):
    """Surface-ish sample of a held banana (axis along hand x through the TCP)."""
    pts = []
    for s in np.linspace(-half_len, half_len, 13):
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            p = np.array([s, radius * np.cos(a), radius * np.sin(a)])
            pts.append(T_tcp[:3, :3] @ p + T_tcp[:3, 3])
    return np.array(pts)


def clearance_holding(q, fryer, table_z=0.09, width=0.04):
    """clearance() plus the held banana."""
    worst = clearance(q, fryer, width=width, table_z=table_z, skip_fingers=True)
    pts = held_banana_points(fk_all(q)[-1])
    dist = min(fryer.distance(pts).min(), (pts[:, 2] - table_z).min(), obstacle_distance(pts).min())
    return (dist, 'banana') if dist < worst[0] else worst


def held_ball_points(T_tcp, radius=0.025):
    """Surface sample of a held ball centred at the TCP."""
    pts = []
    for a in np.linspace(0, np.pi, 6):
        for b in np.linspace(0, 2 * np.pi, 10, endpoint=False):
            p = radius * np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
            pts.append(T_tcp[:3, 3] + p)
    return np.array(pts)


def clearance_holding_ball(q, fryer, table_z=0.09, radius=0.025):
    worst = clearance(q, fryer, table_z=table_z, skip_fingers=True)
    pts = held_ball_points(fk_all(q)[-1], radius)
    dist = min(fryer.distance(pts).min(), (pts[:, 2] - table_z).min(), obstacle_distance(pts).min())
    return (dist, 'ball') if dist < worst[0] else worst
