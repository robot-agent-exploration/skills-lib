"""Dark (black) table top: dark depth points at table height -> top height and top rectangle; its sides;
the near edge below / beside a held object; clearance of the floor in front of a side.

Tested on the black-top drawer island: rectangle 1.0 x 0.62 m from 1.9 m (0.97 x 0.62 close up), top
0.67-0.74 by depth depending on the view (0.67 by touch). Seen from far, the fitted edge sits a few cm inside
the real edge (the drawer fronts appeared 8-15 cm outside it)."""
import cv2
import numpy as np

from ..core.frames import clusters, frame_dir, load

from ..params import TABLE_Z, DARK_MAX, TABLE_OBST_Z as OBST_Z  # hyper-parameters: boxpnp/params.py


def find_table(fdir, near=None, radius=1.5, max_range=3.5, min_side=0.3):
    pts, pix, rgb, meta = load(fdir, max_depth=max_range + 1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    dark = gray[pix[:, 1], pix[:, 0]] < DARK_MAX
    m = (pts[:, 2] > TABLE_Z[0]) & (pts[:, 2] < TABLE_Z[1]) & (pts[:, 0] > 0.25) & (pts[:, 0] < max_range) & dark
    if near is not None:
        m &= np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1]) < radius
    P = pts[m]
    if len(P) < 200:
        return {'ok': False, 'why': f'only {len(P)} dark points at table height'}
    z = np.median(P[:, 2])
    P = P[np.abs(P[:, 2] - z) < 0.04]
    n, lab = clusters(P, 0.02)
    sizes = np.bincount(lab, minlength=n)
    sizes[0] = 0
    Q = P[lab == int(np.argmax(sizes))]
    rect = cv2.minAreaRect(Q[:, :2].astype(np.float32))
    if max(rect[1]) < min_side:
        return {'ok': False, 'why': f'dark blob too small {rect[1]}'}
    return {'ok': True, 'top_z': float(np.median(Q[:, 2])),
            'z_spread': float(np.percentile(Q[:, 2], 90) - np.percentile(Q[:, 2], 10)),
            'center': [float(rect[0][0]), float(rect[0][1])], 'size': [float(rect[1][0]), float(rect[1][1])],
            'angle_deg': float(rect[2]), 'corners': cv2.boxPoints(rect).tolist(), 'n': int(len(Q)),
            'frame': frame_dir(fdir).name}


def sides(corners):
    """The 4 sides of a rectangle: midpoint, direction, outward normal, length."""
    C = np.array(corners, float)
    centre = C.mean(0)
    out = []
    for i in range(4):
        a, b = C[i], C[(i + 1) % 4]
        m = (a + b) / 2
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([d[1], -d[0]])
        if n @ (m - centre) < 0:
            n = -n
        out.append({'mid': m, 'dir': d, 'normal': n, 'length': float(np.linalg.norm(b - a))})
    return out


def side_clear(frame_ids, side, strip=(0.20, 1.0), self_box=(0.8, 0.65)):
    """Obstacle and floor points in the strip in front of a side (base frame of the frames' capture).
    Points of the robot itself / a held object (x < self_box[0], |y| < self_box[1]) are skipped."""
    n_obst, n_floor = 0, 0
    for fid in frame_ids:
        pts, _, _, _ = load(fid, stride=3, max_depth=4.0)
        rel = pts[:, :2] - side['mid']
        out = rel @ side['normal']
        along = rel @ side['dir']
        m = (out > strip[0]) & (out < strip[1]) & (np.abs(along) < side['length'] / 2 - 0.05)
        m &= ~((pts[:, 0] < self_box[0]) & (np.abs(pts[:, 1]) < self_box[1]))
        n_obst += int((m & (pts[:, 2] > OBST_Z[0]) & (pts[:, 2] < 0.6)).sum())
        n_floor += int((m & (pts[:, 2] < 0.04)).sum())
    return {'obstacle_points': n_obst, 'floor_points': n_floor}


def near_edge_x(fdir, x_range=(0.25, 1.2), y_max=0.6, z=(0.6, 0.8), min_points=100):
    """x of the table's near edge (2nd percentile of the dark top points), e.g. below a held box."""
    pts, pix, rgb, meta = load(fdir, stride=2, max_depth=2.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    dark = gray[pix[:, 1], pix[:, 0]] < DARK_MAX
    m = dark & (pts[:, 2] > z[0]) & (pts[:, 2] < z[1]) & (pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1]) & \
        (np.abs(pts[:, 1]) < y_max)
    if m.sum() < min_points:
        return None, int(m.sum())
    return float(np.percentile(pts[m, 0], 2)), int(m.sum())
