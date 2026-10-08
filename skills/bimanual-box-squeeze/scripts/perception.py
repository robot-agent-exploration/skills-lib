"""Box localization from a saved head-camera frame (runtime/frames/<id>/), base frame.

Depth is good for the box centre and top height from >= ~1 m with the torso at `ready`; it was off
by 3-5 cm for the box END faces and picks up the robot's own knee when squatting. So: localize
before squatting, and find the end faces by touch (touch.py).
"""
import json
from pathlib import Path

import cv2
import numpy as np

FLOOR_Z_MIN = 0.04          # floor noise is about +-0.03 m
OBJECT_Z_MAX = 0.45
GRID = 0.01
BOX_MIN_SIDE, BOX_MAX_SIDE = 0.10, 0.60
SELF_X = 0.36               # ignore points closer than this (robot front / knee) within |y| < SELF_Y
SELF_Y = 0.32


def load_points(fdir, stride=2, max_depth=4.5):
    """(N,3) base-frame points, (N,2) pixels and meta for a saved frame directory."""
    fdir = Path(fdir)
    meta = json.loads((fdir / 'meta.json').read_text())
    cam = meta['cameras']['head_camera']
    K, T = np.array(cam['K']), np.array(cam['T_base_cam'])
    depth = cv2.imread(str(fdir / cam['depth']), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    d = depth[::stride, ::stride]
    ok = (d > 0.2) & (d < max_depth)
    u, v, d = u[ok], v[ok], d[ok]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d)], 0)
    return (T @ pc)[:3].T, np.stack([u, v], 1), meta


def project(meta, p):
    cam = meta['cameras']['head_camera']
    K, T = np.array(cam['K']), np.array(cam['T_base_cam'])
    pc = np.linalg.inv(T) @ np.array([*p, 1.0])
    uv = K @ pc[:3]
    return np.array([uv[0] / uv[2], uv[1] / uv[2]]), float(pc[2])


def find_box(fdir, near=None, radius=1.0, top_band=0.025):
    """Largest box-like object on the floor -> centre (top face), top_z, size [long, short], axes, ends.

    long_axis points to +y (robot left); short_axis points away from the robot (+x)."""
    pts, pix, meta = load_points(fdir)
    m = (pts[:, 2] > FLOOR_Z_MIN) & (pts[:, 2] < OBJECT_Z_MAX) & (pts[:, 0] > 0.2) & (pts[:, 0] < 4.0) & (np.abs(pts[:, 1]) < 2.0)
    m &= ~((pts[:, 0] < SELF_X) & (np.abs(pts[:, 1]) < SELF_Y))
    if near is not None:
        m &= np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1]) < radius
    P = pts[m]
    if len(P) < 50:
        return {'ok': False, 'why': f'only {len(P)} object points in the search window'}
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    ij = np.floor((P[:, :2] - [x0, y0]) / GRID).astype(int)
    occ = np.zeros((ij[:, 0].max() + 1, ij[:, 1].max() + 1), np.uint8)
    occ[ij[:, 0], ij[:, 1]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, _, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    best = None
    for k in range(1, n):
        sel = labels[ij[:, 0], ij[:, 1]] == k
        if sel.sum() < 50:
            continue
        Q = P[sel]
        top = np.percentile(Q[:, 2], 95)
        band = Q[np.abs(Q[:, 2] - top) < top_band]
        if len(band) < 30:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(band[:, :2].astype(np.float32))
        if not (BOX_MIN_SIDE <= min(w, h) and max(w, h) <= BOX_MAX_SIDE):
            continue
        if best is None or len(band) > best[0]:
            best = (len(band), band, (cx, cy, w, h, ang))
    if best is None:
        return {'ok': False, 'why': 'no box-sized object'}
    _, band, (cx, cy, w, h, ang) = best
    a = np.deg2rad(ang)
    ax_w, ax_h = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    la = ax_w if w >= h else ax_h
    la = -la if la[1] < 0 else la
    sa = np.array([la[1], -la[0]])
    sa = -sa if sa[0] < 0 else sa
    rel = band[:, :2] - [cx, cy]
    lo_l, hi_l = np.percentile(rel @ la, [1, 99])          # 3-97 % shrank the box by ~6 %
    lo_s, hi_s = np.percentile(rel @ sa, [1, 99])
    c = np.array([cx, cy]) + la * (lo_l + hi_l) / 2 + sa * (lo_s + hi_s) / 2
    L, S = hi_l - lo_l, hi_s - lo_s
    return {'ok': True, 'center': c.tolist(), 'top_z': float(np.median(band[:, 2])), 'size': [float(L), float(S)],
            'long_axis': la.tolist(), 'short_axis': sa.tolist(), 'yaw_short': float(np.arctan2(sa[1], sa[0])),
            'ends': [(c - la * L / 2).tolist(), (c + la * L / 2).tolist()],
            'near_face_x': float((c - sa * S / 2)[0]), 'n_top': int(len(band)), 'frame': Path(fdir).name}


def moved_with_hands(fdir_before, fdir_after, box_point, half=40, min_expected_px=8.0):
    """Was the box carried during a torso move (head joints unchanged)? Hypotheses, in the 'after' image:
    carried -> the box top keeps its pixels (camera and hands move together with the chest);
    left on the floor -> it appears where the fixed base-frame point `box_point` projects with the
    'after' camera pose. The printed box top matches well (score ~0.9); the bare floor does not, so
    no floor reference is used. held=None when the two hypotheses are too close to tell apart."""
    a = cv2.imread(str(Path(fdir_before) / 'head_camera.jpg'), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(Path(fdir_after) / 'head_camera.jpg'), cv2.IMREAD_GRAYSCALE)
    ma = json.loads((Path(fdir_before) / 'meta.json').read_text())
    mb = json.loads((Path(fdir_after) / 'meta.json').read_text())
    uv_before, _ = project(ma, box_point)
    uv_static, _ = project(mb, box_point)
    expected = float(np.linalg.norm(uv_static - uv_before))
    u, v = int(uv_before[0]), int(uv_before[1])
    pad = int(expected) + 30
    if not (half + pad <= u < a.shape[1] - half - pad and half + pad <= v < a.shape[0] - half - pad):
        return {'held': None, 'why': 'box patch too close to the image border', 'expected_static_shift_px': expected}
    tpl = a[v - half:v + half, u - half:u + half]
    res = cv2.matchTemplate(b[v - half - pad:v + half + pad, u - half - pad:u + half + pad], tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    found = np.array([u + loc[0] - pad, v + loc[1] - pad], float)
    d_held, d_static = float(np.linalg.norm(found - uv_before)), float(np.linalg.norm(found - uv_static))
    held = None if expected < min_expected_px or score < 0.5 else bool(d_held < d_static)
    return {'held': held, 'expected_static_shift_px': round(expected, 1), 'found_shift_px': round(d_held, 1),
            'dist_to_static_px': round(d_static, 1), 'score': round(float(score), 3)}
