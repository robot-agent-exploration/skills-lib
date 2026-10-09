"""Perception for push_chair: Gemini points on the head-camera image, metres from depth.

chair_pose(ctx, frame)  -> column (x, y), facing angle, backrest bar ends / height, casters, pixels
desk_edge(ctx, frame)   -> front-edge line of the desk top (two ends, height), its direction and inward normal

Gemini gives pixels (identity, where); geometry (depth, base frame) is computed here. Nothing moves the robot.
"""
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np

import geom

ROOT = geom.ROOT
_spec = importlib.util.spec_from_file_location('vlm', ROOT / '.agents/skills/gemini/scripts/vlm.py')
vlm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vlm)

CHAIR_POINTS = {
    'bar_left': 'the left end of the top edge of the office chair backrest (as seen in the image)',
    'bar_right': 'the right end of the top edge of the office chair backrest (as seen in the image)',
    'bar_mid': 'the middle of the top edge of the office chair backrest',
    'hub': 'the centre of the chair base, where the five legs of the star base meet below the seat column',
    'casters': ('each visible caster wheel of the office chair', 5),
}
DESK_POINTS = {
    'edge_left': 'the left end of the front edge of the light-wood desk top (the edge nearest the camera)',
    'edge_right': 'the right end of the front edge of the light-wood desk top (the edge nearest the camera) '
                  'that is visible in the image',
    'edge_pts': ('points along the front edge of the light-wood desk top, nearest the camera, evenly spaced over '
                 'the visible part of the edge', 8),
}


def depth_point(f, uv, radius=3):
    """Base-frame point of pixel uv from the median valid depth in a window; None if too little depth."""
    u, v = int(round(uv[0])), int(round(uv[1]))
    D = f['depth']
    H, W = D.shape
    w = D[max(0, v - radius):min(H, v + radius + 1), max(0, u - radius):min(W, u + radius + 1)]
    ok = w[(w > 0.2) & (w < 6)]
    if ok.size < 0.4 * w.size:
        return None
    z = float(np.median(ok))
    K, T = f['K'], f['T']
    pc = np.array([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    return T[:3, :3] @ pc + T[:3, 3]


def fit_circle(xy, r=None):
    """Least squares circle centre (optionally fixed radius r) of points (N, 2)."""
    xy = np.asarray(xy, float)
    if r is None:
        A = np.c_[2 * xy, np.ones(len(xy))]
        b = (xy ** 2).sum(1)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        return c[:2], math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    c = xy.mean(0)
    for _ in range(50):
        d = xy - c
        n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        c = (xy - r * d / n).mean(0)
    return c, r


def chair_pose(ctx, frame, log=print, star_r=0.33):
    """Chair column, facing and backrest bar from one frame (frame id or directory)."""
    f = geom.load(frame)
    r = vlm.point(ctx, f['dir'], 'Locate these parts of the office chair (the one with a mesh backrest and a '
                  'five-leg star base on casters).', CHAIR_POINTS, camera=geom.CAM, log=log)
    if not r['ok']:
        return {'ok': False, 'why': 'gemini: ' + r.get('why', ''), 'frame': str(f['dir'])}
    px = r['px']
    P = {k: depth_point(f, px[k], 2) for k in ('bar_left', 'bar_right', 'bar_mid', 'hub') if px.get(k)}
    cas = [(uv, depth_point(f, uv, 2)) for uv in px.get('casters') or []]
    cas = [(uv, p) for uv, p in cas if p is not None and p[2] < 0.12]
    out = {'ok': False, 'frame': str(f['dir']), 'px': px, 'points': {k: (None if v is None else np.round(v, 3).tolist())
                                                                       for k, v in P.items()},
           'casters': [np.round(p, 3).tolist() for _, p in cas]}
    if P.get('bar_left') is None or P.get('bar_right') is None:
        out['why'] = 'no depth at a backrest bar end'
        return out
    bl, br = P['bar_left'], P['bar_right']
    d = br[:2] - bl[:2]
    width = float(np.linalg.norm(d))
    n = np.array([-d[1], d[0]]) / (width + 1e-9)        # perpendicular to the bar
    # column: caster circle if 3+ casters, else hub point
    col = None
    if len(cas) >= 3:
        col, _ = fit_circle([p[:2] for _, p in cas], star_r)
        out['column_from'] = f'{len(cas)} casters'
    elif P.get('hub') is not None:
        col = P['hub'][:2]
        out['column_from'] = 'hub'
    bar_c = (bl[:2] + br[:2]) / 2
    if col is None:
        out['why'] = 'no column (casters / hub)'
        return out
    if np.dot(col - bar_c, n) < 0:                        # facing = from the backrest towards the column/seat
        n = -n
    out.update(ok=True, column=np.round(col, 3).tolist(), facing_rad=round(math.atan2(n[1], n[0]), 3),
               bar_left=np.round(bl, 3).tolist(), bar_right=np.round(br, 3).tolist(),
               bar_centre=np.round(bar_c, 3).tolist(), bar_z=round(float((bl[2] + br[2]) / 2), 3),
               bar_width=round(width, 3), bar_behind_column=round(float(np.dot(col - bar_c, n)), 3))
    return out


def desk_edge(ctx, frame, log=print):
    """Front edge of the desk top as a line in the base frame (fit to Gemini edge points with depth)."""
    f = geom.load(frame)
    r = vlm.point(ctx, f['dir'], 'Locate the front edge of the light-wood (yellowish) desk top.', DESK_POINTS,
                  camera=geom.CAM, log=log)
    if not r['ok']:
        return {'ok': False, 'why': 'gemini: ' + r.get('why', ''), 'frame': str(f['dir'])}
    px = r['px']
    uvs = [px['edge_left'], px['edge_right']] + list(px.get('edge_pts') or [])
    pts = [depth_point(f, uv, 2) for uv in uvs if uv]
    pts = np.array([p for p in pts if p is not None and 0.5 < p[2] < 1.0])
    if len(pts):                                        # keep points at the desk-top height (edge pixels can hit the floor)
        pts = pts[np.abs(pts[:, 2] - np.median(pts[:, 2])) < 0.03]
    out = {'ok': False, 'frame': str(f['dir']), 'px': px, 'points': np.round(pts, 3).tolist()}
    if len(pts) < 3:
        out['why'] = f'only {len(pts)} edge points with depth'
        return out
    xy = pts[:, :2]
    c = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - c)
    u = vt[0]
    s = (xy - c) @ u
    resid = np.abs((xy - c) @ np.array([-u[1], u[0]]))
    nrm = np.array([u[1], -u[0]])                       # inward normal: away from the robot (origin)
    if np.dot(-c, nrm) > 0:
        nrm = -nrm
    out.update(ok=True, centre=np.round(c, 3).tolist(), direction=np.round(u, 3).tolist(),
               ends=[np.round(c + s.min() * u, 3).tolist(), np.round(c + s.max() * u, 3).tolist()],
               inward_normal=np.round(nrm, 3).tolist(), facing_rad=round(math.atan2(nrm[1], nrm[0]), 3),
               height=round(float(np.median(pts[:, 2])), 3), max_resid=round(float(resid.max()), 3))
    return out


def draw(frame, result, path):
    """Draw the Gemini pixels of a result on the frame image."""
    f = geom.load(frame)
    im = f['rgb'].copy()
    for k, v in (result.get('px') or {}).items():
        for uv in (v if v and isinstance(v[0], list) else [v]):
            if uv:
                cv2.circle(im, tuple(int(a) for a in uv), 6, (0, 0, 255), 2)
                cv2.putText(im, k, (int(uv[0]) + 8, int(uv[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(path), cv2.resize(im, (640, 360)))


def backrest_bar(frame, x_range=(0.4, 1.4), y_half=0.45, z_range=(0.95, 1.07), v_max=90):
    """Backrest top bar from depth + colour (dark points in a height band in front of the robot), robot base frame.
    Fits x = a*y + b. -> {ok, a, b, yaw_rad (chair facing, 0 = robot +x), centre (x, y), y_min, y_max, z_top, n}"""
    f = geom.load(frame)
    pts, px, cols = geom.cloud(f, step=1, max_range=3)
    hsv = cv2.cvtColor(cols.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sel = ((hsv[:, 2] < v_max) & (pts[:, 2] > z_range[0]) & (pts[:, 2] < z_range[1]) & (pts[:, 0] > x_range[0])
           & (pts[:, 0] < x_range[1]) & (np.abs(pts[:, 1]) < y_half))
    q = pts[sel]
    if len(q) < 200:
        return {'ok': False, 'why': f'only {len(q)} bar points', 'n': int(len(q))}
    a, b = np.polyfit(q[:, 1], q[:, 0], 1)
    r = q[:, 0] - (a * q[:, 1] + b)
    q = q[np.abs(r) < 0.03]
    a, b = np.polyfit(q[:, 1], q[:, 0], 1)
    y0, y1 = np.percentile(q[:, 1], [2, 98])
    yc = (y0 + y1) / 2
    return {'ok': True, 'a': round(float(a), 4), 'b': round(float(b), 4), 'yaw_rad': round(float(-np.arctan(a)), 4),
            'centre': [round(float(a * yc + b), 3), round(float(yc), 3)], 'y_min': round(float(y0), 3),
            'y_max': round(float(y1), 3), 'z_top': round(float(np.percentile(q[:, 2], 95)), 3), 'n': int(len(q))}


def desk_front(frame, z_range=(0.70, 0.80), x_range=(0.6, 2.4), y_half=0.9, v_min=140, s_max=90):
    """Desk-top front edge from the light-wood colour + depth: per 0.1 m y-bin the nearest desk-top x.
    -> {ok, bins: [(y, x_front)], x_front_median, height}"""
    f = geom.load(frame)
    pts, px, cols = geom.cloud(f, step=1, max_range=3)
    hsv = cv2.cvtColor(cols.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sel = ((hsv[:, 2] > v_min) & (hsv[:, 1] < s_max) & (pts[:, 2] > z_range[0]) & (pts[:, 2] < z_range[1])
           & (pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1]) & (np.abs(pts[:, 1]) < y_half))
    q = pts[sel]
    if len(q) < 500:
        return {'ok': False, 'why': f'only {len(q)} desk-top points'}
    bins = []
    for yb in np.arange(-y_half, y_half, 0.1):
        s = q[(q[:, 1] >= yb) & (q[:, 1] < yb + 0.1)]
        if len(s) > 50:
            bins.append((round(float(yb + 0.05), 2), round(float(np.percentile(s[:, 0], 1)), 3)))
    xs = np.array([b[1] for b in bins])
    return {'ok': True, 'bins': bins, 'x_front_median': round(float(np.median(xs)), 3),
            'height': round(float(np.median(q[:, 2])), 3)}
