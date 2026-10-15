"""Frame geometry for pick & place (no robot motion): object masks -> base-frame points, support height, round
objects from their silhouette ray-cast at rim height, desk front edge from depth, top-down check images.

Everything takes a loaded frame (geom.load) or a frame id and returns data."""
from pathlib import Path

import cv2
import numpy as np

from office_chair_tuck import geom

from .params import COLOURS  # hyper-parameters: params.py


def colour_mask(hsv, name):
    m = np.zeros(hsv.shape[:2], bool)
    for lo, hi in COLOURS[name]:
        m |= cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0
    return m


def frame(f):
    return geom.load(f) if isinstance(f, str) else f


def mask_points(f, mask, erode=2, max_range=3.0):
    """Base-frame points (N, 3) and pixels (N, 2) of a mask (uint8 / bool, full size); eroded against edge depth
    bleeding; points further than 0.12 m (camera depth) from the median are dropped."""
    f = frame(f)
    m = mask.astype(np.uint8)
    if erode:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    P, uv, _ = geom.cloud(f, step=1, max_range=max_range, mask=m)
    if len(P) == 0:
        return P, uv
    d = f['depth'][uv[:, 1], uv[:, 0]]
    k = np.abs(d - np.median(d)) < 0.12
    return P[k], uv[k]


def support_z(f, mask, inner=8, outer=30):
    """Height of the surface the object stands on: median z of a ring of pixels around the mask."""
    f = frame(f)
    m = mask.astype(np.uint8)
    ring = cv2.dilate(m, np.ones((outer, outer), np.uint8)) & ~cv2.dilate(m, np.ones((inner, inner), np.uint8))
    P, _, _ = geom.cloud(f, step=2, max_range=3.0, mask=ring)
    if len(P) < 20:
        return None
    z = P[:, 2]
    lo = np.percentile(z, 5)
    z = z[z < lo + 0.06]                          # the support and its noise, not neighbouring objects
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.004, 0.002))
    i = int(np.argmax(hist))
    near = z[np.abs(z - (edges[i] + 0.001)) < 0.004]
    return float(np.median(near))


def ray_plane(f, uv, z):
    """Base-frame points where the camera rays of pixels uv (N, 2) meet the horizontal plane at height z."""
    f = frame(f)
    K, T = f['K'], f['T']
    uv = np.atleast_2d(np.asarray(uv, float))
    d_cam = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    d = d_cam @ T[:3, :3].T
    o = T[:3, 3]
    t = (z - o[2]) / d[:, 2]
    return o + t[:, None] * d


def round_from_silhouette(f, mask, rim_z):
    """Bowl / bucket (round rim, widest at the rim) from its silhouette: the left and right extreme pixels are the
    rim's widest points; ray-cast to the rim height they give the centre and the radius."""
    vs, us = np.nonzero(mask)
    left = [us.min(), float(np.median(vs[us == us.min()]))]
    right = [us.max(), float(np.median(vs[us == us.max()]))]
    top = [float(np.median(us[vs == vs.min()])), vs.min()]
    L, R, Tp = ray_plane(f, [left, right, top], rim_z)
    r = float(np.hypot(*(L[:2] - R[:2])) / 2)
    c = (L[:2] + R[:2]) / 2
    return {'centre': c.round(3).tolist(), 'r': round(r, 3), 'rim_z': round(float(rim_z), 3),
            'far_rim': Tp.round(3).tolist(), 'extreme_px': [left, right, top]}


def desk_front(f, y_half=0.3, z_band=(0.6, 0.9)):
    """Front edge of the table top ahead (depth only): the table height is the strongest depth level in z_band in
    front of the robot; the edge is the nearest top point per 5 cm y bin (2nd percentile), median over bins.
    -> {ok, x, z, slope (dx/dy), bins}"""
    f = frame(f)
    P, _, _ = geom.cloud(f, step=2, max_range=2.5)
    k = (np.abs(P[:, 1]) < y_half) & (P[:, 2] > z_band[0]) & (P[:, 2] < z_band[1]) & (P[:, 0] > 0.1)
    if k.sum() < 200:
        return {'ok': False, 'why': 'no table points ahead'}
    z = P[k, 2]
    hist, edges = np.histogram(z, bins=np.arange(z_band[0], z_band[1], 0.005))
    zt = float(edges[np.argmax(hist)] + 0.0025)
    top = P[k][np.abs(z - zt) < 0.012]
    bins = []
    for y0 in np.arange(-y_half, y_half, 0.05):
        b = top[(top[:, 1] >= y0) & (top[:, 1] < y0 + 0.05)]
        if len(b) > 15:
            bins.append((round(y0 + 0.025, 3), round(float(np.percentile(b[:, 0], 2)), 3)))
    if len(bins) < 3:
        return {'ok': False, 'why': 'table edge not seen'}
    b = np.array(bins)
    slope = float(np.polyfit(b[:, 0], b[:, 1], 1)[0])
    return {'ok': True, 'x': round(float(np.median(b[:, 1])), 3), 'z': round(zt, 3), 'slope': round(slope, 3),
            'bins': bins}


def topdown_check(f, centre, mask=None, half=0.13, fingers=(), path=None, zobj=None, zdark=(0.76, 1.1)):
    """Top-down image (x up, y left, 4 mm/px, x4) of the object's points (orange), dark (finger) points coloured by
    height and planned finger points (green). Same camera for both, so calibration errors largely cancel."""
    f = frame(f)
    hsv = cv2.cvtColor(f['rgb'], cv2.COLOR_BGR2HSV)
    P, uv, _ = geom.cloud(f, step=1, max_range=2.0)
    cx, cy = centre
    box = (P[:, 0] > cx - half) & (P[:, 0] < cx + half) & (P[:, 1] > cy - half) & (P[:, 1] < cy + half)
    obj = np.zeros(len(P), bool) if mask is None else mask[uv[:, 1], uv[:, 0]] > 0
    if zobj is not None:
        obj &= (P[:, 2] > zobj[0]) & (P[:, 2] < zobj[1])
    dark = (hsv[uv[:, 1], uv[:, 0]][:, 2] < 70) & (P[:, 2] > zdark[0]) & (P[:, 2] < zdark[1]) & ~obj
    res, n = 0.004, int(2 * half / 0.004)
    img = np.full((n, n, 3), 235, np.uint8)

    def ij(x, y):
        return int((cx + half - x) / res), int((cy + half - y) / res)

    for p in P[obj & box]:
        i, j = ij(p[0], p[1])
        if 0 <= i < n and 0 <= j < n:
            img[i, j] = (0, 200, 255)
    for p in P[dark & box]:
        i, j = ij(p[0], p[1])
        if 0 <= i < n and 0 <= j < n:
            t = np.clip((p[2] - zdark[0]) / (zdark[1] - zdark[0]), 0, 1)
            img[i, j] = (int(255 * t), 0, int(255 * (1 - t)))
    big = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    for (x, y) in fingers:
        i, j = ij(x, y)
        cv2.circle(big, (4 * j, 4 * i), 6, (0, 140, 0), -1)
    for k in range(-int(half * 100), int(half * 100) + 1, 4):
        i = int((half - k / 100) / res * 4)
        cv2.putText(big, f'x{cx + k / 100:.2f}', (2, i + 4), 0, 0.35, (0, 0, 0), 1)
        cv2.putText(big, f'y{cy + k / 100:.2f}', (i, big.shape[0] - 4), 0, 0.35, (0, 0, 0), 1)
    path = path or str(geom.ROOT / 'output' / 'pick_place' / f"topdown_{f['dir'].name}.png")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, big)
    return path
