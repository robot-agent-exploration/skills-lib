"""Locate wooden pieces on the table from the two fixed exterior RGB-D cameras.

The exterior cameras agree with the robot kinematics within ~2-3 mm (fingertip projection
check, 2026-10-10); the wrist camera showed view-dependent errors up to ~1 cm, so absolute
positions come from here.
"""
import json
from pathlib import Path

import cv2
import numpy as np

FRAMES = Path('runtime') / 'frames'     # saved frames, relative to the working directory (workspace root)
from bandu_params import TABLE_Z, REGION  # hyper-parameters: bandu_params.py
WOOD_H, WOOD_S_MIN, WOOD_V_MIN = (5, 30), 45, 40


def wood_points(frame_id, camera, min_height=0.004):
    folder = FRAMES / frame_id
    meta = json.loads((folder / 'meta.json').read_text())
    cam = meta['cameras'][camera]
    K, T = np.array(cam['K']), np.array(cam['T_base_cam'])
    rgb = cv2.imread(str(folder / cam['rgb']))
    depth = cv2.imread(str(folder / cam['depth']), cv2.IMREAD_UNCHANGED)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] >= WOOD_H[0]) & (hsv[..., 0] <= WOOD_H[1]) &
            (hsv[..., 1] > WOOD_S_MIN) & (hsv[..., 2] > WOOD_V_MIN)).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))          # drop edge pixels (mixed depth)
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u].astype(np.float64) / 1000.0
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], 1) @ T.T
    p = pc[:, :3]
    keep = ((p[:, 0] > REGION['x'][0]) & (p[:, 0] < REGION['x'][1]) & (p[:, 1] > REGION['y'][0]) &
            (p[:, 1] < REGION['y'][1]) & (p[:, 2] > TABLE_Z + min_height) & (p[:, 2] < 0.35))
    return p[keep]


def clusters(points, cell=0.006, min_points=40):
    """Connected components of occupied xy cells; returns list of point arrays."""
    if len(points) == 0:
        return []
    ij = np.floor(points[:, :2] / cell).astype(int)
    ij -= ij.min(0)
    grid = np.zeros(ij.max(0) + 1, np.uint8)
    grid[ij[:, 0], ij[:, 1]] = 1
    grid = cv2.dilate(grid, np.ones((2, 2), np.uint8))
    n, lab = cv2.connectedComponents(grid, connectivity=8)
    labels = lab[ij[:, 0], ij[:, 1]]
    out = [points[labels == k] for k in range(1, n)]
    return [c for c in out if len(c) >= min_points]


def describe(c):
    top = np.percentile(c[:, 2], 97)
    upper = c[c[:, 2] > top - 0.006]
    return {'n': int(len(c)), 'top_z': float(top), 'height': float(top - TABLE_Z),
            'center_top': [float(v) for v in np.median(upper[:, :2], 0)],
            'center_all': [float(v) for v in np.median(c[:, :2], 0)],
            'extent_xy': [float(v) for v in (np.percentile(c[:, :2], 98, 0) - np.percentile(c[:, :2], 2, 0))]}


def locate(frame_id, cameras=('left', 'right')):
    pts = np.vstack([wood_points(frame_id, c) for c in cameras])
    return [describe(c) for c in clusters(pts)]
