"""Saved head-camera frames (runtime/frames/<id>/: head_camera.jpg, head_camera_depth.png, meta.json)."""
import json
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = next((p for p in Path(__file__).resolve().parents if (p / 'api' / 'runtime.py').exists()),
                 Path(__file__).resolve().parents[3])
FRAMES = WORKSPACE / 'runtime' / 'frames'


def frame_dir(frame_id):
    p = Path(str(frame_id))
    return p if (p / 'meta.json').exists() else FRAMES / str(frame_id)


def load(fdir, stride=2, max_depth=4.5, drop_edges=False, edge_jump=0.05):
    """Base-frame points (N,3) of the valid depth pixels, their pixels (N,2), the RGB image and meta.json.
    drop_edges: skip pixels at depth jumps (> edge_jump m within 5x5): 'flying pixels' between an object
    and the background (they lined the held box's edges and looked like obstacles)."""
    fdir = frame_dir(fdir)
    meta = json.loads((fdir / 'meta.json').read_text())
    cam = meta['cameras']['head_camera']
    K, T = np.array(cam['K']), np.array(cam['T_base_cam'])
    depth = cv2.imread(str(fdir / cam['depth']), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    rgb = cv2.imread(str(fdir / cam['rgb']))
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    d = depth[::stride, ::stride]
    ok = (d > 0.2) & (d < max_depth)
    if drop_edges:
        big = np.where(depth > 0, depth, 10.0)
        k = np.ones((5, 5), np.uint8)
        jump = (cv2.dilate(big, k) - cv2.erode(big, k))[::stride, ::stride]
        ok &= jump < edge_jump
    u, v, d = u[ok], v[ok], d[ok]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d)], 0)
    return (T @ pc)[:3].T, np.stack([u, v], 1), rgb, meta


def project(meta, p):
    cam = meta['cameras']['head_camera']
    K, T = np.array(cam['K']), np.array(cam['T_base_cam'])
    pc = np.linalg.inv(T) @ np.array([*p, 1.0])
    uv = K @ pc[:3]
    return np.array([uv[0] / uv[2], uv[1] / uv[2]]), float(pc[2])


def clusters(P, grid):
    """Connected components of xy points on a grid -> (n, label per point)."""
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    ij = np.floor((P[:, :2] - [x0, y0]) / grid).astype(int)
    occ = np.zeros((ij[:, 0].max() + 1, ij[:, 1].max() + 1), np.uint8)
    occ[ij[:, 0], ij[:, 1]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, _, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    return n, labels[ij[:, 0], ij[:, 1]]
