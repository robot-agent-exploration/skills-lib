"""Frame loading, pixel <-> base-frame geometry for saved observation frames."""
import json
from pathlib import Path

import cv2
import numpy as np

# Saved observation frames, relative to the working directory (the workspace root, harness convention).
# (In the session this was <workspace>/runtime/frames resolved from this file's location.)
FRAMES_DIR = Path('runtime') / 'frames'


def load_frame(frame_id, frames_dir=None):
    d = Path(frames_dir or FRAMES_DIR) / frame_id
    meta = json.loads((d / 'meta.json').read_text())
    out = {'meta': meta, 'dir': d}
    for cam, m in meta['cameras'].items():
        out[cam] = {
            'K': np.array(m['K']),
            'T': np.array(m['T_base_cam']),
            'rgb_path': d / m['rgb'],
            'depth_path': d / m['depth'] if m.get('depth') else None,
        }
    return out


def rgb(frame, cam):
    return cv2.imread(str(frame[cam]['rgb_path']))


def depth(frame, cam):
    return cv2.imread(str(frame[cam]['depth_path']), cv2.IMREAD_UNCHANGED)


def pixel_ray(frame, cam, uv):
    """Camera origin and unit ray direction (base frame) through pixel uv."""
    K, T = frame[cam]['K'], frame[cam]['T']
    d_cam = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    d = T[:3, :3] @ d_cam
    return T[:3, 3].copy(), d / np.linalg.norm(d)


def pixel_on_plane(frame, cam, uv, z):
    """Base-frame point where the ray through pixel uv meets the plane z = const."""
    o, d = pixel_ray(frame, cam, uv)
    t = (z - o[2]) / d[2]
    return o + t * d


def project(frame, cam, pts):
    """Base-frame points (N,3) -> pixels (N,2) (no distortion)."""
    K, T = frame[cam]['K'], frame[cam]['T']
    pts = np.atleast_2d(np.asarray(pts, float))
    Tinv = np.linalg.inv(T)
    pc = (Tinv[:3, :3] @ pts.T).T + Tinv[:3, 3]
    uv = (K @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]


def backproject_depth(frame, cam, uv, radius=2):
    """Base-frame point from the median valid depth around uv (like franka_pixel_to_position)."""
    D = depth(frame, cam)
    u, v = int(round(uv[0])), int(round(uv[1]))
    win = D[max(0, v - radius):v + radius + 1, max(0, u - radius):u + radius + 1].astype(float)
    win = win[win > 0]
    if win.size == 0:
        return None
    z = np.median(win) / 1000.0
    K, T = frame[cam]['K'], frame[cam]['T']
    pc = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0]) * z
    return T[:3, :3] @ pc + T[:3, 3]
