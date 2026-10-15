"""Frame loading, back-projection and projection helpers (base frame fr3_link0)."""
import json
from pathlib import Path

import cv2 as cv
import numpy as np

# Saved frames live under runtime/frames/ of the working directory (the workspace root, harness
# convention). Library import: was Path(__file__).parents[2] / 'runtime' / 'frames' (workspace layout).
FRAMES = Path('runtime') / 'frames'
# In the source session the left exterior camera was bumped (2026-10-04 ~10:50) and the service
# kept reporting its old extrinsics; a left_cam_correction.json next to this file held a pose
# re-estimated by registering SIFT features (3-D, both depths) against pre-collision frames. It is
# applied only while the service reports exactly the stale pose it was made for and only to frames
# captured after valid_after_epoch. That file is specific to that rig and bump and is NOT shipped:
# without it no correction is applied. Keys: T_base_cam_meta, T_base_cam_corrected (4x4),
# valid_after_epoch (s). The right camera moved too far to correct and was not used.
_CORR = Path(__file__).resolve().parent / 'left_cam_correction.json'
LEFT_FIX = json.loads(_CORR.read_text()) if _CORR.exists() else None
from .params import USABLE_CAMERAS  # hyper-parameters: params.py


def corrected_T(name, T_meta, t_capture=None):
    if (name == 'left' and LEFT_FIX is not None and np.allclose(T_meta, LEFT_FIX['T_base_cam_meta'], atol=1e-6)
            and (t_capture is None or t_capture >= LEFT_FIX.get('valid_after_epoch', 0))):
        return np.array(LEFT_FIX['T_base_cam_corrected'])
    return T_meta


def load_frame(frame_id, cameras=('left', 'right', 'wrist')):
    """{camera: {'rgb', 'depth' (m, float32, 0 invalid), 'K', 'T'}} for a saved frame."""
    folder = FRAMES / frame_id
    meta = json.loads((folder / 'meta.json').read_text())
    out = {'meta': meta}
    for name in cameras:
        info = meta['cameras'].get(name)
        if info is None:
            continue
        rgb = cv.imread(str(folder / info['rgb']))
        depth = None
        if info.get('depth') and (folder / info['depth']).exists():
            depth = cv.imread(str(folder / info['depth']), cv.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        out[name] = {'rgb': rgb, 'depth': depth, 'K': np.array(info['K']),
                     'T': corrected_T(name, np.array(info['T_base_cam']), meta.get('time'))}
    return out


def backproject(cam, mask=None, stride=1, max_depth=2.0):
    """Base-frame points (N,3) and their pixels (N,2 as u,v) for valid depth inside mask."""
    depth = cam['depth']
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[vs, us]
    ok = (d > 0) & (d < max_depth)
    if mask is not None:
        ok &= mask[vs, us] > 0
    us, vs, d = us[ok], vs[ok], d[ok]
    K = cam['K']
    x = (us - K[0, 2]) / K[0, 0] * d
    y = (vs - K[1, 2]) / K[1, 1] * d
    pc = np.stack([x, y, d, np.ones_like(d)], axis=1)
    pb = (cam['T'] @ pc.T).T[:, :3]
    return pb, np.stack([us, vs], axis=1)


def project(points, cam):
    """Pixels (N,2) and camera depth (N,) of base-frame points."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    T_inv = np.linalg.inv(cam['T'])
    pc = (T_inv[:3, :3] @ points.T).T + T_inv[:3, 3]
    K = cam['K']
    u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
    v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1), pc[:, 2]
