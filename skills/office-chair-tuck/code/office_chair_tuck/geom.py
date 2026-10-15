"""Frame geometry: depth -> base-frame point cloud, top-down maps, pixel <-> base projections.

Frames are observation directories runtime/frames/<frame_id>/ with head_camera.jpg, head_camera_depth.png
(uint16 mm, 0 = invalid) and meta.json (K, T_base_cam). Nothing here moves the robot.
"""
import json
from pathlib import Path

import cv2
import numpy as np

def _workspace():
    """The agent workspace: the first directory (cwd or above, then this file's parents) holding runtime/ and
    robot-model/."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (start, *start.parents):
            if (d / 'runtime').is_dir() and (d / 'robot-model').exists():
                return d
    return Path.cwd()


ROOT = _workspace()
CAM = 'head_camera'


def frame_dir(frame):
    p = Path(str(frame))
    return p if p.is_absolute() or p.exists() else ROOT / 'runtime' / 'frames' / str(frame)


def load(frame):
    """-> dict(rgb BGR, depth metres (0 = invalid), K, T (base<-cam), meta)."""
    d = frame_dir(frame)
    meta = json.loads((d / 'meta.json').read_text())
    cam = meta['cameras'][CAM]
    rgb = cv2.imread(str(d / cam['rgb']))
    depth = cv2.imread(str(d / cam['depth']), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return {'rgb': rgb, 'depth': depth, 'K': np.array(cam['K']), 'T': np.array(cam['T_base_cam']), 'meta': meta,
            'dir': d}


def cloud(f, step=2, max_range=6.0, mask=None):
    """Base-frame points (N, 3), their pixels (N, 2) [u, v] and colours (N, 3) BGR, every `step` px."""
    depth, K, T = f['depth'], f['K'], f['T']
    H, W = depth.shape
    vs, us = np.mgrid[0:H:step, 0:W:step]
    z = depth[vs, us]
    ok = (z > 0.2) & (z < max_range)
    if mask is not None:
        ok &= mask[vs, us] > 0
    us, vs, z = us[ok], vs[ok], z[ok]
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    pc = np.c_[x, y, z]
    pb = pc @ T[:3, :3].T + T[:3, 3]
    return pb, np.c_[us, vs], f['rgb'][vs, us]


def project(f, pts):
    """Base-frame points (N, 3) -> pixels (N, 2) and camera depth (N,)."""
    T = np.linalg.inv(f['T'])
    pc = np.asarray(pts, float) @ T[:3, :3].T + T[:3, 3]
    uv = pc[:, :2] / pc[:, 2:3] * [f['K'][0, 0], f['K'][1, 1]] + [f['K'][0, 2], f['K'][1, 2]]
    return uv, pc[:, 2]


def topdown(pts, cols=None, xlim=(-0.5, 4.0), ylim=(-3.0, 1.5), res=0.01, zlim=(0.03, 2.0), color_by='height'):
    """Top-down image (x up, y left) of base-frame points; -> (BGR image, to_px function)."""
    W = int((ylim[1] - ylim[0]) / res)
    H = int((xlim[1] - xlim[0]) / res)
    img = np.full((H, W, 3), 255, np.uint8)
    sel = (pts[:, 2] > zlim[0]) & (pts[:, 2] < zlim[1])
    p = pts[sel]
    r = ((xlim[1] - p[:, 0]) / res).astype(int)
    c = ((ylim[1] - p[:, 1]) / res).astype(int)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    if color_by == 'height':
        h = np.clip((p[:, 2] - zlim[0]) / (zlim[1] - zlim[0]), 0, 1)
        colors = cv2.applyColorMap((h * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    else:
        colors = cols[sel]
    order = np.argsort(p[:, 2])        # higher points drawn last
    rr, cc, kk = r[order], c[order], colors[order]
    okk = ok[order]
    img[rr[okk], cc[okk]] = kk[okk]

    def to_px(x, y):
        return int((ylim[1] - y) / res), int((xlim[1] - x) / res)

    # grid every 0.5 m
    for gx in np.arange(np.ceil(xlim[0] * 2) / 2, xlim[1], 0.5):
        cv2.line(img, to_px(gx, ylim[0]), to_px(gx, ylim[1]), (220, 220, 220), 1)
        cv2.putText(img, f'{gx:.1f}', to_px(gx, ylim[1] - 0.02), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    for gy in np.arange(np.ceil(ylim[0] * 2) / 2, ylim[1], 0.5):
        cv2.line(img, to_px(xlim[0], gy), to_px(xlim[1], gy), (220, 220, 220), 1)
        cv2.putText(img, f'{gy:.1f}', to_px(xlim[0] + 0.02, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    # robot footprint (approx 0.6 x 0.6 m around the base origin)
    cv2.rectangle(img, to_px(0.3, 0.3), to_px(-0.3, -0.3), (0, 0, 0), 1)
    cv2.arrowedLine(img, to_px(0, 0), to_px(0.4, 0), (0, 0, 0), 2)
    return img, to_px
