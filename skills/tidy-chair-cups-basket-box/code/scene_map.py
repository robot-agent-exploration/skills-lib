"""Top-down obstacle map in the odometry frame from saved head-camera frames (depth), for planning base moves.

Frames carry T_base_cam but not the odometry: the caller supplies the odometry pose of each frame (recorded in
output/frame_odom.json by `note_frame`). Nothing here moves the robot.

    python code/compose/scene_map.py out.png [x0 x1 y0 y1]     # map of all noted frames in the CURRENT base frame
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

SKILLS = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[4]          # the workspace (output/, runtime/frames/)
sys.path.insert(0, str(SKILLS / 'office-chair-tuck' / 'code'))
import geom  # noqa: E402

REG = ROOT / 'output' / 'frame_odom.json'


def note_frame(frame_id, odom):
    reg = json.loads(REG.read_text()) if REG.exists() else {}
    reg[frame_id] = {'x': odom['x'], 'y': odom['y'], 'yaw': odom['yaw']}
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(reg, indent=1))


def to_odom(odom, p):
    c, s = math.cos(odom['yaw']), math.sin(odom['yaw'])
    p = np.atleast_2d(p)
    return np.c_[odom['x'] + c * p[:, 0] - s * p[:, 1], odom['y'] + s * p[:, 0] + c * p[:, 1]]


def from_odom(odom, q):
    c, s = math.cos(odom['yaw']), math.sin(odom['yaw'])
    q = np.atleast_2d(q)
    dx, dy = q[:, 0] - odom['x'], q[:, 1] - odom['y']
    return np.c_[c * dx + s * dy, -s * dx + c * dy]


def frame_obstacles(frame_id, odom, zlim=(0.06, 1.6), step=3, max_range=4.5):
    """Obstacle points (N, 3) of one frame in the odometry frame (x, y odom; z). The robot's own body (base-frame
    box x < 0.36, |y| < 0.32 and anything within 0.75 m of the camera above 0.9 m: the arms) is removed."""
    f = geom.load(frame_id)
    pts, _, _ = geom.cloud(f, step=step, max_range=max_range)
    keep = (pts[:, 2] > zlim[0]) & (pts[:, 2] < zlim[1])
    keep &= ~((pts[:, 0] < 0.36) & (pts[:, 0] > -0.4) & (np.abs(pts[:, 1]) < 0.32))
    cam = f['T'][:3, 3]
    keep &= ~((np.linalg.norm(pts - cam, axis=1) < 0.75) & (pts[:, 2] > 0.9))
    p = pts[keep]
    return np.c_[to_odom(odom, p[:, :2]), p[:, 2]]


def all_obstacles(frames=None, **kw):
    reg = json.loads(REG.read_text()) if REG.exists() else {}
    out = []
    for fid, od in reg.items():
        if frames is None or fid in frames:
            try:
                out.append(frame_obstacles(fid, od, **kw))
            except Exception as e:  # noqa: BLE001
                print('skip', fid, e)
    return np.vstack(out) if out else np.zeros((0, 3))


def render(pts_base, path, xlim=(-1.5, 2.5), ylim=(-2.0, 2.5), res=0.02, extra=None):
    """Top-down picture (x up, y left) of base-frame obstacle points coloured by height; `extra`: list of
    (polyline [(x, y)...], BGR colour) drawn on top."""
    W, H = int((ylim[1] - ylim[0]) / res), int((xlim[1] - xlim[0]) / res)
    img = np.full((H, W, 3), 255, np.uint8)

    def px(x, y):
        return int((ylim[1] - y) / res), int((xlim[1] - x) / res)

    for gx in np.arange(math.ceil(xlim[0] * 2) / 2, xlim[1], 0.5):
        cv2.line(img, px(gx, ylim[0]), px(gx, ylim[1]), (225, 225, 225), 1)
        cv2.putText(img, f'{gx:.1f}', px(gx, ylim[1] - 0.03), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    for gy in np.arange(math.ceil(ylim[0] * 2) / 2, ylim[1], 0.5):
        cv2.line(img, px(xlim[0], gy), px(xlim[1], gy), (225, 225, 225), 1)
        cv2.putText(img, f'{gy:.1f}', px(xlim[0] + 0.03, gy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
    p = pts_base[np.argsort(pts_base[:, 2])]
    h = np.clip(p[:, 2] / 1.6, 0, 1)
    col = cv2.applyColorMap((h * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    for (x, y, _), c in zip(p, col):
        u, v = px(x, y)
        if 0 <= u < W and 0 <= v < H:
            img[v, u] = c
    poly = [(-0.335, -0.259), (0.304, -0.259), (0.304, 0.259), (-0.335, 0.259)]
    cv2.polylines(img, [np.array([px(*q) for q in poly])], True, (0, 0, 0), 1)
    cv2.arrowedLine(img, px(0, 0), px(0.4, 0), (0, 0, 0), 2)
    for line, colr in extra or []:
        cv2.polylines(img, [np.array([px(*q) for q in line])], False, colr, 2)
    cv2.imwrite(str(path), img)
    return img


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    from api.runtime import Context
    with Context() as ctx:
        od = ctx.value('rby1_status')['odometry']
    P = all_obstacles()
    pb = np.c_[from_odom(od, P[:, :2]), P[:, 2]]
    lim = [float(v) for v in sys.argv[2:6]] if len(sys.argv) >= 6 else [-1.5, 2.5, -2.0, 2.5]
    render(pb, sys.argv[1] if len(sys.argv) > 1 else ROOT / 'output/map.png', xlim=lim[:2], ylim=lim[2:])
    print('points', len(pb), 'odom', od)
