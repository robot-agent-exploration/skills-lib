"""Board pose registration: fit (cx, cy, theta) so projected CAD hole outlines lie on image edges."""
import cv2
import numpy as np

from .board import MODEL, rot2
from .geom import project, rgb


def outline_samples(names=None, step=0.0007):
    """Dense board-frame points along the top outlines of the given holes (all by default)."""
    names = names or list(MODEL['holes'])
    pts = []
    for n in names:
        poly = np.array(MODEL['holes'][n]['outline'])
        for a, b in zip(poly, np.roll(poly, -1, 0)):
            k = max(1, int(np.linalg.norm(b - a) / step))
            t = np.arange(k)[:, None] / k
            pts.append(a + t * (b - a))
    return np.vstack(pts)


def edge_map(img, blur=3):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), blur)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def score(frame, cam, E, pts_b, pose, z):
    cx, cy, th = pose
    xy = pts_b @ rot2(th).T + np.array([cx, cy])
    zz = z(xy) if callable(z) else np.full(len(xy), z)
    P = np.column_stack([xy, zz])
    uv = project(frame, cam, P)
    h, w = E.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    if ok.mean() < 0.5:
        return -1
    u, v = uv[ok, 0], uv[ok, 1]
    return float(cv2.remap(E, u.astype(np.float32).reshape(-1, 1), v.astype(np.float32).reshape(-1, 1),
                           cv2.INTER_LINEAR).mean())


def fit(frame, cam, init, z, names=None, span=(0.03, 0.03, np.radians(8)), steps=(0.002, 0.002, np.radians(1)),
        refine=3, blur=3):
    E = edge_map(rgb(frame, cam), blur)
    pts = outline_samples(names)
    best = (score(frame, cam, E, pts, init, z), tuple(init))
    sx, sy, st = span
    dx, dy, dt = steps
    for it in range(refine + 1):
        c = best[1]
        for x in np.arange(c[0] - sx, c[0] + sx + 1e-9, dx):
            for y in np.arange(c[1] - sy, c[1] + sy + 1e-9, dy):
                for t in np.arange(c[2] - st, c[2] + st + 1e-9, dt):
                    s = score(frame, cam, E, pts, (x, y, t), z)
                    if s > best[0]:
                        best = (s, (x, y, t))
        sx, sy, st = 2 * dx, 2 * dy, 2 * dt
        dx, dy, dt = dx / 3, dy / 3, dt / 3
    return best
