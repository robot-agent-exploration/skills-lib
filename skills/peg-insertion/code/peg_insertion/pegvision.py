"""Peg top-face detection in a wrist frame: centre and yaw on the plane of the peg top."""
import cv2
import numpy as np

from .geom import pixel_on_plane, project, rgb

SYM = {'hexagon': 6, 'rectangle': 4, 'triangle': 3}


def top_face_contour(img, uv0, win=110, dark_max=None):
    """Contour of the dark top face nearest uv0 in a window around it (full-res pixels)."""
    u0, v0 = int(uv0[0]), int(uv0[1])
    h, w = img.shape[:2]
    x0, y0 = max(0, u0 - win), max(0, v0 - win)
    x1, y1 = min(w, u0 + win), min(h, v0 + win)
    g = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    if dark_max is None:
        bg = np.percentile(g, 70)
        dark_max = min(bg - 40, (bg + np.percentile(g, 2)) / 2)
    m = (g < dark_max).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    best, bd = None, 1e9
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 300:
            continue
        d = np.hypot(cent[i, 0] + x0 - u0, cent[i, 1] + y0 - v0)
        if d < bd:
            best, bd = i, d
    if best is None:
        return None
    cs, _ = cv2.findContours((lab == best).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
    return c + [x0, y0]


def fit_top(frame, cam, uv0, z_top, shape, win=110, dark_max=None):
    """Back-project the top-face contour to z_top and fit centre (area centroid) and yaw.

    yaw (rad): for hexagon/rectangle/triangle, the direction of one edge-normal, modulo 2*pi/n
    (n = 6, 4 (rectangle axes; long/short resolved separately), 3). Returns dict or None."""
    img = rgb(frame, cam)
    c = top_face_contour(img, uv0, win, dark_max)
    if c is None or len(c) < 20:
        return None
    P = np.array([pixel_on_plane(frame, cam, uv, z_top)[:2] for uv in c])
    # simplify the metric polygon
    poly = cv2.approxPolyDP((P * 1e5).astype(np.float32).reshape(-1, 1, 2), 0.0006 * 1e5, True).reshape(-1, 2) / 1e5
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    centre = np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)])
    n = SYM[shape]
    e = np.roll(P, -1, 0) - P
    wl = np.linalg.norm(e, axis=1)
    a = np.arctan2(e[:, 1], e[:, 0])
    z = np.sum(wl * np.exp(1j * n * a))
    edge_dir = np.angle(z) / n  # edge direction modulo 2pi/n
    out = {'centre': centre.tolist(), 'area_m2': abs(A), 'edge_dir_rad': float(edge_dir),
           'harmonic_quality': float(abs(z) / wl.sum()), 'n_vertices': len(poly),
           'poly': poly.tolist(), 'contour_px': c}
    if shape == 'rectangle':
        Q = P - centre
        w, V = np.linalg.eigh(Q.T @ Q)
        out['long_axis_rad'] = float(np.arctan2(V[1, 1], V[0, 1]))
        rect = cv2.minAreaRect((P * 1e5).astype(np.float32))
        out['rect_size'] = sorted([rect[1][0] / 1e5, rect[1][1] / 1e5])
    if shape == 'triangle':
        d = P - centre
        r = np.linalg.norm(d, axis=1)
        ang = np.arctan2(d[:, 1], d[:, 0])
        # apex directions: 3-fold harmonic of the radius profile (peaks at apexes)
        zz = np.sum((r - r.mean()) * np.exp(1j * 3 * ang))
        out['apex_dir_rad'] = float(np.angle(zz) / 3)
    return out


def _outline_with_normals(poly, step=0.0004):
    """Dense samples along a closed CCW/CW polygon with outward unit normals."""
    poly = np.asarray(poly, float)
    area2 = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    sgn = 1.0 if area2 > 0 else -1.0  # CCW -> outward normal = (dy, -dx)
    pts, nrm = [], []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        k = max(1, int(L / step))
        t = (np.arange(k)[:, None] + 0.5) / k
        pts.append(a + t * d)
        nrm.append(np.tile(sgn * np.array([d[1], -d[0]]) / L, (k, 1)))
    return np.vstack(pts), np.vstack(nrm)


def _grad(img, blur):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), blur)
    return cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)


def _polar_score(frame, cam, gx, gy, pts, nrm, pose, z, polarity=1.0):
    cx, cy, th = pose
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    P = pts @ R.T + [cx, cy]
    N = nrm @ R.T
    P3 = np.column_stack([P, np.full(len(P), z)])
    uv = project(frame, cam, P3)
    uvn = project(frame, cam, np.column_stack([P + 0.001 * N, np.full(len(P), z)]))
    dn = uvn - uv
    dn /= np.linalg.norm(dn, axis=1, keepdims=True) + 1e-9
    h, w = gx.shape
    ok = (uv[:, 0] >= 1) & (uv[:, 0] < w - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 2)
    if ok.mean() < 0.8:
        return -1e9
    u = uv[ok, 0].astype(np.float32).reshape(-1, 1)
    v = uv[ok, 1].astype(np.float32).reshape(-1, 1)
    ax = cv2.remap(gx, u, v, cv2.INTER_LINEAR).ravel()
    ay = cv2.remap(gy, u, v, cv2.INTER_LINEAR).ravel()
    return float(np.mean(polarity * (ax * dn[ok, 0] + ay * dn[ok, 1])))


def fit_peg_edges(frame, cam, outline, init_xy, z_top, period, span_xy=0.008, blur=1.5, polarity=1.0):
    """Model-based fit of a peg top outline (local frame, metres) to image edges.

    Searches centre within +-span_xy of init_xy and yaw over [0, period). Returns
    dict(centre, yaw (rad, in [0, period)), score, second_best_score_far)."""
    img = rgb(frame, cam)
    gx, gy = _grad(img, blur)
    pts, nrm = _outline_with_normals(outline)
    results = []
    for th in np.arange(0, period, np.radians(3)):
        for x in np.arange(init_xy[0] - span_xy, init_xy[0] + span_xy + 1e-9, 0.001):
            for y in np.arange(init_xy[1] - span_xy, init_xy[1] + span_xy + 1e-9, 0.001):
                results.append((_polar_score(frame, cam, gx, gy, pts, nrm, (x, y, th), z_top, polarity), x, y, th))
    results.sort(reverse=True)
    best = results[0]
    # refine
    s0, x0, y0, t0 = best
    for dxy, dth in ((0.0005, np.radians(1)), (0.0002, np.radians(0.4))):
        for th in t0 + dth * np.arange(-3, 4):
            for x in x0 + dxy * np.arange(-3, 4):
                for y in y0 + dxy * np.arange(-3, 4):
                    s = _polar_score(frame, cam, gx, gy, pts, nrm, (x, y, th), z_top, polarity)
                    if s > best[0]:
                        best = (s, x, y, th)
        s0, x0, y0, t0 = best
    # best score at a clearly different yaw (ambiguity check)
    far = [r for r in results if abs(((r[3] - best[3] + period / 2) % period) - period / 2) > np.radians(12)]
    return {'centre': [best[1], best[2]], 'yaw': float(best[3] % period), 'score': best[0],
            'alt_score': far[0][0] if far else None, 'alt_yaw': float(far[0][3]) if far else None}
