"""Bucket and cabinet-top perception from one head-camera frame: Gemini says WHICH object (target bucket vs the
red distractor, the free part of the cabinet top) and where in the image; depth says where in metres.

Bucket: Gemini segmentation box/mask of the target -> orange pixels (HSV) inside it -> depth points -> rim height
(top percentile) and rim circle (least squares on the top band; inner far wall + outer near wall both lie on it)."""
import cv2
import numpy as np

from lib import frames, vlm

from bimanual_bucket_rim_pnp_params import ORANGE_HSV, RIM_BAND, R_RANGE, R_PRIOR  # hyper-parameters: bimanual_bucket_rim_pnp_params.py
BUCKET_DESC = ('the orange plastic bucket standing on the floor next to the white drawer cabinet that has a cardboard '
               'box on its black top (NOT a red bucket far away between other cabinets)')


def orange_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(ORANGE_HSV[0]), np.array(ORANGE_HSV[1]))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return (m > 0).astype(np.uint8)


def mask_points(fid, mask, stride=2):
    """Base-frame points (N,3) and pixels (N,2) of the mask pixels with valid depth (flying edge pixels dropped)."""
    P, pix, _, _ = frames.load(fid, stride=stride, max_depth=4.0, drop_edges=True, edge_jump=0.04)
    sel = mask[pix[:, 1], pix[:, 0]] > 0
    return P[sel], pix[sel]


def fit_circle(xy):
    """Algebraic (Kasa) circle fit -> (centre xy, radius)."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    return c, float(np.sqrt(max(sol[2] + c @ c, 1e-9)))


def fit_circle_fixed_r(xy, r, c0):
    c = np.array(c0, float)
    for _ in range(50):
        d = xy - c
        n = np.linalg.norm(d, axis=1) + 1e-9
        res = n - r
        J = -d / n[:, None]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c += step
        if np.linalg.norm(step) < 1e-5:
            break
    return c


def rim_from_points(P, support_z=0.0):
    """Rim top height and circle from bucket points (outer near wall + inner far wall)."""
    z = P[:, 2]
    keep = (z > support_z + 0.05) & (z < support_z + 0.6)
    P = P[keep]
    if len(P) < 80:
        return {'ok': False, 'why': f'only {len(P)} bucket depth points'}
    top = float(np.percentile(P[:, 2], 98))
    band = P[P[:, 2] > top - RIM_BAND]
    c, r = fit_circle(band[:, :2]) if len(band) >= 30 else (None, None)
    how = 'rim band'
    if c is None or not (R_RANGE[0] <= r <= R_RANGE[1]):
        c = fit_circle_fixed_r(band[:, :2] if len(band) >= 30 else P[:, :2], R_PRIOR,
                               P[:, :2].mean(0))
        r, how = R_PRIOR, f'fixed r (free fit gave {r})'
    res = np.abs(np.linalg.norm(band[:, :2] - c, axis=1) - r)
    return {'ok': True, 'center': [float(c[0]), float(c[1])], 'radius': float(r), 'rim_z': top,
            'height': top - support_z, 'n_points': int(len(P)), 'n_band': int(len(band)),
            'fit_residual_m': float(np.median(res)), 'fit': how}


def floor_z_near(fid, xy, r_in=0.25, r_out=0.6):
    """Median z of floor-like depth points in a ring around xy (the floor reads a few cm off in some frames)."""
    P, _, _, _ = frames.load(fid, stride=4, max_depth=4.0)
    d = np.linalg.norm(P[:, :2] - np.asarray(xy), axis=1)
    ring = P[(d > r_in) & (d < r_out) & (P[:, 2] < 0.12)]
    return float(np.median(ring[:, 2])) if len(ring) > 50 else 0.0


def locate_bucket(ctx, fid, describe=BUCKET_DESC, ask_details=True, log=print):
    """Gemini picks the target bucket (segmentation box); the convex orange region inside it is the silhouette;
    a truncated-cone model standing on the floor is fitted to it (centre, height); depth gives the initial guess,
    the floor height seen in this frame and a cross-check. -> {ok, center, rim_z, height, radius, iou, floor_z,
    touches_border, depth_center, fully_visible, upright, empty, ears, box_px, frame} or {ok False, why, frame}."""
    seg = vlm.segment(ctx, fid, describe, log=log)
    if not seg['ok']:
        return {'ok': False, 'why': f"gemini segment: {seg.get('why')}", 'frame': fid}
    _, _, rgb, meta = frames.load(fid, stride=16)
    cands = []
    for m in seg['masks']:
        tm = target_mask(rgb, m['box_px'])
        if tm is not None:
            cands.append((int(tm.sum()), m, tm))
    if not cands or max(c[0] for c in cands) < 400:
        return {'ok': False, 'why': f'no orange region in the Gemini boxes {[m["box_px"] for m in seg["masks"]]}',
                'frame': fid}
    _, m, tm = max(cands, key=lambda c: c[0])
    H, W = tm.shape
    ys, xs = np.nonzero(tm)
    border = bool(xs.min() <= 3 or ys.min() <= 3 or xs.max() >= W - 4 or ys.max() >= H - 4)
    P, _ = mask_points(fid, tm & orange_mask(rgb))
    if len(P) < 50:
        return {'ok': False, 'why': f'only {len(P)} depth points on the bucket', 'frame': fid}
    c_depth = np.median(P[:, :2], axis=0)
    fz = floor_z_near(fid, c_depth)
    fit = fit_silhouette(fid, tm, c_depth, z0=fz)
    out = {'ok': True, 'frame': fid, 'center': fit['center'], 'rim_z': fit['rim_z'], 'height': fit['height'],
           'radius': BUCKET_RT, 'iou': fit['iou'], 'floor_z': fz, 'touches_border': border,
           'depth_center': [float(v) for v in c_depth], 'box_px': m['box_px'], 'n_mask_px': int(tm.sum())}
    if ask_details:
        g = vlm.point(ctx, fid, f'Robot head-camera image. Look at {describe}.',
                      points={'ears': ('the places on the outside of that bucket just below its rim where a wire '
                                       'handle is attached or where the two handle mounting bosses are (0, 1 or 2 '
                                       'points)', 2)},
                      flags={'fully_visible': 'that whole bucket is in the image, not cut by the image border or hidden',
                             'upright': 'that bucket stands upright on its bottom, opening up',
                             'empty': 'that bucket looks empty'}, log=log)
        if g['ok']:
            out.update(fully_visible=g['flags']['fully_visible'], upright=g['flags']['upright'],
                       empty=g['flags']['empty'], ears_px=g['px']['ears'])
            ears = []
            for u, v in g['px']['ears']:
                q = frames.ray_plane(fid, u, v, fit['rim_z'] - 0.04)
                if q is not None:
                    ang = float(np.arctan2(q[1] - fit['center'][1], q[0] - fit['center'][0]))
                    ears.append({'px': [u, v], 'xy': [float(q[0]), float(q[1])], 'angle': ang})
            out['ears'] = ears
        else:
            out['gemini_point_why'] = g.get('why')
    if log:
        log(f"bucket: centre {np.round(out['center'], 3).tolist()} rim z {out['rim_z']:.3f} height {out['height']:.3f} "
            f"iou {out['iou']:.3f} floor {fz:+.3f} border {border} depth-median {np.round(c_depth, 3).tolist()} "
            f"ears {[round(e['angle'], 2) for e in out.get('ears', [])]}")
    return out


def draw_bucket(fid, det, path, extra_pts=None):
    """Rim circle (green), centre (red), ears (blue) and optional base-frame points (yellow) on the frame."""
    _, _, rgb, meta = frames.load(fid, stride=16)
    img = rgb.copy()
    c, r, z = np.array(det['center']), det['radius'], det['rim_z']
    pts = []
    for a in np.linspace(0, 2 * np.pi, 72):
        uv, dep = frames.project(meta, [c[0] + r * np.cos(a), c[1] + r * np.sin(a), z])
        if dep > 0:
            pts.append(uv)
    cv2.polylines(img, [np.array(pts, np.int32)], True, (0, 255, 0), 2)
    uv, _ = frames.project(meta, [c[0], c[1], z])
    cv2.circle(img, tuple(int(v) for v in uv), 5, (0, 0, 255), -1)
    for e in det.get('ears', []):
        cv2.circle(img, tuple(int(v) for v in e['px']), 7, (255, 0, 0), 2)
    for p in extra_pts or []:
        uv, dep = frames.project(meta, p)
        if dep > 0:
            cv2.circle(img, tuple(int(v) for v in uv), 5, (0, 255, 255), -1)
    if 'box_px' in det:
        u0, v0, u1, v1 = det['box_px']
        cv2.rectangle(img, (u0, v0), (u1, v1), (255, 0, 255), 1)
    cv2.imwrite(str(path), img)
    return str(path)


# ------------------------------------------------------------------ silhouette fit (no depth on the bucket needed)

BUCKET_RB, BUCKET_RT = 0.131, 0.152      # 5-gallon bucket: bottom / rim outer radius (m)


def _silhouette(meta, c, h, z0, rb=BUCKET_RB, rt=BUCKET_RT, shape=(720, 1280)):
    """Image mask of a truncated cone (bottom centre c at height z0, height h) seen by the frame's camera."""
    cam = meta['cameras']['head_camera']
    K, T = np.array(cam['K']), np.array(cam['T_base_cam'])
    Ti = np.linalg.inv(T)
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = np.c_[np.cos(a), np.sin(a)]
    pts = np.r_[np.c_[c + rb * ring, np.full(48, z0)], np.c_[c + rt * ring, np.full(48, z0 + h)]]
    pc_ = (Ti[:3, :3] @ pts.T).T + Ti[:3, 3]
    if (pc_[:, 2] < 0.1).any():
        return None
    uv = (K @ pc_.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [hull], 1)
    return m


def _iou_fn(fid, mask):
    meta = frames.meta_of(fid)
    H, W = mask.shape
    valid = np.zeros_like(mask)
    valid[4:H - 4, 4:W - 4] = 1
    ys, xs = np.nonzero(mask)
    pad = 120
    v0, v1, u0, u1 = max(0, ys.min() - pad), min(H, ys.max() + pad), max(0, xs.min() - pad), min(W, xs.max() + pad)
    M = (mask & valid)[v0:v1, u0:u1].astype(bool)

    def iou(c, h, z0):
        s = _silhouette(meta, np.asarray(c, float), h, z0, shape=(H, W))
        if s is None:
            return -1.0
        S = (s & valid)[v0:v1, u0:u1].astype(bool)
        return float((S & M).sum() / max(1, (S | M).sum()))
    return iou


def fit_silhouette(fid, mask, c0, z0=0.0, h_range=(0.30, 0.42), span=0.20):
    """Coarse-to-fine search of the bucket's bottom centre (x, y) and height maximising the IoU of the model
    silhouette (truncated cone on the floor plane z0) with `mask` (the target's pixels, convex). Pixels at the
    image border are ignored (cut silhouettes). -> {center, height, rim_z, iou}"""
    iou = _iou_fn(fid, mask)
    best = (-1.0, np.asarray(c0, float), 0.36)
    for step, sp, hs in ((0.02, span, np.arange(h_range[0], h_range[1] + 1e-9, 0.02)),
                         (0.006, 0.024, None), (0.002, 0.006, None)):
        c_ref, h_ref = best[1].copy(), best[2]
        heights = hs if hs is not None else np.arange(h_ref - 2 * step, h_ref + 2 * step + 1e-9, step)
        for h in heights:
            for dx in np.arange(-sp, sp + 1e-9, step):
                for dy in np.arange(-sp, sp + 1e-9, step):
                    c = c_ref + [dx, dy]
                    v = iou(c, h, z0)
                    if v > best[0]:
                        best = (v, c, float(h))
    v, c, h = best
    return {'center': [float(c[0]), float(c[1])], 'height': float(h), 'iou': float(v), 'rim_z': float(z0 + h)}


def target_mask(rgb, box_px, pad=10):
    """Convex hull of the largest orange component inside the Gemini box (the white logo counts as bucket)."""
    om = orange_mask(rgb)
    u0, v0, u1, v1 = box_px
    box = np.zeros_like(om)
    box[max(0, v0 - pad):v1 + pad, max(0, u0 - pad):u1 + pad] = 1
    m = om & box
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    out = np.zeros_like(om)
    cv2.fillPoly(out, [hull], 1)
    return out
