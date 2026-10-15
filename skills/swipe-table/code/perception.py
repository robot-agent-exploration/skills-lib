"""Table-top perception from saved observation frames (runtime/frames/<frame_id>/).

Top-down maps: every base-frame grid cell on the table plane is projected into a camera
(K, T_base_cam from meta.json) and its colour sampled, giving an orthographic view of the
table in base coordinates. Flat things (a stain) land on the same cells in the left and
right maps; raised things (the towel) do not. Depth gives a height map of raised things.

Map layout: row 0 is the far edge (x = MAP_X[1]), rows step towards the robot (-x);
column 0 is the robot's left (y = MAP_Y[1]), columns step to the right (-y).
"""
import json
from pathlib import Path

import cv2 as cv
import numpy as np

FRAMES = Path('runtime')/'frames'     # the session's frame store, relative to the workspace root (cwd)

from swipe_table_params import TABLE_Z, MAP_X, MAP_Y, MAP_RES  # hyper-parameters: swipe_table_params.py


def load_frame(frame_id):
    folder = FRAMES/frame_id
    meta = json.loads((folder/'meta.json').read_text())
    frame = {'id': frame_id, 'meta': meta, 'rgb': {}, 'depth': {}}
    for cam, info in meta['cameras'].items():
        frame['rgb'][cam] = cv.imread(str(folder/info['rgb']))
        depth_path = folder/info.get('depth', f'{cam}_depth.png')
        if depth_path.is_file():
            frame['depth'][cam] = cv.imread(str(depth_path), cv.IMREAD_UNCHANGED)
    return frame


def grid():
    """Base-frame x, y of every map cell (2-D arrays)."""
    xs = MAP_X[1] - MAP_RES * np.arange(int(round((MAP_X[1] - MAP_X[0]) / MAP_RES)))
    ys = MAP_Y[1] - MAP_RES * np.arange(int(round((MAP_Y[1] - MAP_Y[0]) / MAP_RES)))
    return np.meshgrid(xs, ys, indexing='ij')


def cell_of(x, y):
    return int(round((MAP_X[1] - x) / MAP_RES)), int(round((MAP_Y[1] - y) / MAP_RES))


def point_of(row, col):
    return MAP_X[1] - row * MAP_RES, MAP_Y[1] - col * MAP_RES


def region_mask(x0, x1, y0, y1):
    X, Y = grid()
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


def project(meta_cam, pts):
    """Base-frame points (N,3) -> pixels (N,2) and camera-frame depth (N,)."""
    T = np.array(meta_cam['T_base_cam'])
    K = np.array(meta_cam['K'])
    pc = (np.asarray(pts, float) - T[:3, 3]) @ T[:3, :3]
    z = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return np.stack([u, v], 1), z


def top_down(frame, cam, z=TABLE_Z):
    """Orthographic BGR map of the plane z seen from one camera, and its validity mask."""
    X, Y = grid()
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    uv, depth = project(frame['meta']['cameras'][cam], pts)
    img = frame['rgb'][cam]
    h, w = img.shape[:2]
    mapx = uv[:, 0].reshape(X.shape).astype(np.float32)
    mapy = uv[:, 1].reshape(X.shape).astype(np.float32)
    out = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    valid = (mapx >= 0) & (mapx < w - 1) & (mapy >= 0) & (mapy < h - 1) & (depth.reshape(X.shape) > 0)
    return out, valid


def cloud(frame, cam, stride=2, colours=False):
    """Base-frame points (N,3) from one camera's depth (and their BGR colours)."""
    meta_cam = frame['meta']['cameras'][cam]
    full = frame['depth'][cam]
    depth = full[::stride, ::stride].astype(np.float32) / 1000.0
    K = np.array(meta_cam['K'])
    T = np.array(meta_cam['T_base_cam'])
    vs, us = np.mgrid[0:full.shape[0]:stride, 0:full.shape[1]:stride]
    ok = depth > 0
    zc = depth[ok]
    pc = np.stack([(us[ok] - K[0, 2]) * zc / K[0, 0], (vs[ok] - K[1, 2]) * zc / K[1, 1], zc], 1)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    if not colours:
        return pts
    rgb = frame['rgb'][cam]
    if rgb.shape[:2] != full.shape[:2]:
        rgb = cv.resize(rgb, (full.shape[1], full.shape[0]))
    return pts, rgb[::stride, ::stride][ok]


def height_map(frame, cams=('left', 'right'), stride=2, max_z=0.6):
    """Max height above the table per map cell from depth (NaN where no data)."""
    X, _ = grid()
    best = np.full(X.size, -np.inf, np.float32)
    for cam in cams:
        if cam not in frame['depth']:
            continue
        pts = cloud(frame, cam, stride)
        r = np.round((MAP_X[1] - pts[:, 0]) / MAP_RES).astype(int)
        c = np.round((MAP_Y[1] - pts[:, 1]) / MAP_RES).astype(int)
        ok = (r >= 0) & (r < X.shape[0]) & (c >= 0) & (c < X.shape[1]) & (pts[:, 2] < max_z)
        np.maximum.at(best, r[ok] * X.shape[1] + c[ok], (pts[ok, 2] - TABLE_Z).astype(np.float32))
    best[~np.isfinite(best)] = np.nan
    return best.reshape(X.shape)


def _smooth_heights(hm, max_h):
    h = np.where(np.isnan(hm) | (hm > max_h), 0.0, hm)
    return cv.medianBlur((h * 1000).clip(0, 255).astype(np.uint8), 5).astype(np.float32) / 1000


def grey_map(frame, cam):
    img, valid = top_down(frame, cam)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32), valid, img


def towel_mask(frame, min_h=0.006, max_h=0.15, dark_grey=110, min_area_m2=0.004, dilate_m=0.02):
    """Cells covered by the towel: large dark regions (either exterior camera) that are also
    raised above the table in depth. Dilated by dilate_m. May be empty."""
    hm = np.nan_to_num(height_map(frame), nan=0.0)
    raised = cv.dilate(((hm > min_h) & (hm < max_h)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    dark = np.zeros(hm.shape, bool)
    for cam in ('left', 'right'):
        g, valid, _ = grey_map(frame, cam)
        dark |= (g < dark_grey) & valid
    dark = cv.morphologyEx(dark.astype(np.uint8), cv.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    n, lab, st, _ = cv.connectedComponentsWithStats(dark.astype(np.uint8))
    keep = np.zeros(hm.shape, bool)
    for i in range(1, n):
        comp = lab == i
        if st[i, cv.CC_STAT_AREA] * MAP_RES ** 2 > min_area_m2 and (comp & raised).sum() > 0.3 * comp.sum():
            keep |= comp
    k = int(round(dilate_m / MAP_RES)) * 2 + 1
    return cv.dilate(keep.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def towel_cells(frame, cams=('left', 'right'), dark_grey=100, min_h=0.004, max_h=0.15, stride=2):
    """Map cells holding dark 3-D points above the table (the towel), with the max height of
    those points; from each camera's depth + colour at their true positions."""
    X, _ = grid()
    best = np.full(X.size, -np.inf, np.float32)
    for cam in cams:
        if cam not in frame['depth']:
            continue
        pts, bgr = cloud(frame, cam, stride, colours=True)
        grey = bgr.astype(np.float32).mean(1)
        hz = pts[:, 2] - TABLE_Z
        sel = (grey < dark_grey) & (hz > min_h) & (hz < max_h)
        r = np.round((MAP_X[1] - pts[sel, 0]) / MAP_RES).astype(int)
        c = np.round((MAP_Y[1] - pts[sel, 1]) / MAP_RES).astype(int)
        ok = (r >= 0) & (r < X.shape[0]) & (c >= 0) & (c < X.shape[1])
        np.maximum.at(best, r[ok] * X.shape[1] + c[ok], hz[sel][ok].astype(np.float32))
    h = best.reshape(X.shape)
    occ = np.isfinite(h)
    occ = cv.morphologyEx(occ.astype(np.uint8), cv.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    occ = cv.morphologyEx(occ.astype(np.uint8), cv.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return occ, np.where(np.isfinite(h), h, 0.0)


def locate_towel(frame, cams=('left', 'right'), min_area_m2=0.004):
    """The towel = the largest connected region of dark raised points. Returns found, centre
    (x, y) of its footprint, peak (x, y) = centroid of its highest 40 %, top_h (90th
    percentile height above the table), bbox, area and shape 'heap' (top_h >= 2.5 cm) or 'flat'."""
    occ, h = towel_cells(frame, cams)
    n, lab, st, _ = cv.connectedComponentsWithStats(occ.astype(np.uint8))
    if n < 2:
        return {'found': False}
    i = 1 + int(np.argmax(st[1:, cv.CC_STAT_AREA]))
    blob = lab == i
    area = float(blob.sum() * MAP_RES ** 2)
    if area < min_area_m2:
        return {'found': False, 'area_m2': area}
    X, Y = grid()
    w = h[blob]
    w_pos = w[w > 0]
    top_h = float(np.percentile(w_pos, 90)) if w_pos.size else 0.0
    top = blob & (h >= (np.percentile(w_pos, 60) if w_pos.size else 0.0))
    return {'found': True, 'centre': (float(X[blob].mean()), float(Y[blob].mean())),
            'peak': (float(X[top].mean()), float(Y[top].mean())), 'top_h': top_h,
            'shape': 'heap' if top_h >= 0.025 else 'flat', 'area_m2': area, 'mask': blob,
            'bbox': (float(X[blob].min()), float(X[blob].max()), float(Y[blob].min()), float(Y[blob].max()))}


def darkening(frame, cam, ref=None, bg_cells=41):
    """Grey levels darker than clean table per cell (>0 = darker): against a reference frame
    from the same camera, or else against a large median of the current map."""
    g, valid, _ = grey_map(frame, cam)
    if ref is not None:
        g0, valid0, _ = grey_map(ref, cam)
        d = g0 - g
        valid = valid & valid0
    else:
        bg = cv.medianBlur(np.clip(g, 0, 255).astype(np.uint8), bg_cells).astype(np.float32)
        d = bg - g
    d[~valid] = 0
    return d, valid


def detect_stain(frame, ref=None, thresh=35, min_area_m2=2e-5, cluster_m=0.05, exclude=None,
                 roi=(0.28, 0.80, -0.42, 0.42)):
    """Dark marks on the table plane seen by BOTH exterior cameras. The largest mark plus every
    mark within cluster_m of the cluster form the stain. Returns found, mask, area_m2,
    centre (darkness-weighted), bbox (x0, x1, y0, y1), others (marks outside the cluster)."""
    ds = []
    for cam in ('left', 'right'):
        d, valid = darkening(frame, cam, ref)
        ds.append(np.where(valid, d, np.nan))
    both = np.fmin(ds[0], ds[1])     # fmin ignores a NaN side: then the seeing camera decides
    dark = np.nan_to_num(both, nan=0.0)
    mask = (dark > thresh) & region_mask(*roi)
    if exclude is not None:
        mask &= ~exclude
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    n, lab, st, cen = cv.connectedComponentsWithStats(mask.astype(np.uint8))
    comps = []
    for i in range(1, n):
        area = st[i, cv.CC_STAT_AREA] * MAP_RES ** 2
        if area >= min_area_m2:
            comps.append({'id': i, 'area_m2': float(area), 'centre': point_of(cen[i][1], cen[i][0])})
    if not comps:
        return {'found': False, 'mask': mask, 'darkness': dark}
    comps.sort(key=lambda c: -c['area_m2'])
    cluster = [comps[0]]
    grown = True
    while grown:
        grown = False
        for c in comps:
            if c not in cluster and any(np.hypot(c['centre'][0] - k['centre'][0], c['centre'][1] - k['centre'][1])
                                        < cluster_m for k in cluster):
                cluster.append(c)
                grown = True
    smask = np.isin(lab, [c['id'] for c in cluster])
    X, Y = grid()
    xs, ys, w = X[smask], Y[smask], dark[smask]
    return {'found': True, 'mask': smask, 'darkness': dark, 'area_m2': float(smask.sum() * MAP_RES ** 2),
            'centre': (float(np.average(xs, weights=w)), float(np.average(ys, weights=w))),
            'bbox': (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())),
            'n_components': len(cluster), 'others': [c for c in comps if c not in cluster]}


def _lab_maps(frame, cam):
    img, valid = top_down(frame, cam)
    return cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32), valid


def residue(frame, ref=None, db_thresh=5.0, dl_thresh=12.0, roi=(0.28, 0.80, -0.42, 0.42),
            exclude=None, min_cells=4, bg_cells=61):
    """Soy-sauce residue: cells yellower (Lab b*) or darker (L) than clean table in BOTH
    exterior cameras. Clean table = the reference frame (same arm pose, clean table) when
    given, else a large median of the current map. Values are normalised by the ROI median
    to cancel exposure drift. Returns found, mask, area_m2, bbox, centre, db, dl."""
    dbs, dls, valids = [], [], []
    for cam in ('left', 'right'):
        lab, valid = _lab_maps(frame, cam)
        if ref is not None:
            lab0, valid0 = _lab_maps(ref, cam)
            valid = valid & valid0
        else:
            lab0 = np.stack([cv.medianBlur(lab[..., k].clip(0, 255).astype(np.uint8), bg_cells)
                             for k in range(3)], -1).astype(np.float32)
        dbs.append(lab[..., 2] - lab0[..., 2])
        dls.append(lab0[..., 0] - lab[..., 0])
        valids.append(valid)
    valid = valids[0] & valids[1]
    db = np.fmin(dbs[0], dbs[1])
    dl = np.fmin(dls[0], dls[1])
    in_roi = region_mask(*roi) & valid
    if exclude is not None:
        in_roi &= ~exclude
    if in_roi.sum() < 50:
        return {'found': False, 'mask': np.zeros(valid.shape, bool), 'area_m2': 0.0, 'db': db, 'dl': dl,
                'error': 'region not visible in both cameras'}
    db = db - np.median(db[in_roi])
    dl = dl - np.median(dl[in_roi])
    mask = in_roi & ((db > db_thresh) | (dl > dl_thresh))
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    n, lab, st, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(mask.shape, bool)
    for i in range(1, n):
        if st[i, cv.CC_STAT_AREA] >= min_cells:
            keep |= lab == i
    out = {'mask': keep, 'area_m2': float(keep.sum() * MAP_RES ** 2), 'db': db, 'dl': dl,
           'db_p99': float(np.percentile(db[in_roi], 99)), 'dl_p99': float(np.percentile(dl[in_roi], 99))}
    if not keep.any():
        out['found'] = False
        return out
    X, Y = grid()
    xs, ys = X[keep], Y[keep]
    out.update(found=True, centre=(float(xs.mean()), float(ys.mean())),
               bbox=(float(np.percentile(xs, 2)), float(np.percentile(xs, 98)),
                     float(np.percentile(ys, 2)), float(np.percentile(ys, 98))))
    return out


def locate_stain(frame, ref=None, exclude=None, roi=(0.30, 0.80, -0.42, 0.42), cluster_m=0.05,
                 min_area_m2=2e-5):
    """Soy sauce of any shade: cells darker or yellower than clean table in both exterior
    cameras (residue(); without a reference also dark marks from detect_stain()). The largest
    mark plus marks within cluster_m of it form the stain. Returns found, mask, area_m2,
    centre, bbox (x0, x1, y0, y1), n_components, others."""
    r = residue(frame, ref, roi=roi, exclude=exclude)
    mask = r['mask'].copy()
    if ref is None:
        d = detect_stain(frame, None, exclude=exclude, roi=roi)
        if d['found']:
            mask |= d['mask']
    n, lab, st, cen = cv.connectedComponentsWithStats(mask.astype(np.uint8))
    comps = []
    for i in range(1, n):
        area = st[i, cv.CC_STAT_AREA] * MAP_RES ** 2
        if area >= min_area_m2:
            comps.append({'id': i, 'area_m2': float(area), 'centre': point_of(cen[i][1], cen[i][0])})
    if not comps:
        return {'found': False, 'mask': mask, 'area_m2': 0.0}
    comps.sort(key=lambda c: -c['area_m2'])
    X, Y = grid()
    cluster = [comps[0]]
    grown = True
    while grown:
        grown = False
        cm = np.isin(lab, [c['id'] for c in cluster])
        dil = cv.dilate(cm.astype(np.uint8), np.ones((2 * int(cluster_m / MAP_RES) + 1,) * 2, np.uint8)) > 0
        for c in comps:
            if c not in cluster and (dil & (lab == c['id'])).any():
                cluster.append(c)
                grown = True
    smask = np.isin(lab, [c['id'] for c in cluster])
    xs, ys = X[smask], Y[smask]
    return {'found': True, 'mask': smask, 'area_m2': float(smask.sum() * MAP_RES ** 2),
            'centre': (float(xs.mean()), float(ys.mean())),
            'bbox': (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())),
            'n_components': len(cluster), 'others': [c for c in comps if c not in cluster]}


def reference_matches(frame, ref, exclude=None, roi=(0.33, 0.78, -0.40, 0.40), max_median=12.0, max_p95=25.0):
    """Is ref a usable clean-table reference for frame (same fixed cameras, similar light)?
    Compares grey levels on table cells away from anything flagged as residue or excluded."""
    r = residue(frame, ref, roi=roi, exclude=exclude)
    away = ~(cv.dilate(r['mask'].astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)
    stats = {}
    for cam in ('left', 'right'):
        a, va = top_down(frame, cam)
        b, vb = top_down(ref, cam)
        m = va & vb & away & region_mask(*roi)
        if exclude is not None:
            m &= ~exclude
        if m.sum() < 1000:
            return False, {'error': f'{cam}: too few comparable cells'}
        d = np.abs(cv.cvtColor(a, cv.COLOR_BGR2GRAY).astype(np.float32) - cv.cvtColor(b, cv.COLOR_BGR2GRAY).astype(np.float32))
        stats[cam] = (float(np.median(d[m])), float(np.percentile(d[m], 95)))
    ok = all(med <= max_median and p95 <= max_p95 for med, p95 in stats.values())
    return ok, stats


def top_down_fine(frame, cam, roi, res=0.001, z=TABLE_Z):
    """Like top_down() but over roi=(x0, x1, y0, y1) at resolution res (m/cell)."""
    x0, x1, y0, y1 = roi
    xs = x1 - res * np.arange(int(round((x1 - x0) / res)))
    ys = y1 - res * np.arange(int(round((y1 - y0) / res)))
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    uv, depth = project(frame['meta']['cameras'][cam], pts)
    img = frame['rgb'][cam]
    h, w = img.shape[:2]
    mapx = uv[:, 0].reshape(X.shape).astype(np.float32)
    mapy = uv[:, 1].reshape(X.shape).astype(np.float32)
    out = cv.remap(img, mapx, mapy, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    valid = (mapx >= 8) & (mapx < w - 9) & (mapy >= 8) & (mapy < h - 9) & (depth.reshape(X.shape) > 0)
    return out, valid, X, Y


def residue_wrist(frame, ref, roi, res=0.001, db_thresh=4.0, dl_thresh=10.0, min_px=30, exclude_dark=90):
    """Fine residue check with the wrist camera (arm at home in both frames): cells of roi that
    became yellower or darker than in the reference wrist image. Dark things in either frame
    (towel, gripper fingers) are excluded. Returns found, area_m2, bbox, mask, coverage."""
    a, va, X, Y = top_down_fine(frame, 'wrist', roi, res)
    b, vb, _, _ = top_down_fine(ref, 'wrist', roi, res)
    valid = va & vb
    la = cv.cvtColor(a, cv.COLOR_BGR2LAB).astype(np.float32)
    lb = cv.cvtColor(b, cv.COLOR_BGR2LAB).astype(np.float32)
    dark = (la[..., 0] < exclude_dark * 255 / 100 * 0 + exclude_dark) | (lb[..., 0] < exclude_dark)
    dark = cv.dilate(dark.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    valid &= ~dark
    if valid.sum() < 500:
        return {'found': False, 'area_m2': 0.0, 'coverage': float(valid.mean()), 'error': 'wrist view does not cover roi'}
    db = la[..., 2] - lb[..., 2]
    dl = lb[..., 0] - la[..., 0]
    db -= np.median(db[valid])
    dl -= np.median(dl[valid])
    db = cv.GaussianBlur(db, (5, 5), 0)
    dl = cv.GaussianBlur(dl, (5, 5), 0)
    # darker-only changes are shadows (12:34: a shadow line, dl 23, db 3); sauce is also yellower
    mask = valid & ((db > db_thresh) | ((dl > dl_thresh) & (db > 1.5)))
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab, st, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = np.zeros(mask.shape, bool)
    for i in range(1, n):
        if st[i, cv.CC_STAT_AREA] >= min_px:
            keep |= lab == i
    out = {'mask': keep, 'area_m2': float(keep.sum() * res * res), 'coverage': float(valid.mean()),
           'db_p99': float(np.percentile(db[valid], 99)), 'dl_p99': float(np.percentile(dl[valid], 99)),
           'X': X, 'Y': Y, 'db': db, 'dl': dl}
    if not keep.any():
        out['found'] = False
        return out
    xs, ys = X[keep], Y[keep]
    out.update(found=True, bbox=(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())),
               centre=(float(xs.mean()), float(ys.mean())))
    return out
