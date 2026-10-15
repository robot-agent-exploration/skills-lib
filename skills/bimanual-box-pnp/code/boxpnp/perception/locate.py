"""Gemini-seeded localization of a box and a table top in one head-camera frame.

Gemini says WHICH object it is, whether it is fully visible and WHERE it is in the image (pixels); depth says
how far (metres): the top-face rectangle is fitted from depth points around the pointed pixel. This replaces
v1's cardboard-colour gate, the partial-view size rule (its failure caused crash A2) and the odometry table
prior (drifted 0.45-0.56 m). The depth-only detectors in box.py / table.py remain the fallback and the
geometric cross-check (vlm_corner_err_m: Gemini's corners vs the fitted rectangle)."""
import numpy as np

from ..core import frames as pf
from ..core import vlm

from . import box as pbox
from . import table as ptable

BOX_DESC = 'the closed cardboard box'
TABLE_DESC = 'the free-standing white drawer cabinet with a black top'
from ..params import MAX_CORNER_ERR, MIN_MASK_POINTS  # hyper-parameters: boxpnp/params.py


def _rect_corners(d):
    c, la, sa = np.array(d['center']), np.array(d['long_axis']), np.array(d['short_axis'])
    L, S = d['size'][0] / 2, d['size'][1] / 2
    return [c + la * L + sa * S, c - la * L + sa * S, c - la * L - sa * S, c + la * L - sa * S]


def _corner_err(fid, px_corners, rect, z):
    """Mean distance (m) from each Gemini corner (ray met with the plane z) to the nearest rectangle corner."""
    pts = [pf.ray_plane(fid, u, v, z) for u, v in px_corners]
    pts = [p[:2] for p in pts if p is not None]
    if not pts:
        return None
    rc = np.array(rect)
    return float(np.mean([np.min(np.linalg.norm(rc - p, axis=1)) for p in pts]))


def locate_box(ctx, fid, describe=BOX_DESC, support='floor', support_z=0.0, exclude_self=True, log=None):
    """-> box dict (center, size, axes, top_z, ... as box.find_box) + fully_visible, centre_px, vlm_corner_err_m,
    gemini_s; or {ok: False, why, frame}."""
    g = vlm.point(ctx, fid, f'Robot head-camera image. Find {describe} standing on the {support}.',
                  points={'box_top_center': f'centre of the TOP face of {describe}',
                          'box_top_corners': (f'the 4 corners of the TOP face of {describe}', 4)},
                  flags={'box_present': f'{describe} is in the image',
                         'box_fully_visible': f'the whole of {describe} is in the image: not cut by the image '
                                              'border and not hidden behind a robot arm or anything else'},
                  lists={'touching_box': 'other objects standing on or leaning against the box'}, log=log)
    if not g['ok']:
        return {'ok': False, 'why': f"gemini: {g['why']}", 'frame': fid}
    if not g['flags']['box_present'] or not g['px'].get('box_top_center'):
        return {'ok': False, 'why': 'gemini sees no box', 'frame': fid, 'gemini_s': g['seconds']}
    c = g['px']['box_top_center']
    p = pf.pixel_point(fid, *c, radius=6)
    if p is None:
        return {'ok': False, 'why': f'no depth at the box centre pixel {c}', 'frame': fid}
    d = pbox.find_box(fid, near=p[:2], radius=0.35, top_z_hint=float(p[2]), support_z=support_z,
                      require_cardboard=False, exclude_self=exclude_self)
    if not d['ok']:
        return {'ok': False, 'why': f"no box-sized depth top near the Gemini point {np.round(p, 3).tolist()}: "
                                   f"{d.get('why')}", 'frame': fid, 'centre_px': c}
    d.pop('candidates', None)
    err = _corner_err(fid, g['px']['box_top_corners'], _rect_corners(d), d['top_z'])
    d.update(fully_visible=bool(g['flags']['box_fully_visible']), centre_px=c, corners_px=g['px']['box_top_corners'],
             touching=g['lists'].get('touching_box', []), vlm_corner_err_m=err, gemini_s=g['seconds'],
             vlm_centre=p.tolist())
    if log:
        log(f"box: centre {np.round(d['center'], 3).tolist()} size {np.round(d['size'], 3).tolist()} top "
            f"{d['top_z']:.3f} yaw {d['yaw_short']:+.3f} fully_visible {d['fully_visible']} corner err {err}")
    return d


def locate_table(ctx, fid, describe=TABLE_DESC, approx_z=0.70, log=None):
    """-> table dict (center, size, corners, top_z as table.find_table) + top_fully_visible, objects_on_top,
    vlm_corner_err_m; or {ok: False, why, frame}."""
    g = vlm.point(ctx, fid, f'Robot head-camera image. Find {describe}.',
                  points={'top_center': f'centre of the black top of {describe}',
                          'top_corners': (f'the visible corners of the black top of {describe}', 4)},
                  flags={'present': f'{describe} is in the image',
                         'top_fully_visible': 'all 4 corners of its black top are in the image'},
                  lists={'objects_on_top': 'objects standing on its black top'}, log=log)
    if not g['ok']:
        return {'ok': False, 'why': f"gemini: {g['why']}", 'frame': fid}
    if not g['flags']['present'] or not g['px'].get('top_center'):
        return {'ok': False, 'why': 'gemini sees no such table', 'frame': fid}
    seed = pf.ray_plane(fid, *g['px']['top_center'], approx_z)
    cs = [pf.ray_plane(fid, u, v, approx_z) for u, v in g['px']['top_corners']]
    cs = [q[:2] for q in cs if q is not None]
    if seed is None:
        return {'ok': False, 'why': 'table centre ray misses the table plane', 'frame': fid}
    radius = max([float(np.linalg.norm(q - seed[:2])) for q in cs] + [0.5]) + 0.15
    t = ptable.find_table(fid, near=seed[:2], radius=radius)
    if not t['ok']:
        return {'ok': False, 'why': f"no dark top near the Gemini table point: {t.get('why')}", 'frame': fid}
    err = _corner_err(fid, g['px']['top_corners'], t['corners'], t['top_z']) if t['ok'] else None
    t.update(top_fully_visible=bool(g['flags']['top_fully_visible']), objects_on_top=g['lists'].get('objects_on_top', []),
             corners_px=g['px']['top_corners'], vlm_corner_err_m=err, gemini_s=g['seconds'])
    if log:
        log(f"table: centre {np.round(t['center'], 3).tolist()} size {np.round(t['size'], 3).tolist()} top "
            f"{t['top_z']:.3f} corners visible {t['top_fully_visible']} on top {t['objects_on_top']} corner err {err}")
    return t


def table_near_edge_points(ctx, fid, top_z, describe=TABLE_DESC, log=None):
    """Near edge from two Gemini points on it (rays met with z = top_z). LESS ROBUST than table_near_edge (mask):
    with the held box hiding the middle and an arm one side, Gemini put both 'left' / 'right' points on one side
    and one on the drawer face (0.302 m vs 0.394 m). Kept for comparison.
    -> {ok, edge_x (at y = 0), angle (edge direction vs base y), points, frame}"""
    g = vlm.point(ctx, fid, f'Robot head-camera image. The robot may be holding a cardboard box in front of '
                  f'{describe}. Find the NEAR edge of the black table top (the edge closest to the robot).',
                  points={'edge_left': 'a point on that near edge of the black top, left of the held box (or left part)',
                          'edge_right': 'a point on that near edge of the black top, right of the held box (or right part)'},
                  flags={'edge_visible': 'the near edge of the black top is visible on at least one side'}, log=log)
    if not g['ok']:
        return {'ok': False, 'why': f"gemini: {g['why']}", 'frame': fid}
    if not g['flags']['edge_visible']:
        return {'ok': False, 'why': 'gemini: near edge not visible', 'frame': fid}
    pts = [pf.ray_plane(fid, *g['px'][k], top_z) for k in ('edge_left', 'edge_right') if g['px'].get(k)]
    pts = [p for p in pts if p is not None]
    if len(pts) < 2:
        return {'ok': False, 'why': 'edge points missing', 'frame': fid}
    a, b = pts[0][:2], pts[1][:2]
    d = b - a
    t = -a[1] / d[1] if abs(d[1]) > 1e-6 else 0.5
    edge_x = float(a[0] + t * d[0])
    ang = float(np.arctan2(d[0], -d[1]))            # 0 when the edge is parallel to base y
    if log:
        log(f'table near edge: x {edge_x:.3f} at y=0, angle {ang:+.3f}, points {np.round(a, 3).tolist()} '
            f'{np.round(b, 3).tolist()} (gemini {g["seconds"]} s)')
    return {'ok': True, 'edge_x': edge_x, 'angle': ang, 'points': [a.tolist(), b.tolist()], 'frame': fid,
            'gemini_s': g['seconds']}


TOP_DESC = ('the flat black table top surface of the white drawer cabinet (only the black top surface: not the white '
            'cabinet front below it, not the cardboard box or the robot arms above it)')


def table_top_points(ctx, fid, describe=TOP_DESC, z_band=(0.55, 0.90), log=None):
    """Gemini segmentation mask of the table top -> its depth points (base frame), outliers in z removed.
    -> {ok, points (N,3), top_z, mask_px, seconds}"""
    seg = vlm.segment(ctx, fid, describe, log=log)
    if not seg['ok']:
        return {'ok': False, 'why': f"gemini segment: {seg['why']}", 'frame': fid}
    pts, pix, _, _ = pf.load(fid, stride=2, max_depth=3.5)
    P = pts[seg['mask'][pix[:, 1], pix[:, 0]] > 0]
    P = P[(P[:, 2] > z_band[0]) & (P[:, 2] < z_band[1])]
    if len(P) < MIN_MASK_POINTS:
        return {'ok': False, 'why': f'only {len(P)} masked top points: too little of the top in view', 'frame': fid}
    z = float(np.median(P[:, 2]))
    P = P[np.abs(P[:, 2] - z) < 0.03]
    return {'ok': True, 'points': P, 'top_z': z, 'mask_px': int(seg['mask'].sum()), 'seconds': seg['seconds'],
            'frame': fid}


def table_near_edge(ctx, fid, describe=TOP_DESC, y_max=0.65, log=None):
    """Near edge of the table top in front of the robot from Gemini's segmentation mask: the 2nd percentile of x of
    the masked depth points (|y| < y_max), per 0.2 m band of y for the edge direction. Works with the held box
    hiding the middle (points beside it). Measured 0.391 m where 6 Gemini edge points gave 0.394 m and v1's
    far dark-pixel fit gave 0.917 m (true ~0.74 before that drive). -> {ok, edge_x, top_z, bands, frame}"""
    t = table_top_points(ctx, fid, describe, log=log)
    if not t['ok']:
        return t
    P = t['points'][np.abs(t['points'][:, 1]) < y_max]
    if len(P) < MIN_MASK_POINTS // 2:
        return {'ok': False, 'why': f'only {len(P)} top points within |y| < {y_max}', 'frame': fid}
    bands = []
    for lo in np.arange(-y_max, y_max, 0.2):
        b = P[(P[:, 1] >= lo) & (P[:, 1] < lo + 0.2)]
        if len(b) > 40:
            bands.append((round(float(lo + 0.1), 2), round(float(np.percentile(b[:, 0], 2)), 3)))
    edge = float(np.percentile(P[:, 0], 2))
    if log:
        log(f'table near edge (mask): x {edge:.3f}, top z {t["top_z"]:.3f}, bands {bands} ({t["seconds"]} s)')
    return {'ok': True, 'edge_x': edge, 'top_z': t['top_z'], 'bands': bands, 'frame': fid, 'gemini_s': t['seconds']}


def locate_table_mask(ctx, fid, describe=TOP_DESC, log=None):
    """Table top rectangle from Gemini's segmentation mask + depth (minAreaRect of the masked top points).
    -> {ok, center, size, corners, top_z, angle_deg, n, frame} like table.find_table"""
    import cv2
    t = table_top_points(ctx, fid, describe, log=log)
    if not t['ok']:
        return t
    rect = cv2.minAreaRect(t['points'][:, :2].astype(np.float32))
    return {'ok': True, 'top_z': t['top_z'], 'center': [float(rect[0][0]), float(rect[0][1])],
            'size': [float(rect[1][0]), float(rect[1][1])], 'angle_deg': float(rect[2]),
            'corners': cv2.boxPoints(rect).tolist(), 'n': int(len(t['points'])), 'frame': fid, 'gemini_s': t['seconds'],
            'method': 'gemini-mask'}
