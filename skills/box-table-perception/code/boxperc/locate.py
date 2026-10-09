"""Gemini-seeded localization of a box and a table top in one head-camera frame.

Gemini says WHICH object it is, whether it is fully visible and WHERE it is in the image (pixels); depth says
how far (metres): the top-face rectangle is fitted from depth points around the pointed pixel. This replaces
v1's cardboard-colour gate, the partial-view size rule (its failure caused crash A2) and the odometry table
prior (drifted 0.45-0.56 m). The depth-only detectors in box.py / table.py remain the fallback and the
geometric cross-check (vlm_corner_err_m: Gemini's corners vs the fitted rectangle)."""
import numpy as np

from rby1rt import frames as pf
from rby1rt import vlm

from . import box as pbox
from . import table as ptable

BOX_DESC = 'the closed cardboard box'
TABLE_DESC = 'the free-standing white drawer cabinet with a black top'
MAX_CORNER_ERR = 0.08       # m: Gemini corners vs depth rectangle beyond this -> not trusted (checkpoint)


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
