"""Cardboard box on a support surface (floor or table top) from one frame: depth clusters, top-face rectangle,
cardboard colour test and size gates (library skill bimanual-box-squeeze perception.find_box, extended).

Tested: 0.40 x 0.22 x 0.26 m carton; depth gives 0.387 x 0.213, top 0.263 on the floor, from 0.45 to 4 m.
The depth centre came out ~3 cm right (-y) of the centre found by touch at the grasp stance (2 grasps)."""
import cv2
import numpy as np

from ..core.frames import clusters, frame_dir, load

from ..params import (  # hyper-parameters: boxpnp/params.py
    FLOOR_Z_MIN, OBJECT_Z_MAX, GRID, BOX_LONG, BOX_SHORT, BOX_TOP_Z, CARDBOARD_H, CARDBOARD_S,
    MIN_CARDBOARD_FRAC,
)
SELF_X, SELF_Y = 0.36, 0.32       # the robot's own front / knee


def find_box(fdir, near=None, radius=1.0, top_band=0.025, max_range=4.0, exclude_self=True, support_z=0.0,
             top_z_hint=None, require_cardboard=True):
    """-> {ok, center (top face), top_z, size [long, short], long_axis (+y), short_axis (+x, away from the
    robot), yaw_short, ends, near_face_x, cardboard, frame, candidates}

    top_z_hint: the box-top height when known (e.g. from a Gemini-pointed pixel's depth): the height band is set
    around it instead of from support_z. require_cardboard=False skips the colour gate (the caller identified the
    box by other means, e.g. Gemini)."""
    pts, pix, rgb, meta = load(fdir, max_depth=max_range + 0.5)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    # only the height band where a box top can be: floor noise (it bridged the box to a cabinet front) and
    # low parts of other objects are left out; vertical faces of tall things become thin slivers
    zlo, zhi = ((top_z_hint - 0.06, top_z_hint + 0.03) if top_z_hint is not None else
                (support_z + BOX_TOP_Z[0] - 0.03, support_z + BOX_TOP_Z[1] + 0.02))
    m = (pts[:, 2] > zlo) & (pts[:, 2] < zhi) & \
        (pts[:, 0] > 0.2) & (pts[:, 0] < max_range) & (np.abs(pts[:, 1]) < 2.5)
    if exclude_self:
        m &= ~((pts[:, 0] < SELF_X) & (np.abs(pts[:, 1]) < SELF_Y))
    if near is not None:
        m &= np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1]) < radius
    P, U = pts[m], pix[m]
    if len(P) < 50:
        return {'ok': False, 'why': f'only {len(P)} object points in the search window', 'candidates': []}
    n, lab = clusters(P, GRID)
    cands = []
    for k in range(1, n):
        sel = lab == k
        if sel.sum() < 40:
            continue
        Q, UQ = P[sel], U[sel]
        top = np.percentile(Q[:, 2], 95)
        bsel = np.abs(Q[:, 2] - top) < top_band
        band, uband = Q[bsel], UQ[bsel]
        if len(band) < 25:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(band[:, :2].astype(np.float32))
        a = np.deg2rad(ang)
        ax_w, ax_h = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        la = ax_w if w >= h else ax_h
        la = -la if la[1] < 0 else la
        sa = np.array([la[1], -la[0]])
        sa = -sa if sa[0] < 0 else sa
        rel = band[:, :2] - [cx, cy]
        lo_l, hi_l = np.percentile(rel @ la, [1, 99])
        lo_s, hi_s = np.percentile(rel @ sa, [1, 99])
        c = np.array([cx, cy]) + la * (lo_l + hi_l) / 2 + sa * (lo_s + hi_s) / 2
        L, S = hi_l - lo_l, hi_s - lo_s
        hs = hsv[uband[:, 1], uband[:, 0]]
        card = (hs[:, 0] >= CARDBOARD_H[0]) & (hs[:, 0] <= CARDBOARD_H[1]) & (hs[:, 1] >= CARDBOARD_S[0]) & \
               (hs[:, 1] <= CARDBOARD_S[1])
        top_z = float(np.median(band[:, 2]))
        cand = {'center': c.tolist(), 'top_z': top_z, 'support_z': support_z, 'size': [float(L), float(S)],
                'long_axis': la.tolist(), 'short_axis': sa.tolist(), 'yaw_short': float(np.arctan2(sa[1], sa[0])),
                'ends': [(c - la * L / 2).tolist(), (c + la * L / 2).tolist()],
                'near_face_x': float((c - sa * S / 2)[0]), 'n_top': int(len(band)),
                'cardboard': float(card.mean()), 'frame': frame_dir(fdir).name}
        cand['fits'] = bool(BOX_LONG[0] <= L <= BOX_LONG[1] and BOX_SHORT[0] <= S <= BOX_SHORT[1]
                            and BOX_TOP_Z[0] <= top_z - support_z <= BOX_TOP_Z[1]
                            and (cand['cardboard'] >= MIN_CARDBOARD_FRAC or not require_cardboard))
        cands.append(cand)
    good = [c for c in cands if c['fits']]
    if not good:
        return {'ok': False, 'why': 'no cardboard box-sized object', 'candidates': cands}
    best = max(good, key=lambda c: c['n_top'])
    return {'ok': True, **best, 'candidates': cands}
