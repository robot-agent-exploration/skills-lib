"""Perception for the air-fryer task from the exterior RGB-D cameras (base frame).

Fryer frame: c = vertical axis (xy), d = horizontal unit vector from the axis toward the
handle (= drawer pull direction), t = z x d. Coordinates along d / t are written sd / st.
Tested on the white fryer with a black Gamma-shaped handle and a yellow banana.
"""
import numpy as np
import cv2 as cv

from geom import backproject

from air_fry_pan_params import TABLE_Z, FRYER_REGION  # hyper-parameters: air_fry_pan_params.py


def _pts(F, cams, mask_fn, stride=1):
    out = []
    for name in cams:
        cam = F.get(name)
        if cam is None or cam['depth'] is None:
            continue
        hsv = cv.cvtColor(cam['rgb'], cv.COLOR_BGR2HSV)
        p, _ = backproject(cam, mask=mask_fn(hsv).astype(np.uint8), stride=stride)
        out.append(p)
    return np.concatenate(out) if out else np.zeros((0, 3))


def _in_xy(P, region):
    (x0, x1), (y0, y1) = region
    return (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)


def fit_circle(xy):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c[:2], float(np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 1e-9)))


def robust_circle(xy, iters=6, keep_pct=75, min_tol=0.004):
    ctr, r = fit_circle(xy)
    keep = np.ones(len(xy), bool)
    for _ in range(iters):
        res = np.abs(np.linalg.norm(xy - ctr, axis=1) - r)
        keep = res < max(min_tol, np.percentile(res, keep_pct))
        ctr, r = fit_circle(xy[keep])
    return ctr, r, keep


def locate_fryer(F, cams=('left', 'right')):
    """Fryer axis, pull direction and handle features. Returns dict or raises ValueError."""
    white = lambda h: (h[:, :, 2] > 110) & (h[:, :, 1] < 70)
    dark = lambda h: h[:, :, 2] < 70
    # seed: the dark lid on top of the fryer (z ~0.37-0.43)
    L = _pts(F, cams, dark, stride=2)
    L = L[_in_xy(L, FRYER_REGION) & (L[:, 2] > 0.37) & (L[:, 2] < 0.43)]
    if len(L) < 300:
        raise ValueError(f'fryer lid not found ({len(L)} dark points at z 0.37-0.43)')
    seed = np.median(L[:, :2], axis=0)
    W = _pts(F, cams, white, stride=2)
    W = W[_in_xy(W, FRYER_REGION) & (W[:, 2] > 0.30) & (W[:, 2] < 0.40)
          & (np.linalg.norm(W[:, :2] - seed, axis=1) < 0.17)]
    if len(W) < 500:
        raise ValueError(f'fryer body not found ({len(W)} white points at z 0.30-0.40)')
    # Upper body is a cylinder; fit its visible side (white points at mid-height)
    c, r_body, keep = robust_circle(W[:, :2])
    if not (0.08 < r_body < 0.16):
        raise ValueError(f'fryer body radius {r_body:.3f} implausible')
    D = _pts(F, cams, dark)
    D = D[_in_xy(D, FRYER_REGION)]
    rel = D[:, :2] - c
    r = np.linalg.norm(rel, axis=1)
    grip = D[(D[:, 2] > 0.14) & (D[:, 2] < 0.22) & (r > r_body + 0.02) & (r < r_body + 0.12)]
    if len(grip) < 200:
        raise ValueError(f'handle grip not found ({len(grip)} dark points)')
    # drop outliers: keep the largest cluster around the median direction
    ang = np.arctan2(grip[:, 1] - c[1], grip[:, 0] - c[0])
    a0 = np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))
    grip = grip[np.abs(np.angle(np.exp(1j * (ang - a0)))) < 0.35]
    gxy = np.median(grip[:, :2], axis=0)
    d = (gxy - c) / np.linalg.norm(gxy - c)
    t = np.array([-d[1], d[0]])
    H = D[(D[:, 2] > 0.10) & (D[:, 2] < 0.34)]
    rel = H[:, :2] - c
    sd, st = rel @ d, rel @ t
    near = np.abs(st) < 0.022
    sd, st, z = sd[near], st[near], H[near, 2]
    face = {}
    for z0 in (0.14, 0.17, 0.20, 0.22, 0.24):
        m = (z >= z0) & (z < z0 + 0.015) & (sd > r_body + 0.02)
        if m.sum() > 15:
            face[z0] = float(np.percentile(sd[m], 97))
    if 0.22 not in face or 0.14 not in face:
        raise ValueError(f'handle outer face not measured: {face}')
    arm = (sd > face[0.22] - 0.045) & (sd < face[0.22] - 0.005) & (z > 0.22)
    arm_top = float(np.percentile(z[arm], 97)) if arm.sum() > 20 else None
    gm = (z > 0.14) & (z < 0.22) & (sd > r_body + 0.02)
    grip_t = float(np.median(st[gm])) if gm.sum() > 20 else 0.0
    # upper-body front surface next to the handle (white, z 0.27-0.38, beside the handle insert)
    Wb = _pts(F, cams, white, stride=1)
    rel = Wb[:, :2] - c
    bsd, bst = rel @ d, rel @ t
    m = (Wb[:, 2] > 0.27) & (Wb[:, 2] < 0.38) & (np.abs(bst) > 0.02) & (np.abs(bst) < 0.04) & (bsd > 0) & (bsd < 0.2)
    body_front = float(np.percentile(bsd[m], 97) + 0.004) if m.sum() > 30 else None   # +4 mm: |t|~3 cm curvature
    return dict(c=c, d=d, t=t, r_body=r_body, face=face, arm_top=arm_top, grip_t=grip_t,
                body_front=body_front, protrusion=(face[0.22] - body_front) if body_front else None,
                n_white=int(len(W)), n_grip=int(len(grip)))


def handle_state(F, c, d, cams=('left', 'right'), expect_sd=None, top_cams=None, face_ref=None):
    """Re-measure handle face/arm-top along a known axis (e.g. after pulling the drawer).
    Faces come from `cams` (side views); the arm top also uses `top_cams` (views from above)."""
    c = np.asarray(c); d = np.asarray(d); t = np.array([-d[1], d[0]])
    D = _pts(F, cams, lambda h: h[:, :, 2] < 70)
    D = D[(D[:, 2] > 0.09) & (D[:, 2] < 0.34)]
    rel = D[:, :2] - c
    sd, st, z = rel @ d, rel @ t, D[:, 2]
    near = np.abs(st) < 0.03
    if expect_sd is not None:
        near &= (sd > expect_sd - 0.08) & (sd < expect_sd + 0.05)
    sd, st, z = sd[near], st[near], z[near]
    face = {}
    for z0 in (0.12, 0.14, 0.17, 0.20, 0.22, 0.24):
        m = (z >= z0) & (z < z0 + 0.015)
        if m.sum() > 15:
            face[z0] = float(np.percentile(sd[m], 97))
    ref = face_ref if face_ref is not None else (face.get(0.22) or face.get(0.20))
    arm_top, top_t = None, None
    if ref is not None:
        if top_cams:
            Dt = _pts(F, top_cams, lambda h: h[:, :, 2] < 70)
            rel_t = Dt[:, :2] - c
            sdt, stt, zt = rel_t @ d, rel_t @ t, Dt[:, 2]
            arm = (np.abs(stt) < 0.03) & (sdt > ref - 0.045) & (sdt < ref - 0.005) & (zt > 0.19) & (zt < 0.31)
            if arm.sum() > 50:
                arm_top = float(np.percentile(zt[arm], 95))
                top_band = arm & (zt > arm_top - 0.012)
                if top_band.sum() > 50:
                    top_t = float(np.median(stt[top_band]))
        if arm_top is None:
            arm = (sd > ref - 0.045) & (sd < ref - 0.005) & (z > 0.19)
            if arm.sum() > 20:
                arm_top = float(np.percentile(z[arm], 97))
    gm = (z > 0.12) & (z < 0.22)
    grip_t = top_t if top_t is not None else (float(np.median(st[gm])) if gm.sum() > 20 else None)
    return dict(face=face, arm_top=arm_top, grip_t=grip_t, side_t=float(np.median(st[gm])) if gm.sum() > 20 else None,
                n=int(len(sd)))


def locate_banana(F, cams=('left', 'right'), exclude=None):
    """Banana on the table: centroid, axis, convex-side normal, arc middle, top z, chord, sagitta."""
    yellow = lambda h: (h[:, :, 0] > 18) & (h[:, :, 0] < 40) & (h[:, :, 1] > 100) & (h[:, :, 2] > 80)
    P = _pts(F, cams, yellow)
    P = P[(P[:, 2] > TABLE_Z + 0.003) & (P[:, 2] < TABLE_Z + 0.09)]
    if exclude is not None:   # (centre_xy, radius): ignore points near/inside the fryer
        P = P[np.linalg.norm(P[:, :2] - exclude[0], axis=1) > exclude[1]]
    if len(P) < 300:
        raise ValueError(f'banana not found ({len(P)} yellow points)')
    xy = P[:, :2]
    # keep the main cluster
    med = np.median(xy, axis=0)
    xy_ok = np.linalg.norm(xy - med, axis=1) < 0.15
    P, xy = P[xy_ok], xy[xy_ok]
    mu = xy.mean(0)
    _, _, Vt = np.linalg.svd(xy - mu, full_matrices=False)
    ax, nx = Vt[0], Vt[1]
    s, n = (xy - mu) @ ax, (xy - mu) @ nx
    a, b, c0 = np.linalg.lstsq(np.c_[s ** 2, s, np.ones_like(s)], n, rcond=None)[0]
    s0, s1 = np.percentile(s, [1.5, 98.5])
    smid = 0.5 * (s0 + s1)
    # centreline middle; ends curve toward sign(a)*nx, so convex side is -sign(a)*nx
    mid = mu + ax * smid + nx * (a * smid ** 2 + b * smid + c0)
    ends = [mu + ax * sv + nx * (a * sv ** 2 + b * sv + c0) for sv in (s0, s1)]
    chord = float(np.linalg.norm(ends[1] - ends[0]))
    sag = float(np.dot(mid - 0.5 * (ends[0] + ends[1]), -np.sign(a) * nx)) if a != 0 else 0.0
    n_convex = -np.sign(a) * nx if a != 0 else nx
    near_mid = np.abs(s - smid) < 0.02
    top = float(np.percentile(P[near_mid, 2], 95)) if near_mid.sum() > 20 else float(np.percentile(P[:, 2], 95))
    return dict(centroid=mu, axis=ax, n_convex=n_convex, mid=mid, ends=ends, chord=chord,
                sagitta=sag, top_z=top, n=int(len(P)))


def locate_rim(F, c, d, expect_sd, cams=('left', 'right'), radius_guess=0.10):
    """Basket rim circle (dark points z 0.20-0.25) near c + expect_sd*d. Returns (centre, r, n)."""
    c = np.asarray(c); d = np.asarray(d)
    P = _pts(F, cams, lambda h: h[:, :, 2] < 60)
    guess = c + expect_sd * d
    P = P[(P[:, 2] > 0.20) & (P[:, 2] < 0.25) & (np.linalg.norm(P[:, :2] - guess, axis=1) < radius_guess + 0.04)]
    if len(P) < 300:
        return None, None, len(P)
    ctr, r, keep = robust_circle(P[:, :2])
    return ctr, r, int(keep.sum())


def banana_inside(F, rim_ctr, rim_r, rim_z=0.235, cams=('left', 'right', 'wrist'), height_cam='wrist'):
    """Yellow points within the basket circle and below the rim. Heights come from the top-down
    `height_cam` when it sees the banana (side views get rim depth on the banana edge pixels)."""
    yellow = lambda h: (h[:, :, 0] > 18) & (h[:, :, 0] < 40) & (h[:, :, 1] > 100) & (h[:, :, 2] > 80)
    P = _pts(F, cams, yellow)
    P = P[(P[:, 2] > TABLE_Z + 0.003) & (P[:, 2] < 0.45)]
    if len(P) == 0:
        return dict(n=0, inside=0, frac=0.0, zmax=None)
    r = np.linalg.norm(P[:, :2] - rim_ctr, axis=1)
    near = r < rim_r + 0.01
    Ph = _pts(F, (height_cam,), yellow) if height_cam in F else np.zeros((0, 3))
    Ph = Ph[(Ph[:, 2] > TABLE_Z + 0.003) & (Ph[:, 2] < 0.45)] if len(Ph) else Ph
    if len(Ph) > 200:
        zsrc = Ph[np.linalg.norm(Ph[:, :2] - rim_ctr, axis=1) < rim_r + 0.01, 2]
    else:
        zsrc = P[near, 2]
    zmax = float(np.percentile(zsrc, 98)) if len(zsrc) > 10 else None
    return dict(n=int(len(P)), inside=int(near.sum()), frac=float(near.mean()), zmax=zmax,
                n_height=int(len(zsrc)))


def locate_fryer_top(F, rough_c, cams=('wrist',)):
    """Top-view refinement (wrist camera ~0.3 m above the lid): lid-circle centre = fryer axis,
    pull direction from the axis to the handle arm, handle arm top z. Returns dict or raises."""
    rough_c = np.asarray(rough_c, float)
    D = _pts(F, cams, lambda h: h[:, :, 2] < 60)
    lid = D[(D[:, 2] > 0.38) & (D[:, 2] < 0.44) & (np.linalg.norm(D[:, :2] - rough_c, axis=1) < 0.12)]
    if len(lid) < 2000:
        raise ValueError(f'lid not seen from above ({len(lid)} points)')
    ctr, r, keep = robust_circle(lid[:, :2], keep_pct=90)
    if not (0.045 < r < 0.08):
        raise ValueError(f'lid circle radius {r:.3f} implausible')
    rel = D[:, :2] - ctr
    rr = np.linalg.norm(rel, axis=1)
    arm = D[(D[:, 2] > 0.21) & (D[:, 2] < 0.31) & (rr > 0.115) & (rr < 0.20)]
    if len(arm) < 300:
        raise ValueError(f'handle arm not seen from above ({len(arm)} points)')
    ang = np.arctan2(arm[:, 1] - ctr[1], arm[:, 0] - ctr[0])
    a0 = np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))
    arm = arm[np.abs(np.angle(np.exp(1j * (ang - a0)))) < 0.25]
    axy = np.median(arm[:, :2], axis=0)
    d = (axy - ctr) / np.linalg.norm(axy - ctr)
    return dict(c_lid=ctr, r_lid=r, d=d, arm_top=float(np.percentile(arm[:, 2], 95)), n_arm=int(len(arm)))


def locate_ball(F, cams=('left',), exclude=None, zmax=TABLE_Z + 0.09):
    """Orange ball: centre (x, y), radius, top z (zmax ~0.30 to find it inside the basket)."""
    orange = lambda h: (h[:, :, 0] < 18) & (h[:, :, 1] > 140) & (h[:, :, 2] > 120)
    P = _pts(F, cams, orange)
    P = P[(P[:, 2] > TABLE_Z + 0.003) & (P[:, 2] < zmax)]
    if exclude is not None:
        P = P[np.linalg.norm(P[:, :2] - exclude[0], axis=1) > exclude[1]]
    if len(P) < 150:
        raise ValueError(f'ball not found ({len(P)} orange points)')
    med = np.median(P[:, :2], axis=0)
    P = P[np.linalg.norm(P[:, :2] - med, axis=1) < 0.06]
    top = float(np.percentile(P[:, 2], 97))
    r = max(0.015, (top - TABLE_Z) / 2)
    # visible surface is biased toward the camera: centre = surface points shifted by ~r/2 away is
    # unreliable, so use the median of points near the top (seen from any side)
    if zmax > TABLE_Z + 0.09:        # in the basket: size from extent, not height above table
        r = max(0.015, 0.25 * (np.ptp(P[:, 0]) + np.ptp(P[:, 1])))
    near_top = P[P[:, 2] > top - 0.6 * r]
    ctr = np.median(near_top[:, :2], axis=0) if len(near_top) > 30 else med
    return dict(center=ctr, radius=r, top_z=top, n=int(len(P)))
