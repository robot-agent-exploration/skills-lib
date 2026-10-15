"""Visual location of a dark opening (the urn's mouth) with the wrist camera.

The wrist camera's absolute positions carry a view-dependent calibration error, so the opening
is viewed from two opposite sides (camera on +y and on -y of it) and the two estimates are
averaged: errors from the camera's tilt/offset calibration flip sign between the views.
"""
import math

import cv2
import numpy as np

from .perception import Frame, view_pose_for

from .params import TILT, VIEW_HEIGHT, MIN_SCORE, VIEW_YAWS  # hyper-parameters: params.py


def ring_kernel(r0, r1, size):
    yy, xx = np.mgrid[-size:size + 1, -size:size + 1]
    rr = np.hypot(xx, yy)
    k = ((rr >= r0) & (rr < r1)).astype(np.float32)
    return k / k.sum()


def opening_center(frame, xy, z_rim, r_in=0.010, r_out=0.013, search_px=60):
    """Centre (px) of a dark disc of radius r_in inside a brighter rim of radius r_out (both m, at
    height z_rim) near the projection of xy: matched filter score = mean(rim) - mean(disc), so it
    does not depend on how dark the opening looks from this side. Returns (u, v, score)."""
    u0, v0 = frame.project((xy[0], xy[1], z_rim))
    rin = np.mean([np.hypot(*(frame.project((xy[0] + r_in * math.cos(a), xy[1] + r_in * math.sin(a), z_rim)) - (u0, v0)))
                   for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)])
    rout = rin * r_out / r_in
    size = int(rout) + 2
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    disc = cv2.filter2D(gray, -1, ring_kernel(0, 0.85 * rin, size))
    rim = cv2.filter2D(gray, -1, ring_kernel(1.08 * rin, 0.97 * rout, size))
    score = rim - disc
    x0, y0 = int(u0) - search_px, int(v0) - search_px
    win = score[max(0, y0):y0 + 2 * search_px + 1, max(0, x0):x0 + 2 * search_px + 1]
    j, i = np.unravel_index(np.argmax(win), win.shape)
    return float(max(0, x0) + i), float(max(0, y0) + j), float(win[j, i]), float(rin)


def locate_opening(ctx, m, xy, z_rim, r_in=0.010, r_out=0.013, log=print):
    """Measured xy of the opening near xy (rim top at z_rim) from two opposite wrist views.
    Moves at >= 0.30 between views and only descends to the view height (camera VIEW_HEIGHT above
    the rim) at each view. Returns dict(xy, views, spread); raises if a view does not find it."""
    views = []
    travel_z = max(0.30, z_rim + VIEW_HEIGHT - 0.08 + 0.03)
    for k, yaw in enumerate(VIEW_YAWS):
        pose = view_pose_for(xy, yaw=yaw, cam_z=z_rim + VIEW_HEIGHT, back=VIEW_HEIGHT * math.tan(TILT))
        m.transit(pose[0], pose[1], yaw, z=travel_z)
        if k == 1:
            continue                        # intermediate orientation only
        m.move(pose, yaw=yaw, tolerance_mm=0.5)
        fid = ctx.value('franka_observe', cameras=['wrist'])['frame_id']
        f = Frame(fid)
        u, v, score, rin = opening_center(f, xy, z_rim, r_in=r_in, r_out=r_out)
        if score < MIN_SCORE:
            raise RuntimeError(f'opening not seen from yaw {math.degrees(yaw):.0f} (score {score:.0f}, frame {fid})')
        q = f.to_plane((u, v), z_rim)
        views.append(dict(yaw=yaw, frame=fid, px=[u, v], score=score, xy=[float(q[0]), float(q[1])]))
        log(f'opening from yaw {math.degrees(yaw):+.0f}: ({q[0]:.4f},{q[1]:.4f}) score {score:.0f} r {rin:.0f} px  frame {fid}')
    m.lift(z=travel_z)
    p, _ = m.tcp()
    m.move([p[0], p[1], travel_z], yaw=0.0)     # end at yaw 0: a following +90 grasp from -90 is a 180 deg turn (J7)
    a, b = np.array(views[0]['xy']), np.array(views[1]['xy'])
    mid = (a + b) / 2
    return dict(xy=[float(mid[0]), float(mid[1])], views=views, spread=float(np.linalg.norm(a - b)))
