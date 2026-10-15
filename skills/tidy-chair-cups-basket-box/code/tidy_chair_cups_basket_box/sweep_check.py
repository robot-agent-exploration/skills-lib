"""Swept-footprint check of planned base moves against obstacle polygons and depth points (odometry frame).

Why: 2026-10-09 a hand-planned 151 deg in-place turn swung the base's rear into the dark-top cabinet (operator:
"you crashed into the cabinet with the back of the base") - the check had looked at the front only. Every base
move near furniture goes through `check_moves` first. Uses the bimanual-box-pnp footprint (base body + wheels,
turns about the wheel axle 0.228 m ahead of the base origin)."""
import math

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from bimanual_box_pnp.core import footprint as FP

AXLE = 0.228


def poses_along(pose, moves, step_m=0.03, step_rad=0.04):
    """Odometry poses along [('turn', a) | ('drive', d)] moves from pose (x, y, yaw)."""
    x, y, th = pose
    out = [(x, y, th)]
    for kind, v in moves:
        if kind == 'drive':
            n = max(1, int(math.ceil(abs(v) / step_m)))
            for k in range(1, n + 1):
                out.append((x + v * k / n * math.cos(th), y + v * k / n * math.sin(th), th))
            x, y = x + v * math.cos(th), y + v * math.sin(th)
        else:
            ax, ay = x + AXLE * math.cos(th), y + AXLE * math.sin(th)
            n = max(1, int(math.ceil(abs(v) / step_rad)))
            for k in range(1, n + 1):
                t = th + v * k / n
                out.append((ax - AXLE * math.cos(t), ay - AXLE * math.sin(t), t))
            th = th + v
            x, y = ax - AXLE * math.cos(th), ay - AXLE * math.sin(th)
    return out


def swept(pose, moves, margin=0.05, extra=None):
    """Union of the base footprint (plus `extra` base-frame polygons, e.g. a held chair) along the moves."""
    shapes = [FP.BASE_POLY.buffer(margin)] + list(extra or [])
    polys = [FP.place(s, p) for p in poses_along(pose, moves) for s in shapes]
    return unary_union(polys)


def check_moves(pose, moves, obstacles=(), points=None, margin=0.05, extra=None, ignore=None):
    """-> {ok, min_clear (m to the nearest obstacle polygon), hits (named polygons intersected), n_points (depth
    obstacle points inside the swept area)}. obstacles: [(name, shapely polygon in odom)]; points: (N, 2) odom."""
    S = swept(pose, moves, margin=margin, extra=extra)
    hits, clear = [], 9.0
    for name, poly in obstacles:
        if ignore and name in ignore:
            continue
        d = S.distance(poly)
        clear = min(clear, d)
        if S.intersects(poly):
            hits.append(name)
    n_pts = 0
    if points is not None and len(points):
        P = np.asarray(points)[:, :2]
        minx, miny, maxx, maxy = S.bounds
        P = P[(P[:, 0] > minx) & (P[:, 0] < maxx) & (P[:, 1] > miny) & (P[:, 1] < maxy)]
        n_pts = int(sum(S.contains(Point(p)) for p in P[::2]))
    return {'ok': not hits and n_pts <= 25, 'min_clear': round(clear, 3), 'hits': hits, 'n_points': n_pts}
