"""Base routes for a differential-drive RB-Y1 (turns pivot about the wheel axle at base x 0.228): turn-drive-turn
plans, the base footprint swept along them (library footprint module), clearance to known obstacles, and the depth
points of a frame inside the swept footprint (what the camera sees in the way).

All planning is in the ODOMETRY frame (x, y, yaw of the base origin); detections are converted with geom."""
import itertools

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from bimanual_box_pnp.core import footprint, geom
from .lib import frames

AXLE_X = 0.228


def tdt(pose, target, heading):
    """Moves taking the base origin from odom pose (x, y, yaw) to odom point `target` facing odom `heading`:
    turn towards the goal's axle point, drive straight, turn to the heading."""
    x, y, th = pose
    a0 = np.array([x, y]) + AXLE_X * np.array([np.cos(th), np.sin(th)])
    a1 = np.asarray(target, float) + AXLE_X * np.array([np.cos(heading), np.sin(heading)])
    d = a1 - a0
    dist = float(np.linalg.norm(d))
    if dist < 0.02:
        t = geom.wrap(heading - th)
        return [{'turn': t}] if abs(t) > 0.01 else []
    bearing = float(np.arctan2(d[1], d[0]))
    moves = []
    t1 = geom.wrap(bearing - th)
    if abs(t1) > 0.01:
        moves.append({'turn': t1})
    moves.append({'forward': dist})
    t2 = geom.wrap(heading - bearing)
    if abs(t2) > 0.01:
        moves.append({'turn': t2})
    return moves


def end_pose(pose, moves):
    p = tuple(pose)
    for m in moves:
        p = footprint.sample_move(p, m)[-1]
    return p


def swept(pose, moves, poly=None, margin=0.0):
    """Odom polygon swept by the base footprint (or `poly`, base frame) along the moves."""
    poly = (poly if poly is not None else footprint.BASE_POLY).buffer(margin)
    parts, p = [footprint.place(poly, pose)], tuple(pose)
    for m in moves:
        poses = footprint.sample_move(p, m)
        parts += [footprint.place(poly, q) for q in poses[::2] + [poses[-1]]]
        p = poses[-1]
    return unary_union(parts)


def clearance(pose, moves, obstacles, poly=None):
    """{name: smallest distance (m) between the swept footprint and each obstacle polygon (odom)}."""
    g = swept(pose, moves, poly)
    return {k: float(g.distance(o)) for k, o in obstacles.items()}


def circle(c, r):
    return Point(float(c[0]), float(c[1])).buffer(r, 64)


def path_length(moves):
    return sum(abs(m.get('forward', 0.0)) for m in moves), sum(abs(m.get('turn', 0.0)) for m in moves)


def plan_route(pose, goal, heading, obstacles, min_clear=0.12, final_straight=0.5, grid=None, target=None,
               target_clear=0.08):
    """Route ending with a straight `final_straight` m forward leg into (goal, heading): direct turn-drive-turn to
    the pre-goal, else via one intermediate waypoint from `grid` (odom points). Cost: metres + 0.3 m per rad of
    turning. Every move keeps min_clear from all obstacles, except that the final leg may come target_clear close
    to the obstacle named `target` (the object the stance is for). -> {moves, clearance, cost} or None."""
    hv = np.array([np.cos(heading), np.sin(heading)])
    pre = np.asarray(goal, float) - hv * final_straight
    pre_pose = (float(pre[0]), float(pre[1]), float(heading))
    last = [{'forward': float(final_straight)}] if final_straight > 0 else []
    cl_last = clearance(pre_pose, last, obstacles)
    if any(v < (target_clear if k == target else min_clear) for k, v in cl_last.items()):
        return None
    best = None

    def consider(moves):
        nonlocal best
        cl = clearance(pose, moves, obstacles)
        if min(cl.values()) < min_clear:
            return
        f, t = path_length(moves + last)
        cost = f + 0.3 * t
        if best is None or cost < best['cost']:
            best = {'moves': moves + last, 'clearance': cl, 'final_clearance': cl_last, 'cost': cost}

    consider(tdt(pose, pre, heading))
    for w in (grid or []):
        m1 = tdt(pose, w, float(np.arctan2(pre[1] - w[1], pre[0] - w[0])))
        p1 = end_pose(pose, m1)
        consider(m1 + tdt(p1, pre, heading))
    return best


def ring_grid(center, radii=(0.8, 1.1, 1.4), n=16):
    c = np.asarray(center, float)
    return [c + r * np.array([np.cos(a), np.sin(a)]) for r, a in itertools.product(radii, np.linspace(0, 2 * np.pi, n, endpoint=False))]


def depth_hits(fid, frame_odom, pose, moves, ignore=(), zmin=0.12, zmax=1.8, margin=0.05, border=20):
    """Depth points of frame `fid` (taken at odometry `frame_odom`) inside the base footprint swept along `moves`
    from odom `pose`, above zmin (floor) and outside the robot itself and the `ignore` polygons (odom).
    -> (count, pixels of the hits)"""
    pts, pix, _, meta = frames.load(fid, stride=3, max_depth=4.0, drop_edges=True)
    w, h = meta['cameras']['head_camera']['width'], meta['cameras']['head_camera']['height']
    inner = (pix[:, 0] >= border) & (pix[:, 0] < w - border) & (pix[:, 1] >= border) & (pix[:, 1] < h - border)
    pts, pix = pts[inner], pix[inner]
    keep = ~footprint.self_mask(pts, meta['joints'])
    pts, pix = pts[keep], pix[keep]
    xy = np.array([geom.base_to_odom(frame_odom, p[:2])[:2] for p in pts]) if len(pts) else np.zeros((0, 2))
    g = swept(pose, moves, margin=margin)
    from matplotlib.path import Path as MPath
    inside = np.zeros(len(xy), bool)
    for part in (getattr(g, 'geoms', None) or [g]):
        inside |= MPath(np.array(part.exterior.coords)).contains_points(xy)
    for ig in ignore:
        for part in (getattr(ig, 'geoms', None) or [ig]):
            inside &= ~MPath(np.array(part.exterior.coords)).contains_points(xy)
    hit = inside & (pts[:, 2] > zmin) & (pts[:, 2] < zmax)
    return int(hit.sum()), pix[hit]
