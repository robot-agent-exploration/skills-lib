"""Centre a held peg in a loose hole by pushing it to the walls (robot-frame, camera-free).

The midpoint of opposite wall contacts is the TCP xy that centres the peg in that hole at the
current gripper yaw; adding the exact CAD offset between two holes transfers it to another hole.
"""
import time

import cv2
import numpy as np

from motion import move, status

from peg_insertion_params import PUSH, BLOCKED  # hyper-parameters: peg_insertion_params.py


def board_patch():
    """Grey board-edge patch from the newest left-camera frame (for board-motion checks)."""
    from api import stream
    t, p = stream.frames('left', seconds=2)[-1]
    g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return g[470:600, 440:640]


def board_shift(a, b):
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(np.hypot(dx, dy))


def push_to_wall(ctx, centre, z, d, quat, log=print, max_travel=0.004, f_contact=1.5):
    """Push from centre along unit d until the force along d rises f_contact (N) above a fresh
    baseline (a short move alone is not a wall: the arm lags ~0.5 mm). Returns wall xy or None."""
    c = np.asarray(centre, float)
    move(ctx, [c[0], c[1], z], quat, tol_mm=0.6, timeout_s=1.5)
    time.sleep(0.2)
    f0 = np.mean([status(ctx)['tcp_wrench'][:2] for _ in range(3)], axis=0)
    pos = c.copy()
    last_short = False
    for k in range(8):
        target = pos + PUSH * d
        r = move(ctx, [target[0], target[1], z], quat, tol_mm=0.5, timeout_s=1.0)
        m = np.array(r['measured_position'][:2])
        f = np.mean([status(ctx)['tcp_wrench'][:2] for _ in range(2)], axis=0)
        df = float((f - f0) @ d)
        travel = float((m - c) @ d)
        log(f'      push {k}: moved {(m - pos) @ d * 1000:.2f} mm, travel {travel*1000:.2f} mm, dF {df:+.2f} N')
        short = float((m - pos) @ d) < 0.0003
        pos = m
        if df > f_contact or (short and k > 0 and last_short):
            move(ctx, [c[0], c[1], z], quat, tol_mm=0.6, timeout_s=1.5)
            return m
        last_short = short
        if travel > max_travel:
            break
    move(ctx, [c[0], c[1], z], quat, tol_mm=0.6, timeout_s=1.5)
    return None


def centre_in_hole(ctx, centre, z, yaw, quat, log=print):
    """Probe +-a and +-b (gripper axes at yaw); returns dict(ok, centre, spans)."""
    ax = np.array([np.cos(yaw), np.sin(yaw)])
    ay = np.array([np.sin(yaw), -np.cos(yaw)])
    c = np.asarray(centre, float)
    spans = {}
    for name, d in (('a', ax), ('b', ay)):
        w1 = push_to_wall(ctx, c, z, d, quat, log)
        w2 = push_to_wall(ctx, c, z, -d, quat, log)
        if w1 is None or w2 is None:
            return {'ok': False, 'reason': f'no wall along {name}', 'centre': c.tolist()}
        mid = (w1 + w2) / 2
        spans[name] = float((w1 - w2) @ d)
        shift = float((mid - c) @ d)
        c = c + shift * d
        log(f'    axis {name}: span {spans[name]*1000:.2f} mm, centre shift {shift*1000:+.2f} mm')
        move(ctx, [c[0], c[1], z], quat, tol_mm=0.6, timeout_s=1.5)
    return {'ok': True, 'centre': c.tolist(), 'spans_mm': {k: v * 1000 for k, v in spans.items()}}


def centre_triangle(ctx, start, z, hole_yaw, quat, log=print):
    """Centre a triangular peg in an aligned loose triangular hole: push outward along the hole's
    three face normals (yaw + 0, +-120 deg in the CAD outline convention). With travels s_i,
    offset of peg from hole centre = (2/3) * sum((mean(s) - s_i) n_i) (no clearance value needed)."""
    c = np.asarray(start, float)
    normals = [np.array([np.cos(hole_yaw + k), np.sin(hole_yaw + k)]) for k in (0, 2 * np.pi / 3, -2 * np.pi / 3)]
    s = []
    for n in normals:
        w = push_to_wall(ctx, c, z, n, quat, log)
        if w is None:
            return {'ok': False, 'reason': 'no wall', 'centre': c.tolist()}
        s.append(float((w - c) @ n))
    s = np.array(s)
    off = (2.0 / 3.0) * sum((s.mean() - si) * n for si, n in zip(s, normals))
    centre = c - off
    log(f'    triangle travels {np.round(s * 1000, 2).tolist()} mm -> offset {np.round(off * 1000, 2).tolist()} mm')
    move(ctx, [centre[0], centre[1], z], quat, tol_mm=0.6, timeout_s=1.5)
    return {'ok': True, 'centre': centre.tolist(), 'travels_mm': (s * 1000).tolist()}


def gripper_matrix(psi):
    """Columns: base-frame directions of the gripper x (fingers) and y (closing) axes."""
    return np.array([[np.cos(psi), np.sin(psi)], [np.sin(psi), -np.cos(psi)]])


def solve_two_yaw(t1, psi1, t2, psi2):
    """Centred TCPs t1, t2 at gripper yaws psi1, psi2 in the same hole -> (in-hand offset c (a, b),
    hole centre h). Uses t + G(psi) c = h for both."""
    G1, G2 = gripper_matrix(psi1), gripper_matrix(psi2)
    c = np.linalg.solve(G2 - G1, np.asarray(t1) - np.asarray(t2))
    h = np.asarray(t1) + G1 @ c
    return c, h
