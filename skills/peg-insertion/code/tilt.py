"""Tilted-peg (vertex-first) insertion geometry for chamferless tight holes.

Gripper frame: x along the fingers, y = closing direction, z along the fingers away from the hand
(base -z when vertical). Tool orientation for vertical yaw psi: quat [cos(psi/2), sin(psi/2), 0, 0].
The held peg's bottom vertex k sits at gripper-frame point g_k = (c + R(rel) v_k, L_b).
Tilting leans the peg top towards the lead direction u so vertex u becomes the lowest point; the
TCP is computed so that the chosen vertex stays at a commanded world point (pivot about the vertex).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from board import MODEL, rot2


def tool_rot(psi, u=None, theta=0.0):
    """Tool orientation: vertical at yaw psi, then tilted by theta about z x u (top leans to +u)."""
    R = Rotation.from_quat([np.cos(psi / 2), np.sin(psi / 2), 0.0, 0.0])
    if theta and u is not None:
        k = np.cross([0.0, 0.0, 1.0], [u[0], u[1], 0.0])
        k /= np.linalg.norm(k)
        R = Rotation.from_rotvec(theta * k) * R
    return R


def peg_vertices(shape):
    """Bottom-face corner points (peg local frame, metres): farthest outline point per sector."""
    o = np.array(MODEL['pegs'][shape]['outline'])
    n = {'triangle': 3, 'hexagon': 6, 'rectangle': 4}[shape]
    ang = np.arctan2(o[:, 1], o[:, 0])
    r = np.linalg.norm(o, axis=1)
    # the n largest-radius points pairwise separated by > 0.6 * 360/n degrees
    order = np.argsort(-r)
    chosen = []
    for j in order:
        if all(abs((ang[j] - ang[i] + np.pi) % (2 * np.pi) - np.pi) > np.radians(360 / n * 0.6) for i in chosen):
            chosen.append(j)
        if len(chosen) == n:
            break
    return o[sorted(chosen, key=lambda j: ang[j])]


def vertex_gripper_point(shape, k, c, rel, L_b):
    """Gripper-frame 3-D point of peg vertex k (index into peg_vertices)."""
    v = peg_vertices(shape)[k]
    # gripper frame is mirrored w.r.t. base yaw (y_g at psi - 90 deg): world angle psi+rel+phi
    # maps to gripper-frame angle -(rel+phi)
    xy = np.asarray(c, float) + rot2(-rel) @ (v * np.array([1.0, -1.0]))
    return np.array([xy[0], xy[1], L_b])


def vertex_world(tcp, R, g):
    return np.asarray(tcp, float) + R.apply(g)


def tcp_for_vertex(Q, R, g):
    """TCP position that puts gripper-frame point g at world point Q under orientation R."""
    return np.asarray(Q, float) - R.apply(g)


def quat_xyzw(R):
    return [float(v) for v in R.as_quat()]


def place_vertex(ctx, move, Qt, R, g, tries=3, tol=0.00015, max_extra=0.001, timeout_s=1.2):
    """Closed-loop: re-command the TCP until the measured vertex is within tol of Qt (3-D).
    Corrections are capped at max_extra beyond Qt so a blocked vertex cannot build large force.
    Returns (measured vertex, error vector, last reply)."""
    Qt = np.asarray(Qt, float)
    cmd = Qt.copy()
    q = quat_xyzw(R)
    for _ in range(tries):
        r = move(ctx, tcp_for_vertex(cmd, R, g), q, tol_mm=0.5, timeout_s=timeout_s)
        v = vertex_world(r['measured_position'], R, g)
        err = Qt - v
        if np.linalg.norm(err) < tol:
            break
        cmd = Qt + np.clip(cmd - Qt + err, -max_extra, max_extra)
    return v, err, r
