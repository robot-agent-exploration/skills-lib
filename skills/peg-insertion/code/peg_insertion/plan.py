"""Grasp and insertion geometry for the NIST ATB M1 pegs (generic over the CAD outlines).

Conventions: a peg/hole 'yaw' rotates the CAD outline (board_model.json, peg local frame)
into the base frame. Gripper yaw psi: TCP x axis (along the fingers) = (cos psi, sin psi),
TCP y axis (closing direction) = (sin psi, -cos psi). A point's gripper-frame offset
(a, b) from the TCP is a * x_axis + b * y_axis in the base frame.
"""
import numpy as np

from .board import MODEL, rot2

PERIOD = {'hexagon': np.pi / 3, 'triangle': 2 * np.pi / 3, 'rectangle': np.pi}
# gripper yaw = peg yaw + GRASP_PHASE (mod GRASP_STEP) closes the fingers on flat faces
GRASP_PHASE = {'hexagon': 0.0, 'rectangle': 0.0, 'triangle': np.pi / 6}
GRASP_STEP = {'hexagon': np.pi / 3, 'rectangle': np.pi, 'triangle': np.pi / 3}
# local hole yaw (board frame) relative to the CAD peg outline, from the CAD
HOLE_LOCAL_YAW = {}


def axes(psi):
    return np.array([np.cos(psi), np.sin(psi)]), np.array([np.sin(psi), -np.cos(psi)])


def wrap(a, period=2 * np.pi):
    return (a + period / 2) % period - period / 2


def quat(psi):
    return [float(np.cos(psi / 2)), float(np.sin(psi / 2)), 0.0, 0.0]


def outline(shape):
    return np.array(MODEL['pegs'][shape]['outline'])


def hole_yaw_local(name):
    """Yaw (rad) of a hole in the board frame in the peg-outline convention."""
    h = MODEL['holes'][name]
    s = h['shape']
    if s == 'hexagon':
        return np.radians(h['edge_dir_deg_mod60'])
    if s == 'rectangle':
        return np.radians(h['long_axis_deg'])
    if s == 'triangle':
        # peg outline apexes at 180, +-60 deg -> yaw = hole apex - 60 (mod 120)
        return wrap(np.radians(h['apex_dirs_deg'][0]) - np.pi / 3, 2 * np.pi / 3)
    raise ValueError(s)


def hole_pose(board_pose, name):
    """(centre_xy, yaw) of a hole in the base frame."""
    cx, cy, th = board_pose
    c = rot2(th) @ np.array(MODEL['holes'][name]['centre']) + [cx, cy]
    return c, th + hole_yaw_local(name)


def grasp_options(shape, peg_yaw):
    """Gripper yaws (rad, wrapped to [-pi, pi)) that close on flat faces."""
    base = peg_yaw + GRASP_PHASE[shape]
    k = int(round(2 * np.pi / GRASP_STEP[shape]))
    return sorted({round(float(wrap(base + i * GRASP_STEP[shape])), 6) for i in range(k)})


def grasp_plan(shape, peg_xy, peg_yaw, psi, tcp_xy=None):
    """Where the fingers centre the peg for gripper yaw psi.

    Returns dict(tcp_xy to command so the peg is not pushed, width (closing extent),
    in_hand (a, b) of the peg centroid from the TCP after the grasp, rel_yaw)."""
    P = outline(shape) @ rot2(peg_yaw).T + np.asarray(peg_xy)
    ax, ay = axes(psi)
    bs = (P - peg_xy) @ ay
    b_mid = (bs.min() + bs.max()) / 2           # finger-centred point relative to centroid
    tcp = np.asarray(peg_xy) + b_mid * ay if tcp_xy is None else np.asarray(tcp_xy)
    a = float((np.asarray(peg_xy) - tcp) @ ax)
    return {'tcp_xy': tcp.tolist(), 'width': float(bs.max() - bs.min()),
            'in_hand': [a, float(-b_mid)], 'rel_yaw': float(wrap(peg_yaw - psi, PERIOD[shape]))}


def insert_plan(shape, in_hand, rel_yaw, hole_xy, hole_yaw, psi_pref):
    """Gripper yaw and TCP xy that put the held peg on the hole (yaw matched mod symmetry)."""
    per = PERIOD[shape]
    psi0 = hole_yaw - rel_yaw
    psi = psi0 + per * np.round((psi_pref - psi0) / per)
    ax, ay = axes(psi)
    tcp = np.asarray(hole_xy) - (in_hand[0] * ax + in_hand[1] * ay)
    return {'psi': float(psi), 'tcp_xy': tcp.tolist()}
