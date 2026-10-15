"""Rim-pinch grasp geometry for a bucket held with both grippers OPEN around the rim wall (one finger inside, one
outside), the two grasps diametrically opposite across the robot's y axis (bucket centre straight ahead).

Gripper frame: fingers along -z_ee, closing along x_ee. For a grasp at the bucket's left (+y) or right (-y) side
the closing axis is radial (base y) and the fingers may tilt only along the rim tangent (base x): -z_ee =
(sin a, 0, -cos a). The pinch point (middle of the round fingertips, PINCH_D along -z_ee) goes on the wall at
PINCH_DEPTH below the rim top."""
import numpy as np

from lib import BOXPNP_CODE  # noqa: F401  (sys.path for boxpnp)
from boxpnp.core import kin

from bimanual_bucket_rim_pnp_params import PINCH_D, PINCH_DEPTH, WALL_R  # hyper-parameters: bimanual_bucket_rim_pnp_params.py


def grip_rotation(side, tilt, flip=False):
    """ee rotation for a grasp on `side` ('left' at +y, 'right' at -y): closing axis along base y, fingers down
    tilted forward by `tilt` (rad). flip turns the hand 180 deg about the finger axis."""
    s = 1.0 if (side == 'left') != flip else -1.0
    z = np.array([-np.sin(tilt), 0.0, np.cos(tilt)])
    x = np.array([0.0, s, 0.0])
    y = np.cross(z, x)
    return np.c_[x, y, z]


def ee_for_pinch(p_pinch, R):
    return np.asarray(p_pinch, float) + PINCH_D * R[:, 2]


def pinch_points(center, rim_z, wall_r=WALL_R, depth=PINCH_DEPTH, axis_yaw=0.0):
    """Base-frame pinch points (left, right) on the wall: across the bucket along the direction axis_yaw + pi/2."""
    c = np.asarray(center, float)
    d = np.array([-np.sin(axis_yaw), np.cos(axis_yaw)])
    z = rim_z - depth
    return {'left': np.array([*(c + wall_r * d), z]), 'right': np.array([*(c - wall_r * d), z])}


SEEDS = {
    'right': [[-0.9389, -0.5662, 1.2496, -1.4252, -0.2809, 1.0351, 0.1607],     # measured: M1 pre-grasp
              [-0.8812, -0.2855, 1.3004, -0.8154, -0.1534, 0.6627, 0.1213],     # M1 grasp (bucket 0.597, +0.045)
              [-0.5548, -0.2892, 0.6959, -0.9629, 0.0531, 0.6409, 0.6249],      # C1 grasp (bucket 0.583, -0.010)
              [-0.9, -0.3, 0.0, -1.6, 0.0, 0.6, 0.0], [-0.5, -0.4, 0.6, -1.2, -0.4, 0.6, 0.8],
              [-1.2, -0.2, 0.3, -1.0, 0.3, 0.8, -0.8], [-0.7, -0.6, -0.3, -1.5, 0.5, 0.9, 1.5]],
    'left': [[-0.512, 0.5606, -0.6446, -1.5235, 0.1887, 1.001, -0.7351],
             [-0.5727, 0.3671, -0.6873, -0.9673, 0.0698, 0.6066, -0.7076],
             [-0.8092, 0.3929, -1.2871, -0.9293, 0.0948, 0.6578, -0.0979],
             [-0.9, 0.3, 0.0, -1.6, 0.0, 0.6, 0.0], [-0.5, 0.4, -0.6, -1.2, 0.4, 0.6, -0.8],
             [-1.2, 0.2, -0.3, -1.0, -0.3, 0.8, 0.8], [-0.7, 0.6, 0.3, -1.5, -0.5, 0.9, -1.5]],
}


def solve_arm(torso, side, p_ee, R, seeds=None):
    """IK of one arm for ee pose (p_ee, R) at torso `torso` from the given seeds, then the measured / default
    SEEDS: among solutions within 1 mm / 0.01 rad the one with the most joint margin, else the smallest error."""
    cands = []
    for seed in list(seeds or []) + SEEDS[side]:
        q0 = kin.q_from(torso=torso, right=seed if side == 'right' else None, left=seed if side == 'left' else None)
        q, ep, er = kin.solve_ee(q0, side, p_ee, R, iters=400)
        qa = kin.arm_q(q, side)
        lo, hi = kin._model.lowerPositionLimit, kin._model.upperPositionLimit
        iq, _ = kin._arm_idx(side)
        cands.append((ep, er, float(np.min(np.minimum(qa - lo[iq], hi[iq] - qa))), qa))
    good = [c for c in cands if c[0] < 0.001 and c[1] < 0.01]
    ep, er, margin, qa = max(good, key=lambda c: c[2]) if good else min(cands, key=lambda c: c[0] + 0.1 * c[1])
    return {'q': [round(float(v), 4) for v in qa], 'pos_err': ep, 'rot_err': er, 'margin': margin}


CAPSULES = (('link_torso_1', 0.01, 0.13, 0.35, 0.115),      # (link, x offset, z from, z to, radius) from the URDF
            ('link_torso_2', 0.025, 0.15, 0.31, 0.095))
BASE_FRONT_X, BASE_TOP_Z = 0.33, 0.30                       # bumper / wheels front, assumed base cover height


def knee_clearance(torso, center, rim_z, r_rim=0.152, r_bottom=0.131):
    """Smallest gap (m) between the lower torso capsules (URDF collision model) / the base front and the bucket
    (truncated cone r_bottom..r_rim from the floor to rim_z), evaluated in horizontal slices."""
    q = kin.q_from(torso=torso)
    fr = kin.frames(q, [c[0] for c in CAPSULES])
    c = np.asarray(center, float)
    gap = 9.0
    for name, ox, z0, z1, rad in CAPSULES:
        M = fr[name]
        for t in np.linspace(z0, z1, 40):
            p = M.translation + M.rotation @ np.array([ox, 0, t])
            for dz in np.linspace(-rad, rad, 21):
                z = p[2] + dz
                if z < 0 or z > rim_z:
                    continue
                half = np.sqrt(max(rad ** 2 - dz ** 2, 0.0))
                rb = r_bottom + (r_rim - r_bottom) * z / rim_z
                gap = min(gap, float(np.linalg.norm(p[:2] - c)) - half - rb)
    gap = min(gap, float(c[0] - r_bottom - BASE_FRONT_X))
    return gap
