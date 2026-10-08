"""RB-Y1 kinematics for the box squeeze: URDF forward kinematics and ball-tool IK (offline only).

TEMPORARY: this is our own IK because the session's tools offered only single-arm `rby1_move_eef`
on the URDF ee frame. Future sessions will have native IK and control tools - use those instead and
keep only the constants below (ball offset, axle, bumper) and the task logic in plan.py / grasp.py.

The URDF (assets/model.urdf = rby1-sdk models/rby1a/urdf/model.urdf, v1.0) reproduces the measured
end-effector poses to 0.1 mm. It does NOT contain the add-on round fingers: closed, they form a
"ball" whose centre is BALL_D along -z_ee from ee_* (calibrated from RGB on 2026-10-08).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

URDF = Path(__file__).resolve().parents[1] / 'assets' / 'model.urdf'

# Backend limits (rby1_status -> limits, 2026-10-08). Plans keep MARGIN inside them.
LIMITS = {
    'torso': ([-0.162, -0.424, -2.4, -0.685, -0.424, -2.256], [0.162, 1.2, 1.471, 1.471, 0.424, 2.256]),
    'right_arm': ([-3.042, -3.042, -3.042, -2.518, -3.042, -1.471, -2.605], [3.042, -0.083, 3.042, -0.083, 3.042, 1.82, 2.605]),
    'left_arm': ([-3.042, 0.083, -3.042, -2.518, -3.042, -1.471, -2.605], [3.042, 3.042, 3.042, -0.083, 3.042, 1.82, 2.605]),
}
MARGIN = 0.2
READY = {'right': [-0.7854, -0.5236, 0.0, -1.5708, 0.0, 0.7854, 0.0],
         'left': [-0.7854, 0.5236, 0.0, -1.5708, 0.0, 0.7854, 0.0]}

BALL_D = 0.195          # closed round fingers: ball centre along -z_ee (m)
BALL_R = 0.03           # visual radius; touch showed an effective centre-to-face offset of ~0.05
AXLE_X = 0.228          # the base turns in place about this point (wheel axle) in the base frame
BUMPER_X = 0.304        # front of the base


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, angle):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class Model:
    def __init__(self, path=URDF):
        root = ET.parse(str(path)).getroot()
        self.child_joint = {}
        for j in root.findall('joint'):
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            ax = j.find('axis')
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            self.child_joint[j.find('child').get('link')] = {
                'name': j.get('name'), 'type': j.get('type'), 'parent': j.find('parent').get('link'), 'T': T,
                'axis': [float(v) for v in ax.get('xyz').split()] if ax is not None else [1, 0, 0]}

    def fk(self, q, link):
        chain = []
        while link in self.child_joint:
            chain.append(self.child_joint[link])
            link = chain[-1]['parent']
        T = np.eye(4)
        for j in reversed(chain):
            T = T @ j['T']
            if j['type'] in ('revolute', 'continuous'):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j['axis'], q.get(j['name'], 0.0))
                T = T @ R
        return T


MODEL = Model()


def joint_dict(torso=None, right_arm=None, left_arm=None, head=None):
    q = {}
    for prefix, values in (('torso', torso), ('right_arm', right_arm), ('left_arm', left_arm), ('head', head)):
        for i, v in enumerate([] if values is None else values):
            q[f'{prefix}_{i}'] = float(v)
    return q


def torso_pose(a, b=0.0, yaw=0.0):
    """Upright rule: squat a (torso_1), lean b (torso_3 = a + b), chest yaw (torso_5)."""
    return [0.0, a, -2 * a, a + b, 0.0, yaw]


def ee_pose(side, torso, arm):
    return MODEL.fk(joint_dict(torso=torso, **{f'{side}_arm': arm}), f'ee_{side}')


def link_pos(name, torso, right=None, left=None):
    return MODEL.fk(joint_dict(torso=torso, right_arm=right, left_arm=left), name)[:3, 3]


def ball_centre(T):
    return T[:3, 3] - BALL_D * T[:3, 2]


def axis_dir(tilt):
    """Gripper z_ee for fingers pointing forward-down `tilt` rad from vertical (axis in the base x-z plane)."""
    return np.array([-np.sin(tilt), 0.0, np.cos(tilt)])


def arm_margin(side, q):
    lo, hi = (np.array(v) for v in LIMITS[f'{side}_arm'])
    return float(np.minimum(np.asarray(q) - lo, hi - np.asarray(q)).min())


def solve_ball(side, torso, centre, tilt=None, R=None, q0=None, tries=0, w_reg=0.02, margin=MARGIN, seed=0):
    """Arm joints putting the ball centre at `centre`. Orientation: axis `tilt` (free roll) or full `R`.
    With q0 and tries=0 the solution stays on q0's branch (w_reg pulls towards q0)."""
    lo, hi = (np.array(v) for v in LIMITS[f'{side}_arm'])
    lo, hi = lo + margin, hi - margin
    q_ref = np.array(q0 if q0 is not None else READY[side], float)
    centre = np.asarray(centre, float)

    def res(q):
        T = ee_pose(side, torso, q)
        rot = 0.3 * (T[:3, :3] - R).ravel() if R is not None else 0.5 * (T[:3, 2] - axis_dir(tilt))
        return np.concatenate([ball_centre(T) - centre, rot, w_reg * (q - q_ref)])

    rng = np.random.default_rng(seed)
    starts = [np.clip(q_ref, lo, hi)] + [rng.uniform(lo, hi) for _ in range(tries)]
    best = None
    for s in starts:
        sol = least_squares(res, s, bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=2000)
        if best is None or sol.cost < best.cost:
            best = sol
    T = ee_pose(side, torso, best.x)
    return best.x, float(np.linalg.norm(ball_centre(T) - centre)), T
