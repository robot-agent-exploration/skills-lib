"""Pinocchio kinematics of the RB-Y1 (robot-model/model_v1.0.standard.urdf) for planning.

The robot tools stay the authority; this is for offline numbers (ball positions, poses)."""
from pathlib import Path

import numpy as np
import pinocchio as pin

ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'robot-model').exists()),
            Path(__file__).resolve().parents[2])
URDF = ROOT / 'robot-model' / 'model_v1.0.standard.urdf'
BALL_D = 0.195          # closed round fingers: ball centre along -z_ee (library skill, checked in RGB)

_model = pin.buildModelFromUrdf(str(URDF))
_data = _model.createData()

TORSO = [f'torso_{i}' for i in range(6)]
ARM = {s: [f'{s}_arm_{i}' for i in range(7)] for s in ('right', 'left')}


def q_from(torso=None, right=None, left=None, head=None):
    q = pin.neutral(_model)
    for names, vals in ((TORSO, torso), (ARM['right'], right), (ARM['left'], left), (['head_0', 'head_1'], head)):
        if vals is None:
            continue
        for n, v in zip(names, vals):
            q[_model.joints[_model.getJointId(n)].idx_q] = v
    return q


def frames(q, names):
    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)
    return {n: _data.oMf[_model.getFrameId(n)] for n in names}


def ee_and_ball(q):
    out = {}
    for side, f in (('right', 'ee_right'), ('left', 'ee_left')):
        M = frames(q, [f])[f]
        ball = M.translation - BALL_D * M.rotation[:, 2]
        out[side] = {'ee': M.translation.copy(), 'R': M.rotation.copy(), 'ball': ball,
                     'quat_xyzw': pin.Quaternion(M.rotation).coeffs().copy()}
    return out


LIMIT_MARGIN = 0.2      # rad inside the URDF limits (the backend uses 0.1; real stops can be inside)


def _arm_idx(side):
    return [_model.joints[_model.getJointId(n)].idx_q for n in ARM[side]], \
           [_model.joints[_model.getJointId(n)].idx_v for n in ARM[side]]


def solve_ball(q0, side, ball, R_target, iters=300, w_rot=0.3):
    """IK on one arm (torso fixed): ball centre (BALL_D along -z_ee) at `ball`, ee rotation R_target.
    Returns (q, pos_err_m, rot_err_rad)."""
    q = q0.copy()
    iq, iv = _arm_idx(side)
    fid = _model.getFrameId('ee_' + side)
    lo = _model.lowerPositionLimit[iq] + LIMIT_MARGIN
    hi = _model.upperPositionLimit[iq] - LIMIT_MARGIN
    ball = np.asarray(ball, float)
    for _ in range(iters):
        pin.forwardKinematics(_model, _data, q)
        pin.updateFramePlacements(_model, _data)
        M = _data.oMf[fid]
        p_ball = M.translation - BALL_D * M.rotation[:, 2]
        e_p = ball - p_ball
        e_r = pin.log3(R_target @ M.rotation.T)
        J = pin.computeFrameJacobian(_model, _data, q, fid, pin.LOCAL_WORLD_ALIGNED)[:, iv]
        # ball point velocity = v_ee + w x (p_ball - p_ee)
        r = p_ball - M.translation
        Jp = J[:3] - pin.skew(r) @ J[3:]
        Jr = J[3:]
        A = np.vstack([Jp, w_rot * Jr])
        b = np.concatenate([e_p, w_rot * e_r])
        dq = A.T @ np.linalg.solve(A @ A.T + 1e-4 * np.eye(6), b)
        dq = np.clip(dq, -0.1, 0.1)
        q[iq] = np.clip(q[iq] + dq, lo, hi)
        if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
            break
    return q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))


def arm_q(q, side):
    iq, _ = _arm_idx(side)
    return q[iq].copy()


def solve_ee(q0, side, p_ee, R_target, iters=300, w_rot=0.3):
    """IK on one arm (torso fixed) for the ee frame itself. Returns (q, pos_err_m, rot_err_rad)."""
    q = q0.copy()
    iq, iv = _arm_idx(side)
    fid = _model.getFrameId('ee_' + side)
    lo = _model.lowerPositionLimit[iq] + LIMIT_MARGIN
    hi = _model.upperPositionLimit[iq] - LIMIT_MARGIN
    p_ee = np.asarray(p_ee, float)
    for _ in range(iters):
        pin.forwardKinematics(_model, _data, q)
        pin.updateFramePlacements(_model, _data)
        M = _data.oMf[fid]
        e_p = p_ee - M.translation
        e_r = pin.log3(R_target @ M.rotation.T)
        J = pin.computeFrameJacobian(_model, _data, q, fid, pin.LOCAL_WORLD_ALIGNED)[:, iv]
        A = np.vstack([J[:3], w_rot * J[3:]])
        b = np.concatenate([e_p, w_rot * e_r])
        dq = A.T @ np.linalg.solve(A @ A.T + 1e-4 * np.eye(6), b)
        q[iq] = np.clip(q[iq] + np.clip(dq, -0.1, 0.1), lo, hi)
        if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
            break
    return q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
