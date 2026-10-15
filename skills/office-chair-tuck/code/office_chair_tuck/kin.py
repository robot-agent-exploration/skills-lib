"""Local kinematics (pinocchio, model_v1.0.standard.urdf): arm IK for planning joint-space moves.

The robot tools stay the authority: verify every result with rby1_fk / rby1_check before moving.
"""
from pathlib import Path

import numpy as np
import pinocchio as pin

from .geom import ROOT  # noqa: E402  (workspace discovery)
URDF = ROOT / 'robot-model' / 'model_v1.0.standard.urdf'
_model = pin.buildModelFromUrdf(str(URDF))
_data = _model.createData()
TORSO = ['torso_%d' % i for i in range(6)]
ARM = {s: ['%s_arm_%d' % (s, i) for i in range(7)] for s in ('right', 'left')}
EE = {'right': 'ee_right', 'left': 'ee_left'}
# joint limits as the backend reports them (URDF minus 0.1); keep IK 0.2 further inside
LIM = {
    'right': (np.array([-3.042, -3.042, -3.042, -2.518, -3.042, -1.471, -2.605]),
              np.array([3.042, -0.083, 3.042, -0.083, 3.042, 1.82, 2.605])),
    'left': (np.array([-3.042, 0.083, -3.042, -2.518, -3.042, -1.471, -2.605]),
             np.array([3.042, 3.042, 3.042, -0.083, 3.042, 1.82, 2.605])),
}


def _qidx(name):
    return _model.joints[_model.getJointId(name)].idx_q


def q_from(torso, right=None, left=None, head=(0, 0)):
    q = pin.neutral(_model)
    for n, v in zip(TORSO, torso):
        q[_qidx(n)] = v
    for side, arm in (('right', right), ('left', left)):
        if arm is not None:
            for n, v in zip(ARM[side], arm):
                q[_qidx(n)] = v
    q[_qidx('head_0')], q[_qidx('head_1')] = head
    return q


def fk(q, frame):
    pin.framesForwardKinematics(_model, _data, q)
    M = _data.oMf[_model.getFrameId(frame)]
    return M.translation.copy(), M.rotation.copy()


def quat_xyzw(R):
    qq = pin.Quaternion(R)
    return [qq.x, qq.y, qq.z, qq.w]


def R_from_quat(xyzw):
    x, y, z, w = xyzw
    return pin.Quaternion(w, x, y, z).toRotationMatrix()


def ik(side, torso, arm0, pos, quat=None, other=None, iters=300, margin=0.2, rot_weight=0.5):
    """Damped least squares IK for one arm -> (arm joints, pos error m, rot error rad)."""
    arm = np.array(arm0, float)
    lo, hi = LIM[side][0] + margin, LIM[side][1] - margin
    lo = np.minimum(lo, arm)          # never force a start outside the margin into a jump
    hi = np.maximum(hi, arm)
    idx = [_model.joints[_model.getJointId(n)].idx_v for n in ARM[side]]
    fid = _model.getFrameId(EE[side])
    Rt = R_from_quat(quat) if quat is not None else None
    for _ in range(iters):
        q = q_from(torso, arm if side == 'right' else other, arm if side == 'left' else other)
        pin.computeJointJacobians(_model, _data, q)
        pin.framesForwardKinematics(_model, _data, q)
        M = _data.oMf[fid]
        ep = np.asarray(pos) - M.translation
        J = pin.getFrameJacobian(_model, _data, fid, pin.LOCAL_WORLD_ALIGNED)[:, idx]
        if Rt is not None:
            er = pin.log3(Rt @ M.rotation.T)
            e = np.r_[ep, rot_weight * er]
            Jr = np.r_[J[:3], rot_weight * J[3:]]
        else:
            er = np.zeros(3)
            e, Jr = ep, J[:3]
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        dq = Jr.T @ np.linalg.solve(Jr @ Jr.T + 1e-3 * np.eye(len(e)), e)
        arm = np.clip(arm + np.clip(dq, -0.2, 0.2), lo, hi)
    q = q_from(torso, arm if side == 'right' else other, arm if side == 'left' else other)
    p, R = fk(q, EE[side])
    rerr = float(np.linalg.norm(pin.log3(Rt @ R.T))) if Rt is not None else 0.0
    return arm, float(np.linalg.norm(np.asarray(pos) - p)), rerr


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def Ry(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
